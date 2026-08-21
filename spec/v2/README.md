# tensor-layout contract v2

Three document kinds, one engine, and a layout that is computed rather than
stored. The ratified design is
`cozy-creator-tracker/research/tensor-layout-v2/0-DESIGN-v2.md`; this file is
the format.

```
topologies/   {key -> logical shape}, per component. DERIVED, never authored.
rules/        quant(topology) -> the expected header. One per FORMAT, ever.
morphisms/    topology -> topology. Rekey tables and fusion seams.
headers/      the banked reference headers everything is derived and proved from.
baselines/    the v1 engine's own answers, banked before it was deleted.
vectors/      the ratified design's worked example, transplanted verbatim.
```

## topology

```jsonc
{
  "format": "tensorfs-topology-v2",
  "name": "sdxl.diffusers", "version": 1,
  "derived_from": "sdxl-base-sharded (local:...)",   // the reference checkpoint
  "dtype": "BF16",                                    // the reference's DOMINANT element type
  "components": [{
    "name": "unet", "role": "denoiser",
    "islands": {"logit_scale": "F32"},                // keys the reference ships at another type
    "tensors": {"conv_in.weight": [320, 4, 3, 3], "...": []}
  }],
  "digest": "<sha256 of the canonical rendering>"
}
```

Regenerate with `go run ./scripts/build_v2_corpus`, which reads
`CORPUS.tsv` and the banked headers. **Never hand-edit one**: the record would
describe a checkpoint that may not exist, and the derivation test fails on the
digest.

`role` comes from the reference tree's own directory name (`unet` /
`transformer` / `dit` -> denoiser, `vae` -> vae, `text_encoder*` -> text
encoder, a flat checkpoint -> backbone). It exists so a quant rule can scope
itself — `cozy.fp8-rowwise` transforms the denoiser and passes the rest
through.

`dtype` and `islands` are REFERENCE FACTS, not part of what the topology means.
They exist so a `plain.<dtype>` rule can be reference-tolerant: a key the
reference itself ships at a wider compute type is accepted at either, because
both packagings are shipped and refusing the reference would be absurd. A
QUANTIZED element (fp8, packed nibbles) is never tolerated this way — that is a
quantization, not an island.

## quant rule

```jsonc
{
  "format": "tensorfs-quant-rule-v2",
  "name": "cozy.fp8-rowwise", "version": 1,
  "declared_dtype": "float8_e4m3fn",   // the torch spelling a lane declares
  "capability_floor_sm": 89,           // the sm floor, ON THE RULE, not in a lookup table
  "base_dtype": "BF16",                // what untransformed tensors carry
  "reference_tolerant": true,
  "conventions": { "nibble_order": "...", "scale": "per_channel_out", "...": "" },
  "scope_roles": ["denoiser", "backbone"],
  "lossy": true,
  "inverse": "W_bf16[r, c] = weight[r, c] * weight_scale[r]",
  "eligible": {
    "source_dtypes": ["F64", "F32", "F16", "BF16"],
    "rank": 2, "key_suffix": ".weight", "dim_align": [16, 16],
    "require_repeated_block_segment": true,
    "skip_module_substrings": ["embed", "norm", "..."],
    "skip_module_exact": ["proj_in", "proj_out", "proj"]
  },
  "emissions": [
    {"key": "{module}.weight",       "dtype": "F8_E4M3", "shape": ["d0", "d1"]},
    {"key": "{module}.weight_scale", "dtype": "F32",     "shape": ["d0"]}
  ],
  "digest": "..."
}
```

**`conventions` is IDENTITY.** Two formats can agree on every tensor name,
dtype and rank and still be different bytes; `cozy.nvfp4-flat` and
`bfl.nvfp4-preswizzled` differ only in nibble order and scale layout, and
reading one as the other measured LPIPS 1.11. The conventions are in the digest
so the confusion cannot be spelled.

**The eligibility predicate is TRANSCRIBED FROM THE PRODUCER**, with the source
line cited in the description — not from a description of the producer. Its
`skip_module_*` split mirrors how the producer anchors its patterns: `embed`
matches anywhere in the module path, `^proj_in$` matches only a path that IS
`proj_in`, which is why SDXL's `down_blocks.{i}.attentions.{i}.proj_in` DOES
convert.

**Emission shapes are formulas** over the source dims: `d0`, `d1/2` (packed
nibble pairs), `d1/16` (a per-16-block scale), `ceil(d0/128)*128*ceil(d1/16/4)*4`
(a pre-swizzled blocked array). Integers, `dN`, `*`, `/`, `ceil(...)` — that is
the whole language, and it has exactly ONE evaluator, in `quantrule.go`.
Division is exact outside `ceil()`: an inexact `/` means the eligibility
predicate admitted a tensor the emission cannot shape, which is a bug in the
rule and not a number to round.

`optional: true` marks a tensor a calibration may or may not have produced
(`input_scale`, `pre_quant_scale`). Its SHAPE is still fixed — optional means
"absent or exactly this", never "anything".

## morphism

```jsonc
{
  "format": "tensorfs-morphism-v2",
  "name": "minimax-h3.split-to-fused-qkv", "version": 1,
  "from": "minimax-h3.diffusers@1", "to": "minimax-h3.native@1",
  "tier": 3, "invertible": true,
  "rekey": [{"from_key": "transformer_blocks.{i}.attn.to_out.0.weight",
             "to_key":   "blocks.{i}.attn.out_proj.weight"},
            {"from_prefix": "transformer_blocks.", "to_prefix": "blocks."}],
  "seams": [{
    "target": "blocks.{i}.attn.qkv_proj.weight", "groups": 56,
    "parts": [{"role": "q", "share": 1, "key": "transformer_blocks.{i}.attn.to_q.weight"},
              {"role": "k", "share": 1, "key": "..."},
              {"role": "v", "share": 1, "key": "..."}],
    "provenance": "RATIFIED ... 56 HEAD-MAJOR TRIPLES, and the group count is the fact ..."
  }],
  "provenance": ["why this file exists, and what it costs to be wrong"],
  "digest": "..."
}
```

A seam's `parts` are always spelled in the SPLIT packaging and its `target` in
the FUSED one, whichever direction the morphism runs; `rekey` rows map
everything the seams do not touch, first match wins. `invertible: true` gets the
reverse edge for free — and the catalog PROVES it at load by running the
inverse and landing back on the source topology, key for key.

`groups > 1` is HEAD-MAJOR interleaving: the axis is cut into `groups`
repetitions of the part pattern, not one run per part. This is the fact a flat
split gets right for head 0 and wrong for the other 55.

**Every seam states its evidence.** A header cannot show a fusion, so a seam
file's shares are the one thing in this corpus that no measurement can check —
which is why each carries the ratified provenance block that proved it, cited
to the module source line, transplanted from the v1 document it was ratified in
(tensorfs#150). The facts survived the cut; only the container changed.

## tiers

| tier | what | cost | policy |
|---|---|---|---|
| T1 | rekey | new manifest, same chunks — ZERO new bytes | derive freely |
| T2 | contiguous fuse/split | manifest range-slicing, same chunks | derive freely |
| T3 | interleaved permutation | load-time materialization | store ONE packaging |
| T4 | quant, and casts | lossy | gate, produce once, keep provenance |

## who reads what

| consumer | reads | how |
|---|---|---|
| the hub (Go) | everything | links the package; `Catalog.Admit` / `Catalog.Layout` |
| a worker (Python) | rule IDENTITY facts, and the computed header | `tensorfs.layout2` over the vendored documents; the VERDICT arrives binding-carried |
| a producer | the computed header of its target | `go run ./scripts/compute_layout '<pair>'` |

**The worker does not compute a verdict and does not evaluate a rule.** It reads
identity — a rule's `declared_dtype` and `capability_floor_sm`, which under v1
were a document field plus a lookup table keyed on the dtype spelling, so a lane
could silently lose its sm floor by being spelled differently — and it enforces
the header it was handed. Every admission is decided once, in Go, by the hub.

There is deliberately no pyo3 binding for the decision or the evaluator. A Rust
reader of these documents would be a second parser of them, and a Rust evaluator
of the emission formulas would be the second evaluator the design forbids; the
identity facts are plain data and the layout is computed upstream.
