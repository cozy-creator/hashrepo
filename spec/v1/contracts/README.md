# Tensor layout contracts v1

A contract is the **declarative, versioned input** that directs chunking seams
and names the layout a file implements. Format: `tensorfs-contract-v1`, one
JSON document per contract, schema in `contract.schema.json`.

These files are DATA. They are the library of well-known contracts that
replaced the static hint tables (dedup-invariance memo §4, ratified
2026-08-16): the planner grows one mechanism, and new architectures arrive as
documents rather than as planner code.

## The handle

`<producer>.<format>@<major>` — the same spelling gen-worker uses for its
tensor-layout-contract handles, so the stamp a snapshot carries reads the same
on both sides of the serving boundary.

**A published `name@version` is immutable.** Changing what a contract means
requires a new `version`; `crates/tensorfs-core/tests/contract_seams.rs`
(`the_shipped_library_is_pinned_by_digest`) pins each document's digest so an
in-place edit fails CI. This is what makes a stamped snapshot reproducible:
same file + same `contract@version` ⇒ same snapshot id, on every store,
forever.

## Custom contracts: digest identity

`name@version` is a promise only this curated, CI-pinned library can keep, so
it is reserved for documents shipped here. Every OTHER contract — an
author-constructed custom — carries **no name at all**: `name` and `version`
are absent from the document, and its identity is the SHA-256 of its
canonical rendering, stamped `sha256:<64 hex>`. A free-text name on an inline
object validates nothing and can lie or collide; the author's variable name
is their label, and platform surfaces derive a display label from the digest
prefix. Equality everywhere is by digest. A custom document later adopted
into the library gains a name in a NEW document — new digest, new stamp
spelling — while chunk identity stays answerable, because boundaries are a
pure function of (file bytes, document).

A digest stamp IDENTIFIES exactly but DESCRIBES nothing: reading a snapshot
stamped `sha256:…` tells you the layout's identity, not its declarations.
Document recovery is out of band — the release derive document (hub-stored)
and the org's extracted-contract set.

**Matcher scoping is org-scoped.** The builtin library joins every ingest
matcher set, globally. A custom contract joins the ingest matcher set only
for the org whose release declared it — an org's BYOM upload must be able to
stamp the org's own custom at ingest, before any deployment binds it, and
cross-org custom matching never happens. (Deploy-scoped-only was rejected for
exactly the BYOM-before-deploy case.)

Run-preserving conversions (rekey, outer-axis fuse/split, declared permutes)
work for customs with no prior platform knowledge — `compose::derive`
computes them from the two documents' role sets. Anything needing math (cast,
quantization) has no adapter for a custom contract and refuses, typed.

## Fields

- `dtype` (top-level) — **the LANE'S QUANTIZATION**, not a summary of the
  per-tensor dtypes and not a vote over them: `sdxl.diffusers-fp8-rowwise@1` is
  `float8_e4m3fn` with 221 of its 257 declarations BF16, and
  `sdxl.diffusers-nvfp4-flat@1` is `float4_e2m1fn` with no declaration
  spelling fp4 at all (the resident weights are packed `U8` pairs). Nothing
  matches on it — the per-tensor `dtypes` below remain the matcher's business.

  **REQUIRED on a serve lane; optional only on a component FRAGMENT**
  (`sdxl.clip-g-*`, `dit.blocks-fused-qkv`). `lanes={contract: floor}` is
  required on every gen-worker Model subclass (pgw#1597) and the declaration
  reads this field, so a lane without one is undeclarable — a refusal with no
  remedy, discovered in another repo at the far end of a vendor bump.
  `crates/tensorfs-core/tests/contract_seams.rs`
  (`every_serve_lane_declares_its_quantization`) holds the line, with an
  exhaustive fragment allow-list so a forgotten dtype cannot pass as one.

  **Use the torch-precise spelling, and one gen-worker's `DTYPE_MIN_SM` knows.**
  That table is the only producer of a lane's sm floor, and it answers `0` for
  a spelling it does not recognize — correct for fp32, catastrophic for a
  quantized lane, which would then be offered to every card in the fleet with
  no floor at all. Prefer `float4_e2m1fn` over `float4_e2m1fn_x2` for exactly
  this reason: the second is the packed type torch actually ships and reads
  prettier, and it silently drops the sm100 floor.

  Where a layout genuinely has no torch scalar type — a GGUF k-quant container
  (`qwen3.6-27b-mtp.gguf-ud-q4-k-xl@1` declares `q4_k`) — the container's own
  ggml name is the honest spelling, `Contract.torch_dtype` refuses it with a
  typed `MissingDtype` rather than resolving to something wrong, and
  `DTYPE_MIN_SM` must be taught the spelling and its true floor so that floor
  is a decision instead of a default.

  Additive: absent from a document ⇒ absent from the canonical rendering ⇒
  pre-existing digests unchanged.
- `tensors[].pattern` — the tensor's spelling in the file. Literal text with
  `{i}` holes, each matching one non-negative integer without leading zeros.
  No regex: matching is linear and unambiguous.
- `tensors[].role` — the same tensor's spelling-independent identity, carrying
  the same holes in the same order. Two contracts describe the same layout in
  different spellings when their instantiated role sets agree; that is the
  decision procedure for "viewable or real conversion?".
- `tensors[].dtypes`, `tensors[].rank` — constraints. A file whose tensor
  disagrees does not match the contract at all, which is what makes a contract
  falsifiable from the header alone.
- `tensors[].required` — default `true`. Every required declaration must be
  present for the contract to match.
- `tensors[].fusion` — a fusion along the **outermost** axis (`axis: 0`, the
  only value: only the outer axis is byte concatenation). The axis is `groups`
  repetitions of the `parts` cycle, and `parts[].share` are integer shares of
  one group; the outer dimension must divide `groups x sum(share)` or the
  contract does not match. A run's role is the entry's role plus `#part` plus
  `@group`, which is exactly how the other packaging names the same bytes.
- `sets` — named, removable tensor sets (`adaln_projections`, …) for subset
  snapshots.

## Interleaved fusions (`groups > 1`)

`groups: 1` is plain stacking — `concat([q, k, v], dim=0)`.

`groups > 1` is an **interleave**. MiniMax-H3's native DiT fuses
`blocks.N.attn.qkv_proj.weight [21504, 5376]` head-major: `q`, `k` and `v` are
adjacent **inside each of 56 heads**, not three stacked blocks (the naive
`cat` is not a near-miss — it is ~90% error that never crashes). That is still
an ordered concatenation of contiguous runs: 3x56 of them instead of 3. So it
is exactly recoverable by cut points, byte ORDER is untouched, and the split
packaging declares the same runs (`groups: 56`, one unnamed part) so both
sides cut at the same places. Objects are content-addressed, so the two
packagings share every attention byte despite reading their runs in different
orders — 11.56 GB, 17.4% of that DiT.

**The floor.** No declared run may be smaller than **1 MiB**
(`MIN_SEAM_PART_BYTES`); a fusion whose runs fall below it produces NO cut
points and the tensor grids plainly. **This value is FROZEN for v1** — it is a
boundary-deciding input, so snapshot identity is a pure function of (file
bytes, contract@version, this constant, `MAX_OBJECT_SIZE`). Moving it would
make identical inputs chunk differently across planner versions, silently
breaking dedup and identity; like the 64 MiB grid, it changes only with a
format version bump, never as a tunable. This is the memo's rejection of
row-granular splitting expressed as a number: a file cut entirely at the floor
still fits TFM1's 1M-record bound at 1 TiB, while a KB-scale row shuffle blows
through it. Because the floor is a pure function of the tensor's extent and
the declaration, the fused and split packagings cross it at the same size — a
fusion never degrades on one side only. An interleave too fine for the floor
is not expressible as seams and belongs to the adapter (permute) class
instead.

Fusions that are **not** expressible here, by construction: inner-axis fusions
(GPT-2 `c_attn`) and any re-arrangement that does not decompose into ordered
runs of the other packaging. Those are view-expressible, not chunk-shareable,
and route to the adapter vocabulary.

## Generating a candidate: cheap, not optional

`lanes={contract: floor}` is REQUIRED on every gen-worker Model subclass
(pgw#1597), real documents only. That is only survivable if authoring a
document is cheap, so it is: a contract describes a HEADER, and
`tensorfs.generate` derives one from headers alone —
`scripts/generate-contract.py`, HTTP range reads, kilobytes per file, **zero
weight bytes**.

```
scripts/generate-contract.py --name musicgen.transformers-fp16 --version 1 \
    --source stereo-medium=hf:facebook/musicgen-stereo-medium \
    --pin-trivial --out spec/v1/contracts/musicgen.transformers-fp16.v1.json
```

**Generate → human-ratify → publish.** The generator writes a CANDIDATE whose
description opens `GENERATED CANDIDATE - NOT RATIFIED.` and ends with a
RATIFICATION OWED list. Grep that marker to find every unratified document;
only a human removes it.

What is derived: the pattern set (only a WHOLE-SEGMENT integer becomes `{i}`,
so `conv1`/`conv2` stay two declarations), `rank`, `dtypes`, and `required` —
a declaration is required only when EVERY source carries it, so
"one document, two checkpoints" is a measurement. What is NOT derived, because
guessing is worse than owing: `role` (mechanically the pattern), `fusion` (a
header cannot see a fused axis; a wrong one is the ~90%-error split that never
crashes) and `sets`.

⚠️ **`required` is derived per SOURCE; matching happens per MEMBER FILE.** Those
are two axes of one question, and only the first is visible to `candidate()` —
it sees one flat tensor list per source. The bind gate builds one `ArtifactFile`
per `.safetensors` and never groups an index's shards (tensorhub
`internal/bindgate/bindgate.go:388`), so an all-`required` document over a
sharded checkpoint demands every declaration of every shard and **refuses the
tree it was generated from**. tensorfs#150 found all three multi-shard
candidates in exactly that state. `generate-contract.py` now counts the members
a ref resolves to and passes `sharded=`, which makes every declaration optional
— the convention `verdict.go` already states ("a multifolder lane document is
all-optional declarations") and every hand-authored multi-file document here
already follows. The cheap proof either way is `Contract::verdict` over that
checkpoint's own headers; run it before believing a new document.

`fusion` stays underivable on purpose, so the ratified answers for this set live
in `scripts/declare-ratified-fusions.py`, which `generate-the-130-set.sh` calls
for the two documents that carry one. A fusion named there that the document
does not declare is a refusal, not a warning.

A LANE's `dtype` is derived only when the header settles it beyond argument
(one element type, or one plus an F32 island); anything mixed REFUSES with
`UndeclarableLane` and asks for `--dtype`, because a quantized lane's dtype is
authored knowledge like roles and fusions — a majority vote reads every fp8
lane in this library as bf16. A spelling `DTYPE_MIN_SM` cannot price refuses
too. `--fragment` is the only way to emit a document with no dtype.

Coverage is VERIFIED, not asserted: every pattern is expanded back over every
source and must partition it exactly. The report also names its own weak
spots — a DEGENERATE hole (one value, i.e. an `nn.Sequential` position rather
than a layer stack), a non-contiguous index range, a rank conflict.
`--literal 'layers.{i}.attn.to_out.0.weight'` pins a chosen hole back;
`--pin-trivial` does it for every 0/1-valued hole.

**The falsification.** Re-deriving `ernie.diffusers-bf16@1` — a hand-authored,
shipped document — from `baidu/ERNIE-Image`'s real header reproduces its
pattern, rank and dtype set EXACTLY, 24 declarations over 409 tensors, with
three pins. `scripts/generate-the-130-set.sh` keeps that check as its first
step, beside the commands that derived every document below it.

## No document, and why (`hunyuan3d-2.1`)

The one owed document that cannot be generated. `tencent/Hunyuan3D-2.1` carries
exactly ONE `.safetensors` — an image encoder, the family-plural class this
library never declares. Every core model is a pickle: the shape DiT is
`hunyuan3d-dit-v2-1/model.fp16.ckpt`, the paint UNet and both VAEs are `.bin`.
A tensorfs document describes a safetensors header and there is none to read.

The fix is a REPACK, never teaching this format to describe pickles: a
conversion-endpoint invoke, pod-side (weights-locality), `torch.load(...,
weights_only=True)` per member into `save_file` under the upstream key
spellings, published to `tensorhub/hunyuan3d-2.1`; then this generator runs
against the repacked header like any other. Until that lands the class has no
document, and it says so rather than pointing `lanes=` at a guess.

## Matching and tie-break

Identification reads the header inventory only (names, shapes, dtypes; no
tensor bytes). Among all contracts that match, the winner is:

1. **most specific** — explains the most of the file's tensors,
2. then **highest version**,
3. then the lexicographically smallest name.

Key 3 exists so the answer never depends on registry insertion order. The
winner is recorded in the snapshot as `contract@version`; no match is recorded
as `none` and chunks on the plain per-tensor grid.

## Quantized lanes, and where the recipe lives

A lane document's declarations state what its bytes ARE, so they also state how
they are MADE. `cozy.sdxl-*`-era vocabulary put a quant recipe on the endpoint
and cast at serve time; the recipe is a property of the LANE (torchcg tcg#53),
and the two fp8 documents here are that fact written down:

| lane | derived from | converts |
|---|---|---|
| `sdxl.diffusers-fp8-rowwise@1` | `sdxl.diffusers-bf16@1` | 36 of 221 declarations, UNET only |
| `minimax.h3-dit-fp8-rowwise@1` | `minimax.h3-dit-diffusers@1` | 7 of 10 declarations |

The fp8-rowwise packaging is a PAIR: a quantized Linear's `X.weight` is
`F8_E4M3` `[out, in]` and a sibling `X.weight_scale` is `F32` `[out]`, a
per-row DEQUANT multiplier. Modules the recipe skips keep their source dtype
and carry **no** scale leaf, so a reader tells converted from kept per tensor
rather than from a name list. Each scale declaration mirrors its weight's
`required` flag, which is where a document says "an fp8 weight without its
scale is not this layout".

Neither document was hand-written. Both are the bf16 sibling with gen-worker's
own eligibility rule applied — rank-2 float `.weight`, 16-aligned, under a
repeated-block path segment, module path missing every skip pattern — so the
document and the producer cannot disagree about which tensors are fp8. The one
conjunct the format cannot carry is 16-alignment, because contracts are
shapeless on purpose; that stays the producer's refusal, and each document
records why omitting it is sound for its family.

**The recipe is DERIVED from the declarations, and there is no `recipe` field.**
`tensorfs.convert.recipe_for` (Python) and `Contract.Recipe()` (Go) are one
rule: fp8 element types present **and** `.weight_scale` twins present ⇒
`fp8-rowwise`, else `dtype-cast`. Note what it does **not** read — the name. A
document called `…-fp8-rowwise` that failed to declare scales answers
`dtype-cast`, and a differently-named one declaring both answers
`fp8-rowwise`, because a name is a label and the declarations are the
falsifiable part. Storing the recipe instead would be a second assertion about
the same fact, able only to agree with the declarations or contradict them —
and the contradiction is the bug tcg#53 exists to remove. Anything downstream
reading a `recipe` key is a defect, not a missing field.

So a gate that answers `DerivableVia` names the job the producer will actually
run. Proven end to end by `scripts/prove-conversion.sh`: real trees in, real
conversion, and the verdict taken by the real matcher.
