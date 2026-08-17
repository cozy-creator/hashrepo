# Dedup invariance: what shares bytes, what can only share views

Design memo for issues [#80](https://github.com/cozy-creator/tensorfs/issues/80) and
[#81](https://github.com/cozy-creator/tensorfs/issues/81). Every claim below was verified
against the landed planner (`crates/tensorfs-core/src/planner/`) and measured on a real
checkpoint (SmolLM2-135M, bf16, 272 tensors, 269 MB data) — the measurement script chunked
exactly as the planner does.

## 1. What the planner already guarantees (verified in code)

- safetensors: the 8-byte prefix + JSON header becomes Header region(s); each tensor's
  `data_offsets` extent becomes its own region run, with the 64 MiB grid applied **relative to
  the tensor's start** (`append_split_region(header_end + span.start, …)`). `data_offsets` are
  relative to the data section, so header edits never move tensor content relative to its
  chunk grid.
- GGUF: magic/KV/directory are Header regions; each tensor's **unpadded** extent is its own
  region run on the same tensor-relative grid; **alignment padding is emitted as separate
  Header regions, never glued to tensor data**. Cross-format sharing with a safetensors twin
  is therefore structurally enabled, not accidental.
- Chunking is a pure function of file bytes + planner version. This property is load-bearing
  (§4) — same file ⇒ same record list ⇒ same TFM1 snapshot id, on every store.

## 2. The invariance ladder (measured)

| transform | data bytes shared | per-layout cost | mechanism |
|---|---|---|---|
| rename keys | **100%** (272/272 chunks) | 36 KB header vs 269 MB data | nothing — already free |
| repack: 1 file → 3 shards (real `safetensors` lib) | **100%** | ~30 KB of headers | nothing — already free |

Packaging note (Paul's ruling, DESIGN-RULINGS): our canonical packaging is **one safetensors
file per model at any size** — no HF-style sharding (TFM1's 1M-record bound gives ~64 PiB of
headroom per file). Shard↔single-file invariance therefore matters one-way: it is what lets
a *foreign* sharded ingest dedup fully against our canonical single file, not a packaging
choice we make twice.
| GGUF twin, layout-faithful converter | **100%** (272/272 tensors byte-identical) | GGUF KV/directory | nothing — padding isolation makes it work |
| fuse qkv (outer-axis concat), no hints | 87.7% of chunks — **all 12.3% of fusible bytes forfeited** | — | seams needed |
| fuse qkv, with seam cuts | **100%** | header only | contract-directed splitting (§4) |
| llama.cpp GGUF of a llama-family model, dtype-faithful | ~90% (q+k = 9.9% permuted) | — | permute is layout-expressible (§3) |
| transpose / rope-permute / inner-axis fuse | 0% at chunk level, **exactly viewable** | zero as a derived snapshot | adapter (§3) |
| dtype cast, finetune, LoRA-merge, re-quantization | 0%, and correctly so | — | out of scope: math, not layout |

Fuse/split exposure on this model: qkv = 12.3% of bytes, gate/up = 39.5% — ecosystems that
fuse both put ~52% of model bytes behind seams. Embeddings are 21%.

## 3. Two distinct properties — never conflate them

**Chunk-shareability** (two *real* files share objects) holds only for transforms that
preserve contiguous byte runs at tensor granularity: rename, reshape, shard/repack,
concat/split along the **outermost** stored axis. This is the narrow set, and it is exact.

**View-expressibility** (a derived snapshot serves layout B from layout A's bytes with zero
new storage) is much broader: any bijective index remap qualifies — transpose, the llama.cpp
rope-permute (`reshape(n_head, 2, d/2, …) → swapaxes(1,2) → reshape`), inner-axis fusions
(GPT-2 `c_attn`), head interleaves. The reader pays a gather; the store pays nothing.

**The gather is not free on the hot path.** The empirical constraint that birthed the layout
contract (Qwen Image/Edit quantization): disk-order ≠ memory-order cost *hundreds of seconds*
of byte re-ordering on a ~50 GB load; disk-order = memory-order made disk→VRAM ~10 seconds.
Read-time **rename** costs nothing at load; read-time **permute/fuse-reorder** re-introduces
exactly that reorder cost. So derived snapshots serve the non-hot direction and one-time
conversions; the canonical stored copy is laid out **load-order-optimal for the contract
actually served**. Two corollaries: manifest record order should equal the served contract's
memory order (sequential streaming, network→VRAM pipelining); and no seam mechanism may
compromise load order — chunk *boundaries* can move, byte *order* cannot.

The corpus audit (`research/checkpoint-variant-corpus-2026-08.md`, F5) bounds how often the
hot-path cost even arises: SDXL single-file↔diffusers is **95.4% of bytes rename-only**
(residue: one outer-axis CLIP-G qkv fuse of 0.315 GB + 4 MB of legacy reshapes), and
DiT-family spellings are ~100% rename — so deriving IS hot-path-safe for our families, and
the tiers below are a contingency, not a need.

### Dual-layout tiers, for pairs hot in both directions

Ranked by implementation cost against bytes + seconds saved:

- **Tier 1 — partial-dual storage (recommended default; near-zero implementation).** B's
  manifest `inherit()`s every rename-only tensor's chunks from A and admits real chunks only
  for the re-arranged tensors, transform paid once at definition time. Both directions boot
  at full sequential speed; cost = the re-arranged fraction (SDXL: 4.6% ≈ 0.32 GB; llama-GGUF:
  ~10%), never 2×. This is #80's composition applied to a transform — no new machinery.
- **Tier 2 — GPU-side re-arrangement (moderate implementation; dissolves tier-1 bytes for
  dense tensors).** The measured reorder penalty was CPU/disk-side; an in-VRAM permute runs
  at device memory bandwidth (~ms/GB — ~1 s on 50 GB, against a ~10 s load). Upload the
  stored orientation contiguously, permute on device before parameter assignment (we own the
  `from_config` + `load_state_dict` path; scratch = one largest-tensor buffer, freed per
  tensor). Honest limits: works for dense dtypes and for **block-aligned** permutes of
  quantized tensors (row permutations like the rope-permute move whole quant blocks);
  arbitrary element-level permutes of block-quantized tensors need dequant→requant — math,
  out of layout scope — and stay tier 1.
- **Tier 3 — stored deltas per class: mostly NO.** Pure transpose/permute shares no runs in
  either direction on disk (that is *why* tier 2 exists). Fuse/split and outer-axis slices
  decompose into shared runs, but seams already capture that at zero delta. Row-granular
  shuffles are expressible as record runs in principle, but rows are KB-scale — record
  cardinality explodes toward the 1M bound and read performance dies. Reject as a mechanism
  class; nothing here beats tiers 1+2.

Both are decidable mechanically from a layout-contract pair (same tensor multiset + dtypes +
element counts ⇒ viewable; run-preserving ⇒ also chunk-shareable). Consequence for the
adapter vocabulary in #81: plain `transpose` is too weak — the primitive must be
**generalized permute** (reshape → axis permutation → reshape), which subsumes transpose and
covers the rope-permute that real GGUF conversions perform. With it, even a llama.cpp GGUF's
permuted q/k are servable from the safetensors twin's objects.

A second consequence: for run-preserving contract pairs, a derived snapshot needs **no
program at all** — it is an ordinary TFM1 manifest whose record list points at existing
objects under new boundaries/names (exactly #80's rekey shape). Only non-run-preserving
transforms need an adapter program. GC needs zero new machinery either way: the derived
manifest's record list pins its sources through the existing mark walk.

## 4. Boundary sources: contract-directed splitting wins; store-state-directed is rejected

Three candidate sources for fusion-seam cuts inside a fused tensor:

1. **Static hint tables** (per-architecture: qkv, gate/up, `in_proj`). Deterministic and
   order-independent; a wrong hint costs sharing, never correctness (any partition is a valid
   plan; reassembly is concatenation regardless). Cost: table maintenance, drift on new
   architectures, and a planner-version bump whenever the table changes.
2. **Dedup-directed splitting** (consult the store's resident tensor sizes/digests when the
   fused file arrives). Tempting — hint-free and exact when the split twin arrived first — but
   it makes chunk boundaries a function of **store state**: the same file produces different
   record lists (hence different snapshot ids) on different stores, on the same store at
   different times, and before/after a GC. That destroys reproducible snapshot identity,
   makes the planner-vectors corpus unpinnable, and is order-dependent anyway (fused-first
   ingestion shares nothing). **Rejected for identity-bearing chunking.** The store may still
   *suggest* a contract when it notices a resident split twin — as UX, with the contract then
   recorded as the explicit input.
3. **Layout contracts** (Paul's framing): the contract already names the fusion seams, is an
   explicit, versioned, content-addressable **input** to ingestion, and must exist anyway for
   the adapter system. Boundaries = f(file bytes, declared contract) — deterministic and
   reproducible: same file + same contract ⇒ same snapshot id everywhere. Ingesting the same
   file under two contracts yields two ids, but that bifurcation is declared and auditable,
   unlike option 2's silent divergence.

**Recommendation: contract-directed splitting supersedes static tables** — the shipped
"tables" become a library of well-known contracts (data, not planner code), and the planner
grows one mechanism instead of an ever-growing architecture bestiary. Store-derived hints
stay rejected.

## 5. What was steelmanned and stays ruled out

**CDC / smaller grids within tensors.** The honest case for CDC is shifted-but-identical
runs: outer-axis fusion (contracts cover it exactly), vocab-extended embeddings (the extended
tensor shares its prefix — but such checkpoints are finetunes, so the other ~99% of bytes
differ anyway). Against: weight *changes* are dense — finetunes, LoRA-merges, casts and
re-quantizations alter essentially every byte, so content-defined boundaries find nothing;
meanwhile CDC surrenders the deterministic boundary spec and multiplies index cardinality.
Measured here: zero intra-file duplicate tensors; tied embeddings never even serialize twice.
If CDC ever earns a place it is in the **blob lane** (append-mostly datasets), as a future
blob-planner variant — never the tensor lane.

**Transpose-in-the-store.** No chunking scheme shares a permutation's bytes (a transposed
`[R,C]` scatters every contiguous run except along size-1 axes, where "transpose" is a
reshape and already byte-identical). Admission-time transform *detection* stays rejected as
#81 records; §3's view-expressibility is the whole answer.

## 6. Expected-bytes ranking — now corpus-measured, not speculated

The variant-corpus audit (`cozy-creator-tracker/research/checkpoint-variant-corpus-2026-08.md`)
measured the wild; cite it rather than re-deriving. Ranking updated with its numbers:

1. **Repack/rename twins** — the bulk of real duplication: SDXL ships two 6.94 GB single-files
   differing only in a 0.17 GB baked VAE (6.77 GB redundant per pair, F4); single-file↔tree
   pairs are 95.4–100% rename-only by bytes (F5). Already free; ingestion should re-pack
   foreign spellings into our canonical **single-file** packaging (Paul's no-sharding ruling)
   and keep the other spellings as derived views.
2. **Cross-model component sharing** — Wan 5B and A14B ship a byte-identical 11.4 GB UMT5-XXL
   (F1); realistic endpoint sets start 30–50% warm by bytes from chunk residency alone. Zero
   new design needed.
3. **Fuse/split seams** — 0.315 GB per SDXL pair in the wild (F5); up to ~52% of model bytes
   for ecosystems fusing qkv + gate/up (measured here). Contract-directed cuts (§4).
4. **Permuted twins (GGUF llama q/k, ~10%)** — view-expressible via generalized permute.
5. **Precision twins** — casts are math (out of scope by ruling); the corpus flags a
   provenance-link gap (F2's fp16-fix VAE under ≥4 whole-file digests, one byte-equal to the
   fp16 cast of another) — flagged there, not proposed here.
6. **Finetunes/merges/quants** — full-retrain SDXL tunes share ~0 with base (F3); 14 GGUF
   quant levels of one model are all distinct bytes (F7). Zero shareable; do not chase.
