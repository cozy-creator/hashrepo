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
actually served**, and a pair that is hot in *both* directions pays real bytes for its
re-arranged tensors — a deliberate speed-over-storage trade, not a dedup failure. Two
corollaries: manifest record order should equal the served contract's memory order
(sequential streaming, network→VRAM pipelining); and no seam mechanism may compromise load
order — chunk *boundaries* can move, byte *order* cannot.

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

## 6. Expected-bytes ranking (where the redundant bytes actually are)

1. **Repack/rename/shard twins** — the bulk of real-world duplication (ComfyUI single-file vs
   diffusers trees vs HF reshards of the *same* weights). Already 100% free; needs only #69's
   projection so both spellings are cheap to expose.
2. **Fuse/split seams** — 12–52% of model bytes per affected pair; contract-directed cuts (§4).
3. **Permuted twins (GGUF llama q/k, ~10%)** — view-expressible via generalized permute; free
   once #81's adapters land.
4. **Precision twins** — large bytes, but casts are math: real conversion pipeline, out of
   dedup's scope by ruling.
5. **Finetunes/merges/quants** — genuinely distinct weights; zero shareable; do not chase.
