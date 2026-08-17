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
requires a new `version`; `crates/tensorfs-core/tests/contract_registry.rs`
pins each document's digest so an in-place edit fails CI. This is what makes a
stamped snapshot reproducible: same file + same `contract@version` ⇒ same
snapshot id, on every store, forever.

## Fields

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
points and the tensor grids plainly. This is the memo's rejection of
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

## Matching and tie-break

Identification reads the header inventory only (names, shapes, dtypes; no
tensor bytes). Among all contracts that match, the winner is:

1. **most specific** — explains the most of the file's tensors,
2. then **highest version**,
3. then the lexicographically smallest name.

Key 3 exists so the answer never depends on registry insertion order. The
winner is recorded in the snapshot as `contract@version`; no match is recorded
as `none` and chunks on the plain per-tensor grid.
