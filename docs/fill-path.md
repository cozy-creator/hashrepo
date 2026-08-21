# The fill path, and the varena seam

tensorfs owns **the plan and the transform**. varena owns **the address space
and the byte-motion mechanics**. The join is one trait.

## Who owns what, and why it is not what tensorfs#154 first said

`tensorfs#154` reads "tensorfs owns the byte-motion pipeline disk → host →
device". Taken literally that duplicates work that already exists and is
already measured: varena's `refill.rs` is disk (io_uring) → budgeted pinned
slab → async H2D on a copy stream, with dual destinations and chunk recycling,
and pgw#1607 measured it 93–99% H2D-bound. A second one here would be a worse
copy of it.

What varena's refill plane **structurally cannot** do is a morphism: it moves
contiguous `(file, offset, len)` ranges, and a rearrangement is not a range
map. That is the part tensorfs owns, and it is the part the one-implementation
rule is actually about.

So:

| | owner |
|---|---|
| which byte goes where | `tensorfs_core::layout_morphism` — one walk, both directions |
| reading chunks and applying the arrangement | `tensorfs_core::layout_fill::fill` |
| the destination address | the caller, through `FillSink` |
| reservations, backing, stable VAs, paging | varena |
| the production disk and H2D legs | varena's refill plane |

`tensorfs-core` never allocates device memory and is `#![forbid(unsafe_code)]`.
Every `unsafe` in this program lives in `tensorfs-cuda`, a crate whose whole job
is six driver symbols; it is the REFERENCE sink, so the device leg can be proved
here, not the fleet's sink.

## Why the transform runs during the host staging pass

A morphism decomposes into contiguous runs, and the count is computed
(`Plan::run_count`), never assumed:

| arrangement | runs | what that means |
|---|---|---|
| `torch.contiguous@1` | 1 | a whole tensor is one DMA; the fill path is free |
| `cublas.blockscale-128x4@1` | ~1 per 4 elements | short runs, still a gather |
| `torch.channels_last-2d@1` | ~1 per element | its innermost storage axis has source stride H*W |

Handing the last case to a scatter-gather engine is tens of millions of
element-sized transfers. The bytes are already being copied once — chunks to
pinned staging — so the permutation rides *that* copy and the device leg stays
one contiguous DMA. `FillStats::runs_per_element` reports which case a tensor
was, so nobody has to re-derive it.

The decomposition is computed from the strides, not discovered by walking. That
is load-bearing: the first version discovered runs by visiting elements and
noticing adjacency, and measured **16 MiB in 765 ms** for a plan that is one
`memcpy` plus one DMA, because it stepped an odometer four million times to
learn what the strides already said.

## Measured, 4070 Laptop, release, best of 10

| leg | number |
|---|---|
| identity fill, 16 MiB | 3.03 ms = **5.15 GiB/s** (staging memcpy + H2D, one run) |
| channels_last [256,128,3,3] | 9.08 ms for 1.125 MiB = **0.12 GiB/s**, ~31 ns/element |

Both legs are the same walk. The gap between them is the whole story: an
arrangement that folds into long runs is essentially free, and a per-element
gather is not. **The per-element class is not yet fast enough to call the
transform free** — extrapolated, SDXL's 635 MiB conv-weight set is ~5 s of host
time — so the design's "the permutation rides the existing H2D copy at near-zero
marginal cost" is true for the identity, near-true for short-run arrangements,
and not yet true for the per-element class here. Do not quote it as a measured
fact for that class.

Filed as **tensorfs#157**, and filed as LOW: the compile-levers work measured the
inductor-class win this unlocks at only ~1%/step on SDXL, which is a bad trade
against a one-time 5 s host cost. What motivates it is the kernel-library class
(`cublas.blockscale-128x4@1`, `nunchaku.micro-scale@1`), where the win is real —
and those already fold into short runs, so nothing on the fleet is blocked by
this number today.

Single samples of the identity leg spread 2.8/4.4/5.6 ms on this
desktop-driving card, which is why the legs report best of 10 rather than one
number that looks precise.

## Running the on-card tests

Skipped unless sanctioned, and a skip says so:

```
TENSORFS_GPU_WINDOW=1 cargo test -p tensorfs-cuda --test gpu -- --nocapture
```

`TENSORFS_GPU_WINDOW=1` is a coordinator-granted window on the shared box;
`TENSORFS_GPU=1` is for exclusive pods. The gate's own truth table is tested on
CPU, in every run.
