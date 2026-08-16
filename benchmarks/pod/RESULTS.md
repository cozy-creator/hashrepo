# Read/write tables from a dedicated CPU pod

Run 2026-08-15 on a RunPod CPU pod provisioned solely for this measurement and
terminated immediately afterwards.

## Why these numbers are trustworthy and the laptop's were not

The development laptop swings **13×** on the same `dd` command purely with
system load. On this pod the 1-minute load average sat at **0.85 and did not
move** across the entire read run, and the cold-read spread came in at **0.6%**
(3705–3728 MiB/s over five reps). That is the difference the pod bought.

## Hardware

| | |
|---|---|
| CPU | AMD EPYC 4564P, 16 cores / 32 threads, `sha_ni` present |
| RAM | 124 GiB |
| Disk | Samsung SSD 990 EVO Plus 2TB, **local NVMe**, `rotational=0` — *not* network-attached |
| Filesystem | `overlay` (Docker overlay2) on the NVMe |
| `dd` reference | 4.0 GB/s write, 3.6 GB/s read (buffered, 1 GiB) |

The disk is real local NVMe, so these figures are in the same class as the
laptop's PCIe 4.0 NVMe rather than a network volume. They are still a
*different* drive on a *different* CPU — see the caveat at the bottom.

## Reads — 1024 MiB payload, 5 reps, load 0.85 throughout

| path | phase | median | spread | method |
|---|---|---|---|---|
| native | cold | **3722 MiB/s** (3.63 GiB/s) | 3705 – 3728 | O_DIRECT |
| native | warm | **23503 MiB/s** (22.9 GiB/s) | 22810 – 24744 | buffered |
| CAS bypass | cold | **3705 MiB/s** (3.62 GiB/s) | 3697 – 3707 | O_DIRECT |
| CAS bypass | warm | **19890 MiB/s** (19.4 GiB/s) | 19541 – 20471 | buffered |
| FUSE mount | cold/warm | **not measurable** — see below | | |

- **Cold bypass / native = 1.00×.** Splitting a file into sixteen 64 MiB CAS
  objects costs *nothing* on a cold read. This is the headline: the CAS layout
  is free at the storage boundary.
- **Warm bypass / native = 0.85×.** Serving 1 GiB from page cache across 16
  separate opens is ~15% slower than one open. Both numbers are far above any
  plausible model-loading demand.

## Writes — 1024 MiB payload, 3 reps

| arm | median | spread | amplification |
|---|---|---|---|
| native (write + fsync) | **3109 MiB/s** | 2970 – 3148 | **1.00×** |
| CAS direct ingest (bypass) | **414 MiB/s** | 406 – 430 | **1.00×** |
| through the FUSE mount | **not measurable** | | |

Repeated under load ~6.0 the same arms gave 3151 and 390 MiB/s — within the
quiet run's spread, so the write arms are not load-sensitive at this level.

- **The bypass write lane has no amplification at all (1.00×).** It writes each
  object's bytes exactly once. The 3.00× belongs specifically to the FUSE
  mount's spill→compose→journal path, not to the CAS.
- **Direct ingest is SHA-256-bound at 414 MiB/s**, 7.5× slower than a plain
  write, single-threaded, on a core that *has* SHA-NI. This is the real write
  bottleneck for the bypass lane and it is embarrassingly parallel across
  objects — sixteen 64 MiB objects could hash concurrently. That optimisation
  is not implemented and is the obvious next win.

## FUSE could not be measured on RunPod

RunPod CPU containers cannot mount FUSE, established three independent ways:

1. `/dev/fuse` is absent. `mknod` *succeeds* (the container holds `CAP_MKNOD`),
   producing a mode-0666 node — but opening it is still denied by the device
   cgroup, so `[ -w /dev/fuse ]` is false even for root.
2. `CAP_SYS_ADMIN` is absent from `CapEff` **and from `CapBnd`**. Since it is
   dropped from the *bounding* set, even setuid-root `fusermount3` cannot
   acquire it.
3. The empirical check: `tensorfsd mount-snapshot` fails with `mount I/O
   failed`, and `fusermount3 -u` reports `Operation not permitted`.

`/proc/sys/vm/drop_caches` is also read-only in the container.

So the mount rows in this table are genuinely absent rather than estimated.
An earlier version of the runner pointed the mount lane at the native file
when FUSE was unavailable, which emitted "mount" rows that were really native
rows measured twice; that fabrication is fixed — the lane is now omitted.

**The laptop's 3.00× mount write amplification is therefore NOT independently
confirmed on separate hardware.** It has been measured twice on the laptop
(4096 MiB → 12288 MiB, and again at 64 MiB → 192 MiB during this harness's
validation, both exactly 3.00×), but both measurements are the same machine.
Amplification is hardware-agnostic in principle, so it *should* hold — but
that remains an argument, not a measurement. Confirming it needs a box that
grants `CAP_SYS_ADMIN`: a bare-metal host, a privileged container, or a VM.

## A measurement bug worth recording

The first pod run reported cold CAS-bypass reads at 7816 MiB/s — **2.31×
faster than native, and faster than the drive itself**. That was not a
speedup, it was a broken measurement.

`posix_fadvise(POSIX_FADV_DONTNEED)` evicted the 1 GiB native file completely
but plateaued at exactly 256 MiB of the 1 GiB 16-object CAS store no matter
how many sync+evict passes ran — 75% of the "cold" read was served from page
cache. Cold rows now use **O_DIRECT**, which cannot be contaminated, and every
row reports the bytes that actually reached the disk (`read_bytes` from
`/proc/self/io`); a cold row that falls short is flagged and excluded from the
medians. Under O_DIRECT both lanes read a full 1024/1024 MiB from disk and the
ratio settles at the honest 1.00×.

The general lesson: a read that beats the device it sits on is always a bug.

## Caveats

- Different drive and CPU from the laptop; absolute figures should not be
  quoted as Paul's machine's numbers. The **ratios** are what transfer.
- The filesystem is Docker `overlay`, not a bare ext4/xfs mount.
- No FUSE arm at all, per above.
- Payload is 1 GiB. The optional 50 GiB matrix was not run.

## Reproducing

```sh
SIZE_MIB=1024 REPS=5 WREPS=3 bash run_on_pod.sh
```

Raw output is in `results/`: `hardware.txt`, `dd_reference.txt`, `reads.json`,
`writes.json` (loaded), `writes_quiet.json` (quiet), `run.log`.
