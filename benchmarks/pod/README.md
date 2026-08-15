# Pod benchmarks

Load-controlled read and write tables for the three data paths, run on a
dedicated quiet machine.

## Why a pod

The development laptop swung **13×** on the *same* `dd` command purely with
system load (1.6 GB/s at load 18 → 122 MB/s at load 31). Absolute throughput
from a box running several agent sessions is not a measurement, and even
ratios are suspect when two arms are timed minutes apart. A dedicated CPU pod
costs a few cents and removes the variable.

## The three paths

| path | what it measures |
|---|---|
| `native` | plain file on the filesystem — the ceiling |
| `bypass` | reading/writing CAS objects directly, no FUSE in the path |
| `mount` | through the `tensorfsd` FUSE3 mount |

## Method

Both drivers use the same discipline:

- **Round-robin, not batched.** One rep of every path back to back, so a load
  excursion lands on all paths roughly equally instead of poisoning one arm.
- **`/proc/loadavg` sampled before *and* after every single measurement**, and
  printed on the row. A row taken under moving load is visible, not averaged
  away.
- **Identical payload bytes** across all paths within a rep.
- **≥3 reps**, reported as median plus full `[min – max]` spread. Never a
  single number.
- Reads: true cold means `POSIX_FADV_DONTNEED` on the target *and* every
  backing CAS object — otherwise a "cold" mount read is served from its
  objects' page cache and measures nothing.
- Writes: `fsync` completes before the clock stops, and the mount arm
  additionally waits for the daemon's write counter to go quiet, because
  composition into CAS objects continues past the caller's `fsync`.
- Write payloads carry a per-rep stamp byte. Without it the CAS would
  deduplicate rep 2 against rep 1 and report a spectacular, meaningless
  ~0-byte write.

## Write amplification

The headline write number is **not** a throughput but a ratio of counters:
`write_bytes` from `/proc/<pid>/io` over logical bytes written. Amplification
is hardware-agnostic, so it is the one figure that transfers between machines.
The mount path's amplification is paid by the *daemon*, not the writer, so
both processes' counters are summed.

## Files

- `bench.py` — read arms
- `writebench.py` — write arms
- `run_on_pod.sh` — the whole on-pod run: hardware profile, FUSE check, build,
  fixture, both benchmarks
- `../../crates/tensorfs-core/examples/podbench.rs` — fixture builder and the
  direct-ingest write lane

## Running

```sh
# on the pod
SIZE_MIB=1024 REPS=5 WREPS=3 bash run_on_pod.sh
```

## Reading the results

Pod storage is **not** the development laptop's PCIe 4.0 NVMe, and may be
network-attached — check `results/hardware.txt` before quoting any absolute
figure. What transfers between machines is the **ratios between paths** and
the **amplification factors**.
