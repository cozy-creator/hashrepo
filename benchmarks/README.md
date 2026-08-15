# The pgw#1256 measured matrix

`tensorfs-bench` (a Linux-only bin in `crates/tensorfsd`) drives the measured
matrix that is TensorFS's performance and release authority — and the named
trigger for the SQLite→redb metadata-engine swap. It emits one JSONL evidence
stream per run; the row schema lives in `tensorfsd::bench` and is unit-tested
(round-trip, required provenance, the `new + reused == total` accounting
identity, summary folds). Nothing asserts wall-clock floors and CI never runs
the bin: rows are read by people, off a quiet host.

## Row stream

One `meta` row (schema, run id, scale, reps, kernel, filesystem type, hashed
machine id, crate version, start time — every field required), then one `arm`
row per (arm, repetition) with wall/user/sys seconds, process VmHWM,
`/proc/self/io` byte deltas, driver-issued fsync counts, object/byte
accounting, the 1-minute load at arm start and a `load_caveat` flag above 4.0,
then one `summary` row per arm (min/median/max wall).

## Arms

| arm | what it measures |
|---|---|
| `fixture` | corpus generation — timed apart, never a TensorFS number |
| `native_read`, `sha256_floor` | the honesty baselines over the same bytes |
| `import` | cold create+write+fsync of the corpus through a real mount, one fresh store per repetition |
| `seal_reboundary` / `seal_fixpoint` | the planner re-slice at first seal; the no-op reseal |
| `clone` | snapshot→workspace clone (metadata only; zero objects expected) |
| `write_ops` | create/pwrite/truncate/fsync/unlink micro-ops through the mount |
| `overwrite_8k` | the headline COW claim: one small edit, touched objects only |
| `seq_rewrite` | full-file sequential rewrite through the mount |
| `semantic_reuse_safetensors` / `semantic_reuse_gguf` | one-tensor 8 KiB edit, reseal, record-level diff against the baseline snapshot: exact reused/new objects and bytes |
| `ten_workspaces` | ten clones with ten distinct edits: logical vs physical bytes |
| `remount_cold_read` | reopen the store, mount the snapshot, read a file cold |

## Smoke result committed here

`results/` holds a smoke run at small scale from a **shared, loaded box**,
using the **debug profile**. It exists to prove the harness runs end to end
and to freeze the row schema — the numbers themselves decide nothing, and
several rows carry `load_caveat: true`. Do not quote them as performance.

## The real-scale run (the actual release gate)

On a quiet, owned Linux host with ~120 GiB disposable disk and nothing else
running:

    cargo build --release -p tensorfsd --bins
    nice -n 19 target/release/tensorfs-bench run --scale 53687091200 --reps 3 \
        --out /var/tmp/tensorfs-matrix

Requirements: 1-minute load below 1 before starting (every arm records a
caveat above 4.0 regardless); ext4 or the production filesystem; three
repetitions. The 50 GiB corpus plus the main store plus one throwaway import
store peak near 110 GiB.

Boxes this discharges when run at real scale on a quiet host: the
before/after arms, the safetensors/GGUF semantic-reuse arms, the
ten-workspace physical-allocation arm ("about 59 GiB, never 559 GiB"), the
8 KiB-overwrite object-accounting arm, remount/restart, and the
instrumentation-counting box. The crash-matrix, mmap, and FUSE-throughput
boxes belong to their own suites. **Until that run exists, every pgw#1256
checkbox stays unchecked** — this harness makes the run possible; it is not
the run.
