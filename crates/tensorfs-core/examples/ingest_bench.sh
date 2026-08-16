#!/usr/bin/env bash
# Interleaved before/after for the direct ingest lane.
#
# Method notes that make the numbers comparable:
#  - ROUND-ROBIN, not batched: one rep of every arm back to back, so a load
#    excursion hits all arms roughly equally instead of poisoning one.
#  - FRESH STORE per measurement. A repeat run into a warm store hits the
#    resident-object path, which rehashes AND re-verifies -- more work, not
#    less -- so reusing a store would silently measure something else.
#  - The 1-minute load is printed before and after every measurement by the
#    binary itself, so a row taken under a spike is visible, not hidden.
#  - Wall-clock is REPORTED. Nothing here is a gate.
#
# Worker count comes from TENSORFS_ASSEMBLY_BUDGET_BYTES / 64 MiB, clamped by
# core count, so concurrency is varied from out here rather than by the
# binary calling the unsafe set_var.
set -euo pipefail
cd "$(dirname "$0")/../../.."

WORK="${WORK:-/tmp/tensorfs-ingest-bench-$$}"
SIZE_MIB="${SIZE_MIB:-1024}"
REPS="${REPS:-3}"
BIN=target/release/examples/ingest_bench
MIB=$((1024 * 1024))

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

echo "=== building (nice -n 19, -j 2) ==="
nice -n 19 cargo build --release -j 2 --example ingest_bench -p tensorfs-core

echo "=== fixture: ${SIZE_MIB} MiB ==="
# Deterministic and poorly compressible: no layer may cheat by collapsing runs.
nice -n 19 python3 - "$WORK/native.bin" "$SIZE_MIB" <<'PY'
import sys
path, size_mib = sys.argv[1], int(sys.argv[2])
block = bytes(((i * 2654435761) >> 13) & 0xFF for i in range(1 << 20))
with open(path, "wb") as handle:
    for round_index in range(size_mib):
        handle.write(bytes([round_index & 0xFF]) + block[1:])
PY
ls -l "$WORK/native.bin"

run_arm() { # <label> <subcommand> <budget-bytes-or-default>
  local label="$1" subcommand="$2" budget="$3"
  local store="$WORK/store-$label-$RANDOM"
  mkdir -p "$store"
  if [ "$budget" = "default" ]; then
    nice -n 19 "$BIN" "$subcommand" "$store" "$WORK/native.bin"
  else
    TENSORFS_ASSEMBLY_BUDGET_BYTES="$budget" nice -n 19 \
      "$BIN" "$subcommand" "$store" "$WORK/native.bin"
  fi | sed "s/^/${label}\t/"
  rm -rf "$store"
}

echo "=== interleaved arms, ${REPS} reps ==="
for rep in $(seq 1 "$REPS"); do
  echo "--- rep $rep ---"
  run_arm "double-hash-serial" double default
  run_arm "single-pass-serial" single $((64 * MIB))
  run_arm "single-pass-parallel" single default
done
