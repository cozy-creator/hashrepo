#!/usr/bin/env bash
# Everything that runs ON the pod. Self-contained: installs a toolchain,
# builds tensorfs from the public repo, profiles the hardware, then runs the
# load-controlled read and write arms.
#
# It is deliberately noisy about what the machine actually is. A RunPod CPU
# container's storage is NOT the laptop's PCIe 4.0 NVMe, and absolute numbers
# from here should never be quoted as if it were; the value of a quiet box is
# that the RATIOS between the three paths are trustworthy.
set -uo pipefail

BRANCH="${BRANCH:-bench-pod-runner}"
SIZE_MIB="${SIZE_MIB:-1024}"
REPS="${REPS:-5}"
# Writes are as important as reads; they are never dropped for time. They just
# cost more per rep (every rep must use fresh bytes or the CAS deduplicates it
# away), so they get their own, smaller, count.
WREPS="${WREPS:-3}"
ROOT="${ROOT:-/workspace/bench}"
OUT="$ROOT/results"
REPO="$ROOT/tensorfs"

mkdir -p "$ROOT" "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== tensorfs pod benchmark: $(date -u +%FT%TZ) ==="

# ---------------------------------------------------------------- FUSE first
# If FUSE is unavailable the mount arms are impossible; find out now, not
# forty minutes into a build.
FUSE_OK=1
echo "--- FUSE availability ---"
if [ -e /dev/fuse ]; then echo "/dev/fuse: present"; else echo "/dev/fuse: MISSING"; FUSE_OK=0; fi
if command -v fusermount3 >/dev/null 2>&1; then
  echo "fusermount3: $(command -v fusermount3)"
else
  echo "fusermount3: MISSING (will try to install)"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git curl build-essential pkg-config fuse3 libfuse3-dev python3 >/dev/null 2>&1
command -v fusermount3 >/dev/null 2>&1 || { echo "fusermount3 STILL MISSING"; FUSE_OK=0; }
if [ "$FUSE_OK" = 1 ] && [ ! -w /dev/fuse ]; then
  echo "/dev/fuse not writable by $(id -un); mount arms will be skipped"
  FUSE_OK=0
fi
echo "FUSE_OK=$FUSE_OK"

# ------------------------------------------------------------- hardware profile
echo
echo "--- hardware profile ---"
{
  echo "## cpu"
  grep -m1 'model name' /proc/cpuinfo || true
  echo "cores(nproc): $(nproc)"
  grep -c ^processor /proc/cpuinfo | sed 's/^/siblings: /'
  echo "sha_ni: $(grep -m1 -o 'sha_ni' /proc/cpuinfo || echo absent)"
  echo
  echo "## memory"
  grep -E 'MemTotal|MemAvailable' /proc/meminfo
  echo
  echo "## block devices"
  lsblk -o NAME,SIZE,TYPE,ROTA,MODEL 2>/dev/null || echo "lsblk unavailable"
  for q in /sys/block/*/queue/rotational; do
    [ -r "$q" ] && echo "$q = $(cat "$q")"
  done
  echo
  echo "## filesystem under $ROOT"
  df -hT "$ROOT"
  mount | grep -E " $(df --output=target "$ROOT" | tail -1) " || true
  echo
  echo "## load at profile time"
  cat /proc/loadavg
} | tee "$OUT/hardware.txt"

echo
echo "--- raw dd reference (1 GiB, direct where supported) ---"
{
  dd if=/dev/zero of="$ROOT/ddtest.bin" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
  sync
  python3 -c "
import os
fd=os.open('$ROOT/ddtest.bin',os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)"
  dd if="$ROOT/ddtest.bin" of=/dev/null bs=1M 2>&1 | tail -1
} | tee "$OUT/dd_reference.txt"
rm -f "$ROOT/ddtest.bin"

# ------------------------------------------------------------------- toolchain
echo
echo "--- rust toolchain ---"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1090
. "$HOME/.cargo/env" 2>/dev/null || true
rustup update stable >/dev/null 2>&1 || true
rustc --version && cargo --version

# ------------------------------------------------------------------ build
echo
echo "--- build ---"
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 --branch "$BRANCH" https://github.com/cozy-creator/tensorfs.git "$REPO" || {
    echo "FATAL: clone failed"; exit 1; }
fi
cd "$REPO"
git log --oneline -1
cargo build --release --bin tensorfsd --example podbench -p tensorfsd -p tensorfs-core 2>&1 | tail -5
cargo build --release --example podbench -p tensorfs-core 2>&1 | tail -3
TFSD="$REPO/target/release/tensorfsd"
PODBENCH="$REPO/target/release/examples/podbench"
[ -x "$TFSD" ] || { echo "FATAL: tensorfsd not built"; exit 1; }
[ -x "$PODBENCH" ] || { echo "FATAL: podbench not built"; exit 1; }

# ------------------------------------------------------------------ fixture
echo
echo "--- fixture (${SIZE_MIB} MiB) ---"
WORK="$ROOT/work"
rm -rf "$WORK"; mkdir -p "$WORK/store" "$WORK/mnt" "$WORK/wsmnt"
"$PODBENCH" setup "$WORK/store" "$WORK/native.bin" "$SIZE_MIB" | tee "$OUT/setup.txt"
SNAPSHOT=$(grep '^SNAPSHOT=' "$OUT/setup.txt" | cut -d= -f2)
OBJECTS=$(grep '^OBJ=' "$OUT/setup.txt" | cut -d= -f2- | paste -sd,)
echo "snapshot=$SNAPSHOT  objects=$(grep -c '^OBJ=' "$OUT/setup.txt")"

# -------------------------------------------------------------------- mount
MOUNT_FILE="$WORK/mnt/model.bin"
MOUNT_PID=""
WS_PID=""
cleanup() {
  [ -n "$MOUNT_PID" ] && kill "$MOUNT_PID" 2>/dev/null
  [ -n "$WS_PID" ] && kill "$WS_PID" 2>/dev/null
  sleep 1
  fusermount3 -u "$WORK/mnt" 2>/dev/null
  fusermount3 -u "$WORK/wsmnt" 2>/dev/null
}
trap cleanup EXIT

if [ "$FUSE_OK" = 1 ]; then
  "$TFSD" mount-snapshot --store "$WORK/store" --snapshot "$SNAPSHOT" "$WORK/mnt" \
    >"$OUT/mount_snapshot.log" 2>&1 &
  MOUNT_PID=$!
  for _ in $(seq 1 40); do [ -r "$MOUNT_FILE" ] && break; sleep 0.25; done
  if [ -r "$MOUNT_FILE" ]; then
    echo "snapshot mounted (pid $MOUNT_PID): $(stat -c %s "$MOUNT_FILE") bytes"
  else
    echo "mount FAILED; see mount_snapshot.log"; cat "$OUT/mount_snapshot.log"; FUSE_OK=0
  fi
fi
if [ "$FUSE_OK" != 1 ]; then
  echo "NOTE: mount arms unavailable on this pod; native + bypass only."
  MOUNT_FILE="$WORK/native.bin"   # bench.py needs a path; rows get labelled below
fi

# --------------------------------------------------------------- READ ARMS
echo
echo "=================== READ (load-controlled, round-robin) ==================="
python3 "$REPO/benchmarks/pod/bench.py" \
  "$WORK/native.bin" "$MOUNT_FILE" "$OBJECTS" "$REPS" "$OUT/reads.json"

# -------------------------------------------------------------- WRITE ARMS
echo
echo "============= WRITE (load-controlled, round-robin) ============="
WSMNT="-"
if [ "$FUSE_OK" = 1 ]; then
  "$TFSD" mount-workspace --store "$WORK/store" --workspace main "$WORK/wsmnt" \
    >"$OUT/mount_workspace.log" 2>&1 &
  WS_PID=$!
  for _ in $(seq 1 40); do [ -d "$WORK/wsmnt" ] && mountpoint -q "$WORK/wsmnt" && break; sleep 0.25; done
  if mountpoint -q "$WORK/wsmnt"; then
    WSMNT="$WORK/wsmnt"
    echo "workspace mounted (pid $WS_PID)"
  else
    echo "workspace mount FAILED"; cat "$OUT/mount_workspace.log"; WS_PID=""
  fi
fi

rm -rf "$WORK/wstore"; mkdir -p "$WORK/wstore"
"$PODBENCH" setup "$WORK/wstore" "$WORK/seed.bin" 1 >/dev/null 2>&1

python3 "$REPO/benchmarks/pod/writebench.py" \
  "$WORK" "$WSMNT" "${WS_PID:--}" "$SIZE_MIB" \
  "$PODBENCH" "$WORK/wstore" "$OUT/writes.json" "$WREPS"

echo
echo "=== done: $(date -u +%FT%TZ) ==="
echo "results in $OUT"
