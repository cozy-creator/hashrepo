#!/usr/bin/env python3
"""Write arms: native, CAS-direct ingest, and through the FUSE mount.

Same discipline as the read bench: ROUND-ROBIN one rep of each path back to
back, /proc/loadavg sampled before AND after every measurement, >=3 reps,
median plus full spread. A load excursion then lands on all three arms rather
than poisoning one.

Two numbers per arm:

* **throughput** -- wall clock, so it moves with the machine.
* **byte amplification** -- ``write_bytes`` from ``/proc/<pid>/io`` over
  logical bytes. This is hardware-agnostic: it counts what actually reaches
  storage per byte the caller asked to store. The laptop measured the mount
  path at exactly 3.00x (spill file, then composed object, plus journalling).
  If the pod disagrees, that is reported loudly, not reconciled -- one of the
  two measurements would then be wrong and it matters which.

Two things that keep the rows honest:

* All three arms in a rep write the IDENTICAL payload bytes.
* Bytes carry a per-rep stamp. Without it the CAS would deduplicate rep 2
  against rep 1 and report a spectacular, meaningless ~0-byte write.
* fsync completes before the clock stops, and the mount arm additionally
  waits for the daemon's own write counter to go quiet -- composition happens
  after release, so sampling too early undercounts the amplification that is
  the whole point of the row.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

BLOCK = 1 << 20
MIB = 1048576.0


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def proc_io(pid) -> tuple[int, int]:
    """(read_bytes, write_bytes) — bytes that actually hit storage."""
    if pid in (None, "-", ""):
        return (0, 0)
    try:
        text = Path(f"/proc/{pid}/io").read_text()
    except OSError:
        return (0, 0)
    read = write = 0
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            read = int(line.split()[1])
        elif line.startswith("write_bytes:"):
            write = int(line.split()[1])
    return read, write


def evict(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def settle(pid, quiet_for: float = 1.0, cap: float = 60.0) -> int:
    """Wait for a process's write counter to stop moving; return bytes added.

    Composition into CAS objects happens after release, so the daemon keeps
    writing past the point the caller's fsync returns.
    """
    if pid in (None, "-", ""):
        return 0
    _, start = proc_io(pid)
    last = start
    stable_since = time.monotonic()
    deadline = time.monotonic() + cap
    while time.monotonic() < deadline:
        time.sleep(0.1)
        _, now = proc_io(pid)
        if now != last:
            last = now
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= quiet_for:
            break
    return last - start


def make_payload(size_mib: int, rep: int) -> bytes:
    """Deterministic, poorly-compressible, and unique per rep."""
    block = bytes(((i * 2654435761) >> 13) & 0xFF for i in range(BLOCK))
    out = bytearray()
    for round_ in range(size_mib):
        chunk = bytearray(block)
        chunk[0] = round_ & 0xFF
        chunk[1] = rep & 0xFF  # defeats cross-rep CAS dedup
        out.extend(chunk)
    return bytes(out)


def write_file(path: Path, payload: bytes) -> tuple[float, int, int]:
    """Write every byte, fsync, close. Returns (seconds, d_read, d_write)."""
    r0, w0 = proc_io("self")
    view = memoryview(payload)
    start = time.monotonic()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        for offset in range(0, len(payload), BLOCK):
            chunk = view[offset : offset + BLOCK]
            written = 0
            while written < len(chunk):
                written += os.write(fd, chunk[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    secs = time.monotonic() - start
    r1, w1 = proc_io("self")
    return secs, r1 - r0, w1 - w0


def emit(rows, rep, arm, secs, logical, wrote, read, pre, post, note=""):
    mibs = logical / secs / MIB if secs > 0 else 0.0
    amp = wrote / logical if logical else 0.0
    rows.append(
        {
            "rep": rep,
            "arm": arm,
            "mib_s": round(mibs, 1),
            "secs": round(secs, 3),
            "logical_bytes": logical,
            "written_bytes": wrote,
            "read_bytes": read,
            "amplification": round(amp, 3),
            "load_pre": pre,
            "load_post": post,
            "note": note,
        }
    )
    print(
        f"{rep:>3}  {arm:<14} {mibs:>8.0f} MiB/s"
        f"  {logical / MIB:>7.0f} logical"
        f"  {wrote / MIB:>8.0f} written"
        f"  {amp:>6.2f}x"
        f"   load {pre:>5.2f} -> {post:>5.2f}  {note}",
        flush=True,
    )


def main() -> int:
    scratch = Path(sys.argv[1])
    mount_dir = Path(sys.argv[2]) if sys.argv[2] != "-" else None
    daemon_pid = sys.argv[3] if sys.argv[3] not in ("-", "") else None
    size_mib = int(sys.argv[4])
    podbench = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    store_root = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    out_path = Path(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "-" else None
    reps = int(sys.argv[8]) if len(sys.argv) > 8 else 3

    rows: list[dict] = []
    arms = ["native"]
    if podbench and store_root:
        arms.append("direct-ingest")
    if mount_dir is not None:
        arms.append("mount")

    print(f"payload={size_mib} MiB   reps={reps}   arms={','.join(arms)}")
    print(f"daemon pid for mount amplification: {daemon_pid or 'n/a'}")
    print(f"start load {load1():.2f}\n")
    print(
        f"{'rep':>3}  {'arm':<14} {'throughput':>13}"
        f"  {'MiB':>7}         {'MiB':>8}     amp   load"
    )

    for rep in range(1, reps + 1):
        payload = make_payload(size_mib, rep)
        logical = len(payload)
        native_file = scratch / f"native_write_{rep}.bin"

        for arm in arms:
            if arm == "native":
                pre = load1()
                secs, dread, dwrite = write_file(native_file, payload)
                emit(rows, rep, "native", secs, logical, dwrite, dread, pre, load1())

            elif arm == "direct-ingest":
                # Evict first so read_bytes reflects real source I/O rather
                # than the page cache left warm by the native arm.
                evict(native_file)
                pre = load1()
                proc = subprocess.run(
                    [podbench, "direct-ingest", store_root, str(native_file)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                parsed = {}
                for line in proc.stdout.splitlines():
                    if "=" in line:
                        key, _, value = line.partition("=")
                        parsed[key] = value
                if "MIB_S" in parsed:
                    secs = float(parsed.get("WALL_S", 0)) or 1e-9
                    emit(
                        rows,
                        rep,
                        "direct-ingest",
                        secs,
                        int(parsed.get("LOGICAL_BYTES", logical)),
                        int(parsed.get("PROC_WRITE_BYTES", 0)),
                        int(parsed.get("PROC_READ_BYTES", 0)),
                        pre,
                        load1(),
                    )
                else:
                    print(f"{rep:>3}  direct-ingest  FAILED: {proc.stderr.strip()[:160]}")

            else:  # mount
                dr0, dw0 = proc_io(daemon_pid)
                pre = load1()
                secs, self_read, self_write = write_file(
                    mount_dir / f"written_{rep}.bin", payload
                )
                # Composition continues past our fsync; wait for it.
                tail = settle(daemon_pid)
                dr1, dw1 = proc_io(daemon_pid)
                daemon_write = dw1 - dw0
                daemon_read = dr1 - dr0
                emit(
                    rows,
                    rep,
                    "mount",
                    secs,
                    logical,
                    self_write + daemon_write,
                    self_read + daemon_read,
                    pre,
                    load1(),
                    note=f"(writer {self_write / MIB:.0f} + daemon {daemon_write / MIB:.0f} MiB,"
                    f" {tail / MIB:.0f} after fsync)",
                )

        for stale in (native_file,):
            try:
                stale.unlink()
            except OSError:
                pass

    print(f"\nend load {load1():.2f}")

    print("\n=== summary (median of reps, [min-max] spread) ===")
    print(f"{'arm':<14} {'median MiB/s':>13} {'spread':>18} {'median amp':>12} {'amp spread':>18}")
    summary = {}
    for arm in arms:
        vals = [r["mib_s"] for r in rows if r["arm"] == arm]
        amps = [r["amplification"] for r in rows if r["arm"] == arm]
        if not vals:
            continue
        summary[arm] = {
            "median_mib_s": statistics.median(vals),
            "min_mib_s": min(vals),
            "max_mib_s": max(vals),
            "median_amplification": statistics.median(amps),
            "min_amplification": min(amps),
            "max_amplification": max(amps),
        }
        print(
            f"{arm:<14} {statistics.median(vals):>13.0f}"
            f" {f'[{min(vals):.0f} - {max(vals):.0f}]':>18}"
            f" {statistics.median(amps):>11.2f}x"
            f" {f'[{min(amps):.2f} - {max(amps):.2f}]':>18}"
        )

    # The laptop's headline claim, checked independently on quiet hardware.
    if "mount" in summary:
        med = summary["mount"]["median_amplification"]
        print(f"\nmount write amplification on this pod: {med:.2f}x  (laptop measured 3.00x)")
        if abs(med - 3.0) <= 0.15:
            print("  -> CONFIRMS the laptop's 3.00x. Amplification is hardware-agnostic, as expected.")
        else:
            print(
                f"  -> DISAGREES with the laptop's 3.00x by {abs(med - 3.0):.2f}x."
                " Amplification should not vary with hardware, so one of the two"
                " measurements is wrong and this needs chasing down before either"
                " number is quoted."
            )

    if out_path:
        out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
        print(f"\nrows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
