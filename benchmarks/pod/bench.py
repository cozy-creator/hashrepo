#!/usr/bin/env python3
"""Load-controlled read comparison: native vs CAS-bypass vs FUSE mount.

Why this is shaped the way it is
--------------------------------
The laptop these numbers were first attempted on swung 13x on the SAME `dd`
command purely with system load, which makes absolute throughput unusable and
even ratios suspect when the two sides are measured minutes apart. So:

* **Round-robin, not batched.** One rep of every path back to back, so a load
  excursion lands on all three roughly equally instead of poisoning one arm.
* **Load sampled before AND after every single measurement**, printed on the
  row. A row whose load moved is visible rather than silently averaged in.
* **Identical payload bytes** for all three paths, identical 1 MiB read size.
* Reported as median and full spread across reps, never a single number.

Getting "cold" right
--------------------
``posix_fadvise(POSIX_FADV_DONTNEED)`` is NOT sufficient. Measured on an
overlayfs container it evicted a 1 GiB single file completely but plateaued at
exactly 256 MiB of a 1 GiB 16-object CAS store no matter how many passes were
run -- so 75% of the "cold" read was served from page cache and the CAS lane
looked 2.3x FASTER than the disk it sits on. A read that beats the device is
not a fast read, it is a broken measurement.

So cold rows use **O_DIRECT**, which bypasses the page cache outright and
cannot be contaminated. Every row also reports the bytes that actually reached
the disk (``read_bytes`` from ``/proc/self/io``); a cold row whose disk_read
falls short of its logical size is flagged rather than quietly reported.

O_DIRECT is unavailable on some filesystems (notably FUSE mounts without
direct-io support). Those rows fall back to fadvise + buffered reads and are
marked ``fadvise`` in the output, so a fallback is never mistaken for a clean
O_DIRECT number.
"""

from __future__ import annotations

import json
import mmap
import os
import statistics
import sys
import time
from pathlib import Path

BLOCK = 1 << 20
MIB = 1048576.0


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def disk_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def evict(paths) -> None:
    os.sync()
    for path in paths:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def read_direct(paths) -> tuple[int, float]:
    """O_DIRECT sequential read. Page cache is not involved at all.

    The mmap buffer is page-aligned, and offsets/lengths are 1 MiB multiples,
    which satisfies O_DIRECT's alignment requirements.
    """
    buf = mmap.mmap(-1, BLOCK)
    view = memoryview(buf)
    total = 0
    start = time.monotonic()
    for path in paths:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            offset = 0
            while offset < size:
                got = os.preadv(fd, [view], offset)
                if got <= 0:
                    break
                total += got
                offset += got
        finally:
            os.close(fd)
    return total, time.monotonic() - start


def read_buffered(paths) -> tuple[int, float]:
    total = 0
    start = time.monotonic()
    for path in paths:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            while True:
                chunk = os.read(fd, BLOCK)
                if not chunk:
                    break
                total += len(chunk)
        finally:
            os.close(fd)
    return total, time.monotonic() - start


def read_cold(paths) -> tuple[int, float, str]:
    """Prefer O_DIRECT; fall back to eviction + buffered, and say which."""
    try:
        nbytes, secs = read_direct(paths)
        if nbytes > 0:
            return nbytes, secs, "O_DIRECT"
    except OSError:
        pass
    evict(paths)
    nbytes, secs = read_buffered(paths)
    return nbytes, secs, "fadvise"


def row(rows, rep, label, phase, nbytes, secs, disk, pre, post, method):
    mibs = nbytes / secs / MIB if secs > 0 else 0.0
    # A cold row that did not actually reach the disk for most of its bytes is
    # measuring the page cache, not the storage.
    suspect = phase == "cold" and nbytes > 0 and disk < nbytes * 0.9
    rows.append(
        {
            "rep": rep,
            "path": label,
            "phase": phase,
            "method": method,
            "mib_s": round(mibs, 1),
            "bytes": nbytes,
            "disk_read_bytes": disk,
            "secs": round(secs, 3),
            "load_pre": pre,
            "load_post": post,
            "cache_contaminated": suspect,
        }
    )
    flag = "  <-- CACHE-CONTAMINATED" if suspect else ""
    print(
        f"{rep:>3}  {label:<8} {phase:<5} {method:<8} {mibs:>8.0f} MiB/s"
        f"  disk {disk / MIB:>6.0f}/{nbytes / MIB:.0f} MiB"
        f"   load {pre:>5.2f} -> {post:>5.2f}{flag}",
        flush=True,
    )


def main() -> int:
    native = Path(sys.argv[1])
    mount_file = sys.argv[2]
    objects = [Path(p) for p in sys.argv[3].split(",") if p]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None

    lanes = [("native", [native]), ("bypass", objects)]
    if mount_file != "-":
        lanes.append(("mount", [Path(mount_file)]))
    else:
        print("NOTE: no FUSE mount available; native + bypass only.\n")

    rows: list[dict] = []
    print(f"payload={native.stat().st_size / MIB:.0f} MiB  objects={len(objects)}  reps={reps}")
    print(f"start load: {load1():.2f}\n")

    for rep in range(1, reps + 1):
        for label, paths in lanes:
            d0 = disk_read_bytes()
            pre = load1()
            nbytes, secs, method = read_cold(paths)
            row(rows, rep, label, "cold", nbytes, secs,
                disk_read_bytes() - d0, pre, load1(), method)

            # Warm: the buffered read below repopulates cache, then we time a
            # second buffered pass over the now-resident bytes.
            read_buffered(paths)
            d0 = disk_read_bytes()
            pre = load1()
            nbytes, secs = read_buffered(paths)
            row(rows, rep, label, "warm", nbytes, secs,
                disk_read_bytes() - d0, pre, load1(), "buffered")

    print(f"\nend load: {load1():.2f}")

    print("\n=== summary (median of reps, [min-max] spread) ===")
    print(f"{'path':<8} {'phase':<5} {'median MiB/s':>13} {'spread':>22}  method")
    summary = {}
    for label, _ in lanes:
        for phase in ("cold", "warm"):
            picked = [
                r for r in rows
                if r["path"] == label and r["phase"] == phase and not r["cache_contaminated"]
            ]
            if not picked:
                continue
            vals = [r["mib_s"] for r in picked]
            med = statistics.median(vals)
            summary[f"{label}_{phase}"] = med
            methods = sorted({r["method"] for r in picked})
            print(
                f"{label:<8} {phase:<5} {med:>13.0f} "
                f"{f'[{min(vals):.0f} - {max(vals):.0f}]':>22}  {','.join(methods)}"
            )

    dropped = sum(1 for r in rows if r["cache_contaminated"])
    if dropped:
        print(f"\n{dropped} row(s) excluded from the medians as cache-contaminated.")

    if "native_cold" in summary:
        base = summary["native_cold"]
        print()
        for label in ("bypass", "mount"):
            key = f"{label}_cold"
            if key in summary and base:
                print(f"cold {label}/native: {summary[key] / base:.2f}x")

    if out_path:
        out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
        print(f"\nrows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
