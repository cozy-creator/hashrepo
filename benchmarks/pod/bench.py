#!/usr/bin/env python3
"""Load-controlled read/write comparison: native vs CAS-bypass vs FUSE mount.

Why this is shaped the way it is
--------------------------------
The laptop these numbers were first attempted on swung 13x on the SAME `dd`
command purely with system load, which makes absolute throughput unusable and
even ratios suspect when the two sides are measured minutes apart. So:

* **Round-robin, not batched.** One rep of every path back to back, so a load
  excursion lands on all three roughly equally instead of poisoning one arm.
* **Load sampled before AND after every single measurement**, printed on the
  row. A row whose load moved is visible rather than silently averaged in.
* **Identical payload bytes** for all three paths, identical 1 MiB read size,
  identical eviction protocol.
* **True cold** means `POSIX_FADV_DONTNEED` on the target *and* on every
  backing CAS object, otherwise the "cold" mount read is served from the page
  cache of its objects and measures nothing.
* Warm rows read the same file a second time; the first read populated cache.

Reported as median and full spread across reps, never a single number.
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


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def evict(paths) -> None:
    for path in paths:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def read_stream(path) -> tuple[int, float]:
    """Sequential 1 MiB reads; returns (bytes, seconds)."""
    fd = os.open(str(path), os.O_RDONLY)
    total = 0
    start = time.monotonic()
    try:
        while True:
            chunk = os.read(fd, BLOCK)
            if not chunk:
                break
            total += len(chunk)
    finally:
        os.close(fd)
    return total, time.monotonic() - start


def read_concat(paths) -> tuple[int, float]:
    """The bypass lane: read the backing objects directly, in order."""
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


def proc_io(pid: str | int) -> tuple[int, int]:
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


def row(rep, path_label, phase, nbytes, secs, pre, post, rows):
    mibs = nbytes / secs / 1048576 if secs > 0 else 0.0
    rows.append(
        {
            "rep": rep,
            "path": path_label,
            "phase": phase,
            "mib_s": round(mibs, 1),
            "bytes": nbytes,
            "secs": round(secs, 3),
            "load_pre": pre,
            "load_post": post,
        }
    )
    print(
        f"{rep:>3}  {path_label:<8} {phase:<5} {mibs:>9.0f} MiB/s"
        f"   load {pre:>5.2f} -> {post:>5.2f}",
        flush=True,
    )


def main() -> int:
    native = Path(sys.argv[1])
    mount_file = Path(sys.argv[2])
    objects = [Path(p) for p in sys.argv[3].split(",") if p]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out_path = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    rows: list[dict] = []
    print(f"payload={native.stat().st_size / 1048576:.0f} MiB  objects={len(objects)}  reps={reps}")
    print(f"start load: {load1():.2f}\n")
    print(f"{'rep':>3}  {'path':<8} {'phase':<5} {'MiB/s':>14}   load")

    for rep in range(1, reps + 1):
        for label in ("native", "bypass", "mount"):
            if label == "native":
                target, backing, reader = native, [native], lambda: read_stream(native)
            elif label == "bypass":
                target, backing, reader = None, objects, lambda: read_concat(objects)
            else:
                target, backing, reader = (
                    mount_file,
                    objects + [mount_file],
                    lambda: read_stream(mount_file),
                )

            evict(backing)
            pre = load1()
            nbytes, secs = reader()
            row(rep, label, "cold", nbytes, secs, pre, load1(), rows)

            pre = load1()
            nbytes, secs = reader()
            row(rep, label, "warm", nbytes, secs, pre, load1(), rows)

    print(f"\nend load: {load1():.2f}")

    print("\n=== summary (median of reps, [min-max] spread) ===")
    print(f"{'path':<8} {'phase':<5} {'median MiB/s':>14} {'spread':>22}")
    summary = {}
    for label in ("native", "bypass", "mount"):
        for phase in ("cold", "warm"):
            vals = [r["mib_s"] for r in rows if r["path"] == label and r["phase"] == phase]
            if not vals:
                continue
            med = statistics.median(vals)
            summary[f"{label}_{phase}"] = med
            print(
                f"{label:<8} {phase:<5} {med:>14.0f} "
                f"{f'[{min(vals):.0f} - {max(vals):.0f}]':>22}"
            )

    if "bypass_cold" in summary and "mount_cold" in summary and summary["mount_cold"]:
        print(
            f"\ncold bypass/mount ratio: "
            f"{summary['bypass_cold'] / summary['mount_cold']:.2f}x"
        )
    if "native_cold" in summary and summary["native_cold"]:
        for label in ("bypass", "mount"):
            key = f"{label}_cold"
            if key in summary:
                print(
                    f"cold {label}/native ratio:  "
                    f"{summary[key] / summary['native_cold']:.2f}x"
                )

    if out_path:
        out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
        print(f"\nrows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
