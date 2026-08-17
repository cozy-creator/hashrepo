"""Peak RSS of a conversion over a shard at the scale #61 actually names.

#61's fifth acceptance row asks for peak RSS over a transform of a snapshot
whose largest shard is **>= 8 GiB**, and explicitly refuses a small fixture as
a stand-in: "a 384 MiB fixture is not this measurement". So this runs at that
scale, on synthetic bytes, and reports the number with the scale beside it.

Three arms, each in its own process so ``VmHWM`` is a true high-water mark for
that arm alone:

* ``author``    -- stream the shard in through ``TensorWriter.add``.
* ``inherit``   -- the conversion loop: rewrite ONE tensor, inherit the rest.
                   This is the arm the row describes.
* ``transform`` -- the pathological loop: read, transform and rewrite EVERY
                   tensor. Nothing in the pipeline does this today, but it is
                   where a reader that retains mappings would show up.

No model, no network, no inference: the bytes are generated in the process, and
the scratch tree is removed unless ``--keep`` is given.

    python3 python/benchmarks/writer_peak_rss.py --gib 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfs import (  # noqa: E402
    LocalCAS,
    RepositoryManifest,
    TensorWriter,
    open_tensors,
)

BLOCK = 1 << 20
MIB = 1 << 20


def peak_rss_bytes() -> int:
    """The kernel's own high-water mark, not a sample of it."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def blocks(name: str, nbytes: int) -> Iterator[bytes]:
    """Distinct 1 MiB blocks, generated lazily and never held together.

    Every block differs, so the CAS cannot dedup the stream away and quietly
    turn an 8 GiB measurement into a 64 MiB one.
    """

    base = random.Random(name).randbytes(BLOCK)
    counter = 0
    written = 0
    while written < nbytes:
        cut = min(BLOCK, nbytes - written)
        piece = bytearray(base[:cut])
        piece[0:8] = struct.pack("<Q", counter)
        counter += 1
        written += cut
        yield bytes(piece)


def plan(total_bytes: int) -> list[tuple[str, int]]:
    """One large tensor plus a tail of smaller ones, summing to `total_bytes`.

    The largest single tensor is the denominator the row's "small multiple of
    the largest single tensor" is measured against, so it is stated rather than
    left to arithmetic.
    """

    largest = min(total_bytes // 4, 2 * (1 << 30))
    largest -= largest % 4
    tensors = [("denoiser.big", largest)]
    remaining = total_bytes - largest
    index = 0
    while remaining > 0:
        size = min(remaining, 512 * MIB)
        size -= size % 4
        tensors.append((f"denoiser.block.{index}", size))
        remaining -= size
        index += 1
    return tensors


def author(root: Path, total_bytes: int) -> dict[str, object]:
    cas = LocalCAS(root / "cas")
    writer = TensorWriter(cas, "model.safetensors")
    for name, size in plan(total_bytes):
        writer.add(name, "F32", (size // 4,), blocks(name, size))
    entry = writer.finish()
    manifest = RepositoryManifest((entry,))
    (root / "manifest.json").write_text(str(cas.store_manifest(manifest)))
    return {"file_bytes": entry.size_bytes, "objects": len(entry.chunks)}


def halve(view: object) -> Iterator[bytes]:
    """Read every source byte, emit half of them.

    Reading is the point. An arm that allocated the output without touching
    the input would never fault a single mapped page in, and would report the
    peak RSS of a loop that does no work -- the "green while testing nothing"
    failure. `bytes(piece)` faults the whole piece.
    """

    for piece in view.pieces():  # type: ignore[attr-defined]
        whole = bytes(piece)
        yield whole[: len(whole) // 2]


def convert(root: Path, everything: bool, verify: bool) -> dict[str, object]:
    cas = LocalCAS(root / "cas")
    manifest = cas.load_manifest((root / "manifest.json").read_text().strip())
    rewritten = 0
    read = 0
    with open_tensors(cas, manifest, verify=verify) as source:
        writer = TensorWriter(cas, "converted.safetensors")
        for name in source:
            view = source[name]
            if everything or name == "denoiser.block.0":
                # The transform: halve the width, streamed piece by piece, so
                # the tensor is never contiguous in either direction.
                writer.add(name, "U16", (view.nbytes // 4,), halve(view))
                rewritten += 1
                read += view.nbytes
            else:
                writer.inherit(view)
        entry = writer.finish()
    return {"rewritten": rewritten, "bytes_read": read, "file_bytes": entry.size_bytes}


def largest_tensor(total_bytes: int) -> int:
    return max(size for _name, size in plan(total_bytes))


def run_arm(arm: str, root: Path, total_bytes: int, verify: bool) -> dict[str, object]:
    started = time.monotonic()
    if arm == "author":
        detail = author(root, total_bytes)
    else:
        detail = convert(root, everything=arm == "transform", verify=verify)
    return {
        "arm": arm,
        "seconds": round(time.monotonic() - started, 1),
        "peak_rss_bytes": peak_rss_bytes(),
        **detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=8.0)
    parser.add_argument("--root", default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    # The library default. Verification hashes every mapped object, which is
    # what forces every page of every object a conversion reads to be resident.
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    arguments = parser.parse_args()

    total_bytes = int(arguments.gib * (1 << 30))
    total_bytes -= total_bytes % 4

    if arguments.arm:
        result = run_arm(arguments.arm, Path(arguments.root), total_bytes, arguments.verify)
        print(json.dumps(result))
        return 0

    root = Path(arguments.root or os.environ.get("TMPDIR", "/tmp")) / "tensorfs-rss"
    root.mkdir(parents=True, exist_ok=True)
    try:
        results = []
        for arm in ("author", "inherit", "transform"):
            completed = subprocess.run(
                [sys.executable, __file__, "--arm", arm, "--root", str(root),
                 "--gib", str(arguments.gib)]
                + ([] if arguments.verify else ["--no-verify"]),
                capture_output=True,
                text=True,
                check=True,
            )
            results.append(json.loads(completed.stdout.strip().splitlines()[-1]))

        biggest = largest_tensor(total_bytes)
        print(f"scale: {total_bytes / (1 << 30):.2f} GiB of tensor bytes in one shard")
        print(f"largest single tensor: {biggest / MIB:.0f} MiB")
        print()
        for result in results:
            peak = int(result["peak_rss_bytes"])
            print(
                f"{result['arm']:<10} peak RSS {peak / MIB:8.1f} MiB "
                f"({peak / biggest:.3f}x the largest tensor) "
                f"in {result['seconds']}s  {result}"
            )
        return 0
    finally:
        if not arguments.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
