"""Local CPU-side microbench for the #115 stream surface.

Measures store->buffer bandwidth for one multi-GiB tensor container:

- ``TensorStreamReader.readinto`` buffered (the default arm),
- ``TensorStreamReader.readinto`` with ``direct=True`` (O_DIRECT),
- ``RecordsReader.read_at`` (the pre-#115 path: allocates bytes per call,
  holds the GIL for the assembly),
- a plain ``file.readinto`` over the original bytes as the local floor.

This is NOT the pod benchmark (e2e#1906 measures store->VRAM on rented
hardware); it is the CPU-side sanity number for the wheel surface. Page cache
state is whatever ingest left behind: buffered numbers are warm unless noted.

Usage: python python/benchmarks/stream_readinto.py [--gib 2]
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from tensorfs.native import FileRecord, ObjectStore, RecordsReader, TensorStreamReader

MIB = 1024 * 1024
GIB = 1024 * MIB


def build_file(path: Path, size: int) -> None:
    import json
    import struct

    header_object = {
        f"t{index}": {
            "dtype": "U8",
            "shape": [256 * MIB],
            "data_offsets": [index * 256 * MIB, (index + 1) * 256 * MIB],
        }
        for index in range(size // (256 * MIB))
    }
    header = json.dumps(header_object).encode()
    with path.open("wb") as sink:
        sink.write(struct.pack("<Q", len(header)))
        sink.write(header)
        block = bytes(range(256)) * (4 * MIB // 256)
        remaining = size
        while remaining:
            sink.write(block[: min(len(block), remaining)])
            remaining -= min(len(block), remaining)


def timed(label: str, total: int, runs: list[float]) -> None:
    best = min(runs)
    mean = statistics.mean(runs)
    print(
        f"{label:32s} best {total / best / GIB:6.2f} GiB/s   "
        f"mean {total / mean / GIB:6.2f} GiB/s   ({len(runs)} runs)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=2.0)
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()
    size = int(arguments.gib * GIB)

    with tempfile.TemporaryDirectory(prefix="tensorfs-streambench-") as scratch:
        root = Path(scratch)
        source = root / "model.safetensors"
        build_file(source, size)
        store = ObjectStore(root / "store")
        _plan, admitted = store.admit_file(source)
        records = [FileRecord.data(item.digest, item.length) for item in admitted]

        buffered = TensorStreamReader(store, records)
        total = sum(tensor.nbytes for tensor in buffered.tensors)
        destination = bytearray(total)
        view = memoryview(destination)

        def stream(reader: TensorStreamReader) -> float:
            start = time.perf_counter()
            at = 0
            for tensor in reader.tensors:
                reader.readinto(tensor.offset, tensor.nbytes, view[at : at + tensor.nbytes])
                at += tensor.nbytes
            return time.perf_counter() - start

        timed(
            "stream readinto (buffered)",
            total,
            [stream(buffered) for _ in range(arguments.runs)],
        )

        try:
            direct = TensorStreamReader(store, records, direct=True)
            timed(
                "stream readinto (O_DIRECT)",
                total,
                [stream(direct) for _ in range(arguments.runs)],
            )
        except OSError as error:
            print(f"O_DIRECT arm unavailable here: {error}")

        legacy = RecordsReader(store, records)

        def read_at() -> float:
            start = time.perf_counter()
            at = 0
            for tensor in buffered.tensors:
                chunk = legacy.read_at(tensor.offset, tensor.nbytes)
                view[at : at + tensor.nbytes] = chunk
                at += tensor.nbytes
            return time.perf_counter() - start

        timed(
            "RecordsReader.read_at + copy",
            total,
            [read_at() for _ in range(arguments.runs)],
        )

        def raw() -> float:
            start = time.perf_counter()
            with source.open("rb") as handle:
                handle.seek(buffered.tensors[0].offset)
                handle.readinto(view)
            return time.perf_counter() - start

        timed("plain file readinto (floor)", total, [raw() for _ in range(arguments.runs)])


if __name__ == "__main__":
    main()
