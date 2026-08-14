"""Opt-in real-scale benchmark for the tensor-aligned local writer.

Run from the repository root, for example:

    uv run python python/benchmarks/safetensors_dedup.py

The default needs about 4 GiB of temporary disk. It is deliberately excluded
from pytest and CI; shared-runner timing is not a stable correctness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import BinaryIO

from hashrepo import FileEntry, LocalCAS, RepositoryManifest

_MIB = 1 << 20
_COPY_BUFFER = 8 << 20


def _write_repeated(handle: BinaryIO, value: int, length: int) -> None:
    block = bytes([value]) * min(_MIB, length)
    remaining = length
    while remaining:
        data = block[:remaining]
        handle.write(data)
        remaining -= len(data)


def _write_parent(path: Path, body_bytes: int, tensor_count: int) -> int:
    tensor_bytes, remainder = divmod(body_bytes, tensor_count)
    if remainder or tensor_bytes < 8:
        raise ValueError("body bytes must divide evenly into tensors of at least 8 bytes")
    cursor = 0
    tensors: dict[str, object] = {}
    for index in range(tensor_count):
        tensors[f"tensor_{index:05d}"] = {
            "dtype": "U8",
            "shape": [tensor_bytes],
            "data_offsets": [cursor, cursor + tensor_bytes],
        }
        cursor += tensor_bytes
    encoded = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    header = struct.pack("<Q", len(encoded)) + encoded
    with path.open("wb") as handle:
        handle.write(header)
        for index in range(tensor_count):
            _write_repeated(handle, (index % 251) + 1, tensor_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    return len(header)


def _copy_child(parent: Path, child: Path, changed_offset: int) -> None:
    with parent.open("rb") as source, child.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=_COPY_BUFFER)
        destination.flush()
        os.fsync(destination.fileno())
    with child.open("r+b") as handle:
        handle.seek(changed_offset)
        original = handle.read(8)
        if original == b"changed!":
            raise RuntimeError("benchmark mutation would not change the source")
        handle.seek(changed_offset)
        handle.write(b"changed!")
        handle.flush()
        os.fsync(handle.fileno())


def _inventory(cas: LocalCAS) -> dict[str, int]:
    return {
        path.name: path.stat().st_size
        for path in cas.objects.rglob("*")
        if path.is_file()
    }


def _evict(path: Path) -> None:
    with path.open("rb") as handle:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _ingest(cas: LocalCAS, source: Path) -> tuple[FileEntry, dict[str, object]]:
    _evict(source)
    before_objects = _inventory(cas)
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    entry = cas.ingest_file(source, manifest_path="model.safetensors")
    manifest_ref = cas.store_manifest(RepositoryManifest((entry,)))
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    after_objects = _inventory(cas)
    new_digests = after_objects.keys() - before_objects.keys()
    reused = [
        (ref, size) for ref, size in entry.objects() if ref.digest not in new_digests
    ]
    return entry, {
        "wall_seconds": wall,
        "cpu_user_seconds": after.ru_utime - before.ru_utime,
        "cpu_system_seconds": after.ru_stime - before.ru_stime,
        "throughput_mib_s": source.stat().st_size / _MIB / wall,
        "peak_rss_kib": after.ru_maxrss,
        "filesystem_input_blocks": after.ru_inblock - before.ru_inblock,
        "filesystem_output_blocks": after.ru_oublock - before.ru_oublock,
        "file_bytes": source.stat().st_size,
        "file_digest": str(entry.digest),
        "chunks": len(entry.chunks),
        "manifest_ref": str(manifest_ref),
        "new_objects": len(new_digests),
        "new_bytes": sum(after_objects[digest] for digest in new_digests),
        "reused_file_objects": len(reused),
        "reused_file_bytes": sum(size for _, size in reused),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_COPY_BUFFER):
            digest.update(block)
    return digest.hexdigest()


def _verify_materialization(
    cas: LocalCAS, source: Path, entry: FileEntry, destination: Path
) -> dict[str, object]:
    started = time.perf_counter()
    cas.materialize(entry, destination)
    wall = time.perf_counter() - started
    source_digest = _sha256(source)
    rebuilt_digest = _sha256(destination)
    destination.unlink()
    if source_digest != rebuilt_digest or f"sha256:{source_digest}" != str(entry.digest):
        raise RuntimeError("materialized bytes do not match the source and manifest")
    return {
        "wall_seconds": wall,
        "throughput_mib_s": entry.size_bytes / _MIB / wall,
        "sha256": source_digest,
        "byte_identical": True,
    }


def _run(body_mib: int, tensor_count: int, changed_tensor: int) -> dict[str, object]:
    if body_mib <= 0 or tensor_count <= 0 or not 0 <= changed_tensor < tensor_count:
        raise ValueError("size and tensor count must be positive; changed tensor must exist")
    body_bytes = body_mib * _MIB
    tensor_bytes, remainder = divmod(body_bytes, tensor_count)
    if remainder:
        raise ValueError("body MiB must divide evenly across the tensor count")
    with tempfile.TemporaryDirectory(prefix="hashrepo-dedup-") as raw_root:
        root = Path(raw_root)
        parent = root / "parent.safetensors"
        child = root / "child.safetensors"
        header_bytes = _write_parent(parent, body_bytes, tensor_count)
        changed_offset = header_bytes + changed_tensor * tensor_bytes + tensor_bytes // 2
        _copy_child(parent, child, changed_offset)
        cas = LocalCAS(root / "cas")
        parent_entry, parent_metrics = _ingest(cas, parent)
        child_entry, child_metrics = _ingest(cas, child)
        return {
            "environment": {
                "hashrepo": importlib.metadata.version("hashrepo"),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "fixture": {
                "body_mib": body_mib,
                "file_bytes": parent.stat().st_size,
                "header_bytes": header_bytes,
                "tensor_count": tensor_count,
                "tensor_bytes": tensor_bytes,
                "changed_tensor": changed_tensor,
                "changed_bytes": 8,
            },
            "parent": parent_metrics,
            "child": child_metrics,
            "materialize_parent": _verify_materialization(
                cas, parent, parent_entry, root / "rebuilt.safetensors"
            ),
            "materialize_child": _verify_materialization(
                cas, child, child_entry, root / "rebuilt.safetensors"
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-mib", type=int, default=1024)
    parser.add_argument("--tensors", type=int, default=16)
    parser.add_argument("--changed-tensor", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(_run(args.body_mib, args.tensors, args.changed_tensor), indent=2))


if __name__ == "__main__":
    main()
