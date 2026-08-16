"""Test-side ingestion. The one chunker is the Rust planner.

The legacy Python chunker is deleted. Tests that need a committed snapshot
build one here, either under the Rust planner's grid (`tensorfs.native`) or
under an explicit grid the test states outright — the manifest wire accepts
any grid, and the reader must serve all of them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tensorfs import (
    MAX_CHUNK_SIZE,
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    native,
)


def fixed_lengths(size: int) -> list[int]:
    """The bounded fixed grid: 64 MiB slices from offset zero."""

    full, remainder = divmod(size, MAX_CHUNK_SIZE)
    return [MAX_CHUNK_SIZE] * full + ([remainder] if remainder else [])


def ingest_with_grid(
    cas: LocalCAS, source: Path, lengths: list[int], *, manifest_path: str
) -> FileEntry:
    """Commit one file under an explicit object grid."""

    size = source.stat().st_size
    if sum(lengths) != size:
        raise ValueError(f"grid sums to {sum(lengths)}, file is {size} bytes")
    whole = hashlib.sha256()
    chunks: list[Chunk] = []
    with source.open("rb") as handle:
        for length in lengths:
            data = handle.read(length)
            assert len(data) == length
            whole.update(data)
            chunks.append(Chunk(cas.put_bytes(data), length))
    digest = CASRef(whole.hexdigest())
    if not chunks:
        return FileEntry(manifest_path, size, digest)
    return FileEntry(manifest_path, size, digest, tuple(chunks))


def ingest_file(cas: LocalCAS, source: Path, *, manifest_path: str | None = None) -> FileEntry:
    """Commit one file under the Rust planner's grid."""

    plan = native.plan_file(source)
    lengths = [region.length for region in plan.regions]
    return ingest_with_grid(cas, source, lengths, manifest_path=manifest_path or source.name)


def ingest_repository(cas: LocalCAS, root: Path) -> RepositoryManifest:
    """Commit every regular file below a directory, Rust-planned."""

    entries = [
        ingest_file(cas, path, manifest_path=path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.is_symlink()
    ]
    return RepositoryManifest(tuple(entries))
