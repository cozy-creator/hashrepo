from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .manifest import CHUNK_SIZE, Chunk, FileEntry
from .refs import CASRef


class FileChangedError(OSError):
    """The source changed while its immutable declaration was being derived."""


def hash_file(path: Path, *, manifest_path: str | None = None) -> FileEntry:
    """Hash a file and derive its v1 chunk manifest in one sequential read."""

    source = Path(path)
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        whole = hashlib.sha256()
        chunks: list[Chunk] = []
        read = 0
        while True:
            data = handle.read(CHUNK_SIZE)
            if not data:
                break
            read += len(data)
            whole.update(data)
            if before.st_size > CHUNK_SIZE:
                chunks.append(Chunk(CASRef.digest_bytes(data), len(data)))
        after = os.fstat(handle.fileno())
    if (
        read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise FileChangedError(f"{source} changed while it was being hashed")
    return FileEntry(
        path=manifest_path or source.name,
        size_bytes=read,
        digest=CASRef(whole.hexdigest()),
        chunks=tuple(chunks),
    )
