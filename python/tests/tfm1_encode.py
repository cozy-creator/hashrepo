"""A minimal, independent TFM1 encoder for tests.

Written from ``spec/v1/TFM1.md`` rather than imported from the library, so a
test that projects a tree is describing a manifest rather than asking the
implementation what it would have produced. Every manifest built here is fed
through the real ``decode_snapshot``, which refuses anything this file gets
wrong -- so the independence costs nothing in confidence.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_KIND_DIRECTORY = 1
_KIND_FILE = 2
_RECORD_DATA = 1
_PLANNERS = {"safetensors-v1": 1, "gguf-v1": 2, "blob-v1": 4}


@dataclass(frozen=True, slots=True)
class Directory:
    path: str


@dataclass(frozen=True, slots=True)
class Blob:
    """One whole-file object: the entry a projection turns into a symlink."""

    path: str
    digest: str
    size: int


@dataclass(frozen=True, slots=True)
class Tensor:
    """A chunked tensor container: the entry a projection turns into a stub."""

    path: str
    planner: str
    records: tuple[tuple[str, int], ...]


Entry = Directory | Blob | Tensor


def _path(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def encode(entries: list[Entry]) -> bytes:
    """Canonical TFM1 bytes. Entries are sorted, as the format requires."""
    ordered = sorted(entries, key=lambda entry: entry.path.encode("utf-8"))
    out = bytearray(b"TFM1")
    out.append(0)  # no parent
    out += struct.pack("<Q", len(ordered))
    for entry in ordered:
        out += _path(entry.path)
        if isinstance(entry, Directory):
            out.append(_KIND_DIRECTORY)
            continue
        out.append(_KIND_FILE)
        out.append(0)  # not executable
        if isinstance(entry, Blob):
            out.append(_PLANNERS["blob-v1"])
            out += struct.pack("<Q", entry.size)
            out += bytes.fromhex(entry.digest)
            continue
        out.append(_PLANNERS[entry.planner])
        out += struct.pack("<Q", sum(length for _, length in entry.records))
        out += struct.pack("<Q", len(entry.records))
        for digest, length in entry.records:
            out.append(_RECORD_DATA)
            out += bytes.fromhex(digest)
            out += struct.pack("<Q", length)
    return bytes(out)
