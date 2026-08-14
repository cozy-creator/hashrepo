from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .refs import CASRef

CHUNK_SIZE = 64 << 20
FORMAT = 1
MAX_SIZE = (1 << 63) - 1


def _valid_path(path: str) -> str:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"manifest path {path!r} must be valid UTF-8") from exc
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"manifest path {path!r} must be a relative forward-slash path")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ValueError(f"manifest path {path!r} contains a control character")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"manifest path {path!r} contains an unsafe component")
    return path


def _require_string(raw: object, field: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    return raw


def _require_integer(raw: object, field: str) -> int:
    if type(raw) is not int:
        raise ValueError(f"{field} must be an integer")
    return raw


def _require_mapping(raw: object, field: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be an object")
    return raw


def _ascii_fold(value: str) -> str:
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


@dataclass(frozen=True, slots=True)
class Chunk:
    digest: CASRef
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", CASRef.parse(self.digest))
        if type(self.length) is not int:
            raise ValueError("chunk length must be an integer")
        if not 0 < self.length <= CHUNK_SIZE:
            raise ValueError(f"chunk length must be in [1, {CHUNK_SIZE}], got {self.length}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Chunk:
        if set(raw) != {"digest", "len"}:
            raise ValueError("chunk must contain exactly digest and len")
        return cls(
            CASRef.parse(_require_string(raw["digest"], "chunk digest")),
            _require_integer(raw["len"], "chunk length"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"digest": str(self.digest), "len": self.length}


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    size_bytes: int
    digest: CASRef
    chunks: tuple[Chunk, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _valid_path(self.path))
        object.__setattr__(self, "digest", CASRef.parse(self.digest))
        object.__setattr__(self, "chunks", tuple(self.chunks))
        if type(self.size_bytes) is not int:
            raise ValueError("file size must be an integer")
        if not 0 <= self.size_bytes <= MAX_SIZE:
            raise ValueError(f"file size must be in [0, {MAX_SIZE}]")
        if self.size_bytes <= CHUNK_SIZE:
            if self.chunks:
                raise ValueError("files at or below 64 MiB must be stored as one whole object")
            return
        expected_count = (self.size_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
        if len(self.chunks) != expected_count:
            raise ValueError(
                f"chunked file requires {expected_count} chunks, got {len(self.chunks)}"
            )
        for index, chunk in enumerate(self.chunks):
            expected = min(CHUNK_SIZE, self.size_bytes - index * CHUNK_SIZE)
            if chunk.length != expected:
                raise ValueError(f"chunk {index} length is {chunk.length}, expected {expected}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FileEntry:
        allowed = {"path", "size_bytes", "digest", "chunks"}
        if set(raw) - allowed or not {"path", "size_bytes", "digest"}.issubset(raw):
            raise ValueError("file has missing or unknown fields")
        chunks_raw = raw.get("chunks", ())
        if not isinstance(chunks_raw, Sequence) or isinstance(chunks_raw, (str, bytes)):
            raise ValueError("file chunks must be an array")
        if "chunks" in raw and not chunks_raw:
            raise ValueError("file chunks, when present, must not be empty")
        return cls(
            path=_require_string(raw["path"], "file path"),
            size_bytes=_require_integer(raw["size_bytes"], "file size"),
            digest=CASRef.parse(_require_string(raw["digest"], "file digest")),
            chunks=tuple(Chunk.from_dict(_require_mapping(item, "chunk")) for item in chunks_raw),
        )

    def to_dict(self) -> dict[str, object]:
        raw: dict[str, object] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "digest": str(self.digest),
        }
        if self.chunks:
            raw["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return raw

    def objects(self) -> tuple[tuple[CASRef, int], ...]:
        if not self.chunks:
            return ((self.digest, self.size_bytes),)
        return tuple((chunk.digest, chunk.length) for chunk in self.chunks)


@dataclass(frozen=True, slots=True)
class RepositoryManifest:
    files: tuple[FileEntry, ...]
    format: int = FORMAT

    def __post_init__(self) -> None:
        if self.format != FORMAT:
            raise ValueError(f"unsupported manifest format {self.format}; v1 accepts only 1")
        ordered = tuple(sorted(self.files, key=lambda item: item.path))
        object.__setattr__(self, "files", ordered)
        seen: set[str] = set()
        folded: set[str] = set()
        for entry in ordered:
            if entry.path in seen:
                raise ValueError(f"duplicate manifest path {entry.path!r}")
            case_key = _ascii_fold(entry.path)
            if case_key in folded:
                raise ValueError(f"case-insensitive manifest path collision at {entry.path!r}")
            seen.add(entry.path)
            folded.add(case_key)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RepositoryManifest:
        if set(raw) != {"format", "files"}:
            raise ValueError("manifest must contain exactly format and files")
        files = raw["files"]
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            raise ValueError("manifest files must be an array")
        return cls(
            files=tuple(FileEntry.from_dict(_require_mapping(item, "file")) for item in files),
            format=_require_integer(raw["format"], "manifest format"),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> RepositoryManifest:
        raw = json.loads(data)
        if not isinstance(raw, Mapping):
            raise ValueError("manifest root must be an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, object]:
        return {"format": self.format, "files": [entry.to_dict() for entry in self.files]}

    def canonical_bytes(self) -> bytes:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return encoded.replace(b"\xe2\x80\xa8", b"\\u2028").replace(b"\xe2\x80\xa9", b"\\u2029")

    def digest(self) -> CASRef:
        return CASRef.digest_bytes(self.canonical_bytes())
