from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CASRef:
    """A canonical v1 content reference."""

    digest: str

    def __post_init__(self) -> None:
        digest = self.digest.lower()
        if not _HEX.fullmatch(digest):
            raise ValueError("cas ref: sha256 digest must be 64 hexadecimal characters")
        object.__setattr__(self, "digest", digest)

    @classmethod
    def parse(cls, value: str | CASRef) -> CASRef:
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("cas ref: empty")
        if ":" not in text:
            raise ValueError(
                'cas ref: not algorithm-tagged (bare hex is refused; write "sha256:<hex>")'
            )
        algorithm, digest = text.split(":", 1)
        if algorithm != "sha256":
            raise ValueError(f"cas ref: unsupported algorithm {algorithm!r}")
        return cls(digest)

    @classmethod
    def digest_bytes(cls, data: bytes) -> CASRef:
        return cls(hashlib.sha256(data).hexdigest())

    def __str__(self) -> str:
        return f"sha256:{self.digest}"

    def object_key(self, prefix: str = "objects") -> str:
        clean = prefix.strip("/")
        if not clean or any(part in ("", ".", "..") for part in clean.split("/")):
            raise ValueError("object prefix must contain safe non-empty path components")
        return str(PurePosixPath(clean, "sha256", self.digest[:2], self.digest[2:4], self.digest))
