"""Content-addressed local storage and chunk manifests."""

from typing import TYPE_CHECKING, Any

from .journal import TransferJournal, TransferSession
from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .refs import CASRef

if TYPE_CHECKING:
    from .transfer import (
        TransferGrant,
        TransferReport,
        download,
        upload,
    )

_TRANSFER_EXPORTS = frozenset(
    (
        "TransferGrant",
        "TransferReport",
        "download",
        "upload",
    )
)


def __getattr__(name: str) -> Any:
    if name not in _TRANSFER_EXPORTS:
        raise AttributeError(name)
    from . import transfer

    value = getattr(transfer, name)
    globals()[name] = value
    return value


__all__ = [
    "CASRef",
    "Chunk",
    "DigestMismatch",
    "FileEntry",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "RefConflict",
    "RepositoryManifest",
    "TransferGrant",
    "TransferJournal",
    "TransferReport",
    "TransferSession",
    "download",
    "upload",
]
