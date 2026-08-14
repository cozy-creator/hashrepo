"""Content-addressed local storage and chunk manifests."""

from typing import TYPE_CHECKING, Any

from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .refs import CASRef

if TYPE_CHECKING:
    from .transfer import (
        GrantExpired,
        HTTPStatusError,
        HTTPTransport,
        TransferFailure,
        TransferGrant,
        TransferRefused,
        TransferReport,
        TransientTransferError,
        download,
        upload,
    )

_TRANSFER_EXPORTS = frozenset(
    (
        "GrantExpired",
        "HTTPStatusError",
        "HTTPTransport",
        "TransferFailure",
        "TransferGrant",
        "TransferRefused",
        "TransferReport",
        "TransientTransferError",
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
    "CHUNK_SIZE",
    "Chunk",
    "DigestMismatch",
    "FileEntry",
    "GrantExpired",
    "HTTPStatusError",
    "HTTPTransport",
    "LocalCAS",
    "RefConflict",
    "RepositoryManifest",
    "TransferFailure",
    "TransferGrant",
    "TransferRefused",
    "TransferReport",
    "TransientTransferError",
    "download",
    "upload",
]
