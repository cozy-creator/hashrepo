"""Content-addressed local storage and chunk manifests."""

from typing import TYPE_CHECKING, Any

from .journal import TransferJournal, TransferSession
from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .refs import CASRef
from .tensors import (
    DTYPE_BITS,
    BlockLayout,
    FileTooLarge,
    TensorError,
    TensorReader,
    TensorView,
    dtype_itemsize,
    open_tensors,
    read_entry,
)
from .writer import TensorWriter

if TYPE_CHECKING:
    from .daemon import (
        DaemonClient,
        DaemonError,
        DaemonHello,
        DaemonMount,
        DaemonStatus,
        MountedPath,
    )
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

_DAEMON_EXPORTS = frozenset(
    (
        "DaemonClient",
        "DaemonError",
        "DaemonHello",
        "DaemonMount",
        "DaemonStatus",
        "MountedPath",
    )
)


def __getattr__(name: str) -> Any:
    if name in _TRANSFER_EXPORTS:
        from . import transfer

        value = getattr(transfer, name)
    elif name in _DAEMON_EXPORTS:
        from . import daemon

        value = getattr(daemon, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "DTYPE_BITS",
    "BlockLayout",
    "FileTooLarge",
    "CASRef",
    "Chunk",
    "DaemonClient",
    "DaemonError",
    "DaemonHello",
    "DaemonMount",
    "DaemonStatus",
    "DigestMismatch",
    "FileEntry",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "MountedPath",
    "RefConflict",
    "RepositoryManifest",
    "TensorError",
    "TensorReader",
    "TensorView",
    "TensorWriter",
    "TransferGrant",
    "TransferJournal",
    "TransferReport",
    "TransferSession",
    "download",
    "dtype_itemsize",
    "open_tensors",
    "read_entry",
    "upload",
]
