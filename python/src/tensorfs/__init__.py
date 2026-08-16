"""Content-addressed local storage and direct tensor reads and writes."""

from typing import TYPE_CHECKING, Any

from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .refs import CASRef
from .tensors import (
    DTYPE_BITS,
    EXTRACT_SIZE_LIMIT,
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
    if name in _DAEMON_EXPORTS:
        from . import daemon

        value = getattr(daemon, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "DTYPE_BITS",
    "EXTRACT_SIZE_LIMIT",
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
    "dtype_itemsize",
    "open_tensors",
    "read_entry",
]
