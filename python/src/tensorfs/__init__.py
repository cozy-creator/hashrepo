"""Content-addressed local storage and direct tensor reads and writes."""

from .local import DigestMismatch, LocalCAS, RefConflict
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .project import (
    STUB_MAGIC,
    TENSOR_SUFFIXES,
    ProjectionError,
    Stub,
    is_tensor_container,
    parse_stub,
    project_snapshot,
    read_stub,
    stub_bytes,
    tree_bytes,
)
from .refs import CASRef
from .tensors import (
    DTYPE_BITS,
    GGUF_FORMAT,
    SAFETENSORS_FORMAT,
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

__all__ = [
    "DTYPE_BITS",
    "GGUF_FORMAT",
    "SAFETENSORS_FORMAT",
    "STUB_MAGIC",
    "TENSOR_SUFFIXES",
    "BlockLayout",
    "FileTooLarge",
    "CASRef",
    "Chunk",
    "DigestMismatch",
    "FileEntry",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "ProjectionError",
    "RefConflict",
    "RepositoryManifest",
    "Stub",
    "TensorError",
    "TensorReader",
    "TensorView",
    "TensorWriter",
    "dtype_itemsize",
    "is_tensor_container",
    "open_tensors",
    "parse_stub",
    "project_snapshot",
    "read_entry",
    "read_stub",
    "stub_bytes",
    "tree_bytes",
]
