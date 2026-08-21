"""Content-addressed local storage and direct tensor reads and writes."""

from .layout2 import ExpectedHeader, LayoutTensor, Quant
from .local import DigestMismatch, LocalCAS, Reclaimed, RefConflict, TempCollection
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
    "ExpectedHeader",
    "FileEntry",
    "LayoutTensor",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "ProjectionError",
    "Quant",
    "Reclaimed",
    "RefConflict",
    "RepositoryManifest",
    "Stub",
    "TensorError",
    "TempCollection",
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

# The v1 `Contract` surface stood here behind a PEP 562 `__getattr__`, because
# constructing one ran the Rust validator and `import tensorfs` must stay
# pure-Python for the vendored projection path. Its replacement,
# `tensorfs.layout2`, reads what the Go engine already computed and needs
# nothing compiled, so it is a plain import again.
