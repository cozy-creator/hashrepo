"""Content-addressed local storage and direct tensor reads and writes."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

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
    "FileEntry",
    "LocalCAS",
    "MAX_CHUNK_SIZE",
    "ProjectionError",
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
    "Contract",
    "Fusion",
    "MissingDtype",
    "Permute",
    "TensorDecl",
    "contracts",
]

if TYPE_CHECKING:
    from . import contracts
    from .contract import Contract, Fusion, MissingDtype, Permute, TensorDecl

# The contract surface needs the compiled extension (construction runs the
# Rust validator), while `import tensorfs` alone must stay pure-Python — the
# projection path is vendored extension-free. PEP 562 keeps both true: the
# extension loads on first ACCESS of a contract name, never at import.
_CONTRACT_EXPORTS = {
    "Contract": "contract",
    "Fusion": "contract",
    "MissingDtype": "contract",
    "Permute": "contract",
    "TensorDecl": "contract",
    "contracts": "contracts",
}


def __getattr__(name: str) -> Any:
    module_name = _CONTRACT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value: Any = module if name == "contracts" else getattr(module, name)
    globals()[name] = value
    return value
