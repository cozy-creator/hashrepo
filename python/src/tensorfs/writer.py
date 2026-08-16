"""Compose a new safetensors snapshot one tensor at a time.

A conversion reads one tensor, transforms it, and writes it back. Nothing here
ever holds a shard: new tensors are admitted as CAS objects as they arrive, and
tensors the conversion did not touch are carried over **by reference**, keeping
their existing digests so their objects are not rewritten and the hub has
nothing new to fetch for them.

Be precise about what that does and does not avoid. Inheriting a tensor skips
holding its bytes and skips admitting them as new objects. It does **not** skip
hashing them: v1 identifies a file by the digest of its bytes, so
:meth:`TensorWriter.finish` still reads every byte once. Removing that pass
needs a format change, not an API change.

The emitted grid is the seal planner's: a header object, then one object per
tensor, split every 64 MiB from that tensor's own start
(``crates/tensorfs-core/src/planner/safetensors.rs``). That is what makes the
result inheritable by the *next* conversion in turn.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chunking import DTYPE_BITS
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry
from .refs import CASRef
from .tensors import TensorError, TensorView

if TYPE_CHECKING:
    from .local import LocalCAS

__all__ = ["TensorWriter"]


@dataclass(slots=True)
class _Pending:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    # Exactly one of these is set. `inherited` reuses existing objects
    # verbatim; `source` is a view whose bytes must be admitted afresh.
    inherited: tuple[tuple[CASRef, int], ...] | None
    chunks: tuple[tuple[CASRef, int], ...] | None


def _split(total: int) -> Iterator[int]:
    """Object lengths for a region, split every 64 MiB from its own start."""

    while total > MAX_CHUNK_SIZE:
        yield MAX_CHUNK_SIZE
        total -= MAX_CHUNK_SIZE
    if total:
        yield total


class TensorWriter:
    """Build one safetensors file from new and inherited tensors.

    Peak memory is one tensor for :meth:`add`, or one 64 MiB piece when the
    caller streams. No file is created at any point; :meth:`finish` returns a
    :class:`~tensorfs.manifest.FileEntry` describing objects already durable in
    the CAS.
    """

    def __init__(self, cas: LocalCAS, path: str) -> None:
        self._cas = cas
        self._path = path
        self._pending: list[_Pending] = []
        self._names: set[str] = set()

    def _claim(self, name: str) -> None:
        if name in self._names:
            raise TensorError(f"tensor {name!r} was added twice")
        self._names.add(name)

    def add(
        self,
        name: str,
        dtype: str,
        shape: Iterable[int],
        data: bytes | Iterable[bytes],
    ) -> None:
        """Admit a new or transformed tensor.

        ``data`` may be one buffer or an iterable of buffers, so a tensor
        larger than memory can be streamed in without ever being contiguous.
        """

        self._claim(name)
        if dtype not in DTYPE_BITS:
            raise TensorError(f"unknown safetensors dtype {dtype!r}")
        dimensions = tuple(shape)
        blocks = data if not isinstance(data, (bytes, bytearray, memoryview)) else (data,)

        chunks: list[tuple[CASRef, int]] = []
        carry = bytearray()
        total = 0

        def flush(final: bool) -> None:
            nonlocal total
            while len(carry) >= MAX_CHUNK_SIZE or (final and carry):
                cut = min(MAX_CHUNK_SIZE, len(carry))
                piece = bytes(carry[:cut])
                del carry[:cut]
                chunks.append((self._cas.put_bytes(piece), len(piece)))
                total += len(piece)

        for block in blocks:
            carry += memoryview(block).cast("B")
            flush(False)
        flush(True)

        expected = _declared_bytes(dtype, dimensions)
        if total != expected:
            raise TensorError(
                f"{name}: supplied {total} bytes but {dtype}{list(dimensions)} needs {expected}"
            )
        self._pending.append(
            _Pending(name, dtype, dimensions, total, None, tuple(chunks))
        )

    def inherit(self, view: TensorView) -> None:
        """Carry an untouched tensor over without moving or re-admitting its bytes.

        Raises when the tensor does not occupy whole objects, because then it
        has no digest of its own. That happens when the source was committed
        under a grid that packs small tensors together; use :meth:`add` with
        the tensor's bytes instead.
        """

        self._claim(view.name)
        if view.format != "safetensors-v1":
            raise TensorError(
                f"{view.name}: cannot inherit a {view.format} tensor into a "
                "safetensors file; re-add its bytes instead"
            )
        span = view._reader.object_span(view)
        if span is None:
            raise TensorError(
                f"{view.name}: is not object-aligned in {view.file!r}, so it "
                "cannot be inherited by reference; add its bytes instead"
            )
        self._pending.append(
            _Pending(view.name, view.dtype, view.shape, view.nbytes, span, None)
        )

    def finish(self) -> FileEntry:
        """Seal the file and return its manifest entry.

        The whole-file digest is unavoidable: v1 identifies a file by the hash
        of its bytes, so every byte is hashed here even for inherited tensors.
        Their *objects* are still never rewritten and never re-uploaded, which
        is where the saving actually lives -- but the hash pass is real, and
        removing it would take a format change, not an API change.
        """

        header: dict[str, object] = {}
        cursor = 0
        for item in self._pending:
            header[item.name] = {
                "dtype": item.dtype,
                "shape": list(item.shape),
                "data_offsets": [cursor, cursor + item.nbytes],
            }
            cursor += item.nbytes
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        prefix = len(encoded).to_bytes(8, "little") + encoded

        whole = hashlib.sha256()
        chunks: list[Chunk] = []
        size = 0

        at = 0
        for length in _split(len(prefix)):
            piece = prefix[at : at + length]
            chunks.append(Chunk(self._cas.put_bytes(piece), length))
            whole.update(piece)
            size += length
            at += length

        for item in self._pending:
            for ref, length in item.inherited or item.chunks or ():
                path = self._cas.verify_object(ref, size=length)
                with path.open("rb") as handle:
                    while block := handle.read(1 << 20):
                        whole.update(block)
                chunks.append(Chunk(ref, length))
                size += length

        return FileEntry(self._path, size, CASRef(whole.hexdigest()), tuple(chunks))


def _declared_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    bits = elements * DTYPE_BITS[dtype]
    if bits % 8:
        raise TensorError(f"{dtype}{list(shape)} is not a whole number of bytes")
    return bits // 8
