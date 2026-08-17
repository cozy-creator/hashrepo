"""Compose a new snapshot one tensor at a time, in safetensors or in GGUF.

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

The emitted grid is the seal planner's, in both formats — header objects, then
one object per tensor split every 64 MiB from that tensor's own start, plus
GGUF's per-tensor alignment padding as its own object
(``crates/tensorfs-core/src/planner/{safetensors,gguf}.rs``). That is what
makes the result inheritable by the *next* conversion in turn, and it is
asserted against the real planner rather than against a hand-written list of
lengths.

**Emission order is the caller's, and it is load-bearing.** Tensors are laid
out in the order they were added, which is what lets a conversion reproduce its
source's layout exactly by iterating the source's own header. See
``TensorWriter``'s class docstring for what a caller who invents a fresh order
does and does not give up.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import gguf
from .manifest import MAX_CHUNK_SIZE, Chunk, FileEntry
from .refs import CASRef
from .tensors import (
    DTYPE_BITS,
    GGUF_FORMAT,
    SAFETENSORS_FORMAT,
    TensorError,
    TensorView,
)

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
    """Build one safetensors or GGUF file from new and inherited tensors.

    Peak memory is one tensor for :meth:`add`, or one 64 MiB piece when the
    caller streams. No file is created at any point; :meth:`finish` returns a
    :class:`~tensorfs.manifest.FileEntry` describing objects already durable in
    the CAS.

    The format follows the path's suffix. A GGUF file additionally needs the
    metadata block it is being written with, because a GGUF file without one is
    not a model — pass the source's, read with
    :meth:`~tensorfs.tensors.TensorReader.gguf_header`. A conversion rewrites
    tensors, not model metadata.

    **Order.** Tensors are emitted in the order they were added. That is what
    lets the conversion loop reproduce its source exactly: it iterates the
    source's header, so it adds in the order the source was written in, and the
    result is byte-identical to the source for the tensors it did not touch.

    A caller that invents a fresh order instead gets a perfectly valid file
    that is **not** byte-identical to the one the reference library would have
    written for the same weights. What that costs is precisely one thing: the
    file digest, and therefore the snapshot id, differ from the canonical
    copy's. It does **not** cost dedup — the object grid is per tensor and
    objects are named by their own bytes, so every tensor object is shared with
    the canonical copy regardless of order, and only the header object differs.
    ``test_reordering_costs_the_header_object_and_nothing_else`` is that claim
    as a test.
    """

    def __init__(
        self,
        cas: LocalCAS,
        path: str,
        *,
        gguf_header: gguf.GGUFHeader | None = None,
    ) -> None:
        self._cas = cas
        self._path = path
        self._pending: list[_Pending] = []
        self._names: set[str] = set()
        self._format = GGUF_FORMAT if path.endswith(".gguf") else SAFETENSORS_FORMAT
        if self._format == GGUF_FORMAT:
            if gguf_header is None:
                raise TensorError(
                    f"{path}: a GGUF file needs its metadata block; pass the "
                    "source's gguf_header"
                )
            alignment = gguf_header.alignment
            if alignment < gguf.MIN_ALIGNMENT or alignment & (alignment - 1):
                raise TensorError(
                    f"{path}: general.alignment {alignment} is not a power of "
                    f"two at least {gguf.MIN_ALIGNMENT}"
                )
        elif gguf_header is not None:
            raise TensorError(f"{path}: gguf_header is meaningless for a {self._format} file")
        self._gguf = gguf_header

    @property
    def format(self) -> str:
        """The container being composed: ``safetensors-v1`` or ``gguf-v1``."""

        return self._format

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

        ``dtype`` is a safetensors dtype (``"F32"``) for a safetensors file and
        a ggml type name (``"Q4_K"``) for a GGUF one; ``shape`` is logical
        row-major for the first and GGUF's own ``ne`` order for the second,
        which is exactly what :class:`~tensorfs.tensors.TensorView` reports in
        each case. So "read a tensor, transform it, write it back" needs no
        translation of either.
        """

        self._claim(name)
        if self._format == SAFETENSORS_FORMAT and dtype not in DTYPE_BITS:
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
                # One copy, not two: slicing the bytearray would materialise a
                # second 64 MiB buffer before `bytes()` copied it again, and
                # this is the resident high-water mark of the whole writer.
                with memoryview(carry) as window:
                    piece = bytes(window[:cut])
                del carry[:cut]
                chunks.append((self._cas.put_bytes(piece), len(piece)))
                total += len(piece)

        for block in blocks:
            carry += memoryview(block).cast("B")
            flush(False)
        flush(True)

        expected = self._declared_bytes(dtype, dimensions)
        if total != expected:
            raise TensorError(
                f"{name}: supplied {total} bytes but {dtype}{list(dimensions)} needs {expected}"
            )
        self._pending.append(
            _Pending(name, dtype, dimensions, total, None, tuple(chunks))
        )

    def _declared_bytes(self, dtype: str, shape: tuple[int, ...]) -> int:
        if self._format == GGUF_FORMAT:
            try:
                return gguf.tensor_nbytes(gguf.type_id(dtype), shape)
            except gguf.GGUFError as error:
                raise TensorError(str(error)) from None
        return _declared_bytes(dtype, shape)

    def inherit(self, view: TensorView) -> None:
        """Carry an untouched tensor over without moving or re-admitting its bytes.

        Raises when the tensor does not occupy whole objects, because then it
        has no digest of its own. That happens when the source was committed
        under a grid that packs small tensors together; use :meth:`add` with
        the tensor's bytes instead.
        """

        self._claim(view.name)
        if view.format != self._format:
            raise TensorError(
                f"{view.name}: cannot inherit a {view.format} tensor into a "
                f"{self._format} file; re-add its bytes instead"
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

    # -- the header, per format ------------------------------------------

    def _safetensors_domains(self) -> tuple[list[bytes], list[int]]:
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
        return [prefix], [0] * len(self._pending)

    def _gguf_domains(self) -> tuple[list[bytes], list[int]]:
        """The three header domains the GGUF planner keeps apart, plus padding.

        `crates/tensorfs-core/src/planner/gguf.rs::plan_layout` emits the
        metadata block, the tensor directory and the pre-data padding as
        SEPARATE regions, and every tensor's trailing alignment padding as a
        region of its own rather than as part of the tensor. Isolating padding
        is what lets a GGUF share tensor objects with a safetensors twin, so
        the writer has to cut in the same places or seal re-admits everything.
        """

        if self._gguf is None:  # pragma: no cover - the constructor refuses this
            raise TensorError(f"{self._path}: no GGUF header")
        alignment = self._gguf.alignment
        prefix = (
            gguf.encode_prefix(
                self._gguf.version, len(self._pending), self._gguf.metadata_count
            )
            + self._gguf.metadata
        )
        directory = bytearray()
        padding: list[int] = []
        offset = 0
        for item in self._pending:
            try:
                identifier = gguf.type_id(item.dtype)
                directory += gguf.encode_tensor_info(
                    item.name, item.shape, identifier, offset
                )
            except gguf.GGUFError as error:
                raise TensorError(f"{item.name}: {error}") from None
            padded = gguf.align_up(item.nbytes, alignment)
            padding.append(padded - item.nbytes)
            offset += padded

        directory_end = len(prefix) + len(directory)
        # An empty GGUF has no data section to align to, which is the one case
        # where the planner does not round the directory up.
        data_start = gguf.align_up(directory_end, alignment) if self._pending else directory_end
        return [prefix, bytes(directory), b"\0" * (data_start - directory_end)], padding

    def finish(self) -> FileEntry:
        """Seal the file and return its manifest entry.

        The whole-file digest is unavoidable: v1 identifies a file by the hash
        of its bytes, so every byte is hashed here even for inherited tensors.
        Their *objects* are still never rewritten and never re-uploaded, which
        is where the saving actually lives -- but the hash pass is real, and
        removing it would take a format change, not an API change.
        """

        if self._format == GGUF_FORMAT:
            domains, padding = self._gguf_domains()
        else:
            domains, padding = self._safetensors_domains()

        whole = hashlib.sha256()
        chunks: list[Chunk] = []
        size = 0

        def emit(data: bytes) -> None:
            """Admit one header domain, split on the planner's own grid."""

            nonlocal size
            at = 0
            for length in _split(len(data)):
                piece = data[at : at + length]
                chunks.append(Chunk(self._cas.put_bytes(piece), length))
                whole.update(piece)
                size += length
                at += length

        for domain in domains:
            emit(domain)

        for item, pad in zip(self._pending, padding, strict=True):
            for ref, length in item.inherited or item.chunks or ():
                self._absorb(ref, length, whole)
                chunks.append(Chunk(ref, length))
                size += length
            emit(b"\0" * pad)

        return FileEntry(self._path, size, CASRef(whole.hexdigest()), tuple(chunks))

    def _absorb(self, ref: CASRef, length: int, whole: hashlib._Hash) -> None:
        """Fold one object into the whole-file digest, verifying it on the way.

        ONE pass, not two. Reading the object to hash the file and then calling
        ``verify_object`` to check the object would hash the same bytes twice,
        and SHA-256 is what bounds this method -- the same double-pass the CAS
        writer's own docstring warns about. The verification is not weakened:
        the per-object digest is computed from the identical bytes and compared
        to the name the object is stored under.
        """

        object_digest = hashlib.sha256()
        path = self._cas.object_path(ref)
        # Resolving under the shared store lock keeps collection from unlinking
        # the object between resolution and open; the open fd pins it after.
        with self._cas._store_lock():
            handle = path.open("rb")
        with handle:
            observed = 0
            while block := handle.read(1 << 20):
                whole.update(block)
                object_digest.update(block)
                observed += len(block)
        if observed != length:
            raise TensorError(f"{ref}: object is {observed} bytes, expected {length}")
        if object_digest.hexdigest() != ref.digest:
            raise TensorError(f"{ref}: object bytes do not match their digest")


def _declared_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    bits = elements * DTYPE_BITS[dtype]
    if bits % 8:
        raise TensorError(f"{dtype}{list(shape)} is not a whole number of bytes")
    return bits // 8
