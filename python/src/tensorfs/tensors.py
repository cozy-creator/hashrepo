"""Read individual tensors out of a committed manifest without building a file.

The seal-time planner already gives a safetensors file a tensor-aligned object
grid, so "load tensor X" is "read these CAS objects". This module is the
reference implementation of that read path: a lazy ``Mapping`` from tensor name
to a handle that fetches only the objects its own bytes live in.

This is the executable specification for the Rust/PyO3 reader described in
``docs/direct-tensor-reads.md``; the byte-exactness properties it proves are
properties of the object layout, not of this language.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import mmap
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from . import gguf
from .manifest import FileEntry, RepositoryManifest
from .refs import CASRef

if TYPE_CHECKING:
    from .local import LocalCAS

__all__ = [
    "DTYPE_BITS",
    "BlockLayout",
    "FileTooLarge",
    "TensorError",
    "TensorReader",
    "TensorView",
    "dtype_itemsize",
    "open_tensors",
    "read_entry",
]

_SAFETENSORS_SUFFIX = ".safetensors"
_GGUF_SUFFIX = ".gguf"
_SAFETENSORS_FORMAT = "safetensors-v1"
_GGUF_FORMAT = "gguf-v1"
_PREFIX_SIZE = 8
# safetensors 0.8.0's own reader refuses header lengths above this. Matching it
# keeps a malformed prefix from provoking a huge allocation here.
_MAX_HEADER_BYTES = 100_000_000

# Mirrors safetensors::tensor::Dtype::bitsize at upstream 6eb4dc9, and the Rust
# planner's dtype_bits (crates/tensorfs-core/src/planner/safetensors.rs).
DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E4M3FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "C64": 64,
}


class TensorError(ValueError):
    """A manifest did not describe the tensors a caller asked for."""


class FileTooLarge(TensorError):
    """A caller tried to extract a file the escape hatch must not carry."""




def dtype_itemsize(dtype: str) -> float:
    """Bytes per element, which is fractional for the sub-byte float types."""

    try:
        return DTYPE_BITS[dtype] / 8
    except KeyError:
        raise TensorError(f"unknown safetensors dtype {dtype!r}") from None


def _safetensors_block(dtype: str) -> BlockLayout:
    """Express a safetensors dtype in the same block shape GGUF needs.

    Whole-byte dtypes are one element per block. The sub-byte float types pack
    several elements into a byte, which is the same relationship a quantized
    GGUF block describes, so one vocabulary covers both.
    """

    bits = DTYPE_BITS[dtype]
    if bits >= 8:
        return BlockLayout(1, bits // 8)
    elements = 8 // math.gcd(bits, 8)
    return BlockLayout(elements, elements * bits // 8)


@dataclass(frozen=True, slots=True)
class BlockLayout:
    """How many elements one encoded block holds, and how many bytes it takes.

    This is the one shape that describes both formats without the caller
    branching. A plain safetensors ``F32`` is ``(1, 4)``; its sub-byte ``F4`` is
    ``(2, 1)``; a GGUF ``Q4_K`` is ``(256, 144)``. A caller that needs to know
    whether it must build a quantized parameter asks :attr:`quantized`.
    """

    elements: int
    nbytes: int

    @property
    def quantized(self) -> bool:
        return self.elements > 1


@dataclass(frozen=True, slots=True)
class TensorView:
    """A lazy handle on one tensor. No tensor bytes are read to create it."""

    name: str
    file: str
    format: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    offset: int
    block: BlockLayout
    _reader: TensorReader = field(repr=False, compare=False)

    def pieces(self) -> Iterator[memoryview]:
        """Yield this tensor's bytes as ordered zero-copy views.

        Each view is a slice of one mmapped CAS object, so nothing is copied
        and no more than one object is resident per step. A tensor that spans
        several objects yields several views; concatenating them is the
        caller's choice, and it is the only place a copy becomes unavoidable.
        """

        return self._reader._pieces(self.file, self.offset, self.nbytes)

    def tobytes(self) -> bytes:
        """The whole tensor as one contiguous buffer. Copies, by definition."""

        if self.nbytes == 0:
            return b""
        parts = list(self.pieces())
        if len(parts) == 1:
            return bytes(parts[0])
        # join sizes the result once and copies each piece in exactly once;
        # a bytearray followed by bytes() would copy the whole tensor twice.
        return b"".join(parts)

    def readinto(self, buffer: memoryview | bytearray) -> int:
        """Copy this tensor into caller-owned memory, e.g. a pinned buffer."""

        target = memoryview(buffer).cast("B")
        if len(target) < self.nbytes:
            raise TensorError(
                f"{self.name}: buffer holds {len(target)} bytes, tensor needs {self.nbytes}"
            )
        at = 0
        for piece in self.pieces():
            target[at : at + len(piece)] = piece
            at += len(piece)
        return at


class TensorReader(Mapping[str, TensorView]):
    """Every tensor in a manifest, addressable without materializing a file.

    Opening reads only the safetensors header of each contributing file. A
    tensor's own bytes are read when a :class:`TensorView` is asked for them,
    so one tensor of a 50 GiB checkpoint costs that tensor and nothing else.
    """

    def __init__(
        self,
        cas: LocalCAS,
        manifest: RepositoryManifest,
        *,
        verify: bool = True,
    ) -> None:
        self._cas = cas
        self._manifest = manifest
        self._verify = verify
        self._entries = {entry.path: entry for entry in manifest.files}
        # (file path) -> (object start offsets, (ref, length) pairs)
        self._grid: dict[str, tuple[list[int], tuple[tuple[CASRef, int], ...]]] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._verified: set[str] = set()
        self._index: dict[str, TensorView] | None = None
        self._closed = False

    # -- Mapping ---------------------------------------------------------

    def __getitem__(self, name: str) -> TensorView:
        return self._tensors()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors())

    def __len__(self) -> int:
        return len(self._tensors())

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Drop this reader's mappings.

        A mapping the caller still holds a ``memoryview`` into is *not*
        force-unmapped — that would invalidate live buffers. It is released
        instead, so the mapping survives exactly as long as the last view of
        it. This is the same ownership rule the Rust reader gets from holding
        each object in an ``Arc<Mmap>``; here refcounting supplies it.
        """

        self._closed = True
        for handle in self._maps.values():
            try:
                handle.close()
            except BufferError:
                pass
        self._maps.clear()

    def __enter__(self) -> TensorReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- generic byte access ---------------------------------------------

    def files(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def read_range(self, path: str, offset: int, length: int) -> bytes:
        """Read an arbitrary byte range, touching only overlapping objects.

        This is the primitive the whole module rests on, and the honest
        replacement for whole-file materialization: materializing a file was
        only ever ``read_range(path, 0, size)`` with a write and two extra
        hash passes bolted on.
        """

        if length == 0:
            return b""
        parts = list(self._pieces(path, offset, length))
        if len(parts) == 1:
            return bytes(parts[0])
        return b"".join(parts)

    def read_file(self, path: str) -> bytes:
        """A whole small file, e.g. ``config.json`` or ``model_index.json``."""

        entry = self._entry(path)
        return self.read_range(path, 0, entry.size_bytes)

    def extract(
        self,
        path: str,
        destination: str | Path,
        *,
        limit: int | None = None,
    ) -> Path:
        """Write ONE non-tensor file to disk, for consumers that need a path.

        This is the whole remaining reason to create a file: config JSON a
        third-party constructor insists on reading from a directory, or any
        other non-tensor member of a repo. Tensors never come through here --
        they are read with :class:`TensorView`.

        There is NO size cap -- ruled by Paul, 2026-08-16: "no hard-cap on
        materialization size; we just want to avoid large non-tensor files
        being in tensorfs (our chunked CAS system)." The control is scope at
        ingestion (CAS holds repos only; datasets and compiled-graph artifacts
        never enter it -- DESIGN-RULINGS "CAS scope: repos only"), not a limit
        here. The stream is O(one block) of memory regardless of file size, so
        extraction itself has no reason to refuse. ``limit`` remains for a
        caller that wants to bound ITS OWN request; when given, exceeding it
        raises :class:`FileTooLarge`.

        The write is atomic: the destination either does not exist or is
        complete, and its bytes are verified on the way out.
        """

        entry = self._entry(path)
        if limit is not None and entry.size_bytes > limit:
            raise FileTooLarge(
                f"{path} is {entry.size_bytes} bytes, above the caller-supplied "
                f"{limit}-byte bound"
            )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(raw)
        try:
            digest = hashlib.sha256()
            with os.fdopen(handle, "wb") as sink:
                for piece in self._pieces(path, 0, entry.size_bytes):
                    sink.write(piece)
                    digest.update(piece)
                sink.flush()
                os.fsync(sink.fileno())
            if digest.hexdigest() != entry.digest.digest:
                raise TensorError(f"{path}: extracted bytes do not match the manifest")
            os.replace(temporary, target)
            return target
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def object_span(self, view: TensorView) -> tuple[tuple[CASRef, int], ...] | None:
        """The objects this tensor occupies, if it occupies them exactly.

        Returns ``None`` when the tensor starts or ends mid-object, because
        then it has no digest of its own and cannot be carried into a new
        snapshot by reference -- its bytes would have to be rewritten.

        This is the property that decides whether a conversion can leave a
        tensor untouched. The Rust seal planner — the only chunker — gives
        every tensor its own object, so it always holds for grids it planned.
        The manifest wire allows arbitrary grids, so snapshots committed under
        the retired packing chunker may not satisfy it.
        """

        starts, objects = self._file_grid(view.file)
        begin = bisect.bisect_left(starts, view.offset)
        if begin == len(starts) or starts[begin] != view.offset:
            return None
        span: list[tuple[CASRef, int]] = []
        total = 0
        index = begin
        while total < view.nbytes and index < len(objects):
            ref, size = objects[index]
            span.append((ref, size))
            total += size
            index += 1
        if total != view.nbytes:
            return None
        return tuple(span)

    def rekeyed(self, names: Mapping[str, str]) -> Mapping[str, TensorView]:
        """Serve these tensors under other names, admitting nothing.

        The zero-storage half of a re-key: a consumer on the native read path
        who only wants the other naming scheme -- ComfyUI's spelling of a
        diffusers checkpoint, say -- needs no second snapshot at all. Composing
        one (``tensorfs.native.rekey``) is for handing the other spelling to a
        foreign tool that opens files.

        ``names`` maps a stored name to the name it is served under; a tensor
        it does not name keeps its own. Two tensors may not land on one name.
        """

        index = self._tensors()
        unknown = sorted(set(names) - set(index))
        if unknown:
            raise TensorError(f"this manifest holds no tensor named {unknown[0]!r}")
        translated: dict[str, TensorView] = {}
        for name, view in index.items():
            served = names.get(name, name)
            if served in translated:
                raise TensorError(f"two tensors would both be served as {served!r}")
            translated[served] = replace(view, name=served)
        return MappingProxyType(translated)

    # -- internals -------------------------------------------------------

    def _entry(self, path: str) -> FileEntry:
        try:
            return self._entries[path]
        except KeyError:
            raise TensorError(f"{path!r} is not in this manifest") from None

    def _file_grid(self, path: str) -> tuple[list[int], tuple[tuple[CASRef, int], ...]]:
        """Object start offsets and (ref, length) pairs, in file order."""

        cached = self._grid.get(path)
        if cached is not None:
            return cached
        objects = self._entry(path).objects()
        starts: list[int] = []
        position = 0
        for _ref, length in objects:
            starts.append(position)
            position += length
        built = (starts, objects)
        self._grid[path] = built
        return built

    def _map(self, ref: CASRef, size: int) -> memoryview:
        """An mmap of one CAS object, verified at most once per reader."""

        key = ref.digest
        handle = self._maps.get(key)
        if handle is None:
            # Opening under the shared store lock keeps collection from
            # unlinking the object between path resolution and open. Once the
            # mapping exists the bytes are pinned by the inode, so the lock is
            # not held for the lifetime of the view.
            with self._cas._store_lock():
                path = self._cas.object_path(ref)
                with path.open("rb") as source:
                    handle = mmap.mmap(source.fileno(), 0, prot=mmap.PROT_READ)
            self._maps[key] = handle
        view = memoryview(handle)
        if len(view) != size:
            raise TensorError(f"{ref}: object is {len(view)} bytes, manifest says {size}")
        if self._verify and key not in self._verified:
            # Hashing the mapping is a single pass and copies nothing.
            if hashlib.sha256(view).hexdigest() != key:
                raise TensorError(f"{ref}: object bytes do not match their digest")
            self._verified.add(key)
        return view

    def _pieces(self, path: str, offset: int, length: int) -> Iterator[memoryview]:
        if self._closed:
            raise TensorError("this reader is closed")
        entry = self._entry(path)
        if offset < 0 or length < 0 or offset + length > entry.size_bytes:
            raise TensorError(
                f"{path}: range [{offset}, {offset + length}) escapes a "
                f"{entry.size_bytes}-byte file"
            )
        if length == 0:
            return
        starts, objects = self._file_grid(path)
        end = offset + length
        index = bisect.bisect_right(starts, offset) - 1
        while index < len(objects):
            start = starts[index]
            if start >= end:
                break
            ref, size = objects[index]
            view = self._map(ref, size)
            begin = max(offset, start) - start
            stop = min(end, start + size) - start
            yield view[begin:stop]
            index += 1

    # -- tensor index ----------------------------------------------------

    def _tensors(self) -> dict[str, TensorView]:
        if self._index is not None:
            return self._index
        index: dict[str, TensorView] = {}
        for path in self._entries:
            for view in self._parse_container(path):
                if view.name in index:
                    raise TensorError(
                        f"tensor {view.name!r} appears in both "
                        f"{index[view.name].file!r} and {path!r}"
                    )
                index[view.name] = view
        self._index = index
        return index

    def _parse_container(self, path: str) -> Iterator[TensorView]:
        """Dispatch on what the file actually is, not on what it is called."""

        if path.endswith(_GGUF_SUFFIX):
            yield from self._parse_gguf(path)
        elif path.endswith(_SAFETENSORS_SUFFIX):
            yield from self._parse_header(path)

    def _parse_gguf(self, path: str) -> Iterator[TensorView]:
        """Enumerate a GGUF file from its directory, reading no tensor bytes.

        GGUF carries its quantization block geometry per tensor, which is the
        one thing safetensors has no equivalent for and the one thing a
        consumer building a quantized parameter needs.
        """

        entry = self._entry(path)

        def read(offset: int, length: int) -> bytes:
            return self.read_range(path, offset, length)

        try:
            declared = list(gguf.parse(read, entry.size_bytes))
        except gguf.GGUFError as error:
            raise TensorError(f"{path}: {error}") from None
        for tensor in declared:
            yield TensorView(
                name=tensor.name,
                file=path,
                format=_GGUF_FORMAT,
                dtype=tensor.dtype,
                shape=tensor.shape,
                nbytes=tensor.nbytes,
                offset=tensor.offset,
                block=BlockLayout(tensor.block_elements, tensor.block_bytes),
                _reader=self,
            )

    def _parse_header(self, path: str) -> Iterator[TensorView]:
        """Recover names, dtypes and shapes, which the plan itself discards.

        The planner sorts tensor spans and keeps only offsets, so the name to
        byte-range mapping lives in the safetensors header. The header is read
        through the same range primitive as everything else; object zero is
        not special-cased.
        """

        entry = self._entry(path)
        if entry.size_bytes < _PREFIX_SIZE + 2:
            return
        header_size = int.from_bytes(self.read_range(path, 0, _PREFIX_SIZE), "little")
        if not 2 <= header_size <= _MAX_HEADER_BYTES:
            return
        header_end = _PREFIX_SIZE + header_size
        if header_end > entry.size_bytes:
            return
        raw = self.read_range(path, _PREFIX_SIZE, header_size)
        try:
            header = json.loads(raw)
        except ValueError:
            return
        if not isinstance(header, dict):
            return
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(descriptor, dict):
                raise TensorError(f"{path}: tensor {name!r} has a malformed header entry")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not isinstance(offsets, list)
                or len(offsets) != 2
            ):
                raise TensorError(f"{path}: tensor {name!r} has a malformed header entry")
            start, stop = offsets
            nbytes = stop - start
            elements = 1
            for dimension in shape:
                elements *= dimension
            # The header states the byte span twice over: explicitly, and
            # implicitly through dtype and shape. Disagreement means the
            # offsets cannot be trusted to slice with.
            if elements * DTYPE_BITS[dtype] != nbytes * 8:
                raise TensorError(
                    f"{path}: tensor {name!r} declares {nbytes} bytes but "
                    f"{dtype}{list(shape)} needs {elements * DTYPE_BITS[dtype] / 8:.0f}"
                )
            yield TensorView(
                name=name,
                file=path,
                format=_SAFETENSORS_FORMAT,
                dtype=dtype,
                shape=tuple(shape),
                nbytes=nbytes,
                offset=header_end + start,
                block=_safetensors_block(dtype),
                _reader=self,
            )


def open_tensors(
    cas: LocalCAS,
    manifest: RepositoryManifest | CASRef | str,
    *,
    verify: bool = True,
) -> TensorReader:
    """Open every tensor in a manifest, by value or by stored reference."""

    if not isinstance(manifest, RepositoryManifest):
        manifest = cas.load_manifest(manifest)
    return TensorReader(cas, manifest, verify=verify)


def read_entry(cas: LocalCAS, entry: FileEntry, *, verify: bool = True) -> bytes:
    """One committed file's whole contents, straight from its CAS objects.

    This is what whole-file materialization was actually asked for whenever the
    caller only wanted the bytes: no temporary, no copy on disk, no fsync.

    Because this reads the entire file it can also confirm the reconstruction
    against the manifest's whole-file digest, which a partial tensor read
    cannot. Per-tensor reads therefore verify objects only, and that is the
    point -- the whole-file hash is exactly the cost direct reads exist to
    stop paying on every load.
    """

    reader = TensorReader(cas, RepositoryManifest((entry,)), verify=verify)
    try:
        data = reader.read_range(entry.path, 0, entry.size_bytes)
    finally:
        reader.close()
    if verify and hashlib.sha256(data).hexdigest() != entry.digest.digest:
        raise TensorError(
            f"{entry.path}: reconstructed bytes do not match the file manifest"
        )
    return data
