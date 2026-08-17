"""Enumerate a GGUF file's tensors from its directory, reading no tensor bytes.

The seal planner gives GGUF the same tensor-aligned object treatment
safetensors gets, so a GGUF tensor is addressable exactly the same way once its
name, geometry and data offset are known. Recovering those means walking the
GGUF header, which is what this module does.

Every bound and the GGML block table below mirror
``crates/tensorfs-core/src/planner/gguf.rs`` so the two implementations accept
and reject the same files.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
MAX_METADATA_COUNT = 1_000_000
MAX_TENSOR_COUNT = 1_000_000
MAX_SYMBOL_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 16_000_000
MAX_ARRAY_DEPTH = 16
MAX_TENSOR_NAME_LEN = 63
MAX_DIMENSIONS = 4

_ALIGNMENT_KEY = b"general.alignment"

# (elements per encoded block, bytes per encoded block), keyed by ggml type id.
# Pinned against ggml-org/llama.cpp 7e4c0a96880dae4fc4268ad441f8a6446bd5460a;
# ids 4, 5, 31..33 and 36..38 are removed upstream and deliberately absent.
GGML_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 40),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, 56),
    30: (1, 2),
    34: (256, 54),
    35: (256, 66),
    39: (32, 17),
    40: (64, 36),
    41: (128, 18),
    42: (64, 18),
}

GGML_NAME: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}

# The reverse of GGML_NAME, so a `TensorView`'s dtype round-trips back into the
# type id a directory entry has to carry.
GGML_ID: dict[str, int] = {name: identifier for identifier, name in GGML_NAME.items()}

# GGUF metadata value type ids -> fixed width, for the ones that have one.
_FIXED_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# The Rust planner refuses anything below this (crates/tensorfs-core/src/
# planner/gguf.rs), so composing a file that violates it would produce a
# snapshot our own ingest path would reject.
MIN_ALIGNMENT = 8


class GGUFError(ValueError):
    """A byte stream did not parse as the GGUF structure it claimed to be."""


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    name: str
    ggml_type: int
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    block_elements: int
    block_bytes: int


class _Cursor:
    """A forward reader over a byte range supplier, refilled in bounded blocks."""

    __slots__ = ("_at", "_block", "_buffer", "_read", "_start", "_size")

    def __init__(self, read: Callable[[int, int], bytes], size: int) -> None:
        self._read = read
        self._size = size
        self._at = 0
        self._buffer = b""
        self._start = 0
        self._block = 1 << 20

    @property
    def position(self) -> int:
        return self._at

    def take(self, length: int) -> bytes:
        if length < 0 or self._at + length > self._size:
            raise GGUFError("GGUF header runs past the end of the file")
        end = self._at + length
        if not (self._start <= self._at and end <= self._start + len(self._buffer)):
            span = min(max(length, self._block), self._size - self._at)
            self._buffer = self._read(self._at, span)
            self._start = self._at
        begin = self._at - self._start
        self._at = end
        return self._buffer[begin : begin + length]

    def skip(self, length: int) -> None:
        if length < 0 or self._at + length > self._size:
            raise GGUFError("GGUF header runs past the end of the file")
        self._at += length

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.take(8))[0])

    def symbol(self, limit: int) -> bytes:
        length = self.u64()
        if length > limit:
            raise GGUFError("GGUF symbol exceeds its bound")
        return self.take(length)


def _skip_value(cursor: _Cursor, value_type: int, depth: int) -> None:
    if depth > MAX_ARRAY_DEPTH:
        raise GGUFError("GGUF metadata nests too deeply")
    width = _FIXED_WIDTH.get(value_type)
    if width is not None:
        cursor.skip(width)
        return
    if value_type == _TYPE_STRING:
        cursor.symbol(MAX_SYMBOL_BYTES)
        return
    if value_type != _TYPE_ARRAY:
        raise GGUFError(f"unknown GGUF metadata value type {value_type}")
    element_type = cursor.u32()
    count = cursor.u64()
    if count > MAX_ARRAY_ELEMENTS:
        raise GGUFError("GGUF metadata array exceeds its bound")
    element_width = _FIXED_WIDTH.get(element_type)
    if element_width is not None:
        # Fixed-width elements are a flat run; skipping avoids touching them.
        cursor.skip(element_width * count)
        return
    for _ in range(count):
        _skip_value(cursor, element_type, depth + 1)


def _tensor_nbytes(shape: tuple[int, ...], elements: int, block_bytes: int) -> int:
    """Rows are encoded blockwise, so only dimension zero divides by the block."""

    if shape[0] % elements:
        raise GGUFError("GGUF tensor row is not a whole number of blocks")
    total = (shape[0] // elements) * block_bytes
    for dimension in shape[1:]:
        total *= dimension
    return total


@dataclass(frozen=True, slots=True)
class GGUFHeader:
    """Everything before the data section, plus where the data section starts.

    ``metadata`` is the encoded key/value block **verbatim** — the bytes
    between the 24-byte prefix and the tensor directory. Carrying it as bytes
    rather than as decoded values is deliberate: a conversion rewrites tensors,
    not model metadata, and re-encoding a value it never looked at is a chance
    to change a file for no reason.
    """

    version: int
    alignment: int
    metadata_count: int
    metadata: bytes
    directory_start: int
    directory_end: int
    data_start: int
    tensors: tuple[GGUFTensor, ...]


def tensor_nbytes(ggml_type: int, shape: tuple[int, ...]) -> int:
    """Encoded size of a tensor with these GGUF dimensions."""

    layout = GGML_LAYOUT.get(ggml_type)
    if layout is None:
        raise GGUFError(f"unsupported ggml type {ggml_type}")
    elements, block_bytes = layout
    if not shape:
        raise GGUFError("a GGUF tensor needs at least one dimension")
    return _tensor_nbytes(shape, elements, block_bytes)


def read_header(read: Callable[[int, int], bytes], size: int) -> GGUFHeader:
    """Parse the whole directory, reading no tensor body."""

    cursor = _Cursor(read, size)
    if size < 24 or cursor.take(4) != MAGIC:
        raise GGUFError("not a GGUF file")
    version = cursor.u32()
    if version not in (2, 3):
        raise GGUFError(f"unsupported GGUF version {version}")
    tensor_count = cursor.u64()
    metadata_count = cursor.u64()
    if tensor_count > MAX_TENSOR_COUNT or metadata_count > MAX_METADATA_COUNT:
        raise GGUFError("GGUF directory exceeds its bounds")

    alignment = DEFAULT_ALIGNMENT
    for _ in range(metadata_count):
        key = cursor.symbol(MAX_SYMBOL_BYTES)
        value_type = cursor.u32()
        if key == _ALIGNMENT_KEY and value_type == 4:
            alignment = int(struct.unpack("<I", cursor.take(4))[0])
            if alignment == 0 or alignment & (alignment - 1):
                raise GGUFError("general.alignment must be a power of two")
            continue
        _skip_value(cursor, value_type, 0)

    directory_start = cursor.position
    metadata = read(24, directory_start - 24) if directory_start > 24 else b""

    declared: list[GGUFTensor] = []
    for _ in range(tensor_count):
        name = cursor.symbol(MAX_TENSOR_NAME_LEN).decode("utf-8", "strict")
        dimensions = cursor.u32()
        if not 1 <= dimensions <= MAX_DIMENSIONS:
            raise GGUFError(f"GGUF tensor {name!r} declares {dimensions} dimensions")
        shape = tuple(cursor.u64() for _ in range(dimensions))
        ggml_type = cursor.u32()
        offset = cursor.u64()
        layout = GGML_LAYOUT.get(ggml_type)
        if layout is None:
            raise GGUFError(f"GGUF tensor {name!r} uses unsupported ggml type {ggml_type}")
        elements, block_bytes = layout
        declared.append(
            GGUFTensor(
                name=name,
                ggml_type=ggml_type,
                dtype=GGML_NAME.get(ggml_type, f"GGML_{ggml_type}"),
                shape=shape,
                offset=offset,
                nbytes=_tensor_nbytes(shape, elements, block_bytes),
                block_elements=elements,
                block_bytes=block_bytes,
            )
        )

    directory_end = cursor.position
    # Tensor offsets are relative to the aligned start of the data section.
    data_start = align_up(directory_end, alignment)
    absolute: list[GGUFTensor] = []
    for tensor in declared:
        start = data_start + tensor.offset
        if start + tensor.nbytes > size:
            raise GGUFError(f"GGUF tensor {tensor.name!r} runs past the end of the file")
        absolute.append(
            GGUFTensor(
                name=tensor.name,
                ggml_type=tensor.ggml_type,
                dtype=tensor.dtype,
                shape=tensor.shape,
                offset=start,
                nbytes=tensor.nbytes,
                block_elements=tensor.block_elements,
                block_bytes=tensor.block_bytes,
            )
        )
    return GGUFHeader(
        version=version,
        alignment=alignment,
        metadata_count=metadata_count,
        metadata=metadata,
        directory_start=directory_start,
        directory_end=directory_end,
        data_start=data_start,
        tensors=tuple(absolute),
    )


def parse(read: Callable[[int, int], bytes], size: int) -> Iterator[GGUFTensor]:
    """Yield every tensor a GGUF file declares, in declaration order.

    ``read(offset, length)`` supplies bytes; no tensor body is ever requested.
    """

    return iter(read_header(read, size).tensors)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def encode_symbol(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def encode_prefix(version: int, tensor_count: int, metadata_count: int) -> bytes:
    """The 24 bytes every GGUF file opens with."""

    return MAGIC + struct.pack("<IQQ", version, tensor_count, metadata_count)


def encode_tensor_info(
    name: str, shape: tuple[int, ...], ggml_type: int, offset: int
) -> bytes:
    """One tensor directory entry, spelled exactly as the readers expect.

    ``shape`` is GGUF's own ``ne`` order — fastest-varying dimension first —
    which is the order :class:`GGUFTensor` reports, so a tensor read out of one
    file and written into another needs no transposition of its geometry.
    """

    encoded = name.encode("utf-8")
    if not 0 < len(encoded) <= MAX_TENSOR_NAME_LEN:
        raise GGUFError(f"GGUF tensor name {name!r} is not a usable name")
    if not 1 <= len(shape) <= MAX_DIMENSIONS:
        raise GGUFError(f"GGUF tensor {name!r} cannot have {len(shape)} dimensions")
    return (
        encode_symbol(encoded)
        + struct.pack("<I", len(shape))
        + b"".join(struct.pack("<Q", dimension) for dimension in shape)
        + struct.pack("<IQ", ggml_type, offset)
    )


def type_id(dtype: str) -> int:
    """The ggml type id a :class:`~tensorfs.tensors.TensorView` dtype names."""

    identifier = GGML_ID.get(dtype)
    if identifier is None and dtype.startswith("GGML_"):
        try:
            identifier = int(dtype.removeprefix("GGML_"))
        except ValueError:
            identifier = None
    if identifier is None or identifier not in GGML_LAYOUT:
        raise GGUFError(f"unknown ggml type {dtype!r}")
    return identifier
