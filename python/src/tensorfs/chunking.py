from __future__ import annotations

import json
from typing import BinaryIO

from .manifest import MAX_CHUNK_SIZE

# 50 GiB / 64 MiB = 800 objects only when boundaries fill perfectly. A greedy
# small-tensor run of S bytes needs fewer than 2*S/64MiB + 1 packs; large tensors
# use ceil(size / 64 MiB). On 2026-08-13, 186
# unique local layouts measured 3,226 aligned objects vs 2,397 fixed (1.346x);
# 32/16/4 MiB floors cost 2.272x/3.378x/7.988x. Thus 64 MiB controls R2 request
# count while bounding one changed small-tensor pack to <64 MiB. te#185 phase 4
# measures the real savings of the automatic policy on a frozen-base LoRA series.
_FIXED_CHUNK_BYTES = MAX_CHUNK_SIZE
_TENSOR_FLOOR_BYTES = MAX_CHUNK_SIZE
_TENSOR_STRIDE_BYTES = MAX_CHUNK_SIZE
# safetensors 0.8.0's Rust reader refuses header lengths above 100,000,000
# bytes (MAX_HEADER_SIZE in safetensors/src/tensor.rs). Match that format
# boundary instead of conflating it with TensorFS's smaller object ceiling;
# the header region is sub-split below when it exceeds one CAS object.
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
# TensorFS v1 targets 64-bit Linux/POSIX. Safetensors decodes every shape
# dimension into Rust usize before it checks tensor byte lengths.
_MAX_USIZE = (1 << 64) - 1

# This mirrors safetensors::tensor::Dtype::bitsize at upstream 6eb4dc9. Unknown
# future dtypes take the safe fixed-boundary fallback instead of being guessed here.
_DTYPE_BITS = {
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key {key!r}")
        result[key] = value
    return result


def _fixed_lengths(size: int) -> tuple[int, ...]:
    if size <= _FIXED_CHUNK_BYTES:
        return ()
    full, remainder = divmod(size, _FIXED_CHUNK_BYTES)
    lengths = (_FIXED_CHUNK_BYTES,) * full
    return lengths + ((remainder,) if remainder else ())


def _tensor_spans(source: BinaryIO, size: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    if size < 10:
        return None
    source.seek(0)
    prefix = source.read(8)
    if len(prefix) != 8:
        return None
    header_length = int.from_bytes(prefix, "little")
    header_end = 8 + header_length
    if (
        header_length < 2
        or header_length > _MAX_SAFETENSORS_HEADER_BYTES
        or header_end > size
    ):
        return None
    header_bytes = source.read(header_length)
    if len(header_bytes) != header_length or not header_bytes.startswith(b"{"):
        return None
    try:
        raw = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(raw, dict):
        return None
    metadata = raw.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        return None

    spans: list[tuple[int, int]] = []
    for name, value in raw.items():
        if (
            not name
            or not isinstance(value, dict)
            or set(value) != {"dtype", "shape", "data_offsets"}
        ):
            return None
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            return None
        if not isinstance(shape, list) or any(
            type(dimension) is not int or not 0 <= dimension <= _MAX_USIZE
            for dimension in shape
        ):
            return None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            return None
        start, end = offsets
        if start < 0 or end < start or end > _MAX_USIZE:
            return None
        dtype_bits = _DTYPE_BITS[dtype]
        body_bits = (end - start) * 8
        elements = 1
        for dimension in shape:
            if dimension and elements > _MAX_USIZE // dimension:
                return None
            elements *= dimension
        if elements and dtype_bits > _MAX_USIZE // elements:
            return None
        tensor_bits = elements * dtype_bits
        if tensor_bits != body_bits:
            return None
        spans.append((start, end))

    spans.sort()
    data_size = size - header_end
    cursor = 0
    for start, end in spans:
        if start != cursor or end > data_size:
            return None
        cursor = end
    if cursor != data_size:
        return None
    return header_end, tuple(spans)


def _tensor_lengths(header_end: int, spans: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    header_chunks, header_remainder = divmod(header_end, MAX_CHUNK_SIZE)
    lengths = [MAX_CHUNK_SIZE] * header_chunks
    if header_remainder:
        lengths.append(header_remainder)
    packed = 0

    def flush_pack() -> None:
        nonlocal packed
        if packed:
            lengths.append(packed)
            packed = 0

    for start, end in spans:
        tensor_bytes = end - start
        if tensor_bytes == 0:
            continue
        if tensor_bytes >= _TENSOR_FLOOR_BYTES:
            flush_pack()
            full, remainder = divmod(tensor_bytes, _TENSOR_STRIDE_BYTES)
            lengths.extend((_TENSOR_STRIDE_BYTES,) * full)
            if remainder:
                lengths.append(remainder)
            continue
        if packed and packed + tensor_bytes > _TENSOR_FLOOR_BYTES:
            flush_pack()
        packed += tensor_bytes
    flush_pack()
    return tuple(lengths)


def plan_chunks(source: BinaryIO, size: int) -> tuple[int, ...]:
    """Plan the sole current writer layout for one file.

    Structurally valid safetensors use tensor-stable boundaries. Every other
    file uses bounded fixed offsets. This decision is automatic and cannot be
    selected by callers.
    """

    parsed = _tensor_spans(source, size)
    if parsed is None:
        return _fixed_lengths(size)
    return _tensor_lengths(*parsed)
