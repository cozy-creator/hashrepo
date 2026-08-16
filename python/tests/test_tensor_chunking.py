from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import TypedDict, cast

import pytest
import tensorfs
from tensorfs import FileEntry, LocalCAS, RepositoryManifest, read_entry
from tensorfs import chunking as chunking_policy

HEADER_SAMPLES = Path(__file__).parent / "testdata" / "safetensors_header_samples.json"


class _HeaderSample(TypedDict):
    source: str
    observed_header_len: int
    dtype: str
    shape: list[int]
    byte_length: int


_REAL_HEADER_SAMPLES = cast(list[_HeaderSample], json.loads(HEADER_SAMPLES.read_text()))

_OFFICIAL_DTYPE_BITS = {
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


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> bytes:
    offset = 0
    header: dict[str, object] = {}
    for name, payload in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = (
        len(encoded).to_bytes(8, "little") + encoded + b"".join(tensor for _name, tensor in tensors)
    )
    path.write_bytes(payload)
    return payload


def _write_typed_safetensors(path: Path, dtype: str, shape: list[int], body: bytes) -> bytes:
    header = {
        "weight": {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, len(body)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = len(encoded).to_bytes(8, "little") + encoded + body
    path.write_bytes(payload)
    return payload


def _write_sparse_typed_safetensors(
    path: Path, dtype: str, shape: list[int], body_size: int
) -> int:
    header = {
        "weight": {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, body_size],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        handle.seek(body_size - 1, os.SEEK_CUR)
        handle.write(b"\0")
    return 8 + len(encoded)


def _chunk_digests(entry: FileEntry) -> list[str]:
    return [str(chunk.digest) for chunk in entry.chunks]


def _stored_bytes(cas: LocalCAS) -> int:
    return sum(path.stat().st_size for path in cas.objects.rglob("*") if path.is_file())


def test_header_only_safetensors_uses_one_explicit_chunk(tmp_path: Path) -> None:
    source = tmp_path / "empty.safetensors"
    expected = _write_safetensors(source, [])
    cas = LocalCAS(tmp_path / "cas")

    entry = cas.ingest_file(source)

    assert [chunk.length for chunk in entry.chunks] == [len(expected)]
    assert read_entry(cas, entry) == expected


def test_unpadded_two_byte_empty_header_is_valid_safetensors(tmp_path: Path) -> None:
    source = tmp_path / "empty-unpadded.safetensors"
    expected = (2).to_bytes(8, "little") + b"{}"
    source.write_bytes(expected)
    cas = LocalCAS(tmp_path / "cas")

    entry = cas.ingest_file(source)

    assert [chunk.length for chunk in entry.chunks] == [len(expected)]


def test_valid_oversized_header_region_is_split_before_tensor_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    object_ceiling = 64
    monkeypatch.setattr(chunking_policy, "MAX_CHUNK_SIZE", object_ceiling)
    source = tmp_path / "large-header.safetensors"
    header = b'{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    header_length = object_ceiling + 8
    source.write_bytes(
        header_length.to_bytes(8, "little")
        + header
        + b" " * (header_length - len(header))
        + b"x"
    )

    cas = LocalCAS(tmp_path / "cas")
    entry = cas.ingest_file(source)

    assert [chunk.length for chunk in entry.chunks] == [object_ceiling, 16, 1]
    assert len(read_entry(cas, entry)) == source.stat().st_size


@pytest.mark.parametrize(("dtype", "bits"), _OFFICIAL_DTYPE_BITS.items())
def test_every_current_safetensors_dtype_uses_tensor_aligned_chunks(
    tmp_path: Path, dtype: str, bits: int
) -> None:
    assert chunking_policy.DTYPE_BITS == _OFFICIAL_DTYPE_BITS
    elements = 4 if bits == 6 else 2 if bits == 4 else 1
    body = bytes(range(elements * bits // 8))
    source = tmp_path / f"{dtype}.safetensors"
    expected = _write_typed_safetensors(source, dtype, [elements], body)
    cas = LocalCAS(tmp_path / "cas")

    entry = cas.ingest_file(source)

    header_end = len(expected) - len(body)
    assert [chunk.length for chunk in entry.chunks] == [header_end, len(body)]
    assert read_entry(cas, entry) == expected


def test_misaligned_sub_byte_tensor_uses_bounded_fallback(tmp_path: Path) -> None:
    source = tmp_path / "misaligned.safetensors"
    _write_typed_safetensors(source, "F4", [1], b"x")
    cas = LocalCAS(tmp_path / "cas")

    requested = cas.ingest_file(source)

    assert requested.chunks == ()


@pytest.mark.parametrize("sample", _REAL_HEADER_SAMPLES, ids=lambda row: row["source"])
def test_real_fp8_header_samples_use_tensor_aligned_chunks(
    tmp_path: Path, sample: _HeaderSample
) -> None:
    source = tmp_path / f"{sample['source']}.safetensors"
    body_size = sample["byte_length"]
    header_end = _write_sparse_typed_safetensors(
        source,
        sample["dtype"],
        sample["shape"],
        body_size,
    )

    with source.open("rb") as handle:
        lengths = chunking_policy.plan_chunks(
            handle,
            source.stat().st_size,
        )

    assert lengths == (header_end, body_size)
    assert sample["observed_header_len"] >= header_end


def test_pathologically_nested_header_falls_back_instead_of_escaping_parser(
    tmp_path: Path,
) -> None:
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    header = b'{"x":{"dtype":"U8","shape":' + nested + b',"data_offsets":[0,0]}}'
    source = tmp_path / "nested.safetensors"
    source.write_bytes(len(header).to_bytes(8, "little") + header)
    cas = LocalCAS(tmp_path / "cas")

    requested = cas.ingest_file(source)

    assert requested.chunks == ()


def test_huge_shape_product_is_bounded_by_declared_tensor_bytes(tmp_path: Path) -> None:
    dimension = b"9" * 1000
    shape = b"[" + b",".join([dimension] * 2000) + b"]"
    header = b'{"x":{"dtype":"U8","shape":' + shape + b',"data_offsets":[0,1]}}'
    source = tmp_path / "huge-shape.safetensors"
    source.write_bytes(len(header).to_bytes(8, "little") + header + b"x")
    started = time.monotonic()

    with source.open("rb") as handle:
        lengths = chunking_policy.plan_chunks(
            handle,
            source.stat().st_size,
        )

    assert time.monotonic() - started < 1
    assert lengths == ()
    cas = LocalCAS(tmp_path / "cas")
    requested = cas.ingest_file(source)
    assert requested.chunks == ()


def test_zero_dimension_avoids_large_shape_products(tmp_path: Path) -> None:
    source = tmp_path / "zero.safetensors"
    expected = _write_typed_safetensors(source, "U8", [(1 << 64) - 1, 0], b"")
    cas = LocalCAS(tmp_path / "cas")

    entry = cas.ingest_file(source)

    assert [chunk.length for chunk in entry.chunks] == [len(expected)]


def test_shape_dimension_above_64_bit_usize_falls_back(tmp_path: Path) -> None:
    source = tmp_path / "overflow.safetensors"
    _write_typed_safetensors(source, "U8", [1 << 64, 0], b"")
    cas = LocalCAS(tmp_path / "cas")

    requested = cas.ingest_file(source)

    assert requested.chunks == ()


def test_shape_product_overflow_before_zero_dimension_falls_back(tmp_path: Path) -> None:
    source = tmp_path / "product-overflow.safetensors"
    _write_typed_safetensors(source, "U8", [(1 << 64) - 1, 2, 0], b"")
    cas = LocalCAS(tmp_path / "cas")

    requested = cas.ingest_file(source)

    assert requested.chunks == ()


@pytest.mark.parametrize(
    ("changed_names", "change_header", "expected_changed_chunks"),
    [
        ({"a"}, False, {1, 2}),
        ({"b"}, False, {3}),
        ({"c"}, False, {3}),
        ({"d"}, False, {4}),
        ({"e"}, False, {5}),
        ({"a", "d"}, True, {0, 1, 2, 4}),
    ],
)
def test_property_only_changed_tensors_header_and_packs_get_new_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_names: set[str],
    change_header: bool,
    expected_changed_chunks: set[int],
) -> None:
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 64)
    original_tensors = [
        ("a", b"a" * 80),
        ("b", b"b" * 10),
        ("c", b"c" * 12),
        ("d", b"d" * 64),
        ("e", b"e" * 5),
    ]
    changed_tensors = [
        (name, payload.upper() if name in changed_names else payload)
        for name, payload in original_tensors
    ]
    first = tmp_path / "first.safetensors"
    duplicate = tmp_path / "duplicate.bin"
    changed = tmp_path / "changed.safetensors"
    _write_safetensors(first, original_tensors, metadata={"revision": "1"})
    duplicate.write_bytes(first.read_bytes())
    _write_safetensors(
        changed,
        changed_tensors,
        metadata={"revision": "2" if change_header else "1"},
    )

    cas = LocalCAS(tmp_path / "cas")
    first_entry = cas.ingest_file(first)
    duplicate_entry = cas.ingest_file(duplicate)
    changed_entry = cas.ingest_file(changed)

    header_end = 8 + int.from_bytes(first.read_bytes()[:8], "little")
    assert [chunk.length for chunk in first_entry.chunks] == [header_end, 64, 16, 22, 64, 5]
    assert _chunk_digests(first_entry) == _chunk_digests(duplicate_entry)
    assert {
        index
        for index, (before, after) in enumerate(
            zip(_chunk_digests(first_entry), _chunk_digests(changed_entry), strict=True)
        )
        if before != after
    } == expected_changed_chunks


def test_small_tensor_change_invalidates_only_its_bounded_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 64)
    original = tmp_path / "original.safetensors"
    changed = tmp_path / "changed.safetensors"
    tensors = [("a", b"a" * 20), ("b", b"b" * 20), ("c", b"c" * 20), ("d", b"d" * 20)]
    _write_safetensors(original, tensors)
    _write_safetensors(changed, [tensors[0], ("b", b"B" * 20), *tensors[2:]])

    cas = LocalCAS(tmp_path / "cas")
    before = cas.ingest_file(original)
    after = cas.ingest_file(changed)

    assert [chunk.length for chunk in before.chunks[1:]] == [60, 20]
    changed_indexes = {
        index
        for index, (left, right) in enumerate(
            zip(_chunk_digests(before), _chunk_digests(after), strict=True)
        )
        if left != right
    }
    assert changed_indexes == {1}
    assert before.chunks[1].length <= chunking_policy._TENSOR_FLOOR_BYTES


def test_small_tensor_set_edit_can_repack_the_rest_of_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 64)
    tensors = [(f"tensor-{index}", bytes([index]) * 20) for index in range(30)]
    parent = tmp_path / "parent.safetensors"
    child = tmp_path / "child.safetensors"
    _write_safetensors(parent, tensors)
    _write_safetensors(child, tensors[1:])
    cas = LocalCAS(tmp_path / "cas")

    before = cas.ingest_file(parent)
    after = cas.ingest_file(child)

    assert set(_chunk_digests(before)[1:]).isdisjoint(_chunk_digests(after)[1:])


def test_scaled_50_plus_1_publish_adds_only_changed_tensor_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One KiB represents one GiB. This runs the real local writer and CAS while
    # preserving Paul's 50 GiB parent + 1 GiB changed-tensor ratio.
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 1024)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 1024)
    parent = tmp_path / "parent.safetensors"
    child = tmp_path / "child.safetensors"
    tensors = [
        (f"tensor_{index:02d}", hashlib.sha256(str(index).encode()).digest() * 32)
        for index in range(50)
    ]
    _write_safetensors(parent, tensors)
    changed = list(tensors)
    changed[17] = (changed[17][0], b"changed!" * 128)
    _write_safetensors(child, changed)

    cas = LocalCAS(tmp_path / "cas")
    parent_manifest = RepositoryManifest(
        (cas.ingest_file(parent),)
    )
    cas.store_manifest(parent_manifest)
    parent_bytes = _stored_bytes(cas)

    child_manifest = RepositoryManifest(
        (cas.ingest_file(child),)
    )
    cas.store_manifest(child_manifest)
    delta = _stored_bytes(cas) - parent_bytes

    assert delta == 1024 + len(child_manifest.canonical_bytes())


def test_automatic_fixed_and_tensor_layouts_read_and_collect_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_policy, "_FIXED_CHUNK_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 64)
    tensor_source = tmp_path / "model.safetensors"
    tensor_bytes = _write_safetensors(
        tensor_source,
        [("a", b"a" * 80), ("b", b"b" * 10), ("c", b"c" * 90)],
    )
    opaque_source = tmp_path / "opaque.bin"
    opaque_bytes = b"opaque" * 30
    opaque_source.write_bytes(opaque_bytes)
    cas = LocalCAS(tmp_path / "cas")
    fixed = cas.ingest_file(opaque_source, manifest_path="opaque.bin")
    aligned = cas.ingest_file(
        tensor_source,
        manifest_path="model.safetensors",
    )
    assert [chunk.length for chunk in fixed.chunks] == [64, 64, 52]
    assert [chunk.length for chunk in aligned.chunks] != [64, 64, 52]

    manifest = RepositoryManifest((fixed, aligned))
    assert read_entry(cas, fixed) == opaque_bytes
    assert read_entry(cas, aligned) == tensor_bytes

    orphan = cas.put_bytes(b"orphan")
    old = time.time() - 3600
    os.utime(cas.object_path(orphan), (old, old))
    for entry in manifest.files:
        for ref, _size in entry.objects():
            os.utime(cas.object_path(ref), (old, old))
    report = cas.collect_garbage(manifests=(manifest,), older_than=60)
    assert report.deleted == 1
    for entry in manifest.files:
        for ref, size in entry.objects():
            cas.verify_object(ref, size=size)


def test_malformed_safetensors_uses_bounded_fixed_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_policy, "_FIXED_CHUNK_BYTES", 64)
    source = tmp_path / "malformed.safetensors"
    source.write_bytes((16).to_bytes(8, "little") + b"not-json-at-all!!" + b"x" * 200)
    cas = LocalCAS(tmp_path / "cas")

    requested = cas.ingest_file(source, manifest_path="file")

    assert [chunk.length for chunk in requested.chunks] == [64, 64, 64, 33]


def test_writer_is_automatic_and_retired_selector_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunking_policy, "_FIXED_CHUNK_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_FLOOR_BYTES", 64)
    monkeypatch.setattr(chunking_policy, "_TENSOR_STRIDE_BYTES", 64)
    source = tmp_path / "model.safetensors"
    expected = _write_safetensors(source, [("weights", b"w" * 100)])
    cas = LocalCAS(tmp_path / "cas")

    entry = cas.ingest_file(source)
    header_end = len(expected) - 100
    assert [chunk.length for chunk in entry.chunks] == [header_end, 64, 36]
    assert "WriterPolicy" not in tensorfs.__all__
    assert not hasattr(tensorfs, "WriterPolicy")
    assert "writer_policy" not in inspect.signature(LocalCAS.ingest_file).parameters
    assert "writer_policy" not in inspect.signature(LocalCAS.ingest_repository).parameters
    with pytest.raises(TypeError, match="writer_policy"):
        cas.ingest_file(source, writer_policy="retired")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="writer_policy"):
        cas.ingest_repository(tmp_path, writer_policy="retired")  # type: ignore[call-arg]
