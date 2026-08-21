"""tensorfs#159: the no-torch Python client for the one Rust fill path."""

from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path

import pytest
from tensorfs.native import FileRecord, LayoutError, ObjectStore, TensorStreamReader


def _container(shape: tuple[int, ...]) -> tuple[bytes, bytes]:
    elements = 1
    for dim in shape:
        elements *= dim
    payload = bytes((index * 37 + 11) % 251 for index in range(elements))
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": list(shape),
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode()
    return struct.pack("<Q", len(header)) + header + payload, payload


def _reader(tmp_path: Path, shape: tuple[int, ...]) -> tuple[TensorStreamReader, bytes]:
    data, payload = _container(shape)
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)
    store = ObjectStore(tmp_path / "store")
    _plan, admitted = store.admit_file(path)
    records = [FileRecord.data(item.digest, item.length) for item in admitted]
    return TensorStreamReader(store, records), payload


def test_identity_fill_writes_caller_memory_and_reports_the_walk(tmp_path: Path) -> None:
    reader, payload = _reader(tmp_path, (2, 3, 4, 5))
    destination = bytearray(7 + len(payload))

    stats = reader.fill_host_into("weight", destination, destination_offset=7)

    assert destination[:7] == b"\0" * 7
    assert destination[7:] == payload
    assert stats.source_bytes == len(payload)
    assert stats.destination_bytes == len(payload)
    assert stats.padding_bytes == 0
    assert stats.runs == 1
    assert stats.chunks >= 1


def test_host_address_fill_uses_only_primitive_destination_data(tmp_path: Path) -> None:
    reader, payload = _reader(tmp_path, (2, 3, 4, 5))
    allocation = (ctypes.c_ubyte * (9 + len(payload)))()

    stats = reader.fill_host_address(
        "weight", ctypes.addressof(allocation), len(allocation), destination_offset=9
    )

    assert bytes(allocation[:9]) == b"\0" * 9
    assert bytes(allocation[9:]) == payload
    assert stats.destination_bytes == len(payload)


def test_morphism_is_applied_by_the_same_bound_fill(tmp_path: Path) -> None:
    shape = (2, 3, 4, 5)
    reader, payload = _reader(tmp_path, shape)
    destination = bytearray(len(payload))

    stats = reader.fill_host_into("weight", destination, layout="torch.channels_last-2d@1")

    expected = bytes(
        payload[((n * shape[1] + c) * shape[2] + h) * shape[3] + w]
        for n in range(shape[0])
        for h in range(shape[2])
        for w in range(shape[3])
        for c in range(shape[1])
    )
    assert destination == expected
    assert stats.runs == len(payload)


def test_fill_refusals_cross_the_binding_typed(tmp_path: Path) -> None:
    reader, payload = _reader(tmp_path, (2, 3, 4, 5))

    with pytest.raises(LayoutError, match="buffer holds"):
        reader.fill_host_into("weight", bytearray(len(payload) - 1))
    with pytest.raises(LayoutError, match="pointer is null"):
        reader.fill_host_address("weight", 0, len(payload))
    allocation = (ctypes.c_ubyte * len(payload))()
    with pytest.raises(LayoutError, match="buffer holds"):
        reader.fill_host_address("weight", ctypes.addressof(allocation), len(payload) - 1)
    with pytest.raises(LayoutError, match="rank-5"):
        reader.fill_host_into("weight", bytearray(len(payload)), layout="torch.channels_last-3d@1")
    with pytest.raises(LayoutError, match="no such arrangement"):
        reader.fill_host_into("weight", bytearray(len(payload)), layout="candidate.unratified@1")
    with pytest.raises(Exception, match="no tensor named"):
        reader.fill_host_into("absent", bytearray(len(payload)))


def test_no_torch_type_or_import_exists_at_the_fill_seam() -> None:
    import inspect

    import tensorfs.native as native

    signatures = (
        inspect.signature(native.TensorStreamReader.fill_host_into),
        inspect.signature(native.TensorStreamReader.fill_host_address),
        inspect.signature(native.CudaFillClient.fill),
    )
    for signature in signatures:
        assert all(
            parameter.annotation is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
        assert signature.return_annotation is inspect.Signature.empty
        assert "tensor" not in set(signature.parameters)
    assert "torch" not in Path(native.__file__).read_text(encoding="utf-8")
