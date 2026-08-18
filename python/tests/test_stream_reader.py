"""#115: the streamed store->VRAM read surface, on the Python boundary.

Three claims. (1) ``tensors`` is FILE-OFFSET order by contract — proven
against a fixture whose header key order differs from its offset order.
(2) ``readinto`` is byte-exact across the buffered and O_DIRECT arms,
multi-object tensors and holes included, into caller-owned buffers.
(3) The copy releases the GIL: a second Python thread makes progress during
a large ``readinto``.
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import pytest
from tensorfs.native import (
    FileRecord,
    ObjectStore,
    RecordsReader,
    TensorStreamReader,
)

MIB = 1024 * 1024


def _safetensors(entries: list[tuple[str, int, int]], payload_length: int) -> bytes:
    """A file whose HEADER lists ``entries`` in the given order, while the
    byte layout follows each entry's own offsets — the two orders need not
    agree, and for this suite's fixtures they deliberately do not."""

    header_object: dict[str, dict[str, object]] = {}
    for name, start, stop in entries:
        header_object[name] = {
            "dtype": "U8",
            "shape": [stop - start],
            "data_offsets": [start, stop],
        }
    header = json.dumps(header_object).encode()
    payload = bytes((index * 37 + 11) % 251 for index in range(payload_length))
    return struct.pack("<Q", len(header)) + header + payload


def _ingest(store: ObjectStore, path: Path, data: bytes) -> list[FileRecord]:
    path.write_bytes(data)
    _plan, admitted = store.admit_file(path)
    return [FileRecord.data(item.digest, item.length) for item in admitted]


def test_tensors_iterate_in_file_offset_order_not_header_order(tmp_path: Path) -> None:
    # Header order: z first, then a — but a's bytes come FIRST in the file.
    data = _safetensors([("z", 2048, 4096), ("a", 0, 2048)], 4096)
    store = ObjectStore(tmp_path / "store")
    records = _ingest(store, tmp_path / "f.safetensors", data)

    reader = TensorStreamReader(store, records)
    assert reader.format == "safetensors-v1"
    names = [tensor.name for tensor in reader.tensors]
    assert names == ["a", "z"], "ascending offset, whatever the header spelled"
    offsets = [tensor.offset for tensor in reader.tensors]
    assert offsets == sorted(offsets)
    assert [tensor.nbytes for tensor in reader.tensors] == [2048, 2048]


def _direct_or_skip(store: ObjectStore, records: list[FileRecord]) -> TensorStreamReader:
    reader = TensorStreamReader(store, records, direct=True)
    probe = bytearray(512)
    try:
        reader.readinto(0, 512, probe)
    except OSError as error:  # pragma: no cover - filesystem-dependent
        pytest.skip(f"this filesystem refuses O_DIRECT: {error}")
    return reader


def test_readinto_is_byte_exact_across_arms_holes_included(tmp_path: Path) -> None:
    # One 96 MiB tensor (two objects: 64 + 32 MiB) and a small neighbour.
    big = 96 * MIB
    data = _safetensors(
        [("big", 0, big), ("small", big, big + 1024)], big + 1024
    )
    store = ObjectStore(tmp_path / "store")
    records = _ingest(store, tmp_path / "big.safetensors", data)

    # Punch a hole: the big tensor's second object becomes sparse.
    hole_index = 2  # header, 64 MiB, 32 MiB, small
    hole_length = records[hole_index].length
    assert hole_length == 32 * MIB
    records[hole_index] = FileRecord.hole(hole_length)

    reference = RecordsReader(store, records)
    buffered = TensorStreamReader(store, records)
    assert buffered.direct is False

    (big_meta, small_meta) = buffered.tensors
    assert big_meta.name == "big" and big_meta.nbytes == big

    # The whole multi-object tensor, into a caller-owned buffer.
    destination = bytearray(big)
    assert buffered.read_tensor_into("big", destination) == big
    assert bytes(destination) == reference.read_at(big_meta.offset, big)

    # The hole read zeros — zero-fill, never skip.
    hole_start = sum(record.length for record in records[:hole_index])
    probe = bytearray(1024)
    buffered.readinto(hole_start + 5 * MIB, 1024, probe)
    assert bytes(probe) == b"\x00" * 1024

    # The O_DIRECT arm returns the identical bytes, tail and hole included.
    direct = _direct_or_skip(store, records)
    assert direct.direct is True
    twin = bytearray(big)
    assert direct.read_tensor_into("big", twin) == big
    assert twin == destination
    probes = [
        (0, 4096),
        (big_meta.offset + 63 * MIB - 7, 2 * MIB),
        (buffered.length - 999, 999),
    ]
    for offset, length in probes:
        ours, theirs = bytearray(length), bytearray(length)
        buffered.readinto(offset, length, ours)
        direct.readinto(offset, length, theirs)
        assert ours == theirs, (offset, length)


def test_readinto_refuses_short_and_readonly_buffers(tmp_path: Path) -> None:
    data = _safetensors([("a", 0, 2048)], 2048)
    store = ObjectStore(tmp_path / "store")
    records = _ingest(store, tmp_path / "f.safetensors", data)
    reader = TensorStreamReader(store, records)

    with pytest.raises(Exception, match="holds 16 bytes"):
        reader.read_tensor_into("a", bytearray(16))
    with pytest.raises(Exception, match="no tensor named"):
        reader.read_tensor_into("missing", bytearray(16))
    with pytest.raises(Exception, match="exceeds the committed length"):
        reader.readinto(reader.length - 1, 2, bytearray(2))


def test_readinto_releases_the_gil(tmp_path: Path) -> None:
    """A second thread makes progress during a large readinto."""

    big = 192 * MIB
    data = _safetensors([("big", 0, big)], big)
    store = ObjectStore(tmp_path / "store")
    records = _ingest(store, tmp_path / "gil.safetensors", data)
    reader = TensorStreamReader(store, records)
    destination = bytearray(big)

    running = threading.Event()
    finished = threading.Event()
    ticks = 0

    def spin() -> None:
        nonlocal ticks
        running.set()
        while not finished.is_set():
            ticks += 1

    thread = threading.Thread(target=spin)
    thread.start()
    running.wait()
    baseline = ticks
    reader.read_tensor_into("big", destination)
    during = ticks - baseline
    finished.set()
    thread.join()

    # If the copy held the GIL, the spinner could not run at all between the
    # baseline capture and the return: `during` would be ~0.
    assert during > 100, f"the spinner made {during} ticks during readinto"
