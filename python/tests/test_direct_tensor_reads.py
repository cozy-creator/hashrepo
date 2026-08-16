"""Direct tensor reads: byte-exact, lazy, and without creating a file.

The fixture is built so the assertions can actually fail. Every tensor is
filled with distinct pseudorandom bytes rather than a constant, so an
off-by-one offset changes the bytes it returns; and the tensor sizes are chosen
to force all three interesting shapes at once against the Python chunker:

* ``dense.weight`` is larger than 64 MiB, so it spans several CAS objects;
* ``mid.weight`` is a prefix of a shared, packed object;
* ``small.*`` sit at nonzero offsets inside that same shared object.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import pytest
from tensorfs import (
    CASRef,
    FileTooLarge,
    LocalCAS,
    TensorError,
    open_tensors,
)
from tensorfs.manifest import MAX_CHUNK_SIZE

MIB = 1 << 20

# dense is 104 MiB: two objects, the first a full 64 MiB stride.
_TENSORS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("dense.weight", "F32", (26, 1024, 1024)),
    ("mid.weight", "F32", (8, 1024, 1024)),
    ("small.alpha", "F32", (4, 4)),
    ("small.beta", "BF16", (8, 8)),
    ("small.gamma", "I64", (3,)),
    ("empty.weight", "F32", (0, 16)),
)

_DTYPE_BYTES = {"F32": 4, "BF16": 2, "I64": 8}


def _payload(name: str, size: int) -> bytes:
    """Position-sensitive bytes: a shifted read cannot compare equal."""

    return random.Random(name).randbytes(size)


def _build_safetensors(path: Path) -> dict[str, bytes]:
    bodies: dict[str, bytes] = {}
    header: dict[str, object] = {}
    cursor = 0
    for name, dtype, shape in _TENSORS:
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * _DTYPE_BYTES[dtype]
        bodies[name] = _payload(name, size)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name, _dtype, _shape in _TENSORS:
            handle.write(bodies[name])
    return bodies


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("direct-tensor-reads")
    source = root / "source"
    source.mkdir()
    model = source / "model.safetensors"
    bodies = _build_safetensors(model)
    (source / "config.json").write_text(json.dumps({"architectures": ["Fixture"]}))

    cas = LocalCAS(root / "cas")
    manifest = cas.ingest_repository(source)
    ref = cas.store_manifest(manifest)
    return {
        "cas_root": root / "cas",
        "manifest": manifest,
        "ref": ref,
        "bodies": bodies,
        "model": model,
    }


def _entry(fixture: dict[str, object], path: str):  # type: ignore[no-untyped-def]
    manifest = fixture["manifest"]
    return next(item for item in manifest.files if item.path == path)  # type: ignore[attr-defined]


def test_the_fixture_actually_forces_the_shapes_under_test(
    fixture: dict[str, object],
) -> None:
    """Guard the guard: if the grid collapses, the tests below stop binding."""

    entry = _entry(fixture, "model.safetensors")
    lengths = [chunk.length for chunk in entry.chunks]
    assert len(lengths) >= 4, lengths
    # The 104 MiB tensor occupies whole objects, the first a full stride.
    assert MAX_CHUNK_SIZE in lengths, lengths
    # Something larger than one tensor is packed together, proving that the
    # partial-object read path is exercised rather than merely available.
    assert any(length not in (MAX_CHUNK_SIZE,) and length > 32 * MIB for length in lengths)


def test_every_tensor_is_byte_exact_and_no_file_is_created(
    fixture: dict[str, object],
) -> None:
    cas_root = Path(str(fixture["cas_root"]))
    cas = LocalCAS(cas_root)
    bodies: dict[str, bytes] = fixture["bodies"]  # type: ignore[assignment]

    before = {path: path.stat().st_size for path in sorted(cas_root.rglob("*"))}
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        assert set(tensors) == set(bodies)
        for name, expected in bodies.items():
            view = tensors[name]
            assert view.nbytes == len(expected), name
            assert view.tobytes() == expected, name
    after = {path: path.stat().st_size for path in sorted(cas_root.rglob("*"))}

    # Reading wrote nothing: no temporary, no reconstructed file, no growth.
    assert before == after


def test_a_tensor_larger_than_one_object_is_reassembled_across_the_boundary(
    fixture: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    expected: bytes = fixture["bodies"]["dense.weight"]  # type: ignore[index]
    assert len(expected) > MAX_CHUNK_SIZE

    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        view = tensors["dense.weight"]
        sizes = [len(piece) for piece in view.pieces()]
        assert len(sizes) > 1, "this tensor must span more than one object"
        assert sizes[0] == MAX_CHUNK_SIZE
        assert sum(sizes) == len(expected)
        assert view.tobytes() == expected

        # readinto reassembles into caller memory with the same result.
        buffer = bytearray(view.nbytes)
        assert view.readinto(buffer) == view.nbytes
        assert bytes(buffer) == expected


def test_packed_tensors_are_read_out_of_a_shared_object(
    fixture: dict[str, object],
) -> None:
    """The Python chunker packs sub-64 MiB tensors, so these are partial reads."""

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    bodies: dict[str, bytes] = fixture["bodies"]  # type: ignore[assignment]

    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        shared = [tensors[name] for name in ("mid.weight", "small.alpha", "small.beta")]
        # Each is a single piece that is strictly smaller than the object it
        # came from, i.e. a genuine offset-and-length read inside an object.
        for view in shared:
            pieces = list(view.pieces())
            assert len(pieces) == 1, view.name
            assert len(pieces[0]) == view.nbytes
            assert view.tobytes() == bodies[view.name], view.name

        # They really do share one object rather than each owning one.
        assert len({tensors[name].file for name in ("mid.weight", "small.beta")}) == 1


def test_a_zero_length_tensor_reads_as_no_bytes(fixture: dict[str, object]) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        view = tensors["empty.weight"]
        assert view.nbytes == 0
        assert view.shape == (0, 16)
        assert view.tobytes() == b""
        assert list(view.pieces()) == []


def test_opening_reads_only_the_header(fixture: dict[str, object]) -> None:
    """Laziness is the product claim; assert it structurally, not by timing."""

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    entry = _entry(fixture, "model.safetensors")
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        assert tensors.keys()
        # Only the object(s) holding the header have been mapped, even though
        # every tensor is now listed with its dtype and shape.
        mapped_after_open = len(tensors._maps)
        assert mapped_after_open == 1, mapped_after_open

        tensors["small.alpha"].tobytes()
        after_small = len(tensors._maps)
        assert after_small == 2, after_small

        # The 104 MiB tensor still has not been touched.
        assert after_small < len(entry.chunks)


def test_dtype_and_shape_come_back_from_the_header(fixture: dict[str, object]) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        assert tensors["dense.weight"].dtype == "F32"
        assert tensors["dense.weight"].shape == (26, 1024, 1024)
        assert tensors["small.beta"].dtype == "BF16"
        assert tensors["small.gamma"].shape == (3,)


def test_config_json_is_readable_without_a_directory(
    fixture: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        assert json.loads(tensors.read_file("config.json"))["architectures"] == ["Fixture"]


def test_a_range_outside_the_file_is_refused(fixture: dict[str, object]) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    entry = _entry(fixture, "model.safetensors")
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        with pytest.raises(TensorError):
            tensors.read_range("model.safetensors", entry.size_bytes - 4, 64)
        with pytest.raises(TensorError):
            tensors.read_range("absent.bin", 0, 1)


def test_verification_catches_a_corrupted_object(tmp_path: Path) -> None:
    """The discriminating test for verify=True: flip one byte, demand a raise."""

    source = tmp_path / "source"
    source.mkdir()
    bodies = _build_safetensors(source / "model.safetensors")
    cas = LocalCAS(tmp_path / "cas")
    manifest = cas.ingest_repository(source)
    ref = cas.store_manifest(manifest)

    entry = next(item for item in manifest.files if item.path == "model.safetensors")
    target = cas.object_path(entry.chunks[-1].digest)
    original = target.read_bytes()
    target.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])

    with open_tensors(cas, ref, verify=True) as tensors:
        with pytest.raises(TensorError):
            tensors["small.gamma"].tobytes()

    # verify=False is the documented zero-copy path and does not hash, so it
    # hands back the damaged bytes instead of raising. Asserting the damage is
    # visible keeps the knob honest about exactly what it costs the caller.
    with open_tensors(cas, ref, verify=False) as tensors:
        damaged = tensors["mid.weight"].tobytes()
    assert damaged != bodies["mid.weight"]
    assert damaged[1:] == bodies["mid.weight"][1:]


def test_a_header_whose_span_contradicts_its_dtype_is_refused(tmp_path: Path) -> None:
    """The offsets are only safe to slice with if dtype and shape agree."""

    source = tmp_path / "source"
    source.mkdir()
    # F32 (4, 4) is 64 bytes; claim 32 and the offsets cannot be trusted.
    header = {"bad.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 32]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with (source / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"\x00" * 32)

    cas = LocalCAS(tmp_path / "cas")
    ref = cas.store_manifest(cas.ingest_repository(source))
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(TensorError, match="declares 32 bytes"):
            # Mapping.keys() is a lazy view and would not build the index.
            list(tensors)


def test_a_buffer_outlives_the_reader_that_produced_it(
    fixture: dict[str, object],
) -> None:
    """Closing must not invalidate views the caller still holds."""

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    expected: bytes = fixture["bodies"]["small.alpha"]  # type: ignore[index]

    reader = open_tensors(cas, fixture["ref"])  # type: ignore[arg-type]
    held = next(reader["small.alpha"].pieces())
    reader.close()

    # The mapping survived because the view still references it.
    assert bytes(held) == expected


def test_matches_the_reference_safetensors_implementation(
    fixture: dict[str, object],
) -> None:
    """Validate offsets against the real library, not against our own writer."""

    safetensors = pytest.importorskip(
        "safetensors",
        reason="install safetensors+numpy in a separate venv to run this arm",
    )
    numpy = pytest.importorskip("numpy")
    del numpy

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    model = Path(str(fixture["model"]))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        with safetensors.safe_open(model, framework="numpy") as reference:
            names = set(reference.keys())
            assert names == set(tensors)
            for name in sorted(names):
                view = tensors[name]
                if view.dtype == "BF16":
                    continue  # numpy has no bfloat16; our own arm covers it
                expected = reference.get_tensor(name)
                assert expected.tobytes() == view.tobytes(), name
                assert tuple(expected.shape) == view.shape, name


def test_a_small_non_tensor_file_can_be_extracted_to_a_path(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    """The one remaining reason to create a file, for path-only consumers."""

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        target = tensors.extract("config.json", tmp_path / "out" / "config.json")
    assert json.loads(target.read_bytes())["architectures"] == ["Fixture"]


def _large_sparse_snapshot(tmp_path: Path) -> tuple[LocalCAS, CASRef]:
    """A multi-hundred-MiB file, made sparse so it costs nothing.

    Every 64 MiB run of zeros hashes to the same digest, so the whole thing
    occupies a single CAS object no matter how large it is declared.
    """

    source = tmp_path / "source"
    source.mkdir()
    big = source / "model.safetensors"
    with big.open("wb") as handle:
        handle.seek((513 << 20) - 1)
        handle.write(b"\0")
    cas = LocalCAS(tmp_path / "cas")
    manifest = cas.ingest_repository(source)
    return cas, cas.store_manifest(manifest)


def test_extraction_has_no_size_cap_and_streams_any_file(tmp_path: Path) -> None:
    """No hard cap -- ruled by Paul, 2026-08-16.

    "no hard-cap on materialization size; we just want to avoid large
    non-tensor files being in tensorfs (our chunked CAS system)". The control
    is scope at ingestion, not a limit here: extraction streams record by
    record, so a large file costs O(one block) of memory, and refusing it
    would only push a legitimate caller back toward a worse copy path. An
    earlier revision shipped a 512 MiB `FileTooLarge` ceiling; this test pins
    its absence.
    """

    cas, ref = _large_sparse_snapshot(tmp_path)
    destination = tmp_path / "out" / "model.safetensors"
    with open_tensors(cas, ref) as tensors:
        target = tensors.extract("model.safetensors", destination)
    assert target.stat().st_size == 513 << 20


def test_extraction_honours_a_tightened_bound(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    destination = tmp_path / "config.json"
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        with pytest.raises(FileTooLarge):
            tensors.extract("config.json", destination, limit=2)
    assert not destination.exists()
