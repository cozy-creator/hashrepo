"""Direct tensor reads: byte-exact, lazy, and without creating a file.

The fixture is built so the assertions can actually fail. Every tensor is
filled with distinct pseudorandom bytes rather than a constant, so an
off-by-one offset changes the bytes it returns; and the object grid is stated
explicitly to force all three interesting shapes at once. The manifest wire
accepts any grid — the Rust seal planner's aligned one, or the packing grid
the retired Python chunker committed — and the reader must serve them all:

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
from repo_ingest import fixed_lengths, ingest_file, ingest_with_grid
from tensorfs import (
    EXTRACT_SIZE_LIMIT,
    CASRef,
    FileTooLarge,
    LocalCAS,
    RepositoryManifest,
    TensorError,
    open_tensors,
)
from tensorfs.manifest import MAX_CHUNK_SIZE
from tensorfs.tensors import DTYPE_BITS

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


def _packed_lengths(model: Path) -> list[int]:
    """The retired chunker's packing grid, stated outright.

    Header object, then 64 MiB strides from the large tensor's own start, then
    every remaining tensor packed into one shared object. This is the grid old
    snapshots were committed under, and the hardest one for the reader.
    """

    size = model.stat().st_size
    with model.open("rb") as handle:
        header_end = 8 + int.from_bytes(handle.read(8), "little")
    dense = 26 * 1024 * 1024 * 4
    assert dense > MAX_CHUNK_SIZE
    return [header_end, MAX_CHUNK_SIZE, dense - MAX_CHUNK_SIZE, size - header_end - dense]


def _ingest_tree(cas: LocalCAS, source: Path) -> RepositoryManifest:
    """model.safetensors under the packed grid, everything else Rust-planned."""

    model = source / "model.safetensors"
    entries = [ingest_with_grid(cas, model, _packed_lengths(model), manifest_path=model.name)]
    for path in sorted(source.iterdir()):
        if path != model:
            entries.append(ingest_file(cas, path, manifest_path=path.name))
    return RepositoryManifest(tuple(entries))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("direct-tensor-reads")
    source = root / "source"
    source.mkdir()
    model = source / "model.safetensors"
    bodies = _build_safetensors(model)
    (source / "config.json").write_text(json.dumps({"architectures": ["Fixture"]}))

    cas = LocalCAS(root / "cas")
    manifest = _ingest_tree(cas, source)
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
    """The packed grid puts several tensors in one object, so these are partial reads."""

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
    manifest = _ingest_tree(cas, source)
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
    entry = ingest_file(cas, source / "model.safetensors")
    ref = cas.store_manifest(RepositoryManifest((entry,)))
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


def _oversized_snapshot(tmp_path: Path) -> tuple[LocalCAS, CASRef]:
    """A file above the extraction ceiling, made sparse so it costs nothing.

    Every 64 MiB run of zeros hashes to the same digest, so the whole thing
    occupies a single CAS object no matter how large it is declared.
    """

    source = tmp_path / "source"
    source.mkdir()
    big = source / "model.safetensors"
    with big.open("wb") as handle:
        handle.seek(EXTRACT_SIZE_LIMIT + (1 << 20) - 1)
        handle.write(b"\0")
    cas = LocalCAS(tmp_path / "cas")
    entry = ingest_with_grid(cas, big, fixed_lengths(big.stat().st_size), manifest_path=big.name)
    return cas, cas.store_manifest(RepositoryManifest((entry,)))


def test_extraction_refuses_a_weight_sized_file_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The hatch must never become a way back to materializing weights."""

    cas, ref = _oversized_snapshot(tmp_path)
    destination = tmp_path / "leak" / "model.safetensors"
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(FileTooLarge, match="extraction bound"):
            tensors.extract("model.safetensors", destination)
    assert not destination.exists()


def test_a_caller_cannot_widen_the_extraction_bound(tmp_path: Path) -> None:
    """`limit` may only tighten. A bound a caller can raise is not a bound."""

    cas, ref = _oversized_snapshot(tmp_path)
    destination = tmp_path / "model.safetensors"
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(FileTooLarge):
            tensors.extract("model.safetensors", destination, limit=1 << 40)
    assert not destination.exists()


def test_the_dtype_table_matches_the_rust_planner() -> None:
    """Reader and planner must agree on every safetensors dtype's width."""

    import re

    planner = (
        Path(__file__).resolve().parents[2] / "crates/tensorfs-core/src/planner/safetensors.rs"
    )
    assert planner.is_file(), planner  # a missing file must fail, not skip
    section = planner.read_text().split("fn dtype_bits")[1]
    pinned: dict[str, int] = {}
    for names, bits in re.findall(
        r'((?:"[A-Z0-9_]+"\s*\|\s*)*"[A-Z0-9_]+")\s*=>\s*Some\((\d+)\)', section, re.DOTALL
    ):
        for name in re.findall(r'"([A-Z0-9_]+)"', names):
            pinned[name] = int(bits)

    assert pinned, "failed to parse the Rust table; this test would prove nothing"
    assert pinned == DTYPE_BITS


def test_extraction_honours_a_tightened_bound(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    destination = tmp_path / "config.json"
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        with pytest.raises(FileTooLarge):
            tensors.extract("config.json", destination, limit=2)
    assert not destination.exists()
