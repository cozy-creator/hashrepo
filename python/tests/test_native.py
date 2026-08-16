"""The compiled extension, driven through the real installed package.

These are integration tests over the actual PyO3 boundary, not mocks: every
assertion goes through the same `tensorfs.native` import a consumer uses. The
format assertions are driven by the frozen v1 spec vectors, so a binding that
decodes or plans differently from the Rust reference fails here rather than
agreeing with itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tensorfs import native
from tensorfs.native import (
    FileRecord,
    FormatError,
    ObjectStore,
    RecordsReader,
    StoreError,
    TensorfsError,
)

_SPEC = Path(__file__).resolve().parents[2] / "spec" / "v1"
_PLANNER_VECTORS = _SPEC / "planner-vectors" / "planner-vectors.json"
_TFM1_VECTORS = _SPEC / "tfm1-vectors" / "tfm1-vectors.json"

_ZERO = "0" * 64


def _module_directory() -> Path:
    """Where the installed package lives. `__file__` is optional on
    `ModuleType`, so it is narrowed once here rather than at every use."""
    location = native.__file__
    assert location is not None, "tensorfs.native has no file location"
    return Path(location).resolve().parent


def _fixture_bytes(root: Path, relative: str) -> bytes:
    return bytes.fromhex((root / relative).read_text(encoding="utf-8").strip())


def _bare(digest: str) -> str:
    return digest.removeprefix("sha256:")


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, object] = json.load(handle)
    return loaded


# ---------------------------------------------------------------------------
# The extension is present, typed, and matches its stub
# ---------------------------------------------------------------------------


def test_the_extension_is_compiled_and_versioned() -> None:
    assert isinstance(native.__version__, str)
    assert native.__version__
    # A pure-Python module would have no __file__ extension suffix like this.
    from tensorfs import _tensorfs

    suffix = Path(_tensorfs.__file__).suffix
    assert suffix in {".so", ".pyd", ".dylib"}, suffix


def test_stub_matches_the_extension() -> None:
    """Every name the stub promises exists on the compiled module.

    This is what keeps `py.typed` honest across the extension boundary: PyO3
    emits no typing information, so the `.pyi` is an assertion about the
    module, and an assertion nothing checks is decoration.
    """
    from tensorfs import _tensorfs

    stub = Path(native.__file__).with_name("_tensorfs.pyi")
    promised = {
        line.split("(")[0].removeprefix("def ").removeprefix("class ").strip().rstrip(":")
        for line in stub.read_text(encoding="utf-8").splitlines()
        if line.startswith(("def ", "class "))
    }
    promised |= {
        line.split(":")[0].strip()
        for line in stub.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith((" ", "\t", "#", '"')) and ": Final" in line
    }
    missing = sorted(name for name in promised if not hasattr(_tensorfs, name))
    assert not missing, f"the stub promises names the extension does not have: {missing}"


def test_every_public_name_resolves() -> None:
    missing = sorted(name for name in native.__all__ if not hasattr(native, name))
    assert not missing, missing


def test_py_typed_ships_beside_the_extension() -> None:
    assert (Path(native.__file__).with_name("py.typed")).is_file()


# ---------------------------------------------------------------------------
# The CAS
# ---------------------------------------------------------------------------


def test_object_store_round_trip(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    admitted = store.put_bytes(b"native-round-trip")

    assert len(admitted.digest) == 64
    assert admitted.digest.islower()
    assert admitted.length == len(b"native-round-trip")
    assert admitted.preexisting is False

    assert store.read_object(admitted.digest) == b"native-round-trip"
    assert store.verify(admitted.digest) == admitted.length
    assert store.contains(admitted.digest) is True
    assert store.read_object_range(admitted.digest, 7, 5) == b"round"

    # Re-admitting identical bytes converges on one object, re-verified.
    again = store.put_bytes(b"native-round-trip")
    assert again.digest == admitted.digest
    assert again.preexisting is True


def test_object_store_accepts_the_algorithm_tagged_spelling(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    admitted = store.put_bytes(b"tagged")
    assert store.read_object(f"sha256:{admitted.digest}") == b"tagged"


def test_object_store_refuses_a_malformed_digest(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    with pytest.raises(TensorfsError):
        store.read_object("not-a-digest")
    with pytest.raises(TensorfsError):
        store.read_object(_ZERO.upper())


def test_object_store_refuses_an_absent_object(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    assert store.contains(_ZERO) is False
    with pytest.raises(StoreError):
        store.verify(_ZERO)


def test_collect_abandoned_temps_takes_the_callers_grace(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    store.put_bytes(b"resident")
    # A grace larger than the store's age collects nothing; there is no
    # constant here, the caller supplies the window.
    collected = store.collect_abandoned_temps(3600.0)
    assert collected.deleted == 0
    with pytest.raises(TensorfsError):
        store.collect_abandoned_temps(-1.0)


def test_object_store_root_is_a_path(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ObjectStore(root)
    assert Path(store.root) == root


# ---------------------------------------------------------------------------
# The planner, against the frozen v1 vectors
# ---------------------------------------------------------------------------


def _planner_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = _load(_PLANNER_VECTORS)["cases"]  # type: ignore[assignment]
    return cases


@pytest.mark.skipif(not _PLANNER_VECTORS.is_file(), reason="spec vectors are not present")
@pytest.mark.parametrize("case", _planner_cases(), ids=lambda case: str(case["name"]))
def test_planner_vectors_reproduce_through_the_binding(case: dict[str, object]) -> None:
    root = _PLANNER_VECTORS.parent
    # `zero_tail` is a run of trailing zeros the fixture does not spell out --
    # the fp8 cases are real production shapes with tens of MiB of padding.
    # The Rust vector test synthesizes it through a lazy ByteSource; here it
    # is materialized, which costs a few MiB and keeps the case covered
    # rather than skipped.
    zero_tail = case.get("zero_tail", 0)
    assert isinstance(zero_tail, int)
    data = _fixture_bytes(root, str(case["fixture"])) + bytes(zero_tail)
    expected = case["expected"]
    assert isinstance(expected, dict)

    plan = native.plan_bytes(data)
    assert plan.planner == expected["planner"]
    assert plan.file_size == expected["file_size"]

    objects = expected["objects"]
    assert isinstance(objects, list)
    assert [(region.offset, region.length, region.kind) for region in plan.regions] == [
        (item["offset"], item["length"], item["kind"]) for item in objects
    ]

    hashed = native.plan_and_hash_bytes(data)
    assert [(item.digest, item.length, item.kind) for item in hashed.objects] == [
        (_bare(str(item["digest"])), item["length"], item["kind"]) for item in objects
    ]


def test_plan_file_agrees_with_plan_bytes(tmp_path: Path) -> None:
    root = _PLANNER_VECTORS.parent
    data = _fixture_bytes(root, "fixtures/safetensors-two-tensors.hex")
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)

    from_file = native.plan_file(path)
    from_bytes = native.plan_bytes(data)
    assert from_file.planner == from_bytes.planner == "safetensors-v1"
    assert [(region.offset, region.length, region.kind) for region in from_file.regions] == [
        (region.offset, region.length, region.kind) for region in from_bytes.regions
    ]


def test_admit_file_admits_every_planned_region(tmp_path: Path) -> None:
    root = _PLANNER_VECTORS.parent
    data = _fixture_bytes(root, "fixtures/safetensors-two-tensors.hex")
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)

    store = ObjectStore(tmp_path / "store")
    plan, admitted = store.admit_file(path)

    assert plan.planner == "safetensors-v1"
    assert len(admitted) == len(plan.regions)
    for region, item in zip(plan.regions, admitted, strict=True):
        assert item.length == region.length
        assert store.read_object(item.digest) == data[region.offset : region.offset + region.length]


# ---------------------------------------------------------------------------
# TFM1 and TFP1, against the frozen v1 vectors
# ---------------------------------------------------------------------------


def _tfm1_section(name: str) -> list[dict[str, object]]:
    section: list[dict[str, object]] = _load(_TFM1_VECTORS)[name]  # type: ignore[assignment]
    return section


@pytest.mark.skipif(not _TFM1_VECTORS.is_file(), reason="spec vectors are not present")
@pytest.mark.parametrize("case", _tfm1_section("golden"), ids=lambda case: str(case["name"]))
def test_tfm1_golden_vectors_decode_and_re_encode_byte_identically(
    case: dict[str, object],
) -> None:
    data = _fixture_bytes(_TFM1_VECTORS.parent, str(case["fixture"]))
    snapshot = native.decode_snapshot(data)

    assert snapshot.snapshot_id == case["snapshot_id"]
    assert native.snapshot_id_of(data) == case["snapshot_id"]
    # Re-encoding reproduces the canonical bytes; that is what makes the id an
    # identity rather than a checksum of one particular encoder's output.
    assert snapshot.to_bytes() == data


@pytest.mark.skipif(not _TFM1_VECTORS.is_file(), reason="spec vectors are not present")
@pytest.mark.parametrize("case", _tfm1_section("refusals"), ids=lambda case: str(case["name"]))
def test_tfm1_refusal_vectors_are_refused(case: dict[str, object]) -> None:
    data = _fixture_bytes(_TFM1_VECTORS.parent, str(case["fixture"]))
    with pytest.raises(FormatError):
        native.decode_snapshot(data)


def test_decoded_file_entry_exposes_its_records() -> None:
    data = _fixture_bytes(_TFM1_VECTORS.parent, "fixtures/sparse-file.hex")
    snapshot = native.decode_snapshot(data)
    files = [entry for entry in snapshot.entries if entry.kind == "file"]
    assert files, "the sparse-file vector has at least one file entry"

    entry = files[0]
    assert entry.records is not None
    assert entry.logical_size == sum(record.length for record in entry.records)
    assert any(record.kind == "hole" for record in entry.records)
    assert snapshot.file_records(entry.path) is not None
    assert snapshot.file_records("no/such/path") is None


def test_pack_round_trip() -> None:
    payloads = [b"", b"one", b"two-two", bytes(4096)]
    digests = [native.plan_and_hash_bytes(payload) for payload in payloads]
    pairs = [
        (plan.objects[0].digest, payload)
        for plan, payload in zip(digests, payloads, strict=True)
        if plan.objects
    ]

    encoded = native.encode_pack(pairs)
    decoded = native.decode_pack(encoded)
    # A canonical pack is digest-ordered, not input-ordered.
    assert [(item.digest, item.data) for item in decoded] == sorted(pairs)


def test_a_corrupt_pack_is_refused() -> None:
    with pytest.raises(FormatError):
        native.decode_pack(b"not a pack")


# ---------------------------------------------------------------------------
# The byte-source path: one tensor, no materialized file
# ---------------------------------------------------------------------------


def _committed_records(store: ObjectStore, path: Path) -> list[FileRecord]:
    _, admitted = store.admit_file(path)
    return [FileRecord.data(item.digest, item.length) for item in admitted]


def test_records_reader_reads_a_tensor_without_materializing_the_file(tmp_path: Path) -> None:
    """The headline of this binding.

    A committed file's records are read as a byte source, the planner runs
    over that source to find the tensor boundaries, and exactly one tensor's
    bytes come back -- with no whole-file copy anywhere in the path.
    """
    data = _fixture_bytes(_PLANNER_VECTORS.parent, "fixtures/safetensors-two-tensors.hex")
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)

    store = ObjectStore(tmp_path / "store")
    records = _committed_records(store, path)

    # The file itself is gone. Everything below reads only immutable objects.
    path.unlink()

    reader = RecordsReader(store, records)
    assert reader.length == len(data)

    plan = reader.plan()
    assert plan.planner == "safetensors-v1"
    tensors = [region for region in plan.regions if region.kind == "tensor"]
    assert len(tensors) == 2

    for region in tensors:
        assert reader.read_at(region.offset, region.length) == (
            data[region.offset : region.offset + region.length]
        )
    assert reader.read_at(0, reader.length) == data


def test_records_reader_reads_holes_as_zeros_and_spans_records(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    head = store.put_bytes(b"HEAD")
    tail = store.put_bytes(b"TAIL")
    records = [
        FileRecord.data(head.digest, head.length),
        FileRecord.hole(6),
        FileRecord.data(tail.digest, tail.length),
    ]
    reader = RecordsReader(store, records)

    assert reader.length == 14
    assert reader.read_at(0, 14) == b"HEAD" + bytes(6) + b"TAIL"
    # A read that starts inside one record and ends inside another.
    assert reader.read_at(2, 10) == b"AD" + bytes(6) + b"TA"


def test_records_reader_refuses_a_range_past_the_committed_length(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    only = store.put_bytes(b"short")
    reader = RecordsReader(store, [FileRecord.data(only.digest, only.length)])
    with pytest.raises(OSError):
        reader.read_at(0, 6)


def test_records_reader_refuses_a_missing_object(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "store")
    reader = RecordsReader(store, [FileRecord.data(_ZERO, 4)])
    with pytest.raises(OSError):
        reader.read_at(0, 4)


def test_file_record_validates_its_own_shape() -> None:
    with pytest.raises(TensorfsError):
        FileRecord("data", 4)
    with pytest.raises(TensorfsError):
        FileRecord("hole", 4, _ZERO)
    with pytest.raises(TensorfsError):
        FileRecord("neither", 4)


# ---------------------------------------------------------------------------
# What the wheel does NOT carry
# ---------------------------------------------------------------------------


def test_the_wheel_ships_no_daemon_and_no_console_script() -> None:
    """The `tensorfsd` mount daemon is deliberately absent.

    Pods cannot mount FUSE — opening `/dev/fuse` is denied by the device
    cgroup even for root and `CAP_SYS_ADMIN` is not in the container's
    bounding set — so the wheel ships native reads and the daemon lives on
    the `shelf/tensorfsd` branch (tag `shelf/tensorfsd-split`). Asserted
    rather than assumed: a stray `_bin/` from an old build would otherwise
    ride into a wheel unnoticed.
    """
    package = _module_directory()
    assert not (package / "_bin").exists(), "an unshipped daemon directory reached the package"
    assert not hasattr(native, "daemon_binary")
