"""#83: a trimmed model is a snapshot that simply omits the records it deleted.

minimax-h3 trims its Qwen3-VL conditioner after downloading it, out of a 68 GB
component. A subset snapshot inherits the kept tensors' records and omits the
rest, so the bytes are never requested in the first place — the sync win is
proven in Rust (`crates/tensorfs-core/tests/subset_snapshot.rs`) against a real
pull. What is proven here is the other half: the trimmed file is an ordinary
safetensors file the real library opens, naming exactly the kept tensors.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy
import pytest
import safetensors.numpy
from numpy.typing import NDArray
from safetensors import safe_open
from tensorfs.native import FileRecord, ObjectStore, RecordsReader, subset

Array = NDArray[numpy.float32]

_KEPT = ("model.layers.0.weight", "model.layers.1.weight")
_TRIMMED = ("model.layers.2.weight", "lm_head.weight")


def _bodies() -> dict[str, Array]:
    generator = numpy.random.default_rng(83)
    return {
        name: generator.random((16, 64), dtype=numpy.float32)
        for name in (*_KEPT, *_TRIMMED)
    }


def _objects(store: ObjectStore) -> set[str]:
    return {path.name for path in (store.root / "objects").rglob("*") if path.is_file()}


def _digests(records: list[FileRecord]) -> list[str]:
    return [record.digest for record in records if record.digest is not None]


def test_a_subset_names_exactly_the_kept_tensors(tmp_path: Path) -> None:
    bodies = _bodies()
    original = tmp_path / "model.safetensors"
    safetensors.numpy.save_file(bodies, original)

    store = ObjectStore(tmp_path / "store")
    plan, admitted = store.admit_file(original)
    records = [FileRecord.data(item.digest, item.length) for item in admitted]
    before = _objects(store)

    # Identity pairs: trim without renaming. (Trimming and renaming are one
    # header rewrite, so one call does both when a caller wants both.)
    trimmed = subset(store, plan.planner, records, {name: name for name in _KEPT})

    assert len(_objects(store) - before) == 1, "a trim admits one header and nothing else"
    assert set(_digests(trimmed)[1:]) <= set(_digests(records)), (
        "every kept tensor's object is the source's own"
    )
    assert len(_digests(trimmed)) == 1 + len(_KEPT), "and the trimmed records are simply gone"

    reader = RecordsReader(store, trimmed)
    raw = reader.read_at(0, reader.length)
    header_length: int = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    assert set(header) == set(_KEPT), "the header names exactly the kept tensors"

    # The native read path and the reference library agree on the same bytes.
    proof = tmp_path / "trimmed.safetensors"
    proof.write_bytes(raw)
    with safe_open(proof, framework="numpy") as reference:
        assert set(reference.keys()) == set(_KEPT)
        for name in _KEPT:
            assert numpy.array_equal(reference.get_tensor(name), bodies[name]), name

    # The tensors it dropped are not reachable through it at all.
    omitted = set(_digests(records)) - set(_digests(trimmed))
    assert len(omitted) == 1 + len(_TRIMMED), "the old header and the trimmed tensors"


def test_a_trim_that_names_an_absent_tensor_is_refused(tmp_path: Path) -> None:
    bodies = _bodies()
    original = tmp_path / "model.safetensors"
    safetensors.numpy.save_file(bodies, original)
    store = ObjectStore(tmp_path / "store")
    plan, admitted = store.admit_file(original)
    records = [FileRecord.data(item.digest, item.length) for item in admitted]

    before = _objects(store)
    with pytest.raises(Exception, match="holds no tensor named"):
        subset(store, plan.planner, records, {"model.layers.9.weight": "x"})
    assert _objects(store) == before
