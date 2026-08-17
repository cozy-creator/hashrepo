"""#81: a derived layout, checked against numpy's own answer.

The Rust suite proves the adapter's decision procedure and its object
accounting. What is proven here is the part a self-referential test cannot:
that the generalized permute this system applies is the SAME re-arrangement
the real library performs — `reshape -> transpose -> reshape`, the shape
llama.cpp's converter uses to rope-permute q and k. The red proof is a
corrupted axis list: it produces different bytes, and numpy says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import safetensors.numpy
from numpy.typing import NDArray
from tensorfs.native import FileRecord, ObjectStore, derive

Array = NDArray[numpy.float32]

HEADS = 4
HEAD_DIM = 4
COLUMNS = 8

CANONICAL = json.dumps(
    {
        "format": "tensorfs-contract-v1",
        "name": "test.canonical",
        "version": 1,
        "tensors": [
            {
                "role": "blocks.{i}.attn.q",
                "pattern": "blocks.{i}.attn.q_proj.weight",
                "rank": 2,
            }
        ],
    }
)

#: The llama.cpp rope permute: reshape(heads, 2, head_dim/2, cols),
#: swap axes 1 and 2, reshape back.
ROPE = json.dumps(
    {
        "format": "tensorfs-contract-v1",
        "name": "test.rope-permuted",
        "version": 1,
        "tensors": [
            {
                "role": "blocks.{i}.attn.q",
                "pattern": "blocks.{i}.attn.q_proj.weight",
                "rank": 2,
                "permute": {
                    "view": ["auto", 2, HEAD_DIM // 2, "shape[1]"],
                    "axes": [0, 2, 1, 3],
                },
            }
        ],
    }
)


def _weights() -> Array:
    generator = numpy.random.default_rng(811)
    return generator.random((HEADS * HEAD_DIM, COLUMNS), dtype=numpy.float32)


def _rope_permute(weights: Array, axes: tuple[int, int, int, int]) -> Array:
    """What llama.cpp's converter does, in numpy."""
    return (
        weights.reshape(HEADS, 2, HEAD_DIM // 2, COLUMNS)
        .transpose(axes)
        .reshape(HEADS * HEAD_DIM, COLUMNS)
    )


def _ingest(store: ObjectStore, path: Path) -> tuple[str, list[FileRecord]]:
    plan, admitted = store.admit_file(path)
    return plan.planner, [
        FileRecord.data(item.digest, item.length) for item in admitted
    ]


def _derived_file(tmp_path: Path, store: ObjectStore, records: list[FileRecord],
                  planner: str, source: str, target: str) -> Path:
    derived, stamp = derive(store, planner, records, source, target)
    assert stamp == json.loads(target)["name"] + "@1"
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "derived.safetensors"
    with out.open("wb") as handle:
        for record in derived:
            assert record.digest is not None
            handle.write(store.read_object(record.digest))
    return out


def test_a_derived_permute_matches_numpys_own_rearrangement(tmp_path: Path) -> None:
    weights = _weights()
    original = tmp_path / "canonical.safetensors"
    safetensors.numpy.save_file({"blocks.0.attn.q_proj.weight": weights}, original)

    store = ObjectStore(tmp_path / "store")
    planner, records = _ingest(store, original)
    derived = _derived_file(tmp_path, store, records, planner, CANONICAL, ROPE)

    # The real library reads the derived file, and what it reads is exactly
    # numpy's own reshape -> transpose -> reshape.
    served = safetensors.numpy.load_file(derived)["blocks.0.attn.q_proj.weight"]
    expected = _rope_permute(weights, (0, 2, 1, 3))
    numpy.testing.assert_array_equal(served, expected)
    assert served.shape == weights.shape
    assert not numpy.array_equal(served, weights), "a rope permute is not a no-op"

    # Layout, not math: the multiset of values is untouched.
    numpy.testing.assert_array_equal(numpy.sort(served, axis=None), numpy.sort(weights, axis=None))


def test_deriving_back_returns_the_source_bytes(tmp_path: Path) -> None:
    weights = _weights()
    permuted = tmp_path / "permuted.safetensors"
    safetensors.numpy.save_file(
        {"blocks.0.attn.q_proj.weight": _rope_permute(weights, (0, 2, 1, 3))}, permuted
    )

    store = ObjectStore(tmp_path / "store")
    planner, records = _ingest(store, permuted)
    # Source declares the permute, target does not: the adapter runs it
    # backwards, and the round trip has to be exact.
    back = _derived_file(tmp_path, store, records, planner, ROPE, CANONICAL)
    served = safetensors.numpy.load_file(back)["blocks.0.attn.q_proj.weight"]
    numpy.testing.assert_array_equal(served, weights)


def test_a_corrupted_axis_list_serves_different_bytes(tmp_path: Path) -> None:
    # The red proof. Same file, same machinery, one wrong axis list: numpy
    # sees a different tensor. A test that could not tell these apart would
    # not be testing the permute at all.
    weights = _weights()
    original = tmp_path / "canonical.safetensors"
    safetensors.numpy.save_file({"blocks.0.attn.q_proj.weight": weights}, original)

    store = ObjectStore(tmp_path / "store")
    planner, records = _ingest(store, original)
    corrupted = ROPE.replace('"axes": [0, 2, 1, 3]', '"axes": [0, 1, 3, 2]').replace(
        "test.rope-permuted", "test.corrupted"
    )

    honest = safetensors.numpy.load_file(
        _derived_file(tmp_path, store, records, planner, CANONICAL, ROPE)
    )["blocks.0.attn.q_proj.weight"]
    wrong = safetensors.numpy.load_file(
        _derived_file(tmp_path / "b", store, records, planner, CANONICAL, corrupted)
    )["blocks.0.attn.q_proj.weight"]

    assert not numpy.array_equal(honest, wrong)
    numpy.testing.assert_array_equal(honest, _rope_permute(weights, (0, 2, 1, 3)))
    numpy.testing.assert_array_equal(
        wrong,
        weights.reshape(HEADS, 2, HEAD_DIM // 2, COLUMNS)
        .transpose((0, 1, 3, 2))
        .reshape(HEADS * HEAD_DIM, COLUMNS),
    )
