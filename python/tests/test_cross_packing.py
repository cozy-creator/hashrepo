"""#80: the same pipeline in three packings shares every tensor object.

All-in-one, a diffusers-style split tree under different key names, and an
HF-style shard set are the same bytes cut the same way, because the chunk grid
is relative to each TENSOR's start rather than to a position in the file.
Nothing here composes anything: this is what plain ingestion already does.

Each fixture is written by the REAL ``safetensors`` library, so the packings are
the ones the ecosystem actually produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy
import safetensors.numpy
from numpy.typing import NDArray
from repo_ingest import ingest_file, ingest_with_grid
from tensorfs import FileEntry, LocalCAS

Array = NDArray[numpy.float32]

# One "pipeline": two components under two spellings of the same weights.
_COMPONENTS = {
    "text_encoder": ("encoder.layers.0.weight", "encoder.layers.1.weight"),
    "dit": ("blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight"),
    "vae": ("decoder.conv_out.weight",),
}
_ALL_IN_ONE = {
    "encoder.layers.0.weight": "cond_stage_model.transformer.layers.0.weight",
    "encoder.layers.1.weight": "cond_stage_model.transformer.layers.1.weight",
    "blocks.0.attn.qkv.weight": "model.diffusion_model.blocks.0.attn.qkv.weight",
    "blocks.0.mlp.fc1.weight": "model.diffusion_model.blocks.0.mlp.fc1.weight",
    "decoder.conv_out.weight": "first_stage_model.decoder.conv_out.weight",
}


def _weights() -> dict[str, Array]:
    generator = numpy.random.default_rng(8080)
    return {
        name: generator.random((48, 96), dtype=numpy.float32)
        for names in _COMPONENTS.values()
        for name in names
    }


def _objects(cas: LocalCAS) -> set[str]:
    return {path.name for path in cas.objects.rglob("*") if path.is_file()}


def _tensor_digests(entries: list[FileEntry]) -> set[str]:
    """Every entry's chunks except its leading header object."""

    return {chunk.digest.digest for entry in entries for chunk in entry.chunks[1:]}


def _save(path: Path, weights: dict[str, NDArray[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.numpy.save_file(weights, path)
    return path


def _all_in_one(root: Path, weights: dict[str, Array]) -> Path:
    return _save(
        root / "all-in-one.safetensors",
        {_ALL_IN_ONE[name]: body for name, body in weights.items()},
    )


def _split_tree(root: Path, weights: dict[str, Array]) -> list[Path]:
    return [
        _save(
            root / "tree" / component / "model.safetensors",
            {name: weights[name] for name in names},
        )
        for component, names in _COMPONENTS.items()
    ]


def _shards(root: Path, weights: dict[str, Array]) -> list[Path]:
    names = list(weights)
    halves = [names[: len(names) // 2], names[len(names) // 2 :]]
    return [
        _save(
            root / "shards" / f"model-0000{index + 1}-of-00002.safetensors",
            {name: weights[name] for name in half},
        )
        for index, half in enumerate(halves)
    ]


def test_a_split_tree_costs_only_its_headers_against_the_all_in_one(tmp_path: Path) -> None:
    weights = _weights()
    cas = LocalCAS(tmp_path / "cas")

    single = ingest_file(cas, _all_in_one(tmp_path, weights), manifest_path="model.safetensors")
    assert len(single.chunks) == 1 + len(weights), "header, then one object per tensor"
    after_single = _objects(cas)

    tree = [
        ingest_file(cas, path, manifest_path=path.name) for path in _split_tree(tmp_path, weights)
    ]
    admitted = _objects(cas) - after_single

    assert len(admitted) == 3, "three component headers, and nothing else"
    assert _tensor_digests(tree) == _tensor_digests([single]), (
        "different files, different key names, identical tensor objects"
    )


def test_hf_style_shards_dedup_identically(tmp_path: Path) -> None:
    weights = _weights()
    cas = LocalCAS(tmp_path / "cas")

    single = ingest_file(cas, _all_in_one(tmp_path, weights), manifest_path="model.safetensors")
    after_single = _objects(cas)

    shards = [
        ingest_file(cas, path, manifest_path=path.name) for path in _shards(tmp_path, weights)
    ]
    admitted = _objects(cas) - after_single

    assert len(admitted) == 2, "one header per shard, and nothing else"
    assert _tensor_digests(shards) == _tensor_digests([single])


def test_an_absolute_grid_destroys_the_sharing(tmp_path: Path) -> None:
    """The red arm: the grid is relative to each tensor, and that is load-bearing.

    Cut the same files on a grid measured from the START OF THE FILE instead --
    the wire allows any grid -- and every packing lands its tensor bytes at a
    different absolute offset, so the sharing collapses completely.
    """

    weights = _weights()
    cas = LocalCAS(tmp_path / "cas")
    grid = 4096

    def commit(path: Path) -> FileEntry:
        size = path.stat().st_size
        assert size > grid, "the fixture must span several absolute slices"
        full, remainder = divmod(size, grid)
        lengths = [grid] * full + ([remainder] if remainder else [])
        return ingest_with_grid(cas, path, lengths, manifest_path=path.name)

    single = commit(_all_in_one(tmp_path, weights))
    tree = [commit(path) for path in _split_tree(tmp_path, weights)]

    relative = ingest_file(cas, _all_in_one(tmp_path, weights), manifest_path="relative")
    assert _tensor_digests([single]) & _tensor_digests(tree) == set(), (
        "an absolute grid shares nothing between two packings"
    )
    assert len(_tensor_digests([relative])) == len(weights), (
        "while the real planner still gives one object per tensor"
    )


def test_a_dtype_cast_and_a_transpose_are_the_honest_breakers(tmp_path: Path) -> None:
    """Casts are math and transposes scatter every run; both correctly share 0."""

    weights = _weights()
    cas = LocalCAS(tmp_path / "cas")
    single = ingest_file(cas, _all_in_one(tmp_path, weights), manifest_path="model.safetensors")

    cast = ingest_file(
        cas,
        _save(
            tmp_path / "fp16.safetensors",
            {name: body.astype(numpy.float16) for name, body in weights.items()},
        ),
        manifest_path="fp16.safetensors",
    )
    assert _tensor_digests([cast]) & _tensor_digests([single]) == set(), (
        "a cast changes every byte, and no chunking scheme recovers that"
    )

    victim = next(iter(weights))
    transposed = dict(weights)
    transposed[victim] = numpy.ascontiguousarray(weights[victim].T)
    moved = ingest_file(
        cas,
        _save(tmp_path / "transposed.safetensors", transposed),
        manifest_path="transposed.safetensors",
    )
    shared = _tensor_digests([moved]) & _tensor_digests([single])
    assert len(shared) == len(weights) - 1, (
        "exactly the untouched tensors are shared; the permuted one is not"
    )
