"""#81: a fused checkpoint and its split twin share every byte, through the
real safetensors library.

The Rust suite proves the planner arithmetic (`crates/tensorfs-core/tests/
contract_seams.rs`). What is proven here is that the arithmetic lands on files
the real writer produces, and that the fusion under test is the one production
performs: MiniMax-H3's `fuse_qkv_head_interleaved` stacks q, k and v inside
each head, so the fused tensor reads `q0 k0 v0 q1 k1 v1 ...`. Fusing with a
naive `cat` instead is ~90% error that never crashes, which is why the seam
table has to express the interleave rather than assume three stacked blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import safetensors.numpy
from numpy.typing import NDArray
from tensorfs.native import ContractRegistry, ObjectStore, adopt, plan_and_hash_file

Array = NDArray[numpy.float32]

HEADS = 2
HEAD_ROWS = 256
COLUMNS = 1024
#: One head slice is exactly the 1 MiB seam-run floor: 256 x 1024 x f32.
RUN_BYTES = HEAD_ROWS * COLUMNS * 4

FUSED_CONTRACT = json.dumps(
    {
        "format": "tensorfs-contract-v1",
        "name": "test.h3-fused",
        "version": 1,
        "tensors": [
            {
                "role": "blocks.{i}.attn.qkv",
                "pattern": "blocks.{i}.attn.qkv_proj.weight",
                "rank": 2,
                "fusion": {
                    "axis": 0,
                    "groups": HEADS,
                    "parts": [
                        {"role": "q", "share": 1},
                        {"role": "k", "share": 1},
                        {"role": "v", "share": 1},
                    ],
                },
            }
        ],
    }
)

SPLIT_CONTRACT = json.dumps(
    {
        "format": "tensorfs-contract-v1",
        "name": "test.h3-split",
        "version": 1,
        "tensors": [
            {
                "role": f"blocks.{{i}}.attn.qkv#{part}",
                "pattern": f"blocks.{{i}}.attn.to_{part}.weight",
                "rank": 2,
                "fusion": {
                    "axis": 0,
                    "groups": HEADS,
                    "parts": [{"role": "", "share": 1}],
                },
            }
            for part in ("q", "k", "v")
        ],
    }
)


def _projections() -> dict[str, Array]:
    generator = numpy.random.default_rng(81)
    return {
        part: generator.random((HEADS * HEAD_ROWS, COLUMNS), dtype=numpy.float32)
        for part in ("q", "k", "v")
    }


def _fuse_head_interleaved(parts: dict[str, Array]) -> Array:
    """`h3_native_layout.fuse_qkv_head_interleaved`, transcribed to numpy."""
    reshaped = [
        parts[part].reshape(HEADS, HEAD_ROWS, COLUMNS) for part in ("q", "k", "v")
    ]
    return numpy.stack(reshaped, axis=1).reshape(3 * HEADS * HEAD_ROWS, COLUMNS)


def _write_pair(directory: Path) -> tuple[Path, Path]:
    parts = _projections()
    fused = directory / "native.safetensors"
    split = directory / "diffusers.safetensors"
    safetensors.numpy.save_file(
        {"blocks.0.attn.qkv_proj.weight": _fuse_head_interleaved(parts)}, fused
    )
    safetensors.numpy.save_file(
        {f"blocks.0.attn.to_{part}.weight": array for part, array in parts.items()},
        split,
    )
    return fused, split


def _tensor_digests(path: Path, registry: ContractRegistry | None) -> set[str]:
    plan = plan_and_hash_file(path, registry)
    return {item.digest for item in plan.objects if item.kind == "tensor"}


def test_the_fused_file_and_its_split_twin_share_every_tensor_object(
    tmp_path: Path,
) -> None:
    fused, split = _write_pair(tmp_path)
    registry = ContractRegistry([FUSED_CONTRACT, SPLIT_CONTRACT])

    # Identification is header-only: no tensor byte is read to decide this.
    assert registry.detect_file(fused) == "test.h3-fused@1"
    assert registry.detect_file(split) == "test.h3-split@1"

    with_contracts = _tensor_digests(fused, registry)
    assert with_contracts == _tensor_digests(split, registry)
    assert len(with_contracts) == 3 * HEADS, "one object per head slice"

    # The red proof: the same files, no contracts. The fused tensor becomes one
    # object the split file does not hold, and the sharing is gone.
    plain_fused = _tensor_digests(fused, None)
    plain_split = _tensor_digests(split, None)
    assert len(plain_fused) == 1
    assert len(plain_split) == 3
    assert plain_fused & plain_split == set()


def test_the_fixture_fusion_is_the_one_production_performs(tmp_path: Path) -> None:
    # If this fixture fused with a naive `cat`, the sharing test above would
    # still pass -- against the wrong bytes. This is the check that the file
    # under test is the layout H3 actually ships.
    parts = _projections()
    fused = _fuse_head_interleaved(parts)
    naive = numpy.concatenate([parts["q"], parts["k"], parts["v"]], axis=0)
    assert not numpy.array_equal(fused, naive), "the two fusions must differ"

    # Head-major: rows 0..256 are q of head 0, then k, then v.
    numpy.testing.assert_array_equal(fused[:HEAD_ROWS], parts["q"][:HEAD_ROWS])
    numpy.testing.assert_array_equal(
        fused[HEAD_ROWS : 2 * HEAD_ROWS], parts["k"][:HEAD_ROWS]
    )
    numpy.testing.assert_array_equal(
        fused[3 * HEAD_ROWS : 4 * HEAD_ROWS], parts["q"][HEAD_ROWS:]
    )

    written, _ = _write_pair(tmp_path)
    reread = safetensors.numpy.load_file(written)["blocks.0.attn.qkv_proj.weight"]
    numpy.testing.assert_array_equal(reread, fused)


def test_adopting_a_contract_upgrades_a_plain_ingest_in_place(tmp_path: Path) -> None:
    fused, split = _write_pair(tmp_path)
    store = ObjectStore(tmp_path / "store")

    # Ingested with no contract at all: contract:none, plain per-tensor grid.
    plan, admitted = store.admit_file(fused)
    assert plan.contract == "none"
    from tensorfs.native import FileRecord

    records = [FileRecord.data(item.digest, item.length) for item in admitted]

    registry = ContractRegistry([FUSED_CONTRACT, SPLIT_CONTRACT])
    upgraded, stamp = adopt(store, plan.planner, records, registry)
    assert stamp == "test.h3-fused@1"

    # The header object is inherited verbatim; only the fused tensor is recut.
    header = records[0].digest
    assert upgraded[0].digest == header
    assert len(upgraded) == 1 + 3 * HEADS

    # And the upgraded record list now names exactly the split file's objects.
    shared = {record.digest for record in upgraded} & _tensor_digests(split, registry)
    assert len(shared) == 3 * HEADS


def test_the_shipped_library_identifies_and_lists_removable_sets(
    tmp_path: Path,
) -> None:
    # The built-in contracts are data in `spec/v1/contracts/`, and a file that
    # matches none of them is contract:none rather than a guess.
    registry = ContractRegistry.builtin()
    assert "minimax.h3-dit-native@1" in registry.stamps()

    foreign = tmp_path / "foreign.safetensors"
    safetensors.numpy.save_file(
        {"something.else": numpy.zeros((4, 4), dtype=numpy.float32)}, foreign
    )
    assert registry.detect_file(foreign) == "none"

    # A DiT block file resolves its removable AdaLN set by name -- the input a
    # subset snapshot trims (#83).
    dit = tmp_path / "dit.safetensors"
    safetensors.numpy.save_file(
        {
            "blocks.0.attn.qkv.weight": numpy.zeros((6, 4), dtype=numpy.float32),
            "blocks.0.adaLN_modulation.1.weight": numpy.zeros(
                (4, 4), dtype=numpy.float32
            ),
            "blocks.0.adaLN_modulation.1.bias": numpy.zeros((4,), dtype=numpy.float32),
        },
        dit,
    )
    assert registry.detect_file(dit) == "dit.blocks-fused-qkv@1"
    assert registry.set_members(dit, "adaln_projections") == [
        "blocks.0.adaLN_modulation.1.weight",
        "blocks.0.adaLN_modulation.1.bias",
    ]
