"""#111: contracts are objects you import — the ``tensorfs.contracts`` surface.

The lane surface types ``lanes=(contracts.SDXL_DIFFUSERS_BF16, ...)``: the
constant IS the layout. Customs are anonymous — ``Contract(...)`` has no
``name=`` and its identity is the content digest. Every claim here is proven
against the Rust validator and a real object store, not a re-implementation.
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest
from tensorfs import Contract, Fusion, MissingDtype, TensorDecl, contracts
from tensorfs.native import (
    ContractRegistry,
    FileRecord,
    ObjectStore,
    contract_info,
    derive,
    rekey,
    subset,
)

# The Rust pin test's values (crates/tensorfs-core/tests/contract_seams.rs).
RUST_PINNED = {
    "dit.blocks-fused-qkv@1": (
        "57f09073ca1a631bb173293c84e11f462089d0fa8cf9ce45e142fc33881a17a1"
    ),
    "minimax.h3-dit-diffusers@1": (
        "b1a3e40a7f0a8088d2080b4c8a5dd8960a5ad576cef80d39f717ed0b8d5d6ae7"
    ),
    "minimax.h3-dit-native@1": (
        "074afcc03b5a3d5e6a0c5d720dc5441d335b9def1688f726180906a364c91805"
    ),
    "sdxl.clip-g-fused-qkv@1": (
        "364c0c537e54013eab72994a3e6bf0b913cfb76ab1627dc0822b95cf17b1b262"
    ),
    "sdxl.clip-g-split-qkv@1": (
        "c1bbfc65a89a736154504f68296b2b8be3dc43364d4dc04a192c08a184bf64fa"
    ),
    "sdxl.diffusers-bf16@1": (
        "f1455f56321d1f268772912c223170f015564ac164064d6d8f77007b03bd35df"
    ),
}


# ---------------------------------------------------------------------------
# the packaged library
# ---------------------------------------------------------------------------


def test_the_three_surfaces_hold_the_same_library() -> None:
    """Constants == packaged spec directory == Rust BUILTIN, mechanically."""

    spec = Path(__file__).resolve().parents[2] / "spec" / "v1" / "contracts"
    spec_stamps = set()
    for document in sorted(spec.glob("*.v1.json")):
        raw = json.loads(document.read_text(encoding="utf-8"))
        spec_stamps.add(f"{raw['name']}@{raw['version']}")

    python_stamps = {contract.stamp for contract in contracts.all()}
    rust_stamps = set(ContractRegistry.builtin().stamps())
    assert python_stamps == spec_stamps == rust_stamps

    # And every constant name derives mechanically and resolves.
    for contract in contracts.all():
        assert contract.name is not None
        constant = contract.name.replace(".", "_").replace("-", "_").upper()
        assert getattr(contracts, constant).name == contract.name


def test_the_constants_carry_the_rust_pinned_digests() -> None:
    """The Python object's digest IS the Rust ``Contract::digest``."""

    for stamp, digest in RUST_PINNED.items():
        assert contracts.get(stamp).digest == digest
    assert contracts.SDXL_CLIP_G_FUSED_QKV.digest == RUST_PINNED["sdxl.clip-g-fused-qkv@1"]
    assert contracts.SDXL_DIFFUSERS_BF16.stamp == "sdxl.diffusers-bf16@1"


def test_get_pins_a_version_and_refuses_the_unknown() -> None:
    pinned = contracts.get("sdxl.clip-g-split-qkv@1")
    newest = contracts.get("sdxl.clip-g-split-qkv")
    assert pinned == newest  # one version so far; equality is by digest
    with pytest.raises(KeyError, match="sdxl.clip-g-split-qkv@1"):
        contracts.get("sdxl.nonexistent")
    with pytest.raises(AttributeError, match="SDXL_CLIP_G_SPLIT_QKV"):
        _ = contracts.NO_SUCH_CONSTANT


def test_the_lane_constant_reads_its_load_dtype() -> None:
    """The ``ctx.lane.dtype``-shaped read."""

    assert contracts.SDXL_DIFFUSERS_BF16.dtype == "bfloat16"
    # Undeclared is an author error, refused loudly rather than guessed.
    with pytest.raises(MissingDtype, match="sdxl.clip-g-fused-qkv"):
        _ = contracts.SDXL_CLIP_G_FUSED_QKV.dtype


def test_torch_dtype_resolves_lazily_against_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    fake = types.ModuleType("torch")
    fake.bfloat16 = sentinel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert contracts.SDXL_DIFFUSERS_BF16.torch_dtype is sentinel
    with pytest.raises(MissingDtype):
        _ = contracts.SDXL_CLIP_G_FUSED_QKV.torch_dtype


def test_the_shared_corpus_agrees_with_the_python_surface() -> None:
    """The tensorfs#114 conformance corpus, from the third language: every
    golden document parses here with the Rust-pinned digest and stamp — the
    same file Go's contract_test.go runs."""

    corpus_path = (
        Path(__file__).resolve().parents[2]
        / "spec"
        / "v1"
        / "contract-vectors"
        / "contract-vectors.json"
    )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    nameless = 0
    for case in corpus["golden"]:
        if "file" in case:
            document = (corpus_path.parents[1] / case["file"]).read_text(encoding="utf-8")
        else:
            document = case["document"]
        parsed = Contract.from_document(document)
        assert parsed.digest == case["digest"], case["name"]
        assert parsed.stamp == case["stamp"], case["name"]
        nameless += parsed.name is None
    assert nameless >= 1, "the corpus holds a custom the constructor shape produces"
    for case in corpus["refusals"]:
        if case["reason"] == "json":
            continue  # not JSON at all, or shapes json.loads accepts differently
        with pytest.raises(ValueError):
            Contract.from_document(case["document"])


# ---------------------------------------------------------------------------
# anonymous construction
# ---------------------------------------------------------------------------


def _custom_fused() -> Contract:
    return Contract(
        tensors=[
            TensorDecl(
                role="blocks.{i}.attn.qkv",
                pattern="blocks.{i}.attn.qkv_proj.weight",
                rank=2,
                fusion=Fusion(parts=[("q", 1), ("k", 1), ("v", 1)]),
            ),
        ],
    )


def _custom_split() -> Contract:
    return Contract(
        tensors=[
            TensorDecl(
                role="blocks.{i}.attn.qkv#q",
                pattern="blocks.{i}.attn.to_q.weight",
                rank=2,
            ),
            TensorDecl(
                role="blocks.{i}.attn.qkv#k",
                pattern="blocks.{i}.attn.to_k.weight",
                rank=2,
            ),
            TensorDecl(
                role="blocks.{i}.attn.qkv#v",
                pattern="blocks.{i}.attn.to_v.weight",
                rank=2,
            ),
        ],
    )


def test_a_custom_contract_is_anonymous_and_digest_identified() -> None:
    custom = _custom_fused()
    assert custom.name is None
    assert custom.version is None

    # The emitted document is NAMELESS: the wire format never forces a fake
    # name back into the author surface.
    raw = json.loads(custom.document)
    assert "name" not in raw
    assert "version" not in raw

    # Round-trip: build -> document -> Rust parse -> identical digest.
    info = contract_info(custom.document)
    assert info.digest == custom.digest
    assert info.stamp == custom.stamp == f"sha256:{custom.digest}"
    assert custom.label == f"sha256:{custom.digest[:8]}…"

    # Equality is by digest, whatever the JSON spelling.
    respelled = Contract.from_document(json.dumps(raw, indent=3))
    assert respelled == custom
    assert hash(respelled) == hash(custom)
    assert custom != _custom_split()


def test_the_constructor_has_no_name_kwarg() -> None:
    with pytest.raises(TypeError):
        Contract(  # type: ignore[call-arg]
            name="my.contract",
            tensors=[TensorDecl(role="a", pattern="a")],
        )


def test_a_malformed_custom_refuses_with_the_rust_message() -> None:
    # Adjacent holes are ambiguous — the refusal is the Rust parser's, at
    # construction (= author-module import time), never at deploy.
    with pytest.raises(ValueError, match="pattern"):
        Contract(tensors=[TensorDecl(role="a.{i}{i}", pattern="a.{i}{i}")])
    with pytest.raises(ValueError, match="dtype"):
        Contract(dtype="BF16", tensors=[TensorDecl(role="a", pattern="a")])
    with pytest.raises(ValueError, match="tensors"):
        Contract(tensors=[])


def test_a_library_document_round_trips_through_from_document() -> None:
    fused = contracts.SDXL_CLIP_G_FUSED_QKV
    again = Contract.from_document(fused.document)
    assert again == fused
    assert again.stamp == "sdxl.clip-g-fused-qkv@1"
    assert again.label == "sdxl.clip-g-fused-qkv"
    declarations = again.tensors
    assert declarations[0].fusion is not None
    assert [part for part, _ in declarations[0].fusion.parts] == ["q", "k", "v"]


# ---------------------------------------------------------------------------
# Contract objects end-to-end: real store, real compositions
# ---------------------------------------------------------------------------

MIB = 1024 * 1024


def _fused_file(path: Path) -> Path:
    """One fused qkv tensor, three 1 MiB parts — above the seam floor."""

    rows, columns = 3, MIB
    body = bytes(
        (index * 31 + part * 7) % 256 for part in range(rows) for index in range(columns)
    )
    header = json.dumps(
        {
            "blocks.0.attn.qkv_proj.weight": {
                "dtype": "U8",
                "shape": [rows, columns],
                "data_offsets": [0, rows * columns],
            }
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + body)
    return path


def test_compositions_accept_contract_objects_end_to_end(tmp_path: Path) -> None:
    fused, split = _custom_fused(), _custom_split()
    store = ObjectStore(tmp_path / "store")

    # Ingest under a registry built from the Contract OBJECT: the plan is
    # stamped with the custom's digest, not a name.
    registry = ContractRegistry([fused])
    assert registry.stamps() == [fused.stamp]
    plan, admitted = store.admit_file(_fused_file(tmp_path / "fused.safetensors"), registry)
    assert plan.contract == fused.stamp
    records = [FileRecord.data(item.digest, item.length) for item in admitted]
    data_before = {record.digest for record in records if record.digest is not None}

    # derive takes the two Contract objects; run-preserving = zero new data.
    derived, stamp = derive(store, plan.planner, records, fused, split)
    assert stamp == split.stamp
    assert stamp.startswith("sha256:")
    data_after = {record.digest for record in derived if record.digest is not None}
    assert len(data_after - data_before) == 1, "one rewritten header, zero data objects"

    # rekey and subset take the Contract where they took a stamp string.
    rekeyed = rekey(
        store,
        plan.planner,
        records,
        {"blocks.0.attn.qkv_proj.weight": "renamed.qkv_proj.weight"},
        contract=fused,
    )
    assert {record.digest for record in rekeyed if record.digest is not None} - data_before, (
        "the rewritten header is new"
    )
    trimmed = subset(
        store,
        plan.planner,
        records,
        {"blocks.0.attn.qkv_proj.weight": "blocks.0.attn.qkv_proj.weight"},
        contract=fused,
    )
    assert len(trimmed) == len(records)
