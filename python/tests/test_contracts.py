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
    "flux1.diffusers-bf16@1": (
        "392e9b54605ef75e18e7b7a29da3501f59215c091c228a16d18d403463320a63"
    ),
    "flux2-klein.diffusers-bf16@1": (
        "4d9fada187e6e086a1c4c496d81b785ce9419fc089e64d0141f6427b88e9d40d"
    ),
    "hidream-o1.diffusers-bf16@1": (
        "69003d11e2cb3b52628cb05e02c275916db74e318eabbce2f6e89be625c7e01f"
    ),
    "internvl-u.diffusers-bf16@1": (
        "9ee028aea999f92c4d93609f59ba476f0c6aad8dee3269cee1a22abec97d199d"
    ),
    "ltx-2.diffusers-bf16@1": (
        "71038ae11883111d367077eec59a457b97c8746439c3b7bc0885555e26b7aa12"
    ),
    "minimax.h3-dit-diffusers@1": (
        "22bbb607d3e4351c18ac55e8d86c8a8a3d03296309b80e4419d4bc5153481f28"
    ),
    "minimax.h3-dit-native@1": (
        "5f6b7b4a8cd070653607840b922c213a846838e35379737727111c4b0a8de56c"
    ),
    "sd15.diffusers-bf16@1": (
        "0bc98e52edac1a4b3a8f063162d3785350413b60701ba49d9701e46d69f304d3"
    ),
    "sd2.diffusers-bf16@1": (
        "136b158fb5f96cd05f2a8b3accc0b3d36ea7b07b988a338ee4f7934d54a312e6"
    ),
    "minimax.h3-dit-fp8-rowwise@1": (
        "69a2cc8f338ba925d4415f67719f1ed1643e9d31f43e4af04d9b2ff1dc035d1f"
    ),
    "sdxl.clip-g-fused-qkv@1": (
        "364c0c537e54013eab72994a3e6bf0b913cfb76ab1627dc0822b95cf17b1b262"
    ),
    "sdxl.clip-g-split-qkv@1": (
        "c1bbfc65a89a736154504f68296b2b8be3dc43364d4dc04a192c08a184bf64fa"
    ),
    "sdxl.diffusers-bf16@1": (
        "ef01dd65f57bd95ae05d70f5a9893e9abab6b4f0831b05c4edf68ae9ebb148e8"
    ),
    "sdxl.diffusers-fp8-rowwise@1": (
        "7b78f2e44382dc5a3fe413e0f8f0a62ba63efefc810123304151c2ded931ee37"
    ),
    "stable-audio.diffusers-fp16@1": (
        "c580a73fda845048b99b3d3f2093be9c282da4166e1e5521fea139957b454ed3"
    ),
    "trellis2.dit-bf16@1": (
        "f9763a5aa4b82d552c7b582d7b540cd0fdf576cc5bd234bb9e73ec617738ab52"
    ),
    "wan22.diffusers-bf16@1": (
        "91036beea9878c462311d97a878a5dc94128283182fbf0948b30118852d97bd5"
    ),
    "qwen-image.diffusers-bf16@1": (
        "757379261ff69111f5e50a5cf2a066c6a118759f1dfe1f34b605f8a49325e673"
    ),
    "z-image.diffusers-bf16@1": (
        "f726fae567c094783ac5aa41822b77f4ae3387bfc07daf22f3ca189ed74c1af5"
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


# ---------------------------------------------------------------------------
# tensorfs#124: flux1 is not flux2-klein
# ---------------------------------------------------------------------------


def _flux1_transformer_names(
    *, layers: int = 19, single_layers: int = 38, guidance_embeds: bool = True
) -> dict[str, int]:
    """The diffusers ``FluxTransformer2DModel`` state dict, name -> rank.

    The spellings are the real ones, read off the safetensors headers of the
    two checkpoints the fleet serves (tensorhub/flux1-dev and
    tensorhub/flux1-schnell, revision-pinned clones of the gated BFL repos).
    ``layers``/``single_layers``/``guidance_embeds`` are the three axes
    ``transformer/config.json`` actually varies across the served arms.
    """

    names: dict[str, int] = {
        "x_embedder.weight": 2,
        "x_embedder.bias": 1,
        "context_embedder.weight": 2,
        "context_embedder.bias": 1,
        "proj_out.weight": 2,
        "proj_out.bias": 1,
        "norm_out.linear.weight": 2,
        "norm_out.linear.bias": 1,
    }
    embedders = ["timestep_embedder", "text_embedder"]
    if guidance_embeds:
        embedders.append("guidance_embedder")
    for embedder in embedders:
        for index in (1, 2):
            stem = f"time_text_embed.{embedder}.linear_{index}"
            names[f"{stem}.weight"] = 2
            names[f"{stem}.bias"] = 1
    for block in range(layers):
        stem = f"transformer_blocks.{block}"
        for leaf in (
            "attn.to_q", "attn.to_k", "attn.to_v",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_out.0", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            "norm1.linear", "norm1_context.linear",
        ):
            names[f"{stem}.{leaf}.weight"] = 2
            names[f"{stem}.{leaf}.bias"] = 1
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            names[f"{stem}.attn.{norm}.weight"] = 1
    for block in range(single_layers):
        stem = f"single_transformer_blocks.{block}"
        for leaf in ("attn.to_q", "attn.to_k", "attn.to_v",
                     "proj_mlp", "proj_out", "norm.linear"):
            names[f"{stem}.{leaf}.weight"] = 2
            names[f"{stem}.{leaf}.bias"] = 1
        for norm in ("norm_q", "norm_k"):
            names[f"{stem}.attn.{norm}.weight"] = 1
    return names


#: Outermost dim for the synthetic headers. NOT 2, and that is load-bearing:
#: ``Contract.matches`` refuses outright when a declared ``fusion`` cannot cut
#: the tensor it names, so an outer dim that is not divisible by the parts sum
#: makes a fusion-bearing document — flux2-klein's 1:1:1:6, H3's 56 groups —
#: refuse a file it should match, and the refusal looks exactly like the
#: no-match this test is trying to observe. MEASURED: at 2, a real FLUX.2 Klein
#: tree read as ``flux1.diffusers-bf16@1``, which is a false alarm about THIS
#: document rather than a fact about it. 5040 = 2^4 x 3^2 x 5 x 7 clears every
#: parts sum and group count in the shipped library.
SYNTHETIC_OUTER_DIM = 5040


def _bf16_file(path: Path, names: dict[str, int], dtype: str = "BF16") -> Path:
    """A real safetensors file carrying those names at those RANKS.

    Inner dims are 2: these declarations constrain rank and dtype and never
    shape, which is what lets ONE document span a 12B BFL checkpoint and an
    8-block derivative whose ``x_embedder`` is 196 channels wide instead of 64.

    ``dtype`` is parameterised (tensorfs#139) because the audio family ships
    F16 while every other shipped document is BF16, and because the SEPARATION
    between some documents currently rests on element width rather than on
    architecture — a fact worth being able to write a test about. Both BF16 and
    F16 are two bytes, so the span arithmetic is unchanged.
    """

    header: dict[str, object] = {}
    offset = 0
    for name, rank in names.items():
        dims = [SYNTHETIC_OUTER_DIM] + [2] * (rank - 1) if rank else []
        span = 2  # two bytes per BF16/F16 element
        for dim in dims:
            span *= dim
        header[name] = {
            "dtype": dtype,
            "shape": dims,
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return path


def _flux2_klein_transformer_names() -> dict[str, int]:
    """The FLUX.2 Klein 4B transformer state dict, name -> rank.

    Read off the real header of ``tensorhub/flux2-klein-4b``. Here so the
    no-tie proof runs in BOTH directions: a document that wins its own family
    by stealing the sibling family's files has not been proven, it has been
    half-measured.
    """

    names: dict[str, int] = {
        "x_embedder.weight": 2,
        "context_embedder.weight": 2,
        "proj_out.weight": 2,
        "norm_out.linear.weight": 2,
        "double_stream_modulation_img.linear.weight": 2,
        "double_stream_modulation_txt.linear.weight": 2,
        "single_stream_modulation.linear.weight": 2,
        "time_guidance_embed.timestep_embedder.linear_1.weight": 2,
        "time_guidance_embed.timestep_embedder.linear_2.weight": 2,
    }
    for block in range(5):
        stem = f"transformer_blocks.{block}"
        for leaf in (
            "attn.to_q", "attn.to_k", "attn.to_v",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_out.0", "attn.to_add_out",
            "ff.linear_in", "ff.linear_out",
            "ff_context.linear_in", "ff_context.linear_out",
        ):
            names[f"{stem}.{leaf}.weight"] = 2
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            names[f"{stem}.attn.{norm}.weight"] = 1
    for block in range(20):
        stem = f"single_transformer_blocks.{block}"
        names[f"{stem}.attn.to_qkv_mlp_proj.weight"] = 2
        names[f"{stem}.attn.to_out.weight"] = 2
        names[f"{stem}.attn.norm_q.weight"] = 1
        names[f"{stem}.attn.norm_k.weight"] = 1
    return names


def test_flux1_does_not_steal_the_sibling_familys_files(tmp_path: Path) -> None:
    """The other direction of tensorfs#124's no-tie proof.

    FLUX.1 and FLUX.2 Klein are the closest two MMDiT spellings in the library
    and they share 104 of Klein's 169 tensor NAMES, so the predicate that
    separates them has to be the name SET and not a digest. Klein explains 169
    of its own 169 where flux1 explains 104, so Klein keeps its tree; flux1
    explains 1160 of a FLUX.1 tree where Klein explains 308, so flux1 takes
    that one. Neither count ever ties.
    """

    names = _flux2_klein_transformer_names()
    assert len(names) == 169, "the synthetic header drifted from the real one"
    path = _bf16_file(tmp_path / "flux2-klein-4b.safetensors", names)
    assert (
        ContractRegistry(list(contracts.all())).detect_file(path)
        == "flux2-klein.diffusers-bf16@1"
    ), "flux1 must not win its sibling family's checkpoint"


#: arm -> (config axes, the MEASURED transformer tensor count). The count is
#: what ties this synthetic header to the real ones: it is what the shard
#: headers actually add up to, so a drift in the spellings above stops being
#: invisible.
FLUX1_ARMS = {
    # FLUX.1-dev: guidance-distilled.
    "dev": (dict(layers=19, single_layers=38, guidance_embeds=True), 1160),
    # FLUX.1-schnell: no guidance embedder — the four-tensor delta.
    "schnell": (dict(layers=19, single_layers=38, guidance_embeds=False), 1156),
    # ostris/Flex.2-preview: a FLUX.1-architecture redistill the flux.1-schnell
    # endpoint serves, 8 double blocks.
    "flex2": (dict(layers=8, single_layers=38, guidance_embeds=True), 808),
}


@pytest.mark.parametrize("arm", sorted(FLUX1_ARMS))
def test_a_flux1_tree_is_flux1_and_was_flux2_klein_without_it(
    arm: str, tmp_path: Path
) -> None:
    """tensorfs#124: the near miss this document exists to end.

    ``dit.blocks-fused-qkv@1`` fails LOUDLY on a flux tree — it is the timm
    ``blocks.{i}`` spelling and explains nothing. ``flux2-klein.diffusers-bf16@1``
    is the dangerous one: FLUX.1 and FLUX.2 Klein wear the same diffusers
    vocabulary over different architectures (19+38 blocks against 5+20, split
    to_q/to_k/to_v against a fused to_qkv_mlp_proj, an ungated 1:1:1:4 single
    stream against a gated 1:1:1:6), so Klein matches a FLUX.1 file with no
    dtype or rank refusal and WON it outright before this document existed.

    The without-flux1 arm is the point: it names the baseline, so this assertion
    is known to be able to fail rather than merely never having failed.

    THE BASELINE ASSERTS THE PROPERTY, NOT THE RUNNER-UP'S NAME. It used to pin
    ``flux2-klein.diffusers-bf16@1`` as the family that wins without this
    document, and that went stale the moment another lane added a document that
    explains a FLUX.1 tree better: qwen-image landed and the runner-up became
    ``qwen-image.diffusers-bf16@1``, turning master red for every lane through no
    fault of either document. Neither was wrong — a hardcoded second place is
    simply not a property of this family, it is a property of the registry's
    current contents, and it would go stale again on the next document anyone
    adds. What this test actually cares about survives the change: SOMETHING
    still matches (so the arm is live rather than vacuously passing on a refusal)
    and what it matches is the WRONG FAMILY (so flux1's document is doing real
    work). Recorded for history: the runner-up was flux2-klein at tensorfs#124
    and qwen-image at tensorfs#131.
    """

    axes, measured = FLUX1_ARMS[arm]
    names = _flux1_transformer_names(**axes)  # type: ignore[arg-type]
    assert len(names) == measured, "the synthetic header drifted from the real one"
    path = _bf16_file(tmp_path / f"flux1-{arm}.safetensors", names)

    library = contracts.all()
    assert ContractRegistry(list(library)).detect_file(path) == "flux1.diffusers-bf16@1"

    without_flux1 = [item for item in library if item.name != "flux1.diffusers-bf16"]
    assert len(without_flux1) == len(library) - 1
    runner_up = ContractRegistry(without_flux1).detect_file(path)
    # NOT `assert runner_up`: a no-match renders as the STRING "none", which is
    # truthy, so the obvious spelling of this guard can never fire. Absent must
    # never render as present.
    assert runner_up != "none", (
        "the baseline arm went vacuous: with flux1 removed NOTHING matches a "
        "flux1 tree, so this test no longer proves the document closes anything"
    )
    assert runner_up != "flux1.diffusers-bf16@1", "flux1 was supposed to be removed"
    assert not runner_up.startswith("flux1."), (
        f"the wrong-family win this document closes; got {runner_up}"
    )


# ── stable-audio (tensorfs#139, se#769 audio lane) ───────────────────────────


def _stable_audio_transformer_names(blocks: int = 24) -> dict[str, int]:
    """The StableAudioDiTModel state dict, name -> rank.

    Read off the REAL header of ``tensorhub/stable-audio-open``'s transformer
    (445 tensors, ranged read, http=206) and cross-checked byte-for-byte in
    names/shapes/dtypes against ``tensorhub/foundation-1``. The inner indices
    are the measured ones, not guesses: ``timestep_proj``/``global_proj``/
    ``cross_attention_proj`` are MLPs whose Linear layers sit at 0 and 2 with a
    non-parametric activation at 1, ``to_out`` has only index 0, and ``ff.net``
    is GEGLU at 0 + Linear at 2.
    """

    names: dict[str, int] = {
        # The family-DISTINCTIVE anchors: FLUX has no counterpart for any of
        # these, which is what carries specificity given the shared
        # `transformer_blocks.*` spelling.
        "preprocess_conv.weight": 3,
        "postprocess_conv.weight": 3,
        "proj_in.weight": 2,
        "proj_out.weight": 2,
        "time_proj.weight": 1,
    }
    for i in (0, 2):
        names[f"timestep_proj.{i}.weight"] = 2
        names[f"timestep_proj.{i}.bias"] = 1
        names[f"global_proj.{i}.weight"] = 2
        names[f"cross_attention_proj.{i}.weight"] = 2
    for b in range(blocks):
        stem = f"transformer_blocks.{b}"
        for norm in ("norm1", "norm2", "norm3"):
            names[f"{stem}.{norm}.weight"] = 1
            names[f"{stem}.{norm}.bias"] = 1
        # attn1 = self-attention, attn2 = cross-attention. SPLIT q/k/v (no
        # fusion is declared anywhere in this family), and FLUX.1's patterns
        # are bare `attn`, so they miss both of these.
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v"):
                names[f"{stem}.{attn}.{proj}.weight"] = 2
            names[f"{stem}.{attn}.to_out.0.weight"] = 2
        names[f"{stem}.ff.net.0.proj.weight"] = 2
        names[f"{stem}.ff.net.0.proj.bias"] = 1
        names[f"{stem}.ff.net.2.weight"] = 2
        names[f"{stem}.ff.net.2.bias"] = 1
    return names


#: The MEASURED transformer tensor count, so a drift in the spellings above
#: stops being invisible.
STABLE_AUDIO_TENSORS = 445


def test_a_stable_audio_tree_is_stable_audio_and_nothing_wins_it_without_the_document(
    tmp_path: Path,
) -> None:
    """tensorfs#139: the audio DiT wins its own tree, and the baseline is NAMED.

    The near miss this document closes is real and was measured from BOTH
    sides: 432 of StableAudio's 445 tensors are spelled ``transformer_blocks.*``
    — the same spelling FLUX.1 uses — and the flux1 lane measured
    ``flux1.diffusers-bf16@1`` matching 97 of them (the diffusers ``FeedForward``
    leaf spelling ``ff.net.0.proj.*``/``ff.net.2.*`` plus ``proj_out.weight``,
    shared verbatim). Specificity here comes from the anchors FLUX has no
    counterpart for, plus the ``attn1``/``attn2`` split that FLUX's bare
    ``attn`` patterns miss.

    The without-the-document arm NAMES the baseline so this assertion is known
    to be able to fail. It deliberately does NOT hardcode WHICH other family
    wins: that is exactly the assertion that went red fleet-wide when the
    registry legitimately grew and the runner-up changed. The durable property
    is "removing this document changes the answer, and nothing else claims the
    tree as stable-audio", which no amount of registry growth can invalidate.
    """

    names = _stable_audio_transformer_names()
    assert len(names) == STABLE_AUDIO_TENSORS, (
        "the synthetic header drifted from the real one"
    )
    path = _bf16_file(tmp_path / "stable-audio.safetensors", names, dtype="F16")

    library = list(contracts.all())
    assert ContractRegistry(library).detect_file(path) == "stable-audio.diffusers-fp16@1"

    without = [c for c in library if c.name != "stable-audio.diffusers-fp16"]
    assert len(without) == len(library) - 1
    runner_up = ContractRegistry(without).detect_file(path)
    # The baseline is MEASURED and stated exactly, not merely asserted to
    # differ. `!= "stable-audio…"` would pass VACUOUSLY here: a no-match
    # renders as the string "none", which is not the stamp and is also truthy,
    # so the obvious spelling of this guard could never fire (the same trap
    # #138 fixed in the flux1 arm above).
    #
    # "none" IS the honest answer for this family, and for a reason worth
    # recording: every OTHER shipped document declares BF16, so they all refuse
    # an F16 file outright on dtype rather than scoring against it. This
    # document therefore closes a gap where nothing matched, not a wrong-family
    # win — which is a WEAKER guarantee than flux1's, and the bf16 test below
    # is where that weakness is pinned.
    assert runner_up == "none", (
        "the baseline moved: something now claims an F16 StableAudio tree "
        f"without this document ({runner_up}). That is a real cross-family "
        "capture and wants investigating, not a test update."
    )


def test_the_audio_family_does_not_claim_its_siblings_or_get_claimed(
    tmp_path: Path,
) -> None:
    """The cross-family arms, both directions, at the TENSOR level.

    tensorfs#122's tie is what this guards. Two measured facts drive it:
    the AutoencoderOobleck and the T5 text encoder are BYTE-IDENTICAL across
    stable-audio-open and foundation-1, and StableAudio's T5 is NAME-identical
    with musicgen's (Jaccard 1.0, all 99). Neither is in this document, which
    is why the tie cannot happen — but "is not declared" is only a claim until
    a test reads the shipped document and says so.
    """

    doc = next(c for c in contracts.all() if c.name == "stable-audio.diffusers-fp16")
    patterns = {t.pattern for t in doc.tensors}

    # Transformer-only: no autoencoder, no text encoder, no tokenizer.
    for forbidden in ("vae", "decoder.", "encoder.block", "audio_encoder", "shared."):
        assert not any(forbidden in p for p in patterns), (
            f"{forbidden!r} leaked into a transformer-only document"
        )

    # A FLUX.1 tree must not be claimed by the audio document.
    flux_names = _flux1_transformer_names(
        layers=19, single_layers=38, guidance_embeds=True
    )
    flux_path = _bf16_file(tmp_path / "flux1.safetensors", flux_names)
    assert ContractRegistry(list(contracts.all())).detect_file(flux_path) != (
        "stable-audio.diffusers-fp16@1"
    ), "the audio document must not claim a FLUX tree"


def test_a_bf16_stable_audio_tree_is_not_claimed_by_the_fp16_document(
    tmp_path: Path,
) -> None:
    """🔴 The KNOWN GAP, pinned deliberately rather than papered over.

    The served checkpoints histogram ``{F16: 445}`` and the hub records
    ``dtype=fp16``, so this document declares F16 and NOTHING ELSE. A bf16
    REPACK of the same architecture is therefore NOT claimed by it — the dtype
    refusal is correct behaviour, not a bug, and widening ``dtypes`` to admit
    BF16 would be inventing a packaging the fleet does not ship.

    This matters because the flux1 lane measured that the separation between
    ``flux1.diffusers-bf16@1`` and a StableAudio tree currently rests on
    ELEMENT WIDTH: their 66 declarations are all BF16, so they refuse this F16
    file outright rather than scoring 97 against it. If a bf16 StableAudio
    repack ever ships it needs its OWN document
    (``stable-audio.diffusers-bf16@1``), exactly as sdxl carries both a
    ``diffusers-bf16`` and a ``diffusers-fp8-rowwise`` lane. This test exists
    so that day is a loud failure here rather than a silent mis-serve.
    """

    names = _stable_audio_transformer_names()
    bf16_path = _bf16_file(tmp_path / "stable-audio-bf16.safetensors", names)
    claimed_by = ContractRegistry(list(contracts.all())).detect_file(bf16_path)

    assert claimed_by != "stable-audio.diffusers-fp16@1", (
        "an fp16 document must not claim a bf16 tree"
    )

    # 🔴 AND THE HAZARD IS LIVE, not hypothetical. MEASURED: a bf16 twin of this
    # exact tree is claimed by ANOTHER family today — the flux1 lane predicted
    # `flux1`, and the actual winner turned out to be a different family again,
    # which is precisely why this is asserted as a PROPERTY rather than pinned
    # to a name (a hardcoded winner is hostage to registry growth — the #136
    # fleet-red).
    #
    # So a bf16 repack of Stable Audio would TODAY be silently mis-served as
    # some other architecture. The endpoints still declare `plain.bf16@1`
    # against fp16 bytes, so that repack is a plausible future, not a fantasy.
    # The fix when it lands is a `stable-audio.diffusers-bf16@1` document, the
    # way sdxl carries both a bf16 and an fp8-rowwise lane — never widening
    # this document's `dtypes` to admit a packaging the fleet does not ship.
    assert claimed_by != "none", (
        "a bf16 StableAudio tree is claimed by NOTHING now — good news, and it "
        "means this pin can be relaxed; re-read the comment above before doing so"
    )
