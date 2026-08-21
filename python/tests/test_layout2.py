"""The v2 read surface, against layouts the Go engine actually printed.

The fixtures under `data/` are verbatim `go run ./scripts/compute_layout`
output, committed rather than regenerated: a test that shelled out to `go`
would be testing the toolchain's availability, and one that hand-wrote the
document would be testing this file against itself.

Two pairs, chosen for what they are entitled to prove:

* `ltx2-upsampler.diffusers@2+plain.bf16@1` -- an identity rule. Every key
  keeps its shape, `transformed` is 0, and the whole component carries one
  element type.
* `ernie.diffusers@2+cozy.fp8-rowwise@2` -- a transforming rule. 253 weights
  became F8_E4M3 and grew an F32 `weight_scale` twin, and the quant block
  carries the dequant identity and the sm floor a serving loader needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tensorfs.layout2 import (
    ExpectedHeader,
    LayoutTensor,
    identity_arrangement,
    layouts,
    rules,
    topologies,
)

_DATA = Path(__file__).resolve().parent / "data"
_SPEC = Path(__file__).resolve().parents[2] / "spec" / "v2"


def _layout(name: str) -> ExpectedHeader:
    return ExpectedHeader.from_document((_DATA / name).read_text(encoding="utf-8"))


@pytest.fixture()
def plain() -> ExpectedHeader:
    return _layout("layout-ltx2-upsampler-plain-bf16.json")


@pytest.fixture()
def rowwise() -> ExpectedHeader:
    return _layout("layout-ernie-fp8-rowwise.json")


def test_an_identity_layout_reads_back_whole(plain: ExpectedHeader) -> None:
    assert plain.stamp == "ltx2-upsampler.diffusers@2+plain.bf16@1"
    assert plain.topology == "ltx2-upsampler.diffusers@2"
    assert len(plain.topology_digest) == 64
    assert plain.quant.handle == "plain.bf16@1"
    assert plain.quant.family == "plain.bf16"
    assert plain.quant.declared_dtype == "bfloat16"
    assert plain.quant.capability_floor_sm == 80
    assert plain.quant.lossy is False
    assert plain.quant.transformed == 0
    assert plain.quant.inverse == ""
    # tensorfs#153: the lane is the WHOLE published tree — the upsampler ships
    # with its VAE, so the identity layout carries both components.
    assert plain.components.keys() == {"latent_upsampler", "vae"}
    assert plain.roles["latent_upsampler"] == "other"
    assert plain.roles["vae"] == "vae"

    tensors = plain.component("latent_upsampler")
    assert len(tensors) == 72
    assert tensors["initial_conv.weight"] == LayoutTensor(("BF16",), (1024, 128, 3, 3, 3))
    # An identity rule stamps ONE element type across the whole component, so
    # nothing here is at a second dtype.
    assert {entry.dtypes for entry in tensors.values()} == {("BF16",)}


def test_a_transforming_layout_carries_the_scales_and_the_dequant(
    rowwise: ExpectedHeader,
) -> None:
    assert rowwise.quant.handle == "cozy.fp8-rowwise@2"
    assert rowwise.quant.declared_dtype == "float8_e4m3fn"
    assert rowwise.quant.capability_floor_sm == 89
    assert rowwise.quant.lossy is True
    assert rowwise.quant.inverse == "W_bf16[r, c] = weight[r, c] * weight_scale[r]"
    assert rowwise.quant.conventions["scale"] == "per_channel_out"
    assert rowwise.quant.transformed == 253
    assert rowwise.roles["transformer"] == "denoiser"

    tensors = rowwise.tensors("transformer")
    quantized = [key for key, entry in tensors.items() if entry.dtypes == ("F8_E4M3",)]
    scales = [key for key in tensors if key.endswith(".weight_scale")]
    assert len(quantized) == rowwise.quant.transformed
    assert len(scales) == len(quantized)
    for key in quantized:
        scale = tensors[key.removesuffix(".weight") + ".weight_scale"]
        assert scale.dtypes == ("F32",)
        assert scale.shape == (tensors[key].shape[0],), "a per-ROW scale is one per output row"
    # The rule's scope is a real exclusion, not an oversight: a norm inside a
    # transformed block stays at the base dtype.
    assert tensors["layers.0.self_attention.norm_q.weight"].dtypes == ("BF16",)


def test_reference_tolerance_is_visible_as_several_accepted_dtypes() -> None:
    """A plain rule is REFERENCE-TOLERANT: a key the reference packaging itself
    shipped wider is accepted at either type. `accepts` is that membership, and
    `dtypes[0]` is still the one a producer writes."""

    entry = LayoutTensor(("BF16", "F32"), (77,))
    assert entry.accepts("BF16") and entry.accepts("F32")
    assert not entry.accepts("F16")
    assert entry.dtypes[0] == "BF16"


def test_naming_a_component_is_required_when_there_are_several(
    plain: ExpectedHeader, rowwise: ExpectedHeader
) -> None:
    with pytest.raises(KeyError, match="name one of"):
        plain.tensors()
    assert plain.component("latent_upsampler") is not None
    with pytest.raises(KeyError, match="no component 'unet'"):
        rowwise.component("unet")


def test_a_multi_component_layout_refuses_to_guess() -> None:
    """v1's collision: SDXL's two text encoders carry the same key spellings.
    A layout with more than one component will not answer for an unnamed one."""

    document = {
        "stamp": "two.components@1+plain.bf16@1",
        "topology": "two.components@1",
        "topology_digest": "0" * 64,
        "quant": {
            "handle": "plain.bf16@1",
            "declared_dtype": "bfloat16",
            "capability_floor_sm": 80,
            "conventions": {},
            "lossy": False,
            "transformed": 0,
            "digest": "0" * 64,
        },
        "components": [
            {"name": "a", "role": "text_encoder", "tensors": {}},
            {"name": "b", "role": "text_encoder", "tensors": {}},
        ],
    }
    layout = ExpectedHeader.from_document(json.dumps(document))
    with pytest.raises(KeyError, match="has 2 components"):
        layout.tensors()


# -- the vendored corpus, read for identity facts only ----------------------


def test_rules_replace_the_dtype_keyed_sm_table() -> None:
    """pgw#1621: the capability floor is a property of the RULE, and so is
    everything else a loader needs. `cozy.fp8-storage` and `cozy.fp8-rowwise`
    agree on the declared dtype AND on the sm floor and differ in whether a
    scale tensor exists at all -- a table keyed on the dtype spelling cannot
    tell them apart, and loses the floor entirely for a third spelling."""

    corpus = rules(_SPEC)
    assert corpus["cozy.fp8-rowwise@2"].capability_floor_sm == 89
    assert corpus["plain.f32@1"].capability_floor_sm == 0
    assert corpus["bfl.nvfp4-preswizzled@1"].capability_floor_sm == 100

    storage = corpus["cozy.fp8-storage@1"]
    rowwise = corpus["cozy.fp8-rowwise@2"]
    assert storage.declared_dtype == rowwise.declared_dtype == "float8_e4m3fn"
    assert storage.capability_floor_sm == rowwise.capability_floor_sm
    assert storage.conventions["scale"] == "none"
    assert rowwise.conventions["scale"] == "per_channel_out"
    assert storage.inverse != rowwise.inverse, "a cast back is not a multiply"

    # `transformed` is a property of a LAYOUT -- how many keys the rule moved in
    # some topology -- so a rule read on its own has no count to report.
    assert rowwise.transformed == 0


def test_a_rules_identity_matches_the_layout_go_computed(rowwise: ExpectedHeader) -> None:
    off_disk = rules(_SPEC)[rowwise.quant.handle]
    assert off_disk.declared_dtype == rowwise.quant.declared_dtype
    assert off_disk.capability_floor_sm == rowwise.quant.capability_floor_sm
    assert off_disk.conventions == rowwise.quant.conventions
    assert off_disk.lossy == rowwise.quant.lossy
    assert off_disk.inverse == rowwise.quant.inverse


def test_a_topology_is_shapes_and_no_dtype(plain: ExpectedHeader) -> None:
    """`quant(topology)` splits the two halves: the topology says what keys
    exist and how big they are, the rule says what they are made of."""

    shapes = topologies(_SPEC)["ltx2-upsampler.diffusers@2"]["latent_upsampler"]
    computed = plain.component("latent_upsampler")
    assert shapes.keys() == computed.keys()
    for key, shape in shapes.items():
        assert computed[key].shape == shape


def test_a_corpus_root_that_is_not_there_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rules(tmp_path / "nowhere")


# -- the layout-morphism vocabulary (tensorfs#158) ---------------------------
#
# One producer of a layout handle: these records. Rust vendors them at build
# time and Go embeds them, and both APPLY the arrangement; this reader only
# names it. The tests below are about the READING, not about the arrangements
# -- their correctness is `ratify()`'s job in Rust, which applies the map and
# its inverse and compares bytes.


def test_the_layout_corpus_reads_and_is_not_summarised() -> None:
    corpus = layouts(_SPEC)
    on_disk = sorted(p.name for p in (_SPEC / "layouts").glob("*.json"))
    assert len(corpus) == len(on_disk), "a record was dropped or merged"

    channels_last = corpus["torch.channels_last-2d@1"]
    assert channels_last.cls == "inductor"
    assert channels_last.rank == 4
    assert channels_last.permutation == (0, 2, 3, 1)
    assert [axis.extent for axis in channels_last.sub_axes] == ["d0", "d1", "d2", "d3"]
    assert not channels_last.is_identity
    assert channels_last.applies_to(4)
    # A rank the arrangement is not defined for is NOT an error: a 1-D bias
    # under a channels_last declaration is row-major, because the permutation
    # is the identity there.
    assert not channels_last.applies_to(1)

    # The endpoint-declared class is the one the real wins live in, and its
    # extents are FORMULAS -- read verbatim, never evaluated here.
    blocked = corpus["cublas.blockscale-128x4@1"]
    assert blocked.cls == "endpoint-declared"
    assert blocked.permutation == (0, 3, 2, 1, 4)
    assert [axis.extent for axis in blocked.sub_axes] == [
        "ceil(d0/128)", "4", "32", "ceil(d1/4)", "4",
    ]


def test_the_identity_is_derived_from_the_corpus_and_never_spelled() -> None:
    """A consumer that hard-codes `torch.contiguous@1` becomes a second author
    of the one handle every existing artifact carries. It is derivable: the
    identity is the unique RANKLESS record, because row-major is row-major at
    every rank."""

    identity = identity_arrangement(_SPEC)
    assert identity.rank is None
    assert identity.is_identity
    assert identity.permutation == ()
    assert identity.applies_to(1) and identity.applies_to(5)
    assert identity.handle == "torch.contiguous@1"  # what the corpus says today


def test_the_reader_READS_rather_than_restates(tmp_path: Path) -> None:
    """The red arm of the whole one-producer claim: perturb a record and the
    reader must report the perturbed value. A reader that restated the
    permutation as a literal would pass every other test in this file."""

    corpus_root = tmp_path / "v2"
    (corpus_root / "layouts").mkdir(parents=True)
    for path in (_SPEC / "layouts").glob("*.json"):
        (corpus_root / "layouts" / path.name).write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    record = corpus_root / "layouts" / "torch.channels_last-2d.v1.json"
    document = json.loads(record.read_text(encoding="utf-8"))
    assert document["permutation"] == [0, 2, 3, 1]
    document["permutation"] = [0, 3, 2, 1]
    record.write_text(json.dumps(document), encoding="utf-8")

    assert layouts(corpus_root)["torch.channels_last-2d@1"].permutation == (0, 3, 2, 1)


def test_an_absent_layout_class_refuses_instead_of_reading_as_empty(tmp_path: Path) -> None:
    """`{}` reads as a valid, tiny catalog at every call site, and the call
    site's next move is to fall back to row-major -- silently undoing a
    delivery instead of naming a missing corpus."""

    corpus_root = tmp_path / "v2"
    (corpus_root / "rules").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        layouts(corpus_root)


def test_two_rankless_records_are_an_ambiguous_default_and_refuse(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v2"
    (corpus_root / "layouts").mkdir(parents=True)
    source = (_SPEC / "layouts" / "torch.contiguous.v1.json").read_text(encoding="utf-8")
    (corpus_root / "layouts" / "torch.contiguous.v1.json").write_text(
        source, encoding="utf-8"
    )
    twin = json.loads(source)
    twin["name"] = "torch.also-contiguous"
    (corpus_root / "layouts" / "torch.also-contiguous.v1.json").write_text(
        json.dumps(twin), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly ONE rankless record"):
        identity_arrangement(corpus_root)
