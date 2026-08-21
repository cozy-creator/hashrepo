"""The v2 read surface, against layouts the Go engine actually printed.

The fixtures under `data/` are verbatim `go run ./scripts/compute_layout`
output, committed rather than regenerated: a test that shelled out to `go`
would be testing the toolchain's availability, and one that hand-wrote the
document would be testing this file against itself.

Two pairs, chosen for what they are entitled to prove:

* `ltx2-upsampler.diffusers@1+plain.bf16@1` -- an identity rule. Every key
  keeps its shape, `transformed` is 0, and the whole component carries one
  element type.
* `ernie.diffusers@1+cozy.fp8-rowwise@1` -- a transforming rule. 253 weights
  became F8_E4M3 and grew an F32 `weight_scale` twin, and the quant block
  carries the dequant identity and the sm floor a serving loader needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tensorfs.layout2 import ExpectedHeader, LayoutTensor, rules, topologies

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
    assert plain.stamp == "ltx2-upsampler.diffusers@1+plain.bf16@1"
    assert plain.topology == "ltx2-upsampler.diffusers@1"
    assert len(plain.topology_digest) == 64
    assert plain.quant.handle == "plain.bf16@1"
    assert plain.quant.family == "plain.bf16"
    assert plain.quant.declared_dtype == "bfloat16"
    assert plain.quant.capability_floor_sm == 80
    assert plain.quant.lossy is False
    assert plain.quant.transformed == 0
    assert plain.quant.inverse == ""
    assert plain.components.keys() == {"latent_upsampler"}
    assert plain.roles["latent_upsampler"] == "other"

    tensors = plain.tensors()
    assert len(tensors) == 72
    assert tensors["initial_conv.weight"] == LayoutTensor(("BF16",), (1024, 128, 3, 3, 3))
    # An identity rule stamps ONE element type across the whole component, so
    # nothing here is at a second dtype.
    assert {entry.dtypes for entry in tensors.values()} == {("BF16",)}


def test_a_transforming_layout_carries_the_scales_and_the_dequant(
    rowwise: ExpectedHeader,
) -> None:
    assert rowwise.quant.handle == "cozy.fp8-rowwise@1"
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
    assert plain.tensors() is plain.component("latent_upsampler")
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
    assert corpus["cozy.fp8-rowwise@1"].capability_floor_sm == 89
    assert corpus["plain.f32@1"].capability_floor_sm == 0
    assert corpus["bfl.nvfp4-preswizzled@1"].capability_floor_sm == 100

    storage = corpus["cozy.fp8-storage@1"]
    rowwise = corpus["cozy.fp8-rowwise@1"]
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

    shapes = topologies(_SPEC)["ltx2-upsampler.diffusers@1"]["latent_upsampler"]
    computed = plain.tensors()
    assert shapes.keys() == computed.keys()
    for key, shape in shapes.items():
        assert computed[key].shape == shape


def test_a_corpus_root_that_is_not_there_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rules(tmp_path / "nowhere")
