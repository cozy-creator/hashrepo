"""The conversion PRODUCER, driven end to end on real bytes.

What this suite is for. Two independent hub gates emitted a CONVERTIBLE code
for longer than anything could produce the derived checkpoint (tensorhub
th#2164). So the assertions that matter here are not "the function returns a
plan"; they are:

* an fp16-packaged tree is DERIVABLE to a bf16 layout, and the rounding is
  round-to-nearest-even rather than the truncation that biases every magnitude
  toward zero;
* a bf16 tree converts to `cozy.fp8-rowwise` with the per-row scale twins the
  computed layout declares actually present, and those scales are MULTIPLIERS;
* the conversion rewrites the denoiser and nothing else, so every other
  member's CAS objects survive with their digests;
* the refusals that exist because their failure is silent actually fire.

**Where the target layouts come from.** `data/layout-*.json` is verbatim
`go run ./scripts/compute_layout` output, committed. It is the whole point of
v2 that Python does not compute a layout, so the fixtures the producer aims at
are Go's own — including the shapes, which is what lets the planner refuse a
checkpoint of the right family and the wrong size.

The real layouts' tensors are real sizes (ernie's smallest fp8 Linear is
4096x4096), which no pure-Python kernel can push bytes through in a test. So
the byte-level fp8 loop runs against a SMALL layout built here, carrying the
quant block lifted verbatim from the committed ernie fixture -- the rule
identity is Go's, only the topology half is miniaturized.
"""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tensorfs import LocalCAS, RepositoryManifest, TensorWriter, open_tensors, read_entry
from tensorfs.convert import (
    FP8_E4M3,
    ConversionRefused,
    component_of,
    convert,
    recipe_for,
)
from tensorfs.layout2 import ExpectedHeader

_DATA = Path(__file__).resolve().parent / "data"


def _fixture(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((_DATA / name).read_text(encoding="utf-8"))
    return loaded


_LTX2 = _fixture("layout-ltx2-upsampler-plain-bf16.json")
_ERNIE = _fixture("layout-ernie-fp8-rowwise.json")


# -- writing trees ----------------------------------------------------------


def _count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _body(name: str, dtype: str, shape: tuple[int, ...]) -> bytes:
    """Deterministic, finite, and spread over several binades.

    Weights that are all one magnitude make a per-row scale trivially exact and
    hide a reciprocal-vs-multiplier mistake, which is the fp8 bug that survives
    every shape and dtype check.
    """

    seed = sum(ord(character) for character in name)
    out = bytearray()
    for index in range(_count(shape)):
        value = ((seed + index * 7) % 97 - 48) / 11.0
        if dtype == "F16":
            out += struct.pack("<e", value)
        elif dtype == "BF16":
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            out += struct.pack("<H", (bits >> 16) & 0xFFFF)
        else:
            out += struct.pack("<f", value)
    return bytes(out)


def _seed(
    cas: LocalCAS, dtype: str, tree: dict[str, dict[str, tuple[int, ...]]]
) -> RepositoryManifest:
    entries = []
    for path, tensors in tree.items():
        writer = TensorWriter(cas, path)
        for name, shape in tensors.items():
            writer.add(name, dtype, shape, _body(name, dtype, shape))
        entries.append(writer.finish())
    return RepositoryManifest(tuple(entries))


def _member(cas: LocalCAS, manifest: RepositoryManifest, path: str) -> Any:
    entry = next(e for e in manifest.files if e.path == path)
    return open_tensors(cas, RepositoryManifest((entry,)))


# -- the small fp8 layout, with Go's quant block --------------------------


#: The four members th#2160 drove its bind gate against. Shapes are 16-aligned
#: on both axes wherever the fp8 rule is eligible, because that is a conjunct of
#: the producing rule, and a fixture that ignored it would prove the plan
#: against weights the real producer would refuse.
_UNET = "unet/diffusion_pytorch_model.safetensors"
_VAE = "vae/diffusion_pytorch_model.safetensors"
_TE = "text_encoder/model.safetensors"
_TE2 = "text_encoder_2/model.safetensors"

_UNET_BF16: dict[str, tuple[int, ...]] = {
    "conv_in.weight": (32, 4, 3, 3),
    "conv_in.bias": (32,),
    "time_embedding.linear_1.weight": (32, 32),
    "down_blocks.0.attentions.0.norm.weight": (32,),
}
_UNET_FP8: dict[str, tuple[int, ...]] = {
    "down_blocks.0.attentions.0.proj_in.weight": (32, 32),
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": (32, 32),
    "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight": (64, 32),
    "down_blocks.0.resnets.0.time_emb_proj.weight": (32, 32),
}
_VAE_BF16: dict[str, tuple[int, ...]] = {
    "encoder.conv_in.weight": (32, 3, 3, 3),
    "encoder.conv_in.bias": (32,),
}
# THE v1 COLLISION, kept as a fixture: both text encoders carry this key, at
# different widths. A flat pattern set could not tell CLIP-L from CLIP-G; two
# finite maps can.
_SHARED_KEY = "text_model.encoder.layers.0.self_attn.q_proj.weight"
_TE_BF16: dict[str, tuple[int, ...]] = {_SHARED_KEY: (32, 32)}
_TE2_BF16: dict[str, tuple[int, ...]] = {_SHARED_KEY: (64, 64)}

_TREE: dict[str, dict[str, tuple[int, ...]]] = {
    _UNET: {**_UNET_BF16, **_UNET_FP8},
    _VAE: _VAE_BF16,
    _TE: _TE_BF16,
    _TE2: _TE2_BF16,
}


def _component(name: str, role: str, tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "role": role, "tensors": tensors}


def _plain(shapes: dict[str, tuple[int, ...]], dtype: str = "BF16") -> dict[str, dict[str, Any]]:
    return {key: {"dtypes": [dtype], "shape": list(shape)} for key, shape in shapes.items()}


def _quantized(shapes: dict[str, tuple[int, ...]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, shape in shapes.items():
        out[key] = {"dtypes": ["F8_E4M3"], "shape": list(shape)}
        out[key.removesuffix(".weight") + ".weight_scale"] = {
            "dtypes": ["F32"],
            "shape": [shape[0]],
        }
    return out


def _layout(components: list[dict[str, Any]], quant: dict[str, Any], stamp: str) -> ExpectedHeader:
    transformed = sum(
        1
        for component in components
        for entry in component["tensors"].values()
        if entry["dtypes"] == ["F8_E4M3"]
    )
    quant = {**deepcopy(quant), "transformed": transformed}
    return ExpectedHeader.from_document(
        {
            "stamp": stamp,
            "topology": stamp.split("+")[0],
            "topology_digest": "0" * 64,
            "quant": quant,
            "components": components,
        }
    )


def _fp8_layout() -> ExpectedHeader:
    """The small SDXL-shaped fp8 target. The quant block is Go's, verbatim."""

    return _layout(
        [
            _component(
                "unet",
                "denoiser",
                {**_plain(_UNET_BF16), **_quantized(_UNET_FP8)},
            ),
            _component("vae", "vae", _plain(_VAE_BF16)),
            _component("text_encoder", "text_encoder", _plain(_TE_BF16)),
            _component("text_encoder_2", "text_encoder", _plain(_TE2_BF16)),
        ],
        _ERNIE["quant"],
        "sdxl-small.diffusers@1+cozy.fp8-rowwise@1",
    )


def _bf16_layout() -> ExpectedHeader:
    return _layout(
        [
            _component("unet", "denoiser", _plain({**_UNET_BF16, **_UNET_FP8})),
            _component("vae", "vae", _plain(_VAE_BF16)),
            _component("text_encoder", "text_encoder", _plain(_TE_BF16)),
            _component("text_encoder_2", "text_encoder", _plain(_TE2_BF16)),
        ],
        _LTX2["quant"],
        "sdxl-small.diffusers@1+plain.bf16@1",
    )


@pytest.fixture()
def bf16_tree(tmp_path: Path) -> tuple[LocalCAS, RepositoryManifest]:
    cas = LocalCAS(tmp_path / "cas")
    return cas, _seed(cas, "BF16", _TREE)


# -- the recipe is the quant rule's identity --------------------------------


def test_the_recipe_is_the_quant_rule() -> None:
    assert recipe_for(ExpectedHeader.from_document(_LTX2)) == "dtype-cast"
    assert recipe_for(ExpectedHeader.from_document(_ERNIE)) == "fp8-rowwise"


@pytest.mark.parametrize("handle", ["cozy.nvfp4-flat@1", "cozy.fp8-storage@1"])
def test_a_rule_this_producer_has_no_kernel_for_refuses(handle: str) -> None:
    """The silent default #129 forbids, in both of its shapes. Falling through
    to a cast for a 4-bit rule writes bytes in the target's element type with
    none of the rule's scales. `cozy.fp8-storage` is subtler and worse: it is a
    scale-free fp8 rule, so a plain cast produces a file that looks entirely
    right -- except that torch's fp8 cast does not saturate, and every weight
    the rule would have clamped to 448 is now NaN."""

    foreign = deepcopy(_ERNIE)
    foreign["quant"]["handle"] = handle
    with pytest.raises(ConversionRefused, match="no kernel for quant rule"):
        recipe_for(ExpectedHeader.from_document(foreign))


def test_a_rowwise_rule_with_foreign_conventions_refuses() -> None:
    """Same family, re-versioned with a different scale granularity. The
    kernel below would write confidently wrong scales, so the conventions are
    read rather than assumed from the family name."""

    blockwise = deepcopy(_ERNIE)
    blockwise["quant"]["conventions"]["scale"] = "block_128x128"
    with pytest.raises(ConversionRefused, match="conventions this producer"):
        recipe_for(ExpectedHeader.from_document(blockwise))


# -- direction 1: the over-constraint remedy --------------------------------


def test_an_fp16_tree_converts_to_a_bf16_layout(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    manifest = _seed(cas, "F16", _TREE)
    result = convert(cas, manifest, _bf16_layout())
    assert result.plan.recipe == "dtype-cast"
    assert set(result.rewritten) == set(_TREE)
    for path in _TREE:
        with _member(cas, result.manifest, path) as out:
            for name in out:
                assert out[name].dtype == "BF16", f"{path}:{name}"


def test_the_conversion_is_lossless_within_bf16(tmp_path: Path) -> None:
    """A cast is math, so the claim is not "identical" -- it is "the rounding is
    the rounding". Round-to-nearest-even, not truncation: truncating biases
    every magnitude toward zero, which no shape or dtype check can see."""

    cas = LocalCAS(tmp_path / "cas")
    manifest = _seed(cas, "F16", _TREE)
    name = "down_blocks.0.resnets.0.time_emb_proj.weight"
    with _member(cas, manifest, _UNET) as source:
        before = source[name].tobytes()
    result = convert(cas, manifest, _bf16_layout())
    with _member(cas, result.manifest, _UNET) as out:
        after = out[name].tobytes()
    for index in range(len(before) // 2):
        original = struct.unpack_from("<e", before, index * 2)[0]
        lo, hi = after[index * 2], after[index * 2 + 1]
        rounded = struct.unpack("<f", bytes((0, 0, lo, hi)))[0]
        assert abs(rounded - original) <= abs(original) * 2.0**-8 + 2.0**-20


def test_a_real_computed_layout_drives_the_planner(tmp_path: Path) -> None:
    """The same cast, aimed at a layout this repository did not author: the
    ltx2 upsampler as `compute_layout` printed it, keys and shapes and all."""

    layout = ExpectedHeader.from_document(_LTX2)
    shapes = {
        key: entry.shape
        for key, entry in layout.component("latent_upsampler").items()
        if _count(entry.shape) <= 1024
    }
    assert shapes, "the fixture has no cheap keys to drive bytes through"

    cas = LocalCAS(tmp_path / "cas")
    member = "latent_upsampler/diffusion_pytorch_model.safetensors"
    manifest = _seed(cas, "F16", {member: shapes})
    result = convert(cas, manifest, layout)
    assert result.rewritten == (member,)
    with _member(cas, result.manifest, member) as out:
        assert {out[name].dtype for name in out} == {"BF16"}


def test_a_shape_the_layout_does_not_declare_refuses(tmp_path: Path) -> None:
    """v1 could only compare dtypes, so a checkpoint of the right family and
    the wrong size converted cleanly and failed at load. The computed layout
    carries the shape."""

    layout = ExpectedHeader.from_document(_LTX2)
    cas = LocalCAS(tmp_path / "cas")
    member = "latent_upsampler/diffusion_pytorch_model.safetensors"
    manifest = _seed(cas, "F16", {member: {"final_conv.bias": (127,)}})
    with pytest.raises(ConversionRefused, match="not that model"):
        convert(cas, manifest, layout)


# -- the fp8-rowwise rule ---------------------------------------------------


def test_a_bf16_tree_converts_to_the_fp8_rowwise_layout(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    result = convert(cas, manifest, _fp8_layout())
    assert result.plan.recipe == "fp8-rowwise"
    # THE POINT: the rule scopes itself to the denoiser, so the vae and both
    # text encoders are not touched at all and keep every digest they had.
    assert result.rewritten == (_UNET,)
    with _member(cas, result.manifest, _UNET) as out:
        quantized = sorted(name for name in out if out[name].dtype == FP8_E4M3)
        scales = [name for name in out if name.endswith(".weight_scale")]
        assert quantized == sorted(_UNET_FP8)
        assert len(scales) == len(quantized), (
            "every fp8 weight needs its per-row scale; a mismatch here IS the "
            "half-quantized artifact"
        )
        for name in quantized:
            scale = out[name.removesuffix(".weight") + ".weight_scale"]
            assert scale.dtype == "F32"
            assert scale.shape == (out[name].shape[0],), (
                "a per-ROW scale has one entry per output row"
            )
        for name in _UNET_BF16:
            assert out[name].dtype == "BF16", f"{name} is outside the rule's scope"


def test_the_scale_is_a_multiplier_not_its_reciprocal(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """Dequant is ``fp8 * scale``. Storing ``1/scale`` keeps every name, dtype
    and shape correct and is five orders of magnitude wrong."""

    cas, manifest = bf16_tree
    name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    with _member(cas, manifest, _UNET) as reader:
        source = reader[name]
        rows, columns = source.shape
        raw = source.tobytes()
        original = []
        for index in range(rows * columns):
            lo, hi = raw[index * 2], raw[index * 2 + 1]
            original.append(struct.unpack("<f", bytes((0, 0, lo, hi)))[0])
    result = convert(cas, manifest, _fp8_layout())
    with _member(cas, result.manifest, _UNET) as out:
        weights = out[name].tobytes()
        scale = out[name.removesuffix(".weight") + ".weight_scale"]
        scales = struct.unpack(f"<{rows}f", scale.tobytes())
    for row in range(rows):
        row_slice = original[row * columns : (row + 1) * columns]
        peak = max(abs(value) for value in row_slice)
        if peak == 0.0:
            continue
        recovered = [
            _decode_e4m3(weights[row * columns + column]) * scales[row] for column in range(columns)
        ]
        for observed, expected in zip(recovered, row_slice, strict=True):
            assert abs(observed - expected) <= peak / 8.0


def _decode_e4m3(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    if exponent == 0x0F and mantissa == 0x07:
        return float("nan")
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def test_untouched_members_keep_their_objects(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    untouched = {
        entry.path: read_entry(cas, entry) for entry in manifest.files if entry.path != _UNET
    }
    result = convert(cas, manifest, _fp8_layout())
    for entry in result.manifest.files:
        if entry.path == _UNET:
            continue
        assert read_entry(cas, entry) == untouched[entry.path], entry.path


def test_each_text_encoder_is_measured_against_its_own_component(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """The v1 collision, resolved. `text_encoder` and `text_encoder_2` carry
    the same key at 32x32 and 64x64; if the planner reached for a single flat
    rule set, one of them would refuse on shape."""

    cas, manifest = bf16_tree
    layout = _fp8_layout()
    assert component_of(layout, _TE) == "text_encoder"
    assert component_of(layout, _TE2) == "text_encoder_2"
    assert layout.tensors(_TE.split("/")[0])[_SHARED_KEY].shape == (32, 32)
    assert layout.tensors(_TE2.split("/")[0])[_SHARED_KEY].shape == (64, 64)
    # And it survives the whole producer: neither member is rewritten, and
    # neither raises.
    assert convert(cas, manifest, layout).rewritten == (_UNET,)


def test_a_member_the_layout_does_not_name_is_carried_by_reference(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    stray = TensorWriter(cas, "scheduler/extra.safetensors")
    stray.add("whatever", "BF16", (4, 4), _body("x", "BF16", (4, 4)))
    widened = RepositoryManifest((*manifest.files, stray.finish()))
    before = {entry.path: read_entry(cas, entry) for entry in widened.files}
    result = convert(cas, widened, _fp8_layout())
    assert "scheduler/extra.safetensors" not in result.rewritten
    for entry in result.manifest.files:
        if entry.path != _UNET:
            assert read_entry(cas, entry) == before[entry.path]


# -- the refusals that exist because the failure is silent ------------------


def test_requantizing_an_fp8_tree_is_a_no_op_not_a_second_pass(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    target = _fp8_layout()
    result = convert(cas, manifest, target)
    again = convert(cas, result.manifest, target)
    assert again.plan.converted == 0, "a second pass must not re-quantize fp8 bytes"
    assert again.rewritten == ()


def test_a_rule_that_transforms_nothing_here_refuses(tmp_path: Path) -> None:
    """The module shape moved under the rule. Writing the file anyway would
    stamp a bit-identical copy of the source as an fp8 layout, and the pod
    would then serve a silently-bf16 'fp8' checkpoint."""

    cas = LocalCAS(tmp_path / "cas")
    manifest = _seed(cas, "BF16", {_UNET: _UNET_BF16, _VAE: _VAE_BF16})
    with pytest.raises(ConversionRefused, match="converts none of them"):
        convert(cas, manifest, _fp8_layout())
