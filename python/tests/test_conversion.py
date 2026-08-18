"""The conversion PRODUCER, driven end to end on real bytes.

What this suite is for. `Contract.Verdict` has answered `DerivableVia(dtype-cast)`
since tensorfs#123, and two independent hub gates have emitted a CONVERTIBLE
code for longer than that — while nothing anywhere could produce the derived
checkpoint (tensorhub th#2164). So the assertions that matter here are not
"the function returns a plan"; they are:

* an fp16-packaged SDXL tree — the dominant community packaging, and the named
  over-constraint in Paul's matching-set ruling — is DERIVABLE to the bf16 lane,
  and after the conversion runs it SATISFIES it;
* a bf16 tree converts to the fp8-rowwise lane and satisfies THAT, with the
  per-row scale twins the lane declares actually present;
* the conversion rewrites the unet and nothing else, so every other member's
  CAS objects survive with their digests;
* the refusals that exist because their failure is silent actually fire.

The verdicts themselves are taken by the REAL matcher, in Go, over the bytes
this suite writes: `conversion_proof_test.go`, driven by
`scripts/prove-conversion.sh`. A Python re-implementation of the verdict would
be a third matcher grading its own homework.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest
from tensorfs import LocalCAS, RepositoryManifest, TensorWriter, open_tensors, read_entry
from tensorfs import contracts as library
from tensorfs.convert import (
    FP8_E4M3,
    ConversionRefused,
    _declaration_for,  # the ported claim rule the Go fence re-counts
    convert,
    recipe_for,
)

# One tiny SDXL-shaped diffusers tree: the four members th#2160 drove its bind
# gate against. Shapes are 16-aligned on both axes wherever the fp8 recipe is
# eligible, because that is a conjunct of the producing rule the contract format
# deliberately cannot carry -- a fixture that ignored it would prove the plan
# against weights the real producer would refuse.
_UNET = "unet/diffusion_pytorch_model.safetensors"
_VAE = "vae/diffusion_pytorch_model.safetensors"
_TE = "text_encoder/model.safetensors"
_TE2 = "text_encoder_2/model.safetensors"

_TREE: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    _UNET: (
        ("conv_in.weight", (32, 4, 3, 3)),
        ("conv_in.bias", (32,)),
        ("time_embedding.linear_1.weight", (32, 32)),
        ("time_embedding.linear_1.bias", (32,)),
        ("down_blocks.0.attentions.0.proj_in.weight", (32, 32)),
        ("down_blocks.0.attentions.0.proj_in.bias", (32,)),
        ("down_blocks.0.attentions.0.proj_out.weight", (32, 32)),
        ("down_blocks.0.attentions.0.proj_out.bias", (32,)),
        ("down_blocks.0.attentions.0.norm.weight", (32,)),
        ("down_blocks.0.attentions.0.norm.bias", (32,)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", (32, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight", (32, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight", (32, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight", (32, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.bias", (32,)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight", (64, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.bias", (64,)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight", (32, 32)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.norm1.weight", (32,)),
        ("down_blocks.0.attentions.0.transformer_blocks.0.norm1.bias", (32,)),
        ("down_blocks.0.resnets.0.time_emb_proj.weight", (32, 32)),
        ("down_blocks.0.resnets.0.time_emb_proj.bias", (32,)),
        ("down_blocks.0.resnets.0.conv1.weight", (32, 32, 3, 3)),
        ("down_blocks.0.resnets.0.conv1.bias", (32,)),
        ("down_blocks.0.resnets.0.norm1.weight", (32,)),
        ("down_blocks.0.resnets.0.norm1.bias", (32,)),
    ),
    _VAE: (
        ("encoder.conv_in.weight", (32, 3, 3, 3)),
        ("encoder.conv_in.bias", (32,)),
        ("decoder.conv_in.weight", (32, 4, 3, 3)),
        ("decoder.conv_in.bias", (32,)),
        ("quant_conv.weight", (8, 8, 1, 1)),
        ("quant_conv.bias", (8,)),
    ),
    _TE: (
        ("text_model.encoder.layers.0.self_attn.q_proj.weight", (32, 32)),
        ("text_model.encoder.layers.0.self_attn.q_proj.bias", (32,)),
        ("text_model.encoder.layers.0.self_attn.out_proj.weight", (32, 32)),
        ("text_model.encoder.layers.0.self_attn.out_proj.bias", (32,)),
        ("text_model.encoder.layers.0.mlp.fc1.weight", (64, 32)),
        ("text_model.encoder.layers.0.mlp.fc1.bias", (64,)),
        ("text_model.final_layer_norm.weight", (32,)),
        ("text_model.final_layer_norm.bias", (32,)),
    ),
    _TE2: (
        ("text_model.encoder.layers.0.self_attn.q_proj.weight", (32, 32)),
        ("text_model.encoder.layers.0.self_attn.q_proj.bias", (32,)),
        ("text_model.encoder.layers.0.mlp.fc1.weight", (64, 32)),
        ("text_model.encoder.layers.0.mlp.fc1.bias", (64,)),
        ("text_projection.weight", (32, 32)),
    ),
}


# A tiny MiniMax-H3 DiT, same shape discipline. The attention weights carry the
# native contract's 56-group interleave, so the outer axis is 56 * 2 -- a fusion
# whose arithmetic does not divide is a REFUSAL, and a fixture that ignored it
# would prove the fp8 document against weights the matcher rejects.
_DIT = "transformer/diffusion_pytorch_model.safetensors"
_H3_TREE: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    _DIT: (
        ("transformer_blocks.0.attn.to_q.weight", (112, 32)),
        ("transformer_blocks.0.attn.to_k.weight", (112, 32)),
        ("transformer_blocks.0.attn.to_v.weight", (112, 32)),
        ("transformer_blocks.0.attn.to_out.0.weight", (32, 112)),
        ("transformer_blocks.0.attn.norm_q.weight", (32,)),
        ("transformer_blocks.0.attn.norm_k.weight", (32,)),
        ("transformer_blocks.0.ff.net.0.proj.weight", (64, 32)),
        ("transformer_blocks.0.ff.net.2.weight", (32, 64)),
        ("transformer_blocks.0.adaln_proj.linear.weight", (96, 32)),
        ("transformer_blocks.0.adaln_proj.linear.bias", (96,)),
    ),
}


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
    cas: LocalCAS,
    dtype: str,
    tree: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] | None = None,
) -> RepositoryManifest:
    entries = []
    for path, tensors in (tree or _TREE).items():
        writer = TensorWriter(cas, path)
        for name, shape in tensors:
            writer.add(name, dtype, shape, _body(name, dtype, shape))
        entries.append(writer.finish())
    return RepositoryManifest(tuple(entries))


@pytest.fixture()
def fp16_tree(tmp_path: Path) -> tuple[LocalCAS, RepositoryManifest]:
    cas = LocalCAS(tmp_path / "cas")
    return cas, _seed(cas, "F16")


@pytest.fixture()
def bf16_tree(tmp_path: Path) -> tuple[LocalCAS, RepositoryManifest]:
    cas = LocalCAS(tmp_path / "cas")
    return cas, _seed(cas, "BF16")


def _member(cas: LocalCAS, manifest: RepositoryManifest, path: str) -> object:
    entry = next(e for e in manifest.files if e.path == path)
    return open_tensors(cas, RepositoryManifest((entry,)))


# -- the recipe is the lane's property, not the caller's ---------------------


def test_the_recipe_is_read_out_of_the_target_lane() -> None:
    assert recipe_for(library.get("sdxl.diffusers-bf16@1")) == "dtype-cast"
    assert recipe_for(library.get("sdxl.diffusers-fp8-rowwise@1")) == "fp8-rowwise"
    assert recipe_for(library.get("minimax.h3-dit-fp8-rowwise@1")) == "fp8-rowwise"


# -- direction 1: the over-constraint remedy --------------------------------


def test_an_fp16_tree_converts_to_the_bf16_lane(
    fp16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = fp16_tree
    result = convert(cas, manifest, library.get("sdxl.diffusers-bf16@1"))
    assert result.plan.recipe == "dtype-cast"
    # Every member carries claimed tensors, so every member is rewritten: this
    # lane's dtype constraint is BF16 on all four.
    assert set(result.rewritten) == set(_TREE)
    for path in _TREE:
        with _member(cas, result.manifest, path) as out:  # type: ignore[attr-defined]
            for name in out:
                assert out[name].dtype == "BF16", f"{path}:{name}"


def test_the_conversion_is_lossless_within_bf16(
    fp16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """A cast is math, so the claim is not "identical" -- it is "the rounding is
    the rounding". Round-to-nearest-even, not truncation: truncating biases
    every magnitude toward zero, which no shape or dtype check can see."""

    cas, manifest = fp16_tree
    name = "down_blocks.0.resnets.0.time_emb_proj.weight"
    with _member(cas, manifest, _UNET) as source:  # type: ignore[attr-defined]
        before = source[name].tobytes()
    result = convert(cas, manifest, library.get("sdxl.diffusers-bf16@1"))
    with _member(cas, result.manifest, _UNET) as out:  # type: ignore[attr-defined]
        after = out[name].tobytes()
    for index in range(len(before) // 2):
        source = struct.unpack_from("<e", before, index * 2)[0]
        lo, hi = after[index * 2], after[index * 2 + 1]
        result = struct.unpack("<f", bytes((0, 0, lo, hi)))[0]
        assert abs(result - source) <= abs(source) * 2.0**-8 + 2.0**-20


# -- the fp8-rowwise lane ---------------------------------------------------


def test_a_bf16_tree_converts_to_the_fp8_rowwise_lane(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    result = convert(cas, manifest, library.get("sdxl.diffusers-fp8-rowwise@1"))
    assert result.plan.recipe == "fp8-rowwise"
    # THE POINT: fp8 storage targets the denoiser, so the vae and both text
    # encoders are not touched at all and keep every digest they had.
    assert result.rewritten == (_UNET,)
    with _member(cas, result.manifest, _UNET) as out:  # type: ignore[attr-defined]
        quantized = [name for name in out if out[name].dtype == FP8_E4M3]
        scales = [name for name in out if name.endswith(".weight_scale")]
        assert quantized, "the recipe converted nothing"
        assert len(scales) == len(quantized), (
            "every fp8 weight needs its per-row scale; a mismatch here IS the "
            "half-quantized artifact"
        )
        for name in quantized:
            scale = out[name[: -len(".weight")] + ".weight_scale"]
            assert scale.dtype == "F32"
            assert scale.shape == (out[name].shape[0],), (
                "a per-ROW scale has one entry per output row"
            )
        # The rule's own consequences, asserted so a future edit to the skip
        # list is caught by a test rather than by a render.
        assert "down_blocks.0.attentions.0.proj_in.weight" in quantized
        assert "down_blocks.0.resnets.0.time_emb_proj.weight" in quantized
        assert "time_embedding.linear_1.weight" not in quantized
        assert "conv_in.weight" not in quantized
        assert "down_blocks.0.attentions.0.norm.weight" not in quantized


def test_the_scale_is_a_multiplier_not_its_reciprocal(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """Dequant is ``fp8 * scale``. Storing ``1/scale`` keeps every name, dtype
    and shape correct and is five orders of magnitude wrong."""

    cas, manifest = bf16_tree
    name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    with _member(cas, manifest, _UNET) as reader:  # type: ignore[attr-defined]
        source = reader[name]
        rows, columns = source.shape
        raw = source.tobytes()
        original = []
        for index in range(rows * columns):
            lo, hi = raw[index * 2], raw[index * 2 + 1]
            original.append(struct.unpack("<f", bytes((0, 0, lo, hi)))[0])
    result = convert(cas, manifest, library.get("sdxl.diffusers-fp8-rowwise@1"))
    with _member(cas, result.manifest, _UNET) as out:  # type: ignore[attr-defined]
        weights = out[name].tobytes()
        scale_name = name[: -len(".weight")] + ".weight_scale"
        scales = struct.unpack(f"<{rows}f", out[scale_name].tobytes())
    for row in range(rows):
        peak = max(abs(v) for v in original[row * columns : (row + 1) * columns])
        if peak == 0.0:
            continue
        recovered = [
            _decode_e4m3(weights[row * columns + column]) * scales[row] for column in range(columns)
        ]
        row_slice = original[row * columns : (row + 1) * columns]
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
    result = convert(cas, manifest, library.get("sdxl.diffusers-fp8-rowwise@1"))
    for entry in result.manifest.files:
        if entry.path == _UNET:
            continue
        assert read_entry(cas, entry) == untouched[entry.path], entry.path


# -- the refusals that exist because the failure is silent ------------------


def test_requantizing_an_fp8_tree_is_a_no_op_not_a_second_pass(
    bf16_tree: tuple[LocalCAS, RepositoryManifest],
) -> None:
    cas, manifest = bf16_tree
    target = library.get("sdxl.diffusers-fp8-rowwise@1")
    result = convert(cas, manifest, target)
    again = convert(cas, result.manifest, target)
    assert again.plan.converted == 0, "a second pass must not re-quantize fp8 bytes"
    assert again.rewritten == ()


def test_a_lane_whose_recipe_matches_nothing_refuses(tmp_path: Path) -> None:
    """The module shape moved under the recipe. Writing the file anyway would
    stamp a bit-identical copy of the source as an fp8 lane, and the pod would
    then serve a silently-bf16 'fp8' checkpoint."""

    cas = LocalCAS(tmp_path / "cas")
    writer = TensorWriter(cas, _UNET)
    writer.add("nothing.this.lane.knows", "BF16", (4, 4), _body("x", "BF16", (4, 4)))
    manifest = RepositoryManifest((writer.finish(),))
    with pytest.raises(ConversionRefused, match="converts none of them"):
        convert(cas, manifest, library.get("sdxl.diffusers-fp8-rowwise@1"))


# -- the proof fixture the Go matcher grades --------------------------------


def test_write_the_verdict_proof_fixture(tmp_path: Path) -> None:
    """Emit both trees' headers for `conversion_proof_test.go`.

    The verdict is NOT taken here. This test's whole job is to put real bytes
    where the real matcher can read them, so the claim "the converted tree
    satisfies the lane" is answered by the code the hub actually calls.
    """

    out_dir = os.environ.get("TENSORFS_CONVERSION_PROOF_DIR")
    if not out_dir:
        pytest.skip("set TENSORFS_CONVERSION_PROOF_DIR (scripts/prove-conversion.sh)")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, object]] = {}
    for label, dtype, lane, tree in (
        ("fp16-to-bf16", "F16", "sdxl.diffusers-bf16@1", _TREE),
        ("bf16-to-fp8-rowwise", "BF16", "sdxl.diffusers-fp8-rowwise@1", _TREE),
        ("h3-bf16-to-fp8-rowwise", "BF16", "minimax.h3-dit-fp8-rowwise@1", _H3_TREE),
    ):
        cas = LocalCAS(tmp_path / f"cas-{label}")
        manifest = _seed(cas, dtype, tree)
        before = _headers(cas, manifest)
        result = convert(cas, manifest, library.get(lane))
        after = _headers(cas, result.manifest)
        cases[label] = {
            "lane": lane,
            "before": before,
            "after": after,
            "rewritten": list(result.rewritten),
            # The parity fence. The planner needs to know which declaration
            # claims a tensor, and it answers that with a Python port of a rule
            # whose authority is Rust. A port that silently claims NOTHING is
            # not hypothetical -- it is the bug this suite caught while being
            # written (a two-segment pattern was read as ambiguous, so every
            # `a.{i}.b` declaration matched nothing and every conversion was a
            # no-op that still looked like a pass). So Go re-counts.
            "python_claimed": {
                path: sum(
                    1
                    for tensor in tensors
                    if _declaration_for(library.get(lane), str(tensor["name"]))
                    is not None
                )
                for path, tensors in after.items()
            },
        }

    (destination / "verdict-cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")


def _headers(cas: LocalCAS, manifest: RepositoryManifest) -> dict[str, list[dict[str, object]]]:
    files: dict[str, list[dict[str, object]]] = {}
    for entry in manifest.files:
        with open_tensors(cas, RepositoryManifest((entry,))) as reader:
            files[entry.path] = [
                {
                    "name": reader[name].name,
                    "dtype": reader[name].dtype,
                    "shape": list(reader[name].shape),
                    "length": reader[name].nbytes,
                }
                for name in sorted(reader)
            ]
    return files


def test_the_h3_fp8_document_is_what_the_producer_writes(tmp_path: Path) -> None:
    """The H3 fp8 lane, closed-loop.

    This document was DERIVED (from minimax.h3-dit-diffusers@1 plus gen-worker's
    eligibility rule) rather than read off the live artifact, which this author
    could not reach. So the claim it is entitled to make is exactly this one:
    the producer's output satisfies it. Whether the artifact already on the hub
    was made by this producer is a separate question, named in the document's
    own description and owed in tensorfs#128 -- and if the answer is no, the
    failure is a bind refusal naming the tensor, never a silent mismatch.
    """

    cas = LocalCAS(tmp_path / "cas")
    manifest = _seed(cas, "BF16", _H3_TREE)
    result = convert(cas, manifest, library.get("minimax.h3-dit-fp8-rowwise@1"))
    assert result.plan.recipe == "fp8-rowwise"
    with _member(cas, result.manifest, _DIT) as out:  # type: ignore[attr-defined]
        quantized = sorted(name for name in out if out[name].dtype == FP8_E4M3)
        assert quantized == [
            "transformer_blocks.0.adaln_proj.linear.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.ff.net.2.weight",
        ]
        # The q/k/v scales are REQUIRED by the document, which is how it refuses
        # a half-quantized DiT; assert they are actually there.
        for leaf in ("to_q", "to_k", "to_v"):
            scale = out[f"transformer_blocks.0.attn.{leaf}.weight_scale"]
            assert scale.dtype == "F32" and scale.shape == (112,)
        # The norms are the rule's own exclusions, not an oversight.
        assert out["transformer_blocks.0.attn.norm_q.weight"].dtype == "BF16"
        assert out["transformer_blocks.0.adaln_proj.linear.bias"].dtype == "BF16"
