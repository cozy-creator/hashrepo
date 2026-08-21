#!/usr/bin/env python3
"""Derive ``sdxl.diffusers-nvfp4-flat@1`` from the fp8-rowwise sibling.

NOT HAND-WRITTEN, for the same reason neither fp8 document was: the packaging
is a rule, so applying the rule is the only way the document and the code that
reads the bytes cannot disagree.

Two inputs, each authoritative for exactly one half:

* ``sdxl.diffusers-fp8-rowwise@1`` supplies the SDXL tree and the QUANTIZED SET
  (which Linears a w8a8 export of this UNET converts).
* ``python-gen-worker/src/gen_worker/models/w4a4.py`` supplies the PACKAGING,
  transcribed from the module docstring that its ``@unregistered_decode_path``
  marker refers to: per quantized Linear ``L``,

      L.weight          U8       [out, in/2]   packed e2m1 nibble pairs,
                                               element 2j in the LOW nibble
      L.weight_scale    F8_E4M3  [out, in/16]  per-16-block, FLAT row-major
      L.weight_scale_2  F32      scalar        second-level per-tensor scale
      L.input_scale     F32      scalar        optional static activation scale
      L.pre_quant_scale float    [in]          optional AWQ-lite smoothing

THIS IS NOT ``bfl.nvfp4-preswizzled@1`` AND MUST NEVER ALIAS IT. That layout is
HIGH-nibble with pre-swizzled scales; reading one as the other measured LPIPS
1.11 — every name, dtype and shape correct and every pixel wrong. Two element
orders and two scale layouts are two contracts, and giving them one stamp is
how the mistake becomes unrepresentable-in-the-wrong-direction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

SOURCE = ROOT / "spec" / "v1" / "contracts" / "sdxl.diffusers-fp8-rowwise.v1.json"
TARGET = ROOT / "spec" / "v1" / "contracts" / "sdxl.diffusers-nvfp4-flat.v1.json"

DESCRIPTION = """\
GENERATED CANDIDATE - NOT RATIFIED. The FLAT nvfp4 W4A4 serve layout for SDXL: \
a normal diffusers tree whose UNET Linears hold modelopt-exported nvfp4 weights \
with TWO-LEVEL scales. Derived, not hand-written - the SDXL tree and the \
quantized SET come from sdxl.diffusers-fp8-rowwise@1 and the PACKAGING is \
transcribed from python-gen-worker/src/gen_worker/models/w4a4.py's module \
docstring, which is the decode path that actually reads these bytes. Per \
quantized Linear L: L.weight is U8 [out, in/2] holding packed e2m1 nibble \
pairs with element 2j in the LOW nibble (the torch.float4_e2m1fn_x2 \
convention); L.weight_scale is F8_E4M3 [out, in/16], per-16-block and FLAT \
row-major; L.weight_scale_2 is an F32 per-tensor second-level scale; \
L.input_scale (F32 scalar, static activation second-level scale) and \
L.pre_quant_scale (float [in], AWQ-lite smoothing folded back on dequant) are \
optional. Dequant is W ~= e2m1(weight) * weight_scale.float() * weight_scale_2. \
Excluded Linears carry NO scale tensors at all, which is why w4a4.py detects \
quantization per layer by the (U8 weight + E4M3 weight_scale + weight_scale_2) \
TRIPLE and never by a name list - the same triple is also what tells this \
layout apart from w8a8, whose weight IS e4m3. \
THIS IS DELIBERATELY NOT bfl.nvfp4-preswizzled@1 AND MUST NEVER BE ALIASED TO \
IT. That contract is HIGH-nibble with PRE-SWIZZLED scales; w4a4.py's \
@unregistered_decode_path records the measurement that settles it - conflating \
the two reads LPIPS 1.11, i.e. every name, dtype and shape correct and every \
number wrong. A separate stamp is the only representation in which that \
mistake is a refusal rather than a shipped disaster. \
TOP-LEVEL DTYPE IS float4_e2m1fn, and it names the LANE'S QUANTIZATION rather \
than any tensor's container type - the same thing sdxl.diffusers-fp8-rowwise@1 \
does when it declares float8_e4m3fn over 257 declarations of which only 36 are \
fp8. It is AUTHORED, not derived: no per-tensor dtype here spells fp4 at all \
(the resident weights are U8 pairs), so a document that waited for the header to \
say 'nvfp4' would wait forever. It is also the load-bearing consumer - \
gen-worker derives the sm floor from this field alone \
(DTYPE_MIN_SM['float4_e2m1fn'] = 100, Blackwell), which is the whole reason the \
spelling is not float4_e2m1fn_x2. DO NOT 'FIX' IT TO _x2: that is the packed-pair \
type torch actually ships (torch.float4_e2m1fn_x2 exists; torch.float4_e2m1fn \
does not) and it would read prettier, but DTYPE_MIN_SM does not know it, so the \
lane would silently lose its sm100 floor and be placed on Ampere. A dtype that \
resolves through torch and drops the floor is worse than one that refuses through \
torch and keeps it - and nothing on this path calls ctx.lane.dtype anyway, because \
the w4a4 loader reads the U8 weights and their scales directly. \
RATIFICATION OWED: (1) THE QUANTIZED SET IS PROVISIONAL. modelopt decides it \
during calibration, and w4a4.py states plainly that no real artifact has ever \
passed the publish gate, so there is no header to measure it from; it is \
inherited here from the w8a8 rule that is measured for this same UNET. The \
failure direction is the safe one - a wrong set makes a real artifact REFUSE \
this document by name rather than load silently mis-scaled - and the first \
artifact that publishes must re-derive it into a version 2. (2) ROLES are \
inherited from the fp8 sibling, so the two quantized packagings of one UNET \
already share a role vocabulary; the nvfp4 suffixes (.scale2, .input_scale, \
.pre_quant_scale) are new and are the ratifier's to confirm. (3) SHAPE \
CONJUNCTS ARE ABSENT because contracts are shapeless on purpose (tensorfs#122 \
rec 1): neither the in/2 packing nor the in/16 block count nor the K/N \
alignment (32/16) can be stated here. They stay refusals in the loader, \
exactly as 16-alignment stays one for fp8-rowwise."""


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    quantized = {
        entry["pattern"][: -len(".weight")]
        for entry in source["tensors"]
        if entry["pattern"].endswith(".weight") and "F8_E4M3" in entry.get("dtypes", ())
    }
    if not quantized:
        raise SystemExit("the fp8 sibling declares no fp8 weights; refusing to derive nothing")

    tensors: list[dict[str, object]] = []
    for entry in source["tensors"]:
        pattern = entry["pattern"]
        if pattern.endswith(".weight_scale"):
            continue  # replaced below by the nvfp4 scale family
        base = pattern[: -len(".weight")] if pattern.endswith(".weight") else None
        if base is None or base not in quantized:
            tensors.append(dict(entry))
            continue
        role = entry["role"]
        required = entry.get("required", True)
        tensors.append(
            {"role": role, "pattern": pattern, "dtypes": ["U8"], "rank": 2,
             **({} if required else {"required": False})}
        )
        tensors.append(
            {"role": f"{role}.blockscale", "pattern": f"{base}.weight_scale",
             "dtypes": ["F8_E4M3"], "rank": 2,
             **({} if required else {"required": False})}
        )
        # rank is deliberately UNSTATED on the second-level scales: modelopt has
        # exported them both 0-dim and [1], and a rank this document cannot
        # verify is a conjunct that would refuse a real artifact for a shape it
        # never promised.
        tensors.append(
            {"role": f"{role}.scale2", "pattern": f"{base}.weight_scale_2",
             "dtypes": ["F32"],
             **({} if required else {"required": False})}
        )
        # Calibration-optional, per layer: ALWAYS `required: false`, even where
        # the weight is required. nvfp4 calibrates activations unconditionally
        # but AWQ-lite smoothing is a recipe choice, and a static input scale is
        # absent whenever the export fell back to dynamic amax.
        tensors.append(
            {"role": f"{role}.input_scale", "pattern": f"{base}.input_scale",
             "dtypes": ["F32"], "required": False}
        )
        tensors.append(
            {"role": f"{role}.pre_quant_scale", "pattern": f"{base}.pre_quant_scale",
             "dtypes": [], "rank": 1, "required": False}
        )

    document = {
        "format": "tensorfs-contract-v1",
        "name": "sdxl.diffusers-nvfp4-flat",
        "version": 1,
        # The LANE's quantization. Authored, because no tensor here spells fp4:
        # the resident weights are packed U8 pairs. gen-worker derives the sm100
        # floor from this field alone, which is why it is not `_x2`.
        "dtype": "float4_e2m1fn",
        "description": DESCRIPTION,
        "tensors": tensors,
    }
    text = json.dumps(document, indent=2, ensure_ascii=True) + "\n"

    from tensorfs.contract import Contract

    Contract.from_document(text)
    TARGET.write_text(text, encoding="utf-8")
    print(
        f"{len(quantized)} quantized Linears -> {len(tensors)} declarations "
        f"(from {len(source['tensors'])}); validated; wrote {TARGET.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
