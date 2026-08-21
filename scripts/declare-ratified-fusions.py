#!/usr/bin/env python3
"""The one part of a ratification the GENERATOR cannot derive: `fusion`.

A fused axis is invisible in a header — `[3*d, d]` and `[3d, d]` are the same
numbers — so `tensorfs.generate` refuses to guess one and names it in the
RATIFICATION OWED list instead. A human answers it by reading the module
definition, and the answer belongs in the document.

This script is where those answers live for the tensorfs#130 set, so
`generate-the-130-set.sh` still reproduces exactly what is committed instead of
the derivation and the library drifting apart. Every entry below cites the line
that PROVES it, and the shares were checked twice: against the module's own
`nn.Linear` widths and against the real header's shape.

    scripts/declare-ratified-fusions.py spec/v1/contracts/<document>.json

Passing no path applies every entry. A pattern named here that the document does
not declare is a REFUSAL, not a warning — a fusion silently applied to nothing
is exactly the ~90%-error split this file exists to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: q|k|v in equal thirds along the outer axis — the plain fused-qkv shape
#: `sdxl.clip-g-fused-qkv@1` already declares for OpenCLIP.
THIRDS = {
    "axis": 0,
    "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1}, {"role": "v", "share": 1}],
}

FUSIONS: dict[str, dict[str, dict[str, object]]] = {
    # SigLIP's attention-pooling head is a `torch.nn.MultiheadAttention`
    # (transformers/models/siglip/modeling_siglip.py:626-635), whose
    # `in_proj_weight` is torch's own fused q|k|v. Header agrees: [3456, 1152],
    # hidden 1152.
    "joycaption.llava-bf16": {
        "vision_tower.vision_model.head.attention.in_proj_weight": THIRDS,
        "vision_tower.vision_model.head.attention.in_proj_bias": THIRDS,
    },
    # transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py throughout.
    "qwen3.6-35b-a3b.vllm-fp8": {
        # :989 `nn.Linear(dim, dim * 3)`, split by :1005-1007's reshape to
        # (seq, 3, num_heads, -1). [3456, 1152] = 3 x 1152.
        "model.visual.blocks.{i}.attn.qkv.weight": THIRDS,
        "model.visual.blocks.{i}.attn.qkv.bias": THIRDS,
        # :420 `nn.Linear(hidden, key_dim * 2 + value_dim)`, split at :489-494
        # as [key_dim, key_dim, value_dim]. 16x128 | 16x128 | 32x128 =
        # 2048|2048|4096 over [8192, 2048] — 1:1:2, and 8192/3 is not an integer.
        "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight": {
            "axis": 0,
            "parts": [
                {"role": "q", "share": 1},
                {"role": "k", "share": 1},
                {"role": "v", "share": 2},
            ],
        },
        # :644-646 `nn.Linear(hidden, num_attention_heads * head_dim * 2)`,
        # split at :672-675 by viewing (..., -1, head_dim*2) and chunking the
        # LAST axis — so q and an output GATE, interleaved HEAD-MAJOR. 16 groups
        # of (q, gate), 256 rows each, over [8192, 2048]. A flat two-way split
        # is right for head 0 and wrong for the other fifteen, silently.
        "model.language_model.layers.{i}.self_attn.q_proj.weight": {
            "axis": 0,
            "groups": 16,
            "parts": [{"role": "q", "share": 1}, {"role": "gate", "share": 1}],
        },
        "mtp.layers.0.self_attn.q_proj.weight": {
            "axis": 0,
            "groups": 16,
            "parts": [{"role": "q", "share": 1}, {"role": "gate", "share": 1}],
        },
    },
}


def apply(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    name = str(document.get("name", ""))
    wanted = FUSIONS.get(name)
    if not wanted:
        return 0
    seen = set()
    for declaration in document["tensors"]:
        fusion = wanted.get(str(declaration["pattern"]))
        if fusion is not None:
            declaration["fusion"] = fusion
            seen.add(str(declaration["pattern"]))
    missing = sorted(set(wanted) - seen)
    if missing:
        raise SystemExit(
            f"{path.name}: declares no tensor for {missing} — a fusion that lands on "
            "nothing is the silent half of the failure it exists to prevent"
        )
    path.write_text(json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return len(seen)


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "spec/v1/contracts"
    paths = [Path(a) for a in sys.argv[1:]] or [root / f"{n}.v1.json" for n in FUSIONS]
    for path in paths:
        count = apply(path)
        if count:
            print(f"  {path.name}: {count} declared fusion(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
