#!/usr/bin/env python3
"""The PYTHON half of the verdict parity proof — through the pyo3 binding.

Same corpus, same rendering, one line per case. The shell driver diffs this
against the Go half.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from tensorfs import _tensorfs  # noqa: E402


def document_for(stamp: str) -> str:
    name, _, version = stamp.rpartition("@")
    return (ROOT / "spec" / "v1" / "contracts" / f"{name}.v{version}.json").read_text(
        encoding="utf-8"
    )


def main() -> int:
    corpus = json.loads(
        (ROOT / "scripts" / "verdict-fixtures.json").read_text(encoding="utf-8")
    )
    mutate = "--mutate" in sys.argv
    for case in corpus["cases"]:
        document = document_for(case["contract"])
        files = [
            (
                member["path"],
                [
                    (
                        tensor["name"],
                        tensor["dtype"],
                        tensor["shape"],
                        tensor.get("length", 0),
                    )
                    for tensor in member["tensors"]
                ],
            )
            for member in case["files"]
        ]
        verdict = _tensorfs.contract_verdict(document, files)
        if verdict.kind != case["for"]:
            print(
                f"FIXTURE DRIFT: {case['name']} is labelled {case['for']!r} but answered "
                f"{verdict.kind!r} - the corpus no longer exercises the arm it claims to",
                file=sys.stderr,
            )
            return 2
        text = str(verdict)
        if mutate and case["name"] == "derivable-into-fp8-names-the-quant-recipe":
            # THE RED ARM. This is the exact defect tensorfs#128 found in the Go
            # implementation: a verdict that named the remedy `dtype-cast` for an
            # fp8 lane, which would have enqueued a plain cast and produced fp8
            # bytes with no per-row scales - every name, dtype and shape correct
            # and every number wrong. If the comparison below cannot see this
            # substitution, it could not have seen that bug either.
            text = text.replace("via fp8-rowwise", "via dtype-cast")
        print(f"{case['name']}\t{_tensorfs.contract_recipe(document)}\t{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
