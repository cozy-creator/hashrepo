#!/usr/bin/env bash
# Prove the Rust/pyo3 verdict and the Go verdict are ONE rule, by running both
# over the same corpus and diffing what they say.
#
# Green alone would be worthless: two implementations that both answer
# "incompatible" for everything also agree. So the run has three obligations,
# and it fails if any of them is unmet:
#
#   1. every case's answer matches the arm the corpus labels it with (both
#      halves assert this themselves, so a corpus that stopped exercising
#      `derivable` fails instead of quietly passing);
#   2. the two renderings are byte-identical;
#   3. the RED arm DIVERGES — a deliberate `fp8-rowwise` -> `dtype-cast`
#      substitution, which is tensorfs#128's real defect, must make the diff
#      fail. An instrument that cannot go red is not an instrument.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> go"
nice -n 19 go run ./scripts/verdict_parity "$ROOT" > "$WORK/go.tsv"
echo "==> python (pyo3)"
nice -n 19 python3 scripts/verdict_parity/parity.py > "$WORK/py.tsv"

echo "==> GREEN: the two implementations must agree"
diff -u "$WORK/go.tsv" "$WORK/py.tsv"
cat "$WORK/py.tsv"
echo "    $(wc -l < "$WORK/py.tsv") cases agree"

echo "==> RED: a mutated recipe must break the diff"
nice -n 19 python3 scripts/verdict_parity/parity.py --mutate > "$WORK/red.tsv"
if diff -q "$WORK/go.tsv" "$WORK/red.tsv" >/dev/null; then
  echo "REFUSED: the mutated run still matched. The comparison is not looking at" >&2
  echo "the recipe, so it could not have caught tensorfs#128's defect either." >&2
  exit 1
fi
echo "    red as required:"
diff -u "$WORK/go.tsv" "$WORK/red.tsv" | grep -E '^[+-]derivable-into-fp8' || true
echo "PARITY PROVEN"
