#!/usr/bin/env bash
# The conversion producer, driven end to end across both languages.
#
#   python  writes real safetensors trees, converts them, emits their headers
#   go      takes the verdict with the SAME matcher tensorhub's bind gate calls
#
# Two processes on purpose. The producer needs a tensor kernel, the verdict
# lives where the hub consumes it, and grading the producer with a Python copy
# of the verdict would hide exactly the failure that costs money: a converted
# checkpoint the producer believes is fine and the gate refuses.
#
# Load discipline: this box is shared. Niced, and it refuses to start while the
# box is already saturated.
set -euo pipefail
cd "$(dirname "$0")/.."

LOAD_CEILING="${LOAD_CEILING:-16}"
load() { cut -d' ' -f1 /proc/loadavg; }
if [ "$(printf '%.0f' "$(load)")" -gt "$LOAD_CEILING" ]; then
  echo "load $(load) is above ${LOAD_CEILING}; refusing to build on a saturated shared box" >&2
  exit 1
fi

PROOF="${TENSORFS_CONVERSION_PROOF_DIR:-$(mktemp -d)}"
export TENSORFS_CONVERSION_PROOF_DIR="$PROOF"
echo "==> proof fixture: $PROOF"

echo "==> producer (python): plan, convert, emit headers"
nice -n 19 uv run pytest python/tests/test_conversion.py -q

echo "==> verdict (go): the matcher the hub calls, over the produced bytes"
nice -n 19 go test -run 'TestTheConversion|TestThePythonClaimRule' -v ./... 2>&1 | grep -vE '^(=== RUN|--- PASS|ok  |\?   )' || true
nice -n 19 go test -run 'TestTheConversion|TestThePythonClaimRule' ./...

echo "==> proven"
