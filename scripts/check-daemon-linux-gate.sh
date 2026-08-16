#!/usr/bin/env bash
# Every tensorfsd integration test must open with `#![cfg(target_os = "linux")]`.
#
# The daemon is Linux-only: its FUSE dependencies sit under a
# cfg(target_os = "linux") table, lib.rs gates every module, and main is a
# stub off Linux. The core-cross-platform CI job deliberately does not build
# tensorfsd, so an ungated test file would compile on a platform where no
# runner ever executes it -- the vacuous green this script exists to prevent.
set -euo pipefail

cd "$(dirname "$0")/.."

gate='#![cfg(target_os = "linux")]'
ungated=""
count=0

for test in crates/tensorfsd/tests/*.rs; do
    count=$((count + 1))
    if ! grep -qF "${gate}" "${test}"; then
        ungated="${ungated}  ${test}"$'\n'
    fi
done

if [ -n "${ungated}" ]; then
    printf 'daemon test without the Linux gate:\n%s\n' "${ungated}"
    printf 'TensorFS supports Linux/FUSE3 only (crates/tensorfsd/src/lib.rs).\n'
    printf 'Add `%s`, or, if you are landing the\n' "${gate}"
    printf 'macFUSE/WinFsp port, extend .github/workflows/ci.yaml first so the\n'
    printf 'test actually runs on a platform that can mount.\n'
    exit 1
fi

printf 'all %d daemon tests are Linux-gated\n' "${count}"
