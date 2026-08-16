# tensorfs

`tensorfs` is a content-addressed storage and snapshot engine for ordinary
files and repositories. It provides:

- a canonical SHA-256 manifest with explicit bounded chunk lengths;
- an authoritative local CAS that works without a network or hub;
- atomic file/tree materialization and compare-and-swap logical refs;
- durable transfer-session journals and reachability-driven local collection;
- opaque grant-driven Python upload/download with verified restart resume;
- Go missing-object planning and staged verification/promotion; and
- one set of v1 golden vectors consumed by both Python and Go.

The released 0.3.1 Python and Go implementations are the measured prototype
and migration source. The production v1 data plane is now being hard-cut to
one Rust implementation. Python becomes a typed local control client, Go
remains Tensorhub's non-authoring verifier/promoter, and neither retains a
second chunker, local CAS, filesystem, or materializer.

Importing the local CAS does not load the HTTP transport. Transfer exports are
resolved lazily, so offline/local-only use has no network-stack side effect.

## Status

This repository is the pre-launch v1 extraction from Cozy Creator's existing
model-repository CAS. The Python and Go packages are public; their intentionally
narrow API and pre-launch format may still hard-cut before 1.0. The native Rust
workspace is under active construction and is not a released filesystem yet.

The supported v1 shape is intentionally narrow:

- SHA-256 only;
- one closed automatic planner registry with safetensors, GGUF and raw fallback;
- local storage and opaque remote grants;
- Linux/POSIX durability semantics; and
- no Xet, OCI, plugin, or self-hostable-server compatibility layer.

### Platform support

The mounted filesystem is **Linux/FUSE3 only**. macOS (macFUSE/FSKit) and
Windows (WinFsp) are unimplemented, not merely untested: `crates/tensorfsd`
compiles to an empty shell off Linux and its `tensorfsd` binary exits
non-zero. The Windows named-pipe control plane is a seam, not a feature.

`crates/tensorfs-core` — formats, planners and the storage engine — is
genuinely cross-platform and CI runs its tests on macOS and Windows in the
`core-cross-platform` job. That job builds `tensorfs-core` alone, so no green
check on this repository should be read as macOS or Windows mount coverage.

## Layout

```text
spec/v1/                 format documentation, JSON Schema, golden vectors
crates/tensorfs-core/     Rust canonical formats, planners and storage engine
crates/tensorfsd/        the mount daemon and control plane (Linux only)
python/src/tensorfs/     Python local CAS and grant-transfer data plane
*.go                     Go manifest, planning, and promotion engine
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy python/src
go test ./...
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

The two test suites both read `spec/v1/vectors/manifest.json` and require their
canonical encoders to reproduce it byte-for-byte.

## Semantic writer profiles

The production v1 writer automatically selects one built-in profile from
bounded file bytes. Callers cannot select or supply a planner or hash a
caller-authored partition; one `plan_and_hash` operation owns both steps:

- `safetensors-v1` isolates the header and makes every nonempty tensor an
  independent object domain;
- `gguf-v1` validates the little-endian GGUF v2/v3 directory, alignment,
  bounded metadata values and pinned GGML dense/quantized type geometry before
  applying the same tensor rule; and
- `raw-fixed-64m-v1` is the whole-file fallback for every unrecognized,
  unsupported or malformed byte stream.

Objects are at most 64 MiB and a plan has at most 1,000,000 objects. A semantic
tensor at most that size is one natural
object; a larger tensor is split every 64 MiB from its own start. There is no
canonical packing of neighboring small tensors. Transport may batch small
objects, but insertion, deletion, ordering, sharding and absolute file offsets
never become part of an unchanged tensor object's digest. Readers remain
format-blind and reconstruct solely from ordered digest/length records.

### Released 0.3.1 evidence

`LocalCAS.ingest_file` and `ingest_repository` automatically isolate a valid
safetensors header region into one or more bounded chunks,
anchors 64 MiB chunks at each large tensor, and packs consecutive small tensors
up to 64 MiB. Files that do not pass the bounded structural parser silently use
bounded fixed 64 MiB offsets. The manifest remains format 1 and readers use its
ordered lengths rather than inferring boundaries.
The 64 MiB choice measured 3,226 tensor-aligned objects versus 2,397 fixed
objects (1.346x) over 186 unique local layouts; smaller 32/16/4 MiB floors cost
2.272x/3.378x/7.988x. A perfectly filled 50 GiB body is 800 objects, while one
all-small 50 GiB run needs at most 1,600 body objects plus the header.

This improves reuse for unchanged or frozen tensors, duplicate uploads,
partial-fine-tune and LoRA checkpoint series, and structurally identical model
variants with the same ordered tensor names and sizes. It does not help a full
fine-tune where every tensor changes, and it does not reduce the first cold
download; binding and file selection are outside TensorFS chunking. Adding,
removing, or resizing a small tensor can repack the remainder of its consecutive
small-tensor run up to the next large-tensor boundary. This deterministic greedy
policy is not content-defined chunking and has no rolling-hash resynchronization.
te#185 phase 4 measures real stored-byte and object-count deltas over a 25-step
frozen-base LoRA series; it is measurement, not a runtime selector or rollout gate.

For an opt-in, real-scale local measurement, run:

```bash
uv run python python/benchmarks/safetensors_dedup.py
```

The default builds a 1 GiB, 16-tensor parent, changes eight bytes in one tensor,
ingests both into one local CAS, and reports retained/reused bytes, wall/CPU
time, throughput, peak RSS, filesystem I/O blocks, and verified materialization.
It needs about 4 GiB of temporary disk. Timing is deliberately not a shared-CI
gate; compare runs on the same idle machine and filesystem.

Package `0.2.0` introduced that measured planner and `MAX_CHUNK_SIZE`. Package
`0.3.0` removed the transitional public writer selector. The native hard cut
keeps its proven header/tensor validation vectors but deliberately removes its
greedy-small-run weakness rather than preserving it as a compatibility mode.

Every consumer must reconstruct from the manifest's `(digest, len)` sequence.
`chunk_size_bytes` and `MAX_CHUNK_SIZE` are per-object ceilings, never exact
chunk lengths or a basis for inferring object count.

## Releasing

Package releases use SemVer beginning at `0.1.0`; protocol, manifest, journal,
and local-ref formats independently remain v1. Before launch, format v1 may be
broken in place: no v2 or compatibility reader is added beside it.

For a release, merge the reviewed release commit to `main`, tag that exact
commit with `v` followed by the version in `pyproject.toml`, and push the tag.
The `Publish to PyPI` workflow
reruns the Python and Go gates, builds and smoke-tests the wheel, publishes the
tested wheel and sdist through PyPI Trusted Publishing, and verifies the exact
version endpoint. Tags whose name does not match `pyproject.toml`, or whose
commit is not on `main`, are refused.

No PyPI token is stored in GitHub. The repository's `pypi` environment and the
PyPI publisher must both identify `.github/workflows/publish.yaml`.

## License

MIT
