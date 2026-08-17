# tensorfs

`tensorfs` is a content-addressed storage and snapshot engine for ordinary
files and repositories. It provides:

- a canonical SHA-256 manifest with explicit bounded chunk lengths;
- an authoritative local CAS that works without a network or hub;
- direct per-tensor reads and writes over safetensors and GGUF, with no file
  built at any point (`docs/direct-tensor-reads.md`);
- snapshots composed from a committed one's own objects: re-keying every
  tensor, or trimming some away, costs one header object and shares every
  tensor chunk (`crates/tensorfs-core/src/compose.rs`,
  `docs/dedup-invariance.md`);
- compare-and-swap logical refs;
- Go missing-object planning and staged verification/promotion; and
- one set of v1 golden vectors consumed by both Python and Go.

The released 0.3.1 Python and Go implementations are the measured prototype
and migration source. The production v1 data plane is being hard-cut to one
Rust implementation, and the Python half of that cut has landed: the Python
chunker, grant transfer client, transfer journal and Python GC are deleted.
The Rust planner (`crates/tensorfs-core/src/planner/`) is the only chunker.
Python is a typed local client — the CAS store, the manifest model and direct
tensor reads/writes — and Go remains Tensorhub's non-authoring
verifier/promoter.

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

There is no mounted filesystem in this branch. The FUSE daemon
(`crates/tensorfsd`, Linux/FUSE3 only, with its Python client and mount
benchmarks) is **shelved on the `shelf/tensorfsd` branch** — split point
tagged `shelf/tensorfsd-split` — as a permanent, never-rebased reference.
Revival is a rewrite against then-current core, starting from issues #59 and
#50. Direct tensor reads/writes (`docs/direct-tensor-reads.md`) are the
deployment path.

`crates/tensorfs-core` — formats, planners and the storage engine — is
genuinely cross-platform and CI runs its tests on macOS and Windows in the
`core-cross-platform` job.

## Layout

```text
spec/v1/                 format documentation, JSON Schema, golden vectors
crates/tensorfs-core/     Rust canonical formats, planners and storage engine
crates/tensorfs-py/      the PyO3 extension module, `tensorfs._tensorfs`
python/src/tensorfs/     Python local CAS, direct tensor reads/writes
*.go                     Go manifest, planning, and promotion engine
```

A store on disk:

```text
<root>/objects/sha256/xx/yy/<hex>   every CAS object, 0444: blobs AND chunks
<root>/snapshots/<snapshot-id>/…    projected trees — dirs, relative symlinks
                                    into objects/, TFSSTUB1 stubs for tensors
<root>/refs/<name>                  one snapshot id + LF, swapped by rename(2)
<root>/tmp/                         leased admission temps
<root>/metadata.sqlite3             workspaces, roots index, leases, GC state
```

A manifest is an object at its own id — a TFM1 snapshot id IS the SHA-256 of
its bytes — so there is no manifest namespace. Trees are projections: zero
bytes copied, derivable from the manifest, disposable, and they pin nothing.
`docs/mixed-cas-layout.md` is the design.

## Tensor layout contracts

`spec/v1/contracts/` holds versioned JSON documents describing how a
checkpoint family is spelled on disk: tensor patterns, the fusion seams inside
fused tensors, and named removable tensor sets. They are DATA, not planner
code.

At ingestion a file is identified from its **header alone** — names, shapes,
dtypes; no tensor byte is read — against the registry, with a total tie-break
(most specific, then highest version, then name). The winning contract cuts
fused tensors at their seams *before* the 64 MiB grid, so a fused packaging
and its split twin share every data object, and the snapshot records
`contract@version` so identity stays self-describing.

Seams may be interleaved (`fusion.groups`): MiniMax-H3 fuses qkv head-major,
and its two packagings still share every attention byte. Byte ORDER never
moves — only cut points do. `docs/dedup-invariance.md` §4 is the design, and
`spec/v1/contracts/README.md` the format.

## The Python distribution

`tensorfs` is **one** PyPI distribution carrying two things:

| in the wheel | what it is |
| --- | --- |
| `tensorfs/*.py` | the pure-Python facade — `LocalCAS` and tensor reads/writes |
| `tensorfs/_tensorfs.abi3.so` | the compiled Rust extension, imported in-process |

Every platform ships exactly the same two halves. This is HuggingFace's shape
with the split removed: `huggingface_hub` is a pure `py3-none-any` facade and
`hf_xet` is a separate distribution of compiled wheels. Here both halves are
the same distribution, so there is no pure-Python wheel to fall back to. The
**sdist** is that fallback: a platform outside the wheel matrix builds from
source given a Rust toolchain.

**There is no daemon in the wheel, and none in this branch.** Pods cannot
mount FUSE at all — opening `/dev/fuse` is denied by the device cgroup even
for root, `CAP_SYS_ADMIN` is absent from the container's bounding set, and
there is no API field to grant either. The wheel ships native reads;
`tensorfsd` lives on the `shelf/tensorfsd` branch (tag
`shelf/tensorfsd-split`).

Import the compiled surface from `tensorfs.native`; `tensorfs._tensorfs` is
private. It exposes the CAS (`ObjectStore`), the TFM1/TFP1 decoders, and
`RecordsReader` — a committed file's records read as a random-access byte
source, which is how one tensor is read without materializing its file.

The wheels are **abi3** (`abi3-py311`): one wheel per platform for every
CPython from 3.11 up, rather than one per interpreter version. The floor is
3.11 because CPython added the buffer protocol — `Py_buffer`,
`Py_bf_getbuffer`, `PyMemoryView_FromBuffer` — to the stable ABI in 3.11, and
a zero-copy `memoryview` over a CAS object has to be expressible. Free-threaded
CPython is **not** covered by abi3 and builds from the sdist; a stable
free-threaded ABI exists but its floor is CPython 3.15.

The extension links `tensorfs-core` with `--no-default-features --features
store`: 43 crates rather than 259, with no HTTP client, TLS stack or embedded
SQL engine in a wheel that calls none of them.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy python/src
go test ./...
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --exclude tensorfs-py -- -D warnings
cargo clippy -p tensorfs-py --lib -- -D warnings
cargo test --workspace --all-features --exclude tensorfs-py
```

`tensorfs-py` is excluded from `cargo test` and from `--all-targets` clippy
because it enables `pyo3/extension-module`, which leaves CPython symbols for
the loading interpreter to resolve — correct for a cdylib and impossible to
link into a test executable. Its gate is `python/tests/test_native.py`, which
drives the real built extension.

`uv sync` compiles the extension, so a checkout now needs a Rust toolchain.

The two test suites both read `spec/v1/vectors/manifest.json` and require their
canonical encoders to reproduce it byte-for-byte.

## Reading tensors without a file

Whole-file materialization is gone: `LocalCAS.materialize` and
`materialize_repository` are deleted, with no fallback. A consumer reads the
tensors it wants straight out of the CAS objects the seal planner already
created for them, and writes converted tensors back the same way.

```python
from tensorfs import TensorWriter, open_tensors

with open_tensors(cas, snapshot_ref) as tensors:
    view = tensors["denoiser.blocks.0.attn.weight"]
    view.dtype, view.shape, view.block      # safetensors and GGUF alike
    for piece in view.pieces():             # zero-copy, one object at a time
        ...
```

No daemon, no mount, no `/dev/fuse`. See `docs/direct-tensor-reads.md` for the
API, the torch boundary, the measurements and what remains unproven.

## Semantic writer profiles

The production v1 writer automatically selects one built-in profile from
bounded file bytes. Callers cannot select or supply a planner or hash a
caller-authored partition; one `plan_and_hash` operation owns both steps:

- `safetensors-v1` isolates the header and makes every nonempty tensor an
  independent object domain;
- `gguf-v1` validates the little-endian GGUF v2/v3 directory, alignment,
  bounded metadata values and pinned GGML dense/quantized type geometry before
  applying the same tensor rule; and
- `blob-v1` is every other byte stream — unrecognized, unsupported or
  malformed — as ONE whole unchunked blob of any size, named by its own
  SHA-256.

Tensor-planned objects are at most 64 MiB (the tensor chunk grid constant)
and a plan has at most 1,000,000 objects; a blob plan is exactly one object
of any size. A semantic tensor at most 64 MiB is one natural
object; a larger tensor is split every 64 MiB from its own start. There is no
canonical packing of neighboring small tensors. Transport may batch small
objects, but insertion, deletion, ordering, sharding and absolute file offsets
never become part of an unchanged tensor object's digest. Readers remain
format-blind and reconstruct solely from ordered digest/length records.

### The retired Python chunker

Repo versions through PR #62 carried a second chunker in Python
(`LocalCAS.ingest_file`/`ingest_repository`), whose greedy packing of
consecutive small tensors disagreed with the Rust planner's
one-object-per-tensor grid (issue #64) — packed tensors own no digest and
cannot be inherited by `TensorWriter`. It is deleted; the Rust planner is the
only chunker, and the frozen PyPI `hashrepo` 0.3.1 snapshot is its historical
record. Readers stay format-blind: any wire-legal grid, packed grids from old
snapshots included, reconstructs from the manifest's ordered lengths.

Every consumer must reconstruct from the manifest's `(digest, len)` sequence.
`chunk_size_bytes` and `MAX_CHUNK_SIZE` are per-object ceilings, never exact
chunk lengths or a basis for inferring object count.

## Releasing

Package releases use SemVer beginning at `0.1.0`; protocol, manifest, and
local-ref formats independently remain v1. Before launch, format v1 may be
broken in place: no v2 or compatibility reader is added beside it.

For a release, merge the reviewed release commit to `main`, tag that exact
commit with `v` followed by the version in `pyproject.toml`, and push the tag.
The `Publish to PyPI` workflow reruns the Python, Go and Rust gates, then calls
`wheels.yaml` to build six platform wheels plus an sdist on native runners
for every architecture — manylinux and musllinux on x86_64 and aarch64, and
macOS on both architectures, with no QEMU anywhere. There is no Windows wheel:
`local.py` locks with `fcntl`, so `import tensorfs` cannot succeed there. The
Rust half builds on Windows fine; the Python half is what needs a real
POSIX-lock replacement first, and until it has one the import raises an
`ImportError` naming the supported platforms rather than a bare
`No module named 'fcntl'`. Each
wheel is installed into clean venvs on **CPython 3.11 and 3.13** — the abi3
floor and the newest supported release, because one wheel serving the whole
range is a claim, and installing it only on the version it was built against
tests none of it — and must pass `mypy --strict` across the extension
boundary, a `LocalCAS` round trip, a native `ObjectStore` plus
`RecordsReader` round trip, and a **named tensor read out of a committed
snapshot**: an import smoke does not prove the extension works. The sdist is
separately built and installed from source and used. Publication then goes through PyPI
Trusted Publishing and the exact version endpoint is verified, including that
every published wheel is abi3 and every promised platform tag arrived. Tags
whose name does not match `pyproject.toml`, or whose commit is not on `main`,
are refused.

`wheels.yaml` also runs on `workflow_dispatch` and on any pull request that
touches the build, so the matrix cannot rot between releases. A release tag is
much too late to learn that a runner label moved.

No PyPI token is stored in GitHub. The repository's `pypi` environment and the
PyPI publisher must both identify `.github/workflows/publish.yaml`.

## License

MIT
