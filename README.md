# hashrepo

`hashrepo` is a small content-addressed storage library for immutable files
and repositories. It provides:

- a canonical SHA-256 manifest with explicit bounded chunk lengths;
- an authoritative local CAS that works without a network or hub;
- atomic file/tree materialization and compare-and-swap logical refs;
- durable transfer-session journals and reachability-driven local collection;
- opaque grant-driven Python upload/download with verified restart resume;
- Go missing-object planning and staged verification/promotion; and
- one set of v1 golden vectors consumed by both Python and Go.

The Python and Go implementations are native. They share a format and
conformance corpus, not a C ABI. Python's hashing uses OpenSSL through
`hashlib`, while its filesystem and network operations release the GIL. This
keeps installation and debugging simple without putting a second runtime and
cgo boundary inside Python processes.

Importing the local CAS does not load the HTTP transport. Transfer exports are
resolved lazily, so offline/local-only use has no network-stack side effect.

## Status

This repository is the pre-release v1 extraction from Cozy Creator's existing
model-repository CAS. The initial private staging repository will become public
only after the extraction's package and license review.

The supported v1 shape is intentionally narrow:

- SHA-256 only;
- a default `fixed-v1` writer and an opt-in `tensor-aligned-v2` writer;
- local storage and opaque remote grants;
- Linux/POSIX durability semantics; and
- no Xet, OCI, plugin, or self-hostable-server compatibility layer.

## Layout

```text
spec/v1/                 format documentation, JSON Schema, golden vectors
python/src/hashrepo/     Python local CAS and grant-transfer data plane
*.go                     Go manifest, planning, and promotion engine
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy python/src
go test ./...
```

The two test suites both read `spec/v1/vectors/manifest.json` and require their
canonical encoders to reproduce it byte-for-byte.

## Tensor-aligned writer rollout

Pass `writer_policy="tensor-aligned-v2"` to `LocalCAS.ingest_file` or
`ingest_repository` to opt in. The policy isolates the safetensors header,
anchors 64 MiB chunks at each large tensor, and packs consecutive small tensors
up to 64 MiB. Files that do not pass the bounded structural parser silently use
`fixed-v1`. The manifest remains format 1 and readers use its ordered lengths,
so fixed and tensor-aligned objects can coexist without rechunking old data.
The 64 MiB choice measured 3,226 tensor-aligned objects versus 2,397 fixed
objects (1.346x) over 186 unique local layouts; smaller 32/16/4 MiB floors cost
2.272x/3.378x/7.988x. A perfectly filled 50 GiB body is 800 objects, while one
all-small 50 GiB run needs at most 1,600 body objects plus the header.

This improves reuse for unchanged or frozen tensors, duplicate uploads,
partial-fine-tune and LoRA checkpoint series, and structurally identical model
variants with the same ordered tensor names and sizes. It does not help a full
fine-tune where every tensor changes, and it does not reduce the first cold
download; binding and file selection are outside HashRepo chunking. Adding,
removing, or resizing a small tensor can repack the remainder of its consecutive
small-tensor run up to the next large-tensor boundary. This deterministic greedy
policy is not content-defined chunking and has no rolling-hash resynchronization.
`fixed-v1` remains the default until te#185 phase 4 measures real stored-byte and
object-count deltas over a 25-step frozen-base LoRA series.

The public writer-policy API and `MAX_CHUNK_SIZE` bound first ship in package
version `0.2.0`; the `0.1.x` package exposed only fixed-offset writing through
the now-removed `CHUNK_SIZE` name.

Do not enable the new policy in a worker until every consumer reconstructs from
the manifest's `(digest, len)` sequence. In particular, integrations that still
compare a `chunk_size_bytes` scalar with the old `CHUNK_SIZE` constant must hard
cut to explicit lengths first; `MAX_CHUNK_SIZE` is only a per-object bound.

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
