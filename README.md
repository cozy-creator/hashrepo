# hashrepo

`hashrepo` is a small content-addressed storage library for immutable files
and repositories. It provides:

- a canonical SHA-256 manifest format with fixed 64 MiB chunks;
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
- one fixed 64 MiB writer;
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
