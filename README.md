# chunked-cas

`chunked-cas` is a small content-addressed storage library for immutable files
and repositories. It provides:

- a canonical SHA-256 manifest format with fixed 64 MiB chunks;
- an authoritative local CAS that works without a network or hub;
- atomic materialization and compare-and-swap logical refs;
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
python/src/chunked_cas/  Python local CAS and grant-transfer data plane
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

## License

MIT
