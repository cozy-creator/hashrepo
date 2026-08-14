# hashrepo manifest v1

V1 has one canonical JSON representation. Chunk boundaries are explicit data,
not a manifest-format version axis.

## Content references

A content reference is `sha256:` followed by exactly 64 lowercase hexadecimal
characters. Readers may trim surrounding whitespace and canonicalize uppercase
hexadecimal input, but writers always emit lowercase tagged references. Bare
hex and every other algorithm are refused.

## Files

Files stored as one object use their whole-file digest and have no `chunks`
member; this representation is limited to 64 MiB. A present `chunks` member is
never null or empty. Each chunk is 1 through 64 MiB and the ordered lengths sum
exactly to `size_bytes`. Files larger than 64 MiB therefore require chunks, but
readers do not impose fixed offsets. A one-chunk list is valid, including for a
small header-only safetensors file; because that chunk is the whole file, its
digest must equal the file digest.

The manifest carries every chunk length. Readers reconstruct from the explicit
sequence and never infer offsets from a process-global chunk-size setting.
`size_bytes` must equal the sum of the chunk lengths, and `digest` is always the
SHA-256 digest of the complete reconstructed file. File sizes are signed 64-bit
integers in the inclusive range 0 through 9,223,372,036,854,775,807.

## Repository manifests

The canonical JSON object is:

```json
{"format":1,"files":[{"path":"relative/file","size_bytes":0,"digest":"sha256:..."}]}
```

Files are sorted lexicographically by Unicode code point in Python and UTF-8
byte order in Go; those orders are equivalent for valid UTF-8. Object keys and
fields use the order shown by the schema and golden vector. Encoding is UTF-8,
compact, does not HTML-escape `<`, `>`, or `&`, uses `\\u2028` and `\\u2029`,
and contains no trailing newline. Repository paths are relative forward-slash
paths; empty components, `.`, `..`, backslashes, C0/DEL control characters,
absolute paths, duplicates, ASCII-case-insensitive collisions, and unpaired
Unicode surrogate escapes are refused. The `format`, `files`, `path`,
`size_bytes`, and `digest` members are required even when their value could be
mistaken for a language's zero value; required arrays are never null.

The digest of the canonical JSON bytes is the repository-manifest content
reference.

## Writer policy

HashRepo has one automatic writer with a closed built-in planner registry.
Planner selection uses bounded file bytes, never the path extension or a caller
argument. Exactly these profile identifiers exist in v1:

- `safetensors-v1` validates the bounded JSON header, unique tensor names,
  known dtype/shape byte geometry and complete contiguous data coverage. Its
  8-byte prefix plus header is an isolated region. Every nonempty tensor is an
  independent domain; tensors through 64 MiB are one natural-sized object and
  larger tensors are split every 64 MiB relative to their own start. Zero-byte
  tensors add no data object.
- `gguf-v1` accepts little-endian GGUF v2/v3 only after validating bounded
  metadata and tensor counts/strings, unique keys/names, dimensions, the pinned
  GGML dense/quantized block-size table, power-of-two alignment of at least
  eight bytes, overflow,
  expected padded offsets and exact file coverage. Header, directory and
  padding bytes remain explicit, while every nonempty tensor uses the same
  natural-size or tensor-relative 64 MiB rule.
- `raw-fixed-64m-v1` splits every other byte stream at file-relative 64 MiB
  offsets.

Malformed, unsupported, future-format or ambiguous semantic input falls back
as a whole to raw planning. It never produces a partial semantic partition.
The generic core independently validates zero-free, gap-free, overlap-free,
ordered complete coverage, a 1,000,000-object cardinality ceiling, and hashes
the planned bytes itself. Canonical
identity never packs neighboring small tensors; batching belongs only to the
transfer protocol. Consequently insertion, deletion, reordering, resharing or
absolute-offset movement changes header/manifest bytes but does not re-key an
unchanged tensor's data objects.

The writer always emits manifest format 1. Readers reconstruct files only from
the manifest's ordered digest/length sequence and never infer writer boundaries.

## Object layouts

The library's portable object key for `sha256:abcdef...` is:

```text
objects/sha256/ab/cd/abcdef...
```

Remote adapters may map the same content reference into an existing physical
namespace such as `blobs/sha256/...`; the physical prefix is not part of the
manifest format.

## Transfer and promotion

An upload grant has exactly these fields:

```json
{"digest":"sha256:...","size_bytes":18,"staging_key":"staging/sha256/session-1/...","url":"https://objects.invalid/upload?token=v1","headers":{},"expires_at":"2026-08-13T12:10:00Z"}
```

- `digest` is the exact object content reference.
- `size_bytes` is the exact non-negative request-body length.
- `staging_key` is the server-owned, algorithm-qualified, session-scoped
  destination key `staging/sha256/<session>/<digest hex>`.
- `url` is an opaque upload URL; clients must not infer its provider.
- `headers` is an object of verbatim request headers and is `{}` when empty,
  never `null`.
- `expires_at` is the grant deadline as an RFC 3339 UTC timestamp.

The field is always named `url`, not `put_url` or a provider-specific name.
The Python executor verifies local source objects before upload and installs
downloaded bytes into `LocalCAS` only after digest and length match. Already
verified local objects are the restart journal: a new process skips them and
fetches only absent or corrupt objects. An expired grant is a typed re-plan
result, distinct from a byte failure.

Server-side publication is staged. A store adapter must attest the SHA-256 and
length of a staged PUT as part of an atomic, version-bound promotion into the
immutable content key. The destination must also carry a store-asserted SHA-256
checksum; existence and length alone are never called complete. A serialized
plan is untrusted until its manifest partition, session ID, object sizes, and
derived staging keys have all been validated. Promotion is per-object,
idempotent, and retryable. The generic promoter never deletes staging keys;
stores lifecycle-expire the session-scoped staging namespace.

## Local reachability

Logical refs are atomic roots into immutable objects. A ref may target an
ordinary object or a repository manifest; a manifest root reaches every object
listed by its files. Local collection takes additional roots and manifests
from consumers plus a mandatory age cutoff. HashRepo computes and deletes the
unreachable set, but never invents model, graph, tenant, quota, or retention
policy.
