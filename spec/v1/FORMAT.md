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

## Writer policies

`fixed-v1` is the default during rollout. It stores files through 64 MiB whole
and splits larger files at fixed 64 MiB offsets.

`tensor-aligned-v2` first proves that the file is structurally valid
safetensors. Its 8-byte prefix plus JSON header is one chunk. Tensors at least
64 MiB are split every 64 MiB from that tensor's own start; consecutive smaller
tensors are greedily packed, in physical byte order, to at most 64 MiB. The
ideal 50 GiB / 64 MiB body is 800 objects; one all-small 50 GiB run needs at
most 1,600, while large tensors use ceiling division. A 186-layout local corpus
measured 1.346x as many objects as fixed chunks
at this floor, versus 2.272x at 32 MiB. The 64 MiB floor also bounds collateral
from one changed small tensor to less than 64 MiB. Malformed and non-safetensors
files fall back to `fixed-v1`; no filename extension can bypass parsing.

Pack membership is deterministic for one ordered tensor set, not stable across
arbitrary structural edits. Adding, removing, or resizing a small tensor can
repack the rest of that consecutive small-tensor run until the next large
tensor. There is deliberately no rolling hash or content-defined resync.

Both policies write manifest format 1 and coexist without a backfill. Existing
fixed chunks stay readable; a file only gains cross-version deduplication after
it is republished with the same policy.

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
