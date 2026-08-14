# chunked-cas manifest v1

V1 has one writer and one accepted representation.

## Content references

A content reference is `sha256:` followed by exactly 64 lowercase hexadecimal
characters. Readers may trim surrounding whitespace and canonicalize uppercase
hexadecimal input, but writers always emit lowercase tagged references. Bare
hex and every other algorithm are refused.

## Files

Files of at most 64 MiB are stored as one object under their whole-file digest
and have no `chunks` member. Larger files are split from offset zero into fixed
64 MiB objects. Every non-final chunk is exactly 64 MiB. The final chunk is the
remaining non-zero length.

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
absolute paths, duplicates, and ASCII-case-insensitive collisions are refused.

The digest of the canonical JSON bytes is the repository-manifest content
reference.

## Object layouts

The library's portable object key for `sha256:abcdef...` is:

```text
objects/sha256/ab/cd/abcdef...
```

Remote adapters may map the same content reference into an existing physical
namespace such as `blobs/sha256/...`; the physical prefix is not part of the
manifest format.
