# TFM1 — the TensorFS canonical snapshot manifest

One snapshot has exactly one byte encoding. `SnapshotId` is the SHA-256 of
those bytes; there is no second Merkle identity, and nothing platform-derived
(inode numbers, timestamps, owners, randomness) can enter them. A tensor
file's identity is its record list — it has no whole-file hash; a blob's
identity IS its whole-file hash, because a blob is one object. Equal tree
facts with the same explicit parent produce the same id on every host.
Pre-launch, this format is replaced in place: there is no version field, no
v2 reader, and no compatibility alias.

All integers are fixed-width little-endian. There are no varints.

```text
manifest :=
  magic         "TFM1"                      4 bytes
  parent_flag   u8: 0 none | 1 present
  parent        32 bytes                    iff parent_flag == 1
  entry_count   u64
  entry * entry_count                       strictly ascending unique path bytes
  <end of input; any trailing byte refuses>

entry :=
  path_len      u32 (1..=4096)
  path          UTF-8, validated (below)
  kind          u8: 1 directory | 2 file | 3 symlink | 4 hardlink

file body (kind 2) :=
  executable    u8: 0 | 1
  planner       u8: 1 safetensors-v1 | 2 gguf-v1 | 4 blob-v1
  body          per planner

tensor body (planner 1, 2) :=
  logical_size  u64
  record_count  u64 (<= 1,000,000)
  record * record_count

blob body (planner 4) :=
  logical_size  u64
  digest        32 bytes    SHA-256 of the whole file; SHA-256 of the empty
                            string iff logical_size == 0

record :=
  tag           u8: 1 data | 2 hole
  data          digest 32 bytes, length u64 (1..=67,108,864)
  hole          length u64 (>= 1)

symlink body (kind 3) :=
  target_len    u32 (1..=4096)
  target        UTF-8, no control characters

hardlink body (kind 4) :=
  ordinal       u64, a previously assigned file ordinal
```

Tensor bodies: record lengths must sum exactly to `logical_size` (overflow
refuses). Zero lengths and adjacent holes refuse. A data record never exceeds
64 MiB — the tensor chunk grid constant, which bounds nothing but tensor
records; holes are unbounded. An empty tensor file has zero records.

Blob bodies: every non-tensor file is ONE unchunked blob of any size. The
body is its logical size and the SHA-256 of its whole byte content — there is
no record list, so a chunked blob, a multi-record blob, and a hole in a blob
are unrepresentable rather than refused. A zero-size blob must carry exactly
the SHA-256 of the empty string (refusal `empty-blob-digest`); no other size
is digest-checked at decode, because verification is admission-time. Planner
byte 3 — the retired `raw-fixed-64m-v1` grid — refuses as `unknown-planner`:
retired, not aliased.

Readers reconstruct solely from the body facts (ordered digest/length records
for tensor bodies, the single whole-file object for blob bodies); the planner
byte is provenance and no reader loads a planner.

Object ordinals count regular-file entries in path order, starting at zero.
A hardlink names an already-assigned ordinal, so the carrier of a link group
is always the group's first sorted path and references only point backward.
Redirecting a link or renaming any path changes the id; the referenced object
digests never move with a rename.

Paths are UTF-8, NFC-normalized, relative, `/`-separated, at most 4096 bytes,
and fail closed: no empty, `.` or `..` segments; no control characters,
backslashes, or `: * ? " < > |`; no segment trailing dot or space; no
Windows-reserved device stems (CON, PRN, AUX, NUL, COM1–9, LPT1–9, with any
extension); no two paths equal under Unicode lowercase folding. Every parent
path of every entry must be present as an explicit directory entry. Entries
sort strictly ascending by raw path bytes; the parent of a path always sorts
before it.

Symlink targets are stored bytes, not resolved paths: any non-empty UTF-8
string without control characters, at most 4096 bytes. Resolution policy and
escape handling belong to the materializer, not the format.

Unknown tags, invalid flags, out-of-range counts, noncanonical order, and
declared counts exceeding the remaining input all refuse before allocation.
Decoding then re-encoding reproduces the input byte-for-byte.

The language-neutral conformance corpus lives in `tfm1-vectors/`.
