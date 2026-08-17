# The mixed-CAS layout — blobs, snapshot symlink trees, chunked tensors

Design for the vNext on-disk store, from Paul's proposal (2026-08-16):

> "wouldn't it be easier to just have a mixed-CAS system?
> 1. tensors (safetensors, gguf) get chunked + hashed and stored in CAS. this
>    allows for per-tensor de-duplication, so multiple checkpoints sharing
>    tensors can be merged easily, saving disc space.
> 2. non-tensor files (.so compiled graphs, images, videos, config.json, etc.)
>    are just regular CAS (we hash their contents as their name, treat them as
>    a blob)
> 3. we can compose repos + datasets as snapshots with symlinks"

One store, two admission shapes: **tensor containers chunk at tensor
boundaries** (per-tensor dedup across checkpoints), **everything else is one
whole blob** (whole-file dedup across datasets and repos). Snapshots are
directories of symlinks into the blob store, HF-hub-cache style, plus pointer
stubs where a file is chunked and cannot be a symlink.

## ⚠️ Supersession: "CAS scope: repos only" is retired

DESIGN-RULINGS "CAS scope: repos only" (Paul, 2026-08-16, morning) ruled that
datasets and compiled-graph `.so` artifacts **never enter CAS**. This proposal
(Paul, 2026-08-16, later the same day) **supersedes it**: they DO enter CAS —
as whole, unchunked blobs, deduplicated by content and composed into snapshot
directories by symlink. What motivated the earlier ruling — no chunking
machinery for files that don't benefit from it, plain-file read semantics for
consumers — is preserved by the blob lane itself: a blob is one ordinary file
and a consumer opens it through an ordinary path. What the earlier ruling got
wrong is that "regular file semantics" and "in CAS" were never in tension.

Still standing from the same day's rulings: verification is **admission-time**
(the read path never re-hashes), no magic size caps anywhere (PR #65 — scope
decides what enters, never a number), and the one-chunker hard cut (#64: the
Rust planner is the sole authority).

## 1. Layout

```text
<root>/
  objects/sha256/xx/yy/<64-hex>    # every CAS blob: whole files AND tensor chunks
  snapshots/<snapshot-id>/…        # projected symlink trees (disposable)
  refs/<name>                      # one snapshot id + LF, replaced by rename(2)
  tmp/                             # leased admission temps (unchanged)
  workspace metadata DB            # roots index, leases, GC state (unchanged home)
```

**One blob tree, and it is the existing one.** Paul's `/blobs` is today's
`objects/sha256/xx/yy/<hex>` (`store.rs:148`), unchanged. Whole-file blobs and
tensor chunks are the same kind of thing — verified immutable bytes at their
digest path — so a second tree would be a distinction without a difference.
The two-level fan-out stays: unlike the HF cache's flat `blobs/` (file-count
cardinality), our store holds *per-tensor chunk* cardinality — millions of
entries — and a flat directory degrades. Rename churn buys nothing; the name
`objects/` stays.

**`snapshots/<snapshot-id>/`** is the projected tree: a real directory per
manifest directory entry, a relative symlink per blob-planner file, a pointer
stub per tensor-planner file (§4). It is a **projection, never authoritative**
— derivable from the manifest at any time, deletable at any time, and pins
nothing (§6).

**`refs/<name>`** is a plain text file holding one snapshot id and a line
feed, replaced atomically by `rename(2)` — git's `.git/refs/heads/*` and the
HF cache's `refs/main`.

**This retires the symlink ref this document originally specified** (decided
while implementing #69, forced by measurement, flagged for Paul). The symlink
was chosen for two ergonomic wins over HF's hex file: `readlink` works, and
naive tooling traverses `refs/main/<file>` for free. Under 2,000 concurrent
swaps, on CI, **both wins failed**:

- **`readlink` on a name being renamed transiently returns EINVAL on APFS** —
  about 2 reads in 45,000, while the name itself was never once absent. A
  reader that `lstat`s first and `readlink`s second fails more often still:
  two lookups of a renamed name are not one observation.
- **A multi-component walk THROUGH a renamed name transiently returns
  ENOENT**, on Linux as well as macOS, reproduced locally. It never returns
  wrong bytes and the miss is self-healing, but "traverses it for free" is
  not true while anything is swapping.

`open`-then-`read` of a rename-swapped name yields the old id or the new one
on every platform and nothing else, which is why git and HF both chose it.
The supported consumer pattern is therefore the one a hex file forces anyway:
**resolve the ref once, then use `snapshots/<id>`** — a resolved tree is
immutable and a swap deletes nothing, so the path stays valid for as long as
the snapshot exists. `cd snapshots/$(cat refs/main)` replaces `cd refs/main`.

**There is no `/manifests` directory.** A TFM1 snapshot id IS the SHA-256 of
the manifest bytes, so the manifest is admitted to the blob store **at its own
id** — `objects/sha256/<id[..2]>/<id[2..4]>/<id>`. No second namespace, no
index directory; sync and serving treat manifests as objects like any other.
The **root set** stays where it is: the workspace DB `snapshots` table
(`workspace.rs:205`), which becomes an index of `(id, sealed_generation)` —
the manifest *bytes* move out of the DB into the blob store. The transactional
property the DB row exists for (`workspace.rs:701-708`: read-manifest and
take-pin in one transaction) survives as: pin the row in the transaction, then
read the immutable blob the pin protects.

## 2. TFM1: whole-blob entries

Non-tensor files stop chunking. The 64 MiB raw grid (`raw-fixed-64m-v1`) is
**retired**, planner byte and all. The grammar change:

```text
file body (kind 2) :=
  executable    u8: 0 | 1
  planner       u8: 1 safetensors-v1 | 2 gguf-v1 | 4 blob-v1
  body          per planner

planner 1, 2 (tensor containers) — unchanged:
  logical_size  u64
  record_count  u64 (<= 1,000,000)
  record * record_count             # data record length 1..=64 MiB, as today

planner 4 (blob-v1) :=
  logical_size  u64
  digest        32 bytes            # sha256 of the whole file; sha256("") when empty
```

A blob body has **no record list at all** — one file, one digest, any size.
Invalid states (a hole in a blob, a multi-record blob, a chunked config file)
are unrepresentable rather than refused. The digest of the empty string is
kept for `logical_size = 0` so every file body has the same shape.

Refusals: planner byte 3 refuses (retired, not aliased); a blob body that is
truncated, or followed by record-list framing, refuses as trailing/short input
under the existing end-of-input rule; tensor planners keep every current
refusal including the 64 MiB record cap. Ordinals are untouched — a blob
entry consumes a file ordinal exactly like a tensor entry, so hardlinks work
unchanged.

**The 64 MiB bound changes meaning.** `MAX_OBJECT_SIZE` stops being a
store-wide admission cap and becomes what it always really was: the **tensor
chunk grid constant**. `Plan::validate`'s per-region cap
(`planner/mod.rs:98`) becomes conditional on the planner; `raw_plan` and its
splitting fallback (`planner/mod.rs:192`) are replaced by a blob plan of one
region covering the file. This is not a size cap enforcing a preference — the
*scope* (is it a tensor container?) decides the shape, per PR #65's ruling.

**Vectors.** Pre-launch hard cut, no compatibility: the `tfm1-vectors` corpus
is regenerated — new goldens (a small blob entry; a blob entry with
`logical_size` > 64 MiB, which the old grammar could not express in one
record; an empty blob; a hardlink to a blob ordinal; a mixed tree of tensor +
blob entries) and new refusals (planner 3; blob body with trailing record
framing; blob digest short/truncated). Both decoders regenerate against it —
`crates/tensorfs-core/src/tfm1.rs` and the root `tfm1.go` — same discipline,
both languages, one corpus. The hub consumes the Go decoder as a module
(`tensorhub/go.mod` pins `github.com/cozy-creator/tensorfs`), so the hub's
verifier updates by re-pinning, not by re-implementing.

**A stale caveat dies with this section.** `docs/direct-tensor-reads.md` and
issue #64 both state "TFM1 identifies a file by the digest of its bytes, so
`finish()` must read every byte of the composed file once, even for inherited
tensors." That was true of the v0 JSON manifest (`FORMAT.md`, whole-file
`digest` field) and is **false of TFM1**, whose preamble says "no per-file
whole-file hash" — a tensor file's identity is its record list. Composing a
snapshot from inherited chunks therefore costs **zero** hashing of inherited
bytes, which is exactly the property the mixed-CAS dedup story wants. Blob
entries hash once, at admission, like everything else.

## 3. Wire: TFP1 unchanged; big blobs ride multipart grants

TFP1 caps pack payload at 64 MiB, and that stays. Every tensor chunk fits by
construction; small blobs (configs, most artifacts) ride packs exactly as
today. What cannot ride a pack is a multi-GB blob — a dataset video, a large
`.so` bundle — and the answer is **ranged multipart direct-R2 grants**, not
wire-level chunking:

```text
BlobUploadGrant {
  digest, length,                  # the object being staged
  upload_id,                       # R2/S3 multipart upload on the staging key
  part_size,                       # server-chosen; uniform (R2 requires equal parts, last excepted)
  parts: [(part_number, url, headers)],
}
```

The client PUTs each part to its presigned URL, then reports the part etags to
the sync session; the hub completes the multipart upload, **stream-hashes the
staged object once** against the declared digest (admission-time verification,
per the standing ruling — this is the admission), and promotes it to the
object key. Parts are transport, full stop: **nothing about them enters
identity** — TFM1 sees only the whole-blob sha256.

Rejected alternative: wire-level chunking (split the blob into ≤64 MiB wire
pieces, admit them, have the hub concatenate). It re-creates chunk identity we
just decided blobs don't have, adds a second storage shape hub-side, and pays
a full hub read+write per blob to reassemble — R2's multipart *is* the
reassembler, for free.

Downloads need nothing new: `DownloadGrant` (`sync.rs:130`) is already one
presigned GET per object with a length, agnostic to size; big blobs resume via
`Range`.

**Hub-side consequences of th#1960's landed wire, flagged not designed:** the
sync plan partitions `missing` into a pack lane (≤64 MiB) and a blob-grant
lane; `snapshotSyncMaxPackPayload` (`snapshot_sync_th1960.go`) is untouched;
staging promotion gains the stream-hash step for the blob lane; casgc's
staging sweep must learn to abort expired multipart uploads
(`AbortMultipartUpload`), which are otherwise invisibly billed.

## 4. Tensor files in a snapshot directory: pointer stubs

A chunked file has no single inode to symlink. The choice is **absent vs
pointer stub**, and the decision is a **stub**: a 0444 file named exactly the
real filename, containing one line —

```text
TFSSTUB1 {"body_sha256":"…","size":4000000000,"read":"tensorfs"}
```

**The format is pinned** in `spec/v1/TFSSTUB1.md`, with a byte corpus in
`spec/v1/tfsstub1-vectors/` that both the Rust renderer and the Python
consumer read. Key order, the single space after the magic, the absence of
whitespace and the one trailing line feed are all contract.

**The digest field is `body_sha256`, not `file_sha256`** (decided while
implementing #69, flagged for Paul). §2 above establishes that TFM1 carries
**no whole-file hash** for a tensor container — its identity IS its record
list — and the only files that have a whole-file digest, blobs, project as
symlinks and never as stubs. A field spelled `file_sha256` would therefore
either be a lie about what it holds, or force the projection to read every
tensor byte to compute one, destroying the zero-copy property this whole
layout exists for. `body_sha256` is SHA-256 over the entry's canonical file
**body** encoding — planner tag included, exactly the bytes the manifest
carries for it — so it is derivable from the manifest alone, is stable across
snapshots, and identifies the file precisely. Reverting to the original
spelling is a one-line change if Paul prefers the name over the accuracy.

Argued per consumer experience:

- **`ls` / discovery** — the tree shows its true shape. pgw's GGUF discovery
  (`gguf_local.py`: `is_dir()` + `rglob("*.gguf")`) works because stubs carry
  the real filename; absence would make every weights directory look empty and
  break layout probes far from the cause. Proven below: `rglob` finds the stub.
- **naive `open()`** — reads `TFSSTUB1 …`, which no safetensors u64 header or
  GGUF magic will parse: a **loud error at the parse site** versus absence's
  `FileNotFoundError` (misread as a corrupt snapshot) — fail fast beats fail
  mysterious. Measured against the real libraries in
  `python/tests/test_pointer_stubs.py`:

  | reader | on a stub | on absence |
  |---|---|---|
  | `safetensors.safe_open` | `SafetensorError: Error while deserializing header: header too large` | `FileNotFoundError` |
  | `gguf.GGUFReader` | `ValueError: GGUF magic invalid` | `FileNotFoundError` |

  One correction to the original claim: **neither library names the file in
  the message.** Both raise their own parse-error type, and the caller knows
  the path because it passed it — which is the distinction that matters
  (parse failure vs. `OSError`), and is what the tests assert.
- **`stat` / `du`** — the honest cost: size reads ~128 B, not 4 GB. Accepted:
  a consumer that cares about weight *bytes* is a tensor consumer and must go
  through tensorfs regardless; the stub's `size` field carries the truth for
  tooling that wants it.

Stubs are projection artifacts. The **manifest** keeps the real
tensor-planner entry; stubs are generated when the tree is projected and are
not manifest entries, so snapshot identity is untouched by their format.

## 5. Immutability

Blobs and stubs install at **0444** (HF-style), admitted via the existing
leased-temp / verify / no-clobber rename path (`store.rs`), which stays the
only writer. What each layer protects:

- **0444** stops accidents — editors, `>` redirection, `open("ab")` all
  refuse (proven below). This is the entire practical threat on a pod.
- **A determined chmod-and-edit is not defended**, same stance as git's
  `.git/objects` and the HF cache: the store's owner can always corrupt the
  store, and since verification is admission-time, a corrupted shared blob
  silently poisons every snapshot sharing it. The mitigations are that writes
  have exactly one API (workspace mutation → seal → admission), a `tensorfs
  verify` scrub exists for suspicion, and any blob can be re-fetched by digest.
  Defending further (uid separation, immutable attrs) is a deployment choice,
  not a format property.
- **Filesystems without symlinks** (Windows dev boxes): probe at store open,
  HF-style; the projection falls back to copies — local dedup lost,
  correctness kept. Pods are Linux; this is a dev nicety and gets no more
  design than this paragraph. The 0444 install is POSIX-only for the same
  reason it is a nicety there: Windows' read-only *attribute* would make the
  store's own `remove_file` — of a leased temp, and of a quarantined object in
  GC — fail, so Windows keeps inherited ACLs.
- **The executable bit is not projectable** (recorded while implementing #69).
  A blob's mode belongs to the shared object, and executability is a tree
  fact, not a content fact: the same blob may be executable in one snapshot
  and not in another, so 0555-ing the object would be wrong for its other
  referents. TFM1 keeps the bit; the projected symlink does not carry it —
  the HF cache loses it identically. The one consumer that would care,
  `dlopen` of a compiled graph, does not need `+x` on Linux; a projected
  script that must be executed is a `extract()` case.

## 6. Symlink specifics

- **Relative targets, always** — `../../objects/sha256/xx/yy/<hex>`, depth
  adjusted per entry (proven below at four levels). The store relocates as a
  unit; absolute links would pin it to a mount path.
- **`snapshots/` and `objects/` live under one root** by construction.
  Symlinks themselves cross filesystems fine; what breaks is copying a
  snapshot *tree* out of the store — the relative targets dangle. That is
  unsupported by design: exporting a file is `extract()` (§8), exporting a
  tree is not a thing this system does.
- **Trees pin nothing.** GC reachability comes from manifests (DB roots →
  manifest blob → record digests + blob digests), exactly as today — never
  from walking `snapshots/`. A symlink tree is cache, not evidence.
- **Snapshot deletion** = delete the DB root row, then `rm -rf
  snapshots/<id>`, then drop any `refs/` entry naming it — in that
  order, because a tree without a root is inert garbage while a root without a
  tree is merely unprojected. A dangling tree or dangling ref left by a crash
  between steps is garbage the next scrub removes.
- **Dangling links inside a tree** (target object GC'd) cannot occur for a
  rooted snapshot — its manifest pins its objects. Found anyway, they mean
  the tree's root is gone: the whole tree is garbage, removed as above.

## 7. Dedup accounting

`du` behaves honestly once you know which question each form answers (numbers
from the prototype, two datasets sharing one image blob):

| command | answers | prototype |
|---|---|---|
| `du -s objects/` | real bytes on disk, shared counted once | 80K |
| `du -s snapshots/` | projection overhead: dirs + stubs + link inodes | 56K |
| `du -sL snapshots/<id>` | apparent size of one snapshot, shared re-counted | 60K |

The report a `tensorfs du` command owes per snapshot: **logical** (sum of
`logical_size` — what a plain copy would occupy), **resident** (bytes of its
reachable objects), and **exclusive** (objects reachable from *only* this root
— the true "freed if deleted"). Exclusive is the one worth computing centrally
because it needs the whole root set: mark from every manifest root, count
each object's referencing roots, attribute single-root objects. GC's mark
phase already walks exactly this; the accounting is a by-product, not a second
traversal.

## 8. What each existing issue becomes

| issue | disposition |
|---|---|
| **#56 / #61** (direct tensor read/write) | **Unchanged and load-bearing.** `TensorReader`/`TensorWriter` operate on records and never see the snapshot dir; blob entries are simply not tensor containers. The `docs/direct-tensor-reads.md` API is how every tensor in §4's stubs is actually read. One correction flows back: the `inherit()` "must still hash every byte" caveat is stale under TFM1 (§2). |
| **#58** (old data plane deletion) | **Unchanged.** Already executed by PRs #62/#66; nothing here resurrects it. |
| **#64** (chunker disagreement) | **Unchanged.** The Rust planner is canonical. Strengthened, if anything: non-tensor files no longer chunk at all, so the sub-64 MiB packing question cannot arise for them. |
| **tcg#28** (compiled-graph artifacts) | **Superseded again**, including this lane's earlier comment moving artifacts to plain files *outside* the store. They are whole blobs *inside* it, appearing in snapshot trees as symlinks. Site B (`resolve`, untar-and-discard) reads through the symlink — the materialize step disappears entirely. Site A (`export_artifact`, publishes via `os.link` on the caller's device) becomes `extract()` to that device; hardlinking outsiders onto a 0444 store inode aliases canonical bytes and is refused. |
| **pgw#1295** (snapshot consumption) | The "publish" step becomes **seal + project the symlink tree**. `from_pretrained`-style config/tokenizer reads traverse symlinks as plain files; weights load natively per component. No copy anywhere. |
| **streaming materializer** (`extract`, PR #62/#65) | Survives **only** as the private-copy escape hatch (§9's audit). Not a component of the layout. |

## 9. Materialization is not a component of this system

Paul, refining the proposal:

> "the idea is that we skip materialization entirely, since it's not needed.
> Regular-CAS files are just symlinks, while tensor-files are read through
> tensorFS … we could add an escape hatch that lets you materialize tensors,
> but it's not recommended (defeats the whole purpose of this system)."

Stated as a property of the layout, with the consumer audit as evidence:

| consumer class | served by |
|---|---|
| config / tokenizer / JSON readers (`from_pretrained(dir)` non-weight half) | symlink in the tree — a plain file |
| dataset readers (images, video, audio) | symlinks — shared blobs, stored once |
| compiled-graph `.so` → `dlopen` | symlink; `dlopen` follows it |
| weight loading (safetensors, GGUF), all quantizers except the two below | native tensor reads (#56/#61) |
| `modelopt` conversion | native reads — quantizes post-load, `from_config` + state dict; once mis-classified as blocked, is not |
| GGUF discovery (`rglob("*.gguf")`) | pointer stubs carrying real filenames (§4) |
| tcg artifact export off-store | `extract()` |
| endpoint author slots reading raw weight bytes from a directory (pgw#1303) | **the hatch** — explicitly priced, or deprecated |

**Zero known mandatory materializations; one discouraged-hatch user pending a
deprecation decision.** The hatch is the existing `extract()` (PR #62,
uncapped by #65): streaming, O(one block), atomic, any size. It is
**discouraged** — "defeats the whole purpose", Paul's words — and this list of
users is the whole list; the hatch is not an invitation.

Two adjacent facts, recorded so they aren't re-litigated: bnb nf4/int8
*conversion* is orphaned (no pgw ladder class, no hub precision class names
it; deletion pending only Paul's tenant-contract check) and is therefore not a
hatch user. The serving-side bnb *compat reader* for pre-quantized third-party
uploads (`pgw loading.py:216-237`) is a keep-or-kill product decision that is
Paul's: kept, it is a hatch user for tenant bnb checkpoints; killed, the
refusal says "convert to fp8/nvfp4". That is the one open decision here.

## 10. Migration

Pre-launch, no compatibility — and the fleet claim is verified, not assumed:

- **pgw** pins `hashrepo>=0.3,<0.4` (`pyproject.toml:31`) — the frozen
  pre-rename PyPI package. It has never held a tensorfs-layout store.
- **torchcg** likewise pins `hashrepo>=0.3,<0.4` (tcg#28, verified against its
  `origin/main`).
- **The hub** imports the tensorfs Go module (pinned pseudo-version in
  `tensorhub/go.mod`) and the standing stack holds **dev-only** snapshot rows
  and R2 objects from th#1960 sync sessions — internal test data,
  re-creatable from sources.

So: local stores exist on dev boxes only — deleted and re-admitted. Hub-side:
re-pin the Go module for the new TFM1, wipe the dev snapshots, re-sync.
Nothing shipped is invalidated because nothing shipped consumes this layout.

## Prototype evidence

`proto_mixed_cas.py` (session scratchpad; pure stdlib, seconds to run) built a
real store and proved the contested mechanics — run 2026-08-16:

```text
$ python3 proto_mixed_cas.py store
dataset-a -> snapshots/8cc1fd214e0d…
dataset-b -> snapshots/b649b2bdf63c…
-- naive open() through nested symlink: b'RIFF only-in-a-v'
-- both datasets resolve images/cat.png to the SAME inode: ok
-- write through symlink refused by 0444 blob: Permission denied
-- rglob('*.safetensors') discovery over stubs: ['model.safetensors']
-- naive safetensors open reads b'TFSSTUB1': not a u64 header -> loud parse error
-- 80K  store/objects
-- 56K  store/snapshots
-- 60K  store/snapshots      (du -L)
-- deleted dataset-b (rm ref + rm -rf tree); dataset-a still reads: ok

$ ls -l snapshots/8cc1…a526/clips/train/
v.webm -> ../../../../objects/sha256/cb/cc/cbcc81fc…60d30
```

Covered: relative links resolving from nested depth, cross-snapshot inode
sharing, 0444 refusing a write through a symlink, stub discovery + loud naive
failure, du semantics, manifest-at-its-own-id, and snapshot deletion as tree
removal leaving peers intact. **Not covered** (design only, no gate claimed):
the TFM1 grammar change and its vectors, the multipart grant flow, seal-path
changes, hub-side anything, and Windows fallback behaviour.

## The filed program (hard cut A→B, per Paul's mandate)

- **tensorfs#68** — TFM1 blob-v1 entries; `raw-fixed-64m-v1` dies; Rust+Go
  vectors regenerate in the same cut.
- **tensorfs#69** — the layout: snapshot symlink trees, atomic refs,
  manifests as blobs at their own id; tree builder replaces materialization.
  **Landed.** `tensorfs_core::layout` owns the projection and refs;
  `WorkspaceStore::project_snapshot`/`delete_snapshot` own root ordering. The
  `snapshots` table is `(id, sealed_generation)` — bytes live at
  `objects/sha256/…/<id>` and a root row roots the manifest itself.
- **tensorfs#70** — pointer stubs, format pinned. **Landed.**
  `spec/v1/TFSSTUB1.md` + `spec/v1/tfsstub1-vectors/`; `layout::stub_bytes` /
  `layout::parse_stub` are the only renderer and reader, fenced by a
  magic-uniqueness test. The projection, refs and stub codecs reach Python
  through `tensorfs.native`.
- **tensorfs#71** — GC + accounting over the unified store.
- **th#2064** (tracker) — multipart direct-R2 blob grants; hub re-pins the Go
  module; TFP1 untouched.
- **pgw#1295** (tracker, rewritten a second time) — publish = tree
  construction; **pgw#1308** — the consumption cutover carrying §9's audit
  table.
- **tcg#28** — re-reframed on-thread: artifacts are blobs + symlinks inside
  the store.
