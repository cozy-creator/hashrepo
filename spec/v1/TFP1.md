# TFP1 — the TensorFS canonical upload-staging envelope

One pack carries one or more whole missing TFM1 objects to a receiver that
stream-verifies and admits each object to its digest address. TFP1 bounds
upload request count and nothing else: it **never participates in snapshot or
file identity**, is **never retained** as a local or remote storage format,
and an object is never split across packs. Pre-launch, the format is replaced
in place: no version field, no v2 reader, no compatibility alias.

All integers are fixed-width little-endian. There are no varints.

```text
pack :=
  magic         "TFP1"                    4 bytes
  object_count  u64 (1..=1,000,000)
  row * object_count                      strictly ascending unique digests
  payload                                 objects concatenated in index order

row :=
  digest          32 bytes
  payload_offset  u64
  length          u64 (1..=67,108,864)
```

The payload is exactly the indexed objects concatenated in index order with no
gaps, so every `payload_offset` is derivable; it is committed anyway so a
streaming verifier can seek, and any mismatch refuses. Row lengths sum to at
most 64 MiB (the payload bound equals the object bound, so one maximum-size
object fills a pack); index overhead is separately bounded by the object
cardinality cap. Zero objects, zero-length objects, duplicate digests,
noncanonical index order, offset gaps or overlaps, truncated or trailing
bytes, and declared counts exceeding the remaining input all refuse before
allocation.

A receiver verifies each object's bytes against its indexed digest while
walking the index, one object at a time; a digest mismatch refuses the pack.
Encoding then decoding reproduces the input byte-for-byte; the builder sorts,
bounds, and verifies before emitting, so the canonical form is the only form
either side ever produces or accepts.

The language-neutral conformance corpus lives in `tfp1-vectors/`.
