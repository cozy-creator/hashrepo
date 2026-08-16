# TFP1 conformance vectors

Language-neutral golden and refusal fixtures for the TFP1 upload-staging
envelope (`../TFP1.md`). Every consumer must decode each golden fixture,
verify every indexed object digest, and reproduce the exact bytes through its
canonical encoder where it has one; every refusal fixture must refuse with the
indexed kebab-case reason.

Fixtures are one lowercase-hex line terminated by a line feed
(`lowercase-hex-lf-v1`; a CRLF-normalizing checkout is tolerated by readers).
`fixture_sha256` pins each golden fixture's bytes against drift — it is a
corpus fact, never an identity: TFP1 packs carry no identity of any kind.

`generated_golden` cases are goldens whose bytes are built, not read: each
carries a recipe (`objects[].ramp_seed`, `objects[].length`, where object byte
`i` is `(ramp_seed + i) mod 256`) and `pack_sha256`, the envelope's own digest.
Every consumer generates the bytes, must land on that pin, and then owes the
pack the same decode / verify / re-encode contract as a committed fixture. This
exists for the cap boundary: a 64 MiB pack is 128 MiB as committed hex, which
does not belong in git — and because Go has no encoder, its assembler and
Rust's agreeing on the pin is two independent constructions meeting, not one
echoing the other. A ramp rather than a constant fill so that a shifted or
reordered payload cannot reach the same digest.

The Rust encoder is the corpus author:
`TENSORFS_WRITE_TFP1_VECTORS=1 cargo test --test tfp1_vectors` regenerates the
directory in place. The `object-too-large` and `payload-too-large` refusal
fixtures stay small because those bounds refuse on declared lengths before any
payload is read.
