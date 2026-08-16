# TensorFS TFM1 vectors v1

This directory is the language-neutral conformance corpus for the TFM1
canonical snapshot manifest documented in `../TFM1.md`. Rust authors the
corpus; Go and any later consumer decode the same fixture bytes and compare
against `tfm1-vectors.json`.

Fixture files contain one lowercase hexadecimal encoding of the manifest
bytes followed by one LF, exactly as in `../planner-vectors/`.

- `golden` cases decode, re-encode byte-for-byte, and hash to `snapshot_id`,
  which is by definition the SHA-256 of the fixture bytes. A strict decoder
  must accept nothing the corpus does not show it.
- `refusals` cases must be rejected with the stable kebab-case `reason`
  label. A decoder that accepts any refusal fixture is wrong; matching the
  label keeps error vocabularies aligned across languages.

The corpus is regenerated from the Rust encoder with
`TENSORFS_WRITE_TFM1_VECTORS=1 cargo test --test tfm1_vectors`, which is
legitimate only while the pre-launch format may be replaced in place. After
launch the committed bytes are the contract and regeneration is a format
change, not a refresh.
