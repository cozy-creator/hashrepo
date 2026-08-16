# TensorFS planner vectors v1

This directory is the language-neutral conformance corpus for TensorFS's
closed automatic planner registry. Rust, Go, and Python consumers decode the
same fixture sources and compare the automatic result with the exact ordered
regions and SHA-256 object digests in `planner-vectors.json`.

Fixture files contain one lowercase hexadecimal encoding followed by one LF.
The decoded bytes are the source prefix. A case's optional `zero_tail` appends
that exact number of zero bytes, which compactly preserves the released sparse
FP8 samples without committing tens of megabytes of zeros. Empty input is
represented by a file containing only that LF. Whitespace, uppercase
hexadecimal, prefixes, and odd digit counts are not part of the encoding.

The case classifications are deliberately narrow:

- `semantic` means a valid safetensors or GGUF source selected its built-in
  semantic planner.
- `raw` means ordinary input selected `raw-fixed-64m-v1` without resembling a
  supported semantic format.
- `fallback` means a malformed or unsupported safetensors/GGUF candidate fell
  back atomically to `raw-fixed-64m-v1`.

`fallback` is not a refusal. Every stable in-memory byte stream has an
automatic plan; format parse failure chooses the raw plan. Source I/O/change
errors cannot be represented by immutable fixture bytes and remain native
unit tests. Refusal of a forged planner claim or alternate partition belongs
to the later HRM1/Hub verifier specified by pgw#1259 and th#1960. This corpus
does not invent that verifier or its error vocabulary.

The two sparse FP8 cases reproduce the shapes and byte lengths in the released
`python/tests/testdata/safetensors_header_samples.json`; their source bytes are
the exact header prefix followed by the declared zero tail. Other large
synthetic boundary cases stay in Rust's planner tests. The corpus has no
model-specific fixture generator whose behavior could become a second wire
contract.
