# TensorFS TFSSTUB1 vectors v1

The language-neutral corpus for the pointer stub documented in
`../TFSSTUB1.md`. Rust authors it; the Python suite reads the same fixture
files and compares the same bytes, because a stub is the one projection
artifact a foreign consumer parses.

Fixtures are the stub's **raw ASCII**, one line each including the trailing
LF — not a hex encoding, so `cat` shows the contract. Each `golden` case
records the `body_sha256` and `size` the stub carries; a renderer must
reproduce the fixture byte for byte from that pair, and a parser must recover
the pair from the fixture.

Every case has a distinct digest and a distinct size, so no assertion can pass
by matching the wrong row. `max-u64-size` is deliberate: `size` is a JSON
number that a consumer must read as an integer, never as a float.

`.gitattributes` marks the fixtures `-text` so git never translates their line
endings. That is not hygiene: the Windows runner rewrote a fixture's trailing
LF to CRLF on checkout, and the trailing LF is part of the format.

Regenerate with `TENSORFS_WRITE_TFSSTUB1_VECTORS=1 cargo test --test
tfsstub1_vectors`, legitimate only while the pre-launch format may be replaced
in place. After launch the committed bytes are the contract.
