# TFSSTUB1 — the pointer stub

A chunked tensor file has no single inode to symlink, so a projected snapshot
tree carries a **pointer stub** in its place: a file named exactly the real
filename, mode 0444, containing one line.

```text
TFSSTUB1 {"body_sha256":"<64 lowercase hex>","size":<u64>,"read":"tensorfs"}\n
```

Exactly:

```text
stub := "TFSSTUB1" SP json LF
json := {"body_sha256":"<hex>","size":<decimal>,"read":"tensorfs"}
```

- The magic is the first **8 bytes**, so a tool classifies a stub without
  parsing JSON. It is neither a safetensors `u64` header length nor the GGUF
  magic, which is the point: a naive `open()` fails loudly at the parse site.
- The JSON is emitted by hand, not by a serializer: keys in this order, no
  whitespace, one trailing line feed. The bytes are the contract
  (`spec/v1/tfsstub1-vectors/`), so a Python consumer may compare them
  literally.
- `size` is the file's **logical size in bytes** — the truth `stat` cannot
  tell you, because `stat` sees ~128 B of stub.
- `read` names the reader that can serve the bytes. Always `"tensorfs"`.

## `body_sha256`

SHA-256 over the entry's canonical TFM1 **file body** encoding — the planner
tag followed by the body per planner, exactly the bytes `TFM1.md` specifies
for that entry.

It is not the SHA-256 of the file's content, and the field is not named
`file_sha256` for that reason. TFM1 deliberately gives a tensor container **no
whole-file hash**: its identity is its record list. The only entries that own
a whole-file digest are blobs, and a blob projects as a symlink, never as a
stub. A field holding a real file digest would force the projection to read
every tensor byte, which is exactly the cost the layout exists to avoid.

What it does give: an identity that is stable across snapshots, equal for
equal file bodies, distinct across planners (the tag is hashed), and derivable
from the manifest alone.

## Not part of snapshot identity

Stubs are **projection artifacts**. The manifest keeps the real tensor-planner
entry; a tree builder generates stubs when it projects. Changing this format
changes no snapshot id.

## Reading the real bytes

Through tensorfs — the record-addressed reader (`docs/direct-tensor-reads.md`).
A consumer that wants the file as one contiguous stream uses the `extract()`
escape hatch, which the layout doc §9 prices and discourages.
