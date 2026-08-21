package tensorfs

// The header inventory: what a bind decision reads, and the ONLY thing it
// reads. Kilobytes per member — a safetensors header is the first bytes of the
// file and a GGUF directory sits behind a 24-byte prefix — which is what makes
// every refusal in this package PRE-DOWNLOAD.

// InventoryTensor is one header entry.
//
// Dtype is the container's own spelling ("BF16", "F16", "F32", "F8_E4M3",
// "U8") or the ggml type name. Shape is LOGICAL row-major, outermost axis
// first (GGUF `ne` reversed).
type InventoryTensor struct {
	Name  string
	Dtype string
	Shape []uint64
	// Length is the tensor's byte extent, when the caller has it. Zero means
	// "not supplied" and nothing in v2 requires it: shapes and dtypes settle
	// every question the extent used to be needed for.
	Length uint64
}

// ArtifactFile is one tensor-carrying member of a checkpoint.
//
// v1 matched PER MEMBER and had to invent a `required` flag so a sharded
// checkpoint would not refuse itself — the flag that, set false everywhere,
// let a document admit anything. v2 groups members into components and matches
// the UNION, so sharding is invisible and the flag does not exist.
type ArtifactFile struct {
	Path    string
	Tensors []InventoryTensor
}
