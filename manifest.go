package tensorfs

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	// FormatV1 is the first surviving manifest format.
	FormatV1 = 1
	// MaxChunkSize is a wire bound, not a fixed-offset layout promise.
	MaxChunkSize int64 = 64 << 20
)

// Chunk is one independently content-addressed object in a large file.
type Chunk struct {
	Digest Ref   `json:"digest"`
	Len    int64 `json:"len"`
}

// File is one relative repository path and the objects required to rebuild it.
type File struct {
	Path      string  `json:"path"`
	SizeBytes int64   `json:"size_bytes"`
	Digest    Ref     `json:"digest"`
	Chunks    []Chunk `json:"chunks,omitempty"`
}

// Derivation is the edge that makes a DERIVED tree an identity instead of a
// pile of bytes: this tree is that tree's chunks, read through a named layout
// morphism.
//
// It rides in the manifest and not beside it because THE MANIFEST IS THE LOAD
// PLAN. A reader with the manifest has the chunk list and the arrangement to
// apply to it, which is everything the fill path needs; a derivation edge kept
// in a database would make the plan un-portable, and a checkpoint that only
// loads correctly next to a hub is not a shareable file.
type Derivation struct {
	// Source is the manifest digest of the tree this one is derived from.
	Source Ref `json:"source"`
	// Morphism is the bridge id — `<from-layout>><to-layout>`. It is the whole
	// of what this derivation DOES; the chunk list says what it does it to.
	Morphism string `json:"morphism"`
}

// Validate checks the edge's structure. The morphism must be a bridge id and
// not a bare handle: a derived tree has to say which arrangement it came FROM,
// or applying it backwards is guesswork.
func (d Derivation) Validate() error {
	if d.Source.hex == "" {
		return errors.New("a derivation requires the source tree's manifest digest")
	}
	from, to, err := ParseBridge(d.Morphism)
	if err != nil {
		return fmt.Errorf("derivation morphism: %w", err)
	}
	if from == to {
		return fmt.Errorf("derivation morphism %q arranges %s as itself, which "+
			"derives nothing", d.Morphism, from)
	}
	return nil
}

// Manifest is a deterministic repository root.
type Manifest struct {
	Format int    `json:"format"`
	Files  []File `json:"files"`
	// Derived is set on a derived tree and absent on a stored one. Absent is
	// the overwhelming majority and it encodes to nothing, so every manifest
	// written before layouts existed hashes to exactly what it always did.
	Derived *Derivation `json:"derived_from,omitempty"`
}

func validatePath(value string) error {
	if !utf8.ValidString(value) {
		return errors.New("manifest path must be valid UTF-8")
	}
	if value == "" || strings.HasPrefix(value, "/") || strings.Contains(value, `\`) {
		return fmt.Errorf("manifest path %q must be a relative forward-slash path", value)
	}
	for _, character := range value {
		if character < 32 || character == 127 {
			return fmt.Errorf("manifest path %q contains a control character", value)
		}
	}
	for _, component := range strings.Split(value, "/") {
		if component == "" || component == "." || component == ".." {
			return fmt.Errorf("manifest path %q contains an unsafe component", value)
		}
	}
	return nil
}

func asciiFold(value string) string {
	return strings.Map(func(character rune) rune {
		if character >= 'A' && character <= 'Z' {
			return character + ('a' - 'A')
		}
		return character
	}, value)
}

// Validate checks every v1 structural invariant.
func (f File) Validate() error {
	if err := validatePath(f.Path); err != nil {
		return err
	}
	if f.Digest.hex == "" {
		return errors.New("file digest is required")
	}
	if f.SizeBytes < 0 {
		return errors.New("file size must not be negative")
	}
	if len(f.Chunks) == 0 {
		// A chunkless file is one whole blob of ANY size; the 64 MiB bound
		// is the tensor chunk grid constant, never a blob cap.
		return nil
	}
	var total int64
	for index, chunk := range f.Chunks {
		if chunk.Digest.hex == "" {
			return fmt.Errorf("chunk %d digest is required", index)
		}
		if chunk.Len <= 0 || chunk.Len > MaxChunkSize {
			return fmt.Errorf("chunk %d length is %d, expected 1 through %d", index, chunk.Len, MaxChunkSize)
		}
		if chunk.Len > f.SizeBytes-total {
			return errors.New("chunk lengths exceed the file size")
		}
		total += chunk.Len
	}
	if total != f.SizeBytes {
		return errors.New("chunk lengths must sum exactly to the file size")
	}
	if len(f.Chunks) == 1 && f.Chunks[0].Digest != f.Digest {
		return errors.New("a whole-file chunk must match the file digest")
	}
	return nil
}

func (f *File) UnmarshalJSON(data []byte) error {
	if err := validateJSONStringSurrogates(data); err != nil {
		return err
	}
	var wire struct {
		Path      *string         `json:"path"`
		SizeBytes *int64          `json:"size_bytes"`
		Digest    *Ref            `json:"digest"`
		Chunks    json.RawMessage `json:"chunks"`
	}
	if err := decodeOneJSON(data, &wire, "file"); err != nil {
		return err
	}
	if wire.Path == nil || wire.SizeBytes == nil || wire.Digest == nil {
		return errors.New("file requires path, size_bytes and digest")
	}
	var chunks []Chunk
	if wire.Chunks != nil {
		if bytes.Equal(bytes.TrimSpace(wire.Chunks), []byte("null")) {
			return errors.New("file chunks must be an array, not null")
		}
		if err := decodeOneJSON(wire.Chunks, &chunks, "file chunks"); err != nil {
			return err
		}
		if len(chunks) == 0 {
			return errors.New("file chunks, when present, must not be empty")
		}
	}
	decoded := File{
		Path:      *wire.Path,
		SizeBytes: *wire.SizeBytes,
		Digest:    *wire.Digest,
		Chunks:    chunks,
	}
	if err := decoded.Validate(); err != nil {
		return err
	}
	*f = decoded
	return nil
}

// Objects returns the immutable objects needed to reconstruct this file.
func (f File) Objects() []Object {
	if len(f.Chunks) == 0 {
		return []Object{{Digest: f.Digest, SizeBytes: f.SizeBytes}}
	}
	objects := make([]Object, 0, len(f.Chunks))
	for _, chunk := range f.Chunks {
		objects = append(objects, Object{Digest: chunk.Digest, SizeBytes: chunk.Len})
	}
	return objects
}

// Canonical returns validated, compact v1 JSON with files sorted by path.
func (m Manifest) Canonical() ([]byte, error) {
	if m.Format != FormatV1 {
		return nil, fmt.Errorf("unsupported manifest format %d; v1 accepts only 1", m.Format)
	}
	if m.Derived != nil {
		if err := m.Derived.Validate(); err != nil {
			return nil, err
		}
	}
	files := make([]File, len(m.Files))
	copy(files, m.Files)
	copyOf := Manifest{Format: m.Format, Files: files, Derived: m.Derived}
	sort.Slice(copyOf.Files, func(i, j int) bool { return copyOf.Files[i].Path < copyOf.Files[j].Path })
	seen := map[string]bool{}
	folded := map[string]bool{}
	for _, file := range copyOf.Files {
		if err := file.Validate(); err != nil {
			return nil, fmt.Errorf("%s: %w", file.Path, err)
		}
		if seen[file.Path] {
			return nil, fmt.Errorf("duplicate manifest path %q", file.Path)
		}
		caseKey := asciiFold(file.Path)
		if folded[caseKey] {
			return nil, fmt.Errorf("case-insensitive manifest path collision at %q", file.Path)
		}
		seen[file.Path] = true
		folded[caseKey] = true
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(copyOf); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'}), nil
}

func decodeOneJSON(data []byte, target any, label string) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("%s must contain exactly one JSON value", label)
	}
	return nil
}

func hexDigit(value byte) (uint16, bool) {
	switch {
	case value >= '0' && value <= '9':
		return uint16(value - '0'), true
	case value >= 'a' && value <= 'f':
		return uint16(value-'a') + 10, true
	case value >= 'A' && value <= 'F':
		return uint16(value-'A') + 10, true
	default:
		return 0, false
	}
}

func escapedUTF16Unit(data []byte, slash int) (uint16, bool) {
	if slash+5 >= len(data) || data[slash] != '\\' || data[slash+1] != 'u' {
		return 0, false
	}
	var value uint16
	for _, character := range data[slash+2 : slash+6] {
		digit, ok := hexDigit(character)
		if !ok {
			return 0, false
		}
		value = value*16 + digit
	}
	return value, true
}

// encoding/json replaces unpaired UTF-16 surrogate escapes with U+FFFD. Scan
// JSON string tokens first so invalid paths cannot silently change identity.
func validateJSONStringSurrogates(data []byte) error {
	inString := false
	for index := 0; index < len(data); index++ {
		switch data[index] {
		case '"':
			inString = !inString
		case '\\':
			if !inString || index+1 >= len(data) {
				continue
			}
			if data[index+1] != 'u' {
				index++
				continue
			}
			unit, ok := escapedUTF16Unit(data, index)
			if !ok {
				continue
			}
			switch {
			case unit >= 0xd800 && unit <= 0xdbff:
				low, paired := escapedUTF16Unit(data, index+6)
				if !paired || low < 0xdc00 || low > 0xdfff {
					return errors.New("manifest contains an unpaired Unicode surrogate escape")
				}
				index += 11
			case unit >= 0xdc00 && unit <= 0xdfff:
				return errors.New("manifest contains an unpaired Unicode surrogate escape")
			default:
				index += 5
			}
		}
	}
	return nil
}

func (m *Manifest) UnmarshalJSON(data []byte) error {
	if !utf8.Valid(data) {
		return errors.New("manifest must be valid UTF-8")
	}
	if err := validateJSONStringSurrogates(data); err != nil {
		return err
	}
	var wire struct {
		Format  *int        `json:"format"`
		Files   *[]File     `json:"files"`
		Derived *Derivation `json:"derived_from"`
	}
	if err := decodeOneJSON(data, &wire, "manifest"); err != nil {
		return err
	}
	if wire.Format == nil || wire.Files == nil {
		return errors.New("manifest requires format and a non-null files array")
	}
	decoded := Manifest{Format: *wire.Format, Files: *wire.Files, Derived: wire.Derived}
	if _, err := decoded.Canonical(); err != nil {
		return err
	}
	sort.Slice(decoded.Files, func(i, j int) bool { return decoded.Files[i].Path < decoded.Files[j].Path })
	*m = decoded
	return nil
}

// Digest returns the content reference of the canonical manifest bytes.
func (m Manifest) Digest() (Ref, error) {
	data, err := m.Canonical()
	if err != nil {
		return Ref{}, err
	}
	sum := sha256.Sum256(data)
	return Ref{hex: hex.EncodeToString(sum[:])}, nil
}

// DerivedDigest is a derived tree's CONTENT identity, computed over the source
// digests it references and the morphism applied to them.
//
// The point is that identity does not require the bytes. A tree arranged
// `channels_last-2d@1` out of a stored `contiguous@1` one has a name before
// anything is materialized, so the storage tier can decide whether to
// materialize it at all — and two producers deriving the same thing arrive at
// the same name without talking to each other.
//
// CAS HASHING STAYS DUMB. This is not a content hash of the derived bytes and
// it is not an input to the object store; every chunk in that store is still
// addressed by the sha256 of exactly the bytes it holds. This digest names a
// DERIVATION, and it is recomputed from its inputs on every call rather than
// written down anywhere — a stored copy is a second answer waiting to drift
// from the first.
func (m Manifest) DerivedDigest() (Ref, error) {
	if m.Derived == nil {
		return Ref{}, errors.New("a manifest with no derivation has no derived digest")
	}
	if err := m.Derived.Validate(); err != nil {
		return Ref{}, err
	}
	// The edge itself is excluded from the hashed body and stated in the
	// preamble instead: the identity is (what the source bytes are) plus (what
	// is done to them), and nothing about the document that carries the claim.
	sourceOnly := Manifest{Format: m.Format, Files: m.Files}
	body, err := sourceOnly.Canonical()
	if err != nil {
		return Ref{}, err
	}
	var canonical bytes.Buffer
	canonical.WriteString("tensorfs-derived-v2\n")
	fmt.Fprintf(&canonical, "morphism=%s\n", m.Derived.Morphism)
	fmt.Fprintf(&canonical, "source=%s\n", m.Derived.Source)
	canonical.Write(body)
	sum := sha256.Sum256(canonical.Bytes())
	return Ref{hex: hex.EncodeToString(sum[:])}, nil
}

// ParseManifest decodes and validates a v1 manifest.
func ParseManifest(data []byte) (Manifest, error) {
	if !utf8.Valid(data) {
		return Manifest{}, errors.New("manifest must be valid UTF-8")
	}
	var manifest Manifest
	if err := decodeOneJSON(data, &manifest, "manifest"); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}
