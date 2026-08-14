package chunkedcas

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
	// ChunkSize is the sole v1 writer policy: 64 MiB fixed chunks.
	ChunkSize int64 = 64 << 20
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

// Manifest is a deterministic repository root.
type Manifest struct {
	Format int    `json:"format"`
	Files  []File `json:"files"`
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
	if f.SizeBytes <= ChunkSize {
		if len(f.Chunks) != 0 {
			return errors.New("files at or below 64 MiB must be stored as one whole object")
		}
		return nil
	}
	expectedCount := int(1 + (f.SizeBytes-1)/ChunkSize)
	if len(f.Chunks) != expectedCount {
		return fmt.Errorf("chunked file requires %d chunks, got %d", expectedCount, len(f.Chunks))
	}
	for index, chunk := range f.Chunks {
		if chunk.Digest.hex == "" {
			return fmt.Errorf("chunk %d digest is required", index)
		}
		expected := min(ChunkSize, f.SizeBytes-int64(index)*ChunkSize)
		if chunk.Len != expected {
			return fmt.Errorf("chunk %d length is %d, expected %d", index, chunk.Len, expected)
		}
	}
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
	copyOf := Manifest{Format: m.Format, Files: append([]File(nil), m.Files...)}
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

// Digest returns the content reference of the canonical manifest bytes.
func (m Manifest) Digest() (Ref, error) {
	data, err := m.Canonical()
	if err != nil {
		return Ref{}, err
	}
	sum := sha256.Sum256(data)
	return Ref{hex: hex.EncodeToString(sum[:])}, nil
}

// ParseManifest decodes and validates a v1 manifest.
func ParseManifest(data []byte) (Manifest, error) {
	if !utf8.Valid(data) {
		return Manifest{}, errors.New("manifest must be valid UTF-8")
	}
	var manifest Manifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return Manifest{}, errors.New("manifest must contain exactly one JSON value")
	}
	if _, err := manifest.Canonical(); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}
