package tensorfs

// GROUND TRUTH IS PORTABLE FILES; THE DATABASE IS A REBUILDABLE INDEX.
//
// Paul's consolidation ruling, 2026-08-21: "the less metadata we store, and the
// more consolidated, the better... ideally the system has all the data it needs
// in an easy-to-share, easy-to-package location." A checkpoint and its identity
// have to be shareable as plain files with no hub in the loop, and a fresh hub
// pointed at a bucket must be able to RECONSTRUCT its whole index by scanning.
//
// So the identity of a tree travels beside the tree, in this document:
//
//	the stamp        — (topology, quant, layout), what these bytes ARE
//	the manifest ref — which tree it is the identity of
//
// AND NOTHING ELSE. In particular the derivation edge is NOT here even though
// it is metadata about the same tree: it lives in the manifest, which is the
// load plan, and one fact gets one producer. A sidecar that also carried it
// would be a second copy to disagree with the first.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// SidecarFormat tags the identity document.
const SidecarFormat = "tensorfs-stamp-v2"

// Sidecar is a tree's identity as a portable, CAS-addressed file.
type Sidecar struct {
	Format string `json:"format"`
	// Stamp is the rendered LayoutID. It is a STRING on the wire, in the one
	// spelling th#1809 settled, because this document is read by three
	// languages and a struct-shaped stamp would be three chances to render it
	// differently.
	Stamp string `json:"stamp"`
	// Manifest is the digest of the manifest this identity is about. Without
	// it the document is an assertion about nothing in particular.
	Manifest Ref `json:"manifest"`
}

// NewSidecar builds the identity document for one tree.
func NewSidecar(stamp LayoutID, manifest Ref) Sidecar {
	return Sidecar{Format: SidecarFormat, Stamp: stamp.Normalized().String(), Manifest: manifest}
}

// LayoutID parses the stamp back. It is re-parsed rather than remembered:
// the file is the ground truth, and a reader that trusted a field it did not
// parse would accept a stamp no catalog can resolve.
func (s Sidecar) LayoutID() (LayoutID, error) { return ParseLayoutID(s.Stamp) }

// Validate checks the document is self-consistent.
func (s Sidecar) Validate() error {
	if s.Format != SidecarFormat {
		return fmt.Errorf("not a %s document", SidecarFormat)
	}
	if s.Manifest.hex == "" {
		return errors.New("a stamp sidecar requires the manifest digest it identifies")
	}
	if _, err := s.LayoutID(); err != nil {
		return err
	}
	return nil
}

// Canonical returns the validated compact encoding.
func (s Sidecar) Canonical() ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(s); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'}), nil
}

// Digest is the sidecar's own content address.
func (s Sidecar) Digest() (Ref, error) {
	data, err := s.Canonical()
	if err != nil {
		return Ref{}, err
	}
	sum := sha256.Sum256(data)
	return Ref{hex: hex.EncodeToString(sum[:])}, nil
}

// ParseSidecar decodes and validates one identity document.
func ParseSidecar(data []byte) (Sidecar, error) {
	var sidecar Sidecar
	if err := decodeOneJSON(data, &sidecar, "stamp sidecar"); err != nil {
		return Sidecar{}, err
	}
	if err := sidecar.Validate(); err != nil {
		return Sidecar{}, err
	}
	return sidecar, nil
}
