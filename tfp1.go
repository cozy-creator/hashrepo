package tensorfs

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

// TFP1 is the TensorFS canonical upload-staging envelope (spec/v1/TFP1.md).
// This side is the receiving verifier: it decodes the sorted index and checks
// every whole object's bytes against its digest, one object at a time. TFP1
// never participates in snapshot or file identity and is never retained as a
// storage format; Rust authors the canonical bytes and the shared vectors.

const (
	// TFP1MaxPackObjects mirrors the planner's bounded object cardinality.
	TFP1MaxPackObjects = 1_000_000

	tfp1RowBytes = 32 + 8 + 8
)

// TFP1Object is one whole indexed object. Bytes reference the input envelope.
type TFP1Object struct {
	Digest Ref
	Bytes  []byte
}

// TFP1Pack is one structurally validated envelope.
type TFP1Pack struct {
	Objects []TFP1Object
}

// TFP1Error is a strict decode refusal carrying the stable kebab-case reason
// label shared byte-for-byte with the Rust implementation and the vectors.
type TFP1Error struct {
	reason string
}

func (e *TFP1Error) Error() string { return "tfp1: " + e.reason }

// Reason returns the cross-language refusal label.
func (e *TFP1Error) Reason() string { return e.reason }

func tfp1Refuse(reason string) error { return &TFP1Error{reason: reason} }

// ParseTFP1Index validates the complete structure without hashing any payload
// byte. A receiver that wants verified objects calls VerifyObjects, or
// ParseTFP1, which does both.
func ParseTFP1Index(data []byte) (TFP1Pack, error) {
	if len(data) < 4 {
		return TFP1Pack{}, tfp1Refuse("truncated")
	}
	if string(data[:4]) != "TFP1" {
		return TFP1Pack{}, tfp1Refuse("bad-magic")
	}
	rest := data[4:]
	if len(rest) < 8 {
		return TFP1Pack{}, tfp1Refuse("truncated")
	}
	count := binary.LittleEndian.Uint64(rest[:8])
	rest = rest[8:]
	switch {
	case count == 0:
		return TFP1Pack{}, tfp1Refuse("zero-objects")
	case count > TFP1MaxPackObjects:
		return TFP1Pack{}, tfp1Refuse("object-limit")
	case count > uint64(len(rest))/tfp1RowBytes:
		return TFP1Pack{}, tfp1Refuse("count-exceeds-input")
	}

	objects := make([]TFP1Object, 0, count)
	offsets := make([]uint64, 0, count)
	lengths := make([]uint64, 0, count)
	var previous []byte
	var total uint64
	for index := uint64(0); index < count; index++ {
		row := rest[:tfp1RowBytes]
		rest = rest[tfp1RowBytes:]
		digest := row[:32]
		offset := binary.LittleEndian.Uint64(row[32:40])
		length := binary.LittleEndian.Uint64(row[40:48])
		switch {
		case length == 0:
			return TFP1Pack{}, tfp1Refuse("zero-length-object")
		case length > uint64(MaxChunkSize):
			return TFP1Pack{}, tfp1Refuse("object-too-large")
		case offset != total:
			return TFP1Pack{}, tfp1Refuse("offset-mismatch")
		}
		if previous != nil {
			switch bytes.Compare(previous, digest) {
			case 0:
				return TFP1Pack{}, tfp1Refuse("duplicate-digest")
			case 1:
				return TFP1Pack{}, tfp1Refuse("index-order")
			}
		}
		next := total + length
		if next < total || next > uint64(MaxChunkSize) {
			return TFP1Pack{}, tfp1Refuse("payload-too-large")
		}
		total = next
		previous = digest
		objects = append(objects, TFP1Object{
			Digest: Ref{hex: hex.EncodeToString(digest)},
		})
		offsets = append(offsets, offset)
		lengths = append(lengths, length)
	}

	remaining := uint64(len(rest))
	switch {
	case remaining < total:
		return TFP1Pack{}, tfp1Refuse("truncated")
	case remaining > total:
		return TFP1Pack{}, tfp1Refuse("trailing-bytes")
	}
	for index := range objects {
		objects[index].Bytes = rest[offsets[index] : offsets[index]+lengths[index]]
	}
	return TFP1Pack{Objects: objects}, nil
}

// VerifyObjects hashes each object's bytes against its indexed digest, one
// object at a time, so a receiver never buffers more than one hash state.
func (p TFP1Pack) VerifyObjects() error {
	for _, object := range p.Objects {
		sum := sha256.Sum256(object.Bytes)
		if hex.EncodeToString(sum[:]) != object.Digest.hex {
			return tfp1Refuse("digest-mismatch")
		}
	}
	return nil
}

// ParseTFP1 strictly decodes one canonical envelope and verifies every
// object digest.
func ParseTFP1(data []byte) (TFP1Pack, error) {
	pack, err := ParseTFP1Index(data)
	if err != nil {
		return TFP1Pack{}, err
	}
	if err := pack.VerifyObjects(); err != nil {
		return TFP1Pack{}, err
	}
	return pack, nil
}
