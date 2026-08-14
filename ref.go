package hashrepo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"
)

// Ref is the canonical v1 sha256 content reference.
type Ref struct {
	hex string
}

// ParseRef parses an algorithm-tagged content reference. V1 is sha256-only.
func ParseRef(value string) (Ref, error) {
	text := strings.ToLower(strings.TrimSpace(value))
	if text == "" {
		return Ref{}, errors.New("cas ref: empty")
	}
	algorithm, digest, tagged := strings.Cut(text, ":")
	if !tagged {
		return Ref{}, errors.New(`cas ref: not algorithm-tagged (bare hex is refused; write "sha256:<hex>")`)
	}
	if algorithm != "sha256" {
		return Ref{}, fmt.Errorf("cas ref: unsupported algorithm %q", algorithm)
	}
	if len(digest) != sha256.Size*2 {
		return Ref{}, errors.New("cas ref: sha256 digest must be 64 hexadecimal characters")
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil || len(decoded) != sha256.Size {
		return Ref{}, errors.New("cas ref: sha256 digest must be 64 hexadecimal characters")
	}
	return Ref{hex: digest}, nil
}

// RefBytes computes the v1 content reference for data.
func RefBytes(data []byte) Ref {
	sum := sha256.Sum256(data)
	return Ref{hex: hex.EncodeToString(sum[:])}
}

func (r Ref) String() string {
	if r.hex == "" {
		return ""
	}
	return "sha256:" + r.hex
}

// Hex returns the lowercase sha256 digest without its algorithm tag.
func (r Ref) Hex() string { return r.hex }

// ObjectKey maps the ref into the portable local object layout.
func (r Ref) ObjectKey(prefix string) (string, error) {
	clean := strings.Trim(prefix, "/")
	if clean == "" {
		return "", errors.New("object prefix must not be empty")
	}
	for _, component := range strings.Split(clean, "/") {
		if component == "" || component == "." || component == ".." {
			return "", errors.New("object prefix contains an unsafe component")
		}
	}
	if r.hex == "" {
		return "", errors.New("object key requires a non-zero ref")
	}
	return path.Join(clean, "sha256", r.hex[:2], r.hex[2:4], r.hex), nil
}

func (r Ref) MarshalJSON() ([]byte, error) {
	if r.hex == "" {
		return nil, errors.New("cannot encode a zero content ref")
	}
	return json.Marshal(r.String())
}

func (r *Ref) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	parsed, err := ParseRef(text)
	if err != nil {
		return err
	}
	*r = parsed
	return nil
}
