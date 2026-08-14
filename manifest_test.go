package hashrepo

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoReproducesSharedCanonicalManifests(t *testing.T) {
	for _, name := range []string{"manifest.json", "variable_manifest.json"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", name))
			if err != nil {
				t.Fatal(err)
			}
			manifest, err := ParseManifest(data)
			if err != nil {
				t.Fatal(err)
			}
			canonical, err := manifest.Canonical()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(canonical, bytes.TrimSpace(data)) {
				t.Fatalf("canonical bytes differ\n got: %s\nwant: %s", canonical, data)
			}
		})
	}
}

func TestEverySharedInvalidManifestIsRefused(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("spec", "v1", "vectors", "invalid.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Name     string          `json:"name"`
		Manifest json.RawMessage `json:"manifest"`
	}
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatal(err)
	}
	for _, vector := range vectors {
		t.Run(vector.Name, func(t *testing.T) {
			if _, err := ParseManifest(vector.Manifest); err == nil {
				t.Fatal("invalid manifest was accepted")
			}
		})
	}
}

func TestRefIsCanonicalAndPortable(t *testing.T) {
	ref, err := ParseRef(" SHA256:" + strings.Repeat("AB", 32) + " ")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := ref.String(), "sha256:abababababababababababababababababababababababababababababababab"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	key, err := ref.ObjectKey("objects")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := key, "objects/sha256/ab/ab/abababababababababababababababababababababababababababababababab"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
