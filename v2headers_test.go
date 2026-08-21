package tensorfs_test

// Shared test plumbing: the banked REAL headers.
//
// Every v2 proof runs on real upstream headers rather than on hand-typed
// fixtures, because the defects v2 exists to fix were all invisible to
// hand-typed fixtures — v1's corpus was green the entire time it was admitting
// SDXL's contract onto Z-Image's VAE. The headers are committed (gzipped, 270
// KiB for twenty checkpoints), so the proofs are offline and deterministic.

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

type bankedTensor struct {
	Name   string   `json:"name"`
	Dtype  string   `json:"dtype"`
	Shape  []uint64 `json:"shape"`
	Length uint64   `json:"length"`
}

type bankedFile struct {
	Path    string         `json:"path"`
	Tensors []bankedTensor `json:"tensors"`
}

type bankedHeaders struct {
	ID     string       `json:"id"`
	Source string       `json:"source"`
	Note   string       `json:"note"`
	Files  []bankedFile `json:"files"`
}

func (b bankedHeaders) artifact() []tensorfs.ArtifactFile {
	out := make([]tensorfs.ArtifactFile, 0, len(b.Files))
	for _, member := range b.Files {
		tensors := make([]tensorfs.InventoryTensor, 0, len(member.Tensors))
		for _, tensor := range member.Tensors {
			tensors = append(tensors, tensorfs.InventoryTensor{
				Name: tensor.Name, Dtype: tensor.Dtype,
				Shape: tensor.Shape, Length: tensor.Length,
			})
		}
		out = append(out, tensorfs.ArtifactFile{Path: member.Path, Tensors: tensors})
	}
	return out
}

func (b bankedHeaders) count() int {
	total := 0
	for _, member := range b.Files {
		total += len(member.Tensors)
	}
	return total
}

func loadBankedHeaders(t *testing.T) map[string]bankedHeaders {
	t.Helper()
	entries, err := filepath.Glob(filepath.Join("spec", "v2", "headers", "*.json.gz"))
	if err != nil || len(entries) == 0 {
		t.Fatalf("no banked headers: %v", err)
	}
	out := map[string]bankedHeaders{}
	for _, path := range entries {
		handle, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		reader, err := gzip.NewReader(handle)
		if err != nil {
			t.Fatal(err)
		}
		var headers bankedHeaders
		if err := json.NewDecoder(reader).Decode(&headers); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		_ = reader.Close()
		_ = handle.Close()
		if headers.ID == "" {
			headers.ID = strings.TrimSuffix(filepath.Base(path), ".json.gz")
		}
		out[headers.ID] = headers
	}
	return out
}

// mustLayoutID parses a wire-rendered stamp or fails the test.
func mustLayoutID(t *testing.T, text string) tensorfs.LayoutID {
	t.Helper()
	id, err := tensorfs.ParseLayoutID(text)
	if err != nil {
		t.Fatalf("%s: %v", text, err)
	}
	return id
}

func catalog(t *testing.T) *tensorfs.Catalog {
	t.Helper()
	loaded, err := tensorfs.BuiltinCatalog()
	if err != nil {
		t.Fatalf("the embedded corpus does not load: %v", err)
	}
	return loaded
}

func sortedNames[V any](in map[string]V) []string {
	out := make([]string, 0, len(in))
	for key := range in {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}
