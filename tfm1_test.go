package hashrepo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type tfm1Corpus struct {
	Schema          string `json:"$schema"`
	Format          int    `json:"format"`
	Hash            string `json:"hash"`
	MaxObjectSize   uint64 `json:"max_object_size"`
	FixtureEncoding string `json:"fixture_encoding"`
	Golden          []struct {
		Name        string `json:"name"`
		Fixture     string `json:"fixture"`
		SnapshotID  string `json:"snapshot_id"`
		Description string `json:"description"`
	} `json:"golden"`
	Refusals []struct {
		Name    string `json:"name"`
		Fixture string `json:"fixture"`
		Reason  string `json:"reason"`
	} `json:"refusals"`
}

// The exact reason vocabulary of the Rust decoder. Two Rust variants can
// never reach Go: hardlink-target exists only in the authoring builder, and
// resource-exhausted only on allocation failure.
var tfm1KnownReasons = map[string]bool{
	"truncated": true, "bad-magic": true, "parent-flag": true,
	"count-exceeds-input": true, "trailing-bytes": true, "entry-order": true,
	"duplicate-path": true, "path-utf8": true, "path-length": true,
	"path-not-nfc": true, "path-absolute": true, "path-empty-segment": true,
	"path-dot-segment": true, "path-forbidden-character": true,
	"path-trailing-dot-space": true, "path-windows-reserved": true,
	"case-fold-collision": true, "missing-parent-directory": true,
	"unknown-entry-kind": true, "executable-flag": true,
	"unknown-planner": true, "unknown-record-tag": true,
	"zero-length-record": true, "adjacent-holes": true,
	"data-too-large": true, "record-limit": true,
	"length-sum-mismatch": true, "symlink-target": true,
	"hardlink-ordinal": true,
}

func loadTFM1Corpus(t *testing.T) tfm1Corpus {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("spec", "v1", "tfm1-vectors", "tfm1-vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus tfm1Corpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatal(err)
	}
	if corpus.Format != 1 || corpus.Hash != "sha256" ||
		corpus.MaxObjectSize != uint64(MaxChunkSize) ||
		corpus.FixtureEncoding != "lowercase-hex-lf-v1" {
		t.Fatalf("corpus header drifted: %+v", corpus)
	}
	if len(corpus.Golden) == 0 || len(corpus.Refusals) == 0 {
		t.Fatal("corpus must carry golden and refusal cases")
	}
	return corpus
}

func decodeTFM1Fixture(t *testing.T, fixture string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("spec", "v1", "tfm1-vectors", filepath.FromSlash(fixture)))
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	if !strings.HasSuffix(text, "\n") {
		t.Fatalf("%s: fixture must end in a line feed", fixture)
	}
	text = strings.TrimSuffix(text, "\n")
	text = strings.TrimSuffix(text, "\r")
	if strings.ContainsAny(text, "\r\n") {
		t.Fatalf("%s: fixture must hold exactly one line", fixture)
	}
	if text != strings.ToLower(text) {
		t.Fatalf("%s: fixture hex must be lowercase", fixture)
	}
	decoded, err := hex.DecodeString(text)
	if err != nil {
		t.Fatalf("%s: %v", fixture, err)
	}
	return decoded
}

func TestGoDecodesEverySharedTFM1GoldenVector(t *testing.T) {
	corpus := loadTFM1Corpus(t)
	ids := map[string]string{}
	for _, row := range corpus.Golden {
		t.Run(row.Name, func(t *testing.T) {
			bytes := decodeTFM1Fixture(t, row.Fixture)
			sum := sha256.Sum256(bytes)
			if got := hex.EncodeToString(sum[:]); got != row.SnapshotID {
				t.Fatalf("fixture hash %s does not match the index id %s", got, row.SnapshotID)
			}
			snapshot, err := ParseTFM1(bytes)
			if err != nil {
				t.Fatal(err)
			}
			if snapshot.SnapshotID != row.SnapshotID {
				t.Fatalf("decoded id %s, want %s", snapshot.SnapshotID, row.SnapshotID)
			}
			ids[row.Name] = row.SnapshotID
		})
	}

	t.Run("with-parent names the empty snapshot", func(t *testing.T) {
		bytes := decodeTFM1Fixture(t, "fixtures/with-parent.hex")
		snapshot, err := ParseTFM1(bytes)
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.Parent != ids["empty-snapshot"] {
			t.Fatalf("parent %s, want the empty snapshot id", snapshot.Parent)
		}
	})
}

func TestSharedTFM1GoldenFactsMatchTheRustBuilder(t *testing.T) {
	decode := func(t *testing.T, fixture string) TFM1Snapshot {
		t.Helper()
		snapshot, err := ParseTFM1(decodeTFM1Fixture(t, fixture))
		if err != nil {
			t.Fatal(err)
		}
		return snapshot
	}

	t.Run("single-file", func(t *testing.T) {
		snapshot := decode(t, "fixtures/single-file.hex")
		if len(snapshot.Entries) != 1 || snapshot.Parent != "" {
			t.Fatalf("unexpected shape: %+v", snapshot)
		}
		entry := snapshot.Entries[0]
		if entry.Path != "model.bin" || entry.Kind != TFM1File || entry.Executable ||
			entry.Planner != TFM1PlannerRawFixed64mV1 || entry.LogicalSize != 18 {
			t.Fatalf("unexpected entry: %+v", entry)
		}
		if len(entry.Records) != 1 || entry.Records[0].Hole ||
			entry.Records[0].Length != 18 ||
			entry.Records[0].Digest.Hex() != strings.Repeat("11", 32) {
			t.Fatalf("unexpected records: %+v", entry.Records)
		}
	})

	t.Run("tree", func(t *testing.T) {
		snapshot := decode(t, "fixtures/tree.hex")
		var paths []string
		for _, entry := range snapshot.Entries {
			paths = append(paths, entry.Path)
		}
		want := []string{
			"run.sh", "weights", "weights/model.safetensors",
			"weights/text-encoder", "weights/text-encoder/encoder.safetensors",
		}
		if strings.Join(paths, ",") != strings.Join(want, ",") {
			t.Fatalf("paths %v, want %v", paths, want)
		}
		if !snapshot.Entries[0].Executable {
			t.Fatal("run.sh must be executable")
		}
		if snapshot.Entries[2].Planner != TFM1PlannerSafetensorsV1 {
			t.Fatalf("planner %q", snapshot.Entries[2].Planner)
		}
	})

	t.Run("sparse-file", func(t *testing.T) {
		snapshot := decode(t, "fixtures/sparse-file.hex")
		records := snapshot.Entries[0].Records
		if len(records) != 3 || !records[0].Hole || records[0].Length != 4096 ||
			records[1].Hole || records[1].Length != 100 ||
			!records[2].Hole || records[2].Length != 1<<40 {
			t.Fatalf("unexpected records: %+v", records)
		}
		if snapshot.Entries[0].LogicalSize != 4096+100+1<<40 {
			t.Fatalf("logical size %d", snapshot.Entries[0].LogicalSize)
		}
	})

	t.Run("hardlink-group re-roots onto the first sorted path", func(t *testing.T) {
		snapshot := decode(t, "fixtures/hardlink-group.hex")
		kinds := map[string]TFM1EntryKind{}
		for _, entry := range snapshot.Entries {
			kinds[entry.Path] = entry.Kind
		}
		if kinds["a-alias.bin"] != TFM1File || kinds["b-solo.bin"] != TFM1File ||
			kinds["m-alias.bin"] != TFM1Hardlink || kinds["z-original.bin"] != TFM1Hardlink {
			t.Fatalf("unexpected kinds: %v", kinds)
		}
		for _, entry := range snapshot.Entries {
			if entry.Kind == TFM1Hardlink && entry.Ordinal != 0 {
				t.Fatalf("%s ordinal %d, want 0", entry.Path, entry.Ordinal)
			}
		}
		if carrier, ok := snapshot.OrdinalPath(0); !ok || carrier != "a-alias.bin" {
			t.Fatalf("ordinal 0 resolves to %q", carrier)
		}
		if carrier, ok := snapshot.OrdinalPath(1); !ok || carrier != "b-solo.bin" {
			t.Fatalf("ordinal 1 resolves to %q", carrier)
		}
		if _, ok := snapshot.OrdinalPath(2); ok {
			t.Fatal("ordinal 2 must not resolve")
		}
	})

	t.Run("symlink", func(t *testing.T) {
		snapshot := decode(t, "fixtures/symlink.hex")
		var target string
		for _, entry := range snapshot.Entries {
			if entry.Kind == TFM1Symlink {
				target = entry.Target
			}
		}
		if target != "snapshots/current.bin" {
			t.Fatalf("target %q", target)
		}
	})

	t.Run("boundary-objects", func(t *testing.T) {
		snapshot := decode(t, "fixtures/boundary-objects.hex")
		records := snapshot.Entries[0].Records
		if len(records) != 4 || records[0].Length != 1 ||
			records[1].Length != uint64(MaxChunkSize)-1 ||
			records[2].Length != uint64(MaxChunkSize) ||
			!records[3].Hole || records[3].Length != 1 {
			t.Fatalf("unexpected records: %+v", records)
		}
		empty := snapshot.Entries[1]
		if empty.Path != "zero.bin" || empty.LogicalSize != 0 || len(empty.Records) != 0 {
			t.Fatalf("unexpected empty file: %+v", empty)
		}
	})

	t.Run("gguf-provenance", func(t *testing.T) {
		snapshot := decode(t, "fixtures/gguf-provenance.hex")
		if snapshot.Entries[0].Planner != TFM1PlannerGgufV1 {
			t.Fatalf("planner %q", snapshot.Entries[0].Planner)
		}
	})
}

func TestEverySharedTFM1RefusalIsRefusedWithItsReason(t *testing.T) {
	corpus := loadTFM1Corpus(t)
	for _, row := range corpus.Refusals {
		t.Run(row.Name, func(t *testing.T) {
			if !tfm1KnownReasons[row.Reason] {
				t.Fatalf("index reason %q is outside the shared vocabulary", row.Reason)
			}
			bytes := decodeTFM1Fixture(t, row.Fixture)
			_, err := ParseTFM1(bytes)
			if err == nil {
				t.Fatal("refusal fixture was accepted")
			}
			var refusal *TFM1Error
			if !errors.As(err, &refusal) {
				t.Fatalf("refusal must be a TFM1Error, got %T: %v", err, err)
			}
			if refusal.Reason() != row.Reason {
				t.Fatalf("reason %q, want %q", refusal.Reason(), row.Reason)
			}
		})
	}
}

func TestEverySingleByteTFM1MutationRefusesOrChangesTheID(t *testing.T) {
	original := decodeTFM1Fixture(t, "fixtures/single-file.hex")
	originalSum := sha256.Sum256(original)
	for index := range original {
		mutated := make([]byte, len(original))
		copy(mutated, original)
		mutated[index] ^= 0x01
		snapshot, err := ParseTFM1(mutated)
		if err != nil {
			var refusal *TFM1Error
			if !errors.As(err, &refusal) {
				t.Fatalf("byte %d: refusal must be a TFM1Error, got %T", index, err)
			}
			continue
		}
		if snapshot.SnapshotID == hex.EncodeToString(originalSum[:]) {
			t.Fatalf("byte %d: an accepted mutation must change the identity", index)
		}
	}
}
