package tensorfs

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type tfp1Corpus struct {
	Schema          string `json:"$schema"`
	Format          int    `json:"format"`
	Hash            string `json:"hash"`
	MaxObjectSize   uint64 `json:"max_object_size"`
	FixtureEncoding string `json:"fixture_encoding"`
	Golden          []struct {
		Name          string `json:"name"`
		Fixture       string `json:"fixture"`
		FixtureSHA256 string `json:"fixture_sha256"`
		Description   string `json:"description"`
	} `json:"golden"`
	Refusals []struct {
		Name    string `json:"name"`
		Fixture string `json:"fixture"`
		Reason  string `json:"reason"`
	} `json:"refusals"`
}

// The exact reason vocabulary of the Rust decoder. Every variant is reachable
// from pure decoding, so Go carries the full set.
var tfp1KnownReasons = map[string]bool{
	"truncated": true, "bad-magic": true, "zero-objects": true,
	"object-limit": true, "count-exceeds-input": true, "index-order": true,
	"duplicate-digest": true, "zero-length-object": true,
	"object-too-large": true, "payload-too-large": true,
	"offset-mismatch": true, "trailing-bytes": true, "digest-mismatch": true,
}

func loadTFP1Corpus(t *testing.T) tfp1Corpus {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("spec", "v1", "tfp1-vectors", "tfp1-vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus tfp1Corpus
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

func decodeTFP1Fixture(t *testing.T, fixture string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("spec", "v1", "tfp1-vectors", filepath.FromSlash(fixture)))
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

func TestGoDecodesEverySharedTFP1GoldenVector(t *testing.T) {
	corpus := loadTFP1Corpus(t)
	for _, row := range corpus.Golden {
		t.Run(row.Name, func(t *testing.T) {
			data := decodeTFP1Fixture(t, row.Fixture)
			sum := sha256.Sum256(data)
			if got := hex.EncodeToString(sum[:]); got != row.FixtureSHA256 {
				t.Fatalf("fixture hash %s does not match the index pin %s", got, row.FixtureSHA256)
			}
			pack, err := ParseTFP1(data)
			if err != nil {
				t.Fatal(err)
			}
			if len(pack.Objects) == 0 {
				t.Fatal("a golden pack always carries objects")
			}
			previous := ""
			var walked uint64
			for _, object := range pack.Objects {
				if object.Digest.hex <= previous {
					t.Fatal("objects must walk in strict ascending digest order")
				}
				previous = object.Digest.hex
				if RefBytes(object.Bytes) != object.Digest {
					t.Fatalf("object bytes do not hash to %s", object.Digest.hex)
				}
				walked += uint64(len(object.Bytes))
			}
			// The payload region is exactly the objects, in index order.
			payload := data[uint64(len(data))-walked:]
			var concatenated []byte
			for _, object := range pack.Objects {
				concatenated = append(concatenated, object.Bytes...)
			}
			if string(payload) != string(concatenated) {
				t.Fatal("objects must tile the payload region exactly")
			}
		})
	}
}

func TestGoRefusesEverySharedTFP1RefusalVector(t *testing.T) {
	corpus := loadTFP1Corpus(t)
	for _, row := range corpus.Refusals {
		t.Run(row.Name, func(t *testing.T) {
			if !tfp1KnownReasons[row.Reason] {
				t.Fatalf("the corpus names an unknown reason %q", row.Reason)
			}
			data := decodeTFP1Fixture(t, row.Fixture)
			_, err := ParseTFP1(data)
			if err == nil {
				t.Fatal("the refusal fixture decoded")
			}
			var refusal *TFP1Error
			if !errors.As(err, &refusal) {
				t.Fatalf("unexpected error type: %v", err)
			}
			if refusal.Reason() != row.Reason {
				t.Fatalf("reason %q, want %q", refusal.Reason(), row.Reason)
			}
		})
	}
}

func TestEverySingleByteTFP1MutationRefusesOrChangesContent(t *testing.T) {
	corpus := loadTFP1Corpus(t)
	original := decodeTFP1Fixture(t, corpus.Golden[0].Fixture)
	base, err := ParseTFP1(original)
	if err != nil {
		t.Fatal(err)
	}
	for index := range original {
		mutated := append([]byte(nil), original...)
		mutated[index] ^= 0x01
		pack, err := ParseTFP1(mutated)
		if err != nil {
			continue
		}
		changed := len(pack.Objects) != len(base.Objects)
		for i := range pack.Objects {
			if changed {
				break
			}
			if pack.Objects[i].Digest != base.Objects[i].Digest ||
				string(pack.Objects[i].Bytes) != string(base.Objects[i].Bytes) {
				changed = true
			}
		}
		if !changed {
			t.Fatalf("byte %d: an accepted mutation left the content identical", index)
		}
	}
}

func TestTFP1CapBoundariesWithGeneratedBytes(t *testing.T) {
	full := make([]byte, MaxChunkSize)
	for index := range full {
		full[index] = 0xa5
	}
	digest := sha256.Sum256(full)

	assemble := func(count uint64, rows []byte, payload []byte) []byte {
		data := []byte("TFP1")
		data = binary.LittleEndian.AppendUint64(data, count)
		data = append(data, rows...)
		return append(data, payload...)
	}
	row := func(digest [32]byte, offset, length uint64) []byte {
		out := append([]byte(nil), digest[:]...)
		out = binary.LittleEndian.AppendUint64(out, offset)
		return binary.LittleEndian.AppendUint64(out, length)
	}

	// One maximum-size object fills a pack exactly.
	pack, err := ParseTFP1(assemble(1, row(digest, 0, uint64(MaxChunkSize)), full))
	if err != nil {
		t.Fatal(err)
	}
	if len(pack.Objects) != 1 || uint64(len(pack.Objects[0].Bytes)) != uint64(MaxChunkSize) {
		t.Fatal("the boundary pack must carry the one full object")
	}

	// One declared byte above the object bound refuses structurally.
	_, err = ParseTFP1(assemble(1, row(digest, 0, uint64(MaxChunkSize)+1), nil))
	var refusal *TFP1Error
	if !errors.As(err, &refusal) || refusal.Reason() != "object-too-large" {
		t.Fatalf("want object-too-large, got %v", err)
	}

	// Two full objects overflow the payload bound before any payload is
	// read. The rows are digest-sorted so the order guard stays silent and
	// the refusal is exactly the bound.
	other := sha256.Sum256([]byte("other"))
	low, high := digest, other
	if string(other[:]) < string(digest[:]) {
		low, high = other, digest
	}
	twoRows := append(row(low, 0, uint64(MaxChunkSize)),
		row(high, uint64(MaxChunkSize), uint64(MaxChunkSize))...)
	_, err = ParseTFP1(assemble(2, twoRows, nil))
	if !errors.As(err, &refusal) || refusal.Reason() != "payload-too-large" {
		t.Fatalf("want payload-too-large, got %v", err)
	}
}
