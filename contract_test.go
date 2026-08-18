package tensorfs

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// The shared corpus is authored by the Rust validator; this suite proves the
// Go one agrees on every accept/refuse verdict, refusal label, digest and
// stamp — the cross-language sync mechanism for contract documents.

type contractCorpus struct {
	Format string `json:"format"`
	Golden []struct {
		Name     string  `json:"name"`
		File     *string `json:"file"`
		Document *string `json:"document"`
		Digest   string  `json:"digest"`
		Stamp    string  `json:"stamp"`
	} `json:"golden"`
	Refusals []struct {
		Name     string `json:"name"`
		Document string `json:"document"`
		Reason   string `json:"reason"`
	} `json:"refusals"`
}

func loadContractCorpus(t *testing.T) contractCorpus {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("spec", "v1", "contract-vectors", "contract-vectors.json"))
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	var corpus contractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("parse corpus: %v", err)
	}
	if corpus.Format != "tensorfs-contract-vectors-v1" {
		t.Fatalf("unexpected corpus format %q", corpus.Format)
	}
	if len(corpus.Golden) == 0 || len(corpus.Refusals) == 0 {
		t.Fatal("the corpus is empty")
	}
	return corpus
}

func TestContractGoldenVectorsAgreeWithRust(t *testing.T) {
	corpus := loadContractCorpus(t)
	for _, golden := range corpus.Golden {
		var document []byte
		switch {
		case golden.File != nil && golden.Document == nil:
			data, err := os.ReadFile(filepath.Join("spec", "v1", filepath.FromSlash(*golden.File)))
			if err != nil {
				t.Fatalf("%s: read %s: %v", golden.Name, *golden.File, err)
			}
			document = data
		case golden.File == nil && golden.Document != nil:
			document = []byte(*golden.Document)
		default:
			t.Fatalf("%s: exactly one of file/document", golden.Name)
		}
		contract, err := ParseContract(document)
		if err != nil {
			t.Fatalf("%s: refused: %v", golden.Name, err)
		}
		if digest := contract.Digest(); digest != golden.Digest {
			t.Errorf("%s: digest %s, Rust pinned %s", golden.Name, digest, golden.Digest)
		}
		if stamp := contract.Stamp(); stamp != golden.Stamp {
			t.Errorf("%s: stamp %s, Rust pinned %s", golden.Name, stamp, golden.Stamp)
		}
	}
}

func TestContractRefusalVectorsAgreeWithRust(t *testing.T) {
	corpus := loadContractCorpus(t)
	for _, refusal := range corpus.Refusals {
		_, err := ParseContract([]byte(refusal.Document))
		if err == nil {
			t.Errorf("%s: unexpectedly parsed", refusal.Name)
			continue
		}
		var typed *ContractError
		if !errors.As(err, &typed) {
			t.Errorf("%s: untyped error %v", refusal.Name, err)
			continue
		}
		if typed.Reason != refusal.Reason {
			t.Errorf("%s: reason %q, Rust label %q", refusal.Name, typed.Reason, refusal.Reason)
		}
	}
}

func TestContractStampMatchesTFM1DigestSpelling(t *testing.T) {
	// A nameless custom's stamp is the digest spelling the TFM1 decoder
	// produces for the 0xFF stamp arm: "sha256:" + 64 lowercase hex.
	corpus := loadContractCorpus(t)
	nameless := 0
	for _, golden := range corpus.Golden {
		if golden.Document == nil {
			continue
		}
		contract, err := ParseContract([]byte(*golden.Document))
		if err != nil {
			t.Fatalf("%s: %v", golden.Name, err)
		}
		if contract.Name() != "" {
			continue
		}
		nameless++
		stamp := contract.Stamp()
		if len(stamp) != len("sha256:")+64 || stamp[:7] != "sha256:" {
			t.Errorf("%s: malformed digest stamp %q", golden.Name, stamp)
		}
		if stamp != "sha256:"+contract.Digest() {
			t.Errorf("%s: stamp does not spell the digest", golden.Name)
		}
	}
	if nameless == 0 {
		t.Fatal("the corpus holds no nameless custom; regenerate it")
	}
}
