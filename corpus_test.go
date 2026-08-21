package tensorfs_test

// THE DERIVATION IS THE AUTHORITY, and this is the check that keeps it true.
//
// A topology record is derived MECHANICALLY from a reference checkpoint's
// headers. That claim is worth exactly as much as the ability to re-run it, so
// here every `headers:`-sourced record in spec/v2/CORPUS.tsv is re-extracted
// from the banked headers and compared to what is committed — by DIGEST, which
// is the identity a stamp is pinned by.
//
// The failure this catches is a hand-edit. A topology someone "fixed" by typing
// into the JSON describes a checkpoint that may not exist, which is the class
// tensorfs#150 found three of in the v1 library.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

func TestEveryExtractedTopologyReDerivesToItsCommittedDigest(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "CORPUS.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	extracted, derived := 0, 0
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) < 3 {
			t.Fatalf("CORPUS.tsv: %q", line)
		}
		name, version, source := fields[0], fields[1], fields[2]
		handle, err := tensorfs.ParseHandle(name + "@" + version)
		if err != nil {
			t.Fatal(err)
		}
		committed := loaded.Topology(handle)
		if committed == nil {
			t.Errorf("CORPUS.tsv names %s, which the corpus does not carry", handle)
			continue
		}
		if !strings.HasPrefix(source, "headers:") {
			// Morphism-derived records are re-derived by the catalog loader
			// itself, in both directions for the invertible ones.
			derived++
			continue
		}
		identifier := strings.TrimPrefix(source, "headers:")
		banked, found := headers[identifier]
		if !found {
			t.Errorf("%s is extracted from %q, which is not banked", handle, identifier)
			continue
		}
		again, err := tensorfs.TopologyFromHeaders(name, uint32(handle.Version),
			committed.DerivedFrom(), banked.artifact())
		if err != nil {
			t.Errorf("%s does not re-extract: %v", handle, err)
			continue
		}
		if again.Digest() != committed.Digest() {
			t.Errorf("%s re-extracts to %s but the committed record hashes to %s — "+
				"the record was edited rather than regenerated",
				handle, again.Digest()[:16], committed.Digest()[:16])
			continue
		}
		extracted++
	}
	if extracted == 0 {
		t.Fatal("no record was re-extracted; this test proved nothing")
	}
	t.Logf("%d records re-extracted to their committed digest, %d derived by morphism",
		extracted, derived)
}

// TestEveryCatalogDocumentCarriesItsOwnDigest keeps the self-pinning honest:
// a document with no `digest` field would silently accept any edit.
func TestEveryCatalogDocumentCarriesItsOwnDigest(t *testing.T) {
	for _, directory := range []string{"topologies", "rules", "morphisms"} {
		entries, err := filepath.Glob(filepath.Join("spec", "v2", directory, "*.json"))
		if err != nil || len(entries) == 0 {
			t.Fatalf("%s: %v", directory, err)
		}
		for _, path := range entries {
			document, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(document), "\"digest\"") {
				t.Errorf("%s carries no digest — an unpinned document accepts any edit", path)
			}
		}
	}
}

// TestTheSeamFloorIsNotRelitigated: the 1 MiB seam floor is a v1 mechanic that
// v2 does NOT inherit into the decision. A morphism's tier prices the work; the
// floor belongs to the chunker. Asserting it here would put a chunking constant
// into the admission vocabulary, which is how these two got tangled in v1.
//
// (Kept as a comment rather than a test: there is nothing to assert, and the
// reason there is nothing to assert is the point.)

// TestTheDigestVectorsPinTheWholeCorpus is v2's conformance corpus check.
//
// v1 needed cross-language vectors because three engines parsed its documents
// and had to agree byte for byte on a canonical rendering. v2 has one engine,
// so what is left to conform to is identity: a reader holding these documents
// must be holding the ones that were pinned. A digest that moved without a
// version bump changed what a stamp MEANS.
func TestTheDigestVectorsPinTheWholeCorpus(t *testing.T) {
	loaded := catalog(t)
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "vectors", "digests.json"))
	if err != nil {
		t.Fatal(err)
	}
	var pinned struct {
		Topologies map[string]string `json:"topologies"`
		Rules      map[string]string `json:"rules"`
		Morphisms  map[string]string `json:"morphisms"`
	}
	if err := json.Unmarshal(raw, &pinned); err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, handle := range loaded.Topologies() {
		checked += comparePin(t, "topology", handle.String(),
			loaded.Topology(handle).Digest(), pinned.Topologies)
	}
	for _, handle := range loaded.Rules() {
		checked += comparePin(t, "rule", handle.String(),
			loaded.Rule(handle).Digest(), pinned.Rules)
	}
	for _, handle := range loaded.Morphisms() {
		checked += comparePin(t, "morphism", handle.String(),
			loaded.Morphism(handle).Digest(), pinned.Morphisms)
	}
	if want := len(pinned.Topologies) + len(pinned.Rules) + len(pinned.Morphisms); checked != want {
		t.Errorf("%d documents checked against %d pins — the vectors and the "+
			"corpus have drifted apart", checked, want)
	}
	t.Logf("%d documents pinned by digest", checked)
}

func comparePin(t *testing.T, kind, handle, digest string, pins map[string]string) int {
	t.Helper()
	want, pinned := pins[handle]
	if !pinned {
		t.Errorf("%s %s is in the corpus and not in the vectors", kind, handle)
		return 0
	}
	if want != digest {
		t.Errorf("%s %s hashes to %s, pinned at %s — a document changed meaning "+
			"without a version bump", kind, handle, digest[:16], want[:16])
	}
	return 1
}

// TestEveryDeclaredSourceWasActuallyBanked.
//
// SOURCES.tsv is a DECLARATION and the banked headers are the WORK; nothing
// connected the two. tensorfs#151 shipped `flux1` and `stable-audio` rows whose
// bank had never been run — the repos are gated and the run answered 401 — and
// no gate anywhere went red, because every test in this tree enumerates
// `spec/v2/headers/*.json.gz` and a header that was never written is simply not
// in the glob. The corpus was short two checkpoints and the suite counted what
// it had rather than what was owed.
//
// So: a declared source with no banked header REFUSES, and an orphan header no
// row can regenerate refuses too. Both halves matter — the second is what stops
// a header from becoming un-reproducible the day its source row is edited away.
func TestEveryDeclaredSourceWasActuallyBanked(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "headers", "SOURCES.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	banked := loadBankedHeaders(t)
	declared := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		identifier := strings.SplitN(line, "\t", 2)[0]
		declared[identifier] = true
		if _, found := banked[identifier]; !found {
			t.Errorf("SOURCES.tsv declares %q and spec/v2/headers/%s.json.gz does not exist: "+
				"the bank was never run for it, or it failed and the failure was not read",
				identifier, identifier)
		}
	}
	if len(declared) == 0 {
		t.Fatal("SOURCES.tsv parsed to nothing, so this proves nothing")
	}
	for _, identifier := range sortedNames(banked) {
		if !declared[identifier] {
			t.Errorf("spec/v2/headers/%s.json.gz is banked and SOURCES.tsv does not declare it: "+
				"nothing can regenerate it", identifier)
		}
	}
	if !t.Failed() {
		t.Logf("%d declared sources, %d banked headers, and the two sets are equal",
			len(declared), len(banked))
	}
}
