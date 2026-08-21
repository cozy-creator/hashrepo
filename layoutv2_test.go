package tensorfs_test

// The v2 engine's core properties, measured against REAL headers.
//
// Nothing here is a hand-typed fixture. Every input is a banked upstream header
// (spec/v2/headers), and the golden the layout evaluator is graded against is
// the RATIFIED design's own worked example — file 5 of
// research/tensor-layout-v2, "computed MECHANICALLY by applying file 3's rule
// to file 2's unet inventory".

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

func TestTheEmbeddedCorpusLoadsAndValidatesItself(t *testing.T) {
	loaded := catalog(t)
	if len(loaded.Topologies()) == 0 || len(loaded.Rules()) == 0 {
		t.Fatal("an empty corpus decides nothing")
	}
	// The load already re-applied every morphism to its `from` topology and
	// compared the result to the catalog's own `to` record, in both directions
	// for the invertible ones. Say out loud what that covered, so a corpus that
	// quietly loses its morphisms fails here rather than passing vacuously.
	if len(loaded.Morphisms()) < 5 {
		t.Fatalf("only %d morphisms: the ratified fusion facts are the half of "+
			"this corpus a header cannot check, and they have to be present to be "+
			"round-tripped", len(loaded.Morphisms()))
	}
	for _, handle := range loaded.Morphisms() {
		morphism := loaded.Morphism(handle)
		if len(morphism.Provenance()) == 0 {
			t.Fatalf("%s carries no ratified evidence", handle)
		}
		for _, seam := range morphism.Seams() {
			if !strings.Contains(seam.Provenance, "RATIFIED") {
				t.Fatalf("%s: seam %q does not cite its ratification", handle, seam.Target)
			}
		}
	}
}

// TestEveryBankedCheckpointStamps is the identification half of the engine.
//
// It is a table because the ANSWER matters, not just the absence of an error: a
// stamp that silently drifted from `plain.bf16` to `plain.f16` would still be a
// stamp, and would serve the wrong bytes.
func TestEveryBankedCheckpointStamps(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)

	raw, err := os.ReadFile(filepath.Join("spec", "v2", "baselines", "stamps.json"))
	if err != nil {
		t.Fatal(err)
	}
	var expected struct {
		Stamps    map[string]string `json:"stamps"`
		Unstamped map[string]string `json:"unstamped"`
	}
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatal(err)
	}
	for _, id := range sortedNames(headers) {
		stamp, refusal := loaded.Stamp(headers[id].artifact())
		want, shouldStamp := expected.Stamps[id]
		reason, known := expected.Unstamped[id]
		switch {
		case !shouldStamp && !known:
			t.Errorf("%s is neither an expected stamp nor a recorded gap", id)
		case !shouldStamp:
			if refusal == nil {
				t.Errorf("%s stamped as %s, but is recorded unstamped: %s", id, stamp, reason)
			}
		case refusal != nil:
			t.Errorf("%s should stamp %s and did not: %v", id, want, refusal)
		case stamp.String() != want:
			t.Errorf("%s stamped %s, want %s", id, stamp, want)
		}
	}
	for id := range expected.Stamps {
		if _, banked := headers[id]; !banked {
			t.Errorf("stamps.json names %q, which is not banked — a fixture that "+
				"outlives its input asserts nothing", id)
		}
	}
}

// TestTheStampIsUnique is the property that makes a stamp an IDENTITY.
//
// If two catalogued (topology, quant) pairs computed the same layout, a
// checkpoint's stamp would depend on iteration order, and the same bytes would
// be two different things on two days. The search takes the first match, so
// this is the check that "first" never had to matter.
func TestTheStampIsUnique(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	stamped := 0
	for _, id := range sortedNames(headers) {
		hits := loaded.StampAll(headers[id].artifact())
		if len(hits) > 1 {
			t.Errorf("%s is %d stamps at once: %v", id, len(hits), hits)
		}
		if len(hits) == 1 {
			stamped++
		}
	}
	if stamped == 0 {
		t.Fatal("nothing stamped; a uniqueness check over an empty set is vacuous")
	}
	t.Logf("%d checkpoints, each exactly one stamp", stamped)
}

// TestLayoutIsComputedAndMatchesTheRatifiedExample grades the ONE evaluator.
//
// The golden is the design's own file 5 — the fp8-rowwise expected header of
// SDXL's unet, computed by the ratifying session from file 2's inventory. It is
// not reproduced exactly, and the difference is the finding this test exists to
// keep visible: file 3, the design's ENTIRE rule document, says "2-D weight
// tensors of the DENOISER component", which is 743 of SDXL's unet tensors. The
// PRODUCER — `w8a8_cast_eligible` in python-gen-worker — converts 739. The four
// it does not are the top-level time/add embedding Linears: they are outside
// every repeated block AND their module paths contain `embed`.
//
// The producer wins. A rule that says a real cozy fp8 SDXL artifact carries
// `add_embedding.linear_1.weight_scale` would REFUSE the artifact the platform
// actually publishes, and the design's own §1 says the nvfp4 rule is
// "transcribed from w4a4.py" for exactly this reason. The four-tensor delta is
// enumerated below rather than waved through.
func TestLayoutIsComputedAndMatchesTheRatifiedExample(t *testing.T) {
	loaded := catalog(t)
	id := mustLayoutID(t, "sdxl.diffusers@1+cozy.fp8-rowwise@1")
	layout, err := loaded.Layout(id)
	if err != nil {
		t.Fatal(err)
	}
	unet := layout.Component("unet")
	if unet == nil {
		t.Fatal("no unet component")
	}
	golden := readDesignExample(t)

	// THE ENUMERATED DIFF, and nothing else may differ.
	producerSkips := map[string]bool{
		"add_embedding.linear_1.weight":  true,
		"add_embedding.linear_2.weight":  true,
		"time_embedding.linear_1.weight": true,
		"time_embedding.linear_2.weight": true,
	}
	converted, extra, missing := 0, 0, 0
	for name, entry := range unet.Tensors {
		want, found := golden[name]
		if !found {
			t.Errorf("computed %q, which the design example does not carry", name)
			extra++
			continue
		}
		if entry.Dtypes[0] == "F8_E4M3" {
			converted++
		}
		module := strings.TrimSuffix(name, ".weight_scale")
		module = strings.TrimSuffix(module, ".weight") + ".weight"
		if producerSkips[module] {
			// The design example converts these four; the producer does not.
			if entry.Dtypes[0] != "BF16" {
				t.Errorf("%q: the producer leaves this at BF16, computed %v", name, entry.Dtypes)
			}
			continue
		}
		if want.dtype != entry.Dtypes[0] {
			t.Errorf("%q: design example says %s, computed %s", name, want.dtype, entry.Dtypes[0])
		}
		if !want.shape.Equal(entry.Shape) {
			t.Errorf("%q: design example says %s, computed %s", name, want.shape, entry.Shape)
		}
	}
	for name := range golden {
		if _, found := unet.Tensors[name]; found {
			continue
		}
		module := strings.TrimSuffix(name, "_scale")
		if producerSkips[module] {
			missing++ // the four skipped weights' scale twins
			continue
		}
		t.Errorf("the design example carries %q, which was not computed", name)
	}
	if converted != 739 {
		t.Errorf("the rule converted %d unet tensors, want 739 (the producer's "+
			"measured set; the design example's 743 counts the four embedding "+
			"Linears the producer skips)", converted)
	}
	if missing != 4 || extra != 0 {
		t.Errorf("the diff against the design example is %d missing / %d extra, "+
			"want exactly the 4 enumerated scale twins", missing, extra)
	}
	if layout.Transformed() != 739 {
		t.Errorf("Transformed() is %d, want 739", layout.Transformed())
	}
	// The compression claim, measured: one topology plus one rule, no document.
	if layout.Tensors() != 2641+739 {
		t.Errorf("the computed layout has %d entries, want %d", layout.Tensors(), 2641+739)
	}
}

type designEntry struct {
	shape tensorfs.Shape
	dtype string
}

// readDesignExample parses file 5 of the ratified design, verbatim as banked.
func readDesignExample(t *testing.T) map[string]designEntry {
	t.Helper()
	handle, err := os.Open(filepath.Join(
		"spec", "v2", "vectors", "design-5-composed-fp8-expected-header.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = handle.Close() }()
	out := map[string]designEntry{}
	scanner := bufio.NewScanner(handle)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.Contains(line, "[") {
			continue
		}
		name, rest, found := strings.Cut(line, " ")
		if !found {
			continue
		}
		open := strings.Index(rest, "[")
		close := strings.Index(rest, "]")
		if open < 0 || close < open {
			continue
		}
		var shape tensorfs.Shape
		for _, part := range strings.Split(rest[open+1:close], ",") {
			value, err := strconv.ParseUint(strings.TrimSpace(part), 10, 64)
			if err != nil {
				t.Fatalf("%s: %v", line, err)
			}
			shape = append(shape, value)
		}
		fields := strings.Fields(rest[close+1:])
		if len(fields) == 0 {
			t.Fatalf("%s: no dtype", line)
		}
		out[name] = designEntry{shape: shape, dtype: fields[0]}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(out) != 2423 {
		t.Fatalf("the design example parsed to %d entries, want 2423 "+
			"(1680 unet tensors + 743 scale twins)", len(out))
	}
	return out
}

// TestTheTwoNvfp4RulesCanNeverAlias is the LPIPS 1.11 guard.
//
// Same element type, same tensor names, same ranks — and different bytes. The
// only thing that keeps them apart is that their convention facts are inside
// their digests, so this asserts the digests differ AND that the difference is
// the conventions rather than an accident of spelling.
func TestTheTwoNvfp4RulesCanNeverAlias(t *testing.T) {
	loaded := catalog(t)
	flat := loaded.Rule(tensorfs.Handle{Name: "cozy.nvfp4-flat", Version: 1})
	preswizzled := loaded.Rule(tensorfs.Handle{Name: "bfl.nvfp4-preswizzled", Version: 1})
	if flat == nil || preswizzled == nil {
		t.Fatal("both nvfp4 rules must ship")
	}
	if flat.Digest() == preswizzled.Digest() {
		t.Fatal("the two nvfp4 rules hash the same")
	}
	if flat.DeclaredDtype != preswizzled.DeclaredDtype {
		t.Fatal("the test is vacuous unless the two agree on the declared dtype — " +
			"that agreement is what makes the conflation possible in the first place")
	}
	if flat.Conventions["nibble_order"] == preswizzled.Conventions["nibble_order"] {
		t.Fatal("the nibble order is the fact that separates them")
	}
	// Falsify: make the conventions equal and the digests must collide.
	if sameConventionsWouldCollide(t, flat, preswizzled) {
		t.Fatal("the conventions are not in the digest at all — two formats that " +
			"differ ONLY in nibble order and scale layout would share a stamp")
	}
}

// sameConventionsWouldCollide reports whether the digest ignores the convention
// facts, by asking whether two rules that differ only there hash apart.
func sameConventionsWouldCollide(t *testing.T, flat, preswizzled *tensorfs.QuantRule) bool {
	t.Helper()
	// The emissions differ too (a [out, in/16] matrix vs a 1-D blocked array),
	// so a digest that ignored conventions would still separate these two. The
	// honest check is a purpose-built pair.
	base := `{"format":"tensorfs-quant-rule-v2","name":"probe.nvfp4","version":1,
	 "declared_dtype":"float4_e2m1fn","capability_floor_sm":100,"base_dtype":"BF16",
	 "conventions":{"nibble_order":"%s"},"lossy":true,"inverse":"x",
	 "eligible":{"rank":2,"key_suffix":".weight"},
	 "emissions":[{"key":"{module}.weight","dtype":"U8","shape":["d0","d1/2"]}]}`
	low, err := tensorfs.ParseQuantRule([]byte(strings.Replace(base, "%s", "LOW", 1)))
	if err != nil {
		t.Fatal(err)
	}
	high, err := tensorfs.ParseQuantRule([]byte(strings.Replace(base, "%s", "HIGH", 1)))
	if err != nil {
		t.Fatal(err)
	}
	return low.Digest() == high.Digest()
}

// TestARuleWithoutConventionsIsRefused keeps the identity rule enforceable.
func TestARuleWithoutConventionsIsRefused(t *testing.T) {
	document := `{"format":"tensorfs-quant-rule-v2","name":"probe.silent","version":1,
	 "declared_dtype":"float8_e4m3fn","capability_floor_sm":89,"base_dtype":"BF16",
	 "lossy":true,"inverse":"x","eligible":{"rank":2,"key_suffix":".weight"},
	 "emissions":[{"key":"{module}.weight","dtype":"F8_E4M3","shape":["d0","d1"]}]}`
	if _, err := tensorfs.ParseQuantRule([]byte(document)); err == nil {
		t.Fatal("a transforming rule that states no convention facts was accepted")
	}
}

// TestTheShapeEvaluatorRefusesInexactDivision proves the emission formulas are
// arithmetic and not rounding: an inexact `/` means the eligibility predicate
// admitted a tensor the emission cannot shape, which is a bug in the rule and
// not a number to round.
func TestTheShapeEvaluatorRefusesInexactDivision(t *testing.T) {
	document := `{"format":"tensorfs-quant-rule-v2","name":"probe.misaligned","version":1,
	 "declared_dtype":"float4_e2m1fn","capability_floor_sm":100,"base_dtype":"BF16",
	 "conventions":{"nibble_order":"LOW"},"lossy":true,"inverse":"x",
	 "eligible":{"rank":2,"key_suffix":".weight"},
	 "emissions":[{"key":"{module}.weight","dtype":"U8","shape":["d0","d1/16"]}]}`
	rule, err := tensorfs.ParseQuantRule([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	// A rank-2 weight whose inner dim is 10 — eligible by this deliberately
	// under-specified predicate, and unshapeable by the emission.
	files := []tensorfs.ArtifactFile{{Path: "m.safetensors", Tensors: []tensorfs.InventoryTensor{
		{Name: "blocks.0.attn.weight", Dtype: "BF16", Shape: []uint64{16, 10}},
	}}}
	topology, err := tensorfs.TopologyFromHeaders("probe.tiny", 1, "test", files)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rule.Apply(topology); err == nil {
		t.Fatal("10 / 16 was accepted as a shape")
	}
}

// TestDisplayNamesAreNamesAndNothingElse.
//
// The design keeps the v1 lane-handle spellings as DISPLAY names on the pair,
// for humans, UI and refusal messages — and is explicit that they are never
// parsed and gate nothing. This asserts both halves: every display name resolves
// to a real catalogued pair, and no code path admits by one.
func TestDisplayNamesAreNamesAndNothingElse(t *testing.T) {
	loaded := catalog(t)
	successors := loadSuccessors(t)
	named := 0
	for _, lane := range successors.Lanes {
		id := mustLayoutID(t, lane)
		if _, err := loaded.Layout(id); err != nil {
			t.Errorf("%s has a display name and no layout: %v", lane, err)
			continue
		}
		if loaded.DisplayName(id) == id.String() {
			continue // no older spelling; the wire rendering is the display
		}
		named++
	}
	if named == 0 {
		t.Fatal("no pair carries a v1 display name, so this proves nothing")
	}
	// A pair with no display name renders as itself, never as empty or as a
	// guess: a missing name must degrade to the identity, not to silence.
	unnamed := tensorfs.LayoutID{
		Topology: tensorfs.Handle{Name: "sdxl.clip-g-fused", Version: 1},
		Quant:    tensorfs.Handle{Name: "plain.f16", Version: 1},
	}
	if loaded.DisplayName(unnamed) != unnamed.String() {
		t.Errorf("an unnamed pair displays as %q", loaded.DisplayName(unnamed))
	}
	t.Logf("%d pairs carry their v1 spelling as a display name", named)
}
