package tensorfs_test

// LAYOUT MORPHISMS, graded the way the rest of v2 is: against the corpus files
// themselves, and with a RED ARM for every guard.
//
// The pattern throughout is one pair — the shipped record passes, and the same
// record with ONE index changed is refused. A guard that only ever sees valid
// input is a guard nobody has proved can fire, and a layout that is wrong is
// silent by construction: every name, dtype and shape stays correct and the
// numbers move.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

// layoutDocument reads one shipped record as decoded JSON, for mutation.
func layoutDocument(t *testing.T, name string) map[string]any {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "layouts", name))
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(raw, &document); err != nil {
		t.Fatal(err)
	}
	// The shipped digest is a self-check over the shipped content; a mutated
	// document would be refused for the digest before reaching the guard under
	// test, which would make every red arm below pass for the wrong reason.
	delete(document, "digest")
	return document
}

func reparse(t *testing.T, document map[string]any) (*tensorfs.LayoutMorphism, error) {
	t.Helper()
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return tensorfs.ParseLayoutMorphism(encoded)
}

// TestTheLayoutVocabularyIsClosedAndEveryRecordCitesItsSource is the class's
// load-bearing inventory: the v1 vocabulary is CLOSED, so the catalog either
// carries all of it or the fill path has nowhere to send a wish.
func TestTheLayoutVocabularyIsClosedAndEveryRecordCitesItsSource(t *testing.T) {
	loaded := catalog(t)
	want := map[string]tensorfs.LayoutClass{
		"torch.contiguous":        tensorfs.ClassInductor,
		"torch.channels_last-2d":  tensorfs.ClassInductor,
		"torch.channels_last-3d":  tensorfs.ClassInductor,
		"torch.transposed":        tensorfs.ClassInductor,
		"torch.stride-padding-16": tensorfs.ClassInductor,
		"cublas.blockscale-128x4": tensorfs.ClassEndpointDeclared,
		"nunchaku.micro-scale":    tensorfs.ClassEndpointDeclared,
	}
	for _, handle := range loaded.Arrangements() {
		record := loaded.Arrangement(handle)
		class, expected := want[handle.Name]
		if !expected {
			t.Fatalf("%s is in the corpus and not in the closed vocabulary. A new "+
				"arrangement is a deliberate act; if this one is deliberate, this "+
				"list is where it is declared", handle)
		}
		if record.Class() != class {
			t.Fatalf("%s is class %q, want %q", handle, record.Class(), class)
		}
		if len(record.Provenance()) == 0 {
			t.Fatalf("%s cites nothing", handle)
		}
		delete(want, handle.Name)
	}
	if len(want) != 0 {
		t.Fatalf("the corpus is missing %d of the closed vocabulary: %v", len(want), want)
	}
	// The default coordinate has to be the identity, or every stamp written
	// before layouts existed would silently mean a rearrangement.
	if identity := loaded.Arrangement(tensorfs.DefaultLayout); identity == nil || !identity.Identity() {
		t.Fatalf("%s is not the identity record", tensorfs.DefaultLayout)
	}
}

// TestALayoutThatIsNotABijectionIsRefused is the RED ARM of the structural
// half of auto-ratification. Two sub-axes sent to one storage position is a map
// that drops elements, and it drops them without an error anywhere.
func TestALayoutThatIsNotABijectionIsRefused(t *testing.T) {
	document := layoutDocument(t, "torch.channels_last-2d.v1.json")
	if _, err := reparse(t, document); err != nil {
		t.Fatalf("the shipped record must parse: %v", err)
	}
	// ONE index changed: [0, 2, 3, 1] -> [0, 2, 3, 3]. Storage position 3 now
	// holds sub-axis 3 twice and sub-axis 1 never.
	document["permutation"] = []int{0, 2, 3, 3}
	_, err := reparse(t, document)
	if err == nil {
		t.Fatal("a permutation that sends two sub-axes to one position was accepted")
	}
	if !strings.Contains(err.Error(), "bijection") {
		t.Fatalf("the refusal does not name what is wrong: %v", err)
	}
}

// TestAFactorizationThatDoesNotReachItsAxisIsRefused is the other structural
// red arm, and it is the worse failure of the two: a factorization that
// addresses fewer elements than the axis holds reads a live tail as somebody
// else's bytes.
func TestAFactorizationThatDoesNotReachItsAxisIsRefused(t *testing.T) {
	document := layoutDocument(t, "cublas.blockscale-128x4.v1.json")
	record, err := reparse(t, document)
	if err != nil {
		t.Fatalf("the shipped record must parse: %v", err)
	}
	if _, err := record.Plan(tensorfs.Shape{256, 8}); err != nil {
		t.Fatalf("the shipped record must plan a real scale grid: %v", err)
	}
	// ONE extent changed: the 32-row factor becomes 16, so axis 0 addresses
	// ceil(d0/128)*64 elements where the axis holds ceil(d0/128)*128.
	subAxes := document["sub_axes"].([]any)
	subAxes[2].(map[string]any)["extent"] = "16"
	broken, err := reparse(t, document)
	if err != nil {
		t.Fatalf("the mutation must survive parsing to be caught at plan: %v", err)
	}
	if _, err := broken.Plan(tensorfs.Shape{256, 8}); err == nil {
		t.Fatal("a factorization that reaches half its axis was accepted")
	}
}

// TestTheBlockedScaleLayoutAgreesWithTheRuleThatEmitsIt is the cross-document
// guard, and it is the one that would have caught a transcription slip.
//
// Two independently authored documents describe the same bytes: this layout
// record (transcribed from `to_blocked_scales`, which WRITES them) and
// `bfl.nvfp4-preswizzled@1`'s `weight_scale` emission shape (transcribed from
// the packaging BFL publishes). They must agree on the element count for every
// eligible tensor of a real topology, and they are checked against real headers
// rather than a hand-picked shape.
func TestTheBlockedScaleLayoutAgreesWithTheRuleThatEmitsIt(t *testing.T) {
	loaded := catalog(t)
	blocked := loaded.Arrangement(tensorfs.Handle{Name: "cublas.blockscale-128x4", Version: 1})
	if blocked == nil {
		t.Fatal("the corpus carries no cublas.blockscale-128x4@1")
	}
	rule := tensorfs.Handle{Name: "bfl.nvfp4-preswizzled", Version: 1}

	checked := 0
	for _, topology := range loaded.Topologies() {
		layout, err := loaded.Layout(tensorfs.LayoutID{Topology: topology, Quant: rule})
		if err != nil || layout.Transformed() == 0 {
			continue
		}
		for at := range layout.Components {
			component := &layout.Components[at]
			for _, key := range component.Keys() {
				if !strings.HasSuffix(key, ".weight_scale") {
					continue
				}
				packed, found := component.Tensors[strings.TrimSuffix(key, "_scale")]
				if !found {
					t.Fatalf("%s has no packed weight beside it", key)
				}
				// The rule emits the packed weight as [out, in/2] and the scale
				// as one flat run; the LOGICAL scale grid is [out, in/16].
				grid := tensorfs.Shape{packed.Shape[0], packed.Shape[1] * 2 / 16}
				plan, err := blocked.Plan(grid)
				if err != nil {
					t.Fatalf("%s: %s does not plan %s: %v", topology, blocked, grid, err)
				}
				emitted := component.Tensors[key].Shape
				if len(emitted) != 1 || emitted[0] != plan.Elements {
					t.Fatalf("%s %s: the rule emits %s and the layout arranges %d "+
						"elements. Two documents describing one byte-set disagree",
						topology, key, emitted, plan.Elements)
				}
				// Padding is a property of THIS grid, not of the record: a
				// tile-aligned scale grid pads nothing. Compare against the
				// tiling the record itself declares rather than a fixed answer.
				wantPadded := grid[0]%128 != 0 || grid[1]%4 != 0
				if plan.Padded != wantPadded {
					t.Fatalf("%s %s: grid %s reports padded=%t, want %t",
						topology, key, grid, plan.Padded, wantPadded)
				}
				checked++
			}
		}
		if checked > 0 {
			break
		}
	}
	if checked == 0 {
		t.Fatal("no topology in the corpus emits a preswizzled block scale, so " +
			"this guard proved nothing. It has to run on real headers or it is " +
			"a comment")
	}
	t.Logf("%d block-scale tensors agree with the rule that emits them", checked)

	// A grid that is NOT tile-aligned rounds up to the tiling, which is the
	// case the round-trip-on-the-image half of ratification exists for.
	ragged, err := blocked.Plan(tensorfs.Shape{100, 3})
	if err != nil {
		t.Fatal(err)
	}
	if !ragged.Padded || ragged.Elements != 128*4 {
		t.Fatalf("a [100, 3] grid arranges %d elements (padded=%t), want 512 padded",
			ragged.Elements, ragged.Padded)
	}
}

// TestThePlanVectorsAgreeWithTheRustEvaluator is the CROSS-LANGUAGE guard.
//
// Two evaluators read one shape language: this package decides with it, and
// the Rust crate moves bytes with it. That is the same two-implementations
// bargain tfm1 already makes, and it is held the same way — by banked vectors.
// The Rust suite writes `spec/v2/vectors/layout-plans.json` from its own
// evaluator; this test recomputes every entry with Go's. A drift in either
// direction fails on one side or the other, which is the only reason a second
// evaluator is tolerable at all.
func TestThePlanVectorsAgreeWithTheRustEvaluator(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("spec", "v2", "vectors", "layout-plans.json"))
	if err != nil {
		t.Fatalf("the banked plan vectors are missing: %v", err)
	}
	var banked struct {
		Format string `json:"format"`
		Note   string `json:"note"`
		Plans  []struct {
			Layout  string   `json:"layout"`
			Shape   []uint64 `json:"shape"`
			Source  uint64   `json:"source_elements"`
			Dest    uint64   `json:"dest_elements"`
			Padded  bool     `json:"padded"`
			Extents []uint64 `json:"storage_extents"`
		} `json:"plans"`
	}
	if err := json.Unmarshal(raw, &banked); err != nil {
		t.Fatal(err)
	}
	if banked.Format != "tensorfs-layout-plan-vectors-v1" {
		t.Fatalf("unexpected vector format %q", banked.Format)
	}
	loaded := catalog(t)
	if len(banked.Plans) < len(loaded.Arrangements()) {
		t.Fatalf("%d vectors for %d arrangements: the bank does not cover the "+
			"catalog, so agreement here proves less than it looks like",
			len(banked.Plans), len(loaded.Arrangements()))
	}
	covered := map[string]bool{}
	for _, plan := range banked.Plans {
		handle, err := tensorfs.ParseHandle(plan.Layout)
		if err != nil {
			t.Fatal(err)
		}
		record := loaded.Arrangement(handle)
		if record == nil {
			t.Fatalf("the bank names %s, which this corpus does not carry", plan.Layout)
		}
		computed, err := record.Plan(tensorfs.Shape(plan.Shape))
		if err != nil {
			t.Fatalf("%s %v: Go refuses a shape Rust planned: %v",
				plan.Layout, plan.Shape, err)
		}
		if computed.Elements != plan.Dest || computed.Padded != plan.Padded {
			t.Fatalf("%s %v: Go computes %d elements (padded=%t), Rust banked "+
				"%d (padded=%t)", plan.Layout, plan.Shape,
				computed.Elements, computed.Padded, plan.Dest, plan.Padded)
		}
		if len(computed.Storage) != len(plan.Extents) {
			t.Fatalf("%s %v: %d storage extents vs %d",
				plan.Layout, plan.Shape, len(computed.Storage), len(plan.Extents))
		}
		for at := range plan.Extents {
			if computed.Storage[at] != plan.Extents[at] {
				t.Fatalf("%s %v: storage extent %d is %d in Go and %d in Rust",
					plan.Layout, plan.Shape, at, computed.Storage[at], plan.Extents[at])
			}
		}
		covered[plan.Layout] = true
	}
	for _, handle := range loaded.Arrangements() {
		if !covered[handle.String()] {
			t.Fatalf("%s has no banked vector, so nothing holds the two "+
				"evaluators together for it", handle)
		}
	}
	t.Logf("%d plan vectors agree across two evaluators", len(banked.Plans))
}

// TestTheStampCarriesTheLayoutAndTheDefaultRendersImplicitly is the
// no-CAS-fork property, stated as a test: adding an axis to the stamp must not
// change the TEXT of any stamp that already exists.
func TestTheStampCarriesTheLayoutAndTheDefaultRendersImplicitly(t *testing.T) {
	stored := mustLayoutID(t, "sdxl.diffusers@1+plain.bf16@1")
	if stored.Bytes != tensorfs.DefaultLayout {
		t.Fatalf("a two-part stamp must mean %s, got %s", tensorfs.DefaultLayout, stored.Bytes)
	}
	if got := stored.String(); got != "sdxl.diffusers@1+plain.bf16@1" {
		t.Fatalf("the default layout must render implicitly; got %q", got)
	}
	rearranged := mustLayoutID(t, "sdxl.diffusers@1+plain.bf16@1+torch.channels_last-2d@1")
	if rearranged.Equal(stored) {
		t.Fatal("two arrangements of one checkpoint compare EQUAL, which is the " +
			"silent handoff the layout coordinate exists to make impossible")
	}
	if got := rearranged.String(); got != "sdxl.diffusers@1+plain.bf16@1+torch.channels_last-2d@1" {
		t.Fatalf("a rearranged stamp must render its layout; got %q", got)
	}
	// One arrangement, one spelling: the explicit default is refused, because
	// two texts for one stamp are two CAS addresses for one tree.
	if _, err := tensorfs.ParseLayoutID("sdxl.diffusers@1+plain.bf16@1+torch.contiguous@1"); err == nil {
		t.Fatal("the explicit default spelling was accepted")
	}
}

// TestALaneThatDeclaresAnArrangementDerivesOrRefusesByName is the admission
// half: the transparent-load path and its refusal, both from headers alone.
func TestALaneThatDeclaresAnArrangementDerivesOrRefusesByName(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	candidate, found := headers["sdxl-diffusers"]
	if !found {
		for _, name := range sortedNames(headers) {
			if strings.HasPrefix(name, "sdxl") {
				candidate, found = headers[name], true
				break
			}
		}
	}
	if !found {
		t.Fatal("no banked sdxl headers to admit")
	}
	stamp, refusal := loaded.Stamp(candidate.artifact())
	if refusal != nil {
		t.Fatalf("the banked checkpoint does not stamp: %v", refusal)
	}
	if stamp.Bytes != tensorfs.DefaultLayout {
		t.Fatalf("a stored tree stamps %s, not %s", tensorfs.DefaultLayout, stamp.Bytes)
	}

	// GREEN: the same tensors, the same element type, a declared arrangement.
	// The endpoint gets a DERIVABLE with one relayout step — it never repacks.
	wants := stamp
	wants.Bytes = tensorfs.Handle{Name: "torch.channels_last-2d", Version: 1}
	decision := loaded.Admit(candidate.artifact(), []tensorfs.LayoutID{wants})
	if decision.Kind != tensorfs.DecisionDerivable {
		t.Fatalf("declaring an arrangement gave %s, want derivable: %s",
			decision.Kind, decision)
	}
	relayouts := 0
	for _, step := range decision.Steps {
		if step.Kind == tensorfs.StepRelayout {
			relayouts++
			if step.Detail != tensorfs.Bridge(stamp.Bytes, wants.Bytes) {
				t.Fatalf("the step names %q, want the bridge %q",
					step.Detail, tensorfs.Bridge(stamp.Bytes, wants.Bytes))
			}
		}
	}
	if relayouts != 1 {
		t.Fatalf("%d relayout steps in %s, want exactly one", relayouts, decision)
	}

	// RED: a lane declaring an arrangement the corpus does not carry. The
	// refusal must NAME the arrangement gap, not report a different model.
	unknown := stamp
	unknown.Bytes = tensorfs.Handle{Name: "vendor.not-in-the-catalog", Version: 1}
	refused := loaded.Admit(candidate.artifact(), []tensorfs.LayoutID{unknown})
	if refused.Kind != tensorfs.DecisionRefuse {
		t.Fatalf("an uncatalogued arrangement gave %s", refused.Kind)
	}
	if refused.Refusal.Reason != tensorfs.RefusalNoBridge {
		t.Fatalf("refused for %q, want %q: %s",
			refused.Refusal.Reason, tensorfs.RefusalNoBridge, refused.Refusal)
	}
	if !strings.Contains(refused.Refusal.String(), "vendor.not-in-the-catalog@1") {
		t.Fatalf("the refusal does not name the missing arrangement: %s", refused.Refusal)
	}
}

// TestADerivedDigestIsRecomputedFromItsSources: identity without bytes, and it
// moves when its inputs move.
func TestADerivedDigestIsRecomputedFromItsSources(t *testing.T) {
	source := tensorfs.RefBytes([]byte("the stored tree's manifest"))
	chunk := tensorfs.RefBytes([]byte("a chunk of stored weights"))
	other := tensorfs.RefBytes([]byte("a DIFFERENT chunk of stored weights"))
	bridge := tensorfs.Bridge(tensorfs.DefaultLayout,
		tensorfs.Handle{Name: "torch.channels_last-2d", Version: 1})

	build := func(content tensorfs.Ref, morphism string) tensorfs.Manifest {
		return tensorfs.Manifest{
			Format: tensorfs.FormatV1,
			Files: []tensorfs.File{{
				Path: "unet/diffusion_pytorch_model.safetensors", SizeBytes: 25,
				Digest: content,
				Chunks: []tensorfs.Chunk{{Digest: content, Len: 25}},
			}},
			Derived: &tensorfs.Derivation{Source: source, Morphism: morphism},
		}
	}
	first, err := build(chunk, bridge).DerivedDigest()
	if err != nil {
		t.Fatal(err)
	}
	again, err := build(chunk, bridge).DerivedDigest()
	if err != nil {
		t.Fatal(err)
	}
	if first != again {
		t.Fatal("the derived digest is not deterministic")
	}
	// RED ARM 1: one source digest changes and the derived identity must move.
	// If it did not, two different derived trees would share one name.
	moved, err := build(other, bridge).DerivedDigest()
	if err != nil {
		t.Fatal(err)
	}
	if moved == first {
		t.Fatal("a derived digest computed over DIFFERENT source bytes is the same")
	}
	// RED ARM 2: the same sources through a different morphism are a different
	// tree. This is the whole reason the morphism id is in the digest.
	elsewhere, err := build(chunk, tensorfs.Bridge(tensorfs.DefaultLayout,
		tensorfs.Handle{Name: "torch.transposed", Version: 1})).DerivedDigest()
	if err != nil {
		t.Fatal(err)
	}
	if elsewhere == first {
		t.Fatal("two arrangements of the same bytes share one derived digest")
	}
	// A stored tree has no derived digest to compute, and says so rather than
	// inventing one.
	stored := build(chunk, bridge)
	stored.Derived = nil
	if _, err := stored.DerivedDigest(); err == nil {
		t.Fatal("a manifest with no derivation produced a derived digest")
	}
	// A manifest whose derivation names a bridge from an arrangement to itself
	// derives nothing and is refused at canonicalization, not at load.
	circular := build(chunk, tensorfs.Bridge(tensorfs.DefaultLayout, tensorfs.DefaultLayout))
	if _, err := circular.Canonical(); err == nil {
		t.Fatal("a derivation that arranges a layout as itself was accepted")
	}
}

// TestTheIdentitySidecarIsPortable: a checkpoint plus its identity travels as
// plain files, and the identity re-parses into the same stamp a hub would.
func TestTheIdentitySidecarIsPortable(t *testing.T) {
	stamp := mustLayoutID(t, "sdxl.diffusers@1+plain.bf16@1+torch.channels_last-2d@1")
	manifest := tensorfs.RefBytes([]byte("a manifest"))
	encoded, err := tensorfs.NewSidecar(stamp, manifest).Canonical()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := tensorfs.ParseSidecar(encoded)
	if err != nil {
		t.Fatal(err)
	}
	back, err := parsed.LayoutID()
	if err != nil {
		t.Fatal(err)
	}
	if !back.Equal(stamp) {
		t.Fatalf("the sidecar round-trips to %s, not %s", back, stamp)
	}
	if parsed.Manifest != manifest {
		t.Fatal("the sidecar lost the tree it identifies")
	}
	// RED ARM: a sidecar that names no tree is an assertion about nothing.
	orphan := tensorfs.NewSidecar(stamp, tensorfs.Ref{})
	if _, err := orphan.Canonical(); err == nil {
		t.Fatal("a sidecar with no manifest digest was accepted")
	}
}
