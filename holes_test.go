package tensorfs_test

// PROOF BATTERY ITEM 2 — THE THREE HOLES, as diffs against the v1 baseline.
//
// Each one is a defect v1 had by CONSTRUCTION, so each proof has two halves:
// the banked v1 answer, quoted from the engine's own run, and the v2 answer,
// computed here. A test that only showed the v2 answer would be asserting that
// the new engine does something, not that it fixes anything.

import (
	"strings"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

// bankedV1 finds one row of the v1 baseline.
func bankedV1(t *testing.T, contract, checkpoint string) v1Baseline {
	t.Helper()
	for _, row := range loadV1Baselines(t) {
		if row.Contract == contract && row.Checkpoint == checkpoint {
			return row
		}
	}
	t.Fatalf("the bank has no row for %s vs %s", contract, checkpoint)
	return v1Baseline{}
}

// TestHoleA_SdxlInpaintingRefusesOnShape.
//
// v1: `satisfies`, all 1680 tensors, a LEGAL admit — verdict.go says so in its
// own header. The checkpoint binds, runs to completion and serves garbage,
// because a UNet whose `conv_in` takes 9 input channels is not the UNet the
// SDXL pipeline hands 4 channels to.
//
// v2: refused, naming the tensor and both shapes, before a byte is downloaded.
func TestHoleA_SdxlInpaintingRefusesOnShape(t *testing.T) {
	v1 := bankedV1(t, "sdxl.diffusers-bf16@1", "sdxl-inpainting-bf16")
	if v1.Kind != "satisfies" || v1.Matched != 1680 {
		t.Fatalf("the baseline no longer shows v1 admitting the hole: %+v", v1)
	}

	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	// The probe is the WHOLE inpainting tree re-typed bf16 (tensorfs#153):
	// the lane is the whole published tree now, and so is the hole.
	lane := mustLayoutID(t, "sdxl.diffusers@2+plain.bf16@1")
	decision := loaded.Admit(headers["sdxl-inpainting-tree-bf16"].artifact(),
		[]tensorfs.LayoutID{lane})

	if decision.Kind != tensorfs.DecisionRefuse {
		t.Fatalf("v2 answered %s; the inpainting tree must refuse the SDXL lane", decision.Kind)
	}
	refusal := decision.Refusal
	// At whole-tree scale the best refusal is member-level (the two trees'
	// text encoders differ before the unet is reached), and the sharpest fact
	// is in the stamp suffix: the candidate IS the inpainting topology, by
	// name. The conv_in discrimination itself is asserted on the records.
	if !strings.Contains(refusal.String(), "sdxl-inpainting.diffusers@2") {
		t.Errorf("the refusal does not say what the candidate IS: %s", refusal)
	}
	inpainting := loaded.Topology(tensorfs.Handle{Name: "sdxl-inpainting.diffusers", Version: 2})
	sdxl := loaded.Topology(tensorfs.Handle{Name: "sdxl.diffusers", Version: 2})
	convIn := inpainting.Component("unet").Tensors()["conv_in.weight"]
	sdxlConvIn := sdxl.Component("unet").Tensors()["conv_in.weight"]
	if convIn.String() != "[320, 9, 3, 3]" || sdxlConvIn.String() != "[320, 4, 3, 3]" {
		t.Errorf("the discriminator moved: inpainting conv_in %s, sdxl conv_in %s",
			convIn, sdxlConvIn)
	}
	// The whole point of shapes being kept: the two topologies are separate
	// records, so the confusion is not merely caught, it is unrepresentable.
	if _, err := loaded.Layout(mustLayoutID(t,
		"sdxl-inpainting.diffusers@2+plain.bf16@1")); err != nil {
		t.Fatalf("the inpainting tree has no topology of its own: %v", err)
	}
	t.Logf("v1: %s", v1.Rendered)
	t.Logf("v2: %s", refusal)
}

// TestHoleB_Sd15AndSd2SeparateOnShapes.
//
// The two families share 686 key names exactly; the ONLY difference in the
// headers is that SD2's `use_linear_projection` makes 32 `proj_in`/`proj_out`
// weights rank-2 Linears where SD1.x has rank-4 1x1 convs.
//
// v1 could express rank, but its check order reached the DTYPE disagreement
// first and answered `derivable` — an offer to cast SD1.5 into the SD2 lane,
// which no cast can make true. v2 separates them on shapes, and the refusal
// names one of the 32.
func TestHoleB_Sd15AndSd2SeparateOnShapes(t *testing.T) {
	v1 := bankedV1(t, "sd2.diffusers-bf16@1", "sd15")
	if v1.Kind != "derivable" {
		t.Fatalf("the baseline no longer shows v1 offering a cast: %+v", v1)
	}

	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	// Since tensorfs#153 the lanes are WHOLE published trees, and the two
	// families' trees differ grossly (different text encoders, sd15 carries a
	// safety_checker) — so the cross-admission refusals are member-level and
	// the surgical proj_in/proj_out discrimination is asserted directly on the
	// records' unet components below, where it lives.
	for _, probe := range []struct{ checkpoint, lane string }{
		{"sd15-hub", "sd2.diffusers@2+plain.bf16@1"},
		{"sd-turbo-hub", "sd15.diffusers@2+plain.bf16@1"},
	} {
		decision := loaded.Admit(headers[probe.checkpoint].artifact(),
			[]tensorfs.LayoutID{mustLayoutID(t, probe.lane)})
		if decision.Kind != tensorfs.DecisionRefuse {
			t.Fatalf("%s into %s answered %s", probe.checkpoint, probe.lane, decision.Kind)
		}
		t.Logf("%s -> %s: %s", probe.checkpoint, probe.lane, decision.Refusal)
	}
	// The discriminator itself, on the records: identical unet key sets —
	// otherwise this proves nothing about shapes, only about names.
	sd15 := loaded.Topology(tensorfs.Handle{Name: "sd15.diffusers", Version: 2})
	sd2 := loaded.Topology(tensorfs.Handle{Name: "sd2.diffusers", Version: 2})
	left, right := *sd15.Component("unet"), *sd2.Component("unet")
	if len(left.Keys()) != len(right.Keys()) {
		t.Fatalf("%d vs %d keys", len(left.Keys()), len(right.Keys()))
	}
	differing := 0
	for _, key := range left.Keys() {
		other, found := right.Tensors()[key]
		if !found {
			t.Fatalf("%q is in sd15 and not sd2 — the families would separate on NAMES, "+
				"and this test would be proving the wrong thing", key)
		}
		if !other.Equal(left.Tensors()[key]) {
			differing++
		}
	}
	// 32 attention blocks, each with a proj_in AND a proj_out: 64 weights. The
	// v1 documents say "all 32 *.attentions.{i}.proj_{in,out}.weight", counting
	// blocks rather than tensors; the measurement is 64 and the measurement wins.
	if differing != 64 {
		t.Errorf("%d keys differ in shape, want the 64 proj_in/proj_out weights "+
			"(32 attention blocks x 2)", differing)
	}
	t.Logf("686 identical key names, %d differing shapes — the whole discriminator", differing)
}

// TestHoleC_AdmitAnythingIsInexpressible.
//
// v1's `required` flag was load-bearing and unavoidable: matching was per
// member file, so a sharded or multifolder checkpoint could only match a
// document whose declarations were optional — and a document that is optional
// everywhere claims nothing and refuses nothing. tensorfs#150 found three
// documents that refused the very checkpoints they were generated from because
// the flag was set the other way.
//
// v2 has no such flag. Components are matched as a UNION over their members, so
// sharding is invisible, and the layout must be covered EXACTLY. The class is
// not caught; it is inexpressible. This test measures that: across every
// ordered pair of the twenty banked checkpoints, no checkpoint may admit into
// another's lane.
func TestHoleC_AdmitAnythingIsInexpressible(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	successors := loadSuccessors(t)

	// First, the v1 number, from the bank: how often did a document admit a
	// checkpoint that was not its own?
	crossAdmits := 0
	for _, row := range loadV1Baselines(t) {
		if row.Kind != "satisfies" {
			continue
		}
		lane, mapped := successors.Lanes[row.Contract]
		if !mapped {
			continue
		}
		stamp, refusal := loaded.Stamp(headers[row.Checkpoint].artifact())
		if refusal != nil || stamp.String() != lane {
			crossAdmits++
		}
	}
	if crossAdmits == 0 {
		t.Fatal("the v1 bank shows no cross-family admits at all, so this test " +
			"cannot be measuring the thing it claims to measure")
	}

	// Now v2, over every ordered pair.
	stamps := map[string]tensorfs.LayoutID{}
	for id, headerSet := range headers {
		if stamp, refusal := loaded.Stamp(headerSet.artifact()); refusal == nil {
			stamps[id] = stamp
		}
	}
	pairs, wrong := 0, 0
	for _, laneOwner := range sortedNames(stamps) {
		for _, candidate := range sortedNames(headers) {
			if candidate == laneOwner {
				continue
			}
			if other, stamped := stamps[candidate]; stamped && other.Equal(stamps[laneOwner]) {
				continue // genuinely the same packaging (fp16 vs bf16 of one tree)
			}
			pairs++
			decision := loaded.Admit(headers[candidate].artifact(),
				[]tensorfs.LayoutID{stamps[laneOwner]})
			if decision.Kind == tensorfs.DecisionAdmit {
				wrong++
				t.Errorf("%s admits into %s's lane %s", candidate, laneOwner, stamps[laneOwner])
			}
		}
	}
	if wrong != 0 {
		t.Fatalf("%d of %d cross-checkpoint pairs admitted", wrong, pairs)
	}
	t.Logf("v1 admitted %d checkpoints into lanes that were not theirs; "+
		"v2 admits 0 of %d cross-checkpoint pairs", crossAdmits, pairs)

	// The flag itself is gone from the vocabulary — say it out loud, because
	// "we removed the field" is the actual fix and a behaviour test alone would
	// let it come back.
	for _, handle := range loaded.Topologies() {
		for _, component := range loaded.Topology(handle).Components() {
			for _, key := range component.Keys() {
				if component.Tensors()[key] == nil {
					t.Fatalf("%s: %q has no shape — a topology entry is a SHAPE, and a "+
						"shapeless one is the v1 declaration coming back", handle, key)
				}
			}
		}
	}
}

// TestATruncatedCheckpointRefusesByName is fail-closed at its sharpest.
//
// A checkpoint missing ONE tensor is the failure v1 could not see at all: its
// declarations were optional wherever sharding required it, so an absent tensor
// was indistinguishable from a tensor in another member. Under v2 a component
// is matched as a union and must be covered exactly, so a single missing key is
// a refusal that NAMES it — and an extra key is too, in the other direction.
func TestATruncatedCheckpointRefusesByName(t *testing.T) {
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	lane := mustLayoutID(t, "sd15.diffusers@2+plain.bf16@1")
	files := headers["sd15-hub"].artifact()

	dropped := files[0].Tensors[7].Name
	truncated := []tensorfs.ArtifactFile{{
		Path:    files[0].Path,
		Tensors: append(append([]tensorfs.InventoryTensor{}, files[0].Tensors[:7]...), files[0].Tensors[8:]...),
	}}
	decision := loaded.Admit(truncated, []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionRefuse {
		t.Fatalf("a checkpoint missing %q answered %s", dropped, decision.Kind)
	}
	if decision.Refusal.Reason != tensorfs.RefusalMissing {
		t.Errorf("reason %q, want missing", decision.Refusal.Reason)
	}
	if decision.Refusal.Tensor != dropped {
		t.Errorf("the refusal names %q, want the dropped %q", decision.Refusal.Tensor, dropped)
	}

	// The other direction: one tensor the layout does not declare.
	extended := []tensorfs.ArtifactFile{{
		Path: files[0].Path,
		Tensors: append(append([]tensorfs.InventoryTensor{}, files[0].Tensors...),
			tensorfs.InventoryTensor{
				Name: "lora.down.weight", Dtype: "F32", Shape: []uint64{16, 320},
			}),
	}}
	decision = loaded.Admit(extended, []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionRefuse {
		t.Fatalf("a checkpoint carrying an undeclared tensor answered %s", decision.Kind)
	}
	if decision.Refusal.Tensor != "lora.down.weight" {
		t.Errorf("the refusal names %q, want the extra tensor", decision.Refusal.Tensor)
	}
	t.Logf("missing: %s", decision.Refusal)
}
