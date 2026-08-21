package tensorfs_test

// PROOF BATTERY ITEM 3 — THE INCIDENT REPLAYS.
//
// Three failures that cost real money or real time, replayed against v2 with
// the real headers of the checkpoints involved. Each one is decided from
// HEADERS ALONE, which is the property that matters: te#185's second stop was
// found after a 66 GB download, and the header that would have predicted it is
// the first few kilobytes of the file.

import (
	"os"
	"path/filepath"
	"testing"

	tensorfs "github.com/cozy-creator/tensorfs"
)

// TestReplayA_MinimaxFusedVsSplitIsDecidedFromHeaders replays te#185.
//
// DiffSynth's MiniMaxH3DiT reads the MINIMAX-NATIVE key set — one fused
// `blocks.N.attn.qkv_proj` per block — and every minimax-h3 artifact the
// platform holds is the diffusers repackaging with split `to_q`/`to_k`/`to_v`.
// Under v1 the two were separate documents with no relation between them, so
// the disagreement surfaced at load.
//
// Two arms, and both matter:
//
//	WITHOUT the ratified seam: REFUSE, naming a tensor and quoting the overlap
//	    — te#185's answer, moved from after the 66 GB download to before it.

// WITH the seam: DERIVABLE through a T3 interleaved repack — because the
//
//	bytes really are there, in a different order, and refusing a checkpoint
//	the platform can serve is the over-constraint tensorfs#122 forbids.
func TestReplayA_MinimaxFusedVsSplitIsDecidedFromHeaders(t *testing.T) {
	headers := loadBankedHeaders(t)
	candidate := headers["minimax-h3-diffusers"]
	if candidate.count() != 638 {
		t.Fatalf("the banked diffusers packaging has %d tensors, want 638", candidate.count())
	}
	lane := mustLayoutID(t, "minimax-h3.native@1+plain.bf16@1")

	// --- arm 1: the corpus WITHOUT the seam ---------------------------------
	stripped := catalogWithout(t, "minimax-h3.split-to-fused-qkv.v1.json")
	native := stripped.Topology(lane.Topology)
	if native == nil {
		t.Fatal("the native topology must still be in the stripped corpus")
	}
	if native.Tensors() != 538 {
		t.Fatalf("the native topology has %d tensors, want 538", native.Tensors())
	}
	decision := stripped.Admit(candidate.artifact(), []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionRefuse {
		t.Fatalf("without the seam the answer must be a refusal, got %s", decision.Kind)
	}
	refusal := decision.Refusal
	if refusal.Tensor == "" || refusal.Observed == "" {
		t.Errorf("the refusal names nothing: %+v", refusal)
	}
	if refusal.Matched == 0 {
		t.Error("the refusal must say how much DID line up — 638 keys against 538 " +
			"with hundreds in common is a repack, and a refusal that says nothing " +
			"about the overlap reads as 'a different model'")
	}
	// The refusal leads with the first disagreement it meets in the member,
	// which here is a key the native packaging renames. The 638-vs-535 FACT —
	// the fused qkv the native side wants and the split one the artifact has —
	// is the same disagreement seen from the other side, so assert it directly
	// rather than demanding the message lead with it.
	nativeUnet := native.Components()[0]
	if _, declared := nativeUnet.Tensors()["blocks.0.attn.qkv_proj.weight"]; !declared {
		t.Fatal("the native topology must declare the fused qkv_proj")
	}
	carried := map[string]bool{}
	for _, member := range candidate.artifact() {
		for _, tensor := range member.Tensors {
			carried[tensor.Name] = true
		}
	}
	if carried["blocks.0.attn.qkv_proj.weight"] {
		t.Fatal("the diffusers artifact must NOT carry the fused key")
	}
	for _, split := range []string{
		"transformer_blocks.0.attn.to_q.weight",
		"transformer_blocks.0.attn.to_k.weight",
		"transformer_blocks.0.attn.to_v.weight",
	} {
		if !carried[split] {
			t.Fatalf("the diffusers artifact must carry %q", split)
		}
	}
	t.Logf("638-key candidate vs the 538-key native lane, no seam: %s", refusal)

	// --- arm 2: the corpus WITH the ratified seam ---------------------------
	loaded := catalog(t)
	decision = loaded.Admit(candidate.artifact(), []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionDerivable {
		t.Fatalf("with the seam the answer must be derivable, got %s: %v",
			decision.Kind, decision.Refusal)
	}
	if len(decision.Steps) != 1 || decision.Steps[0].Kind != tensorfs.StepSeam {
		t.Fatalf("want one seam step, got %v", decision.Steps)
	}
	if decision.Steps[0].Tier != tensorfs.TierInterleaved {
		t.Errorf("the qkv seam is HEAD-INTERLEAVED (56 head-major triples) and must "+
			"price as T3; it priced as %s", decision.Steps[0].Tier)
	}
	t.Logf("with the ratified seam: %s", decision)
}

// TestReplayB_AnimaPackagingsAreMorphismRelated.
//
// anima-base spells its 685 tensors `net.*`; anima-turbo spells the same
// network `model.diffusion_model.*`. v1 refused, and the tensorfs#150 ruling
// that followed — ONE DOCUMENT PER PACKAGING — was correct but left the
// relation unexpressible: the alternative, one contract over both, makes every
// declaration optional and can never refuse anything.
//
// v2 says what is true: same network, different names, DERIVABLE at T1 — a new
// manifest over the same chunks and zero new bytes in CAS.
func TestReplayB_AnimaPackagingsAreMorphismRelated(t *testing.T) {
	v1 := bankedV1(t, "anima.diffsynth-bf16@1", "anima-turbo")
	if v1.Kind != "incompatible" {
		t.Fatalf("the baseline no longer shows v1 refusing: %+v", v1)
	}
	loaded := catalog(t)
	headers := loadBankedHeaders(t)
	lane := mustLayoutID(t, "anima.net@1+plain.bf16@1")

	decision := loaded.Admit(headers["anima-turbo"].artifact(), []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionDerivable {
		t.Fatalf("want derivable, got %s: %v", decision.Kind, decision.Refusal)
	}
	if len(decision.Steps) != 1 || decision.Steps[0].Kind != tensorfs.StepRekey {
		t.Fatalf("want one rekey step, got %v", decision.Steps)
	}
	if decision.Steps[0].Tier != tensorfs.TierRekey {
		t.Errorf("a rename is T1; it priced as %s", decision.Steps[0].Tier)
	}
	if decision.Lossy() {
		t.Error("a rename loses nothing and must not be gated as lossy")
	}
	// And the relation is symmetric, because the seam file says invertible and
	// the catalog PROVED it by round-tripping the topology at load.
	back := loaded.Admit(headers["anima-base"].artifact(),
		[]tensorfs.LayoutID{mustLayoutID(t, "anima.diffusion-model@1+plain.bf16@1")})
	if back.Kind != tensorfs.DecisionDerivable {
		t.Fatalf("the reverse direction answered %s", back.Kind)
	}
	t.Logf("v1: %s", v1.Rendered)
	t.Logf("v2: %s", decision)
	t.Logf("v2, reversed: %s", back)
}

// TestReplayC_FusedQkvFp16FineTuneDerivesInTwoSteps.
//
// The shape of a very common arrival: someone's fine-tune of SDXL, packaged
// with the CLIP-G text encoder in the single-file OpenCLIP fusion
// (`in_proj_weight`) and cast to fp16 — two differences from the served lane,
// one structural and one numeric.
//
// v2 answers with a TWO-STEP derivation, in the order a producer runs them:
// the seam inverse first (a T2 contiguous split, manifest range-slicing over
// the same chunks), then the cast (T4, lossy, gated, produced once). Neither
// step is guessed: the seam is the ratified `sdxl.clip-g-split-to-fused` file
// that replaced v1's two CLIP-G fragments, and the cast is the two plain rules'
// declared dtypes.
//
// The candidate's header is COMPUTED from the catalog rather than downloaded,
// which is exactly what the design means by "layout = quant(topology), always
// computed": the expected header of a packaging nobody has published yet is
// still a fact about it.
func TestReplayC_FusedQkvFp16FineTuneDerivesInTwoSteps(t *testing.T) {
	loaded := catalog(t)
	fused := mustLayoutID(t, "sdxl.clip-g-fused@1+plain.f16@1")
	layout, err := loaded.Layout(fused)
	if err != nil {
		t.Fatal(err)
	}
	files := headerOf(layout)

	// It stamps as what it is, from the computed header alone.
	stamp, refusal := loaded.Stamp(files)
	if refusal != nil {
		t.Fatalf("the fused fp16 packaging does not stamp: %v", refusal)
	}
	if !stamp.Equal(fused) {
		t.Fatalf("stamped %s, want %s", stamp, fused)
	}

	lane := mustLayoutID(t, "sdxl.diffusers@1+plain.bf16@1")
	decision := loaded.Admit(files, []tensorfs.LayoutID{lane})
	if decision.Kind != tensorfs.DecisionDerivable {
		t.Fatalf("want derivable, got %s: %v", decision.Kind, decision.Refusal)
	}
	if len(decision.Steps) != 2 {
		t.Fatalf("want two steps (seam inverse, then cast), got %v", decision.Steps)
	}
	if decision.Steps[0].Kind != tensorfs.StepSeam {
		t.Errorf("step 1 is %s, want the seam", decision.Steps[0].Kind)
	}
	if decision.Steps[0].Tier != tensorfs.TierContiguous {
		t.Errorf("the CLIP-G q|k|v seam is contiguous thirds and prices T2; got %s",
			decision.Steps[0].Tier)
	}
	if decision.Steps[1].Kind != tensorfs.StepCast {
		t.Errorf("step 2 is %s, want the cast", decision.Steps[1].Kind)
	}
	if !decision.Lossy() {
		t.Error("fp16 -> bf16 drops mantissa: the derivation is lossy and must " +
			"be gated, produced once and kept with its provenance")
	}
	t.Logf("v2: %s", decision)
}

// headerOf renders a computed layout as the header a checkpoint carrying that
// stamp would have: one member per component, optional entries omitted.
func headerOf(layout *tensorfs.Layout) []tensorfs.ArtifactFile {
	var files []tensorfs.ArtifactFile
	for at := range layout.Components {
		component := &layout.Components[at]
		member := tensorfs.ArtifactFile{Path: component.Name + "/model.safetensors"}
		if component.Name == "" {
			member.Path = "model.safetensors"
		}
		for _, key := range component.Keys() {
			entry := component.Tensors[key]
			if entry.Optional {
				continue
			}
			member.Tensors = append(member.Tensors, tensorfs.InventoryTensor{
				Name: key, Dtype: entry.Dtypes[0], Shape: entry.Shape,
			})
		}
		files = append(files, member)
	}
	return files
}

// catalogWithout loads the corpus with one morphism removed, so a proof can
// show what the engine says when a ratified fact is NOT available. Without this
// arm, "the seam made it derivable" is unfalsifiable.
func catalogWithout(t *testing.T, morphism string) *tensorfs.Catalog {
	t.Helper()
	root := t.TempDir()
	for _, directory := range []string{"topologies", "rules", "morphisms"} {
		target := filepath.Join(root, "spec", "v2", directory)
		if err := os.MkdirAll(target, 0o755); err != nil {
			t.Fatal(err)
		}
		entries, err := filepath.Glob(filepath.Join("spec", "v2", directory, "*.json"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range entries {
			if filepath.Base(path) == morphism {
				continue
			}
			document, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(
				filepath.Join(target, filepath.Base(path)), document, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	loaded, err := tensorfs.LoadCatalog(os.DirFS(root), "spec/v2")
	if err != nil {
		t.Fatalf("the stripped corpus does not load: %v", err)
	}
	return loaded
}
