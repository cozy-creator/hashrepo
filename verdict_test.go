package tensorfs

import (
	"os"
	"path/filepath"
	"testing"
)

// The tensorfs#124 matching-set audit, run against the REAL library document
// rather than a hand-written stub: the whole claim under test is that the
// verdict a bind gate gets is the verdict this contract's 41 declarations
// actually produce.

func loadLibraryContract(t *testing.T, file string) *Contract {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("spec", "v1", "contracts", file))
	if err != nil {
		t.Fatalf("read %s: %v", file, err)
	}
	contract, err := ParseContract(data)
	if err != nil {
		t.Fatalf("parse %s: %v", file, err)
	}
	return contract
}

// sdxlDiffusersTree is a diffusers-multifolder SDXL packaging at the given
// dtype: one member per component, carrying the key spellings the contract
// declares. Every one of these names is a real SDXL key.
func sdxlDiffusersTree(dtype string) []ArtifactFile {
	return []ArtifactFile{
		{Path: "unet/diffusion_pytorch_model.safetensors", Tensors: []InventoryTensor{
			{Name: "time_embedding.linear_1.weight", Dtype: dtype, Shape: []uint64{1280, 320}},
			{Name: "add_embedding.linear_1.weight", Dtype: dtype, Shape: []uint64{1280, 2816}},
			{Name: "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight", Dtype: dtype, Shape: []uint64{640, 640}},
			{Name: "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight", Dtype: dtype, Shape: []uint64{640, 2048}},
			{Name: "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.weight", Dtype: dtype, Shape: []uint64{1280, 1280}},
			// Unclaimed by the contract — the inpainting-style residual that is
			// a LEGAL admit (quality is out of scope, tensorfs#122).
			{Name: "conv_in.weight", Dtype: dtype, Shape: []uint64{320, 9, 3, 3}},
		}},
		{Path: "vae/diffusion_pytorch_model.safetensors", Tensors: []InventoryTensor{
			{Name: "encoder.conv_in.weight", Dtype: dtype, Shape: []uint64{128, 3, 3, 3}},
			{Name: "encoder.down_blocks.0.resnets.0.conv1.weight", Dtype: dtype, Shape: []uint64{128, 128, 3, 3}},
			{Name: "decoder.up_blocks.0.resnets.1.conv2.weight", Dtype: dtype, Shape: []uint64{512, 512, 3, 3}},
			{Name: "quant_conv.weight", Dtype: dtype, Shape: []uint64{8, 8, 1, 1}},
		}},
		{Path: "text_encoder/model.safetensors", Tensors: []InventoryTensor{
			{Name: "text_model.encoder.layers.0.self_attn.q_proj.weight", Dtype: dtype, Shape: []uint64{768, 768}},
			{Name: "text_model.encoder.layers.0.self_attn.q_proj.bias", Dtype: dtype, Shape: []uint64{768}},
			{Name: "text_model.encoder.layers.11.mlp.fc1.weight", Dtype: dtype, Shape: []uint64{3072, 768}},
		}},
		{Path: "text_encoder_2/model.safetensors", Tensors: []InventoryTensor{
			{Name: "text_model.encoder.layers.0.self_attn.k_proj.weight", Dtype: dtype, Shape: []uint64{1280, 1280}},
			{Name: "text_model.encoder.layers.31.mlp.fc2.weight", Dtype: dtype, Shape: []uint64{1280, 5120}},
			{Name: "text_projection.weight", Dtype: dtype, Shape: []uint64{1280, 1280}},
		}},
	}
}

func TestVerdictSatisfiesTheBf16Packaging(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	verdict := contract.Verdict(sdxlDiffusersTree("BF16"))
	if verdict.Kind != VerdictSatisfies {
		t.Fatalf("want satisfies, got %s", verdict)
	}
	if verdict.Explained != 4 {
		t.Fatalf("want all four members explained, got %d (%v)", verdict.Explained, verdict.Unexplained)
	}
	// The 9-channel conv_in rode along unclaimed. That is the ruling: it runs
	// to completion, so it binds, and quality is not admission's business.
	t.Logf("%s", verdict)
}

// DIRECTION 1 of the audit — OVER-constraint. fp16-packaged SDXL is the
// dominant community packaging and the pipeline runs it natively. A flat
// refusal here is the over-constraint Paul ruled against.
func TestVerdictOffersAConversionForFp16Packaging(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	verdict := contract.Verdict(sdxlDiffusersTree("F16"))
	if verdict.Kind != VerdictDerivable {
		t.Fatalf("want derivable (a cast, not a refusal), got %s", verdict)
	}
	if verdict.Conversion == nil || verdict.Conversion.Kind != "dtype-cast" {
		t.Fatalf("want a named dtype-cast conversion, got %+v", verdict.Conversion)
	}
	if len(verdict.Conversion.From) != 1 || verdict.Conversion.From[0] != "F16" {
		t.Fatalf("want the observed dtype named, got %v", verdict.Conversion.From)
	}
	if verdict.Mismatch == nil || verdict.Mismatch.Tensor == "" || verdict.Mismatch.Pattern == "" {
		t.Fatalf("a verdict must name the tensor and the pattern, got %+v", verdict.Mismatch)
	}
	t.Logf("%s", verdict)
}

// The mixed-dtype tree: an fp16-fix VAE beside a bf16 unet. One member
// refusing must not condemn the checkpoint — same remedy, narrower scope.
func TestVerdictOffersAConversionForAMixedDtypeTree(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	tree := sdxlDiffusersTree("BF16")
	for i := range tree {
		if tree[i].Path == "vae/diffusion_pytorch_model.safetensors" {
			for j := range tree[i].Tensors {
				tree[i].Tensors[j].Dtype = "F16"
			}
		}
	}
	verdict := contract.Verdict(tree)
	if verdict.Kind != VerdictDerivable {
		t.Fatalf("want derivable, got %s", verdict)
	}
	if len(verdict.Conversion.Files) != 1 || verdict.Conversion.Files[0] != "vae/diffusion_pytorch_model.safetensors" {
		t.Fatalf("the conversion must name only the member that needs it, got %v", verdict.Conversion.Files)
	}
	t.Logf("%s", verdict)
}

// DIRECTION 2 of the audit — UNDER-constraint, and the honest limit of the
// LAYOUT half. An SD1.5 tree packaged diffusers-multifolder in bf16 uses key
// spellings IDENTICAL to SDXL's, so the layout half CANNOT refuse it and must
// not pretend to: closing this is the config-only dry-load's job (th#2160
// half 2). This test pins that the layout half admits, so that a later "fix"
// which refuses here — re-breaking direction 1 — is caught.
func TestVerdictAdmitsSd15BecauseTheLayoutHalfCannotSeeTheDifference(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	sd15 := []ArtifactFile{
		{Path: "unet/diffusion_pytorch_model.safetensors", Tensors: []InventoryTensor{
			{Name: "time_embedding.linear_1.weight", Dtype: "BF16", Shape: []uint64{1280, 320}},
			{Name: "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight", Dtype: "BF16", Shape: []uint64{640, 640}},
		}},
		// SD1.5 has ONE text encoder, and its keys are the same diffusers CLIP
		// spelling the contract declares.
		{Path: "text_encoder/model.safetensors", Tensors: []InventoryTensor{
			{Name: "text_model.encoder.layers.0.self_attn.q_proj.weight", Dtype: "BF16", Shape: []uint64{768, 768}},
		}},
	}
	verdict := contract.Verdict(sd15)
	if verdict.Kind != VerdictSatisfies {
		t.Fatalf("the layout half cannot tell SD1.5 from SDXL and must not claim to; got %s", verdict)
	}
	t.Logf("layout-half verdict (the dry-load closes this): %s", verdict)
}

// A genuinely different model: no declaration claims anything.
func TestVerdictRefusesAnArtifactItExplainsNothingOf(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	verdict := contract.Verdict([]ArtifactFile{
		{Path: "transformer/model.safetensors", Tensors: []InventoryTensor{
			{Name: "double_blocks.0.img_attn.qkv.weight", Dtype: "BF16", Shape: []uint64{9216, 3072}},
		}},
	})
	if verdict.Kind != VerdictIncompatible || verdict.Mismatch.Kind != MismatchNothingExplained {
		t.Fatalf("want incompatible/nothing-explained, got %s", verdict)
	}
	t.Logf("%s", verdict)
}

// A structural disagreement must beat a conversion: no cast turns a rank-2
// tensor into the rank-4 one the contract declares, so offering one would be
// an under-constraint the other way — work that cannot help.
func TestVerdictPrefersAStructuralRefusalOverAConversion(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.diffusers-bf16.v1.json")
	tree := sdxlDiffusersTree("BF16")
	// One member is merely fp16-packaged (a cast would fix it)...
	for j := range tree[2].Tensors {
		tree[2].Tensors[j].Dtype = "F16"
	}
	// ...and another is structurally wrong at the declared dtype, so no cast
	// helps. The structural answer must win.
	tree[1].Tensors[0] = InventoryTensor{Name: "encoder.conv_in.weight", Dtype: "BF16", Shape: []uint64{128, 3}}
	verdict := contract.Verdict(tree)
	if verdict.Kind != VerdictIncompatible {
		t.Fatalf("want incompatible, got %s", verdict)
	}
	if verdict.Mismatch.Kind != MismatchRank || verdict.Mismatch.Tensor != "encoder.conv_in.weight" {
		t.Fatalf("want a named rank refusal, got %+v", verdict.Mismatch)
	}
	t.Logf("%s", verdict)
}

// The required-declaration and fusion arms, over the single-file CLIP-G
// document — the one checked-in contract that declares both.
func TestMatchHonoursRequiredAndFusion(t *testing.T) {
	contract := loadLibraryContract(t, "sdxl.clip-g-fused-qkv.v1.json")
	const fused = "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"

	// 3840 = 3 * 1280, so the q|k|v seam divides.
	ok, mismatch := contract.Match([]InventoryTensor{
		{Name: fused, Dtype: "BF16", Shape: []uint64{3840, 1280}, Length: 3840 * 1280 * 2},
	})
	if mismatch != nil {
		t.Fatalf("the fused packaging must match: %s", mismatch)
	}
	if ok.Matched != 1 {
		t.Fatalf("want one tensor explained, got %d", ok.Matched)
	}

	// 3841 divides by nothing: the declared seam cannot apply.
	_, mismatch = contract.Match([]InventoryTensor{
		{Name: fused, Dtype: "BF16", Shape: []uint64{3841, 1280}, Length: 3841 * 1280 * 2},
	})
	if mismatch == nil || mismatch.Kind != MismatchFusion {
		t.Fatalf("want a fusion refusal, got %+v", mismatch)
	}

	// The required declaration absent — only the optional bias present.
	_, mismatch = contract.Match([]InventoryTensor{
		{Name: "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias",
			Dtype: "BF16", Shape: []uint64{3840}, Length: 3840 * 2},
	})
	if mismatch == nil || mismatch.Kind != MismatchRequired {
		t.Fatalf("want a required refusal, got %+v", mismatch)
	}
	t.Logf("required refusal: %s", mismatch)
}

// Pattern instances: leading zeros are not instances, so one tensor has one
// spelling and a document cannot claim it twice.
func TestPatternInstanceRules(t *testing.T) {
	pattern, err := parseContractPattern("layers.{i}.attn{i}.weight")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	for _, name := range []string{"layers.0.attn1.weight", "layers.31.attn2.weight"} {
		if !pattern.matches(name) {
			t.Fatalf("%q should be an instance", name)
		}
	}
	for _, name := range []string{
		"layers.01.attn1.weight", // leading zero
		"layers..attn1.weight",   // empty hole
		"layers.0.attn1.bias",    // trailing literal differs
		"layers.0.attn1.weightX", // remainder
		"xlayers.0.attn1.weight", // prefix differs
	} {
		if pattern.matches(name) {
			t.Fatalf("%q should NOT be an instance", name)
		}
	}
}
