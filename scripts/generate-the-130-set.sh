#!/usr/bin/env bash
# tensorfs#130 — regenerate every document the pgw#1597 ruling made owed.
#
# Each command below is the DERIVATION, kept runnable rather than described:
# re-running it re-reads the same headers over HTTP ranges and rewrites the same
# document. Nothing here downloads a weight byte.
#
#   HF_TOKEN=<token> scripts/generate-the-130-set.sh [name ...]
#
# krea/Krea-2-Raw is gated and needs HF_TOKEN; everything else is public.
set -euo pipefail
cd "$(dirname "$0")/.."
GEN="nice -n 19 python3 scripts/generate-contract.py"
OUT=spec/v1/contracts
want() { [ $# -eq 0 ] && return 0; for n in "$@"; do [ "$n" = "$TARGET" ] && return 0; done; return 1; }

# ── the falsification: a document a HUMAN authored, re-derived ───────────────
# Not written to disk. It reproduces spec/v1/contracts/ernie.diffusers-bf16.v1.json's
# pattern/rank/dtypes set EXACTLY (24/24) from the real header, which is what
# makes the generator believable for the documents below.
TARGET=ernie; if want "$@"; then
  $GEN --name ernie.diffusers-bf16 --version 1 \
    --source base=hf:baidu/ERNIE-Image@5346b31d68c9c23758ba56ef8be5e9dc174c7f99:transformer \
    --pin-trivial --out /dev/stdout >/dev/null
fi

TARGET=anima; if want "$@"; then
$GEN --name anima.diffsynth-bf16 --version 1 --pin-trivial \
  --source base=hf:circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors \
  --summary "Anima: a Cosmos-Predict2-2B-backbone DiT served through DiffSynth from a SPLIT
checkpoint - one flat safetensors per release, no diffusers component folders. Derived from
circlestone-labs/Anima split_files/diffusion_models/anima-base-v1.0.safetensors, which is the
file tensorhub/anima binds. DIT ONLY: the repo's qwen_3_06b_base text encoder and
qwen_image_vae are the qwen-image family's own components, so declaring either would tie this
document to a different family's checkpoints (tensorfs#121). ONE PACKAGING PER DOCUMENT, and
this was MEASURED rather than assumed: anima-turbo-v1.0 and anima-aesthetic-v1.1 carry the same
685-tensor network under a 'model.diffusion_model.' prefix where base carries 'net.', so a
single document over all three makes EVERY declaration optional - a contract that can never
refuse anything, which is the implicit-coverage class this issue exists to remove. The prefixed
releases need their own document or a rekey derive; this one describes the served packaging." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, against the
model's own implementation source and the real header." \
  --ratified "THE UNCOVERED PACKAGING, corrected by MEASUREMENT. All eight releases in
circlestone-labs/Anima carry the same 685-tensor network, but the split is not the one
previously written down: the \`net.\` packaging this document describes covers FOUR -
anima-base-v1.0, anima-preview, anima-preview2, anima-preview3-base, key-identical sets, all
BF16 - and the \`model.diffusion_model.\` packaging is exactly anima-turbo-v1.0 and
anima-aesthetic-v1.0/v1.0b/v1.1. The \`-preview*\` releases were previously listed as uncovered;
they are covered. Do NOT widen this document: turbo/aesthetic share ZERO keys with base, so
one document over both makes every declaration optional - a contract that can never refuse
anything. Their document is a pure rekey and compose::derive can answer it." \
  --ratified "ROLES stay literal, and the sibling rekey document must adopt THESE role spellings verbatim
while its patterns carry the \`model.diffusion_model.\` prefix - a shared role vocabulary is
what makes the derive possible instead of a refusal, and it needs no change here." \
  --ratified "FUSIONS: NONE. DiffSynth's \`Attention\` declares four separate Linears - q_proj, k_proj,
v_proj, output_proj (diffsynth/models/anima_dit.py:318-328) - and the header agrees ([2048,
2048] each, q_norm [128] so 16 heads x 128 = 2048). \`GPT2FeedForward\` is layer1 -> GELU ->
layer2 with no gate (:212-228), and 8192 = 4 x 2048 confirms the un-gated ratio. Two tensors
are concatenation-shaped and neither is a packaging seam: \`adaln_modulation_*.2.weight\` [6144,
256] is Sequential(SiLU, Linear(2048, 256), Linear(256, 3*2048)) whose triple is chunked into
shift/scale/gate in place (:619-634), and \`t_embedder.1.linear_2.weight\` [6144, 2048] is the
same shape of thing. That is the treatment \`dit.blocks-fused-qkv@1\` already gives adaLN
modulation - declared, never fused." \
  --ratified "SETS: none. One flat file per release, always whole." \
  --ratified "COMPONENT SCOPE: DiT only, and the repo really does ship the alternatives -
\`split_files/{diffusion_models,text_encoders,vae}\` - so declaring the qwen_3_06b text encoder
or the qwen_image vae would tie this document to the qwen-image family's checkpoints
(tensorfs#121)." \
  --ratified "VERDICT, both directions: \`satisfies\` over the real 685-tensor header of anima-base-v1.0, and
\`incompatible\` against anima-turbo-v1.0 naming
\`net.blocks.{i}.adaln_modulation_cross_attn.{i}.weight\` - the refusal the separate document
exists to produce. Both from the same \`Contract.Verdict\` call the bind gate makes (tensorhub
internal/bindgate/bindgate.go:317)." \
  --out $OUT/anima.diffsynth-bf16.v1.json
fi

TARGET=rife; if want "$@"; then
# The inventory comes from the PRODUCER, not a mirror: tensorhub/rife-4.25 is
# written pod-side by cozy_rife.save_artifact. Regenerate it with
#   serverless-endpoints/wan-2.2/.venv/bin/python -c "import sys,torch,json; \
#     sys.path.insert(0,'src'); import cozy_rife; \
#     torch.set_default_device('meta'); m=cozy_rife.RifeFlowNet(); \
#     json.dump([{'name':k,'dtype':'F32','rank':v.dim()} for k,v in m.state_dict().items()], \
#               open('rife-inventory.json','w'))"
$GEN --name rife.flownet-fp32 --version 1 --pin-trivial \
  --source v4.25="json:${RIFE_INVENTORY:-scripts/rife-4.25-inventory.json}" \
  --summary "RIFE v4.25 flownet: the auxiliary frame interpolator bound beside wan-2.2,
ltx-video-2.3 and minimax-h3. ONE DOCUMENT FOR THREE CLASSES - today the same gen-worker Rife
type is declared three different ways across the fleet (lanes=() on wan-2.2 and ltx, eager_only=
on minimax-h3), which is the drift pgw#1597 cites; this is the document all three now name.
tensorhub/rife-4.25 is NOT a mirror: it is produced pod-side by cozy_rife.save_artifact from
hzwer Practical-RIFE 4.25's flownet.pkl, so the layout's authority is the producing module and
this document is derived from RifeFlowNet's own state_dict on a META device - zero weights, zero
network, no pickle read. That is the same property the fp8 documents have and it is why the
document and the producer cannot disagree. Flat F32: save_pretrained writes the fp32 module the
pickle loads into. NOT a diffusers architecture despite the diffusers-layout tree - the tree is
a packaging choice save_artifact makes so the artifact is a legal SDK slot." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, from the
PRODUCING MODULE rather than a mirror, and re-derived rather than read:
\`RifeFlowNet().state_dict()\` on a META device reproduces \`scripts/rife-4.25-inventory.json\`
EXACTLY - 158/158 name+rank pairs, zero weights, zero network
(serverless-endpoints/wan-2.2/src/cozy_rife.py:495-511, :524-536)." \
  --ratified "ROLES stay literal, and here that is CORRECT rather than merely mechanical. A role vocabulary
exists so a SECOND packaging of the same bytes can be derived instead of refused; the \`net.\`
prefix is imposed by the producer's only path into the artifact (\`_flow_net_state\`,
cozy_rife.py:488-492) and the one other spelling in the world - upstream's \`flownet.pkl\`,
which carries the same names WITHOUT \`net.\` and with an optional DataParallel \`module.\` - is a
producer INPUT that can never be a bound artifact. There is nothing to reconcile with." \
  --ratified "FUSIONS: NONE, and none is possible. The module census on meta is Conv2d x53, ConvTranspose2d
x6, ResConv x40, PixelShuffle x5, LeakyReLU x51 and ZERO nn.Linear - IFNet has no attention
and no projection anywhere (cozy_rife.py:128-198), so no tensor's outer axis is a
concatenation a split could be ~90% wrong about. \`convblock.{i}.beta\` is not an activation
parameter either: it is a per-channel residual scale of shape [1, c, 1, 1], which is why it
declares rank 4 (cozy_rife.py:153-161)." \
  --ratified "SETS: none. A subset snapshot has no subject in a net that is always loaded whole." \
  --ratified "COMPONENT SCOPE: the artifact IS this module - \`save_artifact\` writes
\`RifeInterpolatorPipeline(flownet=RifeFlowNet())\` and nothing else - so the tensorfs#121
shared-component hazard has no instance here." \
  --ratified "ARITHMETIC, independently of any header: 5 IFBlocks x (2 conv0 + 8x3 convblock + 2 lastconv =
30) + 8 encode = 158 tensors, and 5x7 + 8 = 43 declarations, 18 singleton / 25 indexed. F32 is
STRUCTURAL, not observed: \`load_state_dict\` copies into the fp32 module, so the artifact is
fp32 whatever the pickle carries." \
  --ratified "VERDICT: \`satisfies\`, from the same \`Contract.Verdict\` call the bind gate makes (tensorhub
internal/bindgate/bindgate.go:317), over the producing module's own 158-tensor state dict." \
  --out $OUT/rife.flownet-fp32.v1.json
fi

TARGET=musicgen; if want "$@"; then
$GEN --name musicgen.transformers-fp16 --version 1 --pin-trivial \
  --source stereo-medium=hf:facebook/musicgen-stereo-medium \
  --summary "Meta MusicGen stereo-medium (MusicgenForConditionalGeneration): a single-file
transformers tree, flat F16. WHOLE TREE, deliberately, because the tree IS the model - the T5
text encoder, the EnCodec audio codec and the LM decoder are one checkpoint the endpoint loads
with one from_pretrained. The cross-family tie the stable-audio document warned about does NOT
recur: musicgen spells its T5 under a 'text_encoder.' prefix while stable-audio declares the
same 99 tensors in a separate component file under bare 'encoder.' names, so the two documents
share zero pattern spellings and cannot match each other's checkpoints." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, against the
model's own implementation source and the real header." \
  --ratified "THE SAFETENSORS PACKAGING IS THE ONE SERVED, which was the owed question. The repo ships
\`model.safetensors\` beside audiocraft's \`state_dict.bin\`/\`compression_state_dict.bin\`, a
\`pytorch_model.bin\` (+ index) and two ORPHAN fp32 indexes
(\`model.safetensors.index.fp32.json\`, \`pytorch_model.bin.index.fp32.json\`) that name shards
the repo does not contain. The endpoint calls
\`MusicgenForConditionalGeneration.from_pretrained(ctx.checkpoint_dir)\`
(serverless-endpoints/musicgen/src/musicgen/main.py:207), which prefers safetensors and reads
\`model.safetensors.index.json\` - never the \`.fp32.\` spelling - so the single F16 member is
what loads. The .bin twins are unclaimed by this document, and an unclaimed member is a legal
admit (verdict.go's own SDXL-inpainting example), not a refusal." \
  --ratified "ROLES stay literal: one on-disk packaging, no second spelling to reconcile." \
  --ratified "FUSIONS: NONE. \`MusicgenAttention\` declares four separate \`nn.Linear\` - k_proj, v_proj,
q_proj, out_proj (transformers/models/musicgen/modeling_musicgen.py:209-212) - the T5 encoder
is stock split q/k/v/o, and fc1/fc2 [6144, 1536]/[1536, 6144] is a plain 4x MLP with no gate
to fuse. Two tensor families are concatenation-SHAPED and neither is a packaging seam:
\`lstm.weight_ih_l*\` [4096, 1024] is torch's own nn.LSTM (i,f,g,o) gate packing, which every
packaging spells the same way, and \`conv.weight_g\`/\`weight_v\` are a weight-norm pair, not a
split." \
  --ratified "SETS: none. The tree IS the model - one \`from_pretrained\` builds T5 + EnCodec + LM decoder
together - so a subset snapshot would name a fraction of one checkpoint." \
  --ratified "COMPONENT SCOPE: whole tree, deliberately, and the cross-family tie is measured rather than
assumed: stable-audio declares its T5 under bare \`encoder.\` names in a separate component file
while musicgen spells it \`text_encoder.\`, so the two documents share zero pattern spellings." \
  --ratified "STEREO is arithmetic, not a label: config.decoder.num_codebooks is 8 = 4 RVQ codebooks x
audio_channels 2, and the header carries exactly \`embed_tokens.0..7\` and \`lm_heads.0..7\`.
hidden 1536 / 48 layers / 24 heads / ffn 6144 and T5 d_model 768 x 12 layers all match the
declared ranks." \
  --ratified "VERDICT: \`satisfies\`, from the same \`Contract.Verdict\` call the bind gate makes (tensorhub
internal/bindgate/bindgate.go:317), over the real 1004-tensor header." \
  --out $OUT/musicgen.transformers-fp16.v1.json
fi

TARGET=joycaption; if want "$@"; then
$GEN --name joycaption.llava-bf16 --version 1 --pin-trivial \
  --source beta-one=hf:fancyfeast/llama-joycaption-beta-one-hf-llava \
  --summary "JoyCaption Beta One: a LLaVA image captioner (LlavaForConditionalGeneration) as one
transformers snapshot - SigLIP vision tower, multi_modal_projector and Llama language model,
four shards, flat BF16. WHOLE TREE, because there is no core to separate from the encoders here:
the endpoint binds the snapshot with one from_pretrained and the three sub-trees are one
checkpoint." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, against the
model's own implementation source and the real header - and TWO CORRECTIONS were owed to the
bytes rather than to the prose." \
  --ratified "SELF-REFUSAL, FIXED. Every declaration was \`required\`, and matching is PER MEMBER FILE -
\`toArtifactFiles\` builds one ArtifactFile per .safetensors and deliberately does not group an
index's shards (tensorhub internal/bindgate/bindgate.go:388-417). Over this checkpoint's four
shards the document therefore REFUSED the very tree it was generated from (\`in
model-00001-of-00004.safetensors, requires 'language_model.lm_head.weight', which no tensor
satisfies\`). Declarations are now optional, which is the convention verdict.go states and
every hand-authored multi-file document in this library already follows; the generator
computes \`required\` from cross-SOURCE presence and never cross-MEMBER, which is the bug's
origin." \
  --ratified "A REAL FUSED AXIS, DECLARED. SigLIP's attention-pooling head is a
\`torch.nn.MultiheadAttention\` (transformers/models/siglip/modeling_siglip.py:626-635), so
\`vision_tower.vision_model.head.attention.in_proj_weight\` [3456, 1152] = 3 x hidden 1152 is
one fused q|k|v in equal thirds, and \`in_proj_bias\` [3456] with it - the same seam
\`sdxl.clip-g-fused-qkv@1\` already declares for OpenCLIP. Every OTHER projection here is split:
SigLIP's per-layer \`self_attn.{q,k,v,out}_proj\` [1152, 1152] and Llama-3.1-8B's
\`self_attn.q_proj\` [4096, 4096] beside \`k_proj\`/\`v_proj\` [1024, 4096] (GQA, 8 kv heads x 128)
with \`mlp.{gate,up,down}_proj\` [14336, 4096] separate - no gate-up to fuse." \
  --ratified "SCOPE CHECK (tensorfs#121): PASSES, and the discriminating fact is finer than the three
prefixes. No other document in this library declares a \`vision_tower.\`,
\`multi_modal_projector.\` or \`language_model.\` pattern. The near miss is InternVL, which
transformers maps to the SAME llava conversion pattern and so shares all three prefixes - but
its vision tower is InternViT with a FUSED \`attn.qkv.weight\`, so it cannot satisfy this
document's required \`self_attn.q_proj\` and SigLIP-specific \`head.probe\` /
\`head.attention.in_proj_weight\` / \`embeddings.patch_embedding\`; and
internvl-u.diffusers-bf16@1 declares the generation decoder only, never the vlm." \
  --ratified "ROLES stay literal. transformers 5 renames these keys AT LOAD - \`^language_model.model\` ->
\`model.language_model\`, \`^vision_tower\` -> \`model.vision_tower\`
(transformers/conversion_mapping.py:283-288) - but that is a module tree, not a second on-disk
packaging, and the bind gate reads the header." \
  --ratified "SETS: none. One \`from_pretrained\` builds the whole snapshot
(serverless-endpoints/joycaption/src/joycaption/main.py:177-178)." \
  --ratified "VERDICT: \`satisfies\` after the fix, from the same \`Contract.Verdict\` call the bind gate makes
(bindgate.go:317), over the real 743-tensor / 4-member header." \
  --out $OUT/joycaption.llava-bf16.v1.json
# `fusion` is the one field the generator refuses to guess, so the ratified
# answer is applied here rather than lost on every re-run.
nice -n 19 python3 scripts/declare-ratified-fusions.py $OUT/joycaption.llava-bf16.v1.json
fi

TARGET=krea-2; if want "$@"; then
$GEN --name krea-2.diffusers-bf16 --version 1 --pin-trivial \
  --source raw=hf:krea/Krea-2-Raw:transformer \
  --summary "Krea 2 Raw: the diffusers transformer, TRANSFORMER ONLY on the tensorfs#121 rule.
Mixed BF16 + F32, the F32 island being the adaptive-norm / modulation tables, which is the same
shape ltx-2 and z-image ship. Derived from UPSTREAM krea/Krea-2-Raw (a gated repo, read with the
fleet's own token over header ranges), NOT from tensorhub/krea-2-raw. That is sound rather than a
shortcut, and the reason is structural: the recorded mirror defect (ie#632) is a model_index.json
CONFIG key - text_encoder_select_layers, which upstream never carried - and a contract is
LAYOUT-only. Config compatibility is the bind gate's other half (tensorfs#122), not this
document's job, so a config-level mirror defect cannot be baked into a document that declares
only names, ranks and dtypes." \
  --ratified "PROVENANCE: PARTIALLY ratified 2026-08-20 by the tensorfs#150 contract-ratification lane -
four of five items answered from the model's own source and the real header, ONE left open, so
the candidate marker STAYS." \
  --ratified "SELF-REFUSAL, FIXED (not a ratification item; a defect the ratification's verdict run
exposed). Every declaration was \`required\` across the transformer's three shards, so the
document refused the upstream tree it was generated from (\`in
transformer/diffusion_pytorch_model-00001-of-00003.safetensors, requires
'final_layer.linear.bias', which no tensor satisfies\`) - matching is per member file
(tensorhub internal/bindgate/bindgate.go:388-417) and a multi-member document is all-optional
by convention (verdict.go). Declarations are now optional, and the verdict is \`satisfies\`." \
  --ratified "FUSIONS: NONE, and the proof is the asymmetry rather than a naming convention.
transformer/config.json declares num_attention_heads 48 and num_key_value_heads 12 over hidden
6144, and the header agrees exactly: \`attn.to_q\`/\`to_gate\`/\`to_out.0\` are [6144, 6144] while
\`attn.to_k\`/\`to_v\` are [1536, 6144] = 12 x 128. A fused qkv cannot exist across projections of
different sizes, and there is none to mis-split. The feed-forward is SwiGLU with \`ff.gate\` and
\`ff.up\` as SEPARATE [16384, 6144] tensors, so no gate-up seam either. Two tensors are
concatenation-shaped and neither is a packaging seam: \`scale_shift_table\` [6, 6144] is a
modulation table (its rank-2 first axis IS the six-way split, not an outer-axis byte
concatenation) and \`time_mod_proj.weight\` [36864, 6144] = 6 x 6144 is the projection that
feeds it, chunked in place." \
  --ratified "SETS: none - the transformer is one component, always whole." \
  --ratified "COMPONENT SCOPE: transformer ONLY, on the tensorfs#121 rule, and the repo really does ship the
alternatives (text_encoder, vae, plus a single-file raw.safetensors)." \
  --ratified "ROLES stay literal: one served packaging today." \
  --ratify "CONFIRM AGAINST THE MIRROR. This document is derived from UPSTREAM krea/Krea-2-Raw's headers,
and the claim that \`tensorhub/krea-2-raw\`'s TRANSFORMER headers are byte-identical is still
INFERRED from the ie#632 defect being config-scoped (\`text_encoder_select_layers\`, a
model_index key), not measured. It was ATTEMPTED and could not be measured: no reachable stack
catalogues the mirror - the standing master hub answers 404 for cozy/krea-2-raw and for every
other repo ref tried, authenticated. The act that closes it: on a hub that has the mirror,
\`GET /api/v1/repos/cozy/krea-2-raw/resolve?ref=prod\` with a bearer, then a ranged read of the
first 8 bytes + header of each
\`transformer/diffusion_pytorch_model-0000N-of-00003.safetensors\` and a set-compare of (name,
dtype, shape) against upstream's 430 entries. If the mirror differs, the MIRROR is what the
fleet serves and this document follows the bytes." \
  --out $OUT/krea-2.diffusers-bf16.v1.json
fi

TARGET=ltx-2-upsampler; if want "$@"; then
$GEN --name ltx-2-upsampler.diffusers-bf16 --version 1 --pin-trivial \
  --source upsampler=hf:dg845/LTX-2.3-Spatial-Upsampler-Diffusers:latent_upsampler \
  --summary "The LTX-2 2x spatial latent upsampler (LTX2LatentUpsamplerModel): a pure 3-D
convnet bound beside the LTX-2 DiT for the two-stage 1080p/4K recipe, flat BF16.
LATENT_UPSAMPLER ONLY - the same repo ships a vae, and declaring it would tie this document to
the DiT's own checkpoint. This tree shares ZERO keys with ltx-2.diffusers-bf16@1, which is the
measurement that makes Ltx2Upsampler a separate model type rather than a variant of Ltx2." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, against the
model's own implementation source and the real header." \
  --ratified "FUSIONS: none, and none is EXPRESSIBLE. \`LTX2LatentUpsamplerModel\` holds only Conv2d/Conv3d
and GroupNorm and ZERO nn.Linear - initial_conv/initial_norm, res_blocks and
post_upsample_res_blocks (conv1/conv2 + norm1/norm2), the upsampler head, final_conv
(diffusers/pipelines/ltx2/latent_upsampler.py:170-283, ResBlock at :32-55). The ONE
fusion-shaped tensor is \`upsampler.0.weight\` [4096, 1024, 3, 3] = Conv2d(mid, 4*mid): its 4x
is a PixelShuffleND(2) OUTPUT layout consumed in place (:126-130), not a concatenation any
packaging spells apart." \
  --ratified "SETS: none - one component, one file, always whole." \
  --ratified "SCOPE: latent_upsampler ONLY, and the exclusion is real work rather than argument - the repo
root is \`latent_upsampler\`, \`vae\`, \`model_index.json\`, and this tree shares ZERO keys with
ltx-2.diffusers-bf16@1." \
  --ratified "ROLES stay literal. The state_dict keys ARE the module attribute names and exactly one
packaging of this component exists today; a rename now would be a guess about a spelling
nobody has seen. The day Lightricks' native upsampler is SERVED, its document adopts these
role spellings and compose::derive moves between them - that is authored then, from the second
packaging's real header, and it is not owed now." \
  --ratified "THE LAYOUT IS CONDITIONAL ON ONE CONFIG KNOB, measured here and not previously recorded:
\`latent_upsampler/config.json\` carries \`use_rational_resampler: false\` while the class
DEFAULTS it to true, and true builds \`SpatialRationalResampler\` instead - \`upsampler.conv.*\` +
\`upsampler.blur_down.*\`, the SAME [4096, 1024, 3, 3] shape under a different NAME (:137-157).
A rational-resampler checkpoint therefore REFUSES against this document by name rather than
mis-loading, which is the safe direction, but it means this document describes the
\`use_rational_resampler: false\` packaging specifically." \
  --ratified "The config corroborates every count: in_channels 128, mid_channels 1024, num_blocks_per_stage
4, dims 3, spatial_upsample true, temporal_upsample false -> 4x8 + 4x8 + 8 = 72 tensors." \
  --ratified "VERDICT: \`satisfies\`, from the same \`Contract.Verdict\` call the bind gate makes (tensorhub
internal/bindgate/bindgate.go:317), over the real 72-tensor header of
dg845/LTX-2.3-Spatial-Upsampler-Diffusers:latent_upsampler." \
  --out $OUT/ltx-2-upsampler.diffusers-bf16.v1.json
fi

TARGET=qwen-a3b; if want "$@"; then
$GEN --name qwen3.6-35b-a3b.vllm-fp8 --version 1 --pin-trivial --dtype float8_e4m3fn \
  --source fp8=hf:Qwen/Qwen3.6-35B-A3B-FP8 \
  --summary "Qwen3.6-35B-A3B-FP8: a hybrid linear-attention / full-attention MoE served by vLLM,
sharded as outside.safetensors + layers-N.safetensors + mtp.safetensors. MIXED PACKAGING, and it
is expressible exactly as sdxl.diffusers-fp8-rowwise@1 already is: the top-level dtype is the
LANE's load dtype while the per-tensor dtypes carry the matcher's constraint, so a flat
float8_e4m3fn lane legitimately holds BF16 declarations. The fp8 weights come with
weight_scale_inv twins; the vision tower and every norm stay BF16, which config.json's
quantization_config.modules_to_not_convert corroborates. The top-level dtype is DECLARED rather
than derived here: BF16 outnumbers F8_E4M3 by tensor COUNT, and a majority vote over counts is
the wrong instrument for 'what does this lane load as'." \
  --ratified "PROVENANCE: ratified 2026-08-20 by the tensorfs#150 contract-ratification lane, against the
model's own implementation source and the real header - and it is the document of the nine
where the bytes had to change." \
  --ratified "NO LOADER CONSUMES THIS TODAY - confirmed, and it is not read as asserting a pytorch streaming
load. The endpoint declares \`self_loading=\` and hands the checkpoint directory to
\`ctx.engine(VllmServer(...))\`; \`Qwen36A3b\` carries a \`contracts=\` FINGERPRINT pattern and no
canonical lane, \`lanes=()\` is the endpoint's state and \`ctx.lane\` would raise
(python-gen-worker src/gen_worker/models/model_types.py:2008-2047;
serverless-endpoints/qwen3.6-35b-a3b/src/qwen36_35b_a3b/main.py). The document answers
checkpoint compatibility and lane selection only." \
  --ratified "SELF-REFUSAL, FIXED. Every declaration was \`required\` across 42 shards, so the document
refused its own checkpoint (\`in layers-0.safetensors, requires 'lm_head.weight', which no
tensor satisfies\`) - matching is per member file (tensorhub
internal/bindgate/bindgate.go:388-417) and a multi-member document is all-optional by
convention (verdict.go). Declarations are now optional." \
  --ratified "FUSIONS: THREE, declared, each proved from the module definition AND shape arithmetic - this
is the family the ~90%-silent-split warning was written for. (a)
\`model.visual.blocks.{i}.attn.qkv.weight\` [3456, 1152] and \`.bias\`: \`nn.Linear(dim, dim * 3)\`
reshaped \`(seq, 3, num_heads, -1)\`, so flat q|k|v in EQUAL THIRDS
(transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:989, :1005-1007). (b)
\`linear_attn.in_proj_qkv.weight\` [8192, 2048]: \`nn.Linear(hidden, key_dim*2 + value_dim)\`
split \`[key_dim, key_dim, value_dim]\` (:420, :489-494). With linear_num_key_heads 16,
linear_num_value_heads 32 and both head dims 128 that is 2048|2048|4096 - shares **1:1:2, NOT
equal thirds**, and 8192/3 is not even an integer. (c) \`self_attn.q_proj.weight\` [8192, 2048]
(language AND mtp): \`nn.Linear(hidden, num_attention_heads * head_dim * 2)\` viewed \`(..., -1,
head_dim*2)\` and chunked on the LAST axis (:644-646, :672-675), so it is q and an output GATE
interleaved HEAD-MAJOR - 16 groups of (q, gate), 256 rows each. A flat two-way split of this
tensor is right for head 0 and wrong for the other fifteen, and nothing crashes. The fp8
\`weight_scale_inv\` twins split proportionally (64 = 16+16+32 blocks for in_proj_qkv) but carry
no declared fusion: a scale table's block granularity is not the weight's byte seam." \
  --ratified "NOT FUSED, checked rather than assumed: the MoE ships \`experts.{i}.{gate,up,down}_proj\` as
separate [512, 2048] tensors, so there is no gate-up seam on disk even though transformers
re-packs one at load (:734)." \
  --ratified "ROLES stay literal - one on-disk packaging." \
  --ratified "SETS: none, though the shard split (outside / layers-N / mtp) is a natural future subset if
MTP ever becomes optional." \
  --ratified "COMPONENT SCOPE: the whole tree is one checkpoint vLLM self-loads; the vision tower's BF16
exclusion is config.json's \`quantization_config.modules_to_not_convert\`, not an omission." \
  --ratified "VERDICT: \`satisfies\` after the fix, from the same \`Contract.Verdict\` call the bind gate makes
(bindgate.go:317), over the real 64196-tensor / 42-member header - and the declared fusions
divide it: 3456 % 3, 8192 % 4 and 8192 % 32 are all 0." \
  --out $OUT/qwen3.6-35b-a3b.vllm-fp8.v1.json
# `fusion` is the one field the generator refuses to guess, so the ratified
# answer is applied here rather than lost on every re-run.
nice -n 19 python3 scripts/declare-ratified-fusions.py $OUT/qwen3.6-35b-a3b.vllm-fp8.v1.json
fi

TARGET=qwen-mtp; if want "$@"; then
$GEN --name qwen3.6-27b-mtp.gguf-ud-q4-k-xl --version 1 --pin-trivial \
  --dtype q4_k --dtype-unknown-to-gen-worker \
  --source ud-q4-k-xl=hf:unsloth/Qwen3.6-27B-MTP-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --summary "Qwen3.6-27B-MTP as the GGUF the fleet actually serves - unsloth's UD-Q4_K_XL, the
quant the endpoint names. TOP-LEVEL DTYPE IS q4_k, the ggml type name, and this document is where
the field stops meaning 'a torch spelling' and starts meaning what it always actually was: THE
LANE'S QUANTIZATION. The earlier reading - that a block-quant container has no torch spelling and
therefore no declarable dtype - was true about torch and wrong about the field, and it made this
lane UNDECLARABLE under pgw#1597's always-required lanes=, which is a refusal with no remedy
rather than an honest absence. q4_k is the container the endpoint names (UD-Q4_K_XL) and the
plurality of the file's own tensors; the rest are the k-quant mix an unsloth UD build makes,
declared per tensor where the format already admits ggml type names. TWO CONSEQUENCES, RECORDED
SO NEITHER LOOKS LIKE A BUG. (1) Contract.torch_dtype REFUSES this spelling with MissingDtype,
which is correct: there is no torch scalar type for a k-quant block, and a typed refusal naming
the spelling beats resolving to a wrong dtype. Nothing on this path calls it - llama.cpp
self-loads the file and ctx.load is never invoked. (2) gen-worker's DTYPE_MIN_SM does not know
q4_k, so capability_floor_for_dtype answers 0. Zero is the RIGHT number - llama.cpp runs k-quants
on any card - but it is currently reached by that table's unknown-is-silent default rather than
by a decision, so gen-worker should learn q4_k: 0 explicitly. Until it does, the right answer is
an accident. STRUCTURE READ OUT OF THE HEADER, not out of a config: 48
blocks carry the ssm_* tensors and a fused attn_qkv, while the others carry split
attn_q/attn_k/attn_v with q/k norms - the full-attention interval, measured. The nextn.* block
is the MTP head." \
  --ratify "This lane is served by an EXTERNAL BINARY (llama.cpp), which never calls ctx.load. The
document answers checkpoint compatibility and lane selection only; it asserts no pytorch load
path. tensorfs already ships a real gguf-v1 planner profile, so storage was never the gap." \
  --out $OUT/qwen3.6-27b-mtp.gguf-ud-q4-k-xl.v1.json
fi

echo "done"
