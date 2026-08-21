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
  --ratify "The 'model.diffusion_model.*' packaging (anima-turbo-v1.0, anima-aesthetic-v1.0/v1.0b/v1.1,
anima-preview*) is NOT covered here. Its document is a pure rekey of this one, so compose::derive
can answer it - author it, do not widen this one." \
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
  --ratified "ROLES: role == pattern is CORRECT here rather than merely mechanical. A role
vocabulary exists so a SECOND packaging of the same bytes can be derived instead of refused, and
this artifact cannot acquire one: we produce it, from a pickle, with save_artifact - there is no
other spelling in the world to reconcile with." \
  --ratified "FUSIONS: none, and none is EXPRESSIBLE. Every declaration is a plain conv weight or
bias, a PReLU beta, or the PixelShuffle head; IFNet has no attention and no concatenated
projection, so no tensor's outer axis is an ordered concatenation of another packaging's runs." \
  --ratified "SETS: none. Subset snapshots have no subject in a 5.66M-parameter net that is always
loaded whole." \
  --ratified "SCOPE: the whole artifact, which is one component (flownet) of a one-component
pipeline. It ties this document to no other family because it IS this platform's artifact -
produced, not mirrored - so the tensorfs#121 shared-component hazard has no instance here." \
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
  --ratify "The repo also ships a state_dict.bin / compression_state_dict.bin pair (the audiocraft
packaging) and an fp32 index. Only the safetensors packaging is described." \
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
  --ratify "SCOPE CHECK owed against tensorfs#121: 'vision_tower.*' is a stock SigLIP spelling and
'language_model.*' a stock Llama one, so the discriminating fact is the COMBINATION plus
'multi_modal_projector.*'. Confirm no other served family ships the same three-prefix tree before
this document joins the global matcher set." \
  --out $OUT/joycaption.llava-bf16.v1.json
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
  --ratify "CONFIRM against the mirror when a hub is reachable: this document is derived from
upstream headers, and the claim that the mirror's TRANSFORMER headers are byte-identical to
upstream's is inferred from the defect being config-scoped, not measured. If the mirror's
transformer differs, the mirror is what the fleet serves and the document follows the bytes." \
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
  --ratified "FUSIONS: none, and none is EXPRESSIBLE. LTX2LatentUpsamplerModel is a pure 3-D
convnet - initial_conv/initial_norm, res_blocks and post_upsample_res_blocks (conv1/conv2 +
norm1/norm2), the upsampler head, final_conv. There is no attention and no concatenated
projection in the header, so there is no outer-axis concatenation to declare." \
  --ratified "SETS: none." \
  --ratified "SCOPE: latent_upsampler only, and the exclusion is MEASURED rather than argued -
this tree shares ZERO keys with ltx-2.diffusers-bf16@1, and the sibling vae in the same repo is
the LTX-2 pipeline's own, so declaring it would tie this document to the DiT's checkpoint." \
  --ratify "ROLES remain mechanical (role == pattern), and unlike rife this family CAN acquire a
second packaging - Lightricks ships native checkpoints beside the diffusers ones. If a native
upsampler is ever served, its roles must be reconciled with these before compose::derive can move
between them. Nothing is wrong today; the work is owed the day a second packaging appears." \
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
  --ratify "vLLM self-loads a directory, so ctx.load is never called on this path and no loader
consumes this document TODAY. It is authored because lanes= answers checkpoint compatibility and
lane selection independently of compilation (pgw#1597) - it must not be read as asserting a
pytorch streaming load this endpoint does not perform." \
  --out $OUT/qwen3.6-35b-a3b.vllm-fp8.v1.json
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
