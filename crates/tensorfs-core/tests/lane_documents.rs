//! #113: the serving-lane contract document `sdxl.diffusers-bf16.v1`.
//!
//! The lane surface references this document as the canonical SDXL serve
//! layout. Three claims: (1) each component file of a diffusers multifolder
//! SDXL tree DETECTS as it — and an all-optional document never vacuously
//! stamps a file it explains nothing of; (2) the top-level `dtype` field is
//! additive and readable; (3) single-file -> diffusers conversion of the
//! CLIP-G tensors is a run-preserving derive: role-set agreement with the
//! sdxl.clip-g-* vocabulary moves ZERO data objects.

#![cfg(any(unix, windows))]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::compose::derive;
use tensorfs_core::contract::Registry;
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::{PlannerId, inventory};
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{FileBody, FileRecord};

const LANE: &str = "sdxl.diffusers-bf16@1";
const MIB: u64 = 1024 * 1024;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-lane-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// fixtures: faithful diffusers KEY SPELLINGS, rank-correct small shapes
// ---------------------------------------------------------------------------

/// Bytes per element for the fixture dtypes.
fn itemsize(dtype: &str) -> u64 {
    match dtype {
        "BF16" | "F16" => 2,
        other => panic!("fixture dtype {other}"),
    }
}

fn deterministic(seed: u64, length: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut bytes = vec![0_u8; length];
    for chunk in bytes.chunks_mut(8) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        chunk.copy_from_slice(&state.to_le_bytes()[..chunk.len()]);
    }
    bytes
}

fn safetensors(dtype: &str, tensors: &[(&str, Vec<u64>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    let mut payload = Vec::new();
    for (index, (name, shape)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let elements: u64 = shape.iter().product();
        let length = elements * itemsize(dtype);
        let end = offset + length;
        let dimensions: Vec<String> = shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            dimensions.join(",")
        ));
        payload.extend_from_slice(&deterministic(index as u64 + 7, length as usize));
        offset = end;
    }
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    file
}

fn unet_file(dtype: &str) -> Vec<u8> {
    safetensors(
        dtype,
        &[
            ("add_embedding.linear_1.weight", vec![8, 4]),
            ("add_embedding.linear_2.weight", vec![8, 8]),
            ("time_embedding.linear_1.weight", vec![8, 4]),
            ("time_embedding.linear_2.weight", vec![8, 8]),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
                vec![8, 8],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight",
                vec![8, 8],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.weight",
                vec![8, 8],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
                vec![8, 8],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight",
                vec![8, 16],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
                vec![64, 8],
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight",
                vec![8, 32],
            ),
            (
                "mid_block.attentions.0.transformer_blocks.1.attn1.to_q.weight",
                vec![8, 8],
            ),
            (
                "up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_v.weight",
                vec![8, 16],
            ),
            // Undeclared by the contract: matching must skip it, not refuse.
            ("conv_in.weight", vec![8, 4, 3, 3]),
        ],
    )
}

fn vae_file() -> Vec<u8> {
    safetensors(
        "BF16",
        &[
            ("quant_conv.weight", vec![8, 8, 1, 1]),
            ("post_quant_conv.weight", vec![4, 4, 1, 1]),
            ("encoder.conv_in.weight", vec![8, 3, 3, 3]),
            ("decoder.conv_in.weight", vec![8, 4, 3, 3]),
            (
                "encoder.down_blocks.0.resnets.1.conv1.weight",
                vec![8, 8, 3, 3],
            ),
            (
                "decoder.up_blocks.2.resnets.0.conv2.weight",
                vec![8, 8, 3, 3],
            ),
            ("decoder.mid_block.attentions.0.to_q.weight", vec![8, 8]),
        ],
    )
}

/// The diffusers CLIP spelling. With projection = text_encoder_2 (CLIP-G);
/// without = text_encoder (CLIP-L), which shares every other key spelling.
fn text_encoder_file(projection: bool) -> Vec<u8> {
    let mut tensors: Vec<(String, Vec<u64>)> = Vec::new();
    for layer in 0..2 {
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            tensors.push((
                format!("text_model.encoder.layers.{layer}.self_attn.{projection}.weight"),
                vec![8, 8],
            ));
            tensors.push((
                format!("text_model.encoder.layers.{layer}.self_attn.{projection}.bias"),
                vec![8],
            ));
        }
        tensors.push((
            format!("text_model.encoder.layers.{layer}.mlp.fc1.weight"),
            vec![32, 8],
        ));
        tensors.push((
            format!("text_model.encoder.layers.{layer}.mlp.fc2.weight"),
            vec![8, 32],
        ));
    }
    if projection {
        tensors.push(("text_projection.weight".to_owned(), vec![8, 8]));
    }
    let borrowed: Vec<(&str, Vec<u64>)> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.clone()))
        .collect();
    safetensors("BF16", &borrowed)
}

fn detect(bytes: &[u8]) -> String {
    let registry = Registry::builtin().expect("the shipped contracts parse");
    let taken = inventory(bytes)
        .expect("header reads")
        .expect("a tensor container");
    registry.detect(&taken).stamp().to_string()
}

// ---------------------------------------------------------------------------
// detection
// ---------------------------------------------------------------------------

#[test]
fn every_component_file_of_a_diffusers_sdxl_tree_detects_the_lane_document() {
    assert_eq!(detect(&unet_file("BF16")), LANE, "unet");
    assert_eq!(detect(&vae_file()), LANE, "vae");
    // Both text encoders match — the diffusers CLIP spelling is shared — and
    // the lane document beats sdxl.clip-g-split-qkv on specificity: it
    // explains out_proj, the mlp and the projection head, not just qkv.
    assert_eq!(detect(&text_encoder_file(true)), LANE, "text_encoder_2");
    assert_eq!(detect(&text_encoder_file(false)), LANE, "text_encoder");
}

#[test]
fn the_bf16_constraint_is_falsifiable_from_the_header() {
    // The same unet spelled F16 is NOT the bf16 packaging: the per-tensor
    // dtype constraint refuses, and no other shipped contract claims it.
    assert_eq!(detect(&unet_file("F16")), "none");
}

#[test]
fn an_all_optional_document_never_vacuously_stamps_a_foreign_file() {
    // The lane document declares every family optional (matching is per
    // component file). A container it explains NOTHING of must stay
    // contract:none — matched == 0 is a non-match, not a weak match.
    let foreign = safetensors("BF16", &[("foo.weight", vec![8, 8]), ("bar.bias", vec![8])]);
    assert_eq!(detect(&foreign), "none");
}

#[test]
fn the_lane_document_carries_the_load_dtype() {
    let registry = Registry::builtin().expect("parses");
    let stamp = tensorfs_core::contract::Stamp::parse(LANE).unwrap();
    let lane = registry.get(&stamp).expect("shipped");
    assert_eq!(lane.dtype(), Some("bfloat16"));
    // And the pre-existing library is untouched by the field's existence:
    // the digest pin test is the byte-level proof; this is the read-side.
    let fused = registry
        .get(&tensorfs_core::contract::Stamp::parse("sdxl.clip-g-fused-qkv@1").unwrap())
        .expect("shipped");
    assert_eq!(fused.dtype(), None);
}

// ---------------------------------------------------------------------------
// the derive proof: single-file CLIP-G -> diffusers, zero data moved
// ---------------------------------------------------------------------------

#[test]
fn single_file_clip_g_derives_into_the_lane_document_moving_no_data() {
    // 12 rows of 1 MiB: shares 1:1:1 give three 4 MiB projections, all above
    // the seam floor, so the fused contract cuts q|k|v exactly.
    let rows = 12_u64;
    let columns = MIB / 2; // BF16: 2 bytes/element -> 1 MiB rows
    let fused = safetensors(
        "BF16",
        &[(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            vec![rows, columns],
        )],
    );
    assert_eq!(
        detect(fused.as_slice()),
        "sdxl.clip-g-fused-qkv@1",
        "the single-file spelling detects the fused contract"
    );

    let root = TempRoot::new("derive");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let registry = Registry::builtin().expect("parses");
    let source_contract = registry
        .get(&tensorfs_core::contract::Stamp::parse("sdxl.clip-g-fused-qkv@1").unwrap())
        .expect("shipped");
    let target_contract = registry
        .get(&tensorfs_core::contract::Stamp::parse(LANE).unwrap())
        .expect("shipped");

    let plan = tensorfs_core::planner::plan_with(fused.as_slice(), Some(source_contract))
        .expect("plans under the fused contract");
    let admitted = store
        .admit_regions(fused.as_slice(), plan.regions())
        .expect("admits");
    let body = FileBody::Tensor {
        format: PlannerId::SafetensorsV1.tensor_format().unwrap(),
        contract: plan.contract().clone(),
        logical_size: fused.len() as u64,
        records: admitted
            .iter()
            .map(|object| FileRecord::Data {
                digest: object.digest(),
                length: object.length(),
            })
            .collect(),
    };
    let before: HashSet<ObjectDigest> = data_digests(&body);

    let derived = derive(&store, &body, source_contract, target_contract).expect("derives");
    let after = data_digests(&derived);
    let fresh: Vec<&ObjectDigest> = after.difference(&before).collect();
    assert_eq!(
        fresh.len(),
        1,
        "one new header object and ZERO data objects: role-set agreement \
         makes single-file -> diffusers a rekey, not a conversion"
    );

    let FileBody::Tensor {
        contract: stamp, ..
    } = &derived
    else {
        panic!("still a tensor container");
    };
    assert_eq!(stamp.to_string(), LANE);

    // The derived container spells the diffusers split projections.
    let source = tensorfs_core::workspace_source::RecordsSource::new(&store, &derived.records());
    let taken = inventory(&source).expect("reads").expect("a container");
    let names: Vec<&str> = taken.tensors().iter().map(|tensor| tensor.name()).collect();
    assert_eq!(
        names,
        [
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            "text_model.encoder.layers.0.self_attn.k_proj.weight",
            "text_model.encoder.layers.0.self_attn.v_proj.weight",
        ]
    );
}

fn data_digests(body: &FileBody) -> HashSet<ObjectDigest> {
    body.records()
        .iter()
        .filter_map(|record| match record {
            FileRecord::Data { digest, .. } => Some(*digest),
            FileRecord::Hole { .. } => None,
        })
        .collect()
}
