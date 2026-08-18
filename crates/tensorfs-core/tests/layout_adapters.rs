//! #81: serving one layout contract from another contract's bytes.
//!
//! Two claims. First, the decision is MECHANICAL: from the two contracts and
//! the source header — same roles, same dtypes, same element counts — the
//! system answers "view or convert?" without reading a tensor byte. Second,
//! the run-preserving majority costs nothing: a derived container inherits
//! every data object by digest and admits one new header, so a rename or a
//! fuse/split between contracts is storage-free. Only genuinely re-arranged
//! tensors (a generalized permute) are materialized, once, at definition
//! time — which is the load-order ruling, not an optimization.

#![cfg(any(unix, windows))]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::adapter::{Decision, Transform, decide, invert_axes, permute_bytes};
use tensorfs_core::compose::derive;
use tensorfs_core::contract::Contract;
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::{PlannerId, inventory};
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{FileBody, FileRecord};
use tensorfs_core::workspace_source::RecordsSource;

const RUN: u64 = 1024 * 1024;
const HEADS: u64 = 4;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-adapters-{name}-{}", std::process::id()));
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
// contracts
// ---------------------------------------------------------------------------

/// Fused, head-interleaved qkv plus a plainly named mlp: the native spelling.
const NATIVE: &str = r#"{
    "format": "tensorfs-contract-v1",
    "name": "test.native",
    "version": 1,
    "tensors": [
        {"role": "blocks.{i}.attn.qkv", "pattern": "blocks.{i}.attn.qkv_proj.weight",
         "rank": 2,
         "fusion": {"axis": 0, "groups": 4,
                    "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1},
                              {"role": "v", "share": 1}]}},
        {"role": "blocks.{i}.mlp", "pattern": "blocks.{i}.mlp.fc1.weight", "rank": 2}
    ]
}"#;

/// The same weights, split projections and diffusers spelling.
const DIFFUSERS: &str = r#"{
    "format": "tensorfs-contract-v1",
    "name": "test.diffusers",
    "version": 1,
    "tensors": [
        {"role": "blocks.{i}.attn.qkv#q", "pattern": "transformer_blocks.{i}.attn.to_q.weight",
         "rank": 2, "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.attn.qkv#k", "pattern": "transformer_blocks.{i}.attn.to_k.weight",
         "rank": 2, "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.attn.qkv#v", "pattern": "transformer_blocks.{i}.attn.to_v.weight",
         "rank": 2, "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.mlp", "pattern": "transformer_blocks.{i}.ff.net.0.proj.weight",
         "rank": 2}
    ]
}"#;

/// A third layout that keeps every role but rope-permutes the mlp tensor:
/// same bytes, different order INSIDE the tensor.
const PERMUTED: &str = r#"{
    "format": "tensorfs-contract-v1",
    "name": "test.permuted",
    "version": 1,
    "tensors": [
        {"role": "blocks.{i}.attn.qkv", "pattern": "blocks.{i}.attn.qkv_proj.weight",
         "rank": 2,
         "fusion": {"axis": 0, "groups": 4,
                    "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1},
                              {"role": "v", "share": 1}]}},
        {"role": "blocks.{i}.mlp", "pattern": "blocks.{i}.mlp.fc1.weight", "rank": 2,
         "permute": {"view": ["auto", 2, 2, "shape[1]"], "axes": [0, 2, 1, 3]}}
    ]
}"#;

fn contract(document: &str) -> Contract {
    Contract::parse(document).expect("the contract parses")
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn slice(part: u8, head: u64) -> Vec<u8> {
    let mut state = u64::from(part) << 32 | head;
    let mut bytes = vec![0_u8; RUN as usize];
    for chunk in bytes.chunks_mut(8) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        chunk.copy_from_slice(&state.to_le_bytes()[..chunk.len()]);
    }
    bytes
}

fn safetensors(tensors: &[(&str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, shape, data)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let end = offset + data.len() as u64;
        let dimensions: Vec<String> = shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            dimensions.join(",")
        ));
        offset = end;
    }
    header.push('}');

    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    for (_, _, data) in tensors {
        file.extend_from_slice(data);
    }
    file
}

/// Four DISTINCT rows: a permute over identical rows would be invisible, and
/// a test that cannot see the transform proves nothing about it.
fn mlp_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in 0..4 {
        bytes.extend_from_slice(&slice(9, row));
    }
    bytes
}

/// A cheap content fingerprint, so a failed assertion prints a number rather
/// than four megabytes.
fn fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{DefaultHasher, Hash as _, Hasher as _};
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn native_file() -> Vec<u8> {
    let mut fused = Vec::new();
    for head in 0..HEADS {
        for part in 0..3_u8 {
            fused.extend_from_slice(&slice(part, head));
        }
    }
    safetensors(&[
        ("blocks.0.attn.qkv_proj.weight", vec![3 * HEADS, RUN], fused),
        ("blocks.0.mlp.fc1.weight", vec![4, RUN], mlp_bytes()),
    ])
}

fn admit(store: &ObjectStore, bytes: &[u8]) -> FileBody {
    let plan = tensorfs_core::planner::plan_with(bytes, Some(&contract(NATIVE))).expect("plans");
    let admitted = store
        .admit_regions(bytes, plan.regions())
        .expect("admits every region");
    FileBody::Tensor {
        format: PlannerId::SafetensorsV1.tensor_format().unwrap(),
        contract: plan.contract().clone(),
        logical_size: bytes.len() as u64,
        records: admitted
            .iter()
            .map(|object| FileRecord::Data {
                digest: object.digest(),
                length: object.length(),
            })
            .collect(),
    }
}

const fn record_length(record: &FileRecord) -> u64 {
    match record {
        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
    }
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

// ---------------------------------------------------------------------------
// the decision procedure
// ---------------------------------------------------------------------------

#[test]
fn the_viewable_question_is_answered_from_two_contracts_and_a_header() {
    let bytes = native_file();
    let taken = inventory(bytes.as_slice()).unwrap().unwrap();

    // Rename + split: every role survives, nothing is re-arranged.
    let decision = decide(&contract(NATIVE), &taken, &contract(DIFFUSERS)).expect("decides");
    let Decision::RunPreserving(adapter) = &decision else {
        panic!("a rename + split is run-preserving, got {decision:?}");
    };
    assert!(decision.is_viewable());
    let names: Vec<&str> = adapter.tensors().iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
        ]
    );
    // Each projection is 4 head slices of the fused tensor, gathered out of
    // an interleave — and the target's shape is derived, never declared.
    assert_eq!(adapter.tensors()[0].runs().len(), 4);
    assert_eq!(adapter.tensors()[0].shape(), [HEADS, RUN]);
    assert_eq!(adapter.tensors()[3].shape(), [4, RUN]);

    // A declared permute is the one thing that moves bytes inside a tensor.
    let permuted = decide(&contract(NATIVE), &taken, &contract(PERMUTED)).expect("decides");
    let Decision::Rearranged { rearranged, .. } = &permuted else {
        panic!("a permuted target is not run-preserving, got {permuted:?}");
    };
    assert_eq!(rearranged, &["blocks.0.mlp.fc1.weight"]);
    assert!(permuted.is_viewable(), "re-arranged is still servable");
}

#[test]
fn a_target_that_needs_bytes_we_do_not_hold_is_a_conversion_not_a_view() {
    let bytes = native_file();
    let taken = inventory(bytes.as_slice()).unwrap().unwrap();

    // A role the source cannot supply: the answer must be "convert", and it
    // must say which role is missing rather than failing obscurely.
    let extra = contract(&DIFFUSERS.replace(
        r#"{"role": "blocks.{i}.mlp", "pattern": "transformer_blocks.{i}.ff.net.0.proj.weight",
         "rank": 2}"#,
        r#"{"role": "blocks.{i}.mlp", "pattern": "transformer_blocks.{i}.ff.net.0.proj.weight",
         "rank": 2},
        {"role": "blocks.{i}.absent", "pattern": "transformer_blocks.{i}.absent.weight"}"#,
    ));
    let decision = decide(&contract(NATIVE), &taken, &extra).expect("decides");
    assert!(!decision.is_viewable());

    // And a target that silently drops weights is a different model, not a
    // view of this one.
    let partial = contract(
        &DIFFUSERS.replace("test.diffusers", "test.partial").replace(
            r#",
        {"role": "blocks.{i}.mlp", "pattern": "transformer_blocks.{i}.ff.net.0.proj.weight",
         "rank": 2}"#,
            "",
        ),
    );
    let dropped = decide(&contract(NATIVE), &taken, &partial).expect("decides");
    assert!(matches!(dropped, Decision::Conversion { .. }));
    let Decision::Conversion { reason } = dropped else {
        unreachable!()
    };
    assert!(
        reason.contains("blocks.0.mlp"),
        "names the orphan: {reason}"
    );
}

// ---------------------------------------------------------------------------
// the derived container
// ---------------------------------------------------------------------------

#[test]
fn a_run_preserving_derivation_admits_one_header_and_no_data() {
    let root = TempRoot::new("derive");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let bytes = native_file();
    let body = admit(&store, &bytes);
    let before = data_digests(&body);

    let derived = derive(&store, &body, &contract(NATIVE), &contract(DIFFUSERS)).expect("derives");
    let after = data_digests(&derived);

    // Every data object is the source's. The one new object is the header,
    // which is why the derived snapshot costs bytes in the KB, not the GB.
    let fresh: Vec<&ObjectDigest> = after.difference(&before).collect();
    assert_eq!(fresh.len(), 1, "one new header object and nothing else");
    assert_eq!(
        after.len(),
        before.len(),
        "13 objects in, 13 objects out — the header replaced the header"
    );

    // The derived file is an ordinary safetensors container the planner reads
    // back, naming the target layout's tensors, and it is stamped with the
    // contract whose boundaries it carries.
    let FileBody::Tensor {
        contract: stamp, ..
    } = &derived
    else {
        panic!("still a tensor container");
    };
    assert_eq!(stamp.to_string(), "test.diffusers@1");

    let source = RecordsSource::new(&store, &derived.records());
    let taken = inventory(&source).expect("reads").expect("a container");

    // A derived snapshot is INDISTINGUISHABLE from an ingested one: planning
    // the derived bytes under the target contract reproduces exactly the
    // record boundaries the derivation inherited. If it did not, a later seal
    // would re-chunk what a derivation just composed.
    let replanned = tensorfs_core::planner::plan_with(&source, Some(&contract(DIFFUSERS)))
        .expect("the derived file plans");
    let boundaries: Vec<u64> = derived.records().iter().map(record_length).collect();
    assert_eq!(
        boundaries,
        replanned
            .regions()
            .iter()
            .map(tensorfs_core::planner::Region::length)
            .collect::<Vec<_>>(),
        "a derivation composes the boundaries an ingest would produce"
    );

    let names: Vec<&str> = taken.tensors().iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
        ]
    );

    // And the bytes are right: to_q is the four q head-slices, in head order,
    // gathered out of the interleave.
    let mut expected = Vec::new();
    for head in 0..HEADS {
        expected.extend_from_slice(&slice(0, head));
    }
    let mut read = vec![0_u8; expected.len()];
    let q = &taken.tensors()[0];
    tensorfs_core::planner::ByteSource::read_exact_at(&source, q.offset(), &mut read)
        .expect("reads the derived tensor");
    assert_eq!(fingerprint(&read), fingerprint(&expected));
}

#[test]
fn a_derivation_into_a_nameless_custom_moves_no_data_and_stamps_the_digest() {
    // A custom contract needs NO prior platform knowledge for run-preserving
    // conversions: derive computes them from the two documents' role sets,
    // and the derived body is stamped with the custom's canonical digest.
    let root = TempRoot::new("derive-custom");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let bytes = native_file();
    let body = admit(&store, &bytes);
    let before = data_digests(&body);

    let custom_document = DIFFUSERS
        .replace("\"name\": \"test.diffusers\",\n", "")
        .replace("\"version\": 1,\n", "");
    let custom = contract(&custom_document);
    assert_eq!(custom.name(), None);

    let derived = derive(&store, &body, &contract(NATIVE), &custom).expect("derives");
    let after = data_digests(&derived);
    let fresh: Vec<&ObjectDigest> = after.difference(&before).collect();
    assert_eq!(
        fresh.len(),
        1,
        "one new header object and ZERO data objects"
    );

    let FileBody::Tensor {
        contract: stamp, ..
    } = &derived
    else {
        panic!("still a tensor container");
    };
    assert_eq!(*stamp, custom.stamp());
    assert!(stamp.to_string().starts_with("sha256:"), "{stamp}");
}

#[test]
fn a_re_arranged_tensor_is_materialized_once_and_is_exactly_invertible() {
    let root = TempRoot::new("permute");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let bytes = native_file();
    let body = admit(&store, &bytes);
    let before = data_digests(&body);

    let derived = derive(&store, &body, &contract(NATIVE), &contract(PERMUTED)).expect("derives");
    let after = data_digests(&derived);

    // The fused tensor is untouched and inherited; only the permuted mlp is
    // new, plus the header. That is the tier-1 cost: the re-arranged
    // fraction, never a second copy.
    let fresh: Vec<&ObjectDigest> = after.difference(&before).collect();
    assert_eq!(fresh.len(), 2, "the header and the one permuted tensor");
    assert_eq!(
        before
            .iter()
            .filter(|digest| after.contains(digest))
            .count(),
        12,
        "every qkv head slice is inherited verbatim"
    );

    // The materialized bytes are the declared permute of the source's, and
    // the permute is exactly invertible — no value was touched, only moved.
    let source = RecordsSource::new(&store, &derived.records());
    let taken = inventory(&source).expect("reads").expect("a container");
    let mlp = taken
        .tensors()
        .iter()
        .find(|item| item.name() == "blocks.0.mlp.fc1.weight")
        .expect("the mlp survives");
    let mut read = vec![0_u8; mlp.length() as usize];
    tensorfs_core::planner::ByteSource::read_exact_at(&source, mlp.offset(), &mut read)
        .expect("reads the materialized tensor");

    let dims = [1_u64, 2, 2, RUN];
    let expected = permute_bytes(&mlp_bytes(), 1, &dims, &[0, 2, 1, 3]);
    assert_eq!(fingerprint(&read), fingerprint(&expected));
    assert_ne!(
        fingerprint(&read),
        fingerprint(&mlp_bytes()),
        "the permute is not the identity"
    );

    let permuted_dims: Vec<u64> = [0, 2, 1, 3].iter().map(|axis| dims[*axis]).collect();
    assert_eq!(
        fingerprint(&permute_bytes(
            &read,
            1,
            &permuted_dims,
            &invert_axes(&[0, 2, 1, 3])
        )),
        fingerprint(&mlp_bytes()),
        "the round trip returns the source bytes exactly"
    );
}

#[test]
fn a_corrupted_permute_declaration_changes_the_bytes_it_produces() {
    // The red proof for the adapter: a permute that names different axes
    // produces a different tensor. If the transform were being ignored, both
    // would come back as the source bytes and this would not discriminate.
    let identityish = Transform::Forward(
        match contract(&PERMUTED.replace("[0, 2, 1, 3]", "[0, 1, 3, 2]"))
            .entry_for("blocks.0.mlp.fc1.weight")
            .and_then(|entry| entry.permute().cloned())
        {
            Some(permute) => permute,
            None => panic!("the contract declares a permute"),
        },
    );
    let declared = Transform::Forward(
        contract(PERMUTED)
            .entry_for("blocks.0.mlp.fc1.weight")
            .and_then(|entry| entry.permute().cloned())
            .expect("the contract declares a permute"),
    );
    let shape = [4_u64, RUN];
    let honest = declared.apply(&mlp_bytes(), 1, &shape).expect("applies");
    let corrupted = identityish.apply(&mlp_bytes(), 1, &shape).expect("applies");
    assert_ne!(fingerprint(&honest), fingerprint(&corrupted));
    assert_ne!(fingerprint(&honest), fingerprint(&mlp_bytes()));
}
