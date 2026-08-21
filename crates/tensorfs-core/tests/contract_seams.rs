//! #81: contract-directed chunking, proven on bytes.
//!
//! The claim under test is the memo's §4: a fused packaging and a split
//! packaging of the same weights share EVERY data object, because the
//! contract cuts the fused tensor at its declared seams before the 64 MiB
//! grid — including when the fusion is head-interleaved, as MiniMax-H3's is.
//! Removing the contract is the red proof: sharing collapses to zero for the
//! fused tensor and to nothing else.

#![cfg(any(unix, windows))]

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::compose::adopt;
use tensorfs_core::contract::{Contract, MIN_SEAM_PART_BYTES, Registry, Stamp};
use tensorfs_core::object::{ObjectDigest, plan_and_hash_under};
use tensorfs_core::planner::{PlannerId, RegionKind, inventory};
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{FileBody, FileRecord, SnapshotBuilder};

const RUN: u64 = MIN_SEAM_PART_BYTES;
const HEADS: u64 = 4;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-seams-{name}-{}", std::process::id()));
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
// contracts under test
// ---------------------------------------------------------------------------

/// The native/fused spelling: one head-interleaved qkv per block, exactly the
/// shape `h3_native_layout.fuse_qkv_head_interleaved` produces.
const FUSED: &str = r#"{
    "format": "tensorfs-contract-v1",
    "name": "test.h3-fused",
    "version": 1,
    "tensors": [
        {"role": "blocks.{i}.attn.qkv", "pattern": "blocks.{i}.attn.qkv_proj.weight",
         "rank": 2,
         "fusion": {"axis": 0, "groups": 4,
                    "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1},
                              {"role": "v", "share": 1}]}},
        {"role": "blocks.{i}.mlp", "pattern": "blocks.{i}.mlp.fc1.weight", "required": false}
    ]
}"#;

/// The split spelling. Each projection declares the SAME runs — 4 head
/// slices — so both packagings cut at the same places.
const SPLIT: &str = r#"{
    "format": "tensorfs-contract-v1",
    "name": "test.h3-split",
    "version": 1,
    "tensors": [
        {"role": "blocks.{i}.attn.qkv#q", "pattern": "blocks.{i}.attn.to_q.weight", "rank": 2,
         "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.attn.qkv#k", "pattern": "blocks.{i}.attn.to_k.weight", "rank": 2,
         "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.attn.qkv#v", "pattern": "blocks.{i}.attn.to_v.weight", "rank": 2,
         "fusion": {"axis": 0, "groups": 4, "parts": [{"role": "", "share": 1}]}},
        {"role": "blocks.{i}.mlp", "pattern": "blocks.{i}.mlp.fc1.weight", "required": false}
    ]
}"#;

fn contract(document: &str) -> Contract {
    Contract::parse(document).expect("the contract parses")
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// One head slice of one projection: deterministic bytes, distinct per slice.
fn slice(part: u8, head: u64) -> Vec<u8> {
    let mut state = u64::from(part) << 32 | head;
    let mut bytes = vec![0_u8; RUN as usize];
    for chunk in bytes.chunks_mut(8) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let word = state.to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
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

/// The fused file: `q0 k0 v0 q1 k1 v1 …` — head-major, NOT three stacked
/// blocks. Getting this order wrong is the ~90% error `h3-key-sets.md`
/// measured, which is why the seam table has to express it.
fn fused_file() -> Vec<u8> {
    let mut data = Vec::new();
    for head in 0..HEADS {
        for part in 0..3_u8 {
            data.extend_from_slice(&slice(part, head));
        }
    }
    safetensors(&[
        ("blocks.0.attn.qkv_proj.weight", vec![3 * HEADS, RUN], data),
        (
            "blocks.0.mlp.fc1.weight",
            vec![2, RUN],
            slice(9, 0).repeat(2),
        ),
    ])
}

/// The split twin: the same bytes, one tensor per projection.
fn split_file() -> Vec<u8> {
    let projection = |part: u8| {
        let mut data = Vec::new();
        for head in 0..HEADS {
            data.extend_from_slice(&slice(part, head));
        }
        data
    };
    safetensors(&[
        ("blocks.0.attn.to_q.weight", vec![HEADS, RUN], projection(0)),
        ("blocks.0.attn.to_k.weight", vec![HEADS, RUN], projection(1)),
        ("blocks.0.attn.to_v.weight", vec![HEADS, RUN], projection(2)),
        (
            "blocks.0.mlp.fc1.weight",
            vec![2, RUN],
            slice(9, 0).repeat(2),
        ),
    ])
}

fn tensor_digests(bytes: &[u8], contract: Option<&Contract>) -> HashSet<ObjectDigest> {
    plan_and_hash_under(bytes, contract)
        .expect("plans")
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .map(|object| object.digest())
        .collect()
}

// ---------------------------------------------------------------------------
// the claim
// ---------------------------------------------------------------------------

#[test]
fn a_fused_packaging_and_its_split_twin_share_every_data_object() {
    let fused = tensor_digests(&fused_file(), Some(&contract(FUSED)));
    let split = tensor_digests(&split_file(), Some(&contract(SPLIT)));

    // 12 head slices + the shared mlp tensor, on both sides.
    assert_eq!(fused.len(), 13);
    assert_eq!(split.len(), 13);
    assert_eq!(
        fused, split,
        "the fused file's objects must BE the split file's objects"
    );
}

#[test]
fn without_the_contract_the_fused_tensor_shares_nothing() {
    // The red proof. Same bytes, same planner, no seam table: the fused
    // tensor becomes one object nobody else holds, and the sharing that the
    // contract bought — every attention byte — is gone.
    let plain_fused = tensor_digests(&fused_file(), None);
    let plain_split = tensor_digests(&split_file(), None);
    let shared: Vec<&ObjectDigest> = plain_fused.intersection(&plain_split).collect();

    assert_eq!(plain_fused.len(), 2, "one fused tensor, one mlp tensor");
    assert_eq!(plain_split.len(), 4);
    assert_eq!(
        shared.len(),
        1,
        "only the untouched mlp tensor survives without a contract"
    );

    // And it is exactly the seam-covered bytes that were lost: 12 MiB.
    let seamed = tensor_digests(&fused_file(), Some(&contract(FUSED)));
    let seamed_split = tensor_digests(&split_file(), Some(&contract(SPLIT)));
    assert_eq!(seamed.intersection(&seamed_split).count(), 13);
}

#[test]
fn a_part_larger_than_the_grid_is_gridded_identically_in_both_packagings() {
    // A seam part is planned exactly as a standalone tensor would be: the
    // 64 MiB grid restarts at the part's own start. So an oversized part
    // splits the same way on both sides, and the pieces still match.
    const BIG: u64 = 64 * 1024 * 1024 + 1024 * 1024;
    let block = |fill: u8, length: u64| vec![fill; length as usize];
    let rows = |length: u64| length / 1024;

    let fused = safetensors(&[(
        "blocks.0.attn.qkv_proj.weight",
        vec![rows(BIG) + 2 * rows(RUN), 1024],
        [block(1, BIG), block(2, RUN), block(3, RUN)].concat(),
    )]);
    let split = safetensors(&[
        (
            "blocks.0.attn.to_q.weight",
            vec![rows(BIG), 1024],
            block(1, BIG),
        ),
        (
            "blocks.0.attn.to_k.weight",
            vec![rows(RUN), 1024],
            block(2, RUN),
        ),
        (
            "blocks.0.attn.to_v.weight",
            vec![rows(RUN), 1024],
            block(3, RUN),
        ),
    ]);

    // Shares 65:1:1 in one group -- plain stacking, oversized first part.
    let oversized = contract(&FUSED.replace("\"groups\": 4,", "").replace(
        "{\"role\": \"q\", \"share\": 1}",
        "{\"role\": \"q\", \"share\": 65}",
    ));
    let plain_split = contract(&SPLIT.replace(
        "\"fusion\": {\"axis\": 0, \"groups\": 4, \"parts\": [{\"role\": \"\", \"share\": 1}]}",
        "\"required\": true",
    ));

    let left = tensor_digests(&fused, Some(&oversized));
    let right = tensor_digests(&split, Some(&plain_split));
    assert_eq!(
        left.len(),
        4,
        "65 MiB grids into 64 MiB + 1 MiB, plus k and v"
    );
    assert_eq!(left, right);
}

#[test]
fn a_declaration_below_the_run_floor_cuts_nothing_on_either_side() {
    // The memo's rejection of row-granular splitting, as a size: the same
    // contracts that cut MiB-scale runs cut nothing when the runs are small,
    // and they cross the floor at the same size in both packagings.
    let tiny_fused = safetensors(&[(
        "blocks.0.attn.qkv_proj.weight",
        vec![3 * HEADS, 8],
        vec![7_u8; (3 * HEADS * 8) as usize],
    )]);
    let plan = plan_and_hash_under(tiny_fused.as_slice(), Some(&contract(FUSED)))
        .expect("plans")
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .count();
    assert_eq!(plan, 1, "a KB-scale interleave is not seam territory");
}

/// A GGUF v3 file holding one 2-D `I8` tensor: metadata, directory, alignment
/// padding, then the tensor. `ne` is fastest-varying first, which is the point
/// of the fixture — the contract vocabulary is logical row-major, so the
/// planner has to reverse it before a seam declaration can apply.
fn gguf(name: &str, rows: u64, columns: u64, data: &[u8]) -> Vec<u8> {
    const ALIGNMENT: u64 = 32;
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3_u32.to_le_bytes());
    file.extend_from_slice(&1_u64.to_le_bytes());
    file.extend_from_slice(&1_u64.to_le_bytes());

    file.extend_from_slice(&(b"general.alignment".len() as u64).to_le_bytes());
    file.extend_from_slice(b"general.alignment");
    file.extend_from_slice(&4_u32.to_le_bytes());
    file.extend_from_slice(&(ALIGNMENT as u32).to_le_bytes());

    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&2_u32.to_le_bytes());
    file.extend_from_slice(&columns.to_le_bytes());
    file.extend_from_slice(&rows.to_le_bytes());
    file.extend_from_slice(&24_u32.to_le_bytes()); // I8
    file.extend_from_slice(&0_u64.to_le_bytes());

    let padding = (file.len() as u64).next_multiple_of(ALIGNMENT) - file.len() as u64;
    file.extend(std::iter::repeat_n(0_u8, padding as usize));
    file.extend_from_slice(data);
    file
}

#[test]
fn a_gguf_twin_shares_every_seam_run_with_the_safetensors_fused_file() {
    // Cross-format sharing under one contract. The GGUF carries the same
    // fused tensor as `fused_file`, and the contract cuts it at the same 12
    // head-slice boundaries — so the two containers, written by two different
    // ecosystems, name the same objects.
    let mut data = Vec::new();
    for head in 0..HEADS {
        for part in 0..3_u8 {
            data.extend_from_slice(&slice(part, head));
        }
    }
    let twin = gguf("blocks.0.attn.qkv_proj.weight", 3 * HEADS, RUN, &data);

    let plan = plan_and_hash_under(twin.as_slice(), Some(&contract(FUSED))).expect("plans");
    assert_eq!(plan.planner(), PlannerId::GgufV1);
    assert_eq!(plan.contract().to_string(), "test.h3-fused@1");

    let gguf_runs: HashSet<ObjectDigest> = plan
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .map(|object| object.digest())
        .collect();
    assert_eq!(gguf_runs.len(), 12, "56-head interleave in miniature");
    assert!(
        gguf_runs.is_subset(&tensor_digests(&fused_file(), Some(&contract(FUSED)))),
        "the GGUF's seam runs are the safetensors twin's objects"
    );

    // The inventory reads GGUF's reversed `ne` back into logical order and
    // names its type the way safetensors would, which is what lets one
    // contract describe both containers.
    let taken = inventory(twin.as_slice()).unwrap().unwrap();
    assert_eq!(taken.tensors()[0].shape(), [3 * HEADS, RUN]);
    assert_eq!(taken.tensors()[0].dtype(), "I8");

    // Red proof: without the contract the GGUF tensor is one object the
    // safetensors file does not hold.
    let plain = plan_and_hash_under(twin.as_slice(), None).expect("plans");
    let plain_runs: Vec<ObjectDigest> = plain
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .map(|object| object.digest())
        .collect();
    assert_eq!(plain_runs.len(), 1);
    assert!(!gguf_runs.contains(&plain_runs[0]));
}

#[test]
fn the_shipped_h3_pair_declares_the_same_runs_on_both_sides() {
    // Pure arithmetic against the MEASURED H3 geometry (hidden 5376, 56 heads
    // of 128, BF16): fused qkv_proj [21504, 5376] = 231,211,008 B, split
    // to_q/to_k/to_v [7168, 5376] = 77,070,336 B each. 50 blocks put
    // 11.56 GB — 17.4% of the 66.28 GB DiT — behind these seams.
    let registry = Registry::builtin().expect("the shipped contracts parse");
    let native = registry
        .get(&Stamp::named("minimax.h3-dit-native", 1).unwrap())
        .expect("the native contract ships");
    let diffusers = registry
        .get(&Stamp::named("minimax.h3-dit-diffusers", 1).unwrap())
        .expect("the diffusers contract ships");

    let fused = native
        .runs_of("blocks.7.attn.qkv_proj.weight", &[21504, 5376], 231_211_008)
        .expect("the interleave applies");
    assert_eq!(fused.len(), 168, "56 heads x (q, k, v)");
    assert!(fused.iter().all(|run| run.length() == 1_376_256));

    let mut by_role: BTreeMap<&str, u64> = BTreeMap::new();
    for run in &fused {
        *by_role.entry(run.role()).or_default() += run.length();
    }
    assert_eq!(by_role.len(), 168, "every run is named once");

    for (pattern, part) in [
        ("transformer_blocks.7.attn.to_q.weight", "q"),
        ("transformer_blocks.7.attn.to_k.weight", "k"),
        ("transformer_blocks.7.attn.to_v.weight", "v"),
    ] {
        let split = diffusers
            .runs_of(pattern, &[7168, 5376], 77_070_336)
            .expect("the slice applies");
        assert_eq!(split.len(), 56);
        assert!(split.iter().all(|run| run.length() == 1_376_256));
        // Role algebra: `blocks.{i}.attn.qkv` + `#q@3` on the fused side is
        // `blocks.{i}.attn.qkv#q` + `@3` on the split side. Same bytes, same
        // name, different packaging.
        assert!(fused.iter().any(|run| run.role() == format!("#{part}@3")));
        assert!(split.iter().any(|run| run.role() == "@3"));
    }
}

// ---------------------------------------------------------------------------
// identification and the stamp
// ---------------------------------------------------------------------------

#[test]
fn identification_reads_the_header_and_records_a_deterministic_winner() {
    let fused = fused_file();
    let taken = inventory(fused.as_slice())
        .expect("reads")
        .expect("a tensor container");
    assert_eq!(taken.tensors().len(), 2);
    assert_eq!(taken.tensors()[0].dtype(), "U8");

    let mut registry = Registry::new();
    registry.insert(contract(FUSED)).unwrap();
    registry.insert(contract(SPLIT)).unwrap();
    let detection = registry.detect(&taken);
    assert_eq!(detection.stamp().to_string(), "test.h3-fused@1");

    // A contract that claims a tensor it cannot describe does not match: the
    // split contract's required to_q/to_k/to_v are simply absent here.
    assert_eq!(detection.candidates().len(), 1);

    // Nothing in the registry matches a foreign file: contract:none.
    let foreign = safetensors(&[("other.weight", vec![2, 8], vec![3_u8; 16])]);
    let foreign_inventory = inventory(foreign.as_slice()).unwrap().unwrap();
    assert!(registry.detect(&foreign_inventory).stamp().is_none());
}

#[test]
fn the_tie_break_prefers_the_most_specific_then_the_highest_version() {
    // Two contracts match the same file. The winner explains more of it; if
    // they explained the same amount, the higher version would win, and the
    // name is the last key so registry order can never decide.
    let broad = contract(FUSED);
    let narrow = contract(
        &FUSED
            .replace("test.h3-fused", "test.h3-narrow")
            .replace("\"required\": false", "\"required\": true"),
    );
    let fused = fused_file();
    let taken = inventory(fused.as_slice()).unwrap().unwrap();

    let mut registry = Registry::new();
    registry.insert(narrow.clone()).unwrap();
    registry.insert(broad.clone()).unwrap();
    let forward = registry.detect(&taken).stamp().clone();

    let mut reversed = Registry::new();
    reversed.insert(broad).unwrap();
    reversed.insert(narrow).unwrap();
    assert_eq!(
        forward,
        *reversed.detect(&taken).stamp(),
        "insertion order may never decide the winner"
    );
    assert!(
        reversed.detect(&taken).ambiguous(),
        "both explain 2 tensors"
    );
    assert_eq!(forward.to_string(), "test.h3-fused@1", "then the name");

    // A higher version of the same declarations wins outright.
    let newer = contract(&FUSED.replace("\"version\": 1", "\"version\": 4"));
    let mut versioned = Registry::new();
    versioned.insert(contract(FUSED)).unwrap();
    versioned.insert(newer).unwrap();
    assert_eq!(
        versioned.detect(&taken).stamp().to_string(),
        "test.h3-fused@4"
    );
}

#[test]
fn a_snapshot_id_follows_its_stamp_and_not_the_registry() {
    // The self-describing-identity claim. Two ingests produce IDENTICAL
    // record lists — the seams agree — and differ only in which contract
    // directed them. The ids must differ, because the layout a snapshot
    // claims is part of what it is; and re-reading the manifest must give
    // back the stamp, not whatever registry is installed now.
    let bytes = fused_file();
    let renamed = contract(&FUSED.replace("test.h3-fused", "test.h3-renamed"));
    let original = plan_and_hash_under(bytes.as_slice(), Some(&contract(FUSED))).unwrap();
    let under_renamed = plan_and_hash_under(bytes.as_slice(), Some(&renamed)).unwrap();
    assert_eq!(
        original
            .objects()
            .iter()
            .map(|object| (object.digest(), object.length()))
            .collect::<Vec<_>>(),
        under_renamed
            .objects()
            .iter()
            .map(|object| (object.digest(), object.length()))
            .collect::<Vec<_>>(),
        "the two registries cut the same boundaries"
    );

    let snapshot = |plan: &tensorfs_core::object::HashedPlan| {
        let mut builder = SnapshotBuilder::new(None);
        builder.file_under(
            "model.safetensors",
            false,
            PlannerId::SafetensorsV1,
            plan.contract().clone(),
            plan.objects()
                .iter()
                .map(|object| FileRecord::Data {
                    digest: object.digest(),
                    length: object.length(),
                })
                .collect(),
        );
        builder.finish().expect("valid")
    };

    let first = snapshot(&original);
    let second = snapshot(&under_renamed);
    assert_ne!(
        first.snapshot_id(),
        second.snapshot_id(),
        "identical bytes under a different contract are a different snapshot"
    );

    let decoded = tensorfs_core::tfm1::decode(&first.to_bytes()).expect("decodes");
    let tensorfs_core::tfm1::Entry::File {
        body: FileBody::Tensor { contract, .. },
        ..
    } = &decoded.entries()[0].1
    else {
        panic!("the entry is a tensor file");
    };
    assert_eq!(contract.to_string(), "test.h3-fused@1");
}

/// The FUSED declarations with the name and version stripped: an
/// author-constructed custom, identified by digest alone.
fn nameless_fused() -> Contract {
    let document = FUSED
        .replace("\"name\": \"test.h3-fused\",\n", "")
        .replace("\"version\": 1,\n", "");
    assert!(!document.contains("\"name\""), "the fixture edit missed");
    contract(&document)
}

#[test]
fn a_custom_contract_chunks_reproducibly_and_stamps_its_digest() {
    // Dedup-invariance holds for customs: boundaries are a pure function of
    // (file bytes, document), so the identical file under the identical
    // nameless document reproduces the identical snapshot on a SECOND store
    // that has never seen the first.
    let bytes = fused_file();
    let custom = nameless_fused();
    assert!(custom.stamp().to_string().starts_with("sha256:"));

    let commit = |name: &str| {
        let root = TempRoot::new(name);
        let store = ObjectStore::open(root.path()).expect("store opens");
        let plan =
            tensorfs_core::planner::plan_with(bytes.as_slice(), Some(&custom)).expect("plans");
        let admitted = store
            .admit_regions(bytes.as_slice(), plan.regions())
            .expect("admits");
        let mut builder = SnapshotBuilder::new(None);
        builder.file_under(
            "model.safetensors",
            false,
            PlannerId::SafetensorsV1,
            plan.contract().clone(),
            admitted
                .iter()
                .map(|object| FileRecord::Data {
                    digest: object.digest(),
                    length: object.length(),
                })
                .collect(),
        );
        builder.finish().expect("valid").snapshot_id()
    };

    let first = commit("custom-first");
    let second = commit("custom-second");
    assert_eq!(first, second, "custom chunking is store-independent");

    // And the recorded stamp is the digest form, which cuts the same seams
    // as the named twin — the objects are shared across the two identities.
    let named = tensor_digests(&bytes, Some(&contract(FUSED)));
    let custom_objects = tensor_digests(&bytes, Some(&custom));
    assert_eq!(named, custom_objects);
}

// ---------------------------------------------------------------------------
// the upgrade path
// ---------------------------------------------------------------------------

#[test]
fn adopting_a_contract_re_admits_only_the_seam_affected_tensor() {
    let root = TempRoot::new("adopt");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let bytes = fused_file();

    // Ingested with no contract: plain per-tensor grid, contract:none.
    let plain = plan_and_hash_under(bytes.as_slice(), None).unwrap();
    let regions: Vec<_> = tensorfs_core::planner::plan(bytes.as_slice())
        .unwrap()
        .regions()
        .to_vec();
    let admitted = store
        .admit_regions(bytes.as_slice(), &regions)
        .expect("admits");
    let before: Vec<ObjectDigest> = admitted.iter().map(|object| object.digest()).collect();
    let body = FileBody::Tensor {
        format: PlannerId::SafetensorsV1.tensor_format().unwrap(),
        contract: Stamp::None,
        logical_size: bytes.len() as u64,
        records: admitted
            .iter()
            .map(|object| FileRecord::Data {
                digest: object.digest(),
                length: object.length(),
            })
            .collect(),
    };
    assert!(plain.contract().is_none());

    let upgraded = adopt(&store, &body, Some(&contract(FUSED))).expect("adopts");
    let FileBody::Tensor {
        contract: stamp,
        records,
        ..
    } = &upgraded
    else {
        panic!("still a tensor container");
    };
    assert_eq!(stamp.to_string(), "test.h3-fused@1");

    // The object-count assertion: the header and the untouched mlp tensor are
    // inherited by digest; only the fused tensor's 12 head slices are new.
    let after: HashSet<ObjectDigest> = records
        .iter()
        .map(|record| match record {
            FileRecord::Data { digest, .. } => *digest,
            FileRecord::Hole { .. } => panic!("no holes here"),
        })
        .collect();
    let inherited: Vec<&ObjectDigest> = before
        .iter()
        .filter(|digest| after.contains(digest))
        .collect();
    assert_eq!(
        inherited.len(),
        2,
        "the header object and the mlp tensor are inherited verbatim"
    );
    assert_eq!(after.len(), 14, "header + mlp + 12 seam runs");

    // And the upgraded file now shares every attention object with the split
    // twin, which is the whole point of upgrading.
    let split = tensor_digests(&split_file(), Some(&contract(SPLIT)));
    assert_eq!(
        split.iter().filter(|digest| after.contains(digest)).count(),
        13
    );
}

#[test]
fn the_shipped_library_is_pinned_by_digest() {
    // A published `name@version` is immutable: that promise is what makes a
    // stamped snapshot reproducible. Nothing enforces it at runtime — the
    // stamp is name@version, not a digest — so it is enforced HERE. Editing a
    // shipped contract without bumping its version fails this test.
    //
    // THE ONE EXCEPTION, and it is narrow: a document still carrying
    // `GENERATED CANDIDATE - NOT RATIFIED` in its own description has not been
    // published — that marker IS the "not yet" in generate -> human-ratify ->
    // publish (tensorfs#130). Ratifying one may move its digest, and then this
    // pin is updated in the same commit rather than the version bumped, or the
    // library would ship a wrong @1 forever and every `lanes=` written against
    // it would name the hypothesis. The moment the marker is gone the document
    // is published and this pin is a promise again.
    let expected: BTreeMap<&str, &str> = BTreeMap::from([
        (
            "anima.diffsynth-bf16@1",
            "f8680e192e1adf6552665bb4f76ed1ba55d564577375c9082eb0ecbbb162e4fc",
        ),
        (
            "dit.blocks-fused-qkv@1",
            "57f09073ca1a631bb173293c84e11f462089d0fa8cf9ce45e142fc33881a17a1",
        ),
        (
            "ernie.diffusers-bf16@1",
            "b4e726e157035529a98a594a211b3d085cc5c3577182ed1e3b9bb4c1d5811c67",
        ),
        (
            "flux1.diffusers-bf16@1",
            "392e9b54605ef75e18e7b7a29da3501f59215c091c228a16d18d403463320a63",
        ),
        (
            "flux2-klein.diffusers-bf16@1",
            "4d9fada187e6e086a1c4c496d81b785ce9419fc089e64d0141f6427b88e9d40d",
        ),
        (
            "hidream-o1.diffusers-bf16@1",
            "69003d11e2cb3b52628cb05e02c275916db74e318eabbce2f6e89be625c7e01f",
        ),
        (
            "internvl-u.diffusers-bf16@1",
            "9ee028aea999f92c4d93609f59ba476f0c6aad8dee3269cee1a22abec97d199d",
        ),
        (
            "joycaption.llava-bf16@1",
            "dbed1ac6ef5756f0dfe60d1c605c193c645e2bcc7d25f6825b01bd12f159f7f5",
        ),
        (
            "krea-2.diffusers-bf16@1",
            "25309e0c2e3ce980e997d54245fde6dcd8860f85521cda7d4317946ee52b1402",
        ),
        (
            "ltx-2-upsampler.diffusers-bf16@1",
            "395dc3c8ccb4acc80c69135df12bf3c965a5746c88ca2c94fd75359432427621",
        ),
        (
            "ltx-2.diffusers-bf16@1",
            "71038ae11883111d367077eec59a457b97c8746439c3b7bc0885555e26b7aa12",
        ),
        (
            "minimax.h3-dit-diffusers@1",
            "22bbb607d3e4351c18ac55e8d86c8a8a3d03296309b80e4419d4bc5153481f28",
        ),
        (
            "minimax.h3-dit-fp8-rowwise@1",
            "69a2cc8f338ba925d4415f67719f1ed1643e9d31f43e4af04d9b2ff1dc035d1f",
        ),
        (
            "minimax.h3-dit-native@1",
            "5f6b7b4a8cd070653607840b922c213a846838e35379737727111c4b0a8de56c",
        ),
        (
            "musicgen.transformers-fp16@1",
            "4d78d369020fe22b612b7403b16dc9cfd653fe0a4f918d2cc364780d3c6f8ce5",
        ),
        (
            "qwen-image.diffusers-bf16@1",
            "757379261ff69111f5e50a5cf2a066c6a118759f1dfe1f34b605f8a49325e673",
        ),
        (
            "qwen3.6-27b-mtp.gguf-ud-q4-k-xl@1",
            "246696d9f7ad089f597744aa4b44b8281c2c2ea4934c7aaa4898534eca7430e7",
        ),
        (
            "qwen3.6-35b-a3b.vllm-fp8@1",
            "63904fa86d615771cc85518064ce37d4afed1a8bf0f33905a25772ac4c933749",
        ),
        (
            "rife.flownet-fp32@1",
            "4133542da74e3738449a8681c5b38cacfbe8d66083b18f24764b4f0520ed411b",
        ),
        (
            "sd15.diffusers-bf16@1",
            "0bc98e52edac1a4b3a8f063162d3785350413b60701ba49d9701e46d69f304d3",
        ),
        (
            "sd2.diffusers-bf16@1",
            "136b158fb5f96cd05f2a8b3accc0b3d36ea7b07b988a338ee4f7934d54a312e6",
        ),
        (
            "sdxl.clip-g-fused-qkv@1",
            "364c0c537e54013eab72994a3e6bf0b913cfb76ab1627dc0822b95cf17b1b262",
        ),
        (
            "sdxl.clip-g-split-qkv@1",
            "c1bbfc65a89a736154504f68296b2b8be3dc43364d4dc04a192c08a184bf64fa",
        ),
        (
            "sdxl.diffusers-bf16@1",
            "ef01dd65f57bd95ae05d70f5a9893e9abab6b4f0831b05c4edf68ae9ebb148e8",
        ),
        (
            "sdxl.diffusers-fp8-rowwise@1",
            "7b78f2e44382dc5a3fe413e0f8f0a62ba63efefc810123304151c2ded931ee37",
        ),
        (
            "sdxl.diffusers-nvfp4-flat@1",
            "239f70ea2c3038812460bc98b83a0818ca042b7d06d27b333b1657aafaca0283",
        ),
        (
            "stable-audio.diffusers-fp16@1",
            "c580a73fda845048b99b3d3f2093be9c282da4166e1e5521fea139957b454ed3",
        ),
        (
            "trellis2.dit-bf16@1",
            "f9763a5aa4b82d552c7b582d7b540cd0fdf576cc5bd234bb9e73ec617738ab52",
        ),
        (
            "wan22.diffusers-bf16@1",
            "91036beea9878c462311d97a878a5dc94128283182fbf0948b30118852d97bd5",
        ),
        (
            "z-image.diffusers-bf16@1",
            "f726fae567c094783ac5aa41822b77f4ae3387bfc07daf22f3ca189ed74c1af5",
        ),
    ]);
    let registry = Registry::builtin().expect("the shipped contracts parse");
    let actual: BTreeMap<String, String> = registry
        .contracts()
        .iter()
        .map(|contract| {
            let digest: String = contract
                .digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            (contract.stamp().to_string(), digest)
        })
        .collect();
    let flat: BTreeMap<&str, &str> = actual
        .iter()
        .map(|(stamp, digest)| (stamp.as_str(), digest.as_str()))
        .collect();
    assert_eq!(
        flat, expected,
        "a shipped contract changed without a version bump"
    );
}

/// A SERVE LANE must declare its quantization, and only a component FRAGMENT
/// may not.
///
/// `lanes={contract: floor}` is required on every gen-worker Model subclass
/// (pgw#1597) and the declaration reads this field, so a lane document with no
/// top-level `dtype` is not a document with a small gap — it is a document no
/// endpoint can declare. That failure surfaces at the FAR END of a vendor bump,
/// in another repo, as a refusal with no remedy; this test moves it to the
/// commit that introduces it.
///
/// gen-worker also derives the sm floor from this field ALONE, so the field is
/// load-bearing twice over: absent, the class cannot be declared; wrong, the
/// lane is placed on a card whose tensor cores cannot do the arithmetic.
///
/// The allow-list is exhaustive ON PURPOSE. A new fragment must be added here
/// deliberately, because "it's a fragment" is exactly the excuse a lane
/// document with a forgotten dtype would offer.
#[test]
fn every_serve_lane_declares_its_quantization() {
    // Component fragments: parts of a tree, not lanes. Each is claimed as a
    // lane nowhere, and none carries a `dtype` — see the `sdxl.clip-g-*` pair
    // and `dit.blocks-fused-qkv`, whose own descriptions say so.
    const FRAGMENTS: &[&str] = &[
        "dit.blocks-fused-qkv@1",
        "sdxl.clip-g-fused-qkv@1",
        "sdxl.clip-g-split-qkv@1",
    ];

    let registry = Registry::builtin().expect("the shipped library parses");
    let mut undeclarable = Vec::new();
    let mut fragments_with_dtype = Vec::new();
    for contract in registry.contracts() {
        let stamp = contract.stamp().to_string();
        let is_fragment = FRAGMENTS.contains(&stamp.as_str());
        match (contract.dtype(), is_fragment) {
            (None, false) => undeclarable.push(stamp),
            (Some(dtype), true) => fragments_with_dtype.push(format!("{stamp} = {dtype}")),
            _ => {}
        }
    }
    assert!(
        undeclarable.is_empty(),
        "these serve lanes declare no top-level dtype, so no endpoint can declare them: \
         {undeclarable:?}. Add the LANE'S QUANTIZATION (torch-precise, and a spelling \
         gen-worker's DTYPE_MIN_SM knows), or add the document to FRAGMENTS if it really \
         is a component fragment."
    );
    assert!(
        fragments_with_dtype.is_empty(),
        "these are listed as fragments but declare a lane dtype: {fragments_with_dtype:?}. \
         Either it is a lane (drop it from FRAGMENTS) or the dtype does not belong."
    );

    // The list must not rot: a name that no longer ships is a stale exemption,
    // and a stale exemption is how a real lane slips through later.
    for name in FRAGMENTS {
        assert!(
            registry
                .contracts()
                .iter()
                .any(|c| c.stamp().to_string() == *name),
            "{name} is exempted here but no longer ships"
        );
    }
}

/// The floor decides chunk boundaries, so it is identity, not tuning
/// (spec/v1/contracts/README.md "The floor", frozen for v1). If this test is
/// red you are changing snapshot identity for every seam-cut file in every
/// store: that requires a format version bump, a new vector corpus, and a
/// migration story — not an edit.
#[test]
fn seam_floor_is_frozen_for_v1() {
    assert_eq!(tensorfs_core::contract::MIN_SEAM_PART_BYTES, 1024 * 1024);
}
