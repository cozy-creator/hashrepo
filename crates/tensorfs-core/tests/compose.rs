//! #80: a re-key is a header rewrite. Every tensor chunk is shared with the
//! source, because the chunk grid is relative to each tensor's own start and
//! TFM1 identity is the record list.

#![cfg(any(unix, windows))]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::compose::{ComposeError, rekey};
use tensorfs_core::object::{ObjectDigest, plan_and_hash};
use tensorfs_core::planner::{self, ByteSource, MAX_OBJECT_SIZE, PlannerId};
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{FileBody, FileRecord};
use tensorfs_core::workspace_source::RecordsSource;

const MIB: u64 = 1024 * 1024;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-compose-{name}-{}", std::process::id()));
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
// fixtures
// ---------------------------------------------------------------------------

fn safetensors(tensors: &[(&str, u64, u8)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, length, _)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let end = offset + length;
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{length}],\"data_offsets\":[{offset},{end}]}}"
        ));
        offset = end;
    }
    header.push('}');

    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(*fill, *length as usize));
    }
    file
}

const GGUF_ALIGNMENT: u64 = 32;

fn aligned(value: u64) -> u64 {
    value.next_multiple_of(GGUF_ALIGNMENT)
}

/// A GGUF v3 file of one-byte (`I8`) tensors, laid out exactly as the format
/// requires: metadata, directory, alignment padding, then each tensor with its
/// own trailing padding.
fn gguf(tensors: &[(&str, u64, u8)]) -> Vec<u8> {
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3_u32.to_le_bytes());
    file.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    file.extend_from_slice(&2_u64.to_le_bytes());

    file.extend_from_slice(&(b"general.architecture".len() as u64).to_le_bytes());
    file.extend_from_slice(b"general.architecture");
    file.extend_from_slice(&8_u32.to_le_bytes());
    file.extend_from_slice(&(b"llama".len() as u64).to_le_bytes());
    file.extend_from_slice(b"llama");

    file.extend_from_slice(&(b"general.alignment".len() as u64).to_le_bytes());
    file.extend_from_slice(b"general.alignment");
    file.extend_from_slice(&4_u32.to_le_bytes());
    file.extend_from_slice(&(GGUF_ALIGNMENT as u32).to_le_bytes());

    let mut offset = 0_u64;
    for (name, length, _) in tensors {
        file.extend_from_slice(&(name.len() as u64).to_le_bytes());
        file.extend_from_slice(name.as_bytes());
        file.extend_from_slice(&1_u32.to_le_bytes());
        file.extend_from_slice(&length.to_le_bytes());
        file.extend_from_slice(&24_u32.to_le_bytes()); // I8: one byte per element
        file.extend_from_slice(&offset.to_le_bytes());
        offset += aligned(*length);
    }

    let padding = aligned(file.len() as u64) - file.len() as u64;
    file.extend(std::iter::repeat_n(0_u8, padding as usize));
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(*fill, *length as usize));
        file.extend(std::iter::repeat_n(
            0_u8,
            (aligned(*length) - length) as usize,
        ));
    }
    file
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Ingests one file exactly as the seal planner would, and returns the body a
/// snapshot would carry for it.
fn commit(store: &ObjectStore, bytes: &[u8]) -> FileBody {
    let plan = planner::plan(bytes).expect("the fixture plans");
    let format = plan
        .planner()
        .tensor_format()
        .expect("the fixture is a tensor container");
    let records = store
        .admit_regions(bytes, plan.regions())
        .expect("the fixture admits")
        .iter()
        .map(|object| FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        })
        .collect();
    FileBody::Tensor {
        format,
        logical_size: bytes.len() as u64,
        records,
    }
}

fn records_of(body: &FileBody) -> &[FileRecord] {
    match body {
        FileBody::Tensor { records, .. } => records,
        FileBody::Blob { .. } => panic!("the fixture is a tensor container"),
    }
}

fn digests(body: &FileBody) -> Vec<ObjectDigest> {
    records_of(body)
        .iter()
        .filter_map(|record| match record {
            FileRecord::Data { digest, .. } => Some(*digest),
            FileRecord::Hole { .. } => None,
        })
        .collect()
}

fn lengths(body: &FileBody) -> Vec<u64> {
    records_of(body)
        .iter()
        .map(|record| match record {
            FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
        })
        .collect()
}

fn read_back(store: &ObjectStore, body: &FileBody) -> Vec<u8> {
    let source = RecordsSource::new(store, records_of(body));
    let mut bytes = vec![0_u8; source.len() as usize];
    source.read_exact_at(0, &mut bytes).expect("records read");
    bytes
}

fn object_count(root: &Path) -> usize {
    fn walk(path: &Path) -> usize {
        fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| {
                if entry.path().is_dir() {
                    walk(&entry.path())
                } else {
                    1
                }
            })
            .sum()
    }
    walk(&root.join("objects"))
}

fn names(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(old, new)| ((*old).to_owned(), (*new).to_owned()))
        .collect()
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

/// The fixture crosses the grid on purpose: `denoiser.big` is 65 MiB, so it
/// owns two objects and its second one exists only because the grid is cut
/// from the TENSOR's start.
fn big_fixture() -> Vec<u8> {
    safetensors(&[
        ("denoiser.big", 65 * MIB, 0x11),
        ("denoiser.small", 4096, 0x22),
        ("text_encoder.a", 2048, 0x33),
    ])
}

#[test]
fn a_rekey_admits_one_header_and_shares_every_tensor_object() {
    let root = TempRoot::new("rekey");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(&store, &big_fixture());
    let before = object_count(root.path());

    let composed = rekey(
        &store,
        &source,
        &names(&[
            ("denoiser.big", "model.diffusion_model.big"),
            ("denoiser.small", "model.diffusion_model.small"),
            ("text_encoder.a", "cond_stage_model.a"),
        ]),
    )
    .expect("a total renaming composes");

    assert_eq!(
        object_count(root.path()),
        before + 1,
        "the only new object is the rewritten header"
    );
    assert_eq!(
        digests(&composed)[1..],
        digests(&source)[1..],
        "every tensor object is inherited verbatim, in order"
    );
    assert_ne!(
        digests(&composed)[0],
        digests(&source)[0],
        "the header is the one thing that changed"
    );
    assert_eq!(
        lengths(&composed)[1..],
        lengths(&source)[1..],
        "and no tensor was re-chunked"
    );
}

#[test]
fn a_tensor_above_the_grid_keeps_its_chunk_boundaries() {
    let root = TempRoot::new("grid");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(&store, &big_fixture());
    let composed = rekey(
        &store,
        &source,
        &names(&[
            ("denoiser.big", "a"),
            ("denoiser.small", "b"),
            ("text_encoder.a", "c"),
        ]),
    )
    .expect("composes");

    // Header, then 64 MiB + 1 MiB for the big tensor, then the two small ones.
    // The 1 MiB remainder is the assertion: the grid is cut from the tensor's
    // own start, so a header of a different length cannot move it.
    assert_eq!(
        lengths(&source)[1..],
        [MAX_OBJECT_SIZE, MIB, 4096, 2048],
        "the source grid is tensor-relative"
    );
    assert_eq!(lengths(&composed)[1..], lengths(&source)[1..]);
    assert_ne!(
        lengths(&composed)[0],
        lengths(&source)[0],
        "the composed header really is a different length"
    );
}

/// The property that makes a composed snapshot ordinary: re-ingesting the
/// composed BYTES through the planner reproduces the composed RECORD LIST
/// object for object. A foreign tool that writes the renamed file itself
/// therefore dedups against the composition completely.
#[test]
fn the_composed_bytes_replan_to_exactly_the_composed_records() {
    let root = TempRoot::new("replan");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(&store, &big_fixture());
    let composed = rekey(
        &store,
        &source,
        &names(&[
            ("denoiser.big", "big"),
            ("denoiser.small", "small"),
            ("text_encoder.a", "a"),
        ]),
    )
    .expect("composes");

    let bytes = read_back(&store, &composed);
    let replanned = plan_and_hash(bytes.as_slice()).expect("the composed bytes plan");
    assert_eq!(replanned.planner(), PlannerId::SafetensorsV1);
    assert_eq!(
        replanned
            .objects()
            .iter()
            .map(|object| (object.digest(), object.length()))
            .collect::<Vec<_>>(),
        records_of(&composed)
            .iter()
            .map(|record| match record {
                FileRecord::Data { digest, length } => (*digest, *length),
                FileRecord::Hole { .. } => panic!("composition emits no holes"),
            })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn a_renaming_that_is_not_a_bijection_over_the_source_is_refused() {
    let root = TempRoot::new("refusals");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(&store, &safetensors(&[("a", 64, 0x01), ("b", 64, 0x02)]));

    assert!(matches!(
        rekey(&store, &source, &names(&[("a", "x")])),
        Err(ComposeError::UnnamedTensor(name)) if name == "b"
    ));
    assert!(matches!(
        rekey(&store, &source, &names(&[("a", "x"), ("b", "x")])),
        Err(ComposeError::DuplicateName(name)) if name == "x"
    ));
    assert!(matches!(
        rekey(&store, &source, &names(&[("a", "x"), ("b", "y"), ("c", "z")])),
        Err(ComposeError::UnknownTensor(name)) if name == "c"
    ));
    assert!(matches!(
        rekey(
            &store,
            &source,
            &names(&[("a", "x"), ("b", "__metadata__")])
        ),
        Err(ComposeError::UnusableName(_))
    ));
    assert_eq!(
        object_count(root.path()),
        3,
        "a refused composition admits nothing"
    );
}

/// The honest limit: inheritance needs each tensor to own whole objects. The
/// seal planner always gives it, but the manifest wire allows any grid.
#[test]
fn a_packed_source_cannot_be_inherited() {
    let root = TempRoot::new("packed");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let bytes = safetensors(&[("a", 64, 0x01), ("b", 64, 0x02)]);
    let whole = store.put_bytes(&bytes).expect("admits");
    let packed = FileBody::Tensor {
        format: PlannerId::SafetensorsV1
            .tensor_format()
            .expect("safetensors is a tensor format"),
        logical_size: whole.length(),
        records: vec![FileRecord::Data {
            digest: whole.digest(),
            length: whole.length(),
        }],
    };

    assert!(matches!(
        rekey(&store, &packed, &names(&[("a", "x"), ("b", "y")])),
        Err(ComposeError::NotObjectAligned(_))
    ));
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

#[test]
fn a_gguf_rekey_keeps_the_metadata_block_and_every_tensor_object() {
    let root = TempRoot::new("gguf");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(
        &store,
        &gguf(&[
            ("blk.0.attn_q.weight", 4096, 0x41),
            ("blk.0.attn_k.weight", 100, 0x42),
            ("output.weight", 2048, 0x43),
        ]),
    );
    let before = object_count(root.path());

    let composed = rekey(
        &store,
        &source,
        &names(&[
            ("blk.0.attn_q.weight", "layers.0.q_proj.weight"),
            ("blk.0.attn_k.weight", "layers.0.k_proj.weight"),
            ("output.weight", "lm_head.weight"),
        ]),
    )
    .expect("composes");

    let source_digests = digests(&source);
    let composed_digests = digests(&composed);
    assert_eq!(
        composed_digests[0], source_digests[0],
        "the metadata block is untouched by a re-key, so it is inherited"
    );
    assert_ne!(
        composed_digests[1], source_digests[1],
        "the tensor directory is where the names live, so it is rewritten"
    );
    // Four records close both files: three tensors, plus the 28 bytes of
    // alignment padding the 100-byte tensor needs. Padding is never glued to
    // tensor data, which is what makes both of them inheritable.
    assert_eq!(
        composed_digests[composed_digests.len() - 4..],
        source_digests[source_digests.len() - 4..],
        "every tensor and its alignment padding is inherited"
    );
    let fresh = composed_digests
        .iter()
        .filter(|digest| !source_digests.contains(digest))
        .count();
    assert!(fresh <= 2, "only the directory and its padding can be new");
    assert_eq!(
        object_count(root.path()),
        before + fresh,
        "nothing but the rewritten header domains is admitted"
    );

    let bytes = read_back(&store, &composed);
    let replanned = plan_and_hash(bytes.as_slice()).expect("the composed GGUF plans");
    assert_eq!(replanned.planner(), PlannerId::GgufV1);
    assert_eq!(
        replanned
            .objects()
            .iter()
            .map(|object| object.digest())
            .collect::<Vec<_>>(),
        composed_digests,
        "a re-ingest of the composed bytes reproduces the composed records"
    );
}

#[test]
fn a_gguf_name_the_format_cannot_hold_is_refused() {
    let root = TempRoot::new("gguf-name");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let source = commit(&store, &gguf(&[("a", 64, 0x01)]));

    assert!(matches!(
        rekey(&store, &source, &names(&[("a", &"n".repeat(64))])),
        Err(ComposeError::UnusableName(_))
    ));
}
