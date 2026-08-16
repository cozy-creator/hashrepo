//! Seal-time planner re-boundary: grid-composed files regain tensor-aligned
//! semantic boundaries when a snapshot is sealed, and only then.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::planner::{MAX_OBJECT_SIZE, PlannerId};
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const MIB: u64 = 1024 * 1024;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-seal-{name}-{}", std::process::id()));
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

fn build_safetensors(tensors: &[(&str, u64, u8)]) -> Vec<u8> {
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

    let mut file = Vec::with_capacity(8 + header.len() + offset as usize);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(*fill, *length as usize));
    }
    file
}

/// The 64 MiB grid records a fixed-grid writer produces (originally the
/// shelved COW mount's fsync compose).
fn grid_records(store: &WorkspaceStore, bytes: &[u8]) -> Vec<FileRecord> {
    bytes
        .chunks(MAX_OBJECT_SIZE as usize)
        .map(|chunk| {
            let admitted = store.store().put_bytes(chunk).expect("grid slot admits");
            FileRecord::Data {
                digest: admitted.digest(),
                length: chunk.len() as u64,
            }
        })
        .collect()
}

fn count_objects(root: &Path) -> usize {
    let mut count = 0;
    let namespace = root.join("objects").join("sha256");
    for level_one in fs::read_dir(&namespace).expect("store layout exists") {
        let level_one = level_one.expect("dirent reads").path();
        if !level_one.is_dir() {
            continue;
        }
        for level_two in fs::read_dir(&level_one).expect("fanout reads") {
            let level_two = level_two.expect("dirent reads").path();
            if !level_two.is_dir() {
                continue;
            }
            count += fs::read_dir(&level_two)
                .expect("leaf reads")
                .filter(|entry| {
                    entry
                        .as_ref()
                        .map(|entry| entry.path().is_file())
                        .unwrap_or(false)
                })
                .count();
        }
    }
    count
}

fn snapshot_file_records(
    store: &WorkspaceStore,
    id: &SnapshotId,
    path: &str,
) -> (PlannerId, Vec<FileRecord>) {
    let snapshot = store.load_snapshot(id).expect("sealed snapshot loads");
    for (entry_path, entry) in snapshot.entries() {
        if entry_path == path
            && let Entry::File { body, .. } = entry
        {
            return (body.planner_id(), body.records().into_owned());
        }
    }
    panic!("{path} is not a file entry in the snapshot");
}

fn read_back(store: &WorkspaceStore, records: &[FileRecord]) -> Vec<u8> {
    use std::io::Read;
    let mut bytes = Vec::new();
    for record in records {
        match record {
            FileRecord::Hole { length } => {
                bytes.extend(std::iter::repeat_n(0_u8, *length as usize))
            }
            FileRecord::Data { digest, length } => {
                let mut object = store.store().open_object(digest).expect("object opens");
                let mut chunk = Vec::with_capacity(*length as usize);
                object.read_to_end(&mut chunk).expect("object reads");
                assert_eq!(chunk.len() as u64, *length);
                bytes.extend(chunk);
            }
        }
    }
    bytes
}

fn expected_tensor_boundaries(header_len: u64, tensors: &[(&str, u64, u8)]) -> Vec<u64> {
    let mut lengths = vec![8 + header_len];
    for (_, tensor_length, _) in tensors {
        let mut remaining = *tensor_length;
        while remaining > MAX_OBJECT_SIZE {
            lengths.push(MAX_OBJECT_SIZE);
            remaining -= MAX_OBJECT_SIZE;
        }
        if remaining > 0 {
            lengths.push(remaining);
        }
    }
    lengths
}

fn record_lengths(records: &[FileRecord]) -> Vec<u64> {
    records
        .iter()
        .map(|record| match record {
            FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
        })
        .collect()
}

const TENSORS: &[(&str, u64, u8)] = &[
    ("blk.attn.weight", 100 * 1024 * 1024, 0xA1),
    ("blk.ffn.weight", 40 * 1024 * 1024, 0xB2),
    ("blk.norm.weight", 4096, 0xC3),
];

#[test]
fn seal_replans_a_grid_composed_safetensors_file_deterministically() {
    let root = TempRoot::new("replan");
    let store = WorkspaceStore::open(root.path()).expect("store opens");
    let bytes = build_safetensors(TENSORS);
    let header_len = bytes.len() as u64 - 8 - TENSORS.iter().map(|(_, l, _)| *l).sum::<u64>();

    store.create_workspace("w").expect("workspace creates");
    store
        .commit_generation(
            "w",
            &[Mutation::CreateFile {
                path: "model.safetensors".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: grid_records(&store, &bytes),
            }],
        )
        .expect("grid records commit");

    let first = store.seal_snapshot("w", None).expect("first seal");
    let (planner, records) = snapshot_file_records(&store, &first, "model.safetensors");
    assert_eq!(planner, PlannerId::SafetensorsV1);
    assert_eq!(
        record_lengths(&records),
        expected_tensor_boundaries(header_len, TENSORS),
        "sealed records are header plus tensor-relative subdivisions"
    );
    assert_eq!(
        read_back(&store, &records),
        bytes,
        "re-slicing kept the bytes"
    );

    let generation = store.head_generation("w").expect("generation reads");
    let objects = count_objects(root.path());
    let second = store.seal_snapshot("w", None).expect("second seal");
    assert_eq!(
        second, first,
        "sealing an already-canonical tree is a fixpoint"
    );
    assert_eq!(
        store.head_generation("w").expect("generation reads"),
        generation,
        "a canonical tree seals without a generation bump"
    );
    assert_eq!(
        count_objects(root.path()),
        objects,
        "a canonical tree seals without admitting any object"
    );
}

#[test]
fn an_edited_clone_reseals_reusing_every_untouched_tensor_object() {
    let root = TempRoot::new("edited");
    let store = WorkspaceStore::open(root.path()).expect("store opens");
    let bytes = build_safetensors(TENSORS);
    let header_len = bytes.len() as u64 - 8 - TENSORS.iter().map(|(_, l, _)| *l).sum::<u64>();

    store.create_workspace("w").expect("workspace creates");
    store
        .commit_generation(
            "w",
            &[Mutation::CreateFile {
                path: "model.safetensors".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: grid_records(&store, &bytes),
            }],
        )
        .expect("grid records commit");
    let sealed = store.seal_snapshot("w", None).expect("seal");

    // A grid writer re-grids the whole edited file; the edit sits inside the
    // first tensor's SECOND tensor-relative subdivision.
    store
        .create_workspace_from_snapshot("edited", &sealed)
        .expect("clone");
    let mut edited = bytes.clone();
    let edit_at = (8 + header_len + 69 * MIB) as usize;
    edited[edit_at..edit_at + 8192].fill(0x5E);
    store
        .commit_generation(
            "edited",
            &[Mutation::SetRecords {
                path: "model.safetensors".to_owned(),
                records: grid_records(&store, &edited),
            }],
        )
        .expect("edited grid commits");
    let resealed = store.seal_snapshot("edited", None).expect("reseal");

    let (_, before) = snapshot_file_records(&store, &sealed, "model.safetensors");
    let (planner, after) = snapshot_file_records(&store, &resealed, "model.safetensors");
    assert_eq!(planner, PlannerId::SafetensorsV1);
    assert_eq!(record_lengths(&before), record_lengths(&after));
    for (index, (old, new)) in before.iter().zip(&after).enumerate() {
        if index == 2 {
            assert_ne!(old, new, "the edited subdivision must change identity");
        } else {
            assert_eq!(
                old, new,
                "untouched record {index} must keep its object digest across snapshots"
            );
        }
    }
    assert_eq!(read_back(&store, &after), edited);
}

#[test]
fn an_unedited_clone_seals_to_the_same_id_admitting_zero_objects() {
    let root = TempRoot::new("clone");
    let store = WorkspaceStore::open(root.path()).expect("store opens");
    let bytes = build_safetensors(TENSORS);

    store.create_workspace("w").expect("workspace creates");
    store
        .commit_generation(
            "w",
            &[Mutation::CreateFile {
                path: "model.safetensors".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: grid_records(&store, &bytes),
            }],
        )
        .expect("grid records commit");
    let sealed = store.seal_snapshot("w", None).expect("seal");

    store
        .create_workspace_from_snapshot("clone", &sealed)
        .expect("clone");
    let generation = store.head_generation("clone").expect("generation reads");
    let objects = count_objects(root.path());
    let resealed = store.seal_snapshot("clone", None).expect("clone seals");
    assert_eq!(resealed, sealed, "an unedited clone reseals to the same id");
    assert_eq!(
        count_objects(root.path()),
        objects,
        "an unedited clone admits zero new objects at seal"
    );
    assert_eq!(
        store.head_generation("clone").expect("generation reads"),
        generation,
        "an unedited clone seals without a generation bump"
    );
}

#[test]
fn malformed_and_sparse_non_tensor_files_seal_as_one_whole_blob() {
    let root = TempRoot::new("fallback");
    let store = WorkspaceStore::open(root.path()).expect("store opens");

    // Plausible-but-invalid safetensors: the declared header length is valid
    // JSON territory but the body contradicts it, so the planner falls back
    // to blob — the raw grid is dead, so the committed grid run must be
    // re-admitted as ONE object of the whole file.
    let mut malformed = vec![0_u8; 20 * MIB as usize];
    malformed[..8].copy_from_slice(&(usize::MAX as u64).to_le_bytes());
    // A hole in a blob is unrepresentable, so a sparse non-tensor file has
    // exactly one canonical form: its zeros materialize into the blob.
    let sparse_data = store
        .store()
        .put_bytes(b"sixteen db bytes")
        .expect("sparse head admits");

    store.create_workspace("w").expect("workspace creates");
    store
        .commit_generation(
            "w",
            &[
                Mutation::CreateFile {
                    path: "broken.safetensors".to_owned(),
                    executable: false,
                    planner: PlannerId::BlobV1,
                    records: grid_records(&store, &malformed),
                },
                Mutation::CreateFile {
                    path: "sparse.bin".to_owned(),
                    executable: false,
                    planner: PlannerId::BlobV1,
                    records: vec![
                        FileRecord::Data {
                            digest: sparse_data.digest(),
                            length: 16,
                        },
                        FileRecord::Hole { length: 4096 },
                    ],
                },
            ],
        )
        .expect("fixture commits");

    let sealed = store.seal_snapshot("w", None).expect("seal");

    let (broken_planner, broken) = snapshot_file_records(&store, &sealed, "broken.safetensors");
    assert_eq!(broken_planner, PlannerId::BlobV1);
    assert_eq!(
        record_lengths(&broken),
        vec![20 * MIB],
        "ONE object, no grid"
    );
    assert_eq!(read_back(&store, &broken), malformed);

    let mut expected_sparse = b"sixteen db bytes".to_vec();
    expected_sparse.extend(std::iter::repeat_n(0_u8, 4096));
    let (sparse_planner, sparse) = snapshot_file_records(&store, &sealed, "sparse.bin");
    assert_eq!(sparse_planner, PlannerId::BlobV1);
    assert_eq!(
        record_lengths(&sparse),
        vec![16 + 4096],
        "the sparse run materialized into one whole blob"
    );
    assert_eq!(read_back(&store, &sparse), expected_sparse);

    // A second seal is a fixpoint: the blobs are already their single
    // whole-file objects, reused by (offset, length) with zero re-hashing.
    let generation = store.head_generation("w").expect("generation reads");
    let objects = count_objects(root.path());
    let resealed = store.seal_snapshot("w", None).expect("reseal");
    assert_eq!(resealed, sealed, "sealing a canonical tree is a fixpoint");
    assert_eq!(count_objects(root.path()), objects);
    assert_eq!(
        store.head_generation("w").expect("generation reads"),
        generation,
        "nothing to re-boundary means no generation bump"
    );
}

#[test]
fn a_grid_composed_gguf_file_regains_its_provenance_at_seal() {
    let fixture = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/v1/planner-vectors/fixtures/gguf-v3-q4-0.hex"),
    )
    .expect("shared GGUF fixture exists");
    let hex = fixture.trim_end_matches(['\r', '\n']);
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("fixture is hex"))
        .collect();

    let root = TempRoot::new("gguf");
    let store = WorkspaceStore::open(root.path()).expect("store opens");
    store.create_workspace("w").expect("workspace creates");
    store
        .commit_generation(
            "w",
            &[Mutation::CreateFile {
                path: "model.gguf".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: grid_records(&store, &bytes),
            }],
        )
        .expect("grid records commit");

    let sealed = store.seal_snapshot("w", None).expect("seal");
    let (planner, records) = snapshot_file_records(&store, &sealed, "model.gguf");
    assert_eq!(planner, PlannerId::GgufV1, "provenance is restored at seal");
    assert!(
        records.len() > 1,
        "the GGUF header and tensor body become separate dedup domains"
    );
    assert_eq!(read_back(&store, &records), bytes);
}

/// The #61 double-hash fence, extended to blobs: a multi-GB-declared blob
/// entry round-trips through seal planning with the blob reused by
/// `(offset, length)` and ZERO re-hashing of its bytes.
///
/// The fixture object is a SPARSE file installed at its digest path, so
/// nothing multi-hundred-MiB is ever actually written. The fence is made
/// red-provable by corrupting the resident payload after commit: a seal that
/// re-read or re-admitted the blob would either surface the corruption as a
/// fresh (different-digest) object or replace the manifest digest — the
/// reuse path does neither, because verification is admission-time and the
/// committed single record already IS the canonical blob form.
#[test]
fn a_multi_gb_declared_blob_reseals_by_offset_length_with_zero_rehash() {
    use sha2::Digest as _;

    const BLOB_SIZE: u64 = 513 * MIB + 7; // deliberately off every 64 MiB grid line
    let root = TempRoot::new("blob-fence");
    let store = WorkspaceStore::open(root.path()).expect("store opens");
    store.create_workspace("w").expect("workspace creates");

    // sha256 of BLOB_SIZE zero bytes, streamed — the only full pass anywhere
    // in this test, standing in for the real admission that would have
    // happened at ingest time.
    let mut hasher = sha2::Sha256::new();
    let block = vec![0_u8; MIB as usize];
    let mut remaining = BLOB_SIZE;
    while remaining > 0 {
        let take = remaining.min(MIB) as usize;
        hasher.update(&block[..take]);
        remaining -= take as u64;
    }
    let digest = tensorfs_core::object::ObjectDigest::from_bytes(hasher.finalize().into());

    // Install the object as a sparse file at its digest path.
    let hex = digest.to_string();
    let hex = hex.strip_prefix("sha256:").expect("digest displays tagged");
    let object_dir = root
        .path()
        .join("objects")
        .join("sha256")
        .join(&hex[..2])
        .join(&hex[2..4]);
    fs::create_dir_all(&object_dir).expect("fanout creates");
    let object_path = object_dir.join(hex);
    let file = fs::File::create(&object_path).expect("sparse object creates");
    file.set_len(BLOB_SIZE).expect("sparse length sets");
    drop(file);

    store
        .commit_generation(
            "w",
            &[Mutation::CreateFile {
                path: "dataset.tar".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest,
                    length: BLOB_SIZE,
                }],
            }],
        )
        .expect("the single-record blob commits");

    // Corrupt the resident payload. From here on, ANY read that hashes the
    // blob's bytes can no longer reproduce `digest`.
    fs::write(&object_path, b"\xffcorrupted").expect("payload overwrites");
    let file = fs::OpenOptions::new()
        .write(true)
        .open(&object_path)
        .expect("reopens");
    file.set_len(BLOB_SIZE).expect("length restored");
    drop(file);

    let generation = store.head_generation("w").expect("generation reads");
    let objects = count_objects(root.path());
    let sealed = store.seal_snapshot("w", None).expect("seal");
    assert_eq!(
        count_objects(root.path()),
        objects,
        "reuse by (offset, length): the seal admitted nothing"
    );
    assert_eq!(
        store.head_generation("w").expect("generation reads"),
        generation,
        "no re-boundary, no generation bump"
    );

    let (planner, records) = snapshot_file_records(&store, &sealed, "dataset.tar");
    assert_eq!(planner, PlannerId::BlobV1);
    assert_eq!(
        records,
        vec![FileRecord::Data {
            digest,
            length: BLOB_SIZE,
        }],
        "the manifest carries the declared whole-file digest untouched"
    );

    // And a re-seal is the same fixpoint.
    assert_eq!(store.seal_snapshot("w", None).expect("reseal"), sealed);
}
