//! The live Rust-stack proof: push and pull real snapshots against a real
//! Tensorhub.
//!
//! Opt-in, never a CI gate. It needs a hub carrying th#1960's snapshot-sync
//! routes and a bearer token:
//!
//!     TENSORFS_HUB_URL    hub base URL (default http://127.0.0.1:31550)
//!     TENSORFS_HUB_ORG    org slug
//!     TENSORFS_HUB_REPO   repo name (must already exist)
//!     TENSORFS_HUB_TOKEN  bearer token
//!
//! Without them the test reports skip and passes. Every phase asserts on the
//! hub's own answers, not on local bookkeeping: a claim the hub does not
//! corroborate is not proven.

#![cfg(any(unix, windows))]

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::ByteSource;
use tensorfs_core::store::ObjectStore;
use tensorfs_core::sync::http::{HttpTransport, TokenSource};
use tensorfs_core::sync::{SyncTransport, pull_snapshot, push_snapshot};
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const MIB: usize = 1024 * 1024;

struct Slice<'a>(&'a [u8]);

impl ByteSource for Slice<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> std::io::Result<()> {
        let start = usize::try_from(offset).expect("offset fits");
        destination.copy_from_slice(&self.0[start..start + destination.len()]);
        Ok(())
    }

    fn check_unchanged(&self) -> std::io::Result<()> {
        Ok(())
    }
}

fn hub() -> Option<(String, String, String, String)> {
    let url = env::var("TENSORFS_HUB_URL").unwrap_or_else(|_| "http://127.0.0.1:31550".to_owned());
    let org = env::var("TENSORFS_HUB_ORG").ok()?;
    let repo = env::var("TENSORFS_HUB_REPO").ok()?;
    let token = env::var("TENSORFS_HUB_TOKEN").ok()?;
    Some((url, org, repo, token))
}

fn transport(url: &str, org: &str, repo: &str, token: &str) -> HttpTransport {
    HttpTransport::new(url, org, repo).with_token(TokenSource::Static(token.to_owned()))
}

/// A deterministic safetensors file whose tensors straddle the 64 MiB grid, so
/// the planner produces several independent objects per file.
///
/// The nonce must ride the PAYLOAD, not only the tensor names: names live in
/// the header object alone, so constant-fill tensors produce byte-identical
/// data objects every run and a hub that saw an earlier run reports them
/// resident. Without this, the "fresh content must upload" assertion below
/// passes once and is vacuous on every later run against the same hub.
fn safetensors(nonce: &str, tensors: &[(&str, usize, u8)]) -> Vec<u8> {
    let seed = *SnapshotId::of(nonce.as_bytes()).as_bytes();
    let mut header = String::from("{");
    let mut offset = 0_usize;
    for (index, (name, length, _)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{nonce}.{name}":{{"dtype":"U8","shape":[{length}],"data_offsets":[{offset},{}]}}"#,
            offset + length
        ));
        offset += length;
    }
    header.push('}');

    let mut file = Vec::with_capacity(8 + header.len() + offset);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend((0..*length).map(|index| fill ^ seed[index % seed.len()]));
    }
    file
}

/// Admits one file's planned objects and returns its ordered records.
fn admit(store: &ObjectStore, bytes: &[u8]) -> Vec<FileRecord> {
    let hashed = plan_and_hash(&Slice(bytes)).expect("fixture plans");
    let mut records = Vec::new();
    let mut offset = 0_usize;
    for object in hashed.objects() {
        let length = usize::try_from(object.length()).expect("object length fits");
        let admitted = store
            .put_bytes(&bytes[offset..offset + length])
            .expect("object admits");
        assert_eq!(admitted.digest(), object.digest(), "admission is exact");
        records.push(FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        });
        offset += length;
    }
    assert_eq!(offset, bytes.len(), "records cover the whole file");
    records
}

fn seal(root: &Path, files: &[(&str, Vec<u8>)]) -> (WorkspaceStore, SnapshotId) {
    let meta = WorkspaceStore::open(root).expect("store opens");
    meta.create_workspace("main").expect("workspace created");
    let mutations: Vec<Mutation> = files
        .iter()
        .map(|(path, bytes)| Mutation::CreateFile {
            path: (*path).to_owned(),
            executable: false,
            planner: tensorfs_core::planner::PlannerId::SafetensorsV1,
            records: admit(meta.store(), bytes),
        })
        .collect();
    meta.commit_generation("main", &mutations)
        .expect("generation commits");
    let id = meta.seal_snapshot("main", None).expect("snapshot seals");
    (meta, id)
}

fn file_bytes(meta: &WorkspaceStore, id: &SnapshotId, path: &str) -> Vec<u8> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut out = Vec::new();
    for (entry_path, entry) in snapshot.entries() {
        if entry_path != path {
            continue;
        }
        let Entry::File { records, .. } = entry else {
            panic!("{path} is not a regular file");
        };
        for record in records {
            match record {
                FileRecord::Hole { length } => {
                    out.extend(std::iter::repeat_n(0_u8, usize::try_from(*length).unwrap()));
                }
                FileRecord::Data { digest, length } => {
                    let mut file = meta.store().open_object(digest).expect("object opens");
                    let mut buffer = vec![0_u8; usize::try_from(*length).unwrap()];
                    std::io::Read::read_exact(&mut file, &mut buffer).expect("object reads");
                    out.extend(buffer);
                }
            }
        }
    }
    out
}

fn scratch(name: &str) -> PathBuf {
    let base = env::var("TENSORFS_E2E_DIR")
        .map_or_else(|_| env::temp_dir().join("tensorfs-live-e2e"), PathBuf::from);
    let path = base.join(name);
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("scratch dir");
    path
}

#[test]
fn a_real_hub_round_trips_snapshots_with_dedup_and_verified_pulls() {
    let Some((url, org, repo, token)) = hub() else {
        eprintln!("skipping: set TENSORFS_HUB_ORG, TENSORFS_HUB_REPO, TENSORFS_HUB_TOKEN");
        return;
    };
    // A per-run nonce rides the TENSOR NAMES so a long-lived shared hub cannot
    // already hold these objects — otherwise "uploaded" would prove nothing.
    let nonce = format!("live{}", std::process::id());
    println!("hub={url} repo={org}/{repo} nonce={nonce}");

    let alpha = safetensors(
        &nonce,
        &[("blk0.attn", 70 * MIB, 0x11), ("blk0.bias", 4096, 0x12)],
    );
    let beta = safetensors(&nonce, &[("blk1.ffn", 40 * MIB, 0x21)]);

    // ---- push -----------------------------------------------------------
    let producer_root = scratch("producer");
    let (producer, snapshot) = seal(
        &producer_root,
        &[
            ("model-00001.safetensors", alpha.clone()),
            ("model-00002.safetensors", beta.clone()),
        ],
    );
    let client = transport(&url, &org, &repo, &token);

    let base = client.head().expect("head reads");
    let report = push_snapshot(
        &producer,
        &client,
        &snapshot,
        base.as_ref(),
        Default::default(),
    )
    .expect("push succeeds");
    println!("push: {report:?}");
    assert!(report.uploaded_objects > 0, "fresh content must upload");
    assert_eq!(
        report.skipped_remote_resident, 0,
        "a nonced fixture cannot already be resident"
    );

    let head = client.head().expect("head reads").expect("head is set");
    assert_eq!(head, snapshot, "the hub's head is exactly what we pushed");

    // ---- dedup ----------------------------------------------------------
    // One byte inside blk0.attn's SECOND grid object; every other object,
    // including all of model-00002, must be recognized as resident.
    let mut edited = alpha.clone();
    let payload_start = edited.len() - (70 * MIB + 4096);
    edited[payload_start + 64 * MIB + 8] ^= 0xFF;

    let editor_root = scratch("editor");
    let (editor, edited_snapshot) = seal(
        &editor_root,
        &[
            ("model-00001.safetensors", edited.clone()),
            ("model-00002.safetensors", beta.clone()),
        ],
    );
    assert_ne!(edited_snapshot, snapshot, "an edit changes the snapshot id");

    let delta = push_snapshot(
        &editor,
        &client,
        &edited_snapshot,
        Some(&head),
        Default::default(),
    )
    .expect("delta push succeeds");
    println!("delta push: {delta:?}");
    assert!(
        delta.skipped_remote_resident >= 3,
        "the hub must recognize the unchanged objects as resident: {delta:?}"
    );
    assert!(
        delta.uploaded_objects <= 2,
        "only the edited object (plus at most its header) may upload: {delta:?}"
    );
    assert!(
        delta.uploaded_bytes < 70 * MIB as u64,
        "an 8-byte edit must not move the whole tensor: {delta:?}"
    );

    // ---- pull into a fresh store ---------------------------------------
    let consumer_root = scratch("consumer");
    let consumer = WorkspaceStore::open(&consumer_root).expect("consumer store opens");
    let remote_head = client.head().expect("head reads").expect("head is set");
    assert_eq!(remote_head, edited_snapshot, "head advanced to the edit");

    let pulled = pull_snapshot(&consumer, &client, &remote_head).expect("pull succeeds");
    println!("pull: {pulled:?}");
    assert!(pulled.fetched_objects > 0, "an empty store fetches");
    assert_eq!(
        pulled.skipped_local_resident, 0,
        "an empty store holds nothing"
    );

    // Byte-exactness is the claim that matters: reconstruct from the pulled
    // objects alone and compare against the producer's bytes.
    let mut expected = BTreeMap::new();
    expected.insert("model-00001.safetensors", edited.clone());
    expected.insert("model-00002.safetensors", beta.clone());
    for (path, bytes) in &expected {
        let got = file_bytes(&consumer, &remote_head, path);
        assert_eq!(got.len(), bytes.len(), "{path}: length matches");
        assert!(got == *bytes, "{path}: pulled bytes are byte-exact");
    }
    println!(
        "byte-exact reconstruction verified for {} files",
        expected.len()
    );

    // ---- resume: a second pull must move nothing ------------------------
    let again = pull_snapshot(&consumer, &client, &remote_head).expect("second pull succeeds");
    println!("second pull: {again:?}");
    assert_eq!(again.fetched_bytes, 0, "verified residency short-circuits");
    assert_eq!(
        again.skipped_local_resident, pulled.fetched_objects,
        "every object the first pull fetched is skipped by the second"
    );

    // ---- the editor pushes again: the hub already has everything --------
    let noop = push_snapshot(
        &editor,
        &client,
        &edited_snapshot,
        Some(&remote_head),
        Default::default(),
    )
    .expect("idempotent re-push succeeds");
    println!("re-push: {noop:?}");
    assert_eq!(
        noop.uploaded_bytes, 0,
        "a re-push of a resident snapshot moves no bytes"
    );

    println!("LIVE RUST-STACK E2E PASSED");
}
