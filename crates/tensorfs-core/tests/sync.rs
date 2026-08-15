//! Engine-level sync coverage over an in-memory fault-injecting transport.
//! The fake hub uses the real TFM1/TFP1 decoders as its oracle, so every
//! structural claim the engine makes is checked by the same code a real hub
//! would run.

#![cfg(any(unix, windows))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::store::StoreError;
use tensorfs_core::sync::{
    CompleteStatus, DownloadGrant, PackGrant, PullReport, PushOptions, PushReport, SyncError,
    SyncPlan, SyncTransport, TransportError, manifest_object_digest, pull_snapshot, push_snapshot,
};
use tensorfs_core::tfm1::{FileRecord, SnapshotId, decode};
use tensorfs_core::tfp1;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

fn scratch(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-sync-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock is sane")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("scratch root creates");
    root
}

/// Commits one sparse file per row (Data/Hole alternation) so seal keeps the
/// committed multi-object boundaries verbatim, then seals.
fn sealed_workspace(
    root: &PathBuf,
    name: &str,
    files: &[(&str, Vec<Vec<u8>>)],
) -> (WorkspaceStore, SnapshotId) {
    let meta = WorkspaceStore::open(root).expect("workspace store opens");
    meta.create_workspace(name).expect("workspace creates");
    let mutations = mutations_for(&meta, files);
    meta.commit_generation(name, &mutations).expect("commits");
    let id = meta.seal_snapshot(name, None).expect("seals");
    (meta, id)
}

fn mutations_for(meta: &WorkspaceStore, files: &[(&str, Vec<Vec<u8>>)]) -> Vec<Mutation> {
    let mut directories = std::collections::BTreeSet::new();
    for (path, _) in files {
        let mut parent = std::path::Path::new(path).parent();
        while let Some(ancestor) = parent {
            if !ancestor.as_os_str().is_empty() {
                directories.insert(ancestor.to_string_lossy().into_owned());
            }
            parent = ancestor.parent();
        }
    }
    let mkdirs = directories.into_iter().map(|path| Mutation::Mkdir { path });
    mkdirs
        .chain(files.iter().map(|(path, chunks)| {
            let mut records = Vec::new();
            for (index, chunk) in chunks.iter().enumerate() {
                if index != 0 {
                    records.push(FileRecord::Hole { length: 1 });
                }
                let admitted = meta.store().put_bytes(chunk).expect("object admits");
                records.push(FileRecord::Data {
                    digest: admitted.digest(),
                    length: admitted.length(),
                });
            }
            Mutation::CreateFile {
                path: (*path).to_owned(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records,
            }
        }))
        .collect()
}

#[derive(Default)]
struct HubState {
    objects: HashMap<[u8; 32], Vec<u8>>,
    staged: HashMap<String, Vec<u8>>,
    head: Option<SnapshotId>,
    sessions: HashMap<String, Session>,
    next: u64,
    uploads_by_digest: HashMap<[u8; 32], u32>,
    grants_per_call: usize,
    fail_uploads: u32,
    expire_uploads: u32,
    incomplete_completes: u32,
    terminal_complete: Option<String>,
    corrupt_download_of: Option<[u8; 32]>,
}

struct Session {
    snapshot_id: SnapshotId,
    expected_head: Option<SnapshotId>,
    closure: Vec<[u8; 32]>,
}

/// An in-memory hub whose admission and verification run the real decoders.
struct FakeHub {
    state: RefCell<HubState>,
}

impl FakeHub {
    fn new() -> Self {
        Self {
            state: RefCell::new(HubState {
                grants_per_call: 2,
                ..HubState::default()
            }),
        }
    }

    fn plan_for(state: &HubState, session_key: &str) -> SyncPlan {
        let session = &state.sessions[session_key];
        let staged_digests: Vec<ObjectDigest> = state
            .staged
            .values()
            .flat_map(|pack| {
                tfp1::decode(pack)
                    .expect("staged packs re-verify")
                    .objects()
                    .map(|object| object.digest())
                    .collect::<Vec<_>>()
            })
            .collect();
        let have = session
            .closure
            .iter()
            .filter(|digest| state.objects.contains_key(*digest))
            .map(|digest| ObjectDigest::from_bytes(*digest))
            .collect();
        let grants = (0..state.grants_per_call)
            .map(|index| PackGrant {
                staging_key: format!("sk-{}-{index}", state.next),
                url: format!("fake-put://{}-{index}", state.next),
                max_payload: tfp1::MAX_PACK_PAYLOAD,
            })
            .collect();
        SyncPlan {
            snapshot_id: session.snapshot_id,
            session: session_key.to_owned(),
            have,
            staged: staged_digests,
            pack_grants: grants,
        }
    }
}

impl SyncTransport for FakeHub {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        let mut state = self.state.borrow_mut();
        let snapshot = decode(tfm1_bytes).map_err(|error| TransportError::Refused {
            code: "manifest-invalid".to_owned(),
            detail: error.to_string(),
        })?;
        let id = snapshot.snapshot_id();
        if state.head.as_ref() != expected_head {
            return Err(TransportError::Refused {
                code: "head-conflict".to_owned(),
                detail: "expected head does not match".to_owned(),
            });
        }
        // The hub holds the manifest bytes already: the blob is admitted as a
        // digest-addressed object directly from the declare body.
        state
            .objects
            .insert(*manifest_object_digest(&id).as_bytes(), tfm1_bytes.to_vec());
        let mut closure = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_path, entry) in snapshot.entries() {
            if let tensorfs_core::tfm1::Entry::File { records, .. } = entry {
                for record in records {
                    if let FileRecord::Data { digest, .. } = record {
                        if seen.insert(*digest.as_bytes()) {
                            closure.push(*digest.as_bytes());
                        }
                    }
                }
            }
        }
        state.next += 1;
        let key = format!("session-{}", state.next);
        state.sessions.insert(
            key.clone(),
            Session {
                snapshot_id: id,
                expected_head: expected_head.copied(),
                closure,
            },
        );
        Ok(Self::plan_for(&state, &key))
    }

    fn more_grants(&self, session: &str) -> Result<SyncPlan, TransportError> {
        let mut state = self.state.borrow_mut();
        state.next += 1;
        if !state.sessions.contains_key(session) {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session.to_owned(),
            });
        }
        Ok(Self::plan_for(&state, session))
    }

    fn upload_pack(&self, grant: &PackGrant, pack: &[u8]) -> Result<(), TransportError> {
        let mut state = self.state.borrow_mut();
        if state.fail_uploads > 0 {
            state.fail_uploads -= 1;
            return Err(TransportError::Io("injected carrier fault".to_owned()));
        }
        if state.expire_uploads > 0 {
            state.expire_uploads -= 1;
            return Err(TransportError::Expired("injected grant expiry".to_owned()));
        }
        let parsed = tfp1::decode(pack).map_err(|error| TransportError::Refused {
            code: "pack-invalid".to_owned(),
            detail: error.to_string(),
        })?;
        for object in parsed.objects() {
            let key = *object.digest().as_bytes();
            let already_staged = state.staged.values().any(|staged| {
                tfp1::decode(staged)
                    .expect("staged packs re-verify")
                    .objects()
                    .any(|member| member.digest() == object.digest())
            });
            if state.objects.contains_key(&key) || already_staged {
                return Err(TransportError::Refused {
                    code: "object-not-missing".to_owned(),
                    detail: "packs may carry only missing objects".to_owned(),
                });
            }
            *state.uploads_by_digest.entry(key).or_default() += 1;
        }
        state
            .staged
            .insert(grant.staging_key.clone(), pack.to_vec());
        Ok(())
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        let mut state = self.state.borrow_mut();
        if let Some(code) = state.terminal_complete.clone() {
            return Ok(CompleteStatus::Failed { code });
        }
        if state.incomplete_completes > 0 {
            state.incomplete_completes -= 1;
            return Ok(CompleteStatus::Incomplete {
                code: "promote_incomplete".to_owned(),
            });
        }
        let (snapshot_id, expected, closure) = {
            let row = &state.sessions[session];
            (row.snapshot_id, row.expected_head, row.closure.clone())
        };
        let packs: Vec<Vec<u8>> = state.staged.values().cloned().collect();
        for pack in packs {
            let parsed = tfp1::decode(&pack).expect("staged packs re-verify");
            for object in parsed.objects() {
                state
                    .objects
                    .insert(*object.digest().as_bytes(), object.bytes().to_vec());
            }
        }
        state.staged.clear();
        for digest in &closure {
            if !state.objects.contains_key(digest) {
                return Ok(CompleteStatus::Incomplete {
                    code: "promote_incomplete".to_owned(),
                });
            }
        }
        if state.head != expected {
            return Ok(CompleteStatus::Failed {
                code: "head-conflict".to_owned(),
            });
        }
        state.head = Some(snapshot_id);
        Ok(CompleteStatus::Promoted)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        Ok(self.state.borrow().head)
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        let state = self.state.borrow();
        digests
            .iter()
            .map(|digest| {
                state
                    .objects
                    .get(digest.as_bytes())
                    .map(|bytes| DownloadGrant {
                        digest: *digest,
                        length: bytes.len() as u64,
                        url: format!("fake-get://{digest}"),
                    })
                    .ok_or_else(|| TransportError::Refused {
                        code: "unknown-object".to_owned(),
                        detail: digest.to_string(),
                    })
            })
            .collect()
    }

    fn download(&self, grant: &DownloadGrant) -> Result<Vec<u8>, TransportError> {
        let state = self.state.borrow();
        let mut bytes = state.objects[grant.digest.as_bytes()].clone();
        if state.corrupt_download_of == Some(*grant.digest.as_bytes()) {
            bytes[0] ^= 0xFF;
        }
        Ok(bytes)
    }
}

fn push(
    meta: &WorkspaceStore,
    hub: &FakeHub,
    id: &SnapshotId,
    expected: Option<&SnapshotId>,
) -> Result<PushReport, SyncError> {
    push_snapshot(meta, hub, id, expected, PushOptions::default())
}

#[test]
fn push_then_pull_round_trips_a_sealed_snapshot_byte_exactly() {
    let root_a = scratch("push-a");
    let (meta_a, id) = sealed_workspace(
        &root_a,
        "publisher",
        &[
            ("model.bin", vec![vec![1_u8; 4096], vec![2_u8; 8192]]),
            ("weights/extra.bin", vec![vec![3_u8; 2048]]),
        ],
    );
    let hub = FakeHub::new();

    let report = push(&meta_a, &hub, &id, None).expect("push succeeds");
    assert_eq!(report.uploaded_objects, 3);
    assert_eq!(report.skipped_remote_resident, 0);
    assert_eq!(hub.state.borrow().head, Some(id));
    // The manifest blob rides the declare body, never a pack.
    assert!(
        !hub.state
            .borrow()
            .uploads_by_digest
            .contains_key(manifest_object_digest(&id).as_bytes())
    );

    let root_b = scratch("pull-b");
    let meta_b = WorkspaceStore::open(&root_b).expect("puller opens");
    let report: PullReport = pull_snapshot(&meta_b, &hub, &id).expect("pull succeeds");
    assert_eq!(report.fetched_objects, 3);
    assert_eq!(report.skipped_local_resident, 0);

    // Adopted and byte-exact: the snapshot decodes locally and every object's
    // bytes round-trip through the verified read path.
    let pulled = meta_b.load_snapshot(&id).expect("snapshot adopted");
    assert_eq!(pulled.snapshot_id(), id);
    for (_path, entry) in pulled.entries() {
        if let tensorfs_core::tfm1::Entry::File { records, .. } = entry {
            for record in records {
                if let FileRecord::Data { digest, length } = record {
                    let verified = meta_b.store().verify(digest).expect("object verifies");
                    assert_eq!(verified, *length);
                }
            }
        }
    }

    let again = pull_snapshot(&meta_b, &hub, &id).expect("second pull succeeds");
    assert_eq!(again.fetched_objects, 0);
    assert_eq!(again.fetched_bytes, 0);
    assert_eq!(again.skipped_local_resident, 3);
}

#[test]
fn an_edited_clone_pushes_only_its_changed_objects() {
    let root = scratch("dedup");
    let (meta, id1) = sealed_workspace(
        &root,
        "base",
        &[(
            "model.bin",
            vec![vec![10_u8; 4096], vec![11_u8; 4096], vec![12_u8; 4096]],
        )],
    );
    let hub = FakeHub::new();
    push(&meta, &hub, &id1, None).expect("base push succeeds");

    meta.create_workspace_from_snapshot("edit", &id1)
        .expect("clone creates");
    let replacement = meta
        .store()
        .put_bytes(&vec![99_u8; 4096])
        .expect("replacement admits");
    let base = meta.load_snapshot(&id1).expect("base loads");
    let (path, records) = base
        .entries()
        .iter()
        .find_map(|(path, entry)| match entry {
            tensorfs_core::tfm1::Entry::File { records, .. } => {
                Some((path.clone(), records.clone()))
            }
            _ => None,
        })
        .expect("base has the file");
    let mut edited = records;
    for record in &mut edited {
        if let FileRecord::Data { digest, length } = record {
            if *length == 4096 && *digest != replacement.digest() {
                *digest = replacement.digest();
                break;
            }
        }
    }
    meta.commit_generation(
        "edit",
        &[Mutation::SetRecords {
            path,
            records: edited,
        }],
    )
    .expect("edit commits");
    let id2 = meta.seal_snapshot("edit", None).expect("edit seals");

    let report = push(&meta, &hub, &id2, Some(&id1)).expect("delta push succeeds");
    assert_eq!(report.uploaded_objects, 1, "only the changed object moves");
    assert_eq!(report.skipped_remote_resident, 2);
    for count in hub.state.borrow().uploads_by_digest.values() {
        assert_eq!(*count, 1, "no object is ever uploaded twice");
    }
    assert_eq!(hub.state.borrow().head, Some(id2));
}

#[test]
fn an_interrupted_push_resumes_without_retransmitting_staged_objects() {
    let root = scratch("resume");
    let files: Vec<Vec<u8>> = (0_u8..6).map(|seed| vec![seed + 1; 4096]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", files)]);
    let hub = FakeHub::new();

    // First attempt stages everything, then dies before promotion: the hub
    // keeps reporting incompleteness until the engine's bounded complete
    // attempts run out, exactly the shape a killed process leaves behind.
    hub.state.borrow_mut().incomplete_completes = 10;
    let stingy = PushOptions {
        max_complete_attempts: 3,
        ..PushOptions::default()
    };
    let error = push_snapshot(&meta, &hub, &id, None, stingy).expect_err("first push dies staged");
    assert!(matches!(error, SyncError::CompletionExhausted { .. }));
    let staged_after_kill = hub.state.borrow().staged.len();
    assert!(
        staged_after_kill > 0,
        "the killed session left staged packs"
    );

    // The resumed push uploads nothing: the same session's staged state is
    // the hub-side journal, mirroring the local store on pull.
    hub.state.borrow_mut().incomplete_completes = 0;
    let report = push(&meta, &hub, &id, None).expect("resume succeeds");
    assert_eq!(report.uploaded_objects, 0, "resume retransmits nothing");
    assert_eq!(report.skipped_remote_resident, 6);
    for count in hub.state.borrow().uploads_by_digest.values() {
        assert_eq!(*count, 1, "no object was ever uploaded twice");
    }
    assert_eq!(hub.state.borrow().head, Some(id));

    // Expired grants replan rather than fail.
    let root2 = scratch("expiry");
    let (meta2, id2) = sealed_workspace(
        &root2,
        "publisher",
        &[("other.bin", vec![vec![201_u8; 4096], vec![202_u8; 4096]])],
    );
    let hub2 = FakeHub::new();
    hub2.state.borrow_mut().expire_uploads = 1;
    let report = push(&meta2, &hub2, &id2, None).expect("expiry replans");
    assert!(report.replans >= 1);
    for count in hub2.state.borrow().uploads_by_digest.values() {
        assert_eq!(*count, 1);
    }
}

#[test]
fn a_lying_hub_cannot_place_wrong_bytes_in_the_local_store() {
    let root = scratch("liar-src");
    let (meta, id) = sealed_workspace(
        &root,
        "publisher",
        &[("model.bin", vec![vec![7_u8; 4096], vec![8_u8; 4096]])],
    );
    let hub = FakeHub::new();
    push(&meta, &hub, &id, None).expect("push succeeds");

    let victim = data_digests(&meta, &id)[0];
    hub.state.borrow_mut().corrupt_download_of = Some(*victim.as_bytes());

    let root_b = scratch("liar-dst");
    let meta_b = WorkspaceStore::open(&root_b).expect("puller opens");
    let error = pull_snapshot(&meta_b, &hub, &id).expect_err("corrupt bytes refuse");
    assert!(
        matches!(
            &error,
            SyncError::Store(StoreError::DigestMismatch { expected, .. })
                if *expected == victim
        ),
        "admission names the digest lie, got {error:?}"
    );
    // Nothing landed at the expected digest and the snapshot was not adopted.
    assert!(matches!(
        meta_b.store().open_object(&victim),
        Err(StoreError::Missing { .. })
    ));
    assert!(meta_b.load_snapshot(&id).is_err());
}

#[test]
fn packs_hold_whole_sorted_objects_within_the_payload_bound() {
    let mib = 1024 * 1024;
    let root = scratch("packs");
    let chunks: Vec<Vec<u8>> = (0_u8..3).map(|seed| vec![seed + 31; 30 * mib]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("big.bin", chunks)]);
    let hub = FakeHub::new();

    let report = push(&meta, &hub, &id, None).expect("push succeeds");
    assert_eq!(report.uploaded_objects, 3);
    assert!(report.packs >= 2, "90 MiB cannot ride one 64 MiB pack");
    assert_eq!(hub.state.borrow().head, Some(id));
    // The staged packs were verified whole/sorted/bounded by the tfp1 oracle
    // inside the fake hub on every upload; heads or bytes lying would have
    // refused there.
}

#[test]
fn completion_retries_through_incompleteness_and_surfaces_terminal_refusal() {
    let root = scratch("complete");
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", vec![vec![5_u8; 512]])]);
    let hub = FakeHub::new();
    hub.state.borrow_mut().incomplete_completes = 3;
    let report = push(&meta, &hub, &id, None).expect("push drives to promoted");
    assert_eq!(report.complete_attempts, 4);

    let root2 = scratch("terminal");
    let (meta2, id2) =
        sealed_workspace(&root2, "publisher", &[("model.bin", vec![vec![6_u8; 512]])]);
    let hub2 = FakeHub::new();
    hub2.state.borrow_mut().terminal_complete = Some("head-conflict".to_owned());
    let error = push(&meta2, &hub2, &id2, None).expect_err("terminal refusal surfaces");
    assert!(matches!(
        error,
        SyncError::HeadRefused { code } if code == "head-conflict"
    ));
}

fn data_digests(meta: &WorkspaceStore, id: &SnapshotId) -> Vec<ObjectDigest> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut digests = Vec::new();
    for (_path, entry) in snapshot.entries() {
        if let tensorfs_core::tfm1::Entry::File { records, .. } = entry {
            for record in records {
                if let FileRecord::Data { digest, .. } = record {
                    digests.push(*digest);
                }
            }
        }
    }
    digests
}
