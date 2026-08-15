//! Engine-level sync coverage over an in-memory fault-injecting hub that
//! mirrors the LANDED th#1960 wire: declare answers missing (no grants), pack
//! grants bind the client's envelope checksum, staging is session-scoped,
//! promotion is budgeted, and the manifest blob is admitted at complete —
//! strictly before the head moves. Admission and verification run the real
//! TFM1/TFP1 decoders, so every structural claim the engine makes is checked
//! by the same code a real hub runs.

#![cfg(any(unix, windows))]

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use sha2::{Digest as _, Sha256};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::store::StoreError;
use tensorfs_core::sync::{
    CompleteStatus, DownloadGrant, GrantsPlan, PackClaim, PackGrant, PullReport, PushOptions,
    PushReport, StagedPack, SyncError, SyncPlan, SyncTransport, TransportError,
    manifest_object_digest, pull_snapshot, push_snapshot,
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

const MAX_PACKS_PER_REQUEST: usize = 16;

struct PackRow {
    staging_key: String,
    objects: Vec<[u8; 32]>,
}

struct Session {
    snapshot_id: SnapshotId,
    expected_head: Option<SnapshotId>,
    closure: Vec<([u8; 32], u64)>,
    manifest: Vec<u8>,
    packs: BTreeMap<String, PackRow>,
}

#[derive(Default)]
struct HubState {
    objects: HashMap<[u8; 32], Vec<u8>>,
    staged: HashMap<String, Vec<u8>>,
    head: Option<SnapshotId>,
    sessions: HashMap<String, Session>,
    next: u64,
    uploads_by_digest: HashMap<[u8; 32], u32>,
    fail_uploads: u32,
    expire_uploads: u32,
    expire_grant_calls: u32,
    incomplete_completes: u32,
    terminal_complete: Option<String>,
    corrupt_download_of: Option<[u8; 32]>,
    /// Objects promoted per `complete` call; 0 means unlimited. Models the
    /// hub's 64-object promote budget at test scale.
    promote_budget: usize,
}

/// An in-memory hub whose admission and verification run the real decoders
/// and the real wire rules.
struct FakeHub {
    state: RefCell<HubState>,
}

impl FakeHub {
    fn new() -> Self {
        Self {
            state: RefCell::new(HubState::default()),
        }
    }

    /// The session's live missing view: closure minus promoted objects minus
    /// members of this session's staged packs — exactly the hub's rule.
    fn missing_view(state: &HubState, session: &Session) -> Vec<(ObjectDigest, u64)> {
        let mut staged_members = std::collections::HashSet::new();
        for pack in session.packs.values() {
            if state.staged.contains_key(&pack.staging_key) {
                staged_members.extend(pack.objects.iter().copied());
            }
        }
        session
            .closure
            .iter()
            .filter(|(digest, _)| {
                !state.objects.contains_key(digest) && !staged_members.contains(digest)
            })
            .map(|(digest, length)| (ObjectDigest::from_bytes(*digest), *length))
            .collect()
    }

    fn staged_rows(state: &HubState, session: &Session) -> Vec<StagedPack> {
        session
            .packs
            .iter()
            .map(|(sha, pack)| StagedPack {
                sha256: sha.clone(),
                staged: state.staged.contains_key(&pack.staging_key),
            })
            .collect()
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
            code: "declaration_invalid".to_owned(),
            detail: error.to_string(),
        })?;
        let id = snapshot.snapshot_id();
        if state.head.as_ref() != expected_head {
            return Err(TransportError::Refused {
                code: "head_conflict".to_owned(),
                detail: "expected head does not match".to_owned(),
            });
        }
        let mut closure = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_path, entry) in snapshot.entries() {
            if let tensorfs_core::tfm1::Entry::File { records, .. } = entry {
                for record in records {
                    if let FileRecord::Data { digest, length } = record
                        && seen.insert(*digest.as_bytes())
                    {
                        closure.push((*digest.as_bytes(), *length));
                    }
                }
            }
        }
        state.next += 1;
        let key = format!("session-{}", state.next);
        let session = Session {
            snapshot_id: id,
            expected_head: expected_head.copied(),
            closure,
            manifest: tfm1_bytes.to_vec(),
            packs: BTreeMap::new(),
        };
        let have = session
            .closure
            .iter()
            .filter(|(digest, _)| state.objects.contains_key(digest))
            .map(|(digest, _)| ObjectDigest::from_bytes(*digest))
            .collect();
        let missing = Self::missing_view(&state, &session);
        let plan = SyncPlan {
            snapshot_id: id,
            session: key.clone(),
            have,
            staged_packs: Vec::new(),
            missing,
            max_pack_payload: tfp1::MAX_PACK_PAYLOAD,
            max_packs_per_request: MAX_PACKS_PER_REQUEST,
        };
        state.sessions.insert(key, session);
        Ok(plan)
    }

    fn pack_grants(
        &self,
        session_key: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        let mut state = self.state.borrow_mut();
        if state.expire_grant_calls > 0 && !claims.is_empty() {
            state.expire_grant_calls -= 1;
            return Err(TransportError::Expired(
                "injected session lease expiry".to_owned(),
            ));
        }
        if !state.sessions.contains_key(session_key) {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session_key.to_owned(),
            });
        }
        if claims.len() > MAX_PACKS_PER_REQUEST {
            return Err(TransportError::Refused {
                code: "bad_request".to_owned(),
                detail: "too many packs".to_owned(),
            });
        }
        let missing_now: HashMap<[u8; 32], u64> = {
            let session = &state.sessions[session_key];
            Self::missing_view(&state, session)
                .into_iter()
                .map(|(digest, length)| (*digest.as_bytes(), length))
                .collect()
        };
        let mut grants = Vec::new();
        let mut rows = Vec::new();
        for claim in claims {
            if claim.objects.is_empty() {
                return Err(TransportError::Refused {
                    code: "bad_request".to_owned(),
                    detail: "a pack claim declares no objects".to_owned(),
                });
            }
            let mut payload = 0_u64;
            let mut seen = std::collections::HashSet::new();
            for digest in &claim.objects {
                if !seen.insert(*digest.as_bytes()) {
                    return Err(TransportError::Refused {
                        code: "bad_request".to_owned(),
                        detail: "duplicate member".to_owned(),
                    });
                }
                let Some(size) = missing_now.get(digest.as_bytes()) else {
                    return Err(TransportError::Refused {
                        code: "bad_request".to_owned(),
                        detail: format!("{digest} is not a missing object of this snapshot"),
                    });
                };
                payload += size;
            }
            // The hub's TFP1 arithmetic fence: magic+count, 48-byte rows,
            // whole-object payload.
            let expected = 12 + 48 * claim.objects.len() as u64 + payload;
            if claim.size_bytes != expected {
                return Err(TransportError::Refused {
                    code: "bad_request".to_owned(),
                    detail: format!(
                        "claimed size {} but members require exactly {expected}",
                        claim.size_bytes
                    ),
                });
            }
            let staging_key = format!("snapshots/staging/{session_key}/{}.tfp1", claim.sha256);
            grants.push(PackGrant {
                pack_sha256: claim.sha256.clone(),
                staging_key: staging_key.clone(),
                url: format!("fake-put://{staging_key}"),
                headers: vec![("x-amz-checksum-sha256".to_owned(), claim.sha256.clone())],
            });
            rows.push((
                claim.sha256.clone(),
                PackRow {
                    staging_key,
                    objects: claim
                        .objects
                        .iter()
                        .map(|digest| *digest.as_bytes())
                        .collect(),
                },
            ));
        }
        let session = state.sessions.get_mut(session_key).expect("session exists");
        for (sha, row) in rows {
            session.packs.insert(sha, row);
        }
        let session = &state.sessions[session_key];
        Ok(GrantsPlan {
            grants,
            staged_packs: Self::staged_rows(&state, session),
            missing: Self::missing_view(&state, session),
        })
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
        // The store enforces the signed checksum: bytes that do not hash to
        // the granted pack sha are refused at the door.
        let mut hasher = Sha256::new();
        hasher.update(pack);
        let actual = {
            let digest = hasher.finalize();
            let mut hex = String::with_capacity(64);
            for byte in digest {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
            }
            hex
        };
        if actual != grant.pack_sha256 {
            return Err(TransportError::Refused {
                code: "checksum-mismatch".to_owned(),
                detail: "pack bytes do not hash to the granted checksum".to_owned(),
            });
        }
        // The store is checksum-and-key only, like R2: the only-missing rule
        // is enforced at GRANT time against the session's missing view, never
        // here — a dead session's staged object legitimately re-uploads
        // under a new session's grant.
        let parsed = tfp1::decode(pack).map_err(|error| TransportError::Refused {
            code: "pack-invalid".to_owned(),
            detail: error.to_string(),
        })?;
        for object in parsed.objects() {
            *state
                .uploads_by_digest
                .entry(*object.digest().as_bytes())
                .or_default() += 1;
        }
        state
            .staged
            .insert(grant.staging_key.clone(), pack.to_vec());
        Ok(())
    }

    fn complete(&self, session_key: &str) -> Result<CompleteStatus, TransportError> {
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
        let budget = state.promote_budget;
        let (snapshot_id, expected, closure, manifest, staged_pack_keys) = {
            let session = &state.sessions[session_key];
            (
                session.snapshot_id,
                session.expected_head,
                session.closure.clone(),
                session.manifest.clone(),
                session
                    .packs
                    .values()
                    .map(|pack| pack.staging_key.clone())
                    .collect::<Vec<_>>(),
            )
        };
        // Budgeted promotion from THIS session's staged packs only.
        let mut promoted_this_call = 0_usize;
        for key in &staged_pack_keys {
            let Some(pack) = state.staged.get(key).cloned() else {
                continue;
            };
            let parsed = tfp1::decode(&pack).expect("staged packs re-verify");
            for object in parsed.objects() {
                if state.objects.contains_key(object.digest().as_bytes()) {
                    continue;
                }
                if budget != 0 && promoted_this_call >= budget {
                    return Ok(CompleteStatus::Incomplete {
                        code: "promote_incomplete".to_owned(),
                    });
                }
                state
                    .objects
                    .insert(*object.digest().as_bytes(), object.bytes().to_vec());
                promoted_this_call += 1;
            }
        }
        for (digest, _) in &closure {
            if !state.objects.contains_key(digest) {
                return Ok(CompleteStatus::Incomplete {
                    code: "upload_incomplete".to_owned(),
                });
            }
        }
        if state.head != expected {
            return Ok(CompleteStatus::Failed {
                code: "head_conflict".to_owned(),
            });
        }
        // The manifest blob becomes the snapshot-id object, strictly before
        // the head is advanced — the landed wire's ordering.
        state
            .objects
            .insert(*manifest_object_digest(&snapshot_id).as_bytes(), manifest);
        for key in &staged_pack_keys {
            state.staged.remove(key);
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
        // Unknown digests are silently omitted, exactly like the wire's
        // `unknown` list: the engine detects the omission itself.
        let state = self.state.borrow();
        Ok(digests
            .iter()
            .filter_map(|digest| {
                state
                    .objects
                    .get(digest.as_bytes())
                    .map(|bytes| DownloadGrant {
                        digest: *digest,
                        length: bytes.len() as u64,
                        url: format!("fake-get://{digest}"),
                    })
            })
            .collect())
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
    // ... and it is fetchable once the head is visible.
    assert!(
        hub.state
            .borrow()
            .objects
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
        if let FileRecord::Data { digest, length } = record
            && *length == 4096
            && *digest != replacement.digest()
        {
            *digest = replacement.digest();
            break;
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
fn resumption_is_promotion_level_across_sessions_and_exact_within_one() {
    // Across restarts: a re-declare opens a fresh session whose staging
    // starts empty, so only PROMOTED objects are skipped — the landed wire's
    // per-session staging keys make this promotion-level by design.
    let root = scratch("resume");
    let files: Vec<Vec<u8>> = (0_u8..6).map(|seed| vec![seed + 1; 4096]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", files)]);
    let hub = FakeHub::new();

    // First run stages everything and dies mid-promotion: the budget promotes
    // two objects per call and the run allows exactly one call.
    hub.state.borrow_mut().promote_budget = 2;
    let stingy = PushOptions {
        max_complete_attempts: 1,
        ..PushOptions::default()
    };
    let error =
        push_snapshot(&meta, &hub, &id, None, stingy).expect_err("first push dies mid-promotion");
    assert!(matches!(error, SyncError::CompletionExhausted { .. }));
    assert_eq!(
        hub.state.borrow().objects.len(),
        2,
        "the killed run promoted exactly its budget"
    );

    // The resumed push re-declares: promoted objects report resident, the
    // rest (staged in the DEAD session) honestly retransmit.
    hub.state.borrow_mut().promote_budget = 0;
    let report = push(&meta, &hub, &id, None).expect("resume succeeds");
    assert_eq!(report.skipped_remote_resident, 2, "promoted objects skip");
    assert_eq!(report.uploaded_objects, 4, "unpromoted objects retransmit");
    assert_eq!(hub.state.borrow().head, Some(id));
    for (digest, count) in &hub.state.borrow().uploads_by_digest {
        let promoted_first = hub.state.borrow().objects.contains_key(digest);
        assert!(promoted_first, "everything ends promoted");
        assert!(*count <= 2, "no object uploads more than twice across runs");
    }

    // Within one run, expiry replans and never retransmits a staged pack.
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
        assert_eq!(*count, 1, "within a run nothing retransmits");
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
    // Every staged pack passed the hub's claim arithmetic (12 + 48n +
    // payload), its signed checksum, and the tfp1 oracle on upload; a claim
    // lying about size, members or bytes would have refused there.
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
    hub2.state.borrow_mut().terminal_complete = Some("head_conflict".to_owned());
    let error = push(&meta2, &hub2, &id2, None).expect_err("terminal refusal surfaces");
    assert!(matches!(
        error,
        SyncError::HeadRefused { code } if code == "head_conflict"
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
