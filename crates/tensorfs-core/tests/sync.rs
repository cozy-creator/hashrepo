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
    BlobGrant, BlobPart, BlobPartReport, CompleteStatus, DownloadGrant, GrantsPlan, PackClaim,
    PackGrant, Progress, ProgressSink, PullReport, PushOptions, PushReport, StagedPack, SyncError,
    SyncPlan, SyncTransport, TransportError, manifest_object_digest, pull_snapshot, push_snapshot,
};

/// What a hub whose corpus never reaches the blob lane answers if the engine
/// ever asks. It never should — a lane it never lists cannot be driven — so
/// this is a loud refusal rather than a silent success.
fn blob_lane_absent() -> TransportError {
    TransportError::Refused {
        code: "blob_lane_unsupported".to_owned(),
        detail: "this hub lists no blob-lane objects".to_owned(),
    }
}
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
                // Tensor provenance: seal keeps a sparse tensor body's
                // committed records verbatim, which is what preserves the
                // multi-object closure these fixtures exist to exercise. A
                // blob-planner file would materialize into ONE object.
                planner: PlannerId::SafetensorsV1,
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
    /// Packs to accept before the carrier dies for good. `None` accepts every
    /// one. Models a push killed MID-TRANSFER, which is the only way to leave
    /// staged-but-unpromoted packs behind for a later session to adopt.
    accept_uploads: Option<u32>,
    expire_uploads: u32,
    expire_grant_calls: u32,
    incomplete_completes: u32,
    /// `complete` calls that die on the CARRIER before the hub answers. This
    /// is the tensorfs#92 shape: the promotion is fine, the answer is lost.
    io_completes: u32,
    /// When set, every `complete` answers retryable incompleteness and admits
    /// NOTHING — a promotion that is standing still rather than working.
    complete_never_advances: bool,
    terminal_complete: Option<String>,
    corrupt_download_of: Option<[u8; 32]>,
    /// Objects promoted per `complete` call; 0 means unlimited. Models the
    /// hub's 64-object promote budget at test scale.
    promote_budget: usize,
    /// The payload cap this hub advertises; 0 means the protocol maximum.
    /// A small cap makes a fixture push several packs at test scale.
    pack_payload_cap: u64,
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
            if let tensorfs_core::tfm1::Entry::File { body, .. } = entry {
                for record in body.records().iter() {
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
            // This hub carries no blob lane; the multipart path is covered
            // against the full-fidelity hub in `tests/blob_lane.rs`.
            missing_blobs: Vec::new(),
            max_pack_payload: if state.pack_payload_cap == 0 {
                tfp1::MAX_PACK_PAYLOAD
            } else {
                state.pack_payload_cap
            },
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
            // th#2077: staging is CONTENT-ADDRESSED, so the key is the
            // pack's own checksum and a later session finds what an earlier
            // one landed. An adopted envelope is recorded and reported staged,
            // never re-granted — which is what makes a resumed push move only
            // the bytes that are genuinely still owed.
            // th#2077: staging is CONTENT-ADDRESSED, so the key is the
            // pack's own checksum and a later session finds what an earlier
            // one landed. An adopted envelope is recorded and reported staged,
            // never re-granted — which is what makes a resumed push move only
            // the bytes that are genuinely still owed.
            let staging_key = format!("snapshots/staging/packs/{}.tfp1", claim.sha256);
            if !state.staged.contains_key(&staging_key) {
                grants.push(PackGrant {
                    pack_sha256: claim.sha256.clone(),
                    staging_key: staging_key.clone(),
                    url: format!("fake-put://{staging_key}"),
                    headers: vec![("x-amz-checksum-sha256".to_owned(), claim.sha256.clone())],
                });
            }
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
            missing_blobs: Vec::new(),
        })
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        let mut state = self.state.borrow_mut();
        if state.fail_uploads > 0 {
            state.fail_uploads -= 1;
            return Err(TransportError::Io("injected carrier fault".to_owned()));
        }
        match state.accept_uploads {
            Some(0) => {
                return Err(TransportError::Io("injected carrier death".to_owned()));
            }
            Some(remaining) => state.accept_uploads = Some(remaining - 1),
            None => {}
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
        // An in-memory carrier moves its bytes in one go; reporting them is
        // still what a transport owes the engine.
        progress.bytes(pack.len() as u64);
        Ok(())
    }

    fn complete(&self, session_key: &str) -> Result<CompleteStatus, TransportError> {
        let mut state = self.state.borrow_mut();
        if let Some(code) = state.terminal_complete.clone() {
            return Ok(CompleteStatus::Failed { code });
        }
        if state.io_completes > 0 {
            state.io_completes -= 1;
            return Err(TransportError::Io("timed out reading response".to_owned()));
        }
        if state.complete_never_advances {
            return Ok(CompleteStatus::Incomplete {
                code: "promote_incomplete".to_owned(),
                promoted: 0,
                total: 1,
            });
        }
        if state.incomplete_completes > 0 {
            state.incomplete_completes -= 1;
            return Ok(CompleteStatus::Incomplete {
                code: "promote_incomplete".to_owned(),
                // The injected fault admits nothing on purpose: this is the
                // "standing still" shape the client's stall budget exists for.
                promoted: 0,
                total: 0,
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
                        promoted: closure
                            .iter()
                            .filter(|(digest, _)| state.objects.contains_key(digest))
                            .count() as u64,
                        total: closure.len() as u64,
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
                    promoted: closure
                        .iter()
                        .filter(|(digest, _)| state.objects.contains_key(digest))
                        .count() as u64,
                    total: closure.len() as u64,
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

    fn blob_grants(
        &self,
        _session: &str,
        _digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        Err(blob_lane_absent())
    }

    fn upload_blob_part(
        &self,
        _part: &BlobPart,
        _bytes: &[u8],
        _progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        Err(blob_lane_absent())
    }

    fn report_blob_parts(
        &self,
        _session: &str,
        _digest: &ObjectDigest,
        _parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        Err(blob_lane_absent())
    }
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        let state = self.state.borrow();
        let mut bytes = state.objects[grant.digest.as_bytes()].clone();
        if state.corrupt_download_of == Some(*grant.digest.as_bytes()) {
            bytes[0] ^= 0xFF;
        }
        sink.write_all(&bytes)
            .map_err(|error| TransportError::Io(error.to_string()))?;
        progress.bytes(bytes.len() as u64);
        Ok(bytes.len() as u64)
    }
}

fn push(
    meta: &WorkspaceStore,
    hub: &FakeHub,
    id: &SnapshotId,
    expected: Option<&SnapshotId>,
) -> Result<PushReport, SyncError> {
    push_snapshot(
        meta,
        hub,
        id,
        expected,
        PushOptions::default(),
        ProgressSink::silent(),
    )
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
    let report: PullReport =
        pull_snapshot(&meta_b, &hub, &id, ProgressSink::silent()).expect("pull succeeds");
    assert_eq!(report.fetched_objects, 3);
    assert_eq!(report.skipped_local_resident, 0);

    // Adopted and byte-exact: the snapshot decodes locally and every object's
    // bytes round-trip through the verified read path.
    let pulled = meta_b.load_snapshot(&id).expect("snapshot adopted");
    assert_eq!(pulled.snapshot_id(), id);
    for (_path, entry) in pulled.entries() {
        if let tensorfs_core::tfm1::Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, length } = record {
                    let verified = meta_b.store().verify(digest).expect("object verifies");
                    assert_eq!(verified, *length);
                }
            }
        }
    }

    let again =
        pull_snapshot(&meta_b, &hub, &id, ProgressSink::silent()).expect("second pull succeeds");
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
            tensorfs_core::tfm1::Entry::File { body, .. } => {
                Some((path.clone(), body.records().into_owned()))
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
fn resumption_is_staging_level_across_sessions_and_exact_within_one() {
    // ACROSS RESTARTS (tensorfs#93). Staging is content-addressed, so a pack
    // an earlier session landed is found by a later one under the same key
    // and is never re-sent. Before th#2077 the key carried the session id,
    // which made this unexpressible: the 2026-08-16 acceptance re-uploaded
    // 185 MB into a fresh prefix and called it a resume.
    let root = scratch("resume");
    let files: Vec<Vec<u8>> = (0_u8..6).map(|seed| vec![seed + 1; 4096]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", files)]);
    let hub = FakeHub::new();

    // One pack per object, and the first run dies with three of the six
    // staged and NOTHING promoted — the shape of a push killed mid-transfer.
    hub.state.borrow_mut().pack_payload_cap = 4096;
    hub.state.borrow_mut().accept_uploads = Some(3);
    let dying = PushOptions {
        max_upload_attempts: 1,
        ..PushOptions::default()
    };
    let error = push_snapshot(&meta, &hub, &id, None, dying, ProgressSink::silent())
        .expect_err("the first push dies mid-transfer");
    assert!(matches!(error, SyncError::Transport(TransportError::Io(_))));
    let staged_first = hub.state.borrow().staged.len();
    assert_eq!(
        staged_first, 3,
        "three packs landed before the carrier died"
    );
    assert!(
        hub.state.borrow().objects.is_empty(),
        "the run never reached complete, so nothing is promoted"
    );

    // The resumed push opens a NEW session. The three staged packs are
    // adopted by checksum; only the unstaged remainder crosses the wire.
    hub.state.borrow_mut().accept_uploads = None;
    let report = push(&meta, &hub, &id, None).expect("resume succeeds");
    assert_eq!(
        report.skipped_remote_resident, 0,
        "nothing was promoted, so nothing reports resident"
    );
    assert_eq!(
        report.packs, 3,
        "only the three packs the dead run never landed are uploaded: {report:?}"
    );
    assert_eq!(hub.state.borrow().head, Some(id));
    for count in hub.state.borrow().uploads_by_digest.values() {
        assert_eq!(
            *count, 1,
            "an adopted pack is never re-sent, so every object uploads exactly once"
        );
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
    let error = pull_snapshot(&meta_b, &hub, &id, ProgressSink::silent())
        .expect_err("corrupt bytes refuse");
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

/// How a hub departs from the canonical missing set it should have answered
/// with. Everything else about the hub stays honest, so a refusal here can
/// only be the perturbation.
#[derive(Clone, Copy)]
enum Perturbation {
    /// The same set, in the opposite order.
    Reverse,
    /// Canonical order, one length inflated by a byte.
    InflateFirstLength,
}

struct PerturbingHub {
    inner: FakeHub,
    how: Perturbation,
}

impl PerturbingHub {
    fn new(how: Perturbation) -> Self {
        Self {
            inner: FakeHub::new(),
            how,
        }
    }

    fn perturb(&self, mut missing: Vec<(ObjectDigest, u64)>) -> Vec<(ObjectDigest, u64)> {
        match self.how {
            Perturbation::Reverse => missing.reverse(),
            Perturbation::InflateFirstLength => {
                if let Some(first) = missing.first_mut() {
                    first.1 += 1;
                }
            }
        }
        missing
    }
}

impl SyncTransport for PerturbingHub {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        let mut plan = self.inner.declare(tfm1_bytes, expected_head)?;
        plan.missing = self.perturb(plan.missing);
        Ok(plan)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        let mut plan = self.inner.pack_grants(session, claims)?;
        plan.missing = self.perturb(plan.missing);
        Ok(plan)
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        self.inner.upload_pack(grant, pack, progress)
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.inner.download_grants(digests)
    }

    fn blob_grants(
        &self,
        session: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        self.inner.blob_grants(session, digests)
    }

    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        self.inner.upload_blob_part(part, bytes, progress)
    }

    fn report_blob_parts(
        &self,
        session: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        self.inner.report_blob_parts(session, digest, parts)
    }
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        self.inner.download(grant, sink, progress)
    }
}

/// The manifest the client declared is the only authority on which objects
/// exist, how long they are, and in what order they are packed. A hub that
/// answers the right SET in the wrong ORDER is refused, not quietly re-sorted:
/// push assembles packs greedily in the order it is handed, so the order
/// decides pack membership and therefore each pack's checksum — the value a
/// grant binds and a resume must match. Repairing it here would hide a remote
/// free to answer differently on the next replan.
///
/// The shuffle is deterministic (a reversal) and the arm is repeated, because
/// an ordering claim proved once is a claim about one scheduling accident.
#[test]
fn a_hub_that_reorders_the_missing_set_is_refused_and_never_repaired() {
    for round in 0..5 {
        let root = scratch(&format!("shuffled-{round}"));
        let (meta, id) = sealed_workspace(
            &root,
            "publisher",
            &[(
                "model.bin",
                vec![vec![41_u8; 4096], vec![42_u8; 4096], vec![43_u8; 4096]],
            )],
        );
        let canonical = data_digests(&meta, &id);
        assert_eq!(
            canonical.len(),
            3,
            "round {round}: three objects to reorder"
        );

        let hub = PerturbingHub::new(Perturbation::Reverse);
        let error = push_snapshot(
            &meta,
            &hub,
            &id,
            None,
            PushOptions::default(),
            ProgressSink::silent(),
        )
        .expect_err("a reordered missing set must refuse");
        assert!(
            matches!(&error, SyncError::MissingNotCanonical { digest } if *digest == canonical[1]),
            "round {round}: the refusal must name the first out-of-order digest, got {error:?}"
        );
        // Refused before anything moved: no pack was claimed, staged or
        // promoted, and the head never advanced.
        assert!(
            hub.inner.state.borrow().uploads_by_digest.is_empty(),
            "round {round}: bytes moved under a refused plan"
        );
        assert!(hub.inner.state.borrow().staged.is_empty());
        assert_eq!(hub.inner.state.borrow().head, None);

        // The control: the identical fixture against an honest hub pushes.
        let honest = FakeHub::new();
        let report = push(&meta, &honest, &id, None).expect("the honest order pushes");
        assert_eq!(report.uploaded_objects, 3);
        assert_eq!(honest.state.borrow().head, Some(id));

        std::fs::remove_dir_all(&root).ok();
    }
}

/// The same contract on lengths: the remote may not restate an object's size.
#[test]
fn a_hub_that_restates_an_object_length_is_refused() {
    let root = scratch("restated-length");
    let (meta, id) = sealed_workspace(
        &root,
        "publisher",
        &[("model.bin", vec![vec![51_u8; 2048], vec![52_u8; 1024]])],
    );
    let canonical = data_digests(&meta, &id);

    let hub = PerturbingHub::new(Perturbation::InflateFirstLength);
    let error = push_snapshot(
        &meta,
        &hub,
        &id,
        None,
        PushOptions::default(),
        ProgressSink::silent(),
    )
    .expect_err("an inflated length must refuse");
    assert!(
        matches!(
            &error,
            SyncError::MissingLength { digest, expected, actual }
                if *digest == canonical[0] && *expected == 2048 && *actual == 2049
        ),
        "the refusal must name the digest and both lengths, got {error:?}"
    );
    assert!(hub.inner.state.borrow().uploads_by_digest.is_empty());
}

/// The pack layer proves the 64 MiB bound on synthetic bytes; this proves the
/// whole hermetic engine carries one there — committed to a workspace, sealed
/// into a manifest, loaded, packed, claimed, uploaded and promoted. It is the
/// largest object the format admits, and one object at the object cap fills
/// the payload cap exactly, so this is also the pack-count boundary: 64 MiB is
/// one pack, and one byte more could not be.
///
/// Deliberately ONE object: the point is the bound, not a large corpus, and a
/// multi-gigabyte fixture would buy nothing this does not already prove.
#[test]
fn one_maximum_size_object_rides_the_engine_end_to_end() {
    let root = scratch("max-object");
    // The literal bound, NOT `MAX_PACK_PAYLOAD`. Sizing the fixture from the
    // constant would make this arm move with the very regression it guards:
    // halve the constant and a constant-derived object halves with it and the
    // test stays green, proving nothing. Pinned here, a moved bound fails
    // twice — on the format claim below, and on the push itself.
    const BOUND: usize = 64 * 1024 * 1024;
    let bound = BOUND;
    assert_eq!(
        tfp1::MAX_PACK_PAYLOAD,
        BOUND as u64,
        "TFP1.md: the payload bound is 64 MiB and equals the object bound"
    );
    assert_eq!(tensorfs_core::planner::MAX_OBJECT_SIZE, BOUND as u64);
    let (meta, id) = {
        let chunk: Vec<u8> = (0..bound)
            .map(|index| (0xa5_u8).wrapping_add(index as u8))
            .collect();
        sealed_workspace(&root, "publisher", &[("giant.bin", vec![chunk])])
    };

    let closure = data_digests(&meta, &id);
    assert_eq!(closure.len(), 1, "one object, at the bound");
    assert_eq!(
        meta.store()
            .verify(&closure[0])
            .expect("the object verifies"),
        bound as u64
    );

    let hub = FakeHub::new();
    let report = push(&meta, &hub, &id, None).expect("a maximum-size object pushes");
    assert_eq!(report.uploaded_objects, 1);
    assert_eq!(report.uploaded_bytes, bound as u64);
    assert_eq!(
        report.packs, 1,
        "the object cap equals the payload cap: exactly one pack"
    );
    assert_eq!(hub.state.borrow().head, Some(id));

    // The hub promoted the real bytes, at the real length, under the real
    // digest — its own tfp1 decode and checksum gate ran on the way in.
    let state = hub.state.borrow();
    let promoted = &state.objects[closure[0].as_bytes()];
    assert_eq!(promoted.len(), bound);
    assert_eq!(
        ObjectDigest::from_bytes(Sha256::digest(promoted).into()),
        closure[0]
    );
    assert_eq!(state.uploads_by_digest[closure[0].as_bytes()], 1);
    drop(state);

    std::fs::remove_dir_all(&root).ok();
}

/// A hub that fires one shot of local mischief the instant a push has
/// declared and is therefore in flight, then behaves exactly like the real
/// fake for the rest of the transfer.
struct SaboteurHub<'meta> {
    inner: FakeHub,
    meta: &'meta WorkspaceStore,
    snapshot: SnapshotId,
    fired: RefCell<bool>,
}

impl<'meta> SaboteurHub<'meta> {
    fn new(meta: &'meta WorkspaceStore, snapshot: SnapshotId) -> Self {
        Self {
            inner: FakeHub::new(),
            meta,
            snapshot,
            fired: RefCell::new(false),
        }
    }

    /// Deletes the snapshot under the push and runs GC to completion: mark,
    /// hold, delete is three epochs, so this is a collector that has fully
    /// made up its mind while the transfer is still running.
    fn sabotage(&self) {
        if std::mem::replace(&mut *self.fired.borrow_mut(), true) {
            return;
        }
        self.meta
            .delete_snapshot(&self.snapshot)
            .expect("the snapshot deletes mid-push");
        for _ in 0..3 {
            self.meta.collect().expect("collection runs");
        }
    }
}

impl SyncTransport for SaboteurHub<'_> {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        let plan = self.inner.declare(tfm1_bytes, expected_head)?;
        self.sabotage();
        Ok(plan)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        self.inner.pack_grants(session, claims)
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        self.inner.upload_pack(grant, pack, progress)
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.inner.download_grants(digests)
    }

    fn blob_grants(
        &self,
        session: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        self.inner.blob_grants(session, digests)
    }

    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        self.inner.upload_blob_part(part, bytes, progress)
    }

    fn report_blob_parts(
        &self,
        session: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        self.inner.report_blob_parts(session, digest, parts)
    }
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        self.inner.download(grant, sink, progress)
    }
}

/// A push in flight is the ONLY thing keeping its bytes alive once the
/// workspace has moved on: the sealed snapshot row is their last root.
/// Deleting that snapshot mid-transfer strips the root, and `collect` then
/// takes the objects the push is still streaming — the push dies reading
/// bytes that existed when it started.
///
/// The pending-sync lease exists exactly for this window: it pins the
/// snapshot's closure for the transfer's exact lifetime, and not one epoch
/// longer, which the tail of this test also proves.
#[test]
fn a_snapshot_deleted_mid_push_keeps_its_objects_until_the_push_ends() {
    let root = scratch("pending-sync-lease");
    let (meta, id) = sealed_workspace(
        &root,
        "publisher",
        &[("model.bin", vec![vec![0xA1_u8; 4096], vec![0xC3_u8; 2048]])],
    );
    // Drop the workspace's own reference, so the sealed snapshot is the only
    // root these objects have. Without this the object map would pin them
    // and the race could not be constructed at all.
    meta.commit_generation(
        "publisher",
        &[Mutation::Unlink {
            path: "model.bin".to_owned(),
        }],
    )
    .expect("the workspace forgets the file");

    let closure = data_digests(&meta, &id);
    assert_eq!(closure.len(), 2, "the fixture must have objects to lose");

    let hub = SaboteurHub::new(&meta, id);
    let report = push_snapshot(
        &meta,
        &hub,
        &id,
        None,
        PushOptions::default(),
        ProgressSink::silent(),
    )
    .expect("a push whose snapshot is deleted mid-flight still finishes");
    assert_eq!(report.uploaded_objects, closure.len() as u64);
    assert_eq!(hub.inner.state.borrow().head, Some(id));
    for digest in &closure {
        assert!(
            meta.store().verify(digest).is_ok(),
            "{digest} was collected out from under the push that was sending it"
        );
    }

    // The pin ends with the push. Nothing roots these objects now, so the
    // ordinary two-epoch protocol must reclaim them.
    for _ in 0..3 {
        meta.collect().expect("collection runs");
    }
    for digest in &closure {
        assert!(
            matches!(meta.store().verify(digest), Err(StoreError::Missing { .. })),
            "{digest} outlived the push that pinned it"
        );
    }
}

/// Records what a transfer reported, in order, and where from.
#[derive(Default)]
struct Beats {
    objects: RefCell<Vec<ObjectDigest>>,
    bytes: RefCell<Vec<u64>>,
    threads: RefCell<Vec<std::thread::ThreadId>>,
}

impl Beats {
    fn sink(&self) -> impl Fn(Progress) + '_ {
        move |event| {
            self.threads.borrow_mut().push(std::thread::current().id());
            match event {
                Progress::Object { digest, .. } => self.objects.borrow_mut().push(digest),
                Progress::Bytes(moved) => self.bytes.borrow_mut().push(moved),
            }
        }
    }

    fn objects(&self) -> usize {
        self.objects.borrow().len()
    }

    fn moved(&self) -> u64 {
        self.bytes.borrow().iter().sum()
    }
}

/// Refuses to serve any object after the first until an earlier one has been
/// REPORTED. A pull that reports only once every object has landed can never
/// get past the second object, so the timing is asserted rather than the
/// totals — which were always right, and always too late.
struct GatedPull<'hub> {
    inner: &'hub FakeHub,
    beats: &'hub Beats,
    manifest: ObjectDigest,
    served: std::cell::Cell<usize>,
}

impl SyncTransport for GatedPull<'_> {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        self.inner.declare(tfm1_bytes, expected_head)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        self.inner.pack_grants(session, claims)
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        self.inner.upload_pack(grant, pack, progress)
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.inner.download_grants(digests)
    }

    fn blob_grants(
        &self,
        session: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        self.inner.blob_grants(session, digests)
    }

    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        self.inner.upload_blob_part(part, bytes, progress)
    }

    fn report_blob_parts(
        &self,
        session: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        self.inner.report_blob_parts(session, digest, parts)
    }
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        if grant.digest != self.manifest {
            if self.served.get() > 0 && self.beats.objects() == 0 {
                return Err(TransportError::Refused {
                    code: "no-progress-yet".to_owned(),
                    detail: "the pull fetched a second object without reporting \
                             the first — the transfer is invisible while it runs"
                        .to_owned(),
                });
            }
            self.served.set(self.served.get() + 1);
        }
        self.inner.download(grant, sink, progress)
    }
}

/// The push-side twin of [`GatedPull`]: the second pack cannot be uploaded
/// until the first pack's objects have been reported.
struct GatedPush<'hub> {
    inner: &'hub FakeHub,
    beats: &'hub Beats,
    sent: std::cell::Cell<usize>,
}

impl SyncTransport for GatedPush<'_> {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        self.inner.declare(tfm1_bytes, expected_head)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        self.inner.pack_grants(session, claims)
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        if self.sent.get() > 0 && self.beats.objects() == 0 {
            return Err(TransportError::Refused {
                code: "no-progress-yet".to_owned(),
                detail: "the push uploaded a second pack without reporting the \
                         first pack's objects"
                    .to_owned(),
            });
        }
        self.sent.set(self.sent.get() + 1);
        self.inner.upload_pack(grant, pack, progress)
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.inner.download_grants(digests)
    }

    fn blob_grants(
        &self,
        session: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        self.inner.blob_grants(session, digests)
    }

    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        self.inner.upload_blob_part(part, bytes, progress)
    }

    fn report_blob_parts(
        &self,
        session: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        self.inner.report_blob_parts(session, digest, parts)
    }
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        self.inner.download(grant, sink, progress)
    }
}

/// pgw#1287: a 31.6 GB pull reported `0` for its entire duration because the
/// only signal came after every transfer had drained, and the hub condemned a
/// pod that was downloading correctly. The totals were never the problem —
/// their timing was. So the gate here makes a late report impossible to pass
/// rather than merely visible after the fact.
#[test]
fn a_pull_reports_every_object_before_it_fetches_the_next() {
    let root_a = scratch("progress-pull-source");
    let (meta_a, id) = sealed_workspace(
        &root_a,
        "publisher",
        &[
            ("model.bin", vec![vec![1_u8; 4096], vec![2_u8; 4096]]),
            ("weights/extra.bin", vec![vec![3_u8; 4096]]),
        ],
    );
    let hub = FakeHub::new();
    push(&meta_a, &hub, &id, None).expect("push succeeds");

    let root_b = scratch("progress-pull-destination");
    let meta_b = WorkspaceStore::open(&root_b).expect("puller opens");
    let beats = Beats::default();
    let gated = GatedPull {
        inner: &hub,
        beats: &beats,
        manifest: manifest_object_digest(&id),
        served: std::cell::Cell::new(0),
    };
    let sink = beats.sink();

    let report = pull_snapshot(&meta_b, &gated, &id, ProgressSink::new(&sink))
        .expect("the pull reports each object before fetching the next");

    assert_eq!(report.fetched_objects, 3);
    assert_eq!(beats.objects(), 3, "every landed object is reported once");
    // The sink also sees the bytes the transport moved — the manifest blob on
    // top of the data closure — so a consumer can watch either counter.
    assert!(beats.moved() > report.fetched_bytes);
    let caller = std::thread::current().id();
    assert!(
        beats.threads.borrow().iter().all(|id| *id == caller),
        "the sink must run on the thread driving the transfer; callers hand it \
         state that is not theirs to lock"
    );
}

/// A push is just as invisible when it reports only at the end, and a pack is
/// the largest thing it moves in one call.
#[test]
fn a_push_reports_every_pack_as_it_lands() {
    let root = scratch("progress-push");
    let (meta, id) = sealed_workspace(
        &root,
        "publisher",
        &[
            ("model.bin", vec![vec![1_u8; 4096], vec![2_u8; 4096]]),
            ("weights/extra.bin", vec![vec![3_u8; 4096]]),
        ],
    );
    let hub = FakeHub::new();
    // One object per pack, so "reports as each pack lands" is a claim with
    // more than one pack behind it.
    hub.state.borrow_mut().pack_payload_cap = 4096;

    let beats = Beats::default();
    let gated = GatedPush {
        inner: &hub,
        beats: &beats,
        sent: std::cell::Cell::new(0),
    };
    let sink = beats.sink();

    let report = push_snapshot(
        &meta,
        &gated,
        &id,
        None,
        PushOptions::default(),
        ProgressSink::new(&sink),
    )
    .expect("the push reports each pack before sending the next");

    assert_eq!(report.packs, 3, "the fixture must push more than one pack");
    assert_eq!(report.uploaded_objects, 3);
    assert_eq!(beats.objects(), 3);
    assert!(
        beats.moved() >= report.uploaded_bytes,
        "the pack envelopes carry at least the payload they were built from"
    );
}

/// A resumed transfer must look alive, not dead. Objects the far side already
/// holds are completed work toward readiness: a resume that finds most of its
/// closure in place and reports nothing is indistinguishable from a transfer
/// making no headway — which is how the healthiest possible pod gets recycled.
#[test]
fn a_settled_transfer_still_reports_the_work_that_is_already_done() {
    let root_a = scratch("progress-resume-source");
    let (meta_a, id) = sealed_workspace(
        &root_a,
        "publisher",
        &[("model.bin", vec![vec![7_u8; 4096], vec![8_u8; 4096]])],
    );
    let hub = FakeHub::new();
    push(&meta_a, &hub, &id, None).expect("first push succeeds");

    // Every object is already promoted, so this push moves nothing at all —
    // and must still report that its closure is accounted for.
    let pushed = Beats::default();
    let sink = pushed.sink();
    let report = push_snapshot(
        &meta_a,
        &hub,
        &id,
        Some(&id),
        PushOptions::default(),
        ProgressSink::new(&sink),
    )
    .expect("a settled push succeeds");
    assert_eq!(report.uploaded_objects, 0);
    assert_eq!(
        pushed.objects(),
        2,
        "a push whose objects the remote already holds reported nothing"
    );

    let root_b = scratch("progress-resume-destination");
    let meta_b = WorkspaceStore::open(&root_b).expect("puller opens");
    pull_snapshot(&meta_b, &hub, &id, ProgressSink::silent()).expect("first pull succeeds");

    let pulled = Beats::default();
    let sink = pulled.sink();
    let report = pull_snapshot(&meta_b, &hub, &id, ProgressSink::new(&sink))
        .expect("a settled pull succeeds");
    assert_eq!(report.fetched_objects, 0);
    assert_eq!(report.skipped_local_resident, 2);
    assert_eq!(
        pulled.objects(),
        2,
        "a pull whose objects are all resident reported nothing"
    );
}

fn data_digests(meta: &WorkspaceStore, id: &SnapshotId) -> Vec<ObjectDigest> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut digests = Vec::new();
    for (_path, entry) in snapshot.entries() {
        if let tensorfs_core::tfm1::Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, .. } = record {
                    digests.push(*digest);
                }
            }
        }
    }
    digests
}

/// tensorfs#92, the defect this whole loop exists for: the 2026-08-16
/// acceptance uploaded all 280 objects of a 272 MB snapshot, lost the answer
/// to `complete` on a flat 60-second read deadline, and threw the entire push
/// away because a transport error out of `complete` was terminal.
///
/// A lost answer is not a failed promotion, and the remote's completion is
/// idempotent, so the only correct response is to ask again.
#[test]
fn a_lost_answer_to_complete_is_retried_rather_than_discarding_the_push() {
    let root = scratch("complete-io");
    let files: Vec<Vec<u8>> = (0_u8..4).map(|seed| vec![seed + 40; 4096]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", files)]);
    let hub = FakeHub::new();

    // Three carrier failures on `complete`, exactly like a read that gave up
    // while the hub was still admitting objects.
    hub.state.borrow_mut().io_completes = 3;
    let report = push(&meta, &hub, &id, None).expect("a lost answer must not lose the push");

    assert_eq!(
        hub.state.borrow().head,
        Some(id),
        "the snapshot promoted despite the lost answers"
    );
    assert!(
        report.complete_attempts >= 4,
        "every carrier failure costs one more call, never the push: {report:?}"
    );
}

/// The other half of #92: what replaces the deadline. Completion is bounded by
/// calls that ADMIT NOTHING, not by calls that take a while — so a promotion
/// that keeps making progress may take as many round trips as it needs, well
/// past any fixed attempt budget.
#[test]
fn completion_polls_while_the_remote_advances_and_stops_only_when_it_stalls() {
    let root = scratch("complete-progress");
    // Thirty objects promoted one per call needs thirty-one calls — comfortably
    // more than `max_completion_stalls`, which is the point: a flat attempt cap
    // would condemn this healthy promotion.
    let files: Vec<Vec<u8>> = (0_u8..30).map(|seed| vec![seed + 1; 4096]).collect();
    let (meta, id) = sealed_workspace(&root, "publisher", &[("model.bin", files)]);
    let hub = FakeHub::new();
    hub.state.borrow_mut().promote_budget = 1;

    let options = PushOptions::default();
    let report = push(&meta, &hub, &id, None).expect("a slow promotion still promotes");
    assert_eq!(hub.state.borrow().head, Some(id));
    assert!(
        report.complete_attempts > options.max_completion_stalls,
        "progress must buy unbounded calls: {} attempts against a {} stall budget",
        report.complete_attempts,
        options.max_completion_stalls
    );

    // And a promotion that never advances is refused, in bounded time, naming
    // the count it stopped at rather than a clock it exceeded.
    let root2 = scratch("complete-stalled");
    let (meta2, id2) = sealed_workspace(
        &root2,
        "publisher",
        &[("other.bin", vec![vec![9_u8; 4096]])],
    );
    let hub2 = FakeHub::new();
    hub2.state.borrow_mut().complete_never_advances = true;
    let error = push_snapshot(
        &meta2,
        &hub2,
        &id2,
        None,
        PushOptions {
            max_completion_stalls: 3,
            ..PushOptions::default()
        },
        ProgressSink::silent(),
    )
    .expect_err("a completion that never advances is refused");
    let SyncError::CompletionStalled {
        stalls, promoted, ..
    } = error
    else {
        panic!("expected a stall refusal, got {error:?}");
    };
    assert_eq!(stalls, 3, "the budget counts calls that changed nothing");
    assert_eq!(promoted, 0, "and reports how far the remote actually got");
}
