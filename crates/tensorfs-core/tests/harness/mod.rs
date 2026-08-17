//! Shared machinery for the adversarial robustness suite.
//!
//! Three things live here, all deliberately independent of the code under
//! test wherever independence is possible:
//!
//!  * [`Consistency`] — an oracle that re-derives store health from the bytes
//!    on disk (walking `objects/sha256` and rehashing) rather than asking the
//!    library whether it is healthy.
//!  * [`FaultHub`] — an in-memory hub whose admission and verification run
//!    the real TFM1/TFP1 decoders and the real wire rules, with a broad
//!    injectable fault surface covering content-adversarial and
//!    protocol-state failure classes.
//!  * [`FaultServer`] — a scripted loopback HTTP server for the
//!    transport-level and HTTP-semantic classes, exercised against the real
//!    [`tensorfs_core::sync::http::HttpTransport`].
//!
//! Randomised families seed [`Rng`] from an explicit seed that every failure
//! message prints, so any red run reproduces exactly.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use sha2::{Digest as _, Sha256};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::sync::{
    BlobGrant, BlobPart, BlobPartReport, CompleteStatus, DownloadGrant, GrantsPlan, PackClaim,
    PackGrant, ProgressSink, StagedPack, SyncPlan, SyncTransport, TransportError,
    manifest_object_digest,
};
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId, decode};
use tensorfs_core::tfp1;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

/// SplitMix64. Small, seedable, and reproducible: every randomised test
/// prints its seed on failure so a red run replays byte for byte.
pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[low, high)`; `high` must exceed `low`.
    pub fn range(&mut self, low: u64, high: u64) -> u64 {
        assert!(high > low, "empty range");
        low + self.next_u64() % (high - low)
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next_u64() % items.len() as u64) as usize]
    }

    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length)
            .map(|_| (self.next_u64() & 0xFF) as u8)
            .collect()
    }
}

/// The seed a randomised test runs under. Overridable so a failure replays:
/// `TENSORFS_SEED=12345 cargo test ...`.
#[must_use]
pub fn seed_from_env(default: u64) -> u64 {
    std::env::var("TENSORFS_SEED")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(default)
}

/// Heavy randomised families run `iterations` only when explicitly asked;
/// the default run takes the fast subset so the suite stays usable.
#[must_use]
pub fn iterations(fast: u32, heavy: u32) -> u32 {
    std::env::var("TENSORFS_HEAVY").ok().map_or(fast, |_| heavy)
}

// ---------------------------------------------------------------------------
// Scratch roots
// ---------------------------------------------------------------------------

static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch directory that removes itself, so a failing assertion cannot
/// leak a multi-megabyte tree into the shared box's temp space.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-rb-{name}-{}-{}",
            std::process::id(),
            SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch root creates");
        Self(path)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// The independent consistency oracle
// ---------------------------------------------------------------------------

/// What a direct walk of the store root found. Computed from bytes on disk,
/// never from the library's own bookkeeping.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Consistency {
    /// Whether `objects/sha256` existed at the scanned root at all. `walk`
    /// returns silently when `read_dir` fails, so without this an unreadable
    /// or non-existent namespace is indistinguishable from a clean one.
    pub namespace_present: bool,
    pub objects: u64,
    /// Every entry name found under `tmp/`. Names, not a count, because the
    /// only useful questions about a leftover temp are "how many" AND "is it
    /// a library temp the reclaimer will ever pick up".
    pub temps: Vec<String>,
    /// Digest paths whose resident bytes hash to something else. Must always
    /// be empty: an object is installed only after its bytes verified, and
    /// nothing ever rewrites one in place.
    pub corrupt: Vec<String>,
    /// Files under `objects/` whose name is not a well-formed digest, or
    /// which are not regular files. Must always be empty.
    pub malformed: Vec<String>,
}

impl Consistency {
    /// Walks `root` and rehashes every resident object.
    pub fn scan(root: &Path) -> Self {
        let mut report = Self::default();
        let namespace = root.join("objects").join("sha256");
        report.namespace_present = namespace.is_dir();
        walk(&namespace, &mut report);
        report.temps = std::fs::read_dir(root.join("tmp"))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        report.temps.sort();
        report
    }

    /// The store is intact: nothing corrupt, nothing malformed — and the
    /// namespace was actually there to examine.
    ///
    /// Deliberately says NOTHING about [`Self::temps`]. A temp is the normal
    /// mid-write state of a live admission, and this runs against stores that
    /// other processes and other threads are still writing, so folding
    /// `temps.is_empty()` in here would make every concurrent caller flaky for
    /// a reason unrelated to intactness. Callers that KNOW no writer can be
    /// live say so with [`Self::assert_no_temps`].
    ///
    /// The last clause is not pedantry. `walk` returns silently when
    /// `read_dir` fails, so without it a scan of a wrong path, or of a root
    /// that was never opened as a store, reports a perfectly intact store and
    /// passes. `ObjectStore::open` always `create_dir_all`s `objects/sha256`,
    /// so an absent namespace means the scan looked somewhere no store lives —
    /// never a legitimately empty one. `kill_matrix` leans on this as its
    /// primary independent oracle after every round, so it must not be able to
    /// pass by having examined nothing.
    pub fn assert_intact(&self, context: &str) {
        assert!(
            self.namespace_present,
            "{context}: no objects/sha256 namespace at the scanned root — this \
             scan examined nothing, so it cannot vouch for anything"
        );
        assert!(
            self.corrupt.is_empty(),
            "{context}: resident objects whose bytes do not hash to their name: {:?}",
            self.corrupt
        );
        assert!(
            self.malformed.is_empty(),
            "{context}: malformed entries under objects/: {:?}",
            self.malformed
        );
    }

    /// No temp survives at all. Only sound where the caller knows every writer
    /// against this root has exited AND that the operation under test does not
    /// write objects locally at all — a SIGKILLed admission legitimately
    /// strands a temp, which is why [`Self::assert_intact`] cannot say this.
    pub fn assert_no_temps(&self, context: &str) {
        assert!(
            self.temps.is_empty(),
            "{context}: {} temp(s) left under tmp/: {:?}",
            self.temps.len(),
            self.temps
        );
    }

    /// [`Self::assert_intact`], plus proof that at least `minimum` objects
    /// were rehashed. Use where the caller knows the store cannot be empty, so
    /// that "verified 40 objects, all good" can never be confused with "found
    /// nothing to verify".
    pub fn assert_examined(&self, minimum: u64, context: &str) {
        assert!(
            self.objects >= minimum,
            "{context}: the scan rehashed {} objects, expected at least {minimum} \
             — an oracle that examined nothing proves nothing",
            self.objects
        );
        self.assert_intact(context);
    }
}

fn walk(dir: &Path, report: &mut Consistency) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            walk(&path, report);
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !metadata.is_file() {
            report.malformed.push(name);
            continue;
        }
        report.objects += 1;
        if name.len() != 64
            || !name
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            report.malformed.push(name);
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            report.malformed.push(name);
            continue;
        };
        if hex(&Sha256::digest(&bytes)) != name {
            report.corrupt.push(name);
        }
    }
}

/// What a transport whose corpus never reaches the blob lane answers if the
/// engine ever asks. It never should — a lane it never lists cannot be
/// driven — so this is a loud refusal rather than a silent success.
fn hex_bytes(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Splits `fake-blob-put://<upload-id>/<digest-hex>/<part-number>`. The
/// staging URL carries the upload identity, exactly as a presigned part URL
/// does, so a re-grant that changed the upload id genuinely writes somewhere
/// else and the parts already landed are orphaned.
fn parse_blob_part_url(url: &str) -> Result<(String, [u8; 32], u32), TransportError> {
    let bad = |what: &str| TransportError::Refused {
        code: "bad_request".to_owned(),
        detail: format!("{what}: {url}"),
    };
    let tail = url
        .strip_prefix("fake-blob-put://")
        .ok_or_else(|| bad("not a blob part url"))?;
    let (head, number) = tail.rsplit_once('/').ok_or_else(|| bad("no part number"))?;
    let (upload_id, hex) = head.rsplit_once('/').ok_or_else(|| bad("no digest"))?;
    let digest = parse_hex_digest(hex).ok_or_else(|| bad("undecodable digest"))?;
    let number = number.parse().map_err(|_| bad("undecodable part number"))?;
    Ok((upload_id.to_owned(), *digest.as_bytes(), number))
}

pub fn blob_lane_absent() -> TransportError {
    TransportError::Refused {
        code: "blob_lane_unsupported".to_owned(),
        detail: "this transport lists no blob-lane objects".to_owned(),
    }
}

/// The on-disk shapes a reclaimer will consider: an admission temp for
/// `ObjectStore::collect_abandoned_temps`, and a projection lease for
/// `Layout::reap_scratch`.
///
/// Spelled out here rather than imported, because the point is to check the
/// bytes on disk against the shape independently of whatever the library
/// currently believes it writes. A stranded temp outside these shapes is a
/// PERMANENT leak: no reclaimer will ever look at it.
#[must_use]
pub fn is_library_temp(name: &str) -> bool {
    (name.starts_with("obj-") || name.starts_with("building-")) && name.ends_with(".tmp")
}

#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Every data object a snapshot names must be resident and rehash correctly,
/// and the snapshot's stored bytes must still hash to its id. This is the
/// "no head references bytes we did not verify" invariant, checked through
/// the public API against a store that has been through a fault gauntlet.
pub fn assert_snapshot_fully_backed(meta: &WorkspaceStore, id: &SnapshotId, context: &str) {
    let snapshot = meta
        .load_snapshot(id)
        .unwrap_or_else(|error| panic!("{context}: snapshot {id} did not load: {error}"));
    assert_eq!(
        snapshot.snapshot_id(),
        *id,
        "{context}: snapshot {id} re-encodes to a different identity"
    );
    for (path, entry) in snapshot.entries() {
        let Entry::File { body, .. } = entry else {
            continue;
        };
        for record in body.records().iter() {
            if let FileRecord::Data { digest, length } = record {
                let resident = meta.store().verify(digest).unwrap_or_else(|error| {
                    panic!("{context}: {path} references unverifiable {digest}: {error}")
                });
                assert_eq!(
                    resident, *length,
                    "{context}: {path} record length disagrees with resident bytes for {digest}"
                );
            }
        }
    }
}

/// Reconstructs one file's logical bytes from a snapshot's records, reading
/// through the object store exactly as a consumer would.
#[must_use]
pub fn reconstruct(meta: &WorkspaceStore, id: &SnapshotId, path: &str) -> Vec<u8> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut bytes = Vec::new();
    let mut found = false;
    for (entry_path, entry) in snapshot.entries() {
        if entry_path.as_str() != path {
            continue;
        }
        let Entry::File { body, .. } = entry else {
            panic!("{path} is not a regular file");
        };
        found = true;
        for record in body.records().iter() {
            match record {
                FileRecord::Hole { length } => {
                    bytes.extend(std::iter::repeat_n(0_u8, *length as usize));
                }
                FileRecord::Data { digest, .. } => {
                    let mut file = meta.store().open_object(digest).expect("object opens");
                    let mut chunk = Vec::new();
                    file.read_to_end(&mut chunk).expect("object reads");
                    bytes.extend_from_slice(&chunk);
                }
            }
        }
    }
    assert!(found, "{path} is absent from snapshot {id}");
    bytes
}

// ---------------------------------------------------------------------------
// Workspace construction
// ---------------------------------------------------------------------------

/// Commits one file per row as alternating `Data`/`Hole` records, so seal
/// keeps the committed multi-object boundaries verbatim, then seals.
pub fn sealed_workspace(
    root: &Path,
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

pub fn mutations_for(meta: &WorkspaceStore, files: &[(&str, Vec<Vec<u8>>)]) -> Vec<Mutation> {
    let mut directories = std::collections::BTreeSet::new();
    for (path, _) in files {
        let mut parent = Path::new(path).parent();
        while let Some(ancestor) = parent {
            if !ancestor.as_os_str().is_empty() {
                directories.insert(ancestor.to_string_lossy().into_owned());
            }
            parent = ancestor.parent();
        }
    }
    directories
        .into_iter()
        .map(|path| Mutation::Mkdir { path })
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

/// The unique data digests one sealed snapshot names, in manifest order.
#[must_use]
pub fn data_digests(meta: &WorkspaceStore, id: &SnapshotId) -> Vec<ObjectDigest> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut seen = HashSet::new();
    let mut digests = Vec::new();
    for (_path, entry) in snapshot.entries() {
        if let Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, .. } = record
                    && seen.insert(*digest.as_bytes())
                {
                    digests.push(*digest);
                }
            }
        }
    }
    digests
}

// ---------------------------------------------------------------------------
// The fault-injecting hub
// ---------------------------------------------------------------------------

const MAX_PACKS_PER_REQUEST: usize = 16;

/// How one download should misbehave. Every variant models a real-world
/// class in which a hub, a CDN, or a proxy hands back something other than
/// the exact bytes that were asked for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DownloadFault {
    /// Correct digest name, entirely different bytes.
    WrongBytes,
    /// A valid, correctly-hashing object — but not the one requested.
    SwappedWithAnother,
    /// A genuine prefix of the expected object: every byte correct, the
    /// stream just stops early.
    ValidPrefix,
    /// Exactly the right length, one byte different: length checks pass and
    /// only the digest can catch it.
    SameLengthDifferentContent,
    /// One flipped bit.
    BitFlip,
    /// Nothing at all.
    Empty,
    /// More bytes than the grant declared.
    Overlong,
}

/// Everything the hub can be told to do wrong. Defaults are all honest.
#[derive(Clone, Debug, Default)]
pub struct Faults {
    /// Per-digest download misbehaviour.
    pub downloads: HashMap<[u8; 32], DownloadFault>,
    /// Corrupt the manifest blob served for this snapshot id, so its bytes
    /// no longer hash to the id the caller asked for.
    pub corrupt_manifest: bool,
    /// Serve a manifest that is a truncated prefix of the real one.
    pub truncate_manifest: bool,
    /// Transient upload failures to burn before succeeding.
    pub fail_uploads: u32,
    /// Upload grant expiries to burn before succeeding.
    pub expire_uploads: u32,
    /// Grant-request lease expiries to burn.
    pub expire_grant_calls: u32,
    /// Retryable `complete` answers to burn.
    pub incomplete_completes: u32,
    /// Make every `complete` fail terminally with this code.
    pub terminal_complete: Option<String>,
    /// Objects promoted per `complete`; 0 is unlimited.
    pub promote_budget: usize,
    /// Drop this digest from every download-grant answer, modelling a hub
    /// that omits a grant the manifest requires.
    pub omit_download_grant: Option<[u8; 32]>,
    /// Add an unrequested grant to every download-grant answer.
    pub inject_unrequested_grant: bool,
    /// Advance the head out from under an in-flight push after this many
    /// pack-grant calls, so `complete` meets a conflict it did not declare
    /// against.
    pub advance_head_after_grants: Option<u32>,
    /// Refuse every `pack_grants` call with an unknown-session error.
    pub forget_sessions: bool,
    /// Transient I/O failures to burn on downloads before succeeding.
    pub fail_downloads: u32,
    /// The pack-payload bound this hub declares. Zero means the wire's own
    /// 64 MiB. A test lowers it to put small objects in the blob lane and
    /// exercise the multipart path without moving 64 MiB.
    pub pack_payload_bound: u64,
    /// Transient I/O failures to burn on blob PART uploads before succeeding.
    pub fail_blob_parts: u32,
    /// Expire the blob grant ONCE, after this many parts have landed. `Some(0)`
    /// expires the first part. Modelling the lease running out mid-blob is
    /// what makes a resume claim testable: some parts are at the store and
    /// some are not.
    pub expire_blob_parts_after: Option<u32>,
    /// Silently corrupt this part number's bytes as they land, the way a
    /// store that computes no checksum of its own would accept them. Only the
    /// admission-time stream hash can catch it.
    pub corrupt_blob_part: Option<u32>,
    /// Drop this digest from every blob-grant answer.
    pub omit_blob_grant: Option<[u8; 32]>,
    /// Answer a re-grant with a FRESH upload id, orphaning the parts already
    /// landed — the defect th#2064 found in its own resume promise.
    pub forget_blob_upload_id: bool,
    /// Put every missing object in the blob lane regardless of size, so the
    /// client meets a remote that partitioned wrongly.
    pub blob_lane_everything: bool,
}

/// `(pack lane, blob lane)` — what one declare or re-probe answers with.
type LaneViews = (Vec<(ObjectDigest, u64)>, Vec<(ObjectDigest, u64)>);

struct PackRow {
    staging_key: String,
    objects: Vec<[u8; 32]>,
}

/// One blob's live multipart upload, modelled the way the hub models it: an
/// upload id, a uniform part size, and the etags the client has reported.
/// The part BYTES live in `HubState::blob_parts`, which stands in for the
/// object store — the hub cannot see them until it assembles them.
struct BlobUpload {
    upload_id: String,
    length: u64,
    part_size: u64,
    reported: BTreeMap<u32, String>,
}

struct Session {
    snapshot_id: SnapshotId,
    expected_head: Option<SnapshotId>,
    closure: Vec<([u8; 32], u64)>,
    manifest: Vec<u8>,
    packs: BTreeMap<String, PackRow>,
    blobs: BTreeMap<[u8; 32], BlobUpload>,
}

#[derive(Default)]
pub struct HubState {
    pub objects: HashMap<[u8; 32], Vec<u8>>,
    staged: HashMap<String, Vec<u8>>,
    /// The store's view of a multipart upload: `(upload_id, part_number)` to
    /// bytes. Keyed by upload id and not by digest, so adopting the same
    /// upload on a re-grant is what makes a resume find its parts — and
    /// answering with a new id genuinely orphans them.
    blob_parts: HashMap<(String, u32), Vec<u8>>,
    /// Part PUTs the client actually made, per digest.
    pub blob_part_puts: HashMap<[u8; 32], u32>,
    /// Blob grant calls the client made, per digest.
    pub blob_grant_calls: HashMap<[u8; 32], u32>,
    /// Monotonic, so `forget_blob_upload_id` really does mint a NEW id every
    /// time rather than a stable one that accidentally resumes.
    blob_regrants: u64,
    pub head: Option<SnapshotId>,
    sessions: HashMap<String, Session>,
    next: u64,
    pub uploads_by_digest: HashMap<[u8; 32], u32>,
    pub downloads_by_digest: HashMap<[u8; 32], u32>,
    pub grant_calls: u32,
    pub complete_calls: u32,
    pub faults: Faults,
}

/// An in-memory hub that runs the real decoders and the real wire rules, with
/// an injectable fault surface. Single-threaded by construction (`RefCell`),
/// matching the engine's sequential use.
pub struct FaultHub {
    pub state: std::cell::RefCell<HubState>,
}

impl FaultHub {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: std::cell::RefCell::new(HubState::default()),
        }
    }

    #[must_use]
    pub fn with_faults(faults: Faults) -> Self {
        let hub = Self::new();
        hub.state.borrow_mut().faults = faults;
        hub
    }

    pub fn set_faults(&self, faults: Faults) {
        self.state.borrow_mut().faults = faults;
    }

    /// Clears every injected fault, leaving the hub honest but keeping the
    /// objects and head it already holds — the "the outage ends" transition.
    pub fn heal(&self) {
        self.state.borrow_mut().faults = Faults::default();
    }

    /// The payload bound this hub declares, and therefore the lane split.
    fn payload_bound(state: &HubState) -> u64 {
        if state.faults.pack_payload_bound == 0 {
            tfp1::MAX_PACK_PAYLOAD
        } else {
            state.faults.pack_payload_bound
        }
    }

    /// Whether one blob's reported parts cover it, which is the blob-lane
    /// equivalent of a staged pack: the bytes are at the store and the hub
    /// has what it needs to assemble them.
    fn blob_is_staged(session: &Session, digest: &[u8; 32]) -> bool {
        session.blobs.get(digest).is_some_and(|upload| {
            let expected = upload.length.div_ceil(upload.part_size).max(1) as usize;
            upload.reported.len() == expected
        })
    }

    /// The missing set split into the two lanes, in canonical order.
    fn lane_views(state: &HubState, session: &Session) -> LaneViews {
        let bound = Self::payload_bound(state);
        let mut packs = Vec::new();
        let mut blobs = Vec::new();
        for (digest, length) in Self::missing_view(state, session) {
            if state.faults.blob_lane_everything || length > bound {
                // A blob whose parts are all reported is staged: the bytes
                // are at the store and the hub has what it needs to assemble
                // them, so it drops out of the lane exactly as a staged pack
                // drops out of the pack lane.
                if !Self::blob_is_staged(session, digest.as_bytes()) {
                    blobs.push((digest, length));
                }
            } else {
                packs.push((digest, length));
            }
        }
        (packs, blobs)
    }

    fn missing_view(state: &HubState, session: &Session) -> Vec<(ObjectDigest, u64)> {
        let mut staged_members = HashSet::new();
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

impl Default for FaultHub {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncTransport for FaultHub {
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
        let mut seen = HashSet::new();
        for (_path, entry) in snapshot.entries() {
            if let Entry::File { body, .. } = entry {
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
            blobs: BTreeMap::new(),
        };
        let have = session
            .closure
            .iter()
            .filter(|(digest, _)| state.objects.contains_key(digest))
            .map(|(digest, _)| ObjectDigest::from_bytes(*digest))
            .collect();
        let (missing, missing_blobs) = Self::lane_views(&state, &session);
        let plan = SyncPlan {
            snapshot_id: id,
            session: key.clone(),
            have,
            staged_packs: Vec::new(),
            missing,
            missing_blobs,
            max_pack_payload: Self::payload_bound(&state),
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
        state.grant_calls += 1;
        if state.faults.forget_sessions {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session_key.to_owned(),
            });
        }
        if state.faults.expire_grant_calls > 0 && !claims.is_empty() {
            state.faults.expire_grant_calls -= 1;
            return Err(TransportError::Expired(
                "injected session lease expiry".to_owned(),
            ));
        }
        if let Some(after) = state.faults.advance_head_after_grants
            && state.grant_calls > after
        {
            // A competing publisher wins the head mid-push.
            let mut hasher = Sha256::new();
            hasher.update(b"a competing head");
            let bytes: [u8; 32] = hasher.finalize().into();
            state.head = Some(SnapshotId::from_bytes(bytes));
            state.faults.advance_head_after_grants = None;
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
        let bound = Self::payload_bound(&state);
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
            let mut seen = HashSet::new();
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
                // The lanes cannot be confused: a pack grant naming a blob-lane
                // object is refused here exactly as the hub refuses it.
                if *size > bound {
                    return Err(TransportError::Refused {
                        code: "object_exceeds_pack_payload".to_owned(),
                        detail: format!("{digest} belongs to the blob lane"),
                    });
                }
                payload += size;
            }
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
        let (missing, missing_blobs) = Self::lane_views(&state, session);
        Ok(GrantsPlan {
            grants,
            staged_packs: Self::staged_rows(&state, session),
            missing,
            missing_blobs,
        })
    }

    fn blob_grants(
        &self,
        session_key: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError> {
        let mut state = self.state.borrow_mut();
        if digests.len() > tensorfs_core::sync::BLOB_GRANT_BATCH {
            return Err(TransportError::Refused {
                code: "bad_request".to_owned(),
                detail: "too many blob digests".to_owned(),
            });
        }
        let bound = Self::payload_bound(&state);
        let forget = state.faults.forget_blob_upload_id;
        let omit = state.faults.omit_blob_grant;
        let lane_everything = state.faults.blob_lane_everything;
        let Some(session) = state.sessions.get(session_key) else {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session_key.to_owned(),
            });
        };
        let sizes: HashMap<[u8; 32], u64> = session.closure.iter().copied().collect();
        let existing: HashMap<[u8; 32], (String, u64)> = session
            .blobs
            .iter()
            .map(|(digest, upload)| (*digest, (upload.upload_id.clone(), upload.part_size)))
            .collect();

        let mut opened: Vec<([u8; 32], BlobUpload)> = Vec::new();
        let mut answers = Vec::new();
        for digest in digests {
            *state
                .blob_grant_calls
                .entry(*digest.as_bytes())
                .or_default() += 1;
            if omit == Some(*digest.as_bytes()) {
                continue;
            }
            let Some(length) = sizes.get(digest.as_bytes()).copied() else {
                return Err(TransportError::Refused {
                    code: "bad_request".to_owned(),
                    detail: format!("{digest} is not an object of this snapshot"),
                });
            };
            // A blob grant naming a pack-lane digest is refused, the mirror of
            // the refusal in `pack_grants`.
            if length <= bound && !lane_everything {
                return Err(TransportError::Refused {
                    code: "object_below_blob_lane".to_owned(),
                    detail: format!("{digest} belongs to the pack lane"),
                });
            }
            // Part size is the SERVER's, uniform, last part excepted.
            let part_size = bound.max(1);
            let (upload_id, part_size) = match existing.get(digest.as_bytes()) {
                Some((id, size)) if !forget => (id.clone(), *size),
                // Content-addressed, like the key it stages on: a later
                // session adopts the upload an earlier one opened.
                _ => (
                    format!("upload-{}", hex_bytes(digest.as_bytes())),
                    part_size,
                ),
            };
            let upload_id = if forget {
                state.blob_regrants += 1;
                format!("{upload_id}-refreshed-{}", state.blob_regrants)
            } else {
                upload_id
            };
            let count = length.div_ceil(part_size).max(1);
            let mut parts = Vec::new();
            let mut uploaded_parts = Vec::new();
            for index in 0..count {
                let number = (index + 1) as u32;
                let size = part_size.min(length - index * part_size);
                if state.blob_parts.contains_key(&(upload_id.clone(), number)) {
                    uploaded_parts.push(number);
                }
                parts.push(BlobPart {
                    part_number: number,
                    size_bytes: size,
                    url: format!(
                        "fake-blob-put://{upload_id}/{}/{number}",
                        hex_bytes(digest.as_bytes())
                    ),
                    headers: vec![("x-amz-part-number".to_owned(), number.to_string())],
                });
            }
            // The STORE's own part list seeds the etags, not a report this
            // session may never have received — the fake's mirror of the hub
            // reading ListParts. Without it a session that uploaded none of
            // the parts could never complete the upload it adopted.
            let adopted: BTreeMap<u32, String> = state
                .blob_parts
                .iter()
                .filter(|((id, _), _)| *id == upload_id)
                .map(|((_, number), bytes)| (*number, format!("\"{}\"", sha256_hex(bytes))))
                .collect();
            opened.push((
                *digest.as_bytes(),
                BlobUpload {
                    upload_id: upload_id.clone(),
                    length,
                    part_size,
                    reported: adopted,
                },
            ));
            answers.push(BlobGrant {
                digest: *digest,
                length,
                staging_key: format!("snapshots/staging/blobs/{digest}.blob"),
                upload_id,
                part_size,
                parts,
                uploaded_parts,
            });
        }
        let session = state.sessions.get_mut(session_key).expect("session exists");
        for (digest, upload) in opened {
            // Re-granting keeps whatever etags were already reported unless
            // the upload id changed, which is exactly what orphans them.
            match session.blobs.get_mut(&digest) {
                Some(live) if live.upload_id == upload.upload_id => {}
                _ => {
                    session.blobs.insert(digest, upload);
                }
            }
        }
        Ok(answers)
    }

    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        _progress: ProgressSink<'_>,
    ) -> Result<String, TransportError> {
        let mut state = self.state.borrow_mut();
        if state.faults.fail_blob_parts > 0 {
            state.faults.fail_blob_parts -= 1;
            return Err(TransportError::Io("injected part carrier fault".to_owned()));
        }
        if let Some(after) = state.faults.expire_blob_parts_after
            && state.blob_part_puts.values().copied().sum::<u32>() >= after
        {
            state.faults.expire_blob_parts_after = None;
            return Err(TransportError::Expired(
                "injected blob grant expiry".to_owned(),
            ));
        }
        if bytes.len() as u64 != part.size_bytes {
            return Err(TransportError::Refused {
                code: "bad_request".to_owned(),
                detail: format!(
                    "part {} is {} bytes, the grant says {}",
                    part.part_number,
                    bytes.len(),
                    part.size_bytes
                ),
            });
        }
        let (upload_id, digest, number) = parse_blob_part_url(&part.url)?;
        *state.blob_part_puts.entry(digest).or_default() += 1;
        // The store accepts whatever arrives and names it; it computes no
        // digest of its own, which is exactly why the hub must stream-hash
        // the assembled object at admission.
        let mut landed = bytes.to_vec();
        if state.faults.corrupt_blob_part == Some(part.part_number) {
            landed[0] ^= 0xFF;
        }
        let etag = format!("\"{}\"", sha256_hex(&landed));
        state.blob_parts.insert((upload_id, number), landed);
        Ok(etag)
    }

    fn report_blob_parts(
        &self,
        session_key: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError> {
        let mut state = self.state.borrow_mut();
        let landed: HashMap<(String, u32), String> = state
            .blob_parts
            .iter()
            .map(|((id, number), bytes)| {
                ((id.clone(), *number), format!("\"{}\"", sha256_hex(bytes)))
            })
            .collect();
        let Some(session) = state.sessions.get_mut(session_key) else {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session_key.to_owned(),
            });
        };
        let Some(upload) = session.blobs.get_mut(digest.as_bytes()) else {
            return Err(TransportError::Refused {
                code: "bad_request".to_owned(),
                detail: format!("{digest} has no live upload in this session"),
            });
        };
        for part in parts {
            let key = (upload.upload_id.clone(), part.part_number);
            match landed.get(&key) {
                Some(etag) if *etag == part.etag => {
                    upload.reported.insert(part.part_number, part.etag.clone());
                }
                Some(_) => {
                    return Err(TransportError::Refused {
                        code: "part_etag_mismatch".to_owned(),
                        detail: format!("part {} etag does not match", part.part_number),
                    });
                }
                None => {
                    return Err(TransportError::Refused {
                        code: "part_missing".to_owned(),
                        detail: format!("part {} was never uploaded", part.part_number),
                    });
                }
            }
        }
        Ok(())
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        _progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        let mut state = self.state.borrow_mut();
        if state.faults.fail_uploads > 0 {
            state.faults.fail_uploads -= 1;
            return Err(TransportError::Io("injected carrier fault".to_owned()));
        }
        if state.faults.expire_uploads > 0 {
            state.faults.expire_uploads -= 1;
            return Err(TransportError::Expired("injected grant expiry".to_owned()));
        }
        if sha256_hex(pack) != grant.pack_sha256 {
            return Err(TransportError::Refused {
                code: "checksum-mismatch".to_owned(),
                detail: "pack bytes do not hash to the granted checksum".to_owned(),
            });
        }
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
        state.complete_calls += 1;
        if let Some(code) = state.faults.terminal_complete.clone() {
            return Ok(CompleteStatus::Failed { code });
        }
        if state.faults.incomplete_completes > 0 {
            state.faults.incomplete_completes -= 1;
            return Ok(CompleteStatus::Incomplete {
                code: "promote_incomplete".to_owned(),
                // The injected fault admits nothing on purpose: this is the
                // "standing still" shape the client's stall budget exists for.
                promoted: 0,
                total: 0,
            });
        }
        let Some(session) = state.sessions.get(session_key) else {
            return Err(TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session_key.to_owned(),
            });
        };
        let budget = state.faults.promote_budget;
        let (snapshot_id, expected, closure, manifest, staged_pack_keys) = (
            session.snapshot_id,
            session.expected_head,
            session.closure.clone(),
            session.manifest.clone(),
            session
                .packs
                .values()
                .map(|pack| pack.staging_key.clone())
                .collect::<Vec<_>>(),
        );
        // The blob lane admits AT MOST ONE blob per call, because each
        // admission streams a whole object. `promote_incomplete` is the
        // progress mechanism, never a timeout.
        let ready: Vec<([u8; 32], String, Vec<u32>)> = session
            .blobs
            .iter()
            .filter(|(digest, _)| {
                !state.objects.contains_key(*digest) && Self::blob_is_staged(session, digest)
            })
            .map(|(digest, upload)| {
                (
                    *digest,
                    upload.upload_id.clone(),
                    upload.reported.keys().copied().collect(),
                )
            })
            .collect();
        if let Some((digest, upload_id, numbers)) = ready.into_iter().next() {
            let mut assembled = Vec::new();
            for number in numbers {
                let Some(bytes) = state.blob_parts.get(&(upload_id.clone(), number)) else {
                    return Ok(CompleteStatus::Incomplete {
                        code: "upload_incomplete".to_owned(),
                        promoted: closure
                            .iter()
                            .filter(|(digest, _)| state.objects.contains_key(digest))
                            .count() as u64,
                        total: closure.len() as u64,
                    });
                };
                assembled.extend_from_slice(bytes);
            }
            // Stream-hashed exactly once, here, against the declared digest.
            // Wire parts never entered identity.
            if sha256_hex(&assembled) != hex_bytes(&digest) {
                // Terminal, and the staging state is destroyed with it.
                state.blob_parts.retain(|(id, _), _| *id != upload_id);
                if let Some(session) = state.sessions.get_mut(session_key) {
                    session.blobs.remove(&digest);
                }
                return Ok(CompleteStatus::Failed {
                    code: "blob_digest_mismatch".to_owned(),
                });
            }
            state.objects.insert(digest, assembled);
            state.blob_parts.retain(|(id, _), _| *id != upload_id);
            return Ok(CompleteStatus::Incomplete {
                code: "promote_incomplete".to_owned(),
                promoted: closure
                    .iter()
                    .filter(|(digest, _)| state.objects.contains_key(digest))
                    .count() as u64,
                total: closure.len() as u64,
            });
        }
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
        let state = self.state.borrow();
        let mut grants: Vec<DownloadGrant> = digests
            .iter()
            .filter(|digest| state.faults.omit_download_grant != Some(*digest.as_bytes()))
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
            .collect();
        if state.faults.inject_unrequested_grant {
            // An object the caller never asked for and the manifest does not
            // name. The engine must ignore it, never fetch it.
            if let Some((digest, bytes)) = state
                .objects
                .iter()
                .find(|(digest, _)| !digests.iter().any(|want| want.as_bytes() == *digest))
            {
                grants.push(DownloadGrant {
                    digest: ObjectDigest::from_bytes(*digest),
                    length: bytes.len() as u64,
                    url: "fake-get://unrequested".to_owned(),
                });
            }
        }
        Ok(grants)
    }

    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        _progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        let served = self.served_bytes(grant)?;
        sink.write_all(&served)
            .map_err(|error| TransportError::Io(error.to_string()))?;
        Ok(served.len() as u64)
    }
}

impl FaultHub {
    /// What this hub would put on the wire for one grant, faults included.
    /// Split out so `download` is only the streaming half.
    fn served_bytes(&self, grant: &DownloadGrant) -> Result<Vec<u8>, TransportError> {
        let mut state = self.state.borrow_mut();
        if state.faults.fail_downloads > 0 {
            state.faults.fail_downloads -= 1;
            return Err(TransportError::Io("injected download fault".to_owned()));
        }
        *state
            .downloads_by_digest
            .entry(*grant.digest.as_bytes())
            .or_default() += 1;
        let honest = state
            .objects
            .get(grant.digest.as_bytes())
            .cloned()
            .ok_or_else(|| TransportError::Refused {
                code: "not-found".to_owned(),
                detail: grant.digest.to_string(),
            })?;

        // Manifest-specific corruption, applied when the requested digest is
        // the manifest blob of the current head.
        let is_manifest = state
            .head
            .as_ref()
            .is_some_and(|head| manifest_object_digest(head) == grant.digest);
        if is_manifest && state.faults.corrupt_manifest {
            let mut bytes = honest;
            let last = bytes.len() - 1;
            bytes[last] ^= 0xFF;
            return Ok(bytes);
        }
        if is_manifest && state.faults.truncate_manifest {
            let mut bytes = honest;
            bytes.truncate(bytes.len() / 2);
            return Ok(bytes);
        }

        let Some(fault) = state.faults.downloads.get(grant.digest.as_bytes()).cloned() else {
            return Ok(honest);
        };
        Ok(match fault {
            DownloadFault::WrongBytes => b"entirely different bytes from a lying hub".to_vec(),
            DownloadFault::SwappedWithAnother => state
                .objects
                .iter()
                .find(|(digest, _)| *digest != grant.digest.as_bytes())
                .map(|(_, bytes)| bytes.clone())
                .unwrap_or_default(),
            DownloadFault::ValidPrefix => honest[..honest.len() / 2].to_vec(),
            DownloadFault::SameLengthDifferentContent => {
                let mut bytes = honest;
                let last = bytes.len() - 1;
                bytes[last] = bytes[last].wrapping_add(1);
                bytes
            }
            DownloadFault::BitFlip => {
                let mut bytes = honest;
                bytes[0] ^= 0x01;
                bytes
            }
            DownloadFault::Empty => Vec::new(),
            DownloadFault::Overlong => {
                let mut bytes = honest;
                bytes.extend_from_slice(b"trailing garbage");
                bytes
            }
        })
    }
}

// ---------------------------------------------------------------------------
// The adversarial HTTP server
// ---------------------------------------------------------------------------

/// One scripted answer. Each variant models a transport-level or
/// HTTP-semantic failure class a real deployment produces.
#[derive(Clone, Debug)]
pub enum Reply {
    /// A well-formed status + body.
    Body { status: u16, body: String },
    /// A well-formed status + raw octet body (for object downloads).
    Bytes { status: u16, body: Vec<u8> },
    /// Headers promising `declared` bytes, then only `send` of them, then a
    /// hard close: the classic truncated response.
    Short {
        status: u16,
        declared: usize,
        send: Vec<u8>,
    },
    /// Headers promising `declared` bytes, then more than that.
    Overlong {
        status: u16,
        declared: usize,
        body: Vec<u8>,
    },
    /// Status line and headers, then the connection closes with no body at
    /// all despite a non-zero Content-Length.
    HeadersOnly { status: u16, declared: usize },
    /// Accept the connection and close it immediately — no status line.
    HangUp,
    /// A chunked response that emits one chunk and never terminates before
    /// closing, so the framing is incomplete.
    UnterminatedChunked { status: u16, chunk: Vec<u8> },
    /// A redirect to `location`, typically another host.
    Redirect { status: u16, location: String },
    /// A proxy's HTML error page served with a 200.
    HtmlPage,
}

impl Reply {
    pub fn json(status: u16, body: &str) -> Self {
        Self::Body {
            status,
            body: body.to_owned(),
        }
    }
}

/// Serves `answers` sequentially on loopback. Returns the base URL; the
/// listener thread exits when the script is exhausted.
///
/// Requests are recorded so a test can prove what the transport actually
/// emitted, and the receiver is returned alongside the URL.
pub struct FaultServer {
    pub base: String,
    pub requests: mpsc::Receiver<RecordedRequest>,
}

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub authorization: Option<String>,
    pub host: Option<String>,
    pub body: Vec<u8>,
}

/// Ports handed out by [`FaultServer::dead_address`], which no server in this
/// binary may then bind.
///
/// The Windows runner proved the need: `dead_address` frees an ephemeral port
/// and the very next test's server was handed the same one, so a connect that
/// had to be refused was answered with JSON instead. A port promised dead
/// stays dead for the rest of the process.
static DEAD_PORTS: std::sync::Mutex<Vec<u16>> = std::sync::Mutex::new(Vec::new());

fn dead_ports() -> std::sync::MutexGuard<'static, Vec<u16>> {
    DEAD_PORTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl FaultServer {
    /// Binds a loopback listener that plays `answers` in order, never on a
    /// port some other test was promised is dead.
    pub fn start(answers: Vec<Reply>) -> Self {
        let mut rejected = Vec::new();
        let listener = loop {
            let listener = TcpListener::bind("127.0.0.1:0").expect("loopback binds");
            let port = listener.local_addr().expect("bound address").port();
            if !dead_ports().contains(&port) {
                break listener;
            }
            // Held, not dropped, so the next bind cannot be handed it again.
            rejected.push(listener);
        };
        drop(rejected);
        let address = listener.local_addr().expect("bound address");
        let (sender, requests) = mpsc::channel();
        thread::spawn(move || {
            for answer in answers {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let Some(request) = read_request(&mut stream) else {
                    continue;
                };
                let _ = sender.send(request);
                write_reply(&mut stream, &answer);
            }
            // Keep the port bound briefly so a surplus request meets a
            // refusal rather than a different process's socket.
            drop(listener);
        });
        Self {
            base: format!("http://127.0.0.1:{}", address.port()),
            requests,
        }
    }

    /// A base URL nothing is listening on, for connect-failure coverage.
    ///
    /// The port is recorded in [`DEAD_PORTS`] before it is released, so no
    /// later [`FaultServer::start`] can bind it and turn a refusal the test
    /// requires into an answer it must never get.
    #[must_use]
    pub fn dead_address() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("loopback binds");
        let port = listener.local_addr().expect("bound address").port();
        dead_ports().push(port);
        drop(listener);
        format!("http://127.0.0.1:{port}")
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<RecordedRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    let head_end = loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_blank_line(&buffer) {
            break position;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let mut authorization = None;
    let mut host = None;
    let mut content_length = 0_usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_owned();
        match name.as_str() {
            "authorization" => authorization = Some(value),
            "host" => host = Some(value),
            "content-length" => content_length = value.parse().unwrap_or(0),
            _ => {}
        }
    }
    let body_start = head_end + 4;
    let mut have = buffer.len();
    while have < body_start + content_length {
        let Ok(read) = stream.read(&mut chunk) else {
            break;
        };
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        have += read;
    }
    Some(RecordedRequest {
        method,
        path,
        authorization,
        host,
        body: buffer[body_start.min(buffer.len())..].to_vec(),
    })
}

fn write_reply(stream: &mut std::net::TcpStream, answer: &Reply) {
    match answer {
        Reply::Body { status, body } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(body.as_bytes());
        }
        Reply::Bytes { status, body } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-type: application/octet-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(body);
        }
        Reply::Short {
            status,
            declared,
            send,
        } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-length: {declared}\r\nconnection: close\r\n\r\n"
            );
            let _ = stream.write_all(send);
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Both);
        }
        Reply::Overlong {
            status,
            declared,
            body,
        } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-length: {declared}\r\nconnection: close\r\n\r\n"
            );
            let _ = stream.write_all(body);
        }
        Reply::HeadersOnly { status, declared } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-length: {declared}\r\nconnection: close\r\n\r\n"
            );
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Both);
        }
        Reply::HangUp => {
            let _ = stream.shutdown(Shutdown::Both);
        }
        Reply::UnterminatedChunked { status, chunk } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\ntransfer-encoding: chunked\r\n\r\n"
            );
            let _ = write!(stream, "{:x}\r\n", chunk.len());
            let _ = stream.write_all(chunk);
            let _ = stream.write_all(b"\r\n");
            let _ = stream.flush();
            let _ = stream.shutdown(Shutdown::Both);
        }
        Reply::Redirect { status, location } => {
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nlocation: {location}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
        }
        Reply::HtmlPage => {
            let body =
                "<html><head><title>502 Bad Gateway</title></head><body>proxy error</body></html>";
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(body.as_bytes());
        }
    }
    let _ = stream.flush();
}

fn find_blank_line(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

// ---------------------------------------------------------------------------
// A directory-backed hub
// ---------------------------------------------------------------------------

/// A [`SyncTransport`] whose entire state is a directory tree, so a SEPARATE
/// PROCESS can push or pull against it and be SIGKILLed mid-transfer.
///
/// The in-memory [`FaultHub`] cannot do that — its state dies with the
/// process — which would leave the restart-convergence family testing
/// simulated interruptions instead of real ones. This models the same wire
/// rules: session-scoped staging, grants bound to a claimed checksum,
/// promotion at `complete` with the manifest admitted strictly before the
/// head moves.
///
/// Layout:
/// ```text
///   objects/<hex>              promoted objects
///   staged/<session>/<sha>     staged TFP1 packs
///   sessions/<session>.json    session state
///   sessions/<session>.tfm1    the declared manifest bytes
///   head                       current head, hex
/// ```
pub struct DirTransport {
    root: PathBuf,
}

impl DirTransport {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        for directory in ["objects", "staged", "sessions"] {
            std::fs::create_dir_all(root.join(directory)).expect("hub layout creates");
        }
        Self { root }
    }

    fn object_file(&self, digest: &ObjectDigest) -> PathBuf {
        self.root.join("objects").join(digest_hex(digest))
    }

    fn session_file(&self, session: &str) -> PathBuf {
        self.root.join("sessions").join(format!("{session}.json"))
    }

    fn manifest_file(&self, session: &str) -> PathBuf {
        self.root.join("sessions").join(format!("{session}.tfm1"))
    }

    fn read_head(&self) -> Option<SnapshotId> {
        std::fs::read_to_string(self.root.join("head"))
            .ok()
            .and_then(|raw| SnapshotId::parse_hex(raw.trim()))
    }

    fn load_session(&self, session: &str) -> Result<serde_json::Value, TransportError> {
        let raw = std::fs::read_to_string(self.session_file(session)).map_err(|_| {
            TransportError::Refused {
                code: "unknown-session".to_owned(),
                detail: session.to_owned(),
            }
        })?;
        serde_json::from_str(&raw).map_err(|error| TransportError::Io(error.to_string()))
    }

    fn store_session(&self, session: &str, value: &serde_json::Value) {
        std::fs::write(
            self.session_file(session),
            serde_json::to_vec(value).expect("session serialises"),
        )
        .expect("session persists");
    }

    /// Objects of the closure that are neither promoted nor covered by a
    /// staged pack of this session.
    fn missing_view(&self, session: &serde_json::Value) -> Vec<(ObjectDigest, u64)> {
        let mut staged: HashSet<String> = HashSet::new();
        if let Some(packs) = session["packs"].as_object() {
            for pack in packs.values() {
                let key = pack["staging_key"].as_str().unwrap_or_default();
                if self.root.join(key).is_file() {
                    for member in pack["objects"].as_array().into_iter().flatten() {
                        staged.insert(member.as_str().unwrap_or_default().to_owned());
                    }
                }
            }
        }
        session["closure"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|row| {
                let hex = row[0].as_str()?;
                let length = row[1].as_u64()?;
                if staged.contains(hex) || self.root.join("objects").join(hex).is_file() {
                    return None;
                }
                Some((parse_hex_digest(hex)?, length))
            })
            .collect()
    }
}

fn digest_hex(digest: &ObjectDigest) -> String {
    let text = digest.to_string();
    text.strip_prefix("sha256:").unwrap_or(&text).to_owned()
}

fn parse_hex_digest(hex: &str) -> Option<ObjectDigest> {
    SnapshotId::parse_hex(hex).map(|id| ObjectDigest::from_bytes(*id.as_bytes()))
}

impl SyncTransport for DirTransport {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        let snapshot = decode(tfm1_bytes).map_err(|error| TransportError::Refused {
            code: "declaration_invalid".to_owned(),
            detail: error.to_string(),
        })?;
        if self.read_head().as_ref() != expected_head {
            return Err(TransportError::Refused {
                code: "head_conflict".to_owned(),
                detail: "expected head does not match".to_owned(),
            });
        }
        let id = snapshot.snapshot_id();
        let mut closure = Vec::new();
        let mut seen = HashSet::new();
        for (_path, entry) in snapshot.entries() {
            if let Entry::File { body, .. } = entry {
                for record in body.records().iter() {
                    if let FileRecord::Data { digest, length } = record
                        && seen.insert(*digest.as_bytes())
                    {
                        closure.push(serde_json::json!([digest_hex(digest), *length]));
                    }
                }
            }
        }
        // A content-derived session name keeps restarts deterministic, so a
        // killed-and-restarted push resumes the SAME session rather than
        // silently starting a fresh one and re-uploading everything.
        let session = format!("s-{}", &sha256_hex(tfm1_bytes)[..16]);
        // Resuming keeps only the staged-pack record; the declared facts are
        // refreshed every time. Carrying a stale `expected_head` forward
        // would make a later push against a moved head fail as a spurious
        // conflict — the head assertion above has already validated this
        // declare against the live head.
        let mut value = self
            .load_session(&session)
            .unwrap_or_else(|_| serde_json::json!({ "packs": {} }));
        value["snapshot_id"] = serde_json::json!(digest_hex(&manifest_object_digest(&id)));
        value["expected_head"] =
            serde_json::json!(expected_head.map(std::string::ToString::to_string));
        value["closure"] = serde_json::json!(closure);
        std::fs::create_dir_all(self.root.join("staged").join(&session))
            .expect("session staging creates");
        std::fs::write(self.manifest_file(&session), tfm1_bytes).expect("manifest persists");
        self.store_session(&session, &value);

        let have = value["closure"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|row| {
                let hex = row[0].as_str()?;
                self.root
                    .join("objects")
                    .join(hex)
                    .is_file()
                    .then(|| parse_hex_digest(hex))
                    .flatten()
            })
            .collect();
        Ok(SyncPlan {
            snapshot_id: id,
            session,
            have,
            staged_packs: Vec::new(),
            missing: self.missing_view(&value),
            // Every object this transport's corpora carry is a pack-lane
            // object; the blob lane is empty by construction, which is why
            // the three blob methods below refuse rather than pretend.
            missing_blobs: Vec::new(),
            max_pack_payload: tfp1::MAX_PACK_PAYLOAD,
            max_packs_per_request: MAX_PACKS_PER_REQUEST,
        })
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

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        let mut value = self.load_session(session)?;
        let missing: HashMap<String, u64> = self
            .missing_view(&value)
            .into_iter()
            .map(|(digest, length)| (digest_hex(&digest), length))
            .collect();

        let mut grants = Vec::new();
        for claim in claims {
            let mut payload = 0_u64;
            for digest in &claim.objects {
                let Some(size) = missing.get(&digest_hex(digest)) else {
                    return Err(TransportError::Refused {
                        code: "bad_request".to_owned(),
                        detail: format!("{digest} is not missing"),
                    });
                };
                payload += size;
            }
            let expected = 12 + 48 * claim.objects.len() as u64 + payload;
            if claim.size_bytes != expected {
                return Err(TransportError::Refused {
                    code: "bad_request".to_owned(),
                    detail: "claimed size disagrees with members".to_owned(),
                });
            }
            let staging_key = format!("staged/{session}/{}", claim.sha256);
            grants.push(PackGrant {
                pack_sha256: claim.sha256.clone(),
                staging_key: staging_key.clone(),
                url: self.root.join(&staging_key).to_string_lossy().into_owned(),
                headers: Vec::new(),
            });
            value["packs"][&claim.sha256] = serde_json::json!({
                "staging_key": staging_key,
                "objects": claim.objects.iter().map(digest_hex).collect::<Vec<_>>(),
            });
        }
        self.store_session(session, &value);

        let staged_packs = value["packs"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(sha, pack)| StagedPack {
                sha256: sha.clone(),
                staged: self
                    .root
                    .join(pack["staging_key"].as_str().unwrap_or_default())
                    .is_file(),
            })
            .collect();
        Ok(GrantsPlan {
            grants,
            staged_packs,
            missing: self.missing_view(&value),
            missing_blobs: Vec::new(),
        })
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        _progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        if sha256_hex(pack) != grant.pack_sha256 {
            return Err(TransportError::Refused {
                code: "checksum-mismatch".to_owned(),
                detail: "pack bytes do not hash to the granted checksum".to_owned(),
            });
        }
        tfp1::decode(pack).map_err(|error| TransportError::Refused {
            code: "pack-invalid".to_owned(),
            detail: error.to_string(),
        })?;
        let path = self.root.join(&grant.staging_key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("staging directory creates");
        }
        // Atomic publish, so a kill mid-write never leaves a half pack that
        // a resumed run would count as staged.
        let temp = path.with_extension("part");
        std::fs::write(&temp, pack).map_err(|error| TransportError::Io(error.to_string()))?;
        std::fs::rename(&temp, &path).map_err(|error| TransportError::Io(error.to_string()))?;
        Ok(())
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        let value = self.load_session(session)?;
        for pack in value["packs"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(_, p)| p)
        {
            let key = pack["staging_key"].as_str().unwrap_or_default();
            let Ok(bytes) = std::fs::read(self.root.join(key)) else {
                continue;
            };
            let parsed = tfp1::decode(&bytes).expect("staged packs re-verify");
            for object in parsed.objects() {
                let path = self.object_file(&object.digest());
                if !path.is_file() {
                    let temp = path.with_extension("part");
                    std::fs::write(&temp, object.bytes()).expect("promotion writes");
                    std::fs::rename(&temp, &path).expect("promotion publishes");
                }
            }
        }
        let rows: Vec<&serde_json::Value> =
            value["closure"].as_array().into_iter().flatten().collect();
        let admitted = rows
            .iter()
            .filter(|row| {
                self.root
                    .join("objects")
                    .join(row[0].as_str().unwrap_or_default())
                    .is_file()
            })
            .count() as u64;
        if admitted < rows.len() as u64 {
            return Ok(CompleteStatus::Incomplete {
                code: "upload_incomplete".to_owned(),
                promoted: admitted,
                total: rows.len() as u64,
            });
        }
        let expected = value["expected_head"]
            .as_str()
            .and_then(SnapshotId::parse_hex);
        if self.read_head() != expected {
            return Ok(CompleteStatus::Failed {
                code: "head_conflict".to_owned(),
            });
        }
        // The manifest blob becomes the snapshot-id object strictly before
        // the head moves, exactly as the landed wire orders it.
        let manifest = std::fs::read(self.manifest_file(session))
            .map_err(|error| TransportError::Io(error.to_string()))?;
        let id = SnapshotId::of(&manifest);
        let path = self.object_file(&manifest_object_digest(&id));
        if !path.is_file() {
            std::fs::write(&path, &manifest).expect("manifest promotes");
        }
        std::fs::write(self.root.join("head"), id.to_string()).expect("head advances");
        Ok(CompleteStatus::Promoted)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        Ok(self.read_head())
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        Ok(digests
            .iter()
            .filter_map(|digest| {
                let path = self.object_file(digest);
                let length = std::fs::metadata(&path).ok()?.len();
                Some(DownloadGrant {
                    digest: *digest,
                    length,
                    url: path.to_string_lossy().into_owned(),
                })
            })
            .collect())
    }

    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        _progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError> {
        let bytes =
            std::fs::read(&grant.url).map_err(|error| TransportError::Io(error.to_string()))?;
        sink.write_all(&bytes)
            .map_err(|error| TransportError::Io(error.to_string()))?;
        Ok(bytes.len() as u64)
    }
}
