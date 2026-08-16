//! Snapshot sync: push a sealed TFM1 snapshot to a remote grant service and
//! pull one into a local store, moving only missing objects.
//!
//! The engine is transport-abstracted. [`SyncTransport`] mirrors the LANDED
//! th#1960 wire: declare answers with the missing set (no grants), the client
//! assembles whole-object TFP1 packs and requests grants that bind each
//! pack's own envelope checksum, uploads ride those presigned grants, and
//! `complete` is driven through retryable incompleteness to a terminal
//! answer. Bytes never proxy through the control plane; downloads are
//! per-object presigned reads admitted through the local store's verifying
//! writer. The local `ObjectStore` is the pull-side resume journal; on push,
//! staging is scoped to one hub session, so within-run replans never
//! retransmit a staged pack, and across process restarts resumption is
//! promotion-level (a re-declare opens a fresh session whose staging starts
//! empty, while promoted objects report resident).

use std::collections::HashSet;
use std::io::Read;
use std::io::Write as _;
use std::time::Duration;

use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::object::ObjectDigest;
use crate::store::StoreError;
use crate::tfm1::{Entry, FileRecord, Snapshot, SnapshotId, Tfm1Error, decode};
use crate::tfp1::{MAX_PACK_OBJECTS, MAX_PACK_PAYLOAD, Tfp1Error};
use crate::workspace::{LeaseId, WorkspaceError, WorkspaceStore};

/// One bounded download-grant batch, per the wire contract.
pub const DOWNLOAD_GRANT_BATCH: usize = 256;

/// One observation that a transfer advanced.
///
/// This library reports movement; it never judges it. A consumer decides that
/// a transfer is stuck from the ABSENCE of these observations, against a
/// budget only the deployment knows — so nothing here owns a clock, a
/// deadline, or a rate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Progress {
    /// Bytes that crossed the wire since the previous observation, while the
    /// object or pack carrying them is still in flight. This is what makes a
    /// single multi-gigabyte transfer visible while it runs.
    Bytes(u64),
    /// One object is accounted for: staged on the remote, verified into the
    /// local store, or already resident on the far side. Mirrors the Python
    /// data plane's `progress(digest, size)` so both stacks report one
    /// contract, and like it, a RESIDENT object advances readiness exactly as
    /// a moved one does — a resumed transfer is alive, not stalled.
    Object { digest: ObjectDigest, length: u64 },
}

/// Where a transfer reports that it advanced.
///
/// Every call runs synchronously on the thread driving the transfer: the sync
/// engine spawns nothing, and a transport reports its own bytes from inside
/// the call that moves them. The sink is therefore neither `Send` nor `Sync`,
/// which is the compiler's own proof of that promise — callers can hand it
/// state they have not otherwise made shareable, exactly as the Python side's
/// caller-thread callback lets them.
#[derive(Clone, Copy)]
pub struct ProgressSink<'sink>(Option<&'sink dyn Fn(Progress)>);

impl std::fmt::Debug for ProgressSink<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("ProgressSink")
            .field(&if self.0.is_some() {
                "reporting"
            } else {
                "silent"
            })
            .finish()
    }
}

impl Default for ProgressSink<'_> {
    fn default() -> Self {
        Self::silent()
    }
}

impl<'sink> ProgressSink<'sink> {
    /// A transfer nobody is watching.
    #[must_use]
    pub const fn silent() -> Self {
        Self(None)
    }

    #[must_use]
    pub const fn new(sink: &'sink dyn Fn(Progress)) -> Self {
        Self(Some(sink))
    }

    /// Reports bytes that just moved. Zero-length reports are dropped: an
    /// observation must mean something happened.
    pub fn bytes(&self, moved: u64) {
        if moved > 0
            && let Some(sink) = self.0
        {
            sink(Progress::Bytes(moved));
        }
    }

    /// Reports that one object is now accounted for.
    pub fn object(&self, digest: ObjectDigest, length: u64) {
        if let Some(sink) = self.0 {
            sink(Progress::Object { digest, length });
        }
    }
}

/// Bounded transient retries per object fetch. Pull carries no options
/// struct, so this is the contract rather than a caller's choice.
const DOWNLOAD_ATTEMPTS: u32 = 4;

/// One presigned authorization to PUT one exact TFP1 staging pack. The signed
/// headers bind the client-declared pack checksum; they are replayed verbatim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackGrant {
    pub pack_sha256: String,
    pub staging_key: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// One presigned authorization to GET one exact object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadGrant {
    pub digest: ObjectDigest,
    pub length: u64,
    pub url: String,
}

/// One pack the session has granted before, with its live staged state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedPack {
    pub sha256: String,
    pub staged: bool,
}

/// The client's claim for one assembled pack: the envelope's own SHA-256,
/// its exact encoded size, and the member objects it carries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackClaim {
    pub sha256: String,
    pub size_bytes: u64,
    pub objects: Vec<ObjectDigest>,
}

/// The remote's answer to a declare: identity, session, what it already
/// holds, what this session staged, and what is still missing — never grants.
#[derive(Clone, Debug)]
pub struct SyncPlan {
    pub snapshot_id: SnapshotId,
    pub session: String,
    pub have: Vec<ObjectDigest>,
    pub staged_packs: Vec<StagedPack>,
    pub missing: Vec<(ObjectDigest, u64)>,
    pub max_pack_payload: u64,
    pub max_packs_per_request: usize,
}

/// The remote's answer to a pack-grant request (or, with no claims, a pure
/// resume probe): grants for the claimed packs plus the refreshed live view.
#[derive(Clone, Debug)]
pub struct GrantsPlan {
    pub grants: Vec<PackGrant>,
    pub staged_packs: Vec<StagedPack>,
    pub missing: Vec<(ObjectDigest, u64)>,
}

/// The terminal-or-not answer of one `complete` call. Retryability is a
/// property of the code, never of the call site.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompleteStatus {
    Promoted,
    /// Retryable by contract (`promote_incomplete`-class): call again.
    Incomplete {
        code: String,
    },
    /// Terminal: the same bytes cannot succeed (verification, head conflict).
    Failed {
        code: String,
    },
}

#[derive(Debug, Error)]
pub enum TransportError {
    /// A grant or session lease ran out; the caller replans, never fails.
    #[error("transport authorization expired: {0}")]
    Expired(String),
    /// The remote refused with a stable code; retrying identical input is
    /// pointless.
    #[error("transport refused ({code}): {detail}")]
    Refused { code: String, detail: String },
    /// A transient carrier failure worth bounded retries.
    #[error("transport I/O failed: {0}")]
    Io(String),
}

/// The transport half of the sync contract. Implementations carry
/// authentication and spelling; the engine carries every invariant.
pub trait SyncTransport {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError>;
    /// Requests grants for `claims`; with no claims this is a pure resume
    /// probe returning the session's live staged/missing view.
    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError>;
    /// Uploads one pack, reporting bytes to `progress` as they leave — a pack
    /// is the largest thing this engine moves in one call, so a transport that
    /// reports only on return makes its whole upload invisible.
    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError>;
    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError>;
    fn head(&self) -> Result<Option<SnapshotId>, TransportError>;
    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError>;
    /// Fetches one object, reporting bytes to `progress` as they arrive.
    fn download(
        &self,
        grant: &DownloadGrant,
        progress: ProgressSink<'_>,
    ) -> Result<Vec<u8>, TransportError>;
}

#[derive(Debug, Error)]
pub enum SyncError {
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Pack(#[from] Tfp1Error),
    #[error(transparent)]
    Manifest(#[from] Tfm1Error),
    #[error("remote computed snapshot id {remote}, local bytes are {local}")]
    IdentityMismatch {
        local: SnapshotId,
        remote: SnapshotId,
    },
    #[error("remote head promotion refused terminally ({code})")]
    HeadRefused { code: String },
    #[error("remote bytes for snapshot {0} do not hash to it")]
    RemoteManifestCorrupt(SnapshotId),
    #[error("the remote has no snapshot head")]
    NoRemoteHead,
    #[error("local object {digest} is {actual} bytes, manifest records {expected}")]
    LocalObjectLength {
        digest: ObjectDigest,
        expected: u64,
        actual: u64,
    },
    #[error("the remote omitted a grant for requested object {0}")]
    GrantOmitted(ObjectDigest),
    #[error("the remote's missing set departs from canonical manifest order at {digest}")]
    MissingNotCanonical { digest: ObjectDigest },
    #[error(
        "the remote declares object {digest} as {actual} bytes, the manifest records {expected}"
    )]
    MissingLength {
        digest: ObjectDigest,
        expected: u64,
        actual: u64,
    },
    #[error("the remote neither granted nor reported staged the claimed pack {0}")]
    PackGrantOmitted(String),
    #[error("grant replans exhausted after {0} attempts")]
    ReplansExhausted(u32),
    #[error("completion still not terminal after {attempts} calls (last: {last})")]
    CompletionExhausted { attempts: u32, last: String },
}

#[derive(Clone, Copy, Debug)]
pub struct PushOptions {
    /// Bounded replans across expired grants and staged-state refreshes.
    pub max_replans: u32,
    /// Bounded transient retries per pack upload.
    pub max_upload_attempts: u32,
    /// Bounded `complete` calls through retryable incompleteness.
    pub max_complete_attempts: u32,
}

impl Default for PushOptions {
    fn default() -> Self {
        Self {
            max_replans: 16,
            max_upload_attempts: 3,
            max_complete_attempts: 600,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PushReport {
    pub uploaded_objects: u64,
    pub uploaded_bytes: u64,
    pub packs: u64,
    pub skipped_remote_resident: u64,
    pub complete_attempts: u32,
    pub replans: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PullReport {
    pub fetched_objects: u64,
    pub fetched_bytes: u64,
    pub skipped_local_resident: u64,
}

/// The unique `(digest, length)` closure of one snapshot's data records, in
/// canonical manifest order.
fn data_closure(snapshot: &Snapshot) -> Vec<(ObjectDigest, u64)> {
    let mut seen = HashSet::new();
    let mut closure = Vec::new();
    for (_path, entry) in snapshot.entries() {
        if let Entry::File { records, .. } = entry {
            for record in records {
                if let FileRecord::Data { digest, length } = record
                    && seen.insert(*digest.as_bytes())
                {
                    closure.push((*digest, *length));
                }
            }
        }
    }
    closure
}

/// The missing set the remote answers with must be a faithful projection of
/// the manifest the client just declared: the same digests, with the same
/// lengths, in the same canonical order. A subsequence is expected — the
/// remote drops what it already holds — but a reorder is not.
///
/// The client refuses rather than re-sorting. Push assembles packs greedily
/// in the order it is handed, so the order decides which objects share a
/// pack, and therefore each pack's SHA-256 — the checksum a grant binds and a
/// resume must match. Silently re-sorting would repair the run in front of us
/// and leave the remote free to answer differently on the next replan, which
/// is exactly how a resume stops recognising its own staged packs. The order
/// is a wire contract, so a remote that breaks it is broken, and the honest
/// answer is to say so with the digest that proves it.
fn verify_canonical_missing(
    closure: &[(ObjectDigest, u64)],
    missing: &[(ObjectDigest, u64)],
) -> Result<(), SyncError> {
    let mut cursor = 0_usize;
    for (digest, length) in missing {
        // Only ever scans FORWARD, so a digest that is absent from the
        // manifest and one that has already been passed are the same refusal:
        // both mean this row cannot be where canonical order would put it.
        let Some(offset) = closure[cursor..]
            .iter()
            .position(|(candidate, _)| candidate == digest)
        else {
            return Err(SyncError::MissingNotCanonical { digest: *digest });
        };
        cursor += offset + 1;
        let expected = closure[cursor - 1].1;
        if *length != expected {
            return Err(SyncError::MissingLength {
                digest: *digest,
                expected,
                actual: *length,
            });
        }
    }
    Ok(())
}

/// The digest the snapshot's own canonical bytes occupy in the object
/// namespace: the manifest blob is digest-addressed by its snapshot id, so
/// pull needs no second manifest channel. The hub admits the blob from the
/// declare body at complete, strictly before the head becomes visible; push
/// never packs it.
#[must_use]
pub fn manifest_object_digest(id: &SnapshotId) -> ObjectDigest {
    ObjectDigest::from_bytes(*id.as_bytes())
}

/// Who the pending-sync lease belongs to, as recorded in the metadata store.
const PUSH_LEASE_HOLDER: &str = "sync-push";

/// Releases a push's pending-sync lease on every exit — the `?` on any of the
/// two dozen fallible steps below, and a panic just as much.
struct PendingSyncPin<'meta> {
    meta: &'meta WorkspaceStore,
    lease: LeaseId,
}

impl Drop for PendingSyncPin<'_> {
    fn drop(&mut self) {
        // A failed release leaks one lease row, which only delays collection
        // of objects the caller already has; there is no second thing to try
        // and nothing to unwind into.
        let _ = self.meta.release_lease(self.lease);
    }
}

/// Pushes one sealed local snapshot: declare, pack and upload only missing
/// objects under claim-bound grants, then drive `complete` to a terminal
/// answer. Within a run, replans re-probe the session and never retransmit a
/// staged pack; across restarts, promoted objects report resident and are
/// never retransmitted.
///
/// `progress` observes the push as it happens: the objects the remote already
/// holds land first, then each pack's bytes as they leave the wire and each of
/// its members as the pack is accepted.
pub fn push_snapshot<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    snapshot: &SnapshotId,
    expected_head: Option<&SnapshotId>,
    options: PushOptions,
    progress: ProgressSink<'_>,
) -> Result<PushReport, SyncError> {
    // The transfer reads local object bytes for as long as it runs, and until
    // it lands the snapshot row is usually the only thing rooting them. The
    // pending-sync lease pins that exact closure for the transfer's exact
    // lifetime, so a `delete_snapshot` racing this push cannot let a
    // collection pass take bytes still being streamed. Acquiring it also
    // re-verifies the stored blob against its id, and TFM1 re-encoding is
    // byte-exact, so these are the canonical bytes.
    let (lease, decoded) = meta.acquire_pending_sync_lease(snapshot, PUSH_LEASE_HOLDER)?;
    let _pin = PendingSyncPin { meta, lease };
    let tfm1_bytes = decoded.to_bytes();

    let mut report = PushReport::default();
    let plan = transport.declare(&tfm1_bytes, expected_head)?;
    if plan.snapshot_id != *snapshot {
        return Err(SyncError::IdentityMismatch {
            local: *snapshot,
            remote: plan.snapshot_id,
        });
    }
    // What the remote already held before this push moved anything. These
    // objects are completed work toward readiness, so they advance progress
    // exactly as uploaded ones do: a resumed push that finds most of its
    // closure already promoted must not read as a push doing nothing.
    report.skipped_remote_resident = plan.have.len() as u64;
    let lengths: std::collections::HashMap<[u8; 32], u64> = data_closure(&decoded)
        .into_iter()
        .map(|(digest, length)| (*digest.as_bytes(), length))
        .collect();
    for digest in &plan.have {
        if let Some(length) = lengths.get(digest.as_bytes()) {
            progress.object(*digest, *length);
        }
    }

    let session = plan.session.clone();
    let max_payload = if plan.max_pack_payload == 0 {
        MAX_PACK_PAYLOAD
    } else {
        plan.max_pack_payload.min(MAX_PACK_PAYLOAD)
    };

    // The manifest we declared is the only authority on which objects exist
    // and in what order; the remote's answer is checked against it, never
    // trusted for it.
    let closure = data_closure(&decoded);
    let mut missing = plan.missing;
    verify_canonical_missing(&closure, &missing)?;
    loop {
        if missing.is_empty() {
            break;
        }

        // Greedy whole-object pack assembly in manifest order. One pack is
        // encoded, claimed, granted and uploaded at a time, so peak transfer
        // memory is bounded by one payload plus its encoding.
        let mut refresh_early = false;
        let mut group: Vec<(ObjectDigest, u64)> = Vec::new();
        let mut group_bytes = 0_u64;
        let mut groups: Vec<Vec<(ObjectDigest, u64)>> = Vec::new();
        for (digest, length) in &missing {
            let over_payload = group_bytes + length > max_payload;
            let over_count = group.len() >= MAX_PACK_OBJECTS;
            if !group.is_empty() && (over_payload || over_count) {
                groups.push(std::mem::take(&mut group));
                group_bytes = 0;
            }
            group.push((*digest, *length));
            group_bytes += length;
        }
        if !group.is_empty() {
            groups.push(group);
        }

        'groups: for members in groups {
            let loaded = load_pack_members(meta, &members)?;
            let borrowed: Vec<(ObjectDigest, &[u8])> = loaded
                .iter()
                .map(|(digest, bytes)| (*digest, bytes.as_slice()))
                .collect();
            let encoded = crate::tfp1::encode(&borrowed)?;
            let payload: u64 = members.iter().map(|(_, length)| *length).sum();
            let claim = PackClaim {
                sha256: lowercase_hex(&Sha256::digest(&encoded)),
                size_bytes: encoded.len() as u64,
                objects: members.iter().map(|(digest, _)| *digest).collect(),
            };

            let granted = match transport.pack_grants(&session, std::slice::from_ref(&claim)) {
                Ok(granted) => granted,
                Err(TransportError::Expired(_)) => {
                    refresh_early = true;
                    break 'groups;
                }
                Err(error) => return Err(error.into()),
            };
            let grant = granted
                .grants
                .iter()
                .find(|grant| grant.pack_sha256 == claim.sha256);
            let Some(grant) = grant else {
                // No grant: acceptable only when the live view says this
                // exact pack already staged (a raced resume); anything else
                // is a broken remote.
                if granted
                    .staged_packs
                    .iter()
                    .any(|pack| pack.sha256 == claim.sha256 && pack.staged)
                {
                    continue;
                }
                return Err(SyncError::PackGrantOmitted(claim.sha256));
            };

            let mut attempt = 0;
            loop {
                match transport.upload_pack(grant, &encoded, progress) {
                    Ok(()) => break,
                    Err(TransportError::Io(detail)) => {
                        attempt += 1;
                        if attempt >= options.max_upload_attempts {
                            return Err(TransportError::Io(detail).into());
                        }
                        // A carrier that just reset the connection is rarely
                        // ready again the same millisecond. Retrying without a
                        // pause spends the whole budget inside one outage and
                        // turns a transient reset into a terminal push.
                        back_off(attempt);
                    }
                    Err(TransportError::Expired(_)) => {
                        refresh_early = true;
                        break 'groups;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            report.uploaded_objects += members.len() as u64;
            report.uploaded_bytes += payload;
            report.packs += 1;
            // Reported as this pack lands, not once every pack has: a push
            // that only reports at the end is indistinguishable from a dead
            // one for its entire duration, which is precisely how a healthy
            // transfer gets condemned.
            for (digest, length) in &members {
                progress.object(*digest, *length);
            }
        }
        let _ = refresh_early;

        // A pass completed or a grant expired: re-probe the session's live
        // staged view and re-derive what is still missing.
        report.replans += 1;
        if report.replans > options.max_replans {
            return Err(SyncError::ReplansExhausted(report.replans));
        }
        missing = transport.pack_grants(&session, &[])?.missing;
        verify_canonical_missing(&closure, &missing)?;
    }

    loop {
        report.complete_attempts += 1;
        match transport.complete(&session)? {
            CompleteStatus::Promoted => return Ok(report),
            CompleteStatus::Incomplete { code } => {
                if report.complete_attempts >= options.max_complete_attempts {
                    return Err(SyncError::CompletionExhausted {
                        attempts: report.complete_attempts,
                        last: code,
                    });
                }
            }
            CompleteStatus::Failed { code } => return Err(SyncError::HeadRefused { code }),
        }
    }
}

/// Bounded exponential backoff before retrying a transient carrier failure.
/// Capped so a long outage still fails in bounded time rather than hanging.
fn back_off(attempt: u32) {
    const BASE: Duration = Duration::from_millis(250);
    const CEILING: Duration = Duration::from_secs(8);
    let delay = BASE.saturating_mul(1_u32 << attempt.min(6)).min(CEILING);
    std::thread::sleep(delay);
}

/// Fetches one object, retrying only genuinely transient carrier failures. A
/// refusal or an expired grant is never retried here: the first is terminal
/// and the second belongs to the caller's replan.
fn download_with_retry<T: SyncTransport>(
    transport: &T,
    grant: &DownloadGrant,
    attempts: u32,
    progress: ProgressSink<'_>,
) -> Result<Vec<u8>, TransportError> {
    let mut attempt = 0;
    loop {
        match transport.download(grant, progress) {
            Ok(bytes) => return Ok(bytes),
            Err(TransportError::Io(detail)) => {
                attempt += 1;
                if attempt >= attempts {
                    return Err(TransportError::Io(detail));
                }
                back_off(attempt);
            }
            Err(error) => return Err(error),
        }
    }
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Reads one pack's member objects out of the verified local store. A length
/// disagreement between manifest and resident bytes refuses before encoding;
/// `tfp1::encode` then independently re-hashes every member.
fn load_pack_members(
    meta: &WorkspaceStore,
    members: &[(ObjectDigest, u64)],
) -> Result<Vec<(ObjectDigest, Vec<u8>)>, SyncError> {
    let mut loaded = Vec::with_capacity(members.len());
    for (digest, expected) in members {
        let mut file = meta.store().open_object(digest)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|error| SyncError::Store(StoreError::Io(error)))?;
        if bytes.len() as u64 != *expected {
            return Err(SyncError::LocalObjectLength {
                digest: *digest,
                expected: *expected,
                actual: bytes.len() as u64,
            });
        }
        loaded.push((*digest, bytes));
    }
    Ok(loaded)
}

/// Pulls the remote head snapshot: fetch its manifest object, admit every
/// locally missing data object through the verifying writer, then adopt the
/// snapshot so it is locally sealed and readable. Resident objects are the
/// resume journal and are never re-fetched.
pub fn pull_head<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    progress: ProgressSink<'_>,
) -> Result<(SnapshotId, PullReport), SyncError> {
    let head = transport.head()?.ok_or(SyncError::NoRemoteHead)?;
    let report = pull_snapshot(meta, transport, &head, progress)?;
    Ok((head, report))
}

/// Pulls one exact remote snapshot id into the local store.
///
/// `progress` observes the pull as it happens: bytes as each object streams
/// in, then the object itself as the verifying writer accepts it, and every
/// already-resident object as the resume journal is read.
pub fn pull_snapshot<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    id: &SnapshotId,
    progress: ProgressSink<'_>,
) -> Result<PullReport, SyncError> {
    let tfm1_bytes = match meta.load_snapshot(id) {
        Ok(local) => local.to_bytes(),
        Err(WorkspaceError::UnknownSnapshot(_)) => {
            let manifest = manifest_object_digest(id);
            let grants = transport.download_grants(std::slice::from_ref(&manifest))?;
            let grant = grants
                .iter()
                .find(|grant| grant.digest == manifest)
                .ok_or(SyncError::GrantOmitted(manifest))?;
            let bytes = transport.download(grant, progress)?;
            if SnapshotId::of(&bytes) != *id {
                return Err(SyncError::RemoteManifestCorrupt(*id));
            }
            bytes
        }
        Err(error) => return Err(error.into()),
    };
    let snapshot = decode(&tfm1_bytes)?;
    let closure = data_closure(&snapshot);

    let mut report = PullReport::default();
    // A resident object is completed work toward readiness, and reading the
    // journal is not free either — so it advances progress as it is found,
    // rather than leaving a mostly-complete resume looking like a pull that
    // has not moved at all.
    let mut missing: Vec<(ObjectDigest, u64)> = Vec::new();
    for (digest, length) in &closure {
        if meta.store().exists(digest) {
            report.skipped_local_resident += 1;
            progress.object(*digest, *length);
        } else {
            missing.push((*digest, *length));
        }
    }

    for batch in missing.chunks(DOWNLOAD_GRANT_BATCH) {
        let digests: Vec<ObjectDigest> = batch.iter().map(|(digest, _)| *digest).collect();
        let grants = transport.download_grants(&digests)?;
        for (digest, length) in batch {
            let grant = grants
                .iter()
                .find(|grant| grant.digest == *digest)
                .ok_or(SyncError::GrantOmitted(*digest))?;
            // A single transient reset must not abandon a whole pull: a
            // multi-gigabyte fetch crosses too many packets for one-shot
            // transfer to be a defensible contract.
            let bytes = download_with_retry(transport, grant, DOWNLOAD_ATTEMPTS, progress)?;
            let mut writer = meta.store().writer()?;
            writer
                .write_all(&bytes)
                .map_err(|error| SyncError::Store(StoreError::Io(error)))?;
            // The admission boundary is the trust boundary: remote bytes are
            // hashed while written and refused on any length or digest lie.
            writer.finish_expecting(*digest, *length)?;
            report.fetched_objects += 1;
            report.fetched_bytes += length;
            // Reported the moment this object is durable, so the next fetch
            // starts against a counter that has already moved.
            progress.object(*digest, *length);
        }
    }

    meta.adopt_snapshot(&tfm1_bytes)?;
    Ok(report)
}

/// The th#1960 HTTP adapter: the exact landed tensorhub wire.
///
/// Route shapes, field spellings and failure envelopes follow tensorhub PR
/// #1265 (`internal/api/snapshot_sync_th1960.go`) verbatim. Bearer auth rides
/// every control route; presigned PUT/GET grants carry their own authority
/// and are replayed with the granted headers, exactly as issued.
pub mod http {
    use std::io::Read as _;
    use std::path::PathBuf;
    use std::time::Duration;

    use base64::Engine as _;
    use serde::Deserialize;

    use super::{
        CompleteStatus, DownloadGrant, GrantsPlan, ObjectDigest, PackClaim, PackGrant,
        ProgressSink, SnapshotId, StagedPack, SyncPlan, SyncTransport, TransportError,
    };

    const CONTROL_TIMEOUT: Duration = Duration::from_secs(60);
    const TRANSFER_TIMEOUT: Duration = Duration::from_secs(600);

    /// The read buffer one download fills before copying on. It bounds a
    /// single `read` syscall's appetite — it is not a rate, a deadline, or a
    /// judgement about how fast a healthy transfer ought to be.
    const DOWNLOAD_BLOCK: usize = 1 << 20;

    /// Feeds a pack to the carrier while reporting the bytes it hands over, so
    /// one large PUT is visible while it is in flight instead of only when it
    /// returns.
    struct ReportingBody<'body> {
        rest: &'body [u8],
        progress: ProgressSink<'body>,
    }

    impl std::io::Read for ReportingBody<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let taken = out.len().min(self.rest.len());
            out[..taken].copy_from_slice(&self.rest[..taken]);
            self.rest = &self.rest[taken..];
            self.progress.bytes(taken as u64);
            Ok(taken)
        }
    }

    /// Where the bearer token comes from. A file is re-read on every call so
    /// an external refresher can rotate it without restarting the process.
    #[derive(Clone, Debug, Default)]
    pub enum TokenSource {
        #[default]
        None,
        Static(String),
        File(PathBuf),
    }

    /// The tensorhub snapshot-sync client for one `org/repo`.
    #[derive(Debug)]
    pub struct HttpTransport {
        base_url: String,
        org: String,
        repo: String,
        token: TokenSource,
        agent: ureq::Agent,
    }

    impl HttpTransport {
        #[must_use]
        pub fn new(
            base_url: impl Into<String>,
            org: impl Into<String>,
            repo: impl Into<String>,
        ) -> Self {
            Self {
                base_url: base_url.into().trim_end_matches('/').to_owned(),
                org: org.into(),
                repo: repo.into(),
                token: TokenSource::None,
                agent: ureq::AgentBuilder::new()
                    .timeout_connect(Duration::from_secs(15))
                    .build(),
            }
        }

        #[must_use]
        pub fn with_token(mut self, token: TokenSource) -> Self {
            self.token = token;
            self
        }

        #[must_use]
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        fn route(&self, tail: &str) -> String {
            format!(
                "{}/api/v1/repos/{}/{}/snapshot-sync{tail}",
                self.base_url, self.org, self.repo
            )
        }

        fn bearer(&self) -> Result<Option<String>, TransportError> {
            match &self.token {
                TokenSource::None => Ok(None),
                TokenSource::Static(token) => Ok(Some(token.clone())),
                TokenSource::File(path) => std::fs::read_to_string(path)
                    .map(|raw| Some(raw.trim().to_owned()))
                    .map_err(|error| {
                        TransportError::Io(format!("read token file {}: {error}", path.display()))
                    }),
            }
        }

        /// One control-plane exchange. Every HTTP status is returned with its
        /// parsed body so each route maps its own envelope; only carrier
        /// failures become `Io`.
        fn exchange(
            &self,
            method: &str,
            url: &str,
            body: Option<&serde_json::Value>,
        ) -> Result<(u16, serde_json::Value), TransportError> {
            let mut request = self.agent.request(method, url).timeout(CONTROL_TIMEOUT);
            if let Some(token) = self.bearer()? {
                request = request.set("authorization", &format!("Bearer {token}"));
            }
            let outcome = match body {
                Some(json) => request.send_json(json.clone()),
                None => request.call(),
            };
            let (status, text) = match outcome {
                Ok(response) => {
                    let status = response.status();
                    let text = response
                        .into_string()
                        .map_err(|error| TransportError::Io(error.to_string()))?;
                    (status, text)
                }
                Err(ureq::Error::Status(status, response)) => {
                    let text = response.into_string().unwrap_or_default();
                    (status, text)
                }
                Err(ureq::Error::Transport(transport)) => {
                    return Err(TransportError::Io(transport.to_string()));
                }
            };
            let value = if text.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_str(&text).map_err(|error| {
                    TransportError::Io(format!("undecodable response ({status}): {error}"))
                })?
            };
            Ok((status, value))
        }
    }

    #[derive(Deserialize)]
    struct WireError {
        code: String,
        #[serde(default)]
        message: String,
    }

    #[derive(Deserialize)]
    struct WireErrorEnvelope {
        error: WireError,
    }

    /// Maps a non-2xx control answer that is NOT a failure envelope. Session
    /// and grant leases surface as `Expired` so the engine replans.
    fn refuse(status: u16, value: &serde_json::Value) -> TransportError {
        if let Ok(envelope) = serde_json::from_value::<WireErrorEnvelope>(value.clone()) {
            if envelope.error.code.contains("expired") {
                return TransportError::Expired(envelope.error.code);
            }
            return TransportError::Refused {
                code: envelope.error.code,
                detail: envelope.error.message,
            };
        }
        TransportError::Refused {
            code: format!("http-{status}"),
            detail: value.to_string(),
        }
    }

    fn parse_ref(raw: &str) -> Result<ObjectDigest, TransportError> {
        let hex = raw
            .strip_prefix("sha256:")
            .ok_or_else(|| TransportError::Io(format!("untagged ref {raw:?}")))?;
        let id = SnapshotId::parse_hex(hex)
            .ok_or_else(|| TransportError::Io(format!("undecodable ref {raw:?}")))?;
        Ok(ObjectDigest::from_bytes(*id.as_bytes()))
    }

    fn tagged(digest: &ObjectDigest) -> String {
        digest.to_string()
    }

    #[derive(Deserialize)]
    struct WireMissing {
        digest: String,
        size_bytes: u64,
    }

    #[derive(Deserialize)]
    struct WirePackStatus {
        sha256: String,
        staged: bool,
    }

    #[derive(Deserialize)]
    struct WireDeclare {
        snapshot_id: String,
        session_id: String,
        have: Vec<String>,
        #[serde(default)]
        staged_packs: Vec<WirePackStatus>,
        #[serde(default)]
        missing: Vec<WireMissing>,
        #[serde(default)]
        max_pack_payload: u64,
        #[serde(default)]
        max_packs_per_request: usize,
    }

    #[derive(Deserialize)]
    struct WireGrant {
        pack_sha256: String,
        staging_key: String,
        put_url: String,
        #[serde(default)]
        headers: std::collections::BTreeMap<String, String>,
    }

    #[derive(Deserialize)]
    struct WireGrants {
        #[serde(default)]
        grants: Vec<WireGrant>,
        #[serde(default)]
        staged_packs: Vec<WirePackStatus>,
        #[serde(default)]
        missing: Vec<WireMissing>,
    }

    #[derive(Deserialize)]
    struct WireFailure {
        code: String,
        retryable: bool,
    }

    #[derive(Deserialize)]
    struct WireComplete {
        #[serde(default)]
        stage: String,
        #[serde(default)]
        failure: Option<WireFailure>,
    }

    #[derive(Deserialize)]
    struct WireHead {
        snapshot_id: String,
    }

    #[derive(Deserialize)]
    struct WireDownloadGrant {
        digest: String,
        size_bytes: u64,
        get_url: String,
    }

    #[derive(Deserialize)]
    struct WireDownloadGrants {
        #[serde(default)]
        grants: Vec<WireDownloadGrant>,
    }

    fn missing_pairs(rows: Vec<WireMissing>) -> Result<Vec<(ObjectDigest, u64)>, TransportError> {
        rows.into_iter()
            .map(|row| Ok((parse_ref(&row.digest)?, row.size_bytes)))
            .collect()
    }

    fn staged_rows(rows: Vec<WirePackStatus>) -> Vec<StagedPack> {
        rows.into_iter()
            .map(|row| StagedPack {
                sha256: row.sha256,
                staged: row.staged,
            })
            .collect()
    }

    impl SyncTransport for HttpTransport {
        fn declare(
            &self,
            tfm1_bytes: &[u8],
            expected_head: Option<&SnapshotId>,
        ) -> Result<SyncPlan, TransportError> {
            let body = serde_json::json!({
                "tfm1_base64": base64::engine::general_purpose::STANDARD.encode(tfm1_bytes),
                "expected_head": expected_head.map(SnapshotId::to_string).unwrap_or_default(),
            });
            let (status, value) = self.exchange("POST", &self.route(""), Some(&body))?;
            if status != 201 {
                return Err(refuse(status, &value));
            }
            let wire: WireDeclare = serde_json::from_value(value)
                .map_err(|error| TransportError::Io(format!("declare response: {error}")))?;
            let snapshot_id = SnapshotId::parse_hex(&wire.snapshot_id).ok_or_else(|| {
                TransportError::Io(format!("undecodable snapshot id {:?}", wire.snapshot_id))
            })?;
            Ok(SyncPlan {
                snapshot_id,
                session: wire.session_id,
                have: wire
                    .have
                    .iter()
                    .map(|raw| parse_ref(raw))
                    .collect::<Result<_, _>>()?,
                staged_packs: staged_rows(wire.staged_packs),
                missing: missing_pairs(wire.missing)?,
                max_pack_payload: wire.max_pack_payload,
                max_packs_per_request: wire.max_packs_per_request,
            })
        }

        fn pack_grants(
            &self,
            session: &str,
            claims: &[PackClaim],
        ) -> Result<GrantsPlan, TransportError> {
            let body = serde_json::json!({
                "packs": claims
                    .iter()
                    .map(|claim| {
                        serde_json::json!({
                            "sha256": claim.sha256,
                            "size_bytes": claim.size_bytes,
                            "objects": claim
                                .objects
                                .iter()
                                .map(tagged)
                                .collect::<Vec<_>>(),
                        })
                    })
                    .collect::<Vec<_>>(),
            });
            let url = self.route(&format!("/{session}/pack-grants"));
            let (status, value) = self.exchange("POST", &url, Some(&body))?;
            if status != 200 {
                return Err(refuse(status, &value));
            }
            let wire: WireGrants = serde_json::from_value(value)
                .map_err(|error| TransportError::Io(format!("pack-grants response: {error}")))?;
            Ok(GrantsPlan {
                grants: wire
                    .grants
                    .into_iter()
                    .map(|grant| PackGrant {
                        pack_sha256: grant.pack_sha256,
                        staging_key: grant.staging_key,
                        url: grant.put_url,
                        headers: grant.headers.into_iter().collect(),
                    })
                    .collect(),
                staged_packs: staged_rows(wire.staged_packs),
                missing: missing_pairs(wire.missing)?,
            })
        }

        fn upload_pack(
            &self,
            grant: &PackGrant,
            pack: &[u8],
            progress: ProgressSink<'_>,
        ) -> Result<(), TransportError> {
            // Presigned: the grant's own headers are the whole authority and
            // are replayed verbatim — the pack checksum lives inside the
            // signature, so changed bytes refuse at the store.
            let mut request = self.agent.put(&grant.url).timeout(TRANSFER_TIMEOUT);
            for (name, value) in &grant.headers {
                request = request.set(name, value);
            }
            // The body is streamed so its bytes can be reported as they leave,
            // and the length is stated explicitly because a reader body with
            // no `content-length` makes ureq switch to chunked framing — which
            // the presigned signature does not cover. With it set, the wire
            // bytes are identical to handing over the whole slice at once.
            if !grant
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            {
                request = request.set("content-length", &pack.len().to_string());
            }
            let body = ReportingBody {
                rest: pack,
                progress,
            };
            match request.send(body) {
                Ok(_) => Ok(()),
                Err(ureq::Error::Status(403, _)) => Err(TransportError::Expired(
                    "upload grant refused (403)".to_owned(),
                )),
                Err(ureq::Error::Status(status, response)) => Err(TransportError::Refused {
                    code: format!("http-{status}"),
                    detail: response.into_string().unwrap_or_default(),
                }),
                Err(ureq::Error::Transport(transport)) => {
                    Err(TransportError::Io(transport.to_string()))
                }
            }
        }

        fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
            let url = self.route(&format!("/{session}/complete"));
            let body = serde_json::json!({});
            let (status, value) = self.exchange("POST", &url, Some(&body))?;
            if let Ok(wire) = serde_json::from_value::<WireComplete>(value.clone()) {
                if let Some(failure) = wire.failure {
                    return Ok(if failure.retryable {
                        CompleteStatus::Incomplete { code: failure.code }
                    } else {
                        CompleteStatus::Failed { code: failure.code }
                    });
                }
                if status == 200 && wire.stage == "promoted" {
                    return Ok(CompleteStatus::Promoted);
                }
            }
            Err(refuse(status, &value))
        }

        fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
            let (status, value) = self.exchange("GET", &self.route("/head"), None)?;
            if status != 200 {
                return Err(refuse(status, &value));
            }
            let wire: WireHead = serde_json::from_value(value)
                .map_err(|error| TransportError::Io(format!("head response: {error}")))?;
            if wire.snapshot_id.is_empty() {
                return Ok(None);
            }
            SnapshotId::parse_hex(&wire.snapshot_id)
                .map(Some)
                .ok_or_else(|| {
                    TransportError::Io(format!("undecodable head {:?}", wire.snapshot_id))
                })
        }

        fn download_grants(
            &self,
            digests: &[ObjectDigest],
        ) -> Result<Vec<DownloadGrant>, TransportError> {
            let body = serde_json::json!({
                "digests": digests.iter().map(tagged).collect::<Vec<_>>(),
            });
            let url = self.route("/download-grants");
            let (status, value) = self.exchange("POST", &url, Some(&body))?;
            if status != 200 {
                return Err(refuse(status, &value));
            }
            let wire: WireDownloadGrants = serde_json::from_value(value).map_err(|error| {
                TransportError::Io(format!("download-grants response: {error}"))
            })?;
            wire.grants
                .into_iter()
                .map(|grant| {
                    Ok(DownloadGrant {
                        digest: parse_ref(&grant.digest)?,
                        length: grant.size_bytes,
                        url: grant.get_url,
                    })
                })
                .collect()
        }

        fn download(
            &self,
            grant: &DownloadGrant,
            progress: ProgressSink<'_>,
        ) -> Result<Vec<u8>, TransportError> {
            let response = match self.agent.get(&grant.url).timeout(TRANSFER_TIMEOUT).call() {
                Ok(response) => response,
                Err(ureq::Error::Status(403, _)) => {
                    return Err(TransportError::Expired(
                        "download grant refused (403)".to_owned(),
                    ));
                }
                Err(ureq::Error::Status(status, response)) => {
                    return Err(TransportError::Refused {
                        code: format!("http-{status}"),
                        detail: response.into_string().unwrap_or_default(),
                    });
                }
                Err(ureq::Error::Transport(transport)) => {
                    return Err(TransportError::Io(transport.to_string()));
                }
            };
            // Read block by block rather than in one gulp, so a slow
            // multi-megabyte object still reports movement while it arrives.
            let mut bytes = Vec::new();
            let mut reader = response.into_reader().take(grant.length + 1);
            let mut block = vec![0_u8; DOWNLOAD_BLOCK];
            loop {
                let read = reader
                    .read(&mut block)
                    .map_err(|error| TransportError::Io(error.to_string()))?;
                if read == 0 {
                    break;
                }
                bytes.extend_from_slice(&block[..read]);
                progress.bytes(read as u64);
            }
            if bytes.len() as u64 != grant.length {
                return Err(TransportError::Io(format!(
                    "download for {} returned {} bytes, grant says {}",
                    grant.digest,
                    bytes.len(),
                    grant.length
                )));
            }
            Ok(bytes)
        }
    }
}
