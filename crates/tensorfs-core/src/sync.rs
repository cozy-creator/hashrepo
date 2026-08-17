//! Snapshot sync: push a sealed TFM1 snapshot to a remote grant service and
//! pull one into a local store, moving only missing objects.
//!
//! The engine is transport-abstracted. [`SyncTransport`] mirrors the LANDED
//! th#1960 wire: declare answers with the missing set (no grants), the client
//! assembles whole-object TFP1 packs and requests grants that bind each
//! pack's own envelope checksum, uploads ride those presigned grants, and
//! `complete` is driven through retryable incompleteness to a terminal
//! answer. An object too large for a pack rides the th#2064 **blob lane**
//! instead: the remote opens one multipart upload per blob and presigns its
//! parts, the client PUTs parts and reports their etags, and the remote
//! stream-hashes the assembled object once at `complete`. Parts are transport
//! and never enter identity. Bytes never proxy through the control plane;
//! downloads are per-object presigned reads streamed straight into the local
//! store's verifying writer. The local `ObjectStore` is the pull-side resume
//! journal; on push, remote staging is CONTENT-ADDRESSED (th#2077), so resume
//! is staging-level in both directions: within a run a replan never
//! retransmits a staged pack, and across process restarts a fresh session
//! finds the previous one's staged packs by their envelope checksum and its
//! landed blob parts by the object digest, uploading only what is genuinely
//! still owed.
//!
//! NOTHING IN THIS MODULE OWNS A CLOCK. Every budget here counts attempts that
//! achieved nothing — replans that moved no object, `complete` calls that
//! admitted none — so healthy work of any duration is never condemned for
//! taking long. The one deadline is on establishing a connection, where there
//! is no peer yet to be making progress, and the one silence bound is on a
//! transfer socket, where bytes are the progress signal being measured. See
//! [`PushOptions`] and [`http::HttpTransport`].

use std::collections::HashSet;
use std::io::Read;
use std::time::Duration;

use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::object::ObjectDigest;
use crate::planner::ByteSource as _;
use crate::store::StoreError;
use crate::tfm1::{Entry, FileRecord, Snapshot, SnapshotId, Tfm1Error, decode};
use crate::tfp1::{MAX_PACK_OBJECTS, MAX_PACK_PAYLOAD, Tfp1Error};
use crate::workspace::{LeaseId, WorkspaceError, WorkspaceStore};
use crate::workspace_source::RecordsSource;

/// One bounded download-grant batch, per the wire contract.
pub const DOWNLOAD_GRANT_BATCH: usize = 256;

/// How many blobs one `blob-grants` call may ask for, per the landed hub wire
/// (th#2064). Each grant opens a real multipart upload, so the bound is the
/// remote's, not a preference of ours.
pub const BLOB_GRANT_BATCH: usize = 16;

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

/// One presigned authorization to PUT one exact part of one blob's multipart
/// upload. `size_bytes` is the remote's, not a choice: the parts of one upload
/// must be uniform apart from the last.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobPart {
    pub part_number: u32,
    pub size_bytes: u64,
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// One blob's live multipart upload: where it stages, which parts it wants,
/// and which of them the remote already holds.
///
/// Re-asking for a grant adopts the SAME `upload_id` and refreshes
/// `uploaded_parts`, which is what makes resume free — a second upload id
/// would orphan the first one's parts into billed state no listing shows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobGrant {
    pub digest: ObjectDigest,
    pub length: u64,
    pub staging_key: String,
    pub upload_id: String,
    pub part_size: u64,
    pub parts: Vec<BlobPart>,
    pub uploaded_parts: Vec<u32>,
}

/// One part the client has landed, as the store named it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobPartReport {
    pub part_number: u32,
    pub etag: String,
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
    /// The PACK lane: every missing object the pack payload bound can carry.
    pub missing: Vec<(ObjectDigest, u64)>,
    /// The BLOB lane: every missing object above that bound. Disjoint from
    /// `missing` by construction, and the two cannot be confused — a grant
    /// naming the wrong lane is refused at both ends.
    pub missing_blobs: Vec<(ObjectDigest, u64)>,
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
    pub missing_blobs: Vec<(ObjectDigest, u64)>,
}

/// The terminal-or-not answer of one `complete` call. Retryability is a
/// property of the code, never of the call site.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompleteStatus {
    Promoted,
    /// Retryable by contract (`promote_incomplete`-class): call again.
    ///
    /// `promoted` and `total` are the remote's own count of the closure it has
    /// admitted. They are the PROGRESS AXIS of completion: a promotion that
    /// takes many calls is healthy exactly as long as this number moves, and
    /// the client's budget is spent on calls that changed nothing rather than
    /// on calls that merely took a while.
    Incomplete {
        code: String,
        promoted: u64,
        total: u64,
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
    /// Opens or re-opens the multipart uploads for up to [`BLOB_GRANT_BATCH`]
    /// blobs. Re-asking for a digest already in flight adopts its live upload
    /// and reports the parts it already holds — including an upload a DIFFERENT
    /// session opened, since the remote stages a blob on its own digest
    /// (th#2077). That is what makes a killed push resume: the parts that
    /// landed stay landed, and only the rest is sent again.
    fn blob_grants(
        &self,
        session: &str,
        digests: &[ObjectDigest],
    ) -> Result<Vec<BlobGrant>, TransportError>;
    /// Uploads one part and returns the etag the store named it with,
    /// reporting bytes to `progress` as they leave.
    fn upload_blob_part(
        &self,
        part: &BlobPart,
        bytes: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<String, TransportError>;
    /// Reports the etags of one blob's landed parts to the session.
    fn report_blob_parts(
        &self,
        session: &str,
        digest: &ObjectDigest,
        parts: &[BlobPartReport],
    ) -> Result<(), TransportError>;
    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError>;
    fn head(&self) -> Result<Option<SnapshotId>, TransportError>;
    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError>;
    /// Streams one object into `sink`, reporting bytes to `progress` as they
    /// arrive, and returns how many it wrote.
    ///
    /// A sink and not a `Vec`: the blob lane admits objects of any size, and
    /// buffering a multi-gigabyte dataset video in RAM to hand it to a writer
    /// that streams anyway is a cliff, not a simplification. The caller's sink
    /// is the store's verifying writer, so bytes are hashed as they land and
    /// never exist whole anywhere.
    fn download(
        &self,
        grant: &DownloadGrant,
        sink: &mut dyn std::io::Write,
        progress: ProgressSink<'_>,
    ) -> Result<u64, TransportError>;
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
    #[error("{rounds} consecutive replans moved nothing, with {owed} objects still owed")]
    ReplansStalled { rounds: u32, owed: usize },
    #[error(
        "completion stopped advancing: {stalls} consecutive calls admitted nothing beyond \
         {promoted} objects (last: {last})"
    )]
    CompletionStalled {
        stalls: u32,
        promoted: u64,
        last: String,
    },
    #[error("the remote omitted a blob grant for requested object {0}")]
    BlobGrantOmitted(ObjectDigest),
    #[error(
        "the remote put object {digest} ({length} bytes) in the {lane} lane, but the \
         {limit}-byte pack payload bound puts it in the other one"
    )]
    BlobLaneMismatch {
        digest: ObjectDigest,
        length: u64,
        limit: u64,
        lane: &'static str,
    },
    #[error(
        "the remote granted {granted} bytes of parts for object {digest}, which is \
         {length} bytes"
    )]
    BlobPartCoverage {
        digest: ObjectDigest,
        length: u64,
        granted: u64,
    },
    #[error("blob part uploads exhausted after {attempts} attempts for object {digest}")]
    BlobPartAttemptsExhausted { digest: ObjectDigest, attempts: u32 },
    #[error("the remote still lists object {0} as missing after its parts were reported")]
    BlobNotAccepted(ObjectDigest),
}

/// Every bound here counts ATTEMPTS THAT ACHIEVED NOTHING. None of them is a
/// clock, and none of them is a cap on how long healthy work may take: a push
/// that keeps staging objects, landing parts or admitting them may run as long
/// as it needs to. What is bounded is standing still, which is the only thing
/// a caller can honestly call stuck without knowing the deployment.
///
/// The one shape that is a plain repetition count is
/// [`PushOptions::max_upload_attempts`], and it is one because a single PUT
/// has no partial success to measure — see its own note.
#[derive(Clone, Copy, Debug)]
pub struct PushOptions {
    /// Replan rounds that moved nothing. A round that shrinks the missing set,
    /// or that lands a blob part, is progress and is free; a round that
    /// re-probes and finds the same work outstanding is not, and enough of
    /// those in a row is a remote that is not accepting the transfer.
    pub max_stalled_replans: u32,
    /// Attempts at ONE envelope — a whole pack, or one blob part.
    ///
    /// This is a repetition count rather than a progress budget on purpose: a
    /// PUT is all-or-nothing, so an attempt that transferred half the bytes
    /// and reset achieved literally nothing that the next attempt can build
    /// on, and "retry while bytes still move" would loop forever on a carrier
    /// that always dies at the halfway mark. Progress on a BLOB is measured
    /// where it is real — parts that landed and are reported back by the next
    /// grant — and that is `max_stalled_replans`' job.
    ///
    /// Eight rather than three: the 2026-08-16 acceptance exhausted three on
    /// pack PUTs over a residential uplink and lost the push, having measured
    /// ~47 MB of part-level retry as ordinary behaviour on the same link.
    pub max_upload_attempts: u32,
    /// `complete` calls that admitted nothing beyond what the previous one
    /// had. The remote reports `promoted`; while it rises, completion polls on
    /// without spending budget, however many calls that takes.
    pub max_completion_stalls: u32,
}

impl Default for PushOptions {
    fn default() -> Self {
        Self {
            max_stalled_replans: 16,
            max_upload_attempts: 8,
            max_completion_stalls: 24,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PushReport {
    pub uploaded_objects: u64,
    pub uploaded_bytes: u64,
    pub packs: u64,
    /// Objects that rode the blob lane, and the parts they took. `blob_parts`
    /// counts parts actually PUT, so a resumed push that adopts landed parts
    /// reports fewer of them than a cold one.
    pub blobs: u64,
    pub blob_parts: u64,
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
        if let Entry::File { body, .. } = entry {
            for record in body.records().iter() {
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

    let closure = data_closure(&decoded);

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
    let lengths: std::collections::HashMap<[u8; 32], u64> = closure
        .iter()
        .map(|(digest, length)| (*digest.as_bytes(), *length))
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
    let mut missing = plan.missing;
    let mut missing_blobs = plan.missing_blobs;
    verify_lanes(&closure, &missing, &missing_blobs, max_payload)?;

    // The blob lane first. Each blob is one multipart upload of its own, so
    // nothing about it shares state with a pack, and doing it first means a
    // push whose only missing object is a 4 GiB dataset never builds a pack
    // at all.
    if !missing_blobs.is_empty() {
        push_blob_lane(
            meta,
            transport,
            &session,
            &missing_blobs,
            options,
            &mut report,
            progress,
        )?;
        let refreshed = transport.pack_grants(&session, &[])?;
        missing = refreshed.missing;
        missing_blobs = refreshed.missing_blobs;
        verify_lanes(&closure, &missing, &missing_blobs, max_payload)?;
        // The remote accepted the parts or it did not; a blob still in the
        // lane after its parts were reported means the report did not take,
        // and re-uploading the same bytes would loop forever.
        if let Some((digest, _)) = missing_blobs.first() {
            return Err(SyncError::BlobNotAccepted(*digest));
        }
    }

    let mut stalled_replans = 0_u32;
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
        //
        // A round that shrank the missing set did work, however many rounds it
        // takes to drain a large closure. Only rounds that leave exactly the
        // same debt behind are counted against the budget.
        report.replans += 1;
        let owed_before = missing.len();
        let refreshed = transport.pack_grants(&session, &[])?;
        missing = refreshed.missing;
        verify_lanes(&closure, &missing, &refreshed.missing_blobs, max_payload)?;
        if missing.len() < owed_before {
            stalled_replans = 0;
        } else {
            stalled_replans += 1;
            if stalled_replans >= options.max_stalled_replans {
                return Err(SyncError::ReplansStalled {
                    rounds: stalled_replans,
                    owed: missing.len(),
                });
            }
        }
    }

    // COMPLETION IS A POLL, NOT A BET.
    //
    // Every byte is already staged by the time this loop starts, so the only
    // way to lose the push here is to give up on it. Two things make that
    // impossible to do by accident:
    //
    // A carrier failure is not a verdict. A `complete` whose answer never came
    // back is not proof the promotion failed — in the 2026-08-16 acceptance it
    // demonstrably had NOT failed, and a read timeout discarded 272 MB of
    // successful upload. Re-issuing is safe because the remote's completion is
    // idempotent: a fully-staged session promotes, and an already-promoted one
    // says so (th#2077).
    //
    // And the budget is spent on STANDING STILL, not on elapsed calls. The
    // remote's own `promoted` count is the progress axis, so a promotion that
    // needs a thousand calls is healthy as long as it keeps admitting objects,
    // and one that needs twenty while admitting nothing is not.
    let mut admitted = 0_u64;
    let mut stalls = 0_u32;
    loop {
        report.complete_attempts += 1;
        match transport.complete(&session) {
            Ok(CompleteStatus::Promoted) => return Ok(report),
            Ok(CompleteStatus::Incomplete {
                code,
                promoted,
                total: _,
            }) => {
                if promoted > admitted {
                    // Objects landed since the last call; poll straight back,
                    // because the remote is working and a pause is pure
                    // latency.
                    admitted = promoted;
                    stalls = 0;
                    continue;
                }
                stalls += 1;
                if stalls >= options.max_completion_stalls {
                    return Err(SyncError::CompletionStalled {
                        stalls,
                        promoted: admitted,
                        last: code,
                    });
                }
                back_off(stalls);
            }
            Ok(CompleteStatus::Failed { code }) => return Err(SyncError::HeadRefused { code }),
            Err(TransportError::Io(detail)) => {
                stalls += 1;
                if stalls >= options.max_completion_stalls {
                    return Err(TransportError::Io(detail).into());
                }
                back_off(stalls);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Both lanes together must be a faithful projection of the manifest just
/// declared, and each object must be in the lane its size puts it in.
///
/// The lane rule is checked in BOTH directions on purpose. A pack-lane row
/// above the payload bound would be encoded into a pack that can only be
/// refused; a blob-lane row below it would open a multipart upload for an
/// object the pack lane already carries, and the remote refuses cross-lane
/// grants at its end too. Neither is a thing to work around locally: a remote
/// that partitions wrongly is broken, and the honest answer names the object
/// that proves it.
fn verify_lanes(
    closure: &[(ObjectDigest, u64)],
    missing: &[(ObjectDigest, u64)],
    missing_blobs: &[(ObjectDigest, u64)],
    max_payload: u64,
) -> Result<(), SyncError> {
    verify_canonical_missing(closure, missing)?;
    verify_canonical_missing(closure, missing_blobs)?;
    for (digest, length) in missing {
        if *length > max_payload {
            return Err(SyncError::BlobLaneMismatch {
                digest: *digest,
                length: *length,
                limit: max_payload,
                lane: "pack",
            });
        }
    }
    for (digest, length) in missing_blobs {
        if *length <= max_payload {
            return Err(SyncError::BlobLaneMismatch {
                digest: *digest,
                length: *length,
                limit: max_payload,
                lane: "blob",
            });
        }
    }
    Ok(())
}

/// Drives the multipart blob lane for every object the remote put in it.
///
/// Per blob: ask for the grant, PUT every part the remote does not already
/// hold — reading each part's bytes out of the local object by RANGE, so peak
/// memory is one part however large the blob is — and report the etags.
///
/// A grant lease that runs out mid-blob is a REPLAN, not a failure and not a
/// retry against a URL the store has stopped honouring: the blob goes back in
/// the queue and the next round re-asks, which adopts the same upload and
/// skips every part that already landed. Replans are bounded by the caller's
/// budget, so a remote that expires every grant fails in bounded time.
fn push_blob_lane<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    session: &str,
    blobs: &[(ObjectDigest, u64)],
    options: PushOptions,
    report: &mut PushReport,
    progress: ProgressSink<'_>,
) -> Result<(), SyncError> {
    let mut pending: Vec<(ObjectDigest, u64)> = blobs.to_vec();
    let mut stalled = 0_u32;
    while !pending.is_empty() {
        // A part that lands is DURABLE progress: the store keeps it, and the
        // next grant reports it back so the retry sends only the rest. So a
        // round that landed even one part earns another round, and a blob of
        // any size finishes on a link that keeps resetting — while a round
        // that landed nothing and deferred everything spends the budget.
        let landed_before = report.blob_parts;
        let pending_before = pending.len();
        let mut deferred = Vec::new();
        for batch in pending.chunks(BLOB_GRANT_BATCH) {
            let digests: Vec<ObjectDigest> = batch.iter().map(|(digest, _)| *digest).collect();
            let grants = match transport.blob_grants(session, &digests) {
                Ok(grants) => grants,
                Err(TransportError::Expired(_)) => {
                    deferred.extend_from_slice(batch);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            for (digest, length) in batch {
                let grant = grants
                    .iter()
                    .find(|grant| grant.digest == *digest)
                    .ok_or(SyncError::BlobGrantOmitted(*digest))?;
                match push_one_blob(
                    meta, transport, session, grant, *length, options, report, progress,
                ) {
                    Ok(()) => {
                        report.blobs += 1;
                        report.uploaded_objects += 1;
                        report.uploaded_bytes += *length;
                        // Reported as this blob's parts finish landing, not
                        // once every blob has: one blob can be the whole
                        // transfer.
                        progress.object(*digest, *length);
                    }
                    Err(BlobOutcome::Expired) => deferred.push((*digest, *length)),
                    Err(BlobOutcome::Failed(error)) => return Err(error),
                }
            }
        }
        pending = deferred;
        report.replans += 1;
        if report.blob_parts > landed_before || pending.len() < pending_before {
            stalled = 0;
        } else {
            stalled += 1;
            if stalled >= options.max_stalled_replans {
                return Err(SyncError::ReplansStalled {
                    rounds: stalled,
                    owed: pending.len(),
                });
            }
        }
    }
    Ok(())
}

/// Why one blob's upload stopped: a lease that ran out is the caller's
/// replan, anything else is terminal.
enum BlobOutcome {
    Expired,
    Failed(SyncError),
}

impl From<SyncError> for BlobOutcome {
    fn from(error: SyncError) -> Self {
        Self::Failed(error)
    }
}

#[allow(clippy::too_many_arguments)]
fn push_one_blob<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    session: &str,
    grant: &BlobGrant,
    length: u64,
    options: PushOptions,
    report: &mut PushReport,
    progress: ProgressSink<'_>,
) -> Result<(), BlobOutcome> {
    let digest = grant.digest;
    if grant.length != length {
        return Err(SyncError::MissingLength {
            digest,
            expected: length,
            actual: grant.length,
        }
        .into());
    }
    // The granted parts must cover the object exactly. Fewer bytes would
    // assemble a truncated object the remote then refuses on its own
    // stream-hash — a whole transfer spent to learn something arithmetic
    // answers here.
    let granted: u64 = grant.parts.iter().map(|part| part.size_bytes).sum();
    if granted != length {
        return Err(SyncError::BlobPartCoverage {
            digest,
            length,
            granted,
        }
        .into());
    }

    let resident = meta
        .store()
        .open_object(&digest)
        .map_err(SyncError::Store)?
        .metadata()
        .map_err(|error| SyncError::Store(StoreError::Io(error)))?
        .len();
    if resident != length {
        return Err(SyncError::LocalObjectLength {
            digest,
            expected: length,
            actual: resident,
        }
        .into());
    }
    // One whole-file record IS the blob's shape, so this is the same
    // record-addressed source a direct tensor read uses, pointed at one
    // object. Parts are read by RANGE: peak memory is one part.
    let source = RecordsSource::new(meta.store(), &[FileRecord::Data { digest, length }]);

    let mut landed: Vec<BlobPartReport> = Vec::new();
    let mut offset = 0_u64;
    let mut buffer = Vec::new();
    for part in &grant.parts {
        let start = offset;
        offset += part.size_bytes;
        // Resume is free, and taken: a part the remote already holds is
        // skipped without reading a byte of it.
        if grant.uploaded_parts.contains(&part.part_number) {
            continue;
        }
        let take = usize::try_from(part.size_bytes).map_err(|_| {
            SyncError::Store(StoreError::Io(std::io::ErrorKind::InvalidInput.into()))
        })?;
        buffer.clear();
        buffer.resize(take, 0);
        source
            .read_exact_at(start, &mut buffer)
            .map_err(|error| SyncError::Store(StoreError::Io(error)))?;

        let mut attempt = 0;
        let etag = loop {
            match transport.upload_blob_part(part, &buffer, progress) {
                Ok(etag) => break etag,
                Err(TransportError::Io(_)) => {
                    attempt += 1;
                    if attempt >= options.max_upload_attempts {
                        return Err(SyncError::BlobPartAttemptsExhausted {
                            digest,
                            attempts: attempt,
                        }
                        .into());
                    }
                    back_off(attempt);
                }
                Err(TransportError::Expired(_)) => {
                    // Whatever landed stays landed; the re-grant will report
                    // it and the retry sends only the rest.
                    if !landed.is_empty() {
                        transport
                            .report_blob_parts(session, &digest, &landed)
                            .map_err(|error| BlobOutcome::Failed(error.into()))?;
                    }
                    return Err(BlobOutcome::Expired);
                }
                Err(error) => return Err(SyncError::Transport(error).into()),
            }
        };
        landed.push(BlobPartReport {
            part_number: part.part_number,
            etag,
        });
        report.blob_parts += 1;
    }

    if !landed.is_empty() {
        transport
            .report_blob_parts(session, &digest, &landed)
            .map_err(|error| BlobOutcome::Failed(error.into()))?;
    }
    Ok(())
}

/// Bounded exponential backoff before retrying a transient carrier failure.
/// Capped so a long outage still fails in bounded time rather than hanging.
fn back_off(attempt: u32) {
    const BASE: Duration = Duration::from_millis(250);
    const CEILING: Duration = Duration::from_secs(8);
    let delay = BASE.saturating_mul(1_u32 << attempt.min(6)).min(CEILING);
    std::thread::sleep(delay);
}

/// Streams one object into a fresh verifying writer and admits it, retrying
/// only genuinely transient carrier failures. A refusal or an expired grant is
/// never retried here: the first is terminal and the second belongs to the
/// caller's replan.
///
/// A retry starts a NEW writer, because a half-written one has already hashed
/// bytes the retry is about to send again. The abandoned temp is removed by
/// the writer's own `Drop`, and a killed process leaves it to the age-bounded
/// collector.
fn download_and_admit<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    grant: &DownloadGrant,
    attempts: u32,
    progress: ProgressSink<'_>,
) -> Result<(), SyncError> {
    let mut attempt = 0;
    loop {
        let mut writer = meta.store().writer()?;
        match transport.download(grant, &mut writer, progress) {
            Ok(_) => {
                // The admission boundary is the trust boundary: remote bytes
                // were hashed while written and are refused on any length or
                // digest lie.
                writer.finish_expecting(grant.digest, grant.length)?;
                return Ok(());
            }
            Err(TransportError::Io(detail)) => {
                attempt += 1;
                if attempt >= attempts {
                    return Err(TransportError::Io(detail).into());
                }
                back_off(attempt);
            }
            Err(error) => return Err(error.into()),
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
            // A manifest is kilobytes and its identity has to be checked
            // before anything is decoded from it, so this one download is
            // collected in memory on purpose.
            let mut bytes = Vec::new();
            transport.download(grant, &mut bytes, progress)?;
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
            download_and_admit(meta, transport, grant, DOWNLOAD_ATTEMPTS, progress)?;
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
        BlobGrant, BlobPart, BlobPartReport, CompleteStatus, DownloadGrant, GrantsPlan,
        ObjectDigest, PackClaim, PackGrant, ProgressSink, SnapshotId, StagedPack, SyncPlan,
        SyncTransport, TransportError,
    };

    /// How long a connection may take to ESTABLISH. This is the one deadline
    /// in this module that judges nothing about the work: until the socket is
    /// up there is no peer, no request in flight and nothing to be making
    /// progress. A connect that has not completed is not slow, it is absent.
    const CONNECT_LIMIT: Duration = Duration::from_secs(15);

    /// How long an ESTABLISHED transfer socket may stay SILENT.
    ///
    /// Not a deadline on the transfer — a liveness bound on the carrier. A
    /// transfer has a byte-level progress signal (this module reports every
    /// block to a [`ProgressSink`] as it moves), so silence on the socket is
    /// real evidence that nothing is happening, and a 4 GiB blob is never
    /// condemned for being large. `ureq`'s `timeout_read`/`timeout_write` bound
    /// each individual socket operation, which is exactly that shape; the
    /// whole-request `timeout()` — which is a deadline — is deliberately never
    /// set.
    const TRANSFER_SILENCE: Duration = Duration::from_secs(90);

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
    ///
    /// TWO CARRIERS, because the two halves of this wire offer different
    /// evidence and only one of them can be judged.
    ///
    /// A TRANSFER reports bytes as they move, so socket silence means
    /// something and [`TRANSFER_SILENCE`] bounds it.
    ///
    /// A CONTROL exchange does not. The server is legitimately mute for the
    /// whole of its work — `declare` probes residency for every object in the
    /// closure, `complete` verifies and admits staged objects — and that work
    /// scales with the snapshot, not with anything the client can watch. There
    /// is no observation to take, so the client refuses to guess: the control
    /// carrier has NO read deadline at all. A flat 60-second one is what threw
    /// away a fully-uploaded 272 MB snapshot in the 2026-08-16 acceptance
    /// (#92), by making "promotion slower than a minute" indistinguishable
    /// from "hung" and then deciding for the hang.
    ///
    /// What replaces it is where it belongs: `complete` is retryable and the
    /// hub's own `promoted` count is the progress axis, so a caller supervising
    /// this push judges stuckness from the absence of observations — the rule
    /// [`Progress`] states, applied to the one call that was breaking it.
    #[derive(Debug)]
    pub struct HttpTransport {
        base_url: String,
        org: String,
        repo: String,
        token: TokenSource,
        control: ureq::Agent,
        transfer: ureq::Agent,
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
                control: ureq::AgentBuilder::new()
                    .timeout_connect(CONNECT_LIMIT)
                    .build(),
                transfer: ureq::AgentBuilder::new()
                    .timeout_connect(CONNECT_LIMIT)
                    .timeout_read(TRANSFER_SILENCE)
                    .timeout_write(TRANSFER_SILENCE)
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
            let mut request = self.control.request(method, url);
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
    struct WireBlobPart {
        part_number: u32,
        size_bytes: u64,
        put_url: String,
        #[serde(default)]
        headers: std::collections::BTreeMap<String, String>,
    }

    #[derive(Deserialize)]
    struct WireBlobGrant {
        digest: String,
        length: u64,
        #[serde(default)]
        staging_key: String,
        upload_id: String,
        part_size: u64,
        #[serde(default)]
        parts: Vec<WireBlobPart>,
        #[serde(default)]
        uploaded_parts: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct WireBlobGrants {
        #[serde(default)]
        grants: Vec<WireBlobGrant>,
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
        missing_blobs: Vec<WireMissing>,
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
        #[serde(default)]
        missing_blobs: Vec<WireMissing>,
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
        promoted: u64,
        #[serde(default)]
        total: u64,
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
                missing_blobs: missing_pairs(wire.missing_blobs)?,
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
                missing_blobs: missing_pairs(wire.missing_blobs)?,
            })
        }

        fn blob_grants(
            &self,
            session: &str,
            digests: &[ObjectDigest],
        ) -> Result<Vec<BlobGrant>, TransportError> {
            let body = serde_json::json!({
                "digests": digests.iter().map(tagged).collect::<Vec<_>>(),
            });
            let url = self.route(&format!("/{session}/blob-grants"));
            let (status, value) = self.exchange("POST", &url, Some(&body))?;
            if status != 200 {
                return Err(refuse(status, &value));
            }
            let wire: WireBlobGrants = serde_json::from_value(value)
                .map_err(|error| TransportError::Io(format!("blob-grants response: {error}")))?;
            wire.grants
                .into_iter()
                .map(|grant| {
                    Ok(BlobGrant {
                        digest: parse_ref(&grant.digest)?,
                        length: grant.length,
                        staging_key: grant.staging_key,
                        upload_id: grant.upload_id,
                        part_size: grant.part_size,
                        parts: grant
                            .parts
                            .into_iter()
                            .map(|part| BlobPart {
                                part_number: part.part_number,
                                size_bytes: part.size_bytes,
                                url: part.put_url,
                                headers: part.headers.into_iter().collect(),
                            })
                            .collect(),
                        uploaded_parts: grant.uploaded_parts,
                    })
                })
                .collect()
        }

        fn upload_blob_part(
            &self,
            part: &BlobPart,
            bytes: &[u8],
            progress: ProgressSink<'_>,
        ) -> Result<String, TransportError> {
            let mut request = self.transfer.put(&part.url);
            for (name, value) in &part.headers {
                request = request.set(name, value);
            }
            if !part
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            {
                request = request.set("content-length", &bytes.len().to_string());
            }
            let body = ReportingBody {
                rest: bytes,
                progress,
            };
            match request.send(body) {
                // The etag is the store's name for these exact bytes, and the
                // hub needs it verbatim — quotes included — to complete the
                // upload. A part with no etag cannot be completed at all, so
                // its absence is a refusal rather than an empty string.
                Ok(response) => match response.header("etag") {
                    Some(etag) if !etag.is_empty() => Ok(etag.to_owned()),
                    _ => Err(TransportError::Io(format!(
                        "part {} landed without an etag",
                        part.part_number
                    ))),
                },
                Err(ureq::Error::Status(403, _)) => Err(TransportError::Expired(
                    "blob part grant refused (403)".to_owned(),
                )),
                // The multipart upload this part belongs to is gone — the
                // store answers `NoSuchUpload` with a 404. That happens when
                // the remote's staging sweep aborts an upload a resumed push
                // had adopted, and it is an EXPIRY, not a failure: the replan
                // re-asks, the remote opens a fresh upload, and the parts go
                // again. Treating it as terminal would make an adopted upload
                // riskier than a cold one, which would defeat adopting at all.
                Err(ureq::Error::Status(404, _)) => Err(TransportError::Expired(
                    "blob part upload no longer exists (404)".to_owned(),
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

        fn report_blob_parts(
            &self,
            session: &str,
            digest: &ObjectDigest,
            parts: &[BlobPartReport],
        ) -> Result<(), TransportError> {
            let body = serde_json::json!({
                "digest": tagged(digest),
                "parts": parts
                    .iter()
                    .map(|part| serde_json::json!({
                        "part_number": part.part_number,
                        "etag": part.etag,
                    }))
                    .collect::<Vec<_>>(),
            });
            let url = self.route(&format!("/{session}/blob-parts"));
            let (status, value) = self.exchange("POST", &url, Some(&body))?;
            if status != 200 && status != 204 {
                return Err(refuse(status, &value));
            }
            Ok(())
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
            let mut request = self.transfer.put(&grant.url);
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
                        CompleteStatus::Incomplete {
                            code: failure.code,
                            promoted: wire.promoted,
                            total: wire.total,
                        }
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
            sink: &mut dyn std::io::Write,
            progress: ProgressSink<'_>,
        ) -> Result<u64, TransportError> {
            let response = match self.transfer.get(&grant.url).call() {
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
            // Block by block into the caller's sink, so a slow object reports
            // movement while it arrives and a multi-gigabyte one never exists
            // whole in memory. Reading one byte past the granted length is
            // deliberate: a remote sending MORE than it granted must be
            // caught, not silently truncated into a passing admission.
            let mut written = 0_u64;
            let mut reader = response.into_reader().take(grant.length + 1);
            let mut block = vec![0_u8; DOWNLOAD_BLOCK];
            loop {
                let read = reader
                    .read(&mut block)
                    .map_err(|error| TransportError::Io(error.to_string()))?;
                if read == 0 {
                    break;
                }
                sink.write_all(&block[..read])
                    .map_err(|error| TransportError::Io(error.to_string()))?;
                written += read as u64;
                progress.bytes(read as u64);
            }
            if written != grant.length {
                return Err(TransportError::Io(format!(
                    "download for {} returned {written} bytes, grant says {}",
                    grant.digest, grant.length
                )));
            }
            Ok(written)
        }
    }
}
