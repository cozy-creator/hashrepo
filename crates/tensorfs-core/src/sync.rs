//! Snapshot sync: push a sealed TFM1 snapshot to a remote grant service and
//! pull one into a local store, moving only missing objects.
//!
//! The engine is transport-abstracted. [`SyncTransport`] mirrors the
//! th#1960 wire SHAPE (declare → have/staged/pack grants → complete;
//! head → download grants → downloads); the hub lane owns the exact HTTP
//! spellings and the [`http`] adapter tracks them. Bytes never proxy through
//! the control plane: uploads are whole TFP1 packs of missing objects under
//! presigned grants, downloads are per-object presigned reads admitted
//! through the local store's verifying writer. The local `ObjectStore` is the
//! resume journal on both directions — a verified resident object is never
//! transferred again.

use std::collections::HashSet;
use std::io::Read;
use std::io::Write as _;

use thiserror::Error;

use crate::object::ObjectDigest;
use crate::store::StoreError;
use crate::tfm1::{Entry, FileRecord, Snapshot, SnapshotId, Tfm1Error, decode};
use crate::tfp1::{MAX_PACK_OBJECTS, MAX_PACK_PAYLOAD, Tfp1Error};
use crate::workspace::{WorkspaceError, WorkspaceStore};

/// One bounded download-grant batch, per the wire contract.
pub const DOWNLOAD_GRANT_BATCH: usize = 512;

/// One presigned authorization to PUT one TFP1 staging pack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackGrant {
    pub staging_key: String,
    pub url: String,
    pub max_payload: u64,
}

/// One presigned authorization to GET one exact object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadGrant {
    pub digest: ObjectDigest,
    pub length: u64,
    pub url: String,
}

/// The remote's answer to a declare/resume: what it holds, what this session
/// already staged, and where the next packs may go.
#[derive(Clone, Debug)]
pub struct SyncPlan {
    pub snapshot_id: SnapshotId,
    pub session: String,
    pub have: Vec<ObjectDigest>,
    pub staged: Vec<ObjectDigest>,
    pub pack_grants: Vec<PackGrant>,
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
    fn more_grants(&self, session: &str) -> Result<SyncPlan, TransportError>;
    fn upload_pack(&self, grant: &PackGrant, pack: &[u8]) -> Result<(), TransportError>;
    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError>;
    fn head(&self) -> Result<Option<SnapshotId>, TransportError>;
    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError>;
    fn download(&self, grant: &DownloadGrant) -> Result<Vec<u8>, TransportError>;
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
                if let FileRecord::Data { digest, length } = record {
                    if seen.insert(*digest.as_bytes()) {
                        closure.push((*digest, *length));
                    }
                }
            }
        }
    }
    closure
}

/// The digest the snapshot's own canonical bytes occupy in the object
/// namespace: the manifest blob is digest-addressed by its snapshot id, so
/// pull needs no second manifest channel. The hub admits the blob from the
/// declare body itself; push never packs it.
#[must_use]
pub fn manifest_object_digest(id: &SnapshotId) -> ObjectDigest {
    ObjectDigest::from_bytes(*id.as_bytes())
}

/// Pushes one sealed local snapshot: declare, pack and upload only missing
/// objects, then drive `complete` to a terminal answer. Restart-safe: a
/// re-push of an interrupted session uploads only what the remote still
/// reports missing.
pub fn push_snapshot<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    snapshot: &SnapshotId,
    expected_head: Option<&SnapshotId>,
    options: PushOptions,
) -> Result<PushReport, SyncError> {
    // `load_snapshot` re-verifies the stored blob against its id, and TFM1
    // re-encoding is byte-exact, so these are the canonical bytes.
    let decoded = meta.load_snapshot(snapshot)?;
    let tfm1_bytes = decoded.to_bytes();
    let closure = data_closure(&decoded);

    let mut report = PushReport::default();
    let mut plan = transport.declare(&tfm1_bytes, expected_head)?;
    if plan.snapshot_id != *snapshot {
        return Err(SyncError::IdentityMismatch {
            local: *snapshot,
            remote: plan.snapshot_id,
        });
    }

    let mut first_plan = true;
    loop {
        let done: HashSet<[u8; 32]> = plan
            .have
            .iter()
            .chain(plan.staged.iter())
            .map(|digest| *digest.as_bytes())
            .collect();
        let missing: Vec<(ObjectDigest, u64)> = closure
            .iter()
            .filter(|(digest, _)| !done.contains(digest.as_bytes()))
            .copied()
            .collect();
        if first_plan {
            // What the remote already held before this push moved anything;
            // later passes see our own staging and must not count it.
            report.skipped_remote_resident = (closure.len() - missing.len()) as u64;
            first_plan = false;
        }
        if missing.is_empty() {
            break;
        }

        // Greedy whole-object pack assembly in manifest order. One pack is in
        // flight at a time, so peak transfer memory is bounded by one payload
        // plus its encoding.
        let mut grants = plan.pack_grants.clone().into_iter();
        let mut pack_members: Vec<(ObjectDigest, u64)> = Vec::new();
        let mut pack_bytes = 0_u64;
        let mut refreshed_mid_flight = false;
        let flush = |members: &mut Vec<(ObjectDigest, u64)>,
                     grants: &mut dyn Iterator<Item = PackGrant>,
                     report: &mut PushReport|
         -> Result<Option<()>, SyncError> {
            if members.is_empty() {
                return Ok(Some(()));
            }
            let Some(grant) = grants.next() else {
                return Ok(None);
            };
            let loaded = load_pack_members(meta, members)?;
            let borrowed: Vec<(ObjectDigest, &[u8])> = loaded
                .iter()
                .map(|(digest, bytes)| (*digest, bytes.as_slice()))
                .collect();
            let encoded = crate::tfp1::encode(&borrowed)?;
            let payload: u64 = members.iter().map(|(_, length)| *length).sum();
            let mut attempt = 0;
            loop {
                match transport.upload_pack(&grant, &encoded) {
                    Ok(()) => break,
                    Err(TransportError::Io(detail)) => {
                        attempt += 1;
                        if attempt >= options.max_upload_attempts {
                            return Err(TransportError::Io(detail).into());
                        }
                    }
                    Err(TransportError::Expired(_)) => return Ok(None),
                    Err(error) => return Err(error.into()),
                }
            }
            report.uploaded_objects += members.len() as u64;
            report.uploaded_bytes += payload;
            report.packs += 1;
            members.clear();
            Ok(Some(()))
        };

        for (digest, length) in &missing {
            let over_payload = pack_bytes + length > MAX_PACK_PAYLOAD;
            let over_count = pack_members.len() >= MAX_PACK_OBJECTS;
            if !pack_members.is_empty() && (over_payload || over_count) {
                match flush(&mut pack_members, &mut grants, &mut report)? {
                    Some(()) => pack_bytes = 0,
                    None => {
                        refreshed_mid_flight = true;
                        break;
                    }
                }
            }
            pack_members.push((*digest, *length));
            pack_bytes += length;
        }
        if !refreshed_mid_flight && flush(&mut pack_members, &mut grants, &mut report)?.is_none() {
            refreshed_mid_flight = true;
        }

        // Grants ran out, a grant expired, or a pass completed: refresh the
        // remote's staged view and re-derive what is still missing.
        report.replans += 1;
        if report.replans > options.max_replans {
            return Err(SyncError::ReplansExhausted(report.replans));
        }
        let _ = refreshed_mid_flight;
        plan = transport.more_grants(&plan.session)?;
    }

    loop {
        report.complete_attempts += 1;
        match transport.complete(&plan.session)? {
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
/// snapshot so it is locally sealed and mountable. Resident objects are the
/// resume journal and are never re-fetched.
pub fn pull_head<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
) -> Result<(SnapshotId, PullReport), SyncError> {
    let head = transport.head()?.ok_or(SyncError::NoRemoteHead)?;
    let report = pull_snapshot(meta, transport, &head)?;
    Ok((head, report))
}

/// Pulls one exact remote snapshot id into the local store.
pub fn pull_snapshot<T: SyncTransport>(
    meta: &WorkspaceStore,
    transport: &T,
    id: &SnapshotId,
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
            let bytes = transport.download(grant)?;
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
    let missing: Vec<(ObjectDigest, u64)> = closure
        .iter()
        .filter(|(digest, _)| !meta.store().exists(digest))
        .copied()
        .collect();
    report.skipped_local_resident = (closure.len() - missing.len()) as u64;

    for batch in missing.chunks(DOWNLOAD_GRANT_BATCH) {
        let digests: Vec<ObjectDigest> = batch.iter().map(|(digest, _)| *digest).collect();
        let grants = transport.download_grants(&digests)?;
        for (digest, length) in batch {
            let grant = grants
                .iter()
                .find(|grant| grant.digest == *digest)
                .ok_or(SyncError::GrantOmitted(*digest))?;
            let bytes = transport.download(grant)?;
            let mut writer = meta.store().writer()?;
            writer
                .write_all(&bytes)
                .map_err(|error| SyncError::Store(StoreError::Io(error)))?;
            // The admission boundary is the trust boundary: remote bytes are
            // hashed while written and refused on any length or digest lie.
            writer.finish_expecting(*digest, *length)?;
            report.fetched_objects += 1;
            report.fetched_bytes += length;
        }
    }

    meta.adopt_snapshot(&tfm1_bytes)?;
    Ok(report)
}

/// The th#1960 HTTP adapter seam. The hub lane owns the landed route and
/// field spellings; until it reports, every call is an honest typed refusal
/// rather than a guessed wire.
pub mod http {
    use super::{
        CompleteStatus, DownloadGrant, ObjectDigest, PackGrant, SnapshotId, SyncPlan,
        SyncTransport, TransportError,
    };

    /// Placeholder client for the tensorhub snapshot-sync routes.
    #[derive(Clone, Debug)]
    pub struct HttpTransport {
        base_url: String,
    }

    impl HttpTransport {
        #[must_use]
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
            }
        }

        #[must_use]
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        fn unimplemented<V>(&self) -> Result<V, TransportError> {
            Err(TransportError::Refused {
                code: "transport-unimplemented".to_owned(),
                detail: "the th#1960 hub wire has not landed; this adapter tracks it".to_owned(),
            })
        }
    }

    impl SyncTransport for HttpTransport {
        fn declare(
            &self,
            _tfm1_bytes: &[u8],
            _expected_head: Option<&SnapshotId>,
        ) -> Result<SyncPlan, TransportError> {
            self.unimplemented()
        }

        fn more_grants(&self, _session: &str) -> Result<SyncPlan, TransportError> {
            self.unimplemented()
        }

        fn upload_pack(&self, _grant: &PackGrant, _pack: &[u8]) -> Result<(), TransportError> {
            self.unimplemented()
        }

        fn complete(&self, _session: &str) -> Result<CompleteStatus, TransportError> {
            self.unimplemented()
        }

        fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
            self.unimplemented()
        }

        fn download_grants(
            &self,
            _digests: &[ObjectDigest],
        ) -> Result<Vec<DownloadGrant>, TransportError> {
            self.unimplemented()
        }

        fn download(&self, _grant: &DownloadGrant) -> Result<Vec<u8>, TransportError> {
            self.unimplemented()
        }
    }
}
