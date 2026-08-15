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

use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::object::ObjectDigest;
use crate::store::StoreError;
use crate::tfm1::{Entry, FileRecord, Snapshot, SnapshotId, Tfm1Error, decode};
use crate::tfp1::{MAX_PACK_OBJECTS, MAX_PACK_PAYLOAD, Tfp1Error};
use crate::workspace::{WorkspaceError, WorkspaceStore};

/// One bounded download-grant batch, per the wire contract.
pub const DOWNLOAD_GRANT_BATCH: usize = 256;

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
/// declare body at complete, strictly before the head becomes visible; push
/// never packs it.
#[must_use]
pub fn manifest_object_digest(id: &SnapshotId) -> ObjectDigest {
    ObjectDigest::from_bytes(*id.as_bytes())
}

/// Pushes one sealed local snapshot: declare, pack and upload only missing
/// objects under claim-bound grants, then drive `complete` to a terminal
/// answer. Within a run, replans re-probe the session and never retransmit a
/// staged pack; across restarts, promoted objects report resident and are
/// never retransmitted.
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

    let mut report = PushReport::default();
    let plan = transport.declare(&tfm1_bytes, expected_head)?;
    if plan.snapshot_id != *snapshot {
        return Err(SyncError::IdentityMismatch {
            local: *snapshot,
            remote: plan.snapshot_id,
        });
    }
    // What the remote already held before this push moved anything.
    report.skipped_remote_resident = plan.have.len() as u64;

    let session = plan.session.clone();
    let max_payload = if plan.max_pack_payload == 0 {
        MAX_PACK_PAYLOAD
    } else {
        plan.max_pack_payload.min(MAX_PACK_PAYLOAD)
    };

    let mut missing = plan.missing;
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
                match transport.upload_pack(grant, &encoded) {
                    Ok(()) => break,
                    Err(TransportError::Io(detail)) => {
                        attempt += 1;
                        if attempt >= options.max_upload_attempts {
                            return Err(TransportError::Io(detail).into());
                        }
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
        }
        let _ = refresh_early;

        // A pass completed or a grant expired: re-probe the session's live
        // staged view and re-derive what is still missing.
        report.replans += 1;
        if report.replans > options.max_replans {
            return Err(SyncError::ReplansExhausted(report.replans));
        }
        missing = transport.pack_grants(&session, &[])?.missing;
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
        CompleteStatus, DownloadGrant, GrantsPlan, ObjectDigest, PackClaim, PackGrant, SnapshotId,
        StagedPack, SyncPlan, SyncTransport, TransportError,
    };

    const CONTROL_TIMEOUT: Duration = Duration::from_secs(60);
    const TRANSFER_TIMEOUT: Duration = Duration::from_secs(600);

    /// Where the bearer token comes from. A file is re-read on every call so
    /// an external refresher can rotate it without restarting the daemon.
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

        fn upload_pack(&self, grant: &PackGrant, pack: &[u8]) -> Result<(), TransportError> {
            // Presigned: the grant's own headers are the whole authority and
            // are replayed verbatim — the pack checksum lives inside the
            // signature, so changed bytes refuse at the store.
            let mut request = self.agent.put(&grant.url).timeout(TRANSFER_TIMEOUT);
            for (name, value) in &grant.headers {
                request = request.set(name, value);
            }
            match request.send_bytes(pack) {
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

        fn download(&self, grant: &DownloadGrant) -> Result<Vec<u8>, TransportError> {
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
            let mut bytes = Vec::new();
            let mut reader = response.into_reader().take(grant.length + 1);
            reader
                .read_to_end(&mut bytes)
                .map_err(|error| TransportError::Io(error.to_string()))?;
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
