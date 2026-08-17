//! Shared fixtures and instrumentation for the live-hub proofs.
//!
//! Two properties keep every live assertion honest and they are enforced here
//! rather than restated per test: assertions read the HUB'S OWN ANSWER, and a
//! per-run nonce rides the TENSOR NAMES so a long-lived shared hub cannot
//! already hold a fixture. Without the nonce, "uploaded" would prove nothing.

#![allow(dead_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::{ByteSource, PlannerId};
use tensorfs_core::store::ObjectStore;
use tensorfs_core::sync::http::{HttpTransport, TokenSource};
use tensorfs_core::sync::{
    BlobGrant, BlobPart, BlobPartReport, CompleteStatus, DownloadGrant, GrantsPlan, PackClaim,
    PackGrant, ProgressSink, SyncPlan, SyncTransport, TransportError,
};
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

pub const MIB: usize = 1024 * 1024;

pub struct Slice<'a>(pub &'a [u8]);

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

/// The live hub's coordinates.
pub struct Hub {
    pub url: String,
    pub org: String,
    pub repo: String,
    pub token: String,
}

/// Reads the hub coordinates, or explains loudly and exactly what is missing.
/// A silent skip defends nothing, so the skip names every absent variable.
pub fn hub() -> Option<Hub> {
    let url = env::var("TENSORFS_HUB_URL").unwrap_or_else(|_| "http://127.0.0.1:31550".to_owned());
    let mut absent = Vec::new();
    for name in [
        "TENSORFS_HUB_ORG",
        "TENSORFS_HUB_REPO",
        "TENSORFS_HUB_TOKEN",
    ] {
        if env::var(name).is_err() {
            absent.push(name);
        }
    }
    if !absent.is_empty() {
        eprintln!(
            "SKIPPING the live-hub proof: {} not set.\n  \
             Set TENSORFS_HUB_URL (default {url}), TENSORFS_HUB_ORG, TENSORFS_HUB_REPO and \
             TENSORFS_HUB_TOKEN.\n  \
             See crates/tensorfs-core/tests/README-live-hub.md for the exact token mint.",
            absent.join(", ")
        );
        return None;
    }
    Some(Hub {
        url,
        org: env::var("TENSORFS_HUB_ORG").expect("checked above"),
        repo: env::var("TENSORFS_HUB_REPO").expect("checked above"),
        token: env::var("TENSORFS_HUB_TOKEN").expect("checked above"),
    })
}

pub fn transport(hub: &Hub) -> HttpTransport {
    HttpTransport::new(&hub.url, &hub.org, &hub.repo)
        .with_token(TokenSource::Static(hub.token.clone()))
}

/// An env-tunable knob with a documented default, so a run can be scaled
/// without editing the suite.
pub fn knob(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(default)
}

/// A deterministic safetensors file whose tensors straddle the 64 MiB grid, so
/// the planner produces several independent objects per file.
///
/// **The nonce rides the PAYLOAD, not only the tensor names.** Names land in
/// the header object alone; every data object is a pure slice of tensor bytes
/// and is therefore name-independent. A fixture whose payload is a constant
/// fill produces byte-identical data objects on every run, so a hub that saw a
/// previous run reports them resident and "fresh content must upload" quietly
/// stops meaning anything. Seeding the payload from the nonce is what actually
/// makes a run's objects new.
pub fn safetensors(nonce: &str, tensors: &[(&str, usize, u8)]) -> Vec<u8> {
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
        // Deterministic, nonce-seeded, and cheap: every 32-byte lane differs
        // per run, so every planned object differs per run.
        file.extend((0..*length).map(|index| fill ^ seed[index % seed.len()]));
    }
    file
}

/// Admits one file's planned objects and returns its ordered records.
pub fn admit(store: &ObjectStore, bytes: &[u8]) -> Vec<FileRecord> {
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

pub fn seal(root: &Path, files: &[(&str, Vec<u8>)]) -> (WorkspaceStore, SnapshotId) {
    let meta = WorkspaceStore::open(root).expect("store opens");
    meta.create_workspace("main").expect("workspace created");
    let mutations: Vec<Mutation> = files
        .iter()
        .map(|(path, bytes)| Mutation::CreateFile {
            path: (*path).to_owned(),
            executable: false,
            planner: PlannerId::SafetensorsV1,
            records: admit(meta.store(), bytes),
        })
        .collect();
    meta.commit_generation("main", &mutations)
        .expect("generation commits");
    let id = meta.seal_snapshot("main", None).expect("snapshot seals");
    (meta, id)
}

/// Reconstructs one file from the local store alone — the byte-exactness claim.
pub fn file_bytes(meta: &WorkspaceStore, id: &SnapshotId, path: &str) -> Vec<u8> {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    let mut out = Vec::new();
    for (entry_path, entry) in snapshot.entries() {
        if entry_path != path {
            continue;
        }
        let Entry::File { body, .. } = entry else {
            panic!("{path} is not a regular file");
        };
        for record in body.records().iter() {
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

pub fn scratch(name: &str) -> PathBuf {
    let base = env::var("TENSORFS_E2E_DIR")
        .map_or_else(|_| env::temp_dir().join("tensorfs-live-e2e"), PathBuf::from);
    let path = base.join(name);
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("scratch dir");
    path
}

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

/// Exact per-route HTTP call counts and moved bytes. Efficiency is a claim
/// like any other: it is measured here, never assumed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Counts {
    pub declare: u64,
    pub pack_grants: u64,
    pub upload_pack: u64,
    pub complete: u64,
    pub head: u64,
    pub download_grants: u64,
    pub download: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
}

impl Counts {
    /// Every HTTP request the round trip actually issued.
    #[must_use]
    pub fn requests(&self) -> u64 {
        self.declare
            + self.pack_grants
            + self.upload_pack
            + self.complete
            + self.head
            + self.download_grants
            + self.download
    }
}

/// One transfer's wall-clock span, in nanoseconds from a common base. Spans
/// are how we answer "is transfer parallel?" with evidence instead of belief.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start_nanos: u128,
    pub end_nanos: u128,
}

/// A `SyncTransport` decorator that counts calls, bytes and transfer spans
/// without altering a single byte or answer.
pub struct Counting<'a, T: SyncTransport> {
    inner: &'a T,
    counts: Mutex<Counts>,
    spans: Mutex<Vec<Span>>,
    base: Instant,
}

impl<'a, T: SyncTransport> Counting<'a, T> {
    pub fn new(inner: &'a T) -> Self {
        Self {
            inner,
            counts: Mutex::new(Counts::default()),
            spans: Mutex::new(Vec::new()),
            base: Instant::now(),
        }
    }

    pub fn counts(&self) -> Counts {
        *self.counts.lock().expect("counts lock")
    }

    pub fn spans(&self) -> Vec<Span> {
        self.spans.lock().expect("spans lock").clone()
    }

    /// The greatest number of transfers ever in flight at once. 1 means the
    /// engine is strictly serial; anything above 1 means it overlaps them.
    #[must_use]
    pub fn peak_concurrency(&self) -> usize {
        let spans = self.spans();
        let mut edges: Vec<(u128, i32)> = Vec::with_capacity(spans.len() * 2);
        for span in &spans {
            edges.push((span.start_nanos, 1));
            edges.push((span.end_nanos, -1));
        }
        // Ends sort before starts at an equal instant, so touching spans are
        // not miscounted as overlapping.
        edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut live = 0_i32;
        let mut peak = 0_i32;
        for (_, delta) in edges {
            live += delta;
            peak = peak.max(live);
        }
        usize::try_from(peak.max(0)).expect("peak fits")
    }

    fn record<R>(&self, call: impl FnOnce() -> R) -> R {
        let start = self.base.elapsed().as_nanos();
        let outcome = call();
        let end = self.base.elapsed().as_nanos();
        self.spans.lock().expect("spans lock").push(Span {
            start_nanos: start,
            end_nanos: end,
        });
        outcome
    }

    fn bump(&self, field: impl FnOnce(&mut Counts)) {
        field(&mut self.counts.lock().expect("counts lock"));
    }
}

impl<T: SyncTransport> SyncTransport for Counting<'_, T> {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        self.bump(|counts| counts.declare += 1);
        self.inner.declare(tfm1_bytes, expected_head)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        self.bump(|counts| counts.pack_grants += 1);
        self.inner.pack_grants(session, claims)
    }

    fn upload_pack(
        &self,
        grant: &PackGrant,
        pack: &[u8],
        progress: ProgressSink<'_>,
    ) -> Result<(), TransportError> {
        self.bump(|counts| {
            counts.upload_pack += 1;
            counts.uploaded_bytes += pack.len() as u64;
        });
        self.record(|| self.inner.upload_pack(grant, pack, progress))
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.bump(|counts| counts.complete += 1);
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.bump(|counts| counts.head += 1);
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[tensorfs_core::object::ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.bump(|counts| counts.download_grants += 1);
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
        let written = self.record(|| self.inner.download(grant, sink, progress));
        if let Ok(written) = &written {
            self.bump(|counts| {
                counts.download += 1;
                counts.downloaded_bytes += *written;
            });
        }
        written
    }
}

/// A decorator that fails the Nth pack upload with a carrier error, so a push
/// can be interrupted at a chosen point against the REAL hub. The hub's state
/// is genuine; only the client dies. This is a client abort, not a SIGKILL —
/// process-level kills belong to the hermetic suite.
pub struct FailUploadAfter<'a, T: SyncTransport> {
    inner: &'a T,
    budget: Mutex<usize>,
    survived: Mutex<usize>,
}

impl<'a, T: SyncTransport> FailUploadAfter<'a, T> {
    pub fn new(inner: &'a T, allow: usize) -> Self {
        Self {
            inner,
            budget: Mutex::new(allow),
            survived: Mutex::new(0),
        }
    }

    /// How many pack uploads actually reached the hub before the abort.
    pub fn survived(&self) -> usize {
        *self.survived.lock().expect("survived lock")
    }
}

impl<T: SyncTransport> SyncTransport for FailUploadAfter<'_, T> {
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
        {
            let mut budget = self.budget.lock().expect("budget lock");
            if *budget == 0 {
                return Err(TransportError::Io(
                    "injected carrier failure: the client dies mid-transfer".to_owned(),
                ));
            }
            *budget -= 1;
        }
        let outcome = self.inner.upload_pack(grant, pack, progress);
        if outcome.is_ok() {
            *self.survived.lock().expect("survived lock") += 1;
        }
        outcome
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[tensorfs_core::object::ObjectDigest],
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
