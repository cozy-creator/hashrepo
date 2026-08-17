//! #83: a trimmed model stops downloading the bytes it deletes.
//!
//! A subset snapshot omits the trimmed tensors' records, and that is the whole
//! mechanism. Nothing in sync or GC knows what a subset is: the download win
//! falls out of ordinary missing-object computation, and the two snapshots are
//! two ordinary roots for the ordinary mark walk.

#![cfg(any(unix, windows))]

mod harness;

use std::cell::RefCell;
use std::collections::BTreeMap;

use harness::{DirTransport, Scratch, data_digests};
use tensorfs_core::compose::{rekey, subset};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::sync::{
    BlobGrant, BlobPart, BlobPartReport, CompleteStatus, DownloadGrant, GrantsPlan, PackClaim,
    PackGrant, ProgressSink, PushOptions, SyncPlan, SyncTransport, TransportError, pull_snapshot,
    push_snapshot,
};
use tensorfs_core::tfm1::{Entry, FileBody, FileRecord, SnapshotBuilder, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const PATH: &str = "model.safetensors";

/// The conditioner trim from the corpus audit, at test scale: keep the first
/// layers, delete the rest and the head.
const KEPT: [&str; 2] = ["layers.0.weight", "layers.1.weight"];
const TRIMMED: [&str; 2] = ["layers.2.weight", "lm_head.weight"];

fn safetensors() -> Vec<u8> {
    let tensors: [(&str, u64, u8); 4] = [
        ("layers.0.weight", 8192, 0x11),
        ("layers.1.weight", 4096, 0x22),
        ("layers.2.weight", 16384, 0x33),
        ("lm_head.weight", 65536, 0x44),
    ];
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, length, _)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let end = offset + length;
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{length}],\"data_offsets\":[{offset},{end}]}}"
        ));
        offset = end;
    }
    header.push('}');

    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(fill, length as usize));
    }
    file
}

/// Commits the fixture as one whole object and seals it, so the snapshot
/// carries exactly the grid the planner gives a real ingest.
fn published(root: &std::path::Path) -> (WorkspaceStore, SnapshotId) {
    let meta = WorkspaceStore::open(root).expect("workspace store opens");
    meta.create_workspace("w").expect("workspace creates");
    let bytes = safetensors();
    let whole = meta.store().put_bytes(&bytes).expect("fixture admits");
    meta.commit_generation(
        "w",
        &[Mutation::CreateFile {
            path: PATH.to_owned(),
            executable: false,
            planner: PlannerId::BlobV1,
            records: vec![FileRecord::Data {
                digest: whole.digest(),
                length: whole.length(),
            }],
        }],
    )
    .expect("commits");
    let id = meta.seal_snapshot("w", None).expect("seals");
    (meta, id)
}

fn body_of(meta: &WorkspaceStore, id: &SnapshotId) -> FileBody {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    snapshot
        .entries()
        .iter()
        .find_map(|(path, entry)| match entry {
            Entry::File { body, .. } if path == PATH => Some(body.clone()),
            _ => None,
        })
        .expect("the sealed snapshot holds the fixture")
}

fn adopt(meta: &WorkspaceStore, body: &FileBody) -> SnapshotId {
    let FileBody::Tensor {
        format, records, ..
    } = body
    else {
        panic!("a composed body is a tensor container");
    };
    let mut builder = SnapshotBuilder::new(None);
    builder.file(PATH, false, format.planner_id(), records.clone());
    let snapshot = builder.finish().expect("the composed body is a valid file");
    meta.adopt_snapshot(&snapshot.to_bytes())
        .expect("a composition names only resident objects")
}

fn identity(names: &[&str]) -> BTreeMap<String, String> {
    names
        .iter()
        .map(|name| ((*name).to_owned(), (*name).to_owned()))
        .collect()
}

/// Delegates to a real transport and records every digest a pull asks for.
/// The requested set IS the pull's missing set: an object the plan does not
/// list is never granted and never streamed.
struct Recording<'a> {
    inner: &'a DirTransport,
    asked: RefCell<Vec<ObjectDigest>>,
}

impl<'a> Recording<'a> {
    fn new(inner: &'a DirTransport) -> Self {
        Self {
            inner,
            asked: RefCell::new(Vec::new()),
        }
    }
}

impl SyncTransport for Recording<'_> {
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
        self.asked.borrow_mut().extend_from_slice(digests);
        self.inner.download_grants(digests)
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

#[test]
fn a_cold_pull_of_a_subset_never_asks_for_the_trimmed_objects() {
    let scratch = Scratch::new("subset-pull");
    let (meta, source) = published(&scratch.path().join("publisher"));
    let body = body_of(&meta, &source);

    let trimmed = subset(meta.store(), &body, &identity(&KEPT)).expect("the trim composes");
    let renamed = rekey(
        meta.store(),
        &body,
        &identity(&[KEPT[0], KEPT[1], TRIMMED[0], TRIMMED[1]]),
    )
    .expect("the full re-key composes");
    let subset_id = adopt(&meta, &trimmed);
    let full_id = adopt(&meta, &renamed);

    let source_objects = data_digests(&meta, &source);
    let subset_objects = data_digests(&meta, &subset_id);
    let full_objects = data_digests(&meta, &full_id);
    let omitted: Vec<ObjectDigest> = source_objects
        .iter()
        .filter(|digest| !subset_objects.contains(digest))
        .copied()
        .collect();
    assert_eq!(omitted.len(), 3, "two trimmed tensors and the old header");
    // What the trim deleted and the full record list still names: the two
    // trimmed tensors, and the old header too when an identity re-key happens
    // to reproduce it byte for byte.
    let deleted_tensors: Vec<ObjectDigest> = omitted
        .iter()
        .filter(|digest| full_objects.contains(digest))
        .copied()
        .collect();
    assert!(deleted_tensors.len() >= 2, "{deleted_tensors:?}");

    let hub = DirTransport::new(scratch.path().join("hub"));
    let mut head = None;
    for id in [&source, &subset_id, &full_id] {
        push_snapshot(
            &meta,
            &hub,
            id,
            head.as_ref(),
            PushOptions::default(),
            ProgressSink::silent(),
        )
        .expect("push");
        head = Some(*id);
    }

    // Cold: a store that holds nothing at all, so every object it ends up with
    // is one it asked the remote for.
    let cold = WorkspaceStore::open(scratch.path().join("cold")).expect("cold store opens");
    let recording = Recording::new(&hub);
    let report = pull_snapshot(&cold, &recording, &subset_id, ProgressSink::silent())
        .expect("the subset pulls");

    let asked = recording.asked.borrow().clone();
    for digest in &omitted {
        assert!(
            !asked.contains(digest),
            "a subset pull asked for {digest}, which its record list does not name"
        );
    }
    assert_eq!(
        report.fetched_objects as usize,
        subset_objects.len(),
        "exactly the subset's own closure moved"
    );

    // The red arm, in the same test: inherit the FULL record list instead --
    // the same composition, the same store, the same hub -- and the trimmed
    // objects are fetched after all. Omitting the records is what saves the
    // bytes; nothing else does.
    let greedy = WorkspaceStore::open(scratch.path().join("greedy")).expect("store opens");
    let recording = Recording::new(&hub);
    pull_snapshot(&greedy, &recording, &full_id, ProgressSink::silent()).expect("full pull");
    let asked = recording.asked.borrow().clone();
    for digest in &deleted_tensors {
        assert!(
            asked.contains(digest),
            "the full record list must fetch {digest}"
        );
    }
}

#[test]
fn pushing_a_subset_beside_its_source_moves_only_the_new_header() {
    let scratch = Scratch::new("subset-push");
    let (meta, source) = published(&scratch.path().join("publisher"));
    let body = body_of(&meta, &source);
    let subset_id = adopt(
        &meta,
        &subset(meta.store(), &body, &identity(&KEPT)).expect("composes"),
    );

    let hub = DirTransport::new(scratch.path().join("hub"));
    push_snapshot(
        &meta,
        &hub,
        &source,
        None,
        PushOptions::default(),
        ProgressSink::silent(),
    )
    .expect("push");
    let delta = push_snapshot(
        &meta,
        &hub,
        &subset_id,
        Some(&source),
        PushOptions::default(),
        ProgressSink::silent(),
    )
    .expect("push");

    assert_eq!(
        delta.uploaded_objects, 1,
        "the composed header, and nothing else"
    );
}

/// Two roots, one deleted. This is the existing mark walk with nothing added:
/// a composed manifest pins its sources because its record list names them.
#[test]
fn deleting_the_source_keeps_the_subsets_objects_alive() {
    let scratch = Scratch::new("subset-gc");
    let (meta, source) = published(&scratch.path().join("publisher"));
    let body = body_of(&meta, &source);
    let subset_id = adopt(
        &meta,
        &subset(meta.store(), &body, &identity(&KEPT)).expect("composes"),
    );

    let kept = data_digests(&meta, &subset_id);
    let source_objects = data_digests(&meta, &source);
    let omitted: Vec<ObjectDigest> = source_objects
        .iter()
        .filter(|digest| !kept.contains(digest))
        .copied()
        .collect();

    // Two roots become one. The workspace the source was sealed from is a root
    // in its own right — its head tree names the same objects — so the source
    // is gone only once both are.
    meta.delete_snapshot(&source).expect("the source root goes");
    meta.delete_workspace("w").expect("the staging root goes");

    // The two-epoch quarantine deletes on a later sweep, not on the one that
    // first sees an object unreferenced. Sweep until it does, bounded by the
    // number of epochs that rule can possibly need.
    let mut deleted = 0;
    for _ in 0..4 {
        deleted += meta.collect().expect("collect runs").deleted;
    }
    assert!(deleted > 0, "the sweep reclaimed nothing at all");

    for digest in &kept {
        assert!(
            meta.store().verify(digest).is_ok(),
            "the subset's own object {digest} was collected"
        );
    }
    for digest in &omitted {
        assert!(
            meta.store().verify(digest).is_err(),
            "{digest} is named by no root and survived the sweep"
        );
    }
}
