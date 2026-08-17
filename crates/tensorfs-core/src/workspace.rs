//! Transactional workspace/snapshot metadata over the immutable object store.
//!
//! One Turso WAL database owns inodes, dirents, ordered object maps,
//! workspace heads, sealed snapshots, leases, and GC quarantine. One durable
//! transaction advances a workspace generation, and only after every newly
//! referenced object has been verified resident, so recovery always exposes
//! the previous generation or the new one, never a hybrid. GC marks
//! unreferenced objects into quarantine and deletes them only after they have
//! stayed unreferenced for two full collection epochs.
//!
//! This layer is deliberately below any filesystem surface: callers admit
//! bytes to the [`crate::store::ObjectStore`] themselves and record metadata
//! here at digest/length granularity.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use crate::workspace_db::{Connection, OptionalExtension, TransactionBehavior, params};
use thiserror::Error;

use crate::layout::{Layout, LayoutError};
use crate::object::ObjectDigest;
use crate::planner::{ByteSource as _, PlanError, PlannerId, plan};
use crate::store::{ObjectStore, StoreError};
use crate::tfm1::{
    Entry, FileRecord, HeadTree, Snapshot, SnapshotBuilder, SnapshotId, Tfm1Error, TreeEntry,
    case_fold, decode, validate_path, validate_record_run, validate_symlink_target,
};
use crate::workspace_source::RecordsSource;

const KIND_DIRECTORY: i64 = 1;
const KIND_FILE: i64 = 2;
const KIND_SYMLINK: i64 = 3;

/// What a lease pins, as stored in `leases.kind`.
///
/// The two shapes are disjoint and each has exactly one constructor, so a
/// lease can never claim a lifetime it does not hold: an [`Unlinked`] lease
/// names an inode and keeps its object map alive, a [`PendingSync`] lease
/// names the digests an in-flight transfer is reading. A merely open or
/// mmapped file needs neither — its dirent is already a root — so there is
/// no variant for it.
///
/// [`Unlinked`]: LeaseKind::Unlinked
/// [`PendingSync`]: LeaseKind::PendingSync
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeaseKind {
    Unlinked,
    PendingSync,
}

impl LeaseKind {
    const fn to_row(self) -> i64 {
        match self {
            Self::Unlinked => 2,
            Self::PendingSync => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeaseId(i64);

#[derive(Clone, Debug)]
pub enum Mutation {
    Mkdir {
        path: String,
    },
    CreateFile {
        path: String,
        executable: bool,
        planner: PlannerId,
        records: Vec<FileRecord>,
    },
    SetRecords {
        path: String,
        records: Vec<FileRecord>,
    },
    Truncate {
        path: String,
        length: u64,
    },
    Rename {
        from: String,
        to: String,
    },
    Unlink {
        path: String,
    },
    Rmdir {
        path: String,
    },
    Symlink {
        path: String,
        target: String,
    },
    Hardlink {
        path: String,
        target: String,
    },
    SetExecutable {
        path: String,
        executable: bool,
    },
}

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("workspace metadata I/O failed")]
    Db(#[from] crate::workspace_db::Error),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("path fact is not canonical: {0}")]
    Tfm1(#[from] Tfm1Error),
    #[error("workspace {0:?} does not exist")]
    UnknownWorkspace(String),
    #[error("workspace {0:?} already exists")]
    WorkspaceExists(String),
    #[error("path {0:?} does not exist")]
    Missing(String),
    #[error("path {0:?} already exists")]
    Exists(String),
    #[error("path {0:?} is not a directory")]
    NotADirectory(String),
    #[error("path {0:?} is not a regular file")]
    NotAFile(String),
    #[error("directory {0:?} is not empty")]
    NotEmpty(String),
    #[error("a sibling of {0:?} collides under case folding")]
    CaseFoldCollision(String),
    #[error("truncating {path:?} at {length} would cut a data record")]
    MidRecordTruncate { path: String, length: u64 },
    #[error("referenced object {digest} is not verified resident")]
    MissingObject { digest: ObjectDigest },
    #[error("snapshot {0} is unknown")]
    UnknownSnapshot(SnapshotId),
    #[error("stored snapshot bytes no longer hash to their id")]
    SnapshotCorrupt,
    #[error("lease {0:?} is unknown")]
    UnknownLease(i64),
    #[error("rename would move {0:?} into its own subtree")]
    RenameIntoSelf(String),
    #[error("seal-time planning failed")]
    Plan(#[from] PlanError),
    #[error("the workspace kept changing during seal; retries exhausted")]
    SealContention,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcReport {
    pub roots: u64,
    pub resident: u64,
    pub newly_quarantined: u64,
    pub rescued: u64,
    pub deleted: u64,
    pub bytes_deleted: u64,
    pub epoch: u64,
}

/// The one metadata authority beside an object store root.
pub struct WorkspaceStore {
    store: ObjectStore,
    connection: Mutex<Connection>,
}

impl WorkspaceStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        let root = root.as_ref();
        let store = ObjectStore::open(root)?;
        let connection = Connection::open(root.join("metadata.sqlite3"))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.busy_timeout(std::time::Duration::from_secs(30))?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS workspaces (
                 id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL UNIQUE,
                 head_generation INTEGER NOT NULL,
                 root_inode INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS inodes (
                 id INTEGER PRIMARY KEY,
                 workspace_id INTEGER NOT NULL
                     REFERENCES workspaces(id) ON DELETE CASCADE,
                 kind INTEGER NOT NULL,
                 executable INTEGER NOT NULL DEFAULT 0,
                 planner INTEGER,
                 symlink_target TEXT
             );
             CREATE TABLE IF NOT EXISTS dirents (
                 parent_inode INTEGER NOT NULL REFERENCES inodes(id) ON DELETE CASCADE,
                 name TEXT NOT NULL,
                 child_inode INTEGER NOT NULL REFERENCES inodes(id),
                 PRIMARY KEY (parent_inode, name)
             );
             CREATE TABLE IF NOT EXISTS object_maps (
                 inode INTEGER NOT NULL REFERENCES inodes(id) ON DELETE CASCADE,
                 ordinal INTEGER NOT NULL,
                 hole INTEGER NOT NULL,
                 digest BLOB,
                 length INTEGER NOT NULL,
                 PRIMARY KEY (inode, ordinal)
             );
             -- The roots index. Manifest BYTES live in the object store at
             -- their own id (a TFM1 id is the SHA-256 of its bytes), so this
             -- row is a root and a generation stamp, never a second copy.
             CREATE TABLE IF NOT EXISTS snapshots (
                 id BLOB PRIMARY KEY,
                 sealed_generation INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS leases (
                 id INTEGER PRIMARY KEY,
                 kind INTEGER NOT NULL,
                 holder TEXT NOT NULL,
                 workspace_id INTEGER REFERENCES workspaces(id) ON DELETE CASCADE,
                 inode INTEGER REFERENCES inodes(id)
             );
             CREATE TABLE IF NOT EXISTS lease_objects (
                 lease INTEGER NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
                 digest BLOB NOT NULL,
                 PRIMARY KEY (lease, digest)
             );
             CREATE TABLE IF NOT EXISTS gc_state (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 epoch INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS gc_quarantine (
                 digest BLOB PRIMARY KEY,
                 epoch INTEGER NOT NULL
             );
             INSERT OR IGNORE INTO gc_state (id, epoch) VALUES (1, 0);",
        )?;
        Ok(Self {
            store,
            connection: Mutex::new(connection),
        })
    }

    /// Whether this store root can be opened by several processes at once.
    ///
    /// True everywhere production runs (Linux, ext4/xfs/btrfs/overlayfs). False
    /// only where the platform or filesystem cannot support shared WAL
    /// coordination — Windows' default IO backend, or a network filesystem —
    /// in which case the store still opens, single-process. Cross-process
    /// tests should skip on false rather than fail.
    #[must_use]
    pub fn supports_multiprocess(&self) -> bool {
        self.connection
            .lock()
            .expect("metadata mutex is healthy")
            .is_multiprocess()
    }

    #[must_use]
    pub const fn store(&self) -> &ObjectStore {
        &self.store
    }

    pub fn create_workspace(&self, name: &str) -> Result<(), WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if workspace_row(&tx, name).is_ok() {
            return Err(WorkspaceError::WorkspaceExists(name.to_owned()));
        }
        tx.execute(
            "INSERT INTO workspaces (name, head_generation, root_inode) VALUES (?1, 0, 0)",
            params![name],
        )?;
        let workspace_id = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO inodes (workspace_id, kind) VALUES (?1, ?2)",
            params![workspace_id, KIND_DIRECTORY],
        )?;
        let root_inode = tx.last_insert_rowid();
        tx.execute(
            "UPDATE workspaces SET root_inode = ?1 WHERE id = ?2",
            params![root_inode, workspace_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Clones a sealed snapshot's decoded tree into a fresh workspace. The
    /// stored TFM1 bytes are verified against their id before use.
    pub fn create_workspace_from_snapshot(
        &self,
        name: &str,
        snapshot: &SnapshotId,
    ) -> Result<(), WorkspaceError> {
        let decoded = self.load_snapshot(snapshot)?;
        self.create_workspace(name)?;
        let mut mutations = Vec::new();
        for (path, entry) in decoded.entries() {
            match entry {
                Entry::Directory => mutations.push(Mutation::Mkdir { path: path.clone() }),
                Entry::File { executable, body } => mutations.push(Mutation::CreateFile {
                    path: path.clone(),
                    executable: *executable,
                    planner: body.planner_id(),
                    records: body.records().into_owned(),
                }),
                Entry::Symlink { target } => mutations.push(Mutation::Symlink {
                    path: path.clone(),
                    target: target.clone(),
                }),
                Entry::Hardlink { ordinal } => {
                    let target = decoded
                        .ordinal_path(*ordinal)
                        .expect("decoded snapshots resolve every ordinal")
                        .to_owned();
                    mutations.push(Mutation::Hardlink {
                        path: path.clone(),
                        target,
                    });
                }
            }
        }
        self.commit_generation(name, &mutations)?;
        Ok(())
    }

    pub fn delete_workspace(&self, name: &str) -> Result<(), WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let workspace = workspace_row(&tx, name)?;
        tx.execute(
            "DELETE FROM dirents WHERE parent_inode IN
                 (SELECT id FROM inodes WHERE workspace_id = ?1)",
            params![workspace.id],
        )?;
        tx.execute(
            "DELETE FROM workspaces WHERE id = ?1",
            params![workspace.id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn head_generation(&self, name: &str) -> Result<u64, WorkspaceError> {
        let connection = self.connection.lock().expect("metadata mutex is healthy");
        let workspace = workspace_row(&connection, name)?;
        Ok(workspace.head_generation)
    }

    /// Applies one batch of tree mutations as one durable generation advance.
    ///
    /// Every digest the batch newly references is rehash-verified resident in
    /// the object store before the WAL commit; an unverifiable object refuses
    /// the whole batch and the head does not move.
    pub fn commit_generation(
        &self,
        name: &str,
        mutations: &[Mutation],
    ) -> Result<u64, WorkspaceError> {
        let mut referenced = HashSet::new();
        for mutation in mutations {
            match mutation {
                Mutation::CreateFile { records, .. } | Mutation::SetRecords { records, .. } => {
                    // Staging-domain validation: structural run rules only.
                    // The 64 MiB grid cap is a tensor-body rule — an adopted
                    // blob is ONE data record of any size.
                    let logical_size = records_length(records)?;
                    validate_record_run(records, logical_size)?;
                    for record in records {
                        if let FileRecord::Data { digest, .. } = record {
                            referenced.insert(*digest);
                        }
                    }
                }
                _ => {}
            }
        }
        for digest in &referenced {
            self.store
                .verify(digest)
                .map_err(|_| WorkspaceError::MissingObject { digest: *digest })?;
        }
        #[cfg(test)]
        AFTER_VERIFY_SEAM.fire();

        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Re-check presence inside the metadata transaction: `verify` above
        // ran outside it, and a GC pass that marked these still-unreferenced
        // objects epochs ago must not be able to delete them between the
        // rehash and this commit. The transaction serializes against
        // `collect`, so a stat here closes that window completely.
        for digest in &referenced {
            if !self.store.exists(digest) {
                return Err(WorkspaceError::MissingObject { digest: *digest });
            }
        }
        let workspace = workspace_row(&tx, name)?;
        for mutation in mutations {
            apply_mutation(&tx, &workspace, mutation)?;
        }
        let next = workspace.head_generation + 1;
        tx.execute(
            "UPDATE workspaces SET head_generation = ?1 WHERE id = ?2",
            params![next as i64, workspace.id],
        )?;
        tx.commit()?;
        Ok(next)
    }

    /// Builds (without storing) the committed head tree in the staging
    /// vocabulary. This is the one tree-read seam a filesystem surface reads
    /// through (the shelved daemon mounted via it). It is deliberately NOT a
    /// canonical snapshot value: a mid-write non-tensor file is a grid record
    /// run, a shape the recordless `blob-v1` manifest body cannot express —
    /// sealing is what canonicalizes it.
    pub fn head_tree(&self, name: &str) -> Result<HeadTree, WorkspaceError> {
        let connection = self.connection.lock().expect("metadata mutex is healthy");
        let workspace = workspace_row(&connection, name)?;
        build_tree(&connection, &workspace)
    }

    /// Seals the committed head tree as one canonical TFM1 snapshot, stores
    /// its exact bytes, and returns the identity.
    ///
    /// Before sealing, the closed planner registry re-establishes semantic
    /// boundaries: each pure-data file is planned through a records-backed
    /// byte source (bounded header reads only). Tensor containers whose
    /// planned regions differ from the committed record boundaries are
    /// re-sliced — exact-range records keep their digests without a payload
    /// read, every other region is read once and admitted. Everything else
    /// becomes ONE whole blob: a non-tensor file already committed as its
    /// single whole-file object is reused by `(offset, length)` with zero
    /// re-hashing, and any other shape (a grid run, a sparse run — a hole in
    /// a blob is unrepresentable, so its zeros materialize) is streamed once
    /// through the verifying writer into one object of any size. The
    /// re-boundaried records and the sealed snapshot land in one transaction,
    /// so the workspace head and the snapshot always agree. Sparse files with
    /// tensor provenance keep their committed records unchanged — the tensor
    /// grammar keeps holes, and re-planning would materialize them.
    pub fn seal_snapshot(
        &self,
        name: &str,
        parent: Option<SnapshotId>,
    ) -> Result<SnapshotId, WorkspaceError> {
        for _attempt in 0..3 {
            // Phase A: capture the committed tree and its generation.
            let (generation, tree) = {
                let connection = self.connection.lock().expect("metadata mutex is healthy");
                let workspace = workspace_row(&connection, name)?;
                (
                    workspace.head_generation,
                    build_tree(&connection, &workspace)?,
                )
            };

            // Phase B: plan and admit outside the lock — region reads and
            // object admission must not hold the metadata store.
            let mut jobs = Vec::new();
            for (path, entry) in tree.entries() {
                if let TreeEntry::File {
                    planner, records, ..
                } = entry
                    && let Some(job) = self.plan_seal_job(path, *planner, records)?
                {
                    jobs.push(job);
                }
            }
            #[cfg(test)]
            AFTER_SEAL_PLAN_SEAM.fire();

            // Phase C: one transaction re-checks the generation, applies the
            // re-boundary, and seals. A moved head invalidates phase B's
            // analysis, so it retries from the top.
            let mut connection = self.connection.lock().expect("metadata mutex is healthy");
            let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let workspace = workspace_row(&tx, name)?;
            if workspace.head_generation != generation {
                drop(tx);
                continue;
            }
            for job in &jobs {
                for digest in &job.new_digests {
                    if !self.store.exists(digest) {
                        return Err(WorkspaceError::MissingObject { digest: *digest });
                    }
                }
            }
            for job in &jobs {
                if job.update_records {
                    apply_mutation(
                        &tx,
                        &workspace,
                        &Mutation::SetRecords {
                            path: job.path.clone(),
                            records: job.records.clone(),
                        },
                    )?;
                }
                let inode = resolve_path(&tx, &workspace, &job.path)?
                    .ok_or_else(|| WorkspaceError::Missing(job.path.clone()))?;
                tx.execute(
                    "UPDATE inodes SET planner = ?1 WHERE id = ?2",
                    params![planner_to_row(job.planner), inode.id],
                )?;
            }
            let sealed_generation = if jobs.is_empty() {
                workspace.head_generation
            } else {
                let next = workspace.head_generation + 1;
                tx.execute(
                    "UPDATE workspaces SET head_generation = ?1 WHERE id = ?2",
                    params![next as i64, workspace.id],
                )?;
                next
            };
            let workspace = WorkspaceRow {
                head_generation: sealed_generation,
                ..workspace
            };
            let snapshot = build_snapshot(&tx, &workspace, parent)?;
            // The manifest is an object like any other, admitted at its own
            // id through the verifying writer. Admitting before the row lands
            // means a crash between them leaves an unreferenced object GC
            // reclaims, never a root naming bytes that were never written.
            let admitted = self.store.put_bytes(&snapshot.to_bytes())?;
            let id = snapshot.snapshot_id();
            debug_assert_eq!(admitted.digest().as_bytes(), id.as_bytes());
            tx.execute(
                "INSERT OR IGNORE INTO snapshots (id, sealed_generation) VALUES (?1, ?2)",
                params![id.as_bytes().as_slice(), sealed_generation as i64],
            )?;
            tx.commit()?;
            return Ok(id);
        }
        Err(WorkspaceError::SealContention)
    }

    /// Computes one file's seal-time re-boundary work, or `None` when the
    /// committed records already are the canonical form.
    fn plan_seal_job(
        &self,
        path: &str,
        stored_planner: PlannerId,
        records: &[FileRecord],
    ) -> Result<Option<SealJob>, WorkspaceError> {
        if records.is_empty() {
            return Ok(None);
        }
        let has_holes = records
            .iter()
            .any(|record| matches!(record, FileRecord::Hole { .. }));
        if has_holes && stored_planner.tensor_format().is_some() {
            // Tensor bodies keep hole records in the grammar, and re-planning
            // a sparse tensor file would materialize its sparseness. It seals
            // with its committed records unchanged.
            return Ok(None);
        }
        let source = RecordsSource::new(&self.store, records);
        if has_holes {
            // A hole in a blob is unrepresentable, so a sparse non-tensor
            // file has exactly one canonical form: the whole byte stream —
            // zeros included, read through the record source — as one object.
            return Ok(Some(self.blob_seal_job(path, &source)?));
        }
        let planned = plan(&source)?;
        if planned.planner() == PlannerId::BlobV1 {
            // Already committed as its single whole-file object: reuse by
            // (offset, length) with ZERO re-hashing — the blob extension of
            // the #61 double-hash fence. Anything else re-streams once.
            if let [FileRecord::Data { .. }] = records {
                if stored_planner == PlannerId::BlobV1 {
                    return Ok(None);
                }
                return Ok(Some(SealJob {
                    path: path.to_owned(),
                    planner: PlannerId::BlobV1,
                    records: records.to_vec(),
                    update_records: false,
                    new_digests: Vec::new(),
                }));
            }
            return Ok(Some(self.blob_seal_job(path, &source)?));
        }

        let mut committed = Vec::with_capacity(records.len());
        let mut position = 0_u64;
        for record in records {
            let FileRecord::Data { digest, length } = record else {
                unreachable!("hole records were excluded above");
            };
            committed.push((position, *length, *digest));
            position += *length;
        }
        let boundaries_match = committed.len() == planned.regions().len()
            && committed
                .iter()
                .zip(planned.regions())
                .all(|((start, length, _), region)| {
                    *start == region.offset() && *length == region.length()
                });
        if boundaries_match {
            if planned.planner() == stored_planner {
                return Ok(None);
            }
            return Ok(Some(SealJob {
                path: path.to_owned(),
                planner: planned.planner(),
                records: records.to_vec(),
                update_records: false,
                new_digests: Vec::new(),
            }));
        }

        let by_range: HashMap<(u64, u64), ObjectDigest> = committed
            .iter()
            .map(|(start, length, digest)| ((*start, *length), *digest))
            .collect();
        // A region the committed head already covers is reused as-is; only
        // the rest are read and admitted, and those go through the store's
        // bounded concurrent path so a re-plan of a many-slot tensor is not
        // one SHA-256 core's worth of wall-clock.
        let mut pending_regions = Vec::new();
        let mut pending_slots = Vec::new();
        for (slot, region) in planned.regions().iter().enumerate() {
            if !by_range.contains_key(&(region.offset(), region.length())) {
                pending_regions.push(*region);
                pending_slots.push(slot);
            }
        }
        let admitted = self.store.admit_regions(&source, &pending_regions)?;

        let mut new_digests = Vec::with_capacity(admitted.len());
        let mut fresh: HashMap<usize, ObjectDigest> = HashMap::with_capacity(admitted.len());
        for (slot, object) in pending_slots.iter().zip(&admitted) {
            fresh.insert(*slot, object.digest());
            new_digests.push(object.digest());
        }

        let mut new_records = Vec::with_capacity(planned.regions().len());
        for (slot, region) in planned.regions().iter().enumerate() {
            let digest = by_range
                .get(&(region.offset(), region.length()))
                .copied()
                .or_else(|| fresh.get(&slot).copied())
                .expect("every planned region is either already committed or freshly admitted");
            new_records.push(FileRecord::Data {
                digest,
                length: region.length(),
            });
        }
        Ok(Some(SealJob {
            path: path.to_owned(),
            planner: planned.planner(),
            records: new_records,
            update_records: true,
            new_digests,
        }))
    }

    /// Streams one file's whole committed byte run — holes read as zeros —
    /// through the verifying writer as ONE object of any size, and returns
    /// the single-record seal job carrying its whole-file digest.
    fn blob_seal_job(
        &self,
        path: &str,
        source: &RecordsSource<&ObjectStore>,
    ) -> Result<SealJob, WorkspaceError> {
        let admitted = self.store.admit_source_range(source, 0, source.len())?;
        Ok(SealJob {
            path: path.to_owned(),
            planner: PlannerId::BlobV1,
            records: vec![FileRecord::Data {
                digest: admitted.digest(),
                length: admitted.length(),
            }],
            update_records: true,
            new_digests: vec![admitted.digest()],
        })
    }

    /// Loads one sealed snapshot: the row is the root, the bytes are the
    /// immutable object it protects.
    ///
    /// Reading does not re-hash. The manifest object was verified at
    /// admission and the digest path is its identity; a read-path rehash is
    /// exactly the cost the admission-time ruling exists to avoid.
    pub fn load_snapshot(&self, id: &SnapshotId) -> Result<Snapshot, WorkspaceError> {
        let connection = self.connection.lock().expect("metadata mutex is healthy");
        self.read_rooted_snapshot(&connection, id)
    }

    /// Pins the root row and reads the blob it protects, inside whatever
    /// transaction the caller holds. `collect` serializes against that
    /// transaction, so the manifest object cannot be reclaimed between the
    /// two steps.
    fn read_rooted_snapshot(
        &self,
        connection: &Connection,
        id: &SnapshotId,
    ) -> Result<Snapshot, WorkspaceError> {
        let rooted: Option<i64> = connection
            .query_row(
                "SELECT sealed_generation FROM snapshots WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| row.get(0),
            )
            .optional()?;
        if rooted.is_none() {
            return Err(WorkspaceError::UnknownSnapshot(*id));
        }
        self.read_manifest_object(id)
    }

    /// Decodes the manifest object stored at a snapshot's own id.
    ///
    /// A manifest is the root of the reachability graph, so its id check
    /// stays: bounded by the manifest, never on a data read path, and the one
    /// thing that keeps an unreadable root from silently un-rooting every
    /// object it names. Data objects are still never rehashed on read.
    fn read_manifest_object(&self, id: &SnapshotId) -> Result<Snapshot, WorkspaceError> {
        let digest = ObjectDigest::from_bytes(*id.as_bytes());
        let mut file = self
            .store
            .open_object(&digest)
            .map_err(|_| WorkspaceError::SnapshotCorrupt)?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut bytes).map_err(StoreError::from)?;
        if SnapshotId::of(&bytes) != *id {
            return Err(WorkspaceError::SnapshotCorrupt);
        }
        Ok(decode(&bytes)?)
    }

    /// The projected trees and refs beside this store.
    #[must_use]
    pub const fn layout(&self) -> Layout<'_> {
        Layout::new(&self.store)
    }

    /// Projects one rooted snapshot as `snapshots/<id>/…` and returns the
    /// tree path. Zero bytes are copied: directories are directories, blob
    /// files are relative symlinks into `objects/`, tensor files are pointer
    /// stubs. The tree is disposable and pins nothing.
    pub fn project_snapshot(&self, id: &SnapshotId) -> Result<std::path::PathBuf, WorkspaceError> {
        let snapshot = self.load_snapshot(id)?;
        Ok(self.layout().project(&snapshot)?)
    }

    /// Adopts one canonical TFM1 snapshot fetched from a remote. The bytes
    /// are strictly decoded (every grammar and tree rule refuses), every
    /// referenced data object must already be resident — pull admits objects
    /// through the verifying writer before adopting, so a stat here is the
    /// cheap re-check, not the trust boundary — and the blob is stored
    /// idempotently under its self-verifying id.
    pub fn adopt_snapshot(&self, bytes: &[u8]) -> Result<SnapshotId, WorkspaceError> {
        let snapshot = decode(bytes)?;
        for (_path, entry) in snapshot.entries() {
            if let Entry::File { body, .. } = entry {
                for record in body.records().iter() {
                    if let FileRecord::Data { digest, .. } = record
                        && !self.store.exists(digest)
                    {
                        return Err(WorkspaceError::MissingObject { digest: *digest });
                    }
                }
            }
        }
        let id = snapshot.snapshot_id();
        self.store.put_bytes(bytes)?;
        let connection = self.connection.lock().expect("metadata mutex is healthy");
        connection.execute(
            "INSERT OR IGNORE INTO snapshots (id, sealed_generation) VALUES (?1, 0)",
            params![id.as_bytes().as_slice()],
        )?;
        Ok(id)
    }

    /// Deletes one snapshot root, then its projected tree, then every ref
    /// that pointed at it — in that order.
    ///
    /// A tree without a root is inert garbage the next scrub removes, while a
    /// root without a tree is merely unprojected, so a crash between the steps
    /// costs nothing. The manifest object and every object it named become
    /// unreferenced and are GC's problem, not this call's.
    pub fn delete_snapshot(&self, id: &SnapshotId) -> Result<(), WorkspaceError> {
        {
            let connection = self.connection.lock().expect("metadata mutex is healthy");
            let deleted = connection.execute(
                "DELETE FROM snapshots WHERE id = ?1",
                params![id.as_bytes().as_slice()],
            )?;
            if deleted == 0 {
                return Err(WorkspaceError::UnknownSnapshot(*id));
            }
        }
        let layout = self.layout();
        layout.remove_tree(id)?;
        for name in layout.refs_to(id)? {
            layout.remove_ref(&name)?;
        }
        Ok(())
    }

    /// Pins one path's inode (and therefore its object map) for the holder,
    /// so a file unlinked while still open keeps its bytes until the last
    /// descriptor closes.
    pub fn acquire_unlinked_lease(
        &self,
        workspace: &str,
        path: &str,
        holder: &str,
    ) -> Result<LeaseId, WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = workspace_row(&tx, workspace)?;
        let inode = resolve_path(&tx, &row, path)?
            .ok_or_else(|| WorkspaceError::Missing(path.to_owned()))?;
        tx.execute(
            "INSERT INTO leases (kind, holder, workspace_id, inode) VALUES (?1, ?2, ?3, ?4)",
            params![LeaseKind::Unlinked.to_row(), holder, row.id, inode.id],
        )?;
        let lease = LeaseId(tx.last_insert_rowid());
        tx.commit()?;
        Ok(lease)
    }

    /// Pins every object one sealed snapshot names for the exact lifetime of
    /// an in-flight transfer, and hands back the snapshot the caller will
    /// send.
    ///
    /// Until a push lands, the `snapshots` row is often the ONLY root its
    /// objects have — the workspace has usually moved on. A `delete_snapshot`
    /// racing the transfer would strip that root and let [`collect`] take the
    /// bytes still being streamed. The pin is a first-class root instead of a
    /// second copy of the closure, so nothing is duplicated and nothing
    /// outlives the release.
    ///
    /// Reading the manifest and taking the pin happen in ONE transaction, so
    /// there is no window between them: a concurrent delete either wins
    /// outright (`UnknownSnapshot`, before a byte moves) or loses until the
    /// lease is released.
    ///
    /// [`collect`]: Self::collect
    pub fn acquire_pending_sync_lease(
        &self,
        snapshot: &SnapshotId,
        holder: &str,
    ) -> Result<(LeaseId, Snapshot), WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let decoded = self.read_rooted_snapshot(&tx, snapshot)?;
        tx.execute(
            "INSERT INTO leases (kind, holder, workspace_id, inode)
                 VALUES (?1, ?2, NULL, NULL)",
            params![LeaseKind::PendingSync.to_row(), holder],
        )?;
        let lease = LeaseId(tx.last_insert_rowid());
        // The manifest object is pinned with the closure it names: a delete
        // racing the transfer must not take the bytes being streamed, and the
        // manifest is now one of those bytes.
        tx.execute(
            "INSERT OR IGNORE INTO lease_objects (lease, digest) VALUES (?1, ?2)",
            params![lease.0, snapshot.as_bytes().as_slice()],
        )?;
        for (_path, entry) in decoded.entries() {
            if let Entry::File { body, .. } = entry {
                for record in body.records().iter() {
                    if let FileRecord::Data { digest, .. } = record {
                        tx.execute(
                            "INSERT OR IGNORE INTO lease_objects (lease, digest)
                                 VALUES (?1, ?2)",
                            params![lease.0, digest.as_bytes().as_slice()],
                        )?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok((lease, decoded))
    }

    /// Releases one lease and everything it pinned. An orphaned inode whose
    /// last lease is released is removed together with its object map, and a
    /// pending-sync lease's pinned digests stop being roots in the same
    /// transaction.
    pub fn release_lease(&self, lease: LeaseId) -> Result<(), WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let inode: Option<Option<i64>> = tx
            .query_row(
                "SELECT inode FROM leases WHERE id = ?1",
                params![lease.0],
                |row| row.get(0),
            )
            .optional()?;
        let Some(inode) = inode else {
            return Err(WorkspaceError::UnknownLease(lease.0));
        };
        // Children first: the pin must not be able to outlive its lease even
        // where the engine declines to cascade.
        tx.execute(
            "DELETE FROM lease_objects WHERE lease = ?1",
            params![lease.0],
        )?;
        tx.execute("DELETE FROM leases WHERE id = ?1", params![lease.0])?;
        if let Some(inode) = inode {
            drop_inode_if_orphaned(&tx, inode)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// One collection pass: advances the epoch, marks unreferenced resident
    /// objects, rescues re-referenced ones, and deletes only objects that
    /// have stayed quarantined for at least two full epochs. The whole pass
    /// holds the metadata write transaction, so the root set cannot move
    /// under it and a live object can never enter the delete set.
    pub fn collect(&self) -> Result<GcReport, WorkspaceError> {
        let mut connection = self.connection.lock().expect("metadata mutex is healthy");
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let epoch: i64 = tx.query_row("SELECT epoch FROM gc_state WHERE id = 1", [], |row| {
            row.get(0)
        })?;
        let epoch = epoch + 1;
        tx.execute(
            "UPDATE gc_state SET epoch = ?1 WHERE id = 1",
            params![epoch],
        )?;

        let mut roots: HashSet<ObjectDigest> = HashSet::new();
        {
            let mut map_digests =
                tx.prepare("SELECT digest FROM object_maps WHERE digest IS NOT NULL")?;
            let mut rows = map_digests.query([])?;
            while let Some(row) = rows.next()? {
                roots.insert(digest_from_row(row.get::<_, Vec<u8>>(0)?)?);
            }
        }
        {
            let mut ids = Vec::new();
            {
                let mut rooted = tx.prepare("SELECT id FROM snapshots")?;
                let mut rows = rooted.query([])?;
                while let Some(row) = rows.next()? {
                    ids.push(digest_from_row(row.get::<_, Vec<u8>>(0)?)?);
                }
            }
            for id in ids {
                // A manifest is an object at its own id, so a root row roots
                // the manifest itself as well as everything it names.
                roots.insert(id);
                let snapshot =
                    self.read_manifest_object(&SnapshotId::from_bytes(*id.as_bytes()))?;
                for (_, entry) in snapshot.entries() {
                    if let Entry::File { body, .. } = entry {
                        for record in body.records().iter() {
                            if let FileRecord::Data { digest, .. } = record {
                                roots.insert(*digest);
                            }
                        }
                    }
                }
            }
        }

        // Leases are the third root source, and the only one that is not a
        // committed fact: a pending-sync lease pins the closure a transfer is
        // still reading, whose snapshot row may already have been deleted out
        // from under it. Read inside the same transaction as the other two,
        // so the whole root set is one consistent instant.
        {
            let mut pinned = tx.prepare("SELECT digest FROM lease_objects")?;
            let mut rows = pinned.query([])?;
            while let Some(row) = rows.next()? {
                roots.insert(digest_from_row(row.get::<_, Vec<u8>>(0)?)?);
            }
        }

        let mut resident: Vec<(ObjectDigest, u64)> = Vec::new();
        self.store
            .each_object(|digest, length| resident.push((digest, length)))?;

        let mut report = GcReport {
            roots: roots.len() as u64,
            resident: resident.len() as u64,
            epoch: epoch as u64,
            ..GcReport::default()
        };

        let resident_set: HashSet<ObjectDigest> =
            resident.iter().map(|(digest, _)| *digest).collect();
        let mut lengths: HashMap<ObjectDigest, u64> = resident.into_iter().collect();

        // Rescue everything referenced again, and drop rows whose object is
        // already gone; both make the quarantine reflect reality.
        let quarantined: Vec<(ObjectDigest, i64)> = {
            let mut rows_statement = tx.prepare("SELECT digest, epoch FROM gc_quarantine")?;
            let mut rows = rows_statement.query([])?;
            let mut collected = Vec::new();
            while let Some(row) = rows.next()? {
                collected.push((
                    digest_from_row(row.get::<_, Vec<u8>>(0)?)?,
                    row.get::<_, i64>(1)?,
                ));
            }
            collected
        };
        for (digest, marked) in &quarantined {
            if roots.contains(digest) || !resident_set.contains(digest) {
                tx.execute(
                    "DELETE FROM gc_quarantine WHERE digest = ?1",
                    params![digest.as_bytes().as_slice()],
                )?;
                if roots.contains(digest) {
                    report.rescued += 1;
                }
                continue;
            }
            if *marked <= epoch - 2 {
                if self.store.remove_object(digest)? {
                    report.deleted += 1;
                    report.bytes_deleted += lengths.remove(digest).unwrap_or(0);
                }
                tx.execute(
                    "DELETE FROM gc_quarantine WHERE digest = ?1",
                    params![digest.as_bytes().as_slice()],
                )?;
            }
        }

        let already: HashSet<ObjectDigest> =
            quarantined.iter().map(|(digest, _)| *digest).collect();
        for digest in resident_set {
            if !roots.contains(&digest) && !already.contains(&digest) {
                tx.execute(
                    "INSERT OR IGNORE INTO gc_quarantine (digest, epoch) VALUES (?1, ?2)",
                    params![digest.as_bytes().as_slice(), epoch],
                )?;
                report.newly_quarantined += 1;
            }
        }

        tx.commit()?;
        Ok(report)
    }

    /// Rebuilds every workspace named in `pairs` from its snapshot in a fresh
    /// metadata database, proving snapshots alone carry the identity.
    pub fn rebuild_from_snapshot(
        &self,
        name: &str,
        snapshot: &SnapshotId,
    ) -> Result<(), WorkspaceError> {
        self.create_workspace_from_snapshot(name, snapshot)
    }
}

struct WorkspaceRow {
    id: i64,
    head_generation: u64,
    root_inode: i64,
}

/// One file's computed seal-time re-boundary: the canonical records, the
/// planner provenance they came from, and the digests admitted for them.
struct SealJob {
    path: String,
    planner: PlannerId,
    records: Vec<FileRecord>,
    update_records: bool,
    new_digests: Vec<ObjectDigest>,
}

struct ResolvedInode {
    id: i64,
    kind: i64,
}

fn workspace_row(connection: &Connection, name: &str) -> Result<WorkspaceRow, WorkspaceError> {
    connection
        .query_row(
            "SELECT id, head_generation, root_inode FROM workspaces WHERE name = ?1",
            params![name],
            |row| {
                Ok(WorkspaceRow {
                    id: row.get(0)?,
                    head_generation: row.get::<_, i64>(1)? as u64,
                    root_inode: row.get(2)?,
                })
            },
        )
        .optional()?
        .ok_or_else(|| WorkspaceError::UnknownWorkspace(name.to_owned()))
}

fn digest_from_row(bytes: Vec<u8>) -> Result<ObjectDigest, WorkspaceError> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| WorkspaceError::SnapshotCorrupt)?;
    Ok(ObjectDigest::from_bytes(bytes))
}

fn records_length(records: &[FileRecord]) -> Result<u64, Tfm1Error> {
    let mut sum = 0_u64;
    for record in records {
        let length = match record {
            FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
        };
        sum = sum
            .checked_add(length)
            .ok_or(Tfm1Error::LengthSumMismatch)?;
    }
    Ok(sum)
}

fn split_parent(path: &str) -> (Option<&str>, &str) {
    match path.rsplit_once('/') {
        Some((parent, name)) => (Some(parent), name),
        None => (None, path),
    }
}

fn resolve_path(
    connection: &Connection,
    workspace: &WorkspaceRow,
    path: &str,
) -> Result<Option<ResolvedInode>, WorkspaceError> {
    if path.is_empty() {
        return Ok(Some(ResolvedInode {
            id: workspace.root_inode,
            kind: KIND_DIRECTORY,
        }));
    }
    validate_path(path)?;
    let mut current = ResolvedInode {
        id: workspace.root_inode,
        kind: KIND_DIRECTORY,
    };
    for segment in path.split('/') {
        if current.kind != KIND_DIRECTORY {
            return Ok(None);
        }
        let child: Option<(i64, i64)> = connection
            .query_row(
                "SELECT inodes.id, inodes.kind FROM dirents
                     JOIN inodes ON inodes.id = dirents.child_inode
                     WHERE dirents.parent_inode = ?1 AND dirents.name = ?2",
                params![current.id, segment],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        match child {
            Some((id, kind)) => current = ResolvedInode { id, kind },
            None => return Ok(None),
        }
    }
    Ok(Some(current))
}

/// Resolves the parent directory of `path` and validates the leaf name,
/// including sibling case-fold uniqueness.
fn resolve_new_leaf(
    connection: &Connection,
    workspace: &WorkspaceRow,
    path: &str,
) -> Result<(i64, String), WorkspaceError> {
    validate_path(path)?;
    let (parent, name) = split_parent(path);
    let parent_inode = match parent {
        None => workspace.root_inode,
        Some(parent_path) => {
            let resolved = resolve_path(connection, workspace, parent_path)?
                .ok_or_else(|| WorkspaceError::Missing(parent_path.to_owned()))?;
            if resolved.kind != KIND_DIRECTORY {
                return Err(WorkspaceError::NotADirectory(parent_path.to_owned()));
            }
            resolved.id
        }
    };
    let exact: Option<i64> = connection
        .query_row(
            "SELECT child_inode FROM dirents WHERE parent_inode = ?1 AND name = ?2",
            params![parent_inode, name],
            |row| row.get(0),
        )
        .optional()?;
    if exact.is_some() {
        return Err(WorkspaceError::Exists(path.to_owned()));
    }
    let folded = case_fold(name);
    let mut siblings = connection.prepare("SELECT name FROM dirents WHERE parent_inode = ?1")?;
    let mut rows = siblings.query(params![parent_inode])?;
    while let Some(row) = rows.next()? {
        let sibling: String = row.get(0)?;
        if case_fold(&sibling) == folded {
            return Err(WorkspaceError::CaseFoldCollision(path.to_owned()));
        }
    }
    Ok((parent_inode, name.to_owned()))
}

/// Row values mirror the TFM1 planner tags; 3, the retired raw grid, is gone.
fn planner_to_row(planner: PlannerId) -> i64 {
    match planner {
        PlannerId::SafetensorsV1 => 1,
        PlannerId::GgufV1 => 2,
        PlannerId::BlobV1 => 4,
    }
}

fn planner_from_row(tag: i64) -> PlannerId {
    match tag {
        1 => PlannerId::SafetensorsV1,
        2 => PlannerId::GgufV1,
        _ => PlannerId::BlobV1,
    }
}

fn insert_records(
    connection: &Connection,
    inode: i64,
    records: &[FileRecord],
) -> Result<(), WorkspaceError> {
    connection.execute("DELETE FROM object_maps WHERE inode = ?1", params![inode])?;
    for (ordinal, record) in records.iter().enumerate() {
        match record {
            FileRecord::Data { digest, length } => connection.execute(
                "INSERT INTO object_maps (inode, ordinal, hole, digest, length)
                     VALUES (?1, ?2, 0, ?3, ?4)",
                params![
                    inode,
                    ordinal as i64,
                    digest.as_bytes().as_slice(),
                    *length as i64
                ],
            )?,
            FileRecord::Hole { length } => connection.execute(
                "INSERT INTO object_maps (inode, ordinal, hole, digest, length)
                     VALUES (?1, ?2, 1, NULL, ?3)",
                params![inode, ordinal as i64, *length as i64],
            )?,
        };
    }
    Ok(())
}

fn read_records(connection: &Connection, inode: i64) -> Result<Vec<FileRecord>, WorkspaceError> {
    let mut statement = connection.prepare(
        "SELECT hole, digest, length FROM object_maps WHERE inode = ?1 ORDER BY ordinal",
    )?;
    let mut rows = statement.query(params![inode])?;
    let mut records = Vec::new();
    while let Some(row) = rows.next()? {
        let hole: i64 = row.get(0)?;
        let length = row.get::<_, i64>(2)? as u64;
        if hole == 1 {
            records.push(FileRecord::Hole { length });
        } else {
            records.push(FileRecord::Data {
                digest: digest_from_row(row.get::<_, Vec<u8>>(1)?)?,
                length,
            });
        }
    }
    Ok(records)
}

fn link_count(connection: &Connection, inode: i64) -> Result<i64, WorkspaceError> {
    Ok(connection.query_row(
        "SELECT COUNT(*) FROM dirents WHERE child_inode = ?1",
        params![inode],
        |row| row.get(0),
    )?)
}

fn lease_count(connection: &Connection, inode: i64) -> Result<i64, WorkspaceError> {
    Ok(connection.query_row(
        "SELECT COUNT(*) FROM leases WHERE inode = ?1",
        params![inode],
        |row| row.get(0),
    )?)
}

fn drop_inode_if_orphaned(connection: &Connection, inode: i64) -> Result<(), WorkspaceError> {
    if link_count(connection, inode)? == 0 && lease_count(connection, inode)? == 0 {
        connection.execute("DELETE FROM inodes WHERE id = ?1", params![inode])?;
    }
    Ok(())
}

fn apply_mutation(
    connection: &Connection,
    workspace: &WorkspaceRow,
    mutation: &Mutation,
) -> Result<(), WorkspaceError> {
    match mutation {
        Mutation::Mkdir { path } => {
            let (parent, name) = resolve_new_leaf(connection, workspace, path)?;
            connection.execute(
                "INSERT INTO inodes (workspace_id, kind) VALUES (?1, ?2)",
                params![workspace.id, KIND_DIRECTORY],
            )?;
            let inode = connection.last_insert_rowid();
            connection.execute(
                "INSERT INTO dirents (parent_inode, name, child_inode) VALUES (?1, ?2, ?3)",
                params![parent, name, inode],
            )?;
        }
        Mutation::CreateFile {
            path,
            executable,
            planner,
            records,
        } => {
            let (parent, name) = resolve_new_leaf(connection, workspace, path)?;
            connection.execute(
                "INSERT INTO inodes (workspace_id, kind, executable, planner)
                     VALUES (?1, ?2, ?3, ?4)",
                params![
                    workspace.id,
                    KIND_FILE,
                    i64::from(*executable),
                    planner_to_row(*planner)
                ],
            )?;
            let inode = connection.last_insert_rowid();
            connection.execute(
                "INSERT INTO dirents (parent_inode, name, child_inode) VALUES (?1, ?2, ?3)",
                params![parent, name, inode],
            )?;
            insert_records(connection, inode, records)?;
        }
        Mutation::SetRecords { path, records } => {
            let inode = resolve_path(connection, workspace, path)?
                .ok_or_else(|| WorkspaceError::Missing(path.clone()))?;
            if inode.kind != KIND_FILE {
                return Err(WorkspaceError::NotAFile(path.clone()));
            }
            insert_records(connection, inode.id, records)?;
        }
        Mutation::Truncate { path, length } => {
            let inode = resolve_path(connection, workspace, path)?
                .ok_or_else(|| WorkspaceError::Missing(path.clone()))?;
            if inode.kind != KIND_FILE {
                return Err(WorkspaceError::NotAFile(path.clone()));
            }
            let records = read_records(connection, inode.id)?;
            let truncated = truncate_records(records, *length).ok_or_else(|| {
                WorkspaceError::MidRecordTruncate {
                    path: path.clone(),
                    length: *length,
                }
            })?;
            validate_record_run(&truncated, *length)?;
            insert_records(connection, inode.id, &truncated)?;
        }
        Mutation::Rename { from, to } => {
            let source = resolve_path(connection, workspace, from)?
                .ok_or_else(|| WorkspaceError::Missing(from.clone()))?;
            if source.kind == KIND_DIRECTORY && (to == from || to.starts_with(&format!("{from}/")))
            {
                return Err(WorkspaceError::RenameIntoSelf(from.clone()));
            }
            let (new_parent, new_name) = resolve_new_leaf(connection, workspace, to)?;
            let (old_parent, old_name) = split_parent(from);
            let old_parent_inode = match old_parent {
                None => workspace.root_inode,
                Some(parent_path) => {
                    resolve_path(connection, workspace, parent_path)?
                        .expect("the source parent resolved during source lookup")
                        .id
                }
            };
            connection.execute(
                "UPDATE dirents SET parent_inode = ?1, name = ?2
                     WHERE parent_inode = ?3 AND name = ?4",
                params![new_parent, new_name, old_parent_inode, old_name],
            )?;
        }
        Mutation::Unlink { path } => {
            let inode = resolve_path(connection, workspace, path)?
                .ok_or_else(|| WorkspaceError::Missing(path.clone()))?;
            if inode.kind == KIND_DIRECTORY {
                return Err(WorkspaceError::NotAFile(path.clone()));
            }
            let (parent, name) = split_parent(path);
            let parent_inode = match parent {
                None => workspace.root_inode,
                Some(parent_path) => {
                    resolve_path(connection, workspace, parent_path)?
                        .expect("the parent resolved during path lookup")
                        .id
                }
            };
            connection.execute(
                "DELETE FROM dirents WHERE parent_inode = ?1 AND name = ?2",
                params![parent_inode, name],
            )?;
            drop_inode_if_orphaned(connection, inode.id)?;
        }
        Mutation::Rmdir { path } => {
            let inode = resolve_path(connection, workspace, path)?
                .ok_or_else(|| WorkspaceError::Missing(path.clone()))?;
            if inode.kind != KIND_DIRECTORY {
                return Err(WorkspaceError::NotADirectory(path.clone()));
            }
            let children: i64 = connection.query_row(
                "SELECT COUNT(*) FROM dirents WHERE parent_inode = ?1",
                params![inode.id],
                |row| row.get(0),
            )?;
            if children != 0 {
                return Err(WorkspaceError::NotEmpty(path.clone()));
            }
            let (parent, name) = split_parent(path);
            let parent_inode = match parent {
                None => workspace.root_inode,
                Some(parent_path) => {
                    resolve_path(connection, workspace, parent_path)?
                        .expect("the parent resolved during path lookup")
                        .id
                }
            };
            connection.execute(
                "DELETE FROM dirents WHERE parent_inode = ?1 AND name = ?2",
                params![parent_inode, name],
            )?;
            drop_inode_if_orphaned(connection, inode.id)?;
        }
        Mutation::Symlink { path, target } => {
            validate_symlink_target(target)?;
            let (parent, name) = resolve_new_leaf(connection, workspace, path)?;
            connection.execute(
                "INSERT INTO inodes (workspace_id, kind, symlink_target) VALUES (?1, ?2, ?3)",
                params![workspace.id, KIND_SYMLINK, target],
            )?;
            let inode = connection.last_insert_rowid();
            connection.execute(
                "INSERT INTO dirents (parent_inode, name, child_inode) VALUES (?1, ?2, ?3)",
                params![parent, name, inode],
            )?;
        }
        Mutation::Hardlink { path, target } => {
            let target_inode = resolve_path(connection, workspace, target)?
                .ok_or_else(|| WorkspaceError::Missing(target.clone()))?;
            if target_inode.kind != KIND_FILE {
                return Err(WorkspaceError::NotAFile(target.clone()));
            }
            let (parent, name) = resolve_new_leaf(connection, workspace, path)?;
            connection.execute(
                "INSERT INTO dirents (parent_inode, name, child_inode) VALUES (?1, ?2, ?3)",
                params![parent, name, target_inode.id],
            )?;
        }
        Mutation::SetExecutable { path, executable } => {
            let inode = resolve_path(connection, workspace, path)?
                .ok_or_else(|| WorkspaceError::Missing(path.clone()))?;
            if inode.kind != KIND_FILE {
                return Err(WorkspaceError::NotAFile(path.clone()));
            }
            connection.execute(
                "UPDATE inodes SET executable = ?1 WHERE id = ?2",
                params![i64::from(*executable), inode.id],
            )?;
        }
    }
    Ok(())
}

/// Cuts a record run at `length`. Boundary cuts drop whole records, a cut
/// inside a hole shrinks it, extension appends or grows a trailing hole, and
/// a cut inside a data record is refused as `None`.
fn truncate_records(records: Vec<FileRecord>, length: u64) -> Option<Vec<FileRecord>> {
    let mut kept = Vec::new();
    let mut position = 0_u64;
    for record in records {
        if position >= length {
            break;
        }
        let record_length = match &record {
            FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
        };
        let end = position.checked_add(record_length)?;
        if end <= length {
            kept.push(record);
            position = end;
            continue;
        }
        match record {
            FileRecord::Hole { .. } => {
                kept.push(FileRecord::Hole {
                    length: length - position,
                });
                position = length;
            }
            FileRecord::Data { .. } => return None,
        }
        break;
    }
    if position < length {
        match kept.last_mut() {
            Some(FileRecord::Hole {
                length: hole_length,
            }) => *hole_length += length - position,
            _ => kept.push(FileRecord::Hole {
                length: length - position,
            }),
        }
    }
    Some(kept)
}

/// Stages the committed tree into a builder by walking the dirent graph.
/// Hardlink groups pass one carrier file body and point every other path at
/// it; the builder re-roots by sorted order itself.
fn stage_builder(
    connection: &Connection,
    workspace: &WorkspaceRow,
    parent: Option<SnapshotId>,
) -> Result<SnapshotBuilder, WorkspaceError> {
    let mut paths_by_inode: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    let mut directories = Vec::new();
    let mut symlinks: Vec<(String, String)> = Vec::new();
    let mut stack = vec![(workspace.root_inode, String::new())];
    while let Some((inode, prefix)) = stack.pop() {
        let mut children = connection.prepare(
            "SELECT dirents.name, inodes.id, inodes.kind, inodes.symlink_target
                 FROM dirents JOIN inodes ON inodes.id = dirents.child_inode
                 WHERE dirents.parent_inode = ?1",
        )?;
        let mut rows = children.query(params![inode])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get(0)?;
            let child: i64 = row.get(1)?;
            let kind: i64 = row.get(2)?;
            let path = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            match kind {
                KIND_DIRECTORY => {
                    directories.push(path.clone());
                    stack.push((child, path));
                }
                KIND_FILE => paths_by_inode.entry(child).or_default().push(path),
                _ => {
                    let target: String = row.get(3)?;
                    symlinks.push((path, target));
                }
            }
        }
    }

    let mut builder = SnapshotBuilder::new(parent);
    for path in directories {
        builder.directory(path);
    }
    for (path, target) in symlinks {
        builder.symlink(path, target);
    }
    for (inode, mut paths) in paths_by_inode {
        paths.sort();
        let (executable, planner): (i64, i64) = connection.query_row(
            "SELECT executable, planner FROM inodes WHERE id = ?1",
            params![inode],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let records = read_records(connection, inode)?;
        let carrier = paths[0].clone();
        builder.file(
            carrier.clone(),
            executable == 1,
            planner_from_row(planner),
            records,
        );
        for path in paths.into_iter().skip(1) {
            builder.hardlink(path, carrier.clone());
        }
    }
    Ok(builder)
}

/// The committed head tree in the staging vocabulary — the mount's view.
fn build_tree(
    connection: &Connection,
    workspace: &WorkspaceRow,
) -> Result<HeadTree, WorkspaceError> {
    Ok(stage_builder(connection, workspace, None)?.finish_tree()?)
}

/// The canonical TFM1 snapshot of the committed tree. Only meaningful after
/// seal-time re-boundary: every non-tensor file must already be its single
/// whole-file object, or the builder refuses (`blob-records`).
fn build_snapshot(
    connection: &Connection,
    workspace: &WorkspaceRow,
    parent: Option<SnapshotId>,
) -> Result<Snapshot, WorkspaceError> {
    Ok(stage_builder(connection, workspace, parent)?.finish()?)
}

/// A test-only one-shot hook, fired at a point no static on-disk state can
/// reach.
///
/// Both users below guard the same shape: objects are admitted or rehashed
/// OUTSIDE the metadata transaction and rechecked for presence INSIDE it, so
/// a GC pass cannot delete them in between and leave a head naming bytes that
/// no longer exist. Deleting the objects before the call proves nothing —
/// the outer verify (or the admission that just wrote them) refuses first —
/// so the race can only be constructed temporally.
///
/// `#[cfg(test)]` keeps every line of this out of shipped binaries. The hook
/// is ONE-SHOT: it is taken as it fires, so an armed hook cannot leak into
/// another test sharing this binary.
#[cfg(test)]
struct Seam(Mutex<Option<Box<dyn Fn() + Send + Sync>>>);

#[cfg(test)]
impl Seam {
    const fn new() -> Self {
        Self(Mutex::new(None))
    }

    fn arm(&self, hook: impl Fn() + Send + Sync + 'static) {
        *self.0.lock().expect("seam mutex is healthy") = Some(Box::new(hook));
    }

    fn fire(&self) {
        let hook = self.0.lock().expect("seam mutex is healthy").take();
        if let Some(hook) = hook {
            hook();
        }
    }
}

/// Between `commit_generation`'s out-of-transaction `verify` sweep and its
/// in-transaction presence recheck.
#[cfg(test)]
static AFTER_VERIFY_SEAM: Seam = Seam::new();

/// Between `seal_snapshot`'s phase B (plan and admit outside the lock) and
/// phase C's transaction, which rechecks every newly admitted digest.
#[cfg(test)]
static AFTER_SEAL_PLAN_SEAM: Seam = Seam::new();

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    /// The seams are process-global one-shots, so the tests that arm them run
    /// one at a time. Poison is stepped over: a failing seam test must not
    /// cascade into the other.
    static SEAM_LANE: Mutex<()> = Mutex::new(());

    fn seam_lane() -> std::sync::MutexGuard<'static, ()> {
        SEAM_LANE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-workspace-unit-{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// What a GC pass looks like from the outside: every resident object gone.
    /// Returns the count so a seam can prove it actually did something.
    fn wipe_objects(root: &std::path::Path) -> usize {
        let mut removed = 0;
        let mut stack = vec![root.join("objects").join("sha256")];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if std::fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }
        }
        removed
    }

    /// A minimal well-formed safetensors file, so sealing has something to
    /// re-boundary and therefore new objects to admit in phase B.
    fn safetensors(tensors: &[(&str, u64, u8)]) -> Vec<u8> {
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
            file.extend(std::iter::repeat_n(*fill, *length as usize));
        }
        file
    }

    /// The GC race the in-transaction recheck exists to close.
    ///
    /// `commit_generation` verifies every referenced object OUTSIDE the
    /// metadata transaction, then rechecks presence INSIDE it. Deleting the
    /// objects before the call proves nothing — the outer `verify` refuses
    /// first, and the test would still pass with the recheck removed. The
    /// seam deletes them in the window between the two, which is exactly what
    /// a GC pass does to a writer stalled across two collection epochs.
    #[test]
    fn a_delete_between_verify_and_commit_cannot_land_a_head_naming_deleted_bytes() {
        let _lane = seam_lane();
        let root = scratch("gc-race");
        let store = WorkspaceStore::open(&root).expect("store opens");
        store.create_workspace("main").expect("workspace created");

        let bytes = b"bytes a stalled writer already verified".to_vec();
        let admitted = store.store().put_bytes(&bytes).expect("object admits");
        let digest = admitted.digest();

        let before = store
            .head_generation("main")
            .expect("head readable before the race");

        // Fires after `verify` has passed and before the transaction opens.
        let victim = root.clone();
        AFTER_VERIFY_SEAM.arm(move || {
            let store = ObjectStore::open(&victim).expect("collector opens the store");
            assert!(
                store.remove_object(&digest).expect("collector deletes"),
                "the seam must actually remove the object it is simulating a GC for"
            );
        });

        let result = store.commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "model.bin".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest,
                    length: admitted.length(),
                }],
            }],
        );

        match result {
            Err(WorkspaceError::MissingObject { digest: refused }) => {
                assert_eq!(refused, digest, "the refusal must name the deleted object");
            }
            Err(other) => panic!("expected MissingObject, got {other:?}"),
            Ok(generation) => panic!(
                "the commit landed generation {generation} while naming bytes \
                 that no longer exist — the in-transaction recheck is gone"
            ),
        }

        let after = store
            .head_generation("main")
            .expect("head readable after the race");
        assert_eq!(
            after, before,
            "a refused commit must not advance the workspace head"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The same race, one layer up: `seal_snapshot` admits its re-boundaried
    /// objects in phase B OUTSIDE the metadata lock and rechecks them inside
    /// phase C's transaction. That recheck was as unguarded as
    /// `commit_generation`'s — no static on-disk state reaches it, because the
    /// objects are admitted by the very call under test, so only the seam can
    /// delete them in the window.
    ///
    /// A seal that ignored the recheck would store a snapshot naming bytes
    /// that no longer exist: a permanent, self-verifying identity over
    /// unreadable content.
    #[test]
    fn a_delete_between_seal_planning_and_its_transaction_cannot_seal_deleted_bytes() {
        let _lane = seam_lane();
        let root = scratch("seal-race");
        let store = WorkspaceStore::open(&root).expect("store opens");
        store.create_workspace("main").expect("workspace created");

        // Committed as ONE raw region, so sealing must re-slice it into
        // header-plus-tensor regions and admit every one of them anew.
        let bytes = safetensors(&[
            ("blk.attn.weight", 4096, 0xA1),
            ("blk.norm.weight", 512, 0xC3),
        ]);
        let admitted = store.store().put_bytes(&bytes).expect("object admits");
        store
            .commit_generation(
                "main",
                &[Mutation::CreateFile {
                    path: "model.safetensors".to_owned(),
                    executable: false,
                    planner: PlannerId::BlobV1,
                    records: vec![FileRecord::Data {
                        digest: admitted.digest(),
                        length: admitted.length(),
                    }],
                }],
            )
            .expect("the grid record commits");

        let before = store
            .head_generation("main")
            .expect("head readable before the race");

        // Fires after phase B has planned and admitted, before phase C opens
        // its transaction.
        let victim = root.clone();
        AFTER_SEAL_PLAN_SEAM.arm(move || {
            assert!(
                wipe_objects(&victim) > 0,
                "the seam must actually remove the objects it is simulating a GC for"
            );
        });

        match store.seal_snapshot("main", None) {
            Err(WorkspaceError::MissingObject { .. }) => {}
            Err(other) => panic!("expected MissingObject, got {other:?}"),
            Ok(id) => panic!(
                "seal stored snapshot {id:?} naming bytes that no longer exist \
                 — phase C's in-transaction recheck is gone"
            ),
        }

        assert_eq!(
            store
                .head_generation("main")
                .expect("head readable after the race"),
            before,
            "a refused seal must not advance the workspace head"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
