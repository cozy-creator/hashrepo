//! The projected store layout: snapshot symlink trees, pointer stubs, refs.
//!
//! ```text
//! <root>/objects/sha256/xx/yy/<hex>   every CAS blob: whole files AND chunks
//! <root>/snapshots/<snapshot-id>/…    projected trees (disposable)
//! <root>/refs/<name> -> ../snapshots/<id>
//! ```
//!
//! A projection copies **zero bytes**: a directory entry becomes a real
//! directory, a blob-planner file a relative symlink into `objects/`, and a
//! tensor-planner file — which has no single inode to point at — a pointer
//! stub. Trees are derivable from the manifest at any time, pin nothing, and
//! are deleted whenever their root is.
//!
//! Everything here is a pure function of a decoded manifest plus the store
//! root, so it needs no metadata engine: the workspace layer owns roots,
//! deletion order and GC; this module owns the bytes on disk.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use fs4::FileExt;
use thiserror::Error;

use crate::object::ObjectDigest;
use crate::store::{ObjectStore, StoreError, set_read_only};
use crate::tfm1::{Entry, FileBody, Snapshot, SnapshotId};

static SWAP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The scratch names this module owns: a half-built projection under
/// `snapshots/`, its lease under the store's `tmp/`, and a ref mid-swap.
const BUILDING_PREFIX: &str = ".building-";
const REMOVED_PREFIX: &str = ".removed-";
const LEASE_PREFIX: &str = "building-";
const LEASE_SUFFIX: &str = ".tmp";
const SWAP_PREFIX: &str = ".swap-";

/// First eight bytes of every pointer stub, so a tool can classify one
/// without parsing JSON — and so no safetensors u64 header or GGUF magic can
/// be mistaken for it.
pub const STUB_MAGIC: &[u8; 8] = b"TFSSTUB1";

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("projected layout I/O failed")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("ref name {0:?} is not a single safe path segment")]
    InvalidRefName(String),
    #[error("ref {0:?} does not name a snapshot id")]
    UnreadableRef(String),
}

/// What one reap of abandoned projection scratch examined and removed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScratchReap {
    pub examined: u64,
    pub trees_removed: u64,
    pub swaps_removed: u64,
    /// Artifacts a removal had already unlinked from the namespace, whose
    /// bytes only this reap could finally take — a Windows reader holding a
    /// file open is the one thing that defers them.
    pub unlinked_removed: u64,
    /// What was reclaimed, recorded BEFORE each removal.
    ///
    /// A sweep that frees resources and reports only a count leaves nobody
    /// able to say afterwards WHOSE they were — which is how the 2026-08-16
    /// orphaned-FUSE-connection incident (#96) stayed invisible until the box
    /// was unusable. The evidence outlives the artifact because it travels
    /// out in the report, so a caller can log it at whatever level survives.
    pub reclaimed: Vec<ReclaimedScratch>,
}

/// One scratch artifact, described while it still exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReclaimedScratch {
    pub path: PathBuf,
    /// The creating process's token, read off the name that process wrote.
    /// Evidence only: the lease decides, never this.
    pub creator: String,
    pub lease: LeaseState,
}

/// Why a scratch artifact was reclaimable. None of these is a statement about
/// age.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseState {
    /// A lease file existed and its lock was free: the holder is gone.
    Free,
    /// No lease file at all: the holder never got one, or it was reaped.
    Absent,
    /// No live NAME: a removal renamed the artifact out of the namespace, so
    /// nothing can reach it and no lease could make it live again.
    Unreachable,
}

/// What a removal did to the name it was asked to take.
///
/// Three outcomes rather than a `bool`, because Windows cannot promise the
/// second one away: a name is taken, was never there, or **could not be taken
/// right now**.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Removal {
    /// The name is gone. Its bytes are gone too, or are unreachable scratch a
    /// reap will take.
    Taken,
    /// Nothing was there — including the case of losing a removal race, which
    /// is a success for everyone who wanted the name gone.
    Absent,
    /// The name is still there and this caller did nothing wrong: another
    /// handle is inside the artifact, which on Windows refuses the rename that
    /// takes the name away. It says nothing about rights and nothing about the
    /// artifact — the same call succeeds once that handle closes — so the
    /// artifact stays and the next scrub takes it, exactly like one left by a
    /// crash. POSIX never returns this.
    Deferred,
}

impl Removal {
    /// True only for [`Removal::Taken`], for callers whose whole question is
    /// "is that name gone because of me".
    #[must_use]
    pub const fn taken(self) -> bool {
        matches!(self, Self::Taken)
    }
}

/// What one pointer stub says: which file body it stands for, and how many
/// bytes that file really has.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stub {
    pub body_sha256: ObjectDigest,
    pub size: u64,
}

/// One pointer stub's bytes: the magic, one space, one line of JSON.
///
/// Emitted by hand rather than by a serializer because the BYTES are the
/// contract (`spec/v1/TFSSTUB1.md`, `spec/v1/tfsstub1-vectors/`) — key order,
/// absence of whitespace and the single trailing line feed included.
///
/// `body_sha256` is the SHA-256 of the manifest's canonical file **body**
/// encoding — the file's identity under TFM1, which deliberately carries no
/// whole-file hash for a tensor container. It is derivable from the manifest
/// entry alone, so projecting a stub reads no tensor bytes.
#[must_use]
pub fn stub_bytes(body_sha256: &ObjectDigest, size: u64) -> Vec<u8> {
    format!(
        "TFSSTUB1 {{\"body_sha256\":\"{}\",\"size\":{size},\"read\":\"tensorfs\"}}\n",
        body_sha256.to_hex()
    )
    .into_bytes()
}

/// Reads one pointer stub, or `None` when the bytes are not one.
///
/// Cheap enough to run on any small file: the magic decides in eight bytes.
#[must_use]
pub fn parse_stub(bytes: &[u8]) -> Option<Stub> {
    let rest = bytes.strip_prefix(STUB_MAGIC)?.strip_prefix(b" ")?;
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Wire {
        body_sha256: String,
        size: u64,
        read: String,
    }
    let wire: Wire = serde_json::from_slice(rest.strip_suffix(b"\n").unwrap_or(rest)).ok()?;
    if wire.read != "tensorfs" {
        return None;
    }
    Some(Stub {
        body_sha256: parse_hex_digest(&wire.body_sha256)?,
        size: wire.size,
    })
}

fn parse_hex_digest(text: &str) -> Option<ObjectDigest> {
    if text.len() != 64 {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let digits = std::str::from_utf8(pair).ok()?;
        if digits.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return None;
        }
        bytes[index] = u8::from_str_radix(digits, 16).ok()?;
    }
    Some(ObjectDigest::from_bytes(bytes))
}

/// The projected trees and refs beside one object store.
#[derive(Debug)]
pub struct Layout<'store> {
    store: &'store ObjectStore,
    symlinks: bool,
}

impl<'store> Layout<'store> {
    /// Projects with symlinks wherever the store's open-time probe found
    /// them, and with copies where it did not.
    #[must_use]
    pub const fn new(store: &'store ObjectStore) -> Self {
        Self {
            store,
            symlinks: store.supports_symlinks(),
        }
    }

    /// Projects with copies unconditionally — the Windows-dev fallback.
    /// Correctness is identical; local dedup is lost.
    #[must_use]
    pub const fn without_symlinks(store: &'store ObjectStore) -> Self {
        Self {
            store,
            symlinks: false,
        }
    }

    #[must_use]
    pub fn snapshots_dir(&self) -> PathBuf {
        self.store.root().join("snapshots")
    }

    #[must_use]
    pub fn refs_dir(&self) -> PathBuf {
        self.store.root().join("refs")
    }

    #[must_use]
    pub fn tree_path(&self, id: &SnapshotId) -> PathBuf {
        self.snapshots_dir().join(hex_of(id))
    }

    /// Projects one snapshot as `snapshots/<id>/…` and returns its path.
    ///
    /// The tree is built under a private leased name and renamed into place,
    /// so a ref can never resolve into a half-built projection. Projection is
    /// idempotent: an existing tree wins and the loser's scratch is removed.
    ///
    /// The lease is taken BEFORE the scratch directory exists and released
    /// only after the rename, which is what lets [`reap_scratch`] tell a
    /// projection that crashed from one that is merely slow.
    ///
    /// [`reap_scratch`]: Self::reap_scratch
    pub fn project(&self, snapshot: &Snapshot) -> Result<PathBuf, LayoutError> {
        let final_path = self.tree_path(&snapshot.snapshot_id());
        if final_path.exists() {
            return Ok(final_path);
        }
        let snapshots = self.snapshots_dir();
        fs::create_dir_all(&snapshots)?;
        let (token, lease) = self.take_scratch_lease()?;
        let scratch = snapshots.join(format!("{BUILDING_PREFIX}{token}"));

        let projected = self.build_into(snapshot, &scratch, &final_path);
        // A no-op after a successful rename, and the cleanup on every other
        // path. The name is never reused, so nothing else can be behind it.
        let _ = fs::remove_dir_all(&scratch);
        drop(lease);
        let _ = fs::remove_file(self.lease_path(&token));
        projected
    }

    fn build_into(
        &self,
        snapshot: &Snapshot,
        scratch: &Path,
        final_path: &Path,
    ) -> Result<PathBuf, LayoutError> {
        fs::create_dir(scratch)?;
        self.fill(snapshot, scratch)?;
        match fs::rename(scratch, final_path) {
            Ok(()) => Ok(final_path.to_path_buf()),
            // Another projector of the same snapshot won the race. Its tree
            // has the same content by construction — the manifest is the id.
            Err(_) if final_path.exists() => Ok(final_path.to_path_buf()),
            Err(error) => Err(error.into()),
        }
    }

    fn lease_path(&self, token: &str) -> PathBuf {
        self.store
            .tmp_dir()
            .join(format!("{LEASE_PREFIX}{token}{LEASE_SUFFIX}"))
    }

    /// Takes the advisory lease that says a scratch tree's creator is alive.
    ///
    /// It lives in the store's `tmp/` as a file because a directory cannot
    /// portably carry an `flock`, and it is named from the same token as the
    /// scratch so a reaper needs no registry to correlate the two.
    fn take_scratch_lease(&self) -> Result<(String, File), LayoutError> {
        loop {
            let token = scratch_token();
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.lease_path(&token))
            {
                Ok(file) => {
                    file.try_lock_exclusive()?;
                    return Ok((token, file));
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Removes the scratch of projections and ref swaps whose creator is gone.
    ///
    /// Liveness decides, never a name, a PID or a clock. A `.building-…` tree
    /// goes only when its lease is free or already absent, a `.swap-…` file
    /// only when it can be leased and is still the inode the scan listed. A
    /// projection that has been building for an hour is retained; a crashed
    /// one is reclaimable the instant its holder dies.
    pub fn reap_scratch(&self) -> Result<ScratchReap, LayoutError> {
        let mut report = ScratchReap::default();
        for entry in read_dir_or_empty(&self.snapshots_dir())? {
            let name = entry.file_name();
            let Some(token) = name
                .to_str()
                .and_then(|name| name.strip_prefix(BUILDING_PREFIX))
            else {
                continue;
            };
            report.examined += 1;
            let Some(lease) = lease_state(&self.lease_path(token)) else {
                continue;
            };
            // Recorded before the removal: after it, the name is gone and
            // nothing on disk can say who left it behind.
            let evidence = ReclaimedScratch {
                path: entry.path(),
                creator: token.to_owned(),
                lease,
            };
            if fs::remove_dir_all(entry.path()).is_ok() {
                report.trees_removed += 1;
                report.reclaimed.push(evidence);
            }
        }
        for entry in read_dir_or_empty(&self.refs_dir())? {
            let name = entry.file_name();
            let Some(token) = name
                .to_str()
                .and_then(|name| name.strip_prefix(SWAP_PREFIX))
            else {
                continue;
            };
            if !entry.metadata().is_ok_and(|listed| listed.is_file()) {
                continue;
            }
            report.examined += 1;
            // Decided by the same lease a projection uses. The staged file
            // carries no lock of its own, so its liveness is not a question
            // its own inode can answer; the token's lease answers it.
            let Some(lease) = lease_state(&self.lease_path(token)) else {
                continue;
            };
            let evidence = ReclaimedScratch {
                path: entry.path(),
                creator: token.to_owned(),
                lease,
            };
            if fs::remove_file(entry.path()).is_ok() {
                report.swaps_removed += 1;
                report.reclaimed.push(evidence);
            }
        }
        // Whatever a removal already took the name away from. It is
        // unreachable by construction — no name resolves to it and no lease
        // can make it live again — so it goes on sight, and the only thing
        // that can have delayed it is a reader that still holds its bytes.
        for directory in [self.snapshots_dir(), self.refs_dir()] {
            for entry in read_dir_or_empty(&directory)? {
                let name = entry.file_name();
                let Some(token) = name
                    .to_str()
                    .and_then(|name| name.strip_prefix(REMOVED_PREFIX))
                else {
                    continue;
                };
                report.examined += 1;
                let evidence = ReclaimedScratch {
                    path: entry.path(),
                    creator: token.to_owned(),
                    lease: LeaseState::Unreachable,
                };
                if delete_scratch(&entry.path()) {
                    report.unlinked_removed += 1;
                    report.reclaimed.push(evidence);
                }
            }
        }
        // Leases last: a lease whose scratch was renamed into place, or which
        // the sweeps above just finished with. Removing them earlier would
        // answer "why was this reclaimable?" with `Absent` for artifacts whose
        // holder demonstrably died, degrading the evidence to the vaguer of
        // two true answers.
        for entry in read_dir_or_empty(&self.store.tmp_dir())? {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.starts_with(LEASE_PREFIX)
                && name.ends_with(LEASE_SUFFIX)
                && lease_state(&entry.path()).is_some()
            {
                let _ = fs::remove_file(entry.path());
            }
        }
        Ok(report)
    }

    /// Writes every entry into an already-created tree root.
    fn fill(&self, snapshot: &Snapshot, root: &Path) -> Result<(), LayoutError> {
        for (path, entry) in snapshot.entries() {
            let target = root.join(path);
            match entry {
                Entry::Directory => fs::create_dir(&target)?,
                Entry::File { body, .. } => self.project_body(path, body, &target)?,
                Entry::Symlink {
                    target: link_target,
                } => self.link(link_target.as_ref(), &target)?,
                Entry::Hardlink { ordinal } => {
                    let carrier = snapshot
                        .ordinal_path(*ordinal)
                        .expect("a decoded snapshot resolves every ordinal");
                    let body = snapshot
                        .entries()
                        .iter()
                        .find_map(|(candidate, entry)| match entry {
                            Entry::File { body, .. } if candidate == carrier => Some(body),
                            _ => None,
                        })
                        .expect("a hardlink ordinal names a file entry");
                    self.project_body(path, body, &target)?;
                }
            }
        }
        Ok(())
    }

    /// One file entry's projection: a stub for a tensor container, a relative
    /// symlink into `objects/` for a blob, an empty file for an empty blob.
    ///
    /// The dispatch is on the BODY KIND and nothing else, and every value it
    /// needs comes from the manifest entry — never from a source file that
    /// may not exist. A re-keyed or derived entry (#80, #81) is a new
    /// `FileBody` variant that this match will refuse to compile without,
    /// which is the point.
    fn project_body(&self, path: &str, body: &FileBody, target: &Path) -> Result<(), LayoutError> {
        match body {
            FileBody::Tensor { logical_size, .. } => {
                self.install_bytes(target, &stub_bytes(&body.body_sha256(), *logical_size))
            }
            FileBody::Blob {
                logical_size: 0, ..
            } => self.install_bytes(target, b""),
            FileBody::Blob { digest, .. } => {
                if self.symlinks {
                    self.link(&relative_object_target(path, digest), target)
                } else {
                    // No symlinks on this filesystem: the tree carries a copy.
                    // Local dedup is lost, correctness is kept.
                    fs::copy(self.store.object_path(digest), target)?;
                    set_read_only(target)?;
                    Ok(())
                }
            }
        }
    }

    /// Installs a projection artifact — a stub or an empty blob — immutable.
    fn install_bytes(&self, target: &Path, bytes: &[u8]) -> Result<(), LayoutError> {
        fs::write(target, bytes)?;
        set_read_only(target)?;
        Ok(())
    }

    fn link(&self, link_target: &Path, at: &Path) -> Result<(), LayoutError> {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(link_target, at)?;
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(link_target, at)?;
        }
        Ok(())
    }

    /// Removes one snapshot's projected tree. A tree pins nothing, so this is
    /// pure cache eviction; the manifest can re-project it byte for byte.
    ///
    /// Removal is [`unlink_scratch`]: the live name is given up by one
    /// `rename`, and only the private name it leaves behind is deleted.
    ///
    /// [`unlink_scratch`]: Self::unlink_scratch
    pub fn remove_tree(&self, id: &SnapshotId) -> Result<Removal, LayoutError> {
        self.unlink_scratch(&self.tree_path(id), &self.snapshots_dir())
    }

    /// Takes one live name away and deletes what it named, in that order.
    ///
    /// Deleting a live name in place is the wrong primitive under
    /// concurrency, and Windows says so out loud. Two removers of one tree
    /// both call `remove_dir_all` on the same directory and the loser gets
    /// `ERROR_ACCESS_DENIED`, not `NotFound` — a scrub racing a deletion
    /// failed for doing exactly what it is for (#109). A reader holding one
    /// file inside a tree refuses the whole removal for the same reason,
    /// where POSIX would unlink the name and keep the bytes for the reader.
    ///
    /// So the name goes first, by `rename` into a `.removed-…` scratch name
    /// nothing else can reach, and the bytes go second. The rename is the
    /// claim: exactly one remover can win it, the loser is told `NotFound`
    /// and reports the honest `false`, and no two removers ever meet inside
    /// one artifact. A delete that then fails — a Windows reader still has
    /// the bytes open — leaves only unreachable scratch that
    /// [`reap_scratch`] takes on sight, so it is not a removal failure and
    /// is not reported as one: the caller asked for the name to be gone, and
    /// it is.
    ///
    /// [`reap_scratch`]: Self::reap_scratch
    fn unlink_scratch(&self, path: &Path, scratch_dir: &Path) -> Result<Removal, LayoutError> {
        let claimed = scratch_dir.join(format!("{REMOVED_PREFIX}{}", scratch_token()));
        match fs::rename(path, &claimed) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Removal::Absent),
            Err(error) if is_share_refusal(&error) => return Ok(Removal::Deferred),
            Err(error) => return Err(error.into()),
        }
        delete_scratch(&claimed);
        Ok(Removal::Taken)
    }

    /// Points `refs/<name>` at one snapshot, replacing any previous value by
    /// one `rename(2)`.
    ///
    /// A ref is a plain text file holding the id — git's `.git/refs/heads/*`
    /// and the HF cache's `refs/main`, and NOT the symlink this layout was
    /// first designed with. Measurement retired the symlink: `readlink` on a
    /// name being renamed transiently returns EINVAL on APFS, and a
    /// multi-component walk THROUGH such a name transiently returns ENOENT on
    /// Linux too, so the symlink's two advantages — `readlink` and free
    /// traversal — both evaporate under exactly the concurrency a ref exists
    /// to survive. `open`-then-`read` of a rename-swapped name yields the old
    /// id or the new one and nothing else — **on POSIX**. Windows' rename is
    /// not atomic to a concurrent opener and cannot be made so, so there the
    /// `open` itself transiently fails and [`Self::read_ref`] looks again
    /// (#103); a Windows tool reading the file directly must do the same.
    pub fn set_ref(&self, name: &str, id: &SnapshotId) -> Result<(), LayoutError> {
        let refs = self.refs_dir();
        let destination = refs.join(validated_ref_name(name)?);
        fs::create_dir_all(&refs)?;
        // Leased for its whole life like every other scratch artifact — but
        // the lease is a SEPARATE file in `tmp/`, never a lock on the staged
        // ref itself. A lock taken here travels with the inode through the
        // rename, and Windows' is mandatory: it would make the LIVE ref
        // unreadable to every concurrent reader until this handle closed.
        // Nothing that becomes live may carry a lock.
        let (token, lease) = self.take_scratch_lease()?;
        let staged = refs.join(format!("{SWAP_PREFIX}{token}"));
        let swapped = (|| -> io::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)?;
            file.write_all(&ref_bytes(id))?;
            file.sync_all()?;
            drop(file);
            fs::rename(&staged, &destination)
        })();
        let _ = fs::remove_file(&staged);
        drop(lease);
        let _ = fs::remove_file(self.lease_path(&token));
        Ok(swapped?)
    }

    /// Reads `refs/<name>`: the id it names, or `None` when there is no such
    /// ref. One `open` plus one `read`, and it never waits: on Windows the
    /// `open` can transiently fail while a swap replaces the name (#103), and
    /// that failure is REPORTED rather than retried, because the only state a
    /// retry could wait on is a peer's progress — and a peer can be waiting
    /// on this reader's caller. `scrub` held a transaction across exactly that
    /// wait and deadlocked the store. A caller that wants the ref reads again;
    /// nothing that reads a ref may block on one that is being written.
    pub fn read_ref(&self, name: &str) -> Result<Option<SnapshotId>, LayoutError> {
        let path = self.refs_dir().join(validated_ref_name(name)?);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        String::from_utf8(bytes)
            .ok()
            .and_then(|text| SnapshotId::parse_hex(text.trim()))
            .map(Some)
            .ok_or_else(|| LayoutError::UnreadableRef(name.to_owned()))
    }

    /// Drops `refs/<name>`, by the same claim-then-delete as a tree
    /// ([`unlink_scratch`]) and for the same reason: a ref is removed by a
    /// scrub and by `delete_snapshot` at once, and a reader may hold it open.
    ///
    /// [`unlink_scratch`]: Self::unlink_scratch
    pub fn remove_ref(&self, name: &str) -> Result<Removal, LayoutError> {
        let path = self.refs_dir().join(validated_ref_name(name)?);
        self.unlink_scratch(&path, &self.refs_dir())
    }

    /// Every ref name currently pointing at one snapshot, sorted.
    pub fn refs_to(&self, id: &SnapshotId) -> Result<Vec<String>, LayoutError> {
        let mut names = Vec::new();
        for name in self.ref_names()? {
            if self.read_ref(&name).ok().flatten() == Some(*id) {
                names.push(name);
            }
        }
        Ok(names)
    }

    /// Every ref name present, sorted. Names a ref may not have — a leading
    /// dot above all — are skipped, which is what keeps a `.swap-…` staged
    /// file mid-`rename` invisible to enumeration.
    pub fn ref_names(&self) -> Result<Vec<String>, LayoutError> {
        let mut names = Vec::new();
        let entries = match fs::read_dir(self.refs_dir()) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(names),
            Err(error) => return Err(error.into()),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if validated_ref_name(name).is_ok() {
                names.push(name.to_owned());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Every projected tree present, as the snapshot id it claims to hold.
    ///
    /// A name that is not a snapshot id is not a tree: `project` builds under
    /// a leased `.building-…` name and renames into place, so such a name
    /// belongs to a projection whose liveness only its lease can answer.
    /// Enumeration reports trees; [`reap_scratch`] handles the rest.
    ///
    /// [`reap_scratch`]: Self::reap_scratch
    pub fn tree_ids(&self) -> Result<Vec<SnapshotId>, LayoutError> {
        let mut ids = Vec::new();
        let entries = match fs::read_dir(self.snapshots_dir()) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(ids),
            Err(error) => return Err(error.into()),
        };
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && let Some(id) = SnapshotId::parse_hex(name)
            {
                ids.push(id);
            }
        }
        ids.sort_by_key(|id| *id.as_bytes());
        Ok(ids)
    }
}

/// A collision-free name for one scratch artifact. The clock is in it for
/// uniqueness across a restart with the same PID, never as an age: nothing
/// reads it back.
fn scratch_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{}-{}-{nanos}",
        process::id(),
        SWAP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

/// Windows refuses to rename or delete an artifact another handle is inside:
/// `ERROR_ACCESS_DENIED` (5) for a directory whose contents are open, and
/// `ERROR_SHARING_VIOLATION` (32) for a file. Neither is a statement about
/// this caller's rights — the identical call succeeds once the other handle
/// closes — and neither has a POSIX equivalent, where a rename cannot be
/// refused for a busy artifact. So this clause is Windows-only, deliberately:
/// on POSIX every one of these errors is real and propagates.
#[cfg(windows)]
fn is_share_refusal(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(5 | 32))
}

#[cfg(not(windows))]
const fn is_share_refusal(_error: &io::Error) -> bool {
    false
}

/// Deletes one unreachable scratch artifact, directory or file. Best effort
/// on purpose: what a reader still holds open is taken by the next reap.
fn delete_scratch(path: &Path) -> bool {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path).is_ok(),
        Ok(_) => fs::remove_file(path).is_ok(),
        Err(_) => false,
    }
}

/// Why this lease says its creator is gone, or `None` while one holds it.
///
/// One sample answers it, which is the whole advantage of a lease over the
/// drain-check shape #96 needs for FUSE connections: there, liveness has to
/// be inferred from a work counter that fails to move across two samples,
/// because a wedged connection has no owner left to ask. Here the owner IS
/// the lock, so a second sample could only repeat the first.
fn lease_state(path: &Path) -> Option<LeaseState> {
    match OpenOptions::new().read(true).open(path) {
        Ok(file) => file.try_lock_exclusive().ok().map(|()| LeaseState::Free),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Some(LeaseState::Absent),
        Err(_) => None,
    }
}

fn read_dir_or_empty(path: &Path) -> Result<Vec<fs::DirEntry>, LayoutError> {
    match fs::read_dir(path) {
        Ok(entries) => Ok(entries.flatten().collect()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}

/// The relative link from a projected file back to its object.
///
/// Always relative: the store relocates as a unit, and an absolute target
/// would pin it to one mount path. The depth is the entry's own directory
/// depth plus the two components `snapshots/<id>`.
fn relative_object_target(path: &str, digest: &ObjectDigest) -> PathBuf {
    let hex = digest.to_hex();
    let depth = path.matches('/').count() + 2;
    let mut target = PathBuf::new();
    for _ in 0..depth {
        target.push("..");
    }
    target
        .join("objects")
        .join("sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(hex)
}

/// Ref names are one safe path segment. Nesting would change every ref's
/// link depth for no gain, and `.`/`..`/separators would escape `refs/`.
fn validated_ref_name(name: &str) -> Result<&str, LayoutError> {
    let refused = name.is_empty()
        || name.len() > 255
        || name == "."
        || name == ".."
        || name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0');
    if refused {
        return Err(LayoutError::InvalidRefName(name.to_owned()));
    }
    Ok(name)
}

/// One ref file's whole content: the id and a line feed, so `cat` and every
/// line-oriented tool read it the way they read a git ref.
fn ref_bytes(id: &SnapshotId) -> Vec<u8> {
    format!("{}\n", hex_of(id)).into_bytes()
}

fn hex_of(id: &SnapshotId) -> String {
    ObjectDigest::from_bytes(*id.as_bytes()).to_hex()
}
