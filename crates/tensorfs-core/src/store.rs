use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fs4::FileExt;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;

use crate::object::ObjectDigest;
use crate::planner::{ByteSource, MAX_OBJECT_SIZE, Region};

const VERIFY_BUFFER_SIZE: usize = 1024 * 1024;
const TEMP_PREFIX: &str = "obj-";
const TEMP_SUFFIX: &str = ".tmp";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Process-wide ceiling on bytes held in RAM while admitting objects, and the
/// environment variable that overrides it.
///
/// This is deliberately the SAME name and default the shelved daemon's slot
/// assembly used (`shelf/tensorfsd`). Ingest and assembly both hold whole
/// grid slots in RAM for the same reason, so an operator tunes one number
/// rather than discovering a second knob that means almost the same thing.
const INGEST_BUDGET_ENV: &str = "TENSORFS_ASSEMBLY_BUDGET_BYTES";
const DEFAULT_INGEST_BUDGET: u64 = 8 * MAX_OBJECT_SIZE;

/// How many objects may be hashed and admitted at once.
///
/// Each worker holds one whole object in RAM (never more than
/// [`MAX_OBJECT_SIZE`]), so budget-divided-by-object-ceiling *is* the worker
/// count: memory is the binding constraint, not cores. It is then clamped by
/// the machine's parallelism, because SHA-256 is compute-bound even with
/// SHA-NI — more hashing threads than cores buys context switches, not
/// throughput. A 4-vCPU pod therefore runs 4 workers, not 8.
#[must_use]
pub fn ingest_concurrency() -> usize {
    let budget = std::env::var(INGEST_BUDGET_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_INGEST_BUDGET);
    let by_memory = usize::try_from(budget / MAX_OBJECT_SIZE).unwrap_or(1);
    let by_cores = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    by_memory.clamp(1, by_cores.max(1))
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("object store I/O failed")]
    Io(#[from] io::Error),
    #[error("admitted length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: u64, actual: u64 },
    #[error("admitted digest mismatch: expected {expected}, got {actual}")]
    DigestMismatch {
        expected: ObjectDigest,
        actual: ObjectDigest,
    },
    #[error("object {digest} is missing")]
    Missing { digest: ObjectDigest },
    #[error("object {digest} is not a regular file")]
    NotARegularFile { digest: ObjectDigest },
    #[error("object {expected} is corrupt: resident bytes hash to {actual}")]
    CorruptObject {
        expected: ObjectDigest,
        actual: ObjectDigest,
    },
    #[error("temp collection requires a positive grace age")]
    InvalidGrace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmittedObject {
    digest: ObjectDigest,
    length: u64,
    preexisting: bool,
}

impl AdmittedObject {
    #[must_use]
    pub const fn digest(&self) -> ObjectDigest {
        self.digest
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// True when a verified same-digest object was already resident, so this
    /// admission installed nothing.
    #[must_use]
    pub const fn preexisting(&self) -> bool {
        self.preexisting
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TempCollection {
    pub examined: u64,
    pub deleted: u64,
    pub bytes_deleted: u64,
}

/// Crash-safe immutable object storage under one root directory.
///
/// Bytes enter through a unique leased temp, are hashed while written, length
/// and digest checked, fsynced, installed no-clobber at their digest path, and
/// sealed by a containing-directory fsync. A resident same-digest object is
/// accepted only after rehash verification; size, path, or mtime equality is
/// never proof. Workspace metadata, reachability GC, and snapshots are owned
/// by the pgw#1259 metadata layer, not this store.
#[derive(Debug)]
pub struct ObjectStore {
    root: PathBuf,
    symlinks: bool,
}

impl ObjectStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        let sha256 = root.join("objects").join("sha256");
        let tmp = root.join("tmp");
        fs::create_dir_all(&sha256)?;
        fs::create_dir_all(&tmp)?;
        fsync_dir(&root)?;
        fsync_dir(&root.join("objects"))?;
        fsync_dir(&sha256)?;
        fsync_dir(&tmp)?;
        let symlinks = probe_symlinks(&root);
        Ok(Self { root, symlinks })
    }

    /// The store root every projected tree and ref is relative to.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this filesystem let the store create a symlink at open.
    ///
    /// False on a Windows box without the create-symlink privilege, and on
    /// the handful of filesystems that have no symlinks at all. The
    /// projection falls back to copies there: local dedup is lost,
    /// correctness is kept.
    #[must_use]
    pub const fn supports_symlinks(&self) -> bool {
        self.symlinks
    }

    pub(crate) fn sha256_dir(&self) -> PathBuf {
        self.root.join("objects").join("sha256")
    }

    pub(crate) fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    pub(crate) fn object_path(&self, digest: &ObjectDigest) -> PathBuf {
        let hex = digest.to_hex();
        self.sha256_dir().join(&hex[..2]).join(&hex[2..4]).join(hex)
    }

    /// Opens a leased streaming writer whose bytes are hashed as they arrive.
    pub fn writer(&self) -> Result<ObjectWriter<'_>, StoreError> {
        loop {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let name = format!(
                "{TEMP_PREFIX}{}-{}-{nanos}{TEMP_SUFFIX}",
                process::id(),
                TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed),
            );
            let path = self.tmp_dir().join(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    // The advisory lease held for the write/verify/install
                    // lifetime; a freshly created unique temp cannot be held.
                    file.try_lock_exclusive()?;
                    return Ok(ObjectWriter {
                        store: self,
                        file: Some(file),
                        path,
                        hasher: Sha256::new(),
                        written: 0,
                        installed: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Admits one in-memory byte string and returns its verified identity.
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<AdmittedObject, StoreError> {
        let mut writer = self.writer()?;
        writer.write_all(bytes)?;
        writer.finish()
    }

    /// Admits the named regions of one source and returns them in region
    /// order, hashing and installing up to [`ingest_concurrency`] of them at
    /// once.
    ///
    /// Every region is read and hashed exactly **once**. Callers must not hash
    /// first and admit second — that is two full SHA-256 passes over the same
    /// bytes, and SHA-256 is what bounds ingest.
    ///
    /// The per-object crash boundary is unchanged: leased temp, hash while
    /// writing, verify, fsync, no-clobber `hard_link` install, directory
    /// fsync. Concurrency adds throughput without widening the durability
    /// window, and two workers racing the same digest still converge on one
    /// file with both succeeding, because the install is no-clobber.
    ///
    /// On error the regions that already succeeded stay admitted. That is
    /// safe and deliberate: an admitted object nothing references is an
    /// unreferenced CAS entry, which the two-epoch GC reclaims.
    pub fn admit_regions<S>(
        &self,
        source: &S,
        regions: &[Region],
    ) -> Result<Vec<AdmittedObject>, StoreError>
    where
        S: ByteSource + Sync + ?Sized,
    {
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let workers = ingest_concurrency().min(regions.len());
        if workers <= 1 {
            return regions
                .iter()
                .map(|region| self.admit_region(source, region))
                .collect();
        }

        let next = AtomicUsize::new(0);
        let mut collected: Vec<(usize, Result<AdmittedObject, StoreError>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        let next = &next;
                        scope.spawn(move || {
                            let mut mine = Vec::new();
                            loop {
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                let Some(region) = regions.get(index) else {
                                    break;
                                };
                                mine.push((index, self.admit_region(source, region)));
                            }
                            mine
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| match handle.join() {
                        Ok(done) => done,
                        // Never swallow a worker panic into a short result
                        // vector — that would silently drop objects the
                        // caller is about to name in a manifest.
                        Err(payload) => std::panic::resume_unwind(payload),
                    })
                    .collect()
            });
        collected.sort_by_key(|(index, _)| *index);
        // Returns the first failure in REGION order, not in completion order,
        // so a concurrent run reports the same error a serial one would.
        collected.into_iter().map(|(_, result)| result).collect()
    }

    /// Reads one region into a bounded buffer and admits it.
    fn admit_region<S>(&self, source: &S, region: &Region) -> Result<AdmittedObject, StoreError>
    where
        S: ByteSource + ?Sized,
    {
        let length = usize::try_from(region.length())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(length)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        buffer.resize(length, 0);
        source.read_exact_at(region.offset(), &mut buffer)?;
        self.put_bytes(&buffer)
    }

    /// Admits one contiguous source range as ONE object of any size — the
    /// blob lane. Bytes stream through the leased verifying writer in bounded
    /// blocks; nothing here assumes the tensor grid constant, and peak memory
    /// is one block regardless of blob size.
    pub fn admit_source_range<S>(
        &self,
        source: &S,
        offset: u64,
        length: u64,
    ) -> Result<AdmittedObject, StoreError>
    where
        S: ByteSource + ?Sized,
    {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut writer = self.writer()?;
        let mut buffer = vec![0_u8; VERIFY_BUFFER_SIZE];
        let mut position = offset;
        while position < end {
            let take = usize::try_from((end - position).min(VERIFY_BUFFER_SIZE as u64))
                .expect("a block never exceeds usize");
            source.read_exact_at(position, &mut buffer[..take])?;
            writer.write_all(&buffer[..take])?;
            position += take as u64;
        }
        writer.finish()
    }

    /// Opens a resident object for reading, refusing symlinks and every
    /// non-regular file at the digest path.
    pub fn open_object(&self, digest: &ObjectDigest) -> Result<File, StoreError> {
        let path = self.object_path(digest);

        #[cfg(unix)]
        let opened = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path);
        #[cfg(windows)]
        let opened = {
            // Windows cannot refuse symlink traversal at open, so the type
            // check precedes the open under the store's single-writer layout.
            //
            // The check is on the whole file type, not just symlinks: opening
            // a directory on Windows fails with ERROR_ACCESS_DENIED, which
            // would surface as a bare `Io` and make the refusal reason
            // platform-dependent for callers. This stat is already being paid
            // here for the symlink check, so widening it to `!is_file()`
            // costs no extra syscall, keeps `NotARegularFile` meaning the same
            // thing on every platform, and skips an open that was going to
            // fail anyway.
            match fs::symlink_metadata(&path) {
                Ok(metadata) if !metadata.file_type().is_file() => {
                    return Err(StoreError::NotARegularFile { digest: *digest });
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    return Err(StoreError::Missing { digest: *digest });
                }
                Err(error) => return Err(error.into()),
            }
            OpenOptions::new().read(true).open(&path)
        };

        let file = match opened {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(StoreError::Missing { digest: *digest });
            }
            #[cfg(unix)]
            Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
                return Err(StoreError::NotARegularFile { digest: *digest });
            }
            Err(error) => return Err(error.into()),
        };
        if !file.metadata()?.is_file() {
            return Err(StoreError::NotARegularFile { digest: *digest });
        }
        Ok(file)
    }

    /// Rehashes a resident object and reports corruption without deleting it;
    /// repair belongs to a newly verified admission, never to the reader.
    pub fn verify(&self, digest: &ObjectDigest) -> Result<u64, StoreError> {
        let mut file = self.open_object(digest)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; VERIFY_BUFFER_SIZE];
        let mut length = 0_u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            length += read as u64;
        }
        let actual = ObjectDigest::from_bytes(hasher.finalize().into());
        if actual != *digest {
            return Err(StoreError::CorruptObject {
                expected: *digest,
                actual,
            });
        }
        Ok(length)
    }

    /// Reclaims abandoned library temps older than the caller's grace.
    ///
    /// Only known TensorFS temp shapes are considered. Each candidate must be
    /// past the grace, hold no live advisory lease, and still be the exact
    /// inode observed at listing before it is unlinked, so a live writer in
    /// another process is never removed and a recycled name is never touched.
    /// Cadence and grace policy belong to the consumer; there is no background
    /// thread and no startup scan.
    pub fn collect_abandoned_temps(&self, grace: Duration) -> Result<TempCollection, StoreError> {
        if grace.is_zero() {
            return Err(StoreError::InvalidGrace);
        }
        let cutoff = SystemTime::now().checked_sub(grace);
        let mut report = TempCollection::default();
        for entry in fs::read_dir(self.tmp_dir())? {
            let Ok(entry) = entry else { continue };
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.starts_with(TEMP_PREFIX) || !name.ends_with(TEMP_SUFFIX) {
                continue;
            }
            let Ok(listed) = entry.metadata() else {
                continue;
            };
            if !listed.is_file() {
                continue;
            }
            report.examined += 1;
            let old_enough = match (listed.modified().ok(), cutoff) {
                (Some(modified), Some(cutoff)) => modified <= cutoff,
                _ => false,
            };
            if !old_enough {
                continue;
            }
            self.reclaim_candidate(&entry.path(), &temp_identity(&listed), &mut report);
        }
        if report.deleted > 0 {
            fsync_dir(&self.tmp_dir())?;
        }
        Ok(report)
    }

    /// Visits every resident object digest and length. GC's mark phase needs
    /// the complete resident set; visit order is unspecified.
    ///
    /// Reachability GC and remote sync are the only callers, so this and the
    /// three helpers below follow their features. Without the gate a
    /// `store`-only build -- the one the Python extension links -- warns that
    /// they are dead, which is true and not a defect.
    #[cfg(any(feature = "sync", feature = "workspace"))]
    pub(crate) fn each_object(
        &self,
        mut visit: impl FnMut(ObjectDigest, u64),
    ) -> Result<(), StoreError> {
        let namespace = self.sha256_dir();
        for first in fs::read_dir(&namespace)? {
            let Ok(first) = first else { continue };
            if !first.path().is_dir() {
                continue;
            }
            for second in fs::read_dir(first.path())? {
                let Ok(second) = second else { continue };
                if !second.path().is_dir() {
                    continue;
                }
                for object in fs::read_dir(second.path())? {
                    let Ok(object) = object else { continue };
                    let name = object.file_name();
                    let Some(name) = name.to_str() else { continue };
                    let Some(digest) = parse_digest_hex(name) else {
                        continue;
                    };
                    let Ok(metadata) = object.metadata() else {
                        continue;
                    };
                    if metadata.is_file() {
                        visit(digest, metadata.len());
                    }
                }
            }
        }
        Ok(())
    }

    /// Unlinks one object the caller has proven unreferenced. The quarantine
    /// protocol owns that proof; this helper only re-checks the path still
    /// names a regular file so nothing else is ever removed.
    /// A cheap resident-existence stat for callers that already verified the
    /// bytes and only need to re-check presence inside a metadata transaction.
    #[cfg(any(feature = "sync", feature = "workspace"))]
    pub(crate) fn exists(&self, digest: &ObjectDigest) -> bool {
        self.object_path(digest).is_file()
    }

    #[cfg(any(feature = "sync", feature = "workspace"))]
    pub(crate) fn remove_object(&self, digest: &ObjectDigest) -> Result<bool, StoreError> {
        let path = self.object_path(digest);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                fs::remove_file(&path)?;
                fsync_dir(path.parent().expect("object paths always have a parent"))?;
                Ok(true)
            }
            Ok(_) => Ok(false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// One candidate's reclaim decision. Every guard failure retains the file.
    pub(crate) fn reclaim_candidate(
        &self,
        path: &Path,
        listed: &TempIdentity,
        report: &mut TempCollection,
    ) {
        #[cfg(unix)]
        let opened = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path);
        #[cfg(not(unix))]
        let opened = OpenOptions::new().read(true).open(path);
        let Ok(file) = opened else { return };

        // The name may have been recycled between listing and open.
        let Ok(held) = file.metadata() else { return };
        let held = temp_identity(&held);
        if held != *listed {
            return;
        }

        // A live writer holds the lease for its whole write/verify/install
        // lifetime; failing to acquire it non-blocking retains the temp.
        if file.try_lock_exclusive().is_err() {
            return;
        }

        // Re-stat the path after acquiring the lease: the writer we locked
        // out may have finished and a new temp may own this name now.
        let Ok(current) = fs::symlink_metadata(path) else {
            return;
        };
        if temp_identity(&current) != held {
            return;
        }

        if fs::remove_file(path).is_ok() {
            report.deleted += 1;
            report.bytes_deleted += current.len();
        }
    }
}

/// A leased streaming admission. Dropping an unfinished writer removes its
/// temp; a killed process leaves the temp for the age-bounded collector.
#[derive(Debug)]
pub struct ObjectWriter<'store> {
    store: &'store ObjectStore,
    file: Option<File>,
    path: PathBuf,
    hasher: Sha256,
    written: u64,
    installed: bool,
}

impl ObjectWriter<'_> {
    /// Completes the admission, computing the digest from the written bytes.
    pub fn finish(mut self) -> Result<AdmittedObject, StoreError> {
        let file = self.file.take().expect("a writer finishes at most once");
        file.sync_all()?;
        let digest = self.take_digest();
        self.install(file, digest)
    }

    /// Completes the admission only if the bytes match a plan's exact
    /// expected digest and length.
    pub fn finish_expecting(
        mut self,
        expected: ObjectDigest,
        length: u64,
    ) -> Result<AdmittedObject, StoreError> {
        let file = self.file.take().expect("a writer finishes at most once");
        if self.written != length {
            return Err(StoreError::LengthMismatch {
                expected: length,
                actual: self.written,
            });
        }
        file.sync_all()?;
        let digest = self.take_digest();
        if digest != expected {
            return Err(StoreError::DigestMismatch {
                expected,
                actual: digest,
            });
        }
        self.install(file, digest)
    }

    fn take_digest(&mut self) -> ObjectDigest {
        let hasher = std::mem::replace(&mut self.hasher, Sha256::new());
        ObjectDigest::from_bytes(hasher.finalize().into())
    }

    fn install(&mut self, file: File, digest: ObjectDigest) -> Result<AdmittedObject, StoreError> {
        let final_path = self.store.object_path(&digest);
        let leaf = final_path
            .parent()
            .expect("object paths always have a parent");
        fs::create_dir_all(leaf)?;
        // The mode lands on the temp inode BEFORE it is linked into the
        // object tree, so a resident object is never briefly writable. The
        // write fd above is already open and unaffected.
        set_read_only(&self.path)?;
        match fs::hard_link(&self.path, &final_path) {
            Ok(()) => {
                fsync_dir(leaf)?;
                fsync_dir(leaf.parent().expect("fanout directories nest"))?;
                fsync_dir(&self.store.sha256_dir())?;
                drop(file);
                let _ = fs::remove_file(&self.path);
                self.installed = true;
                Ok(AdmittedObject {
                    digest,
                    length: self.written,
                    preexisting: false,
                })
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                drop(file);
                // Existing bytes are accepted only after rehash verification.
                // A divergent resident is reported, never replaced.
                self.store.verify(&digest)?;
                let _ = fs::remove_file(&self.path);
                self.installed = true;
                Ok(AdmittedObject {
                    digest,
                    length: self.written,
                    preexisting: true,
                })
            }
            Err(error) => Err(error.into()),
        }
    }
}

impl Write for ObjectWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let file = self.file.as_mut().expect("a writer finishes at most once");
        file.write_all(buffer)?;
        self.hasher.update(buffer);
        self.written += buffer.len() as u64;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file
            .as_mut()
            .expect("a writer finishes at most once")
            .flush()
    }
}

impl Drop for ObjectWriter<'_> {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            drop(file);
        }
        if !self.installed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TempIdentity {
    length: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    creation_time: u64,
}

pub(crate) fn temp_identity(metadata: &fs::Metadata) -> TempIdentity {
    TempIdentity {
        length: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(windows)]
        creation_time: metadata.creation_time(),
    }
}

/// Only `each_object` reads object filenames back, so this shares its gate.
#[cfg(any(feature = "sync", feature = "workspace"))]
fn parse_digest_hex(name: &str) -> Option<ObjectDigest> {
    if name.len() != 64 {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in name.as_bytes().chunks_exact(2).enumerate() {
        let digits = std::str::from_utf8(pair).ok()?;
        if digits.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return None;
        }
        bytes[index] = u8::from_str_radix(digits, 16).ok()?;
    }
    Some(ObjectDigest::from_bytes(bytes))
}

/// Installs the immutable mode HF and git both use: readable by everyone,
/// writable by nobody.
///
/// This stops accidents — an editor, a `>` redirection, an `open("ab")`
/// through a snapshot symlink — which is the entire practical threat on a
/// pod. A determined chmod-and-edit by the store's owner is explicitly not
/// defended; `verify` is the scrub for suspicion.
///
/// Unix only. Windows' read-only ATTRIBUTE would make the store's own
/// `remove_file` — of the leased temp here, and of a quarantined object in
/// GC — fail, so Windows keeps inherited ACLs and this is a no-op there.
pub(crate) fn set_read_only(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o444))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// One symlink created and removed at the store root, answering whether this
/// filesystem supports the projection's links at all.
///
/// The root and not `tmp/`: a process killed between the create and the
/// remove would otherwise strand a file in the directory whose emptiness is a
/// crash-safety assertion. At the root a stranded probe is inert — nothing
/// reads it, and the next open uses a fresh unique name.
fn probe_symlinks(root: &Path) -> bool {
    let probe = root.join(format!(
        ".symlink-probe-{}-{}",
        process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_file(&probe);
    #[cfg(unix)]
    let created = std::os::unix::fs::symlink("probe-target", &probe);
    #[cfg(windows)]
    let created = std::os::windows::fs::symlink_file("probe-target", &probe);
    #[cfg(not(any(unix, windows)))]
    let created = Err(io::Error::from(io::ErrorKind::Unsupported));
    let supported = created.is_ok();
    let _ = fs::remove_file(&probe);
    supported
}

/// Directory fsync seals a rename/link against the containing directory. The
/// standard library cannot open a Windows directory handle, so the seal is
/// POSIX-only; NTFS journals its own metadata.
fn fsync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recycled_candidate_name_is_never_reclaimed() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-store-unit-{}-{}",
            process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let store = ObjectStore::open(&root).unwrap();
        let path = store.tmp_dir().join("obj-0-0-0.tmp");

        fs::write(&path, b"abandoned bytes").unwrap();
        let listed = temp_identity(&fs::symlink_metadata(&path).unwrap());

        // The abandoned temp is adopted and its name recycled by a new,
        // different temp between the collector's listing and its open.
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"a brand new writer's bytes, longer").unwrap();

        let mut report = TempCollection::default();
        store.reclaim_candidate(&path, &listed, &mut report);

        assert_eq!(report, TempCollection::default());
        assert!(path.exists(), "the recycled temp must survive");
        fs::remove_dir_all(&root).unwrap();
    }
}
