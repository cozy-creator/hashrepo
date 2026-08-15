//! The writable COW workspace mount.
//!
//! Ordinary file semantics over immutable CAS objects, with no whole-file
//! materialization anywhere. Namespace mutations (create, mkdir, rename,
//! unlink, symlink, chmod) commit durably as their own metadata generations
//! the moment they return. File bytes land in a per-inode dirty overlay — an
//! unlinked spill file holding only the dirty bytes at their logical offsets —
//! and reads merge committed records with the newest dirty ranges. A durable
//! flush (`fsync`, `fdatasync`, or any close of a dirty handle) composes only
//! the 64 MiB-grid objects the dirty ranges intersect, admits exactly those
//! through the object store, and commits one new generation; untouched
//! objects keep their digests without being read or rewritten. A daemon
//! killed before the flush loses exactly the dirty bytes: recovery exposes
//! the last committed generation, never a hybrid.
//!
//! # Coherence and mmap
//!
//! Every open rides the kernel page cache (no `FOPEN_DIRECT_IO`), which is
//! what makes `mmap` work at all: page faults and writeback travel the same
//! FUSE `read`/`write` ops as ordinary I/O, so mapped-page writeback lands in
//! the dirty overlay like any write. The contract: all access through this
//! mount shares one page cache, so reads and mappings observe writes from
//! other descriptors immediately (single-host POSIX coherence); committed
//! content only ever changes through this mount's own operations (one daemon
//! owns the workspace, and composing rewrites objects without changing
//! logical bytes), so cached pages never go stale. Durability is unchanged:
//! FLUSH never composes, RELEASE composes best-effort, `fsync` is the
//! guarantee — and `msync(MS_SYNC)` on a shared mapping IS an `fsync` (the
//! kernel writes the dirty pages back and then issues FUSE fsync on the
//! mapped file). The serving process must never touch its own mountpoint;
//! that same-process writeback deadlock is why serving always lives in a
//! dedicated daemon process.
//!
//! # Advisory locks
//!
//! POSIX record locks are served from a per-mount [`locks::LockTable`]:
//! session state that dies with the mount and never enters a snapshot.
//! `setlk` grants or refuses immediately — a blocking `F_SETLKW` under
//! contention is refused with `EAGAIN` rather than suspending the FUSE
//! dispatch loop; blocking waits are deferred until a real consumer needs
//! them. Windows share modes cannot be expressed on Linux FUSE at all; they
//! arrive with the WinFsp adapter.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use fuser::{
    BackgroundSession, FileAttr, FileType, Filesystem, KernelConfig, MountOption, ReplyAttr,
    ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyLock, ReplyOpen, Request, TimeOrNow,
};

use tensorfs_core::planner::{MAX_OBJECT_SIZE, PlannerId};
use tensorfs_core::tfm1::{Entry, FileRecord, Snapshot};
use tensorfs_core::workspace::{LeaseId, LeaseKind, Mutation, WorkspaceError, WorkspaceStore};

use crate::MountError;
use crate::locks::LockTable;

/// Namespace facts change under this mount, so the kernel may only cache them
/// briefly; one daemon owns the workspace, so one second is safe and cheap.
const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;
const LEASE_HOLDER: &str = "tensorfsd";

static SPILL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The dirty overlay of one regular file: an unlinked spill file holding only
/// the dirty bytes at their logical offsets, plus the coalesced range list.
/// The spill is disk-backed and self-cleaning on process death, so overlay
/// memory stays O(ranges) regardless of how many bytes are dirty.
struct Dirty {
    spill: File,
    /// start -> length; disjoint and coalesced.
    ranges: BTreeMap<u64, u64>,
}

impl Dirty {
    fn new() -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "tensorfsd-spill-{}-{}",
            process::id(),
            SPILL_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let spill = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        // Unlink-open: the bytes live exactly as long as this handle.
        std::fs::remove_file(&path)?;
        Ok(Self {
            spill,
            ranges: BTreeMap::new(),
        })
    }

    fn add_range(&mut self, start: u64, length: u64) {
        let mut new_start = start;
        let mut new_end = start + length;
        let touching: Vec<u64> = self
            .ranges
            .range(..=new_end)
            .filter(|(s, l)| **s + **l >= new_start)
            .map(|(s, _)| *s)
            .collect();
        for s in touching {
            let l = self.ranges.remove(&s).expect("listed range exists");
            new_start = new_start.min(s);
            new_end = new_end.max(s + l);
        }
        self.ranges.insert(new_start, new_end - new_start);
    }

    fn shrink_to(&mut self, size: u64) {
        let beyond: Vec<u64> = self.ranges.range(size..).map(|(&s, _)| s).collect();
        for s in beyond {
            self.ranges.remove(&s);
        }
        if let Some((&s, &l)) = self.ranges.range(..size).next_back() {
            if s + l > size {
                self.ranges.insert(s, size - s);
            }
        }
    }

    fn overlaps(&self, start: u64, end: u64) -> bool {
        self.ranges
            .range(..end)
            .next_back()
            .is_some_and(|(&s, &l)| s < end && s + l > start)
    }
}

struct WFile {
    executable: bool,
    committed: Vec<FileRecord>,
    committed_size: u64,
    /// The served logical size; equals `committed_size` while clean.
    size: u64,
    dirty: Option<Dirty>,
    /// Every (parent inode, name) dirent naming this file.
    links: Vec<(u64, String)>,
    open_handles: u32,
    unlink_lease: Option<LeaseId>,
}

enum WNode {
    Directory {
        parent: u64,
        name: String,
        children: BTreeMap<String, u64>,
    },
    File(WFile),
    Symlink {
        parent: u64,
        name: String,
        target: String,
    },
}

fn records_size(records: &[FileRecord]) -> u64 {
    records
        .iter()
        .map(|record| match record {
            FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
        })
        .sum()
}

struct Handle {
    writable: bool,
    objects: HashMap<[u8; 32], File>,
}

struct WorkspaceFs {
    meta: WorkspaceStore,
    workspace: String,
    uid: u32,
    gid: u32,
    nodes: HashMap<u64, WNode>,
    next_ino: u64,
    next_fh: u64,
    /// Per open handle: owning inode, whether the handle may write, and one
    /// lazily opened `File` per committed object read.
    handles: HashMap<u64, Handle>,
    /// Advisory POSIX record locks; mount-lifetime state, never persisted.
    locks: LockTable,
}

fn errno(error: &WorkspaceError) -> i32 {
    match error {
        WorkspaceError::Exists(_) | WorkspaceError::CaseFoldCollision(_) => libc::EEXIST,
        WorkspaceError::Missing(_) | WorkspaceError::UnknownWorkspace(_) => libc::ENOENT,
        WorkspaceError::NotADirectory(_) => libc::ENOTDIR,
        WorkspaceError::NotAFile(_) => libc::EISDIR,
        WorkspaceError::NotEmpty(_) => libc::ENOTEMPTY,
        WorkspaceError::RenameIntoSelf(_) => libc::EINVAL,
        WorkspaceError::Tfm1(_) => libc::EINVAL,
        _ => libc::EIO,
    }
}

/// Builds the mutable inode table from the committed head tree. The decoded
/// snapshot's sorted-parent-first guarantee makes single-pass loading safe,
/// exactly as in the read-only mount.
fn load_tree(snapshot: &Snapshot) -> (HashMap<u64, WNode>, u64) {
    let mut nodes: HashMap<u64, WNode> = HashMap::new();
    nodes.insert(
        ROOT_INO,
        WNode::Directory {
            parent: ROOT_INO,
            name: String::new(),
            children: BTreeMap::new(),
        },
    );
    let mut next_ino = ROOT_INO + 1;
    let mut ino_by_path: HashMap<&str, u64> = HashMap::new();

    for (path, entry) in snapshot.entries() {
        let (parent_path, name) = match path.rsplit_once('/') {
            Some((parent, name)) => (parent, name),
            None => ("", path.as_str()),
        };
        let parent_ino = if parent_path.is_empty() {
            ROOT_INO
        } else {
            ino_by_path[parent_path]
        };
        let ino = match entry {
            Entry::Directory => {
                let ino = next_ino;
                next_ino += 1;
                nodes.insert(
                    ino,
                    WNode::Directory {
                        parent: parent_ino,
                        name: name.to_owned(),
                        children: BTreeMap::new(),
                    },
                );
                ino
            }
            Entry::File {
                executable,
                logical_size,
                records,
                ..
            } => {
                let ino = next_ino;
                next_ino += 1;
                nodes.insert(
                    ino,
                    WNode::File(WFile {
                        executable: *executable,
                        committed: records.clone(),
                        committed_size: *logical_size,
                        size: *logical_size,
                        dirty: None,
                        links: vec![(parent_ino, name.to_owned())],
                        open_handles: 0,
                        unlink_lease: None,
                    }),
                );
                ino
            }
            Entry::Symlink { target } => {
                let ino = next_ino;
                next_ino += 1;
                nodes.insert(
                    ino,
                    WNode::Symlink {
                        parent: parent_ino,
                        name: name.to_owned(),
                        target: target.clone(),
                    },
                );
                ino
            }
            Entry::Hardlink { ordinal } => {
                let carrier = snapshot
                    .ordinal_path(*ordinal)
                    .expect("a decoded snapshot only names assigned ordinals");
                let ino = ino_by_path[carrier];
                if let Some(WNode::File(file)) = nodes.get_mut(&ino) {
                    file.links.push((parent_ino, name.to_owned()));
                }
                ino
            }
        };
        ino_by_path.insert(path, ino);
        if let Some(WNode::Directory { children, .. }) = nodes.get_mut(&parent_ino) {
            children.insert(name.to_owned(), ino);
        }
    }
    (nodes, next_ino)
}

impl WorkspaceFs {
    fn dir_path(&self, mut ino: u64) -> String {
        let mut segments = Vec::new();
        while ino != ROOT_INO {
            let Some(WNode::Directory { parent, name, .. }) = self.nodes.get(&ino) else {
                break;
            };
            segments.push(name.clone());
            ino = *parent;
        }
        segments.reverse();
        segments.join("/")
    }

    fn child_path(&self, parent: u64, name: &str) -> String {
        let parent_path = self.dir_path(parent);
        if parent_path.is_empty() {
            name.to_owned()
        } else {
            format!("{parent_path}/{name}")
        }
    }

    fn file_path(&self, file: &WFile) -> Option<String> {
        file.links
            .first()
            .map(|(parent, name)| self.child_path(*parent, name))
    }

    fn commit(&self, mutations: &[Mutation]) -> Result<(), i32> {
        self.meta
            .commit_generation(&self.workspace, mutations)
            .map(|_| ())
            .map_err(|error| errno(&error))
    }

    fn attr(&self, ino: u64) -> Option<FileAttr> {
        let node = self.nodes.get(&ino)?;
        let (kind, size, perm, nlink) = match node {
            WNode::Directory { children, .. } => {
                let subdirs = children
                    .values()
                    .filter(|child| matches!(self.nodes.get(child), Some(WNode::Directory { .. })))
                    .count() as u32;
                (FileType::Directory, 0, 0o755, 2 + subdirs)
            }
            WNode::File(file) => {
                let perm = if file.executable { 0o755 } else { 0o644 };
                (
                    FileType::RegularFile,
                    file.size,
                    perm,
                    file.links.len().max(1) as u32,
                )
            }
            WNode::Symlink { target, .. } => (FileType::Symlink, target.len() as u64, 0o777, 1),
        };
        Some(FileAttr {
            ino,
            size,
            blocks: size.div_ceil(512),
            atime: SystemTime::UNIX_EPOCH,
            mtime: SystemTime::UNIX_EPOCH,
            ctime: SystemTime::UNIX_EPOCH,
            crtime: SystemTime::UNIX_EPOCH,
            kind,
            perm,
            nlink,
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            blksize: 4096,
            flags: 0,
        })
    }

    fn alloc_ino(&mut self) -> u64 {
        let ino = self.next_ino;
        self.next_ino += 1;
        ino
    }

    fn alloc_fh(&mut self, writable: bool) -> u64 {
        self.next_fh += 1;
        self.handles.insert(
            self.next_fh,
            Handle {
                writable,
                objects: HashMap::new(),
            },
        );
        self.next_fh
    }

    /// Fills `buffer` from committed records starting at `offset`; bytes past
    /// `committed_size` stay zero. The per-handle cache keeps one open `File`
    /// per referenced object, and bytes are trusted from admission-time
    /// verification, never rehashed here.
    fn fill_committed(
        &mut self,
        ino: u64,
        fh: Option<u64>,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<(), i32> {
        let Some(WNode::File(file)) = self.nodes.get(&ino) else {
            return Err(libc::ENOENT);
        };
        let mut plan: Vec<(u64, u64, u64, Option<tensorfs_core::object::ObjectDigest>)> =
            Vec::new();
        let mut start = 0_u64;
        for record in &file.committed {
            let (length, digest) = match record {
                FileRecord::Data { digest, length } => (*length, Some(*digest)),
                FileRecord::Hole { length } => (*length, None),
            };
            let end = start + length;
            if end > offset && start < offset + buffer.len() as u64 {
                let from = offset.max(start);
                let to = end.min(offset + buffer.len() as u64);
                plan.push((from, to, start, digest));
            }
            start = end;
        }
        for (from, to, record_start, digest) in plan {
            let slice_range = (from - offset) as usize..(to - offset) as usize;
            match digest {
                None => {}
                Some(digest) => {
                    let store = self.meta.store();
                    match fh.and_then(|fh| self.handles.get_mut(&fh)) {
                        Some(Handle { objects, .. }) => {
                            if !objects.contains_key(digest.as_bytes()) {
                                let opened = store.open_object(&digest).map_err(|_| libc::EIO)?;
                                objects.insert(*digest.as_bytes(), opened);
                            }
                            objects[digest.as_bytes()]
                                .read_exact_at(&mut buffer[slice_range], from - record_start)
                                .map_err(|_| libc::EIO)?;
                        }
                        None => {
                            let opened = store.open_object(&digest).map_err(|_| libc::EIO)?;
                            opened
                                .read_exact_at(&mut buffer[slice_range], from - record_start)
                                .map_err(|_| libc::EIO)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Serves one read by merging committed records with the newest dirty
    /// ranges, exactly in that order.
    fn read_merged(
        &mut self,
        ino: u64,
        fh: u64,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<(), i32> {
        self.fill_committed(ino, Some(fh), offset, buffer)?;
        let Some(WNode::File(file)) = self.nodes.get(&ino) else {
            return Err(libc::ENOENT);
        };
        if let Some(dirty) = &file.dirty {
            let end = offset + buffer.len() as u64;
            for (&s, &l) in dirty.ranges.range(..end) {
                let range_end = s + l;
                if range_end <= offset {
                    continue;
                }
                let from = offset.max(s);
                let to = end.min(range_end);
                dirty
                    .spill
                    .read_exact_at(
                        &mut buffer[(from - offset) as usize..(to - offset) as usize],
                        from,
                    )
                    .map_err(|_| libc::EIO)?;
            }
        }
        Ok(())
    }

    /// Returns the committed record that covers exactly `[start, start + length)`.
    fn exact_committed(&self, file: &WFile, start: u64, length: u64) -> Option<FileRecord> {
        let mut position = 0_u64;
        for record in &file.committed {
            let record_length = match record {
                FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
            };
            if position == start && record_length == length {
                return Some(record.clone());
            }
            position += record_length;
            if position > start {
                break;
            }
        }
        None
    }

    /// Composes and durably commits one dirty file: only grid slots the dirty
    /// ranges intersect (or whose committed boundaries no longer line up) are
    /// read and re-admitted; every other slot keeps its committed record
    /// untouched. A fully unlinked file has no path to commit under and its
    /// bytes die with the last handle, which is exactly unlink-open semantics.
    fn compose_commit(&mut self, ino: u64) -> Result<(), i32> {
        let Some(WNode::File(file)) = self.nodes.get(&ino) else {
            return Ok(());
        };
        if file.dirty.is_none() {
            return Ok(());
        }
        let Some(path) = self.file_path(file) else {
            return Ok(());
        };
        let size = file.size;
        let committed_size = file.committed_size;

        let mut records: Vec<FileRecord> = Vec::new();
        let mut slot_start = 0_u64;
        while slot_start < size {
            let slot_end = (slot_start + MAX_OBJECT_SIZE).min(size);
            let slot_length = slot_end - slot_start;
            let dirty_hit = match &self.nodes[&ino] {
                WNode::File(file) => file
                    .dirty
                    .as_ref()
                    .expect("dirty state checked above")
                    .overlaps(slot_start, slot_end),
                _ => false,
            };
            if !dirty_hit {
                let reused = match &self.nodes[&ino] {
                    WNode::File(file) => self.exact_committed(file, slot_start, slot_length),
                    _ => None,
                };
                if let Some(record) = reused {
                    records.push(record);
                    slot_start = slot_end;
                    continue;
                }
                if slot_start >= committed_size {
                    records.push(FileRecord::Hole {
                        length: slot_length,
                    });
                    slot_start = slot_end;
                    continue;
                }
            }
            let mut buffer = vec![0_u8; slot_length as usize];
            self.fill_committed(ino, None, slot_start, &mut buffer)?;
            if let Some(WNode::File(file)) = self.nodes.get(&ino) {
                let dirty = file.dirty.as_ref().expect("dirty state checked above");
                for (&s, &l) in dirty.ranges.range(..slot_end) {
                    let range_end = s + l;
                    if range_end <= slot_start {
                        continue;
                    }
                    let from = slot_start.max(s);
                    let to = slot_end.min(range_end);
                    dirty
                        .spill
                        .read_exact_at(
                            &mut buffer[(from - slot_start) as usize..(to - slot_start) as usize],
                            from,
                        )
                        .map_err(|_| libc::EIO)?;
                }
            }
            let admitted = self
                .meta
                .store()
                .put_bytes(&buffer)
                .map_err(|_| libc::EIO)?;
            records.push(FileRecord::Data {
                digest: admitted.digest(),
                length: slot_length,
            });
            slot_start = slot_end;
        }

        // TFM1 refuses adjacent holes, so consecutive hole slots merge.
        let mut merged: Vec<FileRecord> = Vec::new();
        for record in records {
            match (merged.last_mut(), &record) {
                (Some(FileRecord::Hole { length: last }), FileRecord::Hole { length }) => {
                    *last += *length;
                }
                _ => merged.push(record),
            }
        }

        self.commit(&[Mutation::SetRecords {
            path,
            records: merged.clone(),
        }])?;
        if let Some(WNode::File(file)) = self.nodes.get_mut(&ino) {
            file.committed = merged;
            file.committed_size = size;
            file.dirty = None;
        }
        Ok(())
    }

    /// Removes one dirent from a file, pinning the inode with an unlinked
    /// lease first when open handles must keep reading past the last link.
    fn unlink_file(&mut self, ino: u64, parent: u64, name: &str) -> Result<(), i32> {
        let path = self.child_path(parent, name);
        let needs_lease = match self.nodes.get(&ino) {
            Some(WNode::File(file)) => {
                file.links.len() == 1 && file.open_handles > 0 && file.unlink_lease.is_none()
            }
            _ => false,
        };
        if needs_lease {
            let lease = self
                .meta
                .acquire_lease(&self.workspace, &path, LeaseKind::Unlinked, LEASE_HOLDER)
                .map_err(|error| errno(&error))?;
            if let Some(WNode::File(file)) = self.nodes.get_mut(&ino) {
                file.unlink_lease = Some(lease);
            }
        }
        self.commit(&[Mutation::Unlink { path }])?;
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.remove(name);
        }
        let drop_node = match self.nodes.get_mut(&ino) {
            Some(WNode::File(file)) => {
                file.links.retain(|(link_parent, link_name)| {
                    !(*link_parent == parent && link_name == name)
                });
                file.links.is_empty() && file.open_handles == 0
            }
            Some(WNode::Symlink { .. }) => true,
            _ => false,
        };
        if drop_node {
            self.nodes.remove(&ino);
            self.locks.forget_ino(ino);
        }
        Ok(())
    }

    fn lookup_child(&self, parent: u64, name: &OsStr) -> Option<u64> {
        let name = name.to_str()?;
        match self.nodes.get(&parent)? {
            WNode::Directory { children, .. } => children.get(name).copied(),
            _ => None,
        }
    }
}

impl Filesystem for WorkspaceFs {
    fn init(&mut self, _req: &Request<'_>, config: &mut KernelConfig) -> Result<(), libc::c_int> {
        // Without this capability the kernel serves POSIX locks locally and
        // never consults the table; with it, lock state is mount-owned and
        // uniform across the future macFUSE/WinFsp backends.
        let _ = config.add_capabilities(fuser::consts::FUSE_POSIX_LOCKS);
        Ok(())
    }

    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        match self
            .lookup_child(parent, name)
            .and_then(|ino| self.attr(ino))
        {
            Some(attr) => reply.entry(&TTL, &attr, 0),
            None => reply.error(libc::ENOENT),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        match self.attr(ino) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(libc::ENOENT),
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        match self.nodes.get(&ino) {
            Some(WNode::Symlink { target, .. }) => reply.data(target.as_bytes()),
            Some(_) => reply.error(libc::EINVAL),
            None => reply.error(libc::ENOENT),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let Some(WNode::Directory { children, .. }) = self.nodes.get(&ino) else {
            reply.error(libc::ENOTDIR);
            return;
        };
        let mut entries: Vec<(u64, FileType, String)> = vec![
            (ino, FileType::Directory, ".".to_owned()),
            (ino, FileType::Directory, "..".to_owned()),
        ];
        for (name, child) in children {
            let kind = match self.nodes.get(child) {
                Some(WNode::Directory { .. }) => FileType::Directory,
                Some(WNode::Symlink { .. }) => FileType::Symlink,
                _ => FileType::RegularFile,
            };
            entries.push((*child, kind, name.clone()));
        }
        for (index, (child, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            if reply.add(child, (index + 1) as i64, kind, &name) {
                break;
            }
        }
        reply.ok();
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let Some(name) = name.to_str().map(str::to_owned) else {
            reply.error(libc::EINVAL);
            return;
        };
        let path = self.child_path(parent, &name);
        if let Err(code) = self.commit(&[Mutation::Mkdir { path }]) {
            reply.error(code);
            return;
        }
        let ino = self.alloc_ino();
        self.nodes.insert(
            ino,
            WNode::Directory {
                parent,
                name: name.clone(),
                children: BTreeMap::new(),
            },
        );
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.insert(name, ino);
        }
        match self.attr(ino) {
            Some(attr) => reply.entry(&TTL, &attr, 0),
            None => reply.error(libc::EIO),
        }
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: fuser::ReplyCreate,
    ) {
        let Some(name) = name.to_str().map(str::to_owned) else {
            reply.error(libc::EINVAL);
            return;
        };
        let path = self.child_path(parent, &name);
        let executable = mode & 0o111 != 0;
        if let Err(code) = self.commit(&[Mutation::CreateFile {
            path,
            executable,
            planner: PlannerId::RawFixed64mV1,
            records: Vec::new(),
        }]) {
            reply.error(code);
            return;
        }
        let ino = self.alloc_ino();
        self.nodes.insert(
            ino,
            WNode::File(WFile {
                executable,
                committed: Vec::new(),
                committed_size: 0,
                size: 0,
                dirty: None,
                links: vec![(parent, name.clone())],
                open_handles: 1,
                unlink_lease: None,
            }),
        );
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.insert(name, ino);
        }
        let fh = self.alloc_fh(true);
        match self.attr(ino) {
            Some(attr) => reply.created(&TTL, &attr, 0, fh, 0),
            None => reply.error(libc::EIO),
        }
    }

    fn symlink(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        link_name: &OsStr,
        target: &Path,
        reply: ReplyEntry,
    ) {
        let (Some(name), Some(target)) = (
            link_name.to_str().map(str::to_owned),
            target.to_str().map(str::to_owned),
        ) else {
            reply.error(libc::EINVAL);
            return;
        };
        let path = self.child_path(parent, &name);
        if let Err(code) = self.commit(&[Mutation::Symlink {
            path,
            target: target.clone(),
        }]) {
            reply.error(code);
            return;
        }
        let ino = self.alloc_ino();
        self.nodes.insert(
            ino,
            WNode::Symlink {
                parent,
                name: name.clone(),
                target,
            },
        );
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.insert(name, ino);
        }
        match self.attr(ino) {
            Some(attr) => reply.entry(&TTL, &attr, 0),
            None => reply.error(libc::EIO),
        }
    }

    fn link(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        newparent: u64,
        newname: &OsStr,
        reply: ReplyEntry,
    ) {
        let Some(name) = newname.to_str().map(str::to_owned) else {
            reply.error(libc::EINVAL);
            return;
        };
        let Some(WNode::File(file)) = self.nodes.get(&ino) else {
            reply.error(libc::EPERM);
            return;
        };
        let Some(target) = self.file_path(file) else {
            reply.error(libc::ENOENT);
            return;
        };
        let path = self.child_path(newparent, &name);
        if let Err(code) = self.commit(&[Mutation::Hardlink { path, target }]) {
            reply.error(code);
            return;
        }
        if let Some(WNode::File(file)) = self.nodes.get_mut(&ino) {
            file.links.push((newparent, name.clone()));
        }
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&newparent) {
            children.insert(name, ino);
        }
        match self.attr(ino) {
            Some(attr) => reply.entry(&TTL, &attr, 0),
            None => reply.error(libc::EIO),
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(name) = name.to_str().map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(ino) = self.lookup_child(parent, OsStr::new(&name)) else {
            reply.error(libc::ENOENT);
            return;
        };
        if matches!(self.nodes.get(&ino), Some(WNode::Directory { .. })) {
            reply.error(libc::EISDIR);
            return;
        }
        match self.unlink_file(ino, parent, &name) {
            Ok(()) => reply.ok(),
            Err(code) => reply.error(code),
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(name) = name.to_str().map(str::to_owned) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(ino) = self.lookup_child(parent, OsStr::new(&name)) else {
            reply.error(libc::ENOENT);
            return;
        };
        let path = self.child_path(parent, &name);
        if let Err(code) = self.commit(&[Mutation::Rmdir { path }]) {
            reply.error(code);
            return;
        }
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.remove(&name);
        }
        self.nodes.remove(&ino);
        reply.ok();
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        flags: u32,
        reply: ReplyEmpty,
    ) {
        if flags != 0 {
            reply.error(libc::EINVAL);
            return;
        }
        let (Some(name), Some(newname)) = (
            name.to_str().map(str::to_owned),
            newname.to_str().map(str::to_owned),
        ) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(ino) = self.lookup_child(parent, OsStr::new(&name)) else {
            reply.error(libc::ENOENT);
            return;
        };
        let from = self.child_path(parent, &name);
        let to = self.child_path(newparent, &newname);

        // POSIX rename replaces an existing target; the metadata layer's
        // rename refuses occupied names, so replacement rides the same
        // generation as an explicit removal of the target.
        let mut mutations = Vec::new();
        if let Some(existing) = self.lookup_child(newparent, OsStr::new(&newname)) {
            match self.nodes.get(&existing) {
                Some(WNode::Directory { .. }) => {
                    mutations.push(Mutation::Rmdir { path: to.clone() })
                }
                Some(WNode::File(file)) => {
                    let needs_lease = file.links.len() == 1
                        && file.open_handles > 0
                        && file.unlink_lease.is_none();
                    if needs_lease {
                        match self.meta.acquire_lease(
                            &self.workspace,
                            &to,
                            LeaseKind::Unlinked,
                            LEASE_HOLDER,
                        ) {
                            Ok(lease) => {
                                if let Some(WNode::File(file)) = self.nodes.get_mut(&existing) {
                                    file.unlink_lease = Some(lease);
                                }
                            }
                            Err(error) => {
                                reply.error(errno(&error));
                                return;
                            }
                        }
                    }
                    mutations.push(Mutation::Unlink { path: to.clone() });
                }
                _ => mutations.push(Mutation::Unlink { path: to.clone() }),
            }
        }
        mutations.push(Mutation::Rename { from, to });
        if let Err(code) = self.commit(&mutations) {
            reply.error(code);
            return;
        }

        if let Some(existing) = self.lookup_child(newparent, OsStr::new(&newname)) {
            if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&newparent) {
                children.remove(&newname);
            }
            let drop_existing = match self.nodes.get_mut(&existing) {
                Some(WNode::File(file)) => {
                    file.links.retain(|(link_parent, link_name)| {
                        !(*link_parent == newparent && link_name == &newname)
                    });
                    file.links.is_empty() && file.open_handles == 0
                }
                _ => true,
            };
            if drop_existing {
                self.nodes.remove(&existing);
                self.locks.forget_ino(existing);
            }
        }
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&parent) {
            children.remove(&name);
        }
        match self.nodes.get_mut(&ino) {
            Some(WNode::Directory {
                parent: node_parent,
                name: node_name,
                ..
            })
            | Some(WNode::Symlink {
                parent: node_parent,
                name: node_name,
                ..
            }) => {
                *node_parent = newparent;
                *node_name = newname.clone();
            }
            Some(WNode::File(file)) => {
                for link in &mut file.links {
                    if link.0 == parent && link.1 == name {
                        *link = (newparent, newname.clone());
                        break;
                    }
                }
            }
            None => {}
        }
        if let Some(WNode::Directory { children, .. }) = self.nodes.get_mut(&newparent) {
            children.insert(newname, ino);
        }
        reply.ok();
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        // Size and the executable bit are real workspace facts. Ownership
        // does not exist in the format, and timestamps are deliberately
        // canonical, so both are accepted and ignored.
        if let Some(new_size) = size {
            let Some(WNode::File(file)) = self.nodes.get_mut(&ino) else {
                reply.error(libc::EINVAL);
                return;
            };
            if file.dirty.is_none() {
                match Dirty::new() {
                    Ok(dirty) => file.dirty = Some(dirty),
                    Err(_) => {
                        reply.error(libc::EIO);
                        return;
                    }
                }
            }
            if new_size < file.size {
                file.dirty
                    .as_mut()
                    .expect("dirty state was just ensured")
                    .shrink_to(new_size);
            }
            file.size = new_size;
        }
        if let Some(mode) = mode {
            let executable = mode & 0o111 != 0;
            let (path, changed) = match self.nodes.get(&ino) {
                Some(WNode::File(file)) => (self.file_path(file), file.executable != executable),
                _ => (None, false),
            };
            if changed {
                if let Some(path) = path {
                    if let Err(code) = self.commit(&[Mutation::SetExecutable { path, executable }])
                    {
                        reply.error(code);
                        return;
                    }
                }
                if let Some(WNode::File(file)) = self.nodes.get_mut(&ino) {
                    file.executable = executable;
                }
            }
        }
        match self.attr(ino) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(libc::ENOENT),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        let truncate = flags & libc::O_TRUNC != 0;
        let Some(WNode::File(file)) = self.nodes.get_mut(&ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        if truncate {
            if file.dirty.is_none() {
                match Dirty::new() {
                    Ok(dirty) => file.dirty = Some(dirty),
                    Err(_) => {
                        reply.error(libc::EIO);
                        return;
                    }
                }
            }
            let dirty = file.dirty.as_mut().expect("dirty state was just ensured");
            dirty.ranges.clear();
            file.size = 0;
        }
        file.open_handles += 1;
        let writable = flags & libc::O_ACCMODE != libc::O_RDONLY;
        let fh = self.alloc_fh(writable);
        // Page-cache path: mmap needs it, and it cannot go stale — committed
        // content only changes through this mount's own operations, and the
        // kernel keeps its own cache coherent for those (see module docs).
        reply.opened(fh, 0);
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        let Some(WNode::File(file)) = self.nodes.get(&ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let logical_size = file.size;
        let offset = offset.max(0) as u64;
        let end = (offset + size as u64).min(logical_size);
        if end <= offset {
            reply.data(&[]);
            return;
        }
        let mut buffer = vec![0_u8; (end - offset) as usize];
        match self.read_merged(ino, fh, offset, &mut buffer) {
            Ok(()) => reply.data(&buffer),
            Err(code) => reply.error(code),
        }
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: fuser::ReplyWrite,
    ) {
        let Some(WNode::File(file)) = self.nodes.get_mut(&ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let offset = offset.max(0) as u64;
        if file.dirty.is_none() {
            match Dirty::new() {
                Ok(dirty) => file.dirty = Some(dirty),
                Err(_) => {
                    reply.error(libc::EIO);
                    return;
                }
            }
        }
        let dirty = file.dirty.as_mut().expect("dirty state was just ensured");
        if dirty.spill.write_all_at(data, offset).is_err() {
            reply.error(libc::EIO);
            return;
        }
        dirty.add_range(offset, data.len() as u64);
        file.size = file.size.max(offset + data.len() as u64);
        reply.written(data.len() as u32);
    }

    fn flush(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        // FLUSH fires on every fd close, including the CLOEXEC teardown of a
        // mere fork/exec in the writing process and every read-only close a
        // file indexer makes. None of those are durability points: the last
        // real close composes in `release`, and the guaranteed contract is
        // `fsync`, exactly as on any local filesystem. What FLUSH does carry
        // is the closing descriptor's lock owner: POSIX releases that owner's
        // record locks on any close of the file.
        self.locks.release_owner(ino, lock_owner);
        reply.ok();
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        match self.compose_commit(ino) {
            Ok(()) => reply.ok(),
            Err(code) => reply.error(code),
        }
    }

    fn fsyncdir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        // Namespace mutations commit durably as they happen.
        reply.ok();
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        _flags: i32,
        lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        if let Some(owner) = lock_owner {
            self.locks.release_owner(ino, owner);
        }
        let writable = self
            .handles
            .remove(&fh)
            .is_some_and(|handle| handle.writable);
        // The last close of a writable handle is a best-effort durability
        // point; close(2) has already returned, so failures surface on the
        // next explicit fsync rather than here.
        if writable {
            let _ = self.compose_commit(ino);
        }
        let mut drop_node = false;
        let mut release = None;
        if let Some(WNode::File(file)) = self.nodes.get_mut(&ino) {
            file.open_handles = file.open_handles.saturating_sub(1);
            if file.open_handles == 0 && file.links.is_empty() {
                release = file.unlink_lease.take();
                drop_node = true;
            }
        }
        if let Some(lease) = release {
            let _ = self.meta.release_lease(lease);
        }
        if drop_node {
            self.nodes.remove(&ino);
            self.locks.forget_ino(ino);
        }
        reply.ok();
    }

    fn getlk(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        lock_owner: u64,
        start: u64,
        end: u64,
        typ: i32,
        _pid: u32,
        reply: ReplyLock,
    ) {
        let exclusive = typ == libc::F_WRLCK;
        match self
            .locks
            .first_conflict(ino, lock_owner, exclusive, start, end)
        {
            Some(holder) => {
                let typ = if holder.exclusive {
                    libc::F_WRLCK
                } else {
                    libc::F_RDLCK
                };
                reply.locked(holder.start, holder.end, typ, holder.pid);
            }
            None => reply.locked(0, 0, libc::F_UNLCK, 0),
        }
    }

    fn setlk(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        lock_owner: u64,
        start: u64,
        end: u64,
        typ: i32,
        pid: u32,
        _sleep: bool,
        reply: ReplyEmpty,
    ) {
        if !matches!(self.nodes.get(&ino), Some(WNode::File(_))) {
            reply.error(libc::EBADF);
            return;
        }
        match typ {
            t if t == libc::F_UNLCK => {
                self.locks.unset(ino, lock_owner, start, end);
                reply.ok();
            }
            t if t == libc::F_RDLCK || t == libc::F_WRLCK => {
                let exclusive = t == libc::F_WRLCK;
                // A contended blocking request (`sleep`) is refused with
                // EAGAIN instead of suspending the dispatch loop; blocking
                // waits are deferred (see module docs).
                match self.locks.set(ino, lock_owner, pid, exclusive, start, end) {
                    Ok(()) => reply.ok(),
                    Err(_conflict) => reply.error(libc::EAGAIN),
                }
            }
            _ => reply.error(libc::EINVAL),
        }
    }
}

/// One live writable mount; dropping it unmounts.
pub struct WorkspaceMount {
    session: Option<BackgroundSession>,
    mountpoint: PathBuf,
}

impl WorkspaceMount {
    #[must_use]
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Unmounts explicitly; equivalent to dropping the guard.
    pub fn unmount(mut self) {
        self.session.take();
    }
}

impl Drop for WorkspaceMount {
    fn drop(&mut self) {
        self.session.take();
    }
}

/// Serves the workspace `name` from the store at `root`, writable, at
/// `mountpoint`.
pub fn mount_workspace(
    root: impl AsRef<Path>,
    name: &str,
    mountpoint: impl AsRef<Path>,
) -> Result<WorkspaceMount, MountError> {
    let root = root.as_ref();
    let mountpoint = mountpoint.as_ref().to_path_buf();
    if !mountpoint.is_dir() {
        return Err(MountError::NotADirectory(mountpoint));
    }
    let meta = WorkspaceStore::open(root)?;
    let tree = meta.head_tree(name)?;
    let (nodes, next_ino) = load_tree(&tree);
    debug_assert!(nodes.values().all(|node| match node {
        WNode::File(file) => file.size == records_size(&file.committed),
        _ => true,
    }));

    let root_metadata = std::fs::metadata(root)?;
    let fs = WorkspaceFs {
        meta,
        workspace: name.to_owned(),
        uid: root_metadata.uid(),
        gid: root_metadata.gid(),
        nodes,
        next_ino,
        next_fh: 0,
        handles: HashMap::new(),
        locks: LockTable::default(),
    };
    let session = fuser::spawn_mount2(
        fs,
        &mountpoint,
        &[
            MountOption::RW,
            MountOption::FSName("tensorfs".to_owned()),
            MountOption::DefaultPermissions,
        ],
    )?;
    Ok(WorkspaceMount {
        session: Some(session),
        mountpoint,
    })
}
