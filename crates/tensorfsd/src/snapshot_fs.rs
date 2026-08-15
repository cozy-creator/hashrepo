use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use fuser::{
    BackgroundSession, FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyData,
    ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, Request,
};
use thiserror::Error;

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{Entry, FileRecord, Snapshot, SnapshotId};
use tensorfs_core::workspace::{WorkspaceError, WorkspaceStore};

/// Snapshot content is immutable, so the kernel may cache entries and
/// attributes for as long as it likes.
const TTL: Duration = Duration::from_secs(3600);
const ROOT_INO: u64 = 1;

#[derive(Debug, Error)]
pub enum MountError {
    #[error("workspace metadata failed")]
    Workspace(#[from] WorkspaceError),
    #[error("mountpoint is not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("mount I/O failed")]
    Io(#[from] io::Error),
}

/// One byte range of a regular file: object bytes or a zero hole.
enum Segment {
    Data {
        start: u64,
        length: u64,
        digest: ObjectDigest,
    },
    Hole {
        start: u64,
        length: u64,
    },
}

struct FileNode {
    executable: bool,
    logical_size: u64,
    nlink: u32,
    segments: Vec<Segment>,
}

enum Node {
    Directory { children: Vec<(String, u64)> },
    File { file: usize },
    Symlink { target: String },
}

struct Tree {
    /// Indexed by `ino - 1`; hardlinks alias one `File` index from two inodes
    /// deliberately never happens — a link group shares one inode instead.
    nodes: Vec<Node>,
    files: Vec<FileNode>,
}

fn parent_and_name(path: &str) -> (&str, &str) {
    match path.rsplit_once('/') {
        Some((parent, name)) => (parent, name),
        None => ("", path),
    }
}

/// Builds the immutable inode table from one decoded canonical snapshot.
/// TFM1 guarantees sorted paths with every parent present, so parents always
/// resolve before their children, and hardlink ordinals only point backward.
fn build_tree(snapshot: &Snapshot) -> Tree {
    let mut nodes = vec![Node::Directory {
        children: Vec::new(),
    }];
    let mut files = Vec::new();
    let mut inode_by_path: HashMap<&str, u64> = HashMap::new();

    for (path, entry) in snapshot.entries() {
        let (parent, name) = parent_and_name(path);
        let ino = match entry {
            Entry::Directory => {
                nodes.push(Node::Directory {
                    children: Vec::new(),
                });
                nodes.len() as u64
            }
            Entry::File {
                executable,
                logical_size,
                records,
                ..
            } => {
                let mut segments = Vec::with_capacity(records.len());
                let mut start = 0_u64;
                for record in records {
                    segments.push(match record {
                        FileRecord::Data { digest, length } => Segment::Data {
                            start,
                            length: *length,
                            digest: *digest,
                        },
                        FileRecord::Hole { length } => Segment::Hole {
                            start,
                            length: *length,
                        },
                    });
                    start += match record {
                        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
                    };
                }
                files.push(FileNode {
                    executable: *executable,
                    logical_size: *logical_size,
                    nlink: 1,
                    segments,
                });
                nodes.push(Node::File {
                    file: files.len() - 1,
                });
                nodes.len() as u64
            }
            Entry::Symlink { target } => {
                nodes.push(Node::Symlink {
                    target: target.clone(),
                });
                nodes.len() as u64
            }
            Entry::Hardlink { ordinal } => {
                let carrier = snapshot
                    .ordinal_path(*ordinal)
                    .expect("a decoded snapshot only names assigned ordinals");
                let ino = inode_by_path[carrier];
                if let Node::File { file } = &nodes[(ino - 1) as usize] {
                    files[*file].nlink += 1;
                }
                ino
            }
        };
        inode_by_path.insert(path, ino);
        let parent_ino = if parent.is_empty() {
            ROOT_INO
        } else {
            inode_by_path[parent]
        };
        if let Node::Directory { children } = &mut nodes[(parent_ino - 1) as usize] {
            children.push((name.to_owned(), ino));
        }
    }

    Tree { nodes, files }
}

struct SnapshotFs {
    store: ObjectStore,
    tree: Tree,
    uid: u32,
    gid: u32,
    next_fh: u64,
    /// Per open file handle, one lazily opened `File` per referenced object;
    /// bytes are trusted from admission-time verification and never rehashed
    /// on the read path.
    handles: HashMap<u64, HashMap<[u8; 32], File>>,
}

impl SnapshotFs {
    fn node(&self, ino: u64) -> Option<&Node> {
        self.tree.nodes.get((ino.checked_sub(1)?) as usize)
    }

    fn attr(&self, ino: u64) -> Option<FileAttr> {
        let node = self.node(ino)?;
        let (kind, size, perm, nlink) = match node {
            Node::Directory { children } => {
                let subdirs = children
                    .iter()
                    .filter(|(_, child)| matches!(self.node(*child), Some(Node::Directory { .. })))
                    .count() as u32;
                (FileType::Directory, 0, 0o555, 2 + subdirs)
            }
            Node::File { file } => {
                let file = &self.tree.files[*file];
                let perm = if file.executable { 0o555 } else { 0o444 };
                (FileType::RegularFile, file.logical_size, perm, file.nlink)
            }
            Node::Symlink { target } => (FileType::Symlink, target.len() as u64, 0o777, 1),
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

    /// Fills `buffer` from one file's segments starting at `offset`. Data
    /// segments read from their verified object; holes read as zeros.
    fn read_file(
        &mut self,
        file: usize,
        fh: u64,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<(), i32> {
        let node = &self.tree.files[file];
        for segment in &node.segments {
            let (start, length) = match segment {
                Segment::Data { start, length, .. } | Segment::Hole { start, length } => {
                    (*start, *length)
                }
            };
            let end = start + length;
            if end <= offset || start >= offset + buffer.len() as u64 {
                continue;
            }
            let from = offset.max(start);
            let to = end.min(offset + buffer.len() as u64);
            let slice = &mut buffer[(from - offset) as usize..(to - offset) as usize];
            match segment {
                Segment::Hole { .. } => slice.fill(0),
                Segment::Data { digest, .. } => {
                    let objects = self.handles.entry(fh).or_default();
                    if !objects.contains_key(digest.as_bytes()) {
                        let opened = self.store.open_object(digest).map_err(|_| libc::EIO)?;
                        objects.insert(*digest.as_bytes(), opened);
                    }
                    let object = &objects[digest.as_bytes()];
                    object
                        .read_exact_at(slice, from - start)
                        .map_err(|_| libc::EIO)?;
                }
            }
        }
        Ok(())
    }
}

impl Filesystem for SnapshotFs {
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(Node::Directory { children }) = self.node(parent) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(libc::ENOENT);
            return;
        };
        match children.iter().find(|(child, _)| child == name) {
            Some((_, ino)) => match self.attr(*ino) {
                Some(attr) => reply.entry(&TTL, &attr, 0),
                None => reply.error(libc::ENOENT),
            },
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
        match self.node(ino) {
            Some(Node::Symlink { target }) => reply.data(target.as_bytes()),
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
        let Some(Node::Directory { children }) = self.node(ino) else {
            reply.error(libc::ENOTDIR);
            return;
        };
        let mut entries: Vec<(u64, FileType, &str)> = vec![
            (ino, FileType::Directory, "."),
            (ino, FileType::Directory, ".."),
        ];
        for (name, child) in children {
            let kind = match self.node(*child) {
                Some(Node::Directory { .. }) => FileType::Directory,
                Some(Node::Symlink { .. }) => FileType::Symlink,
                _ => FileType::RegularFile,
            };
            entries.push((*child, kind, name));
        }
        for (index, (child, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            if reply.add(child, (index + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        if !matches!(self.node(ino), Some(Node::File { .. })) {
            reply.error(libc::ENOENT);
            return;
        }
        if flags & libc::O_ACCMODE != libc::O_RDONLY || flags & libc::O_TRUNC != 0 {
            reply.error(libc::EROFS);
            return;
        }
        self.next_fh += 1;
        self.handles.insert(self.next_fh, HashMap::new());
        reply.opened(self.next_fh, 0);
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
        let Some(Node::File { file }) = self.node(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let file = *file;
        let logical_size = self.tree.files[file].logical_size;
        let offset = offset.max(0) as u64;
        let end = (offset + size as u64).min(logical_size);
        if end <= offset {
            reply.data(&[]);
            return;
        }
        let mut buffer = vec![0_u8; (end - offset) as usize];
        match self.read_file(file, fh, offset, &mut buffer) {
            Ok(()) => reply.data(&buffer),
            Err(errno) => reply.error(errno),
        }
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        self.handles.remove(&fh);
        reply.ok();
    }

    // Every mutating operation refuses with EROFS in this slice: the mount
    // serves one sealed snapshot, and writable COW workspaces are the next
    // slice, not an error-mapping accident.
    fn setattr(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        _size: Option<u64>,
        _atime: Option<fuser::TimeOrNow>,
        _mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        reply.error(libc::EROFS);
    }

    fn mknod(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _mode: u32,
        _umask: u32,
        _rdev: u32,
        reply: ReplyEntry,
    ) {
        reply.error(libc::EROFS);
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        reply.error(libc::EROFS);
    }

    fn unlink(&mut self, _req: &Request<'_>, _parent: u64, _name: &OsStr, reply: ReplyEmpty) {
        reply.error(libc::EROFS);
    }

    fn rmdir(&mut self, _req: &Request<'_>, _parent: u64, _name: &OsStr, reply: ReplyEmpty) {
        reply.error(libc::EROFS);
    }

    fn symlink(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _link_name: &OsStr,
        _target: &Path,
        reply: ReplyEntry,
    ) {
        reply.error(libc::EROFS);
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _newparent: u64,
        _newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        reply.error(libc::EROFS);
    }

    fn link(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _newparent: u64,
        _newname: &OsStr,
        reply: ReplyEntry,
    ) {
        reply.error(libc::EROFS);
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _offset: i64,
        _data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: fuser::ReplyWrite,
    ) {
        reply.error(libc::EROFS);
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        _parent: u64,
        _name: &OsStr,
        _mode: u32,
        _umask: u32,
        _flags: i32,
        reply: fuser::ReplyCreate,
    ) {
        reply.error(libc::EROFS);
    }
}

/// One live mount. Dropping it unmounts; a leaked guard is the only way to
/// leak a mount, so tests hold it in scope and assert the table after drop.
pub struct SnapshotMount {
    session: Option<BackgroundSession>,
    mountpoint: PathBuf,
}

impl SnapshotMount {
    #[must_use]
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Unmounts explicitly; equivalent to dropping the guard.
    pub fn unmount(mut self) {
        self.session.take();
    }
}

impl Drop for SnapshotMount {
    fn drop(&mut self) {
        self.session.take();
    }
}

/// Serves the sealed snapshot `id` from the store at `root`, read-only, at
/// `mountpoint`.
pub fn mount_snapshot(
    root: impl AsRef<Path>,
    id: &SnapshotId,
    mountpoint: impl AsRef<Path>,
) -> Result<SnapshotMount, MountError> {
    let root = root.as_ref();
    let mountpoint = mountpoint.as_ref().to_path_buf();
    if !mountpoint.is_dir() {
        return Err(MountError::NotADirectory(mountpoint));
    }
    let workspace = WorkspaceStore::open(root)?;
    let snapshot = workspace.load_snapshot(id)?;
    drop(workspace);

    let store = ObjectStore::open(root).map_err(WorkspaceError::from)?;
    let root_metadata = std::fs::metadata(root)?;
    let fs = SnapshotFs {
        store,
        tree: build_tree(&snapshot),
        uid: root_metadata.uid(),
        gid: root_metadata.gid(),
        next_fh: 0,
        handles: HashMap::new(),
    };
    let session = fuser::spawn_mount2(
        fs,
        &mountpoint,
        &[
            MountOption::RO,
            MountOption::FSName("tensorfs".to_owned()),
            MountOption::DefaultPermissions,
        ],
    )?;
    Ok(SnapshotMount {
        session: Some(session),
        mountpoint,
    })
}
