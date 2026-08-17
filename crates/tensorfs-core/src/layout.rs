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

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::object::ObjectDigest;
use crate::store::{ObjectStore, StoreError, set_read_only};
use crate::tfm1::{Entry, FileBody, Snapshot, SnapshotId};

static SWAP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

/// One pointer stub's bytes: the magic, one space, one line of JSON.
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
    /// The tree is built under a private name and renamed into place, so a
    /// ref can never resolve into a half-built projection. Projection is
    /// idempotent: an existing tree wins and the loser's scratch is removed.
    pub fn project(&self, snapshot: &Snapshot) -> Result<PathBuf, LayoutError> {
        let final_path = self.tree_path(&snapshot.snapshot_id());
        if final_path.exists() {
            return Ok(final_path);
        }
        let snapshots = self.snapshots_dir();
        fs::create_dir_all(&snapshots)?;
        let scratch = snapshots.join(format!(
            ".building-{}-{}",
            process::id(),
            SWAP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&scratch);
        fs::create_dir(&scratch)?;

        if let Err(error) = self.fill(snapshot, &scratch) {
            let _ = fs::remove_dir_all(&scratch);
            return Err(error);
        }
        match fs::rename(&scratch, &final_path) {
            Ok(()) => Ok(final_path),
            // Another projector of the same snapshot won the race. Its tree
            // has the same content by construction — the manifest is the id.
            Err(_) if final_path.exists() => {
                let _ = fs::remove_dir_all(&scratch);
                Ok(final_path)
            }
            Err(error) => {
                let _ = fs::remove_dir_all(&scratch);
                Err(error.into())
            }
        }
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
    pub fn remove_tree(&self, id: &SnapshotId) -> Result<bool, LayoutError> {
        let path = self.tree_path(id);
        match fs::remove_dir_all(&path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
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
    /// id or the new one on every platform, and nothing else.
    pub fn set_ref(&self, name: &str, id: &SnapshotId) -> Result<(), LayoutError> {
        let refs = self.refs_dir();
        fs::create_dir_all(&refs)?;
        let staged = refs.join(format!(
            ".swap-{}-{}",
            process::id(),
            SWAP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_file(&staged);
        fs::write(&staged, ref_bytes(id))?;
        match fs::rename(&staged, refs.join(validated_ref_name(name)?)) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_file(&staged);
                Err(error.into())
            }
        }
    }

    /// Reads `refs/<name>`: the id it names, or `None` when there is no such
    /// ref. One `open` plus one `read`, so a concurrent swap is invisible.
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

    pub fn remove_ref(&self, name: &str) -> Result<bool, LayoutError> {
        match fs::remove_file(self.refs_dir().join(validated_ref_name(name)?)) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Every ref name currently pointing at one snapshot, sorted.
    pub fn refs_to(&self, id: &SnapshotId) -> Result<Vec<String>, LayoutError> {
        let mut names = Vec::new();
        let entries = match fs::read_dir(self.refs_dir()) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(names),
            Err(error) => return Err(error.into()),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if validated_ref_name(name).is_err() {
                continue;
            }
            if self.read_ref(name).ok().flatten() == Some(*id) {
                names.push(name.to_owned());
            }
        }
        names.sort();
        Ok(names)
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
