#![forbid(unsafe_code)]

//! The TensorFS daemon.
//!
//! This slice serves one sealed TFM1 snapshot as a read-only FUSE3 mount.
//! Applications read ordinary files whose bytes come from verified CAS
//! objects; hole records read as zeros without touching the store. Writable
//! COW workspaces, the local RPC control plane, and the macFUSE/WinFsp
//! adapters are later slices.

/// The FUSE adapter needs a kernel, so it exists only on Linux in this slice.
#[cfg(target_os = "linux")]
mod snapshot_fs;

#[cfg(target_os = "linux")]
pub use snapshot_fs::{MountError, SnapshotMount, mount_snapshot};
