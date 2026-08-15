#![forbid(unsafe_code)]

//! The TensorFS daemon.
//!
//! This slice serves one sealed TFM1 snapshot as a read-only FUSE3 mount and
//! one workspace as a writable COW mount: ordinary file semantics over
//! immutable CAS objects, with dirty bytes overlaid until a durable flush
//! composes and commits exactly the touched objects. The local RPC control
//! plane and the macFUSE/WinFsp adapters are later slices.

/// The FUSE adapters need a kernel, so they exist only on Linux in this slice.
#[cfg(target_os = "linux")]
mod snapshot_fs;
#[cfg(target_os = "linux")]
mod workspace_fs;

#[cfg(target_os = "linux")]
pub use snapshot_fs::{MountError, SnapshotMount, mount_snapshot};
#[cfg(target_os = "linux")]
pub use workspace_fs::{WorkspaceMount, mount_workspace};
