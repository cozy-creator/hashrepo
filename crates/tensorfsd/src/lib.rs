#![forbid(unsafe_code)]

//! The TensorFS daemon.
//!
//! This slice serves one sealed TFM1 snapshot as a read-only FUSE3 mount and
//! one workspace as a writable COW mount: ordinary file semantics over
//! immutable CAS objects, with dirty bytes overlaid until a durable flush
//! composes and commits exactly the touched objects.
//!
//! # Platform support: Linux only
//!
//! Linux/FUSE3 is the only supported platform. `fuser`, `libc`, `nix` and
//! `signal-hook` are `cfg(target_os = "linux")` dependencies, every module
//! below is gated, `main` is a stub that exits non-zero, and every file in
//! `tests/` opens with `#![cfg(target_os = "linux")]`.
//!
//! macOS (macFUSE/FSKit) and Windows (WinFsp) are unimplemented, not merely
//! untested. On those targets this crate compiles to an empty shell, so a
//! green macOS or Windows CI run says nothing about mounting there. The
//! `core-cross-platform` job builds `tensorfs-core` alone for that reason,
//! and `scripts/check-daemon-linux-gate.sh` fails if a daemon test ever
//! drops its gate and starts to look covered off Linux.
//!
//! The Windows named-pipe control plane is a seam, not a feature: `rpc`
//! serves a Unix socket and nothing implements the pipe.

/// The FUSE adapters need a kernel, so they exist only on Linux.
#[cfg(target_os = "linux")]
mod locks;
#[cfg(target_os = "linux")]
mod snapshot_fs;
#[cfg(target_os = "linux")]
mod workspace_fs;

/// The Unix-socket control plane shares the Linux gate.
#[cfg(target_os = "linux")]
pub mod rpc;

/// Row schema and folds for the pgw#1256 measured matrix; the
/// `tensorfs-bench` bin is the sole producer and CI never runs it.
#[cfg(target_os = "linux")]
pub mod bench;

#[cfg(target_os = "linux")]
pub use snapshot_fs::{MountError, SnapshotMount, mount_snapshot};
#[cfg(target_os = "linux")]
pub use workspace_fs::{
    WorkspaceMount, assembly_budget_bytes, assembly_peak_bytes, mount_workspace,
};
