#![forbid(unsafe_code)]

pub mod object;
pub mod planner;
pub mod source;
/// The object store needs real files, advisory locks, and directory fsync,
/// so it exists only on the platforms the native daemon targets.
#[cfg(any(unix, windows))]
pub mod store;
/// Workspace metadata rides the Turso engine beside the object store, so it
/// shares the store's real-filesystem platform gate.
/// Snapshot sync moves objects between the local store and a remote grant
/// service, so it shares the real-filesystem platform gate.
#[cfg(any(unix, windows))]
pub mod sync;
pub mod tfm1;
pub mod tfp1;
#[cfg(any(unix, windows))]
pub mod workspace;
#[cfg(any(unix, windows))]
mod workspace_db;
#[cfg(any(unix, windows))]
mod workspace_source;
