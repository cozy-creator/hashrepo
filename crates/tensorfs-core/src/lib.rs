#![forbid(unsafe_code)]

pub mod object;
pub mod planner;
pub mod source;
/// The object store needs real files, advisory locks, and directory fsync,
/// so it exists only on the platforms the native daemon targets.
#[cfg(any(unix, windows))]
pub mod store;
pub mod tfm1;
pub mod tfp1;
/// Workspace metadata rides SQLite beside the object store, so it shares the
/// store's real-filesystem platform gate.
#[cfg(any(unix, windows))]
pub mod workspace;
#[cfg(any(unix, windows))]
mod workspace_source;
