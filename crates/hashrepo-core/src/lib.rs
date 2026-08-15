#![forbid(unsafe_code)]

pub mod object;
pub mod planner;
pub mod source;
/// The object store needs real files, advisory locks, and directory fsync,
/// so it exists only on the platforms the native daemon targets.
#[cfg(any(unix, windows))]
pub mod store;
pub mod tfm1;
