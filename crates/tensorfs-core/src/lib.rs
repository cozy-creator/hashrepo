#![forbid(unsafe_code)]

/// Composing a new tensor container out of a committed one's own objects —
/// a re-key or a record subset. Needs the store, so it shares its gate.
#[cfg(all(feature = "store", any(unix, windows)))]
pub mod compose;
/// Tensor layout contracts: the declarative, versioned input that identifies
/// a file's layout and directs its chunking seams. Pure data and pure
/// functions — no filesystem, no store.
pub mod contract;
/// Snapshot symlink trees, pointer stubs and refs beside the object store.
/// Shares the store's real-filesystem platform gate and needs nothing else —
/// a projection is a pure function of a manifest and the store root.
#[cfg(all(feature = "store", any(unix, windows)))]
pub mod layout;
pub mod object;
pub mod planner;
pub mod source;
/// The object store needs real files, advisory locks, and directory fsync,
/// so it exists only where those exist: Unix and Windows.
///
/// The `store` feature also lets a consumer that needs only bytes — the
/// Python extension module — link the store without the HTTP sync client or
/// the embedded metadata engine.
#[cfg(all(feature = "store", any(unix, windows)))]
pub mod store;
/// Snapshot sync moves objects between the local store and a remote grant
/// service, so it shares the real-filesystem platform gate.
#[cfg(all(feature = "sync", any(unix, windows)))]
pub mod sync;
pub mod tfm1;
pub mod tfp1;
/// Workspace metadata rides the Turso engine beside the object store, so it
/// shares the store's real-filesystem platform gate.
#[cfg(all(feature = "workspace", any(unix, windows)))]
pub mod workspace;
#[cfg(all(feature = "workspace", any(unix, windows)))]
mod workspace_db;
/// A committed file's ordered records presented as one byte source. Public
/// because it is the foundation of a direct tensor read: one tensor's bytes
/// addressed inside a committed file that was never materialized.
#[cfg(all(feature = "store", any(unix, windows)))]
pub mod workspace_source;
