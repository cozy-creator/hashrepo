//! Synchronous façade over the Turso engine, confined to this module.
//!
//! Paul's 2026-08-15 ruling moves the metadata engine from SQLite to Turso.
//! Turso's Rust API is async, but its futures are self-stepping (the engine
//! pumps its own IO inside `poll` and uses the waker only for completion), so
//! a parked-thread executor drives them without any runtime dependency. The
//! blocking is confined here: `workspace.rs` keeps its synchronous shape and
//! its exact call vocabulary.
//!
//! Transactions are issued as raw SQL (`BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK`)
//! so [`Transaction`] can borrow [`Connection`] and deref to it exactly like
//! rusqlite's, and an un-committed transaction rolls back on drop. Every
//! statement stays SQLite-compatible on purpose: the escape hatch back to
//! rusqlite is the same file on disk, and the cross-engine reopen tests hold
//! that door open.

use std::fmt;
use std::future::Future;
use std::ops::Deref;
use std::path::Path;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Engine(#[from] turso::Error),
    #[error("query returned no rows")]
    QueryReturnedNoRows,
}

/// Mirrors rusqlite's extension so `query_row(...).optional()` keeps reading
/// the same at every call site.
pub trait OptionalExtension<T> {
    fn optional(self) -> Result<Option<T>, Error>;
}

impl<T> OptionalExtension<T> for Result<T, Error> {
    fn optional(self) -> Result<Option<T>, Error> {
        match self {
            Ok(value) => Ok(Some(value)),
            Err(Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionBehavior {
    Immediate,
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives one self-stepping engine future to completion on this thread.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

/// One bound statement parameter. The closed set matches what the metadata
/// schema actually stores.
#[derive(Clone, Debug)]
pub enum Param {
    Null,
    Integer(i64),
    Text(String),
    Blob(Vec<u8>),
}

impl From<i64> for Param {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<&str> for Param {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<&String> for Param {
    fn from(value: &String) -> Self {
        Self::Text(value.clone())
    }
}

impl From<String> for Param {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&[u8]> for Param {
    fn from(value: &[u8]) -> Self {
        Self::Blob(value.to_vec())
    }
}

impl From<Vec<u8>> for Param {
    fn from(value: Vec<u8>) -> Self {
        Self::Blob(value)
    }
}

impl From<Param> for turso::Value {
    fn from(value: Param) -> Self {
        match value {
            Param::Null => Self::Null,
            Param::Integer(value) => Self::Integer(value),
            Param::Text(value) => Self::Text(value),
            Param::Blob(value) => Self::Blob(value),
        }
    }
}

/// Builds a parameter list; reads identically to rusqlite's macro of the same
/// name at every call site.
macro_rules! params {
    () => { std::vec::Vec::<$crate::workspace_db::Param>::new() };
    ($($value:expr),+ $(,)?) => {
        std::vec![$($crate::workspace_db::Param::from($value)),+]
    };
}
pub(crate) use params;

/// Anything the shim accepts as a statement parameter list.
pub trait IntoParamVec {
    fn into_param_vec(self) -> Vec<Param>;
}

impl IntoParamVec for Vec<Param> {
    fn into_param_vec(self) -> Vec<Param> {
        self
    }
}

impl<const N: usize> IntoParamVec for [Param; N] {
    fn into_param_vec(self) -> Vec<Param> {
        self.into_iter().collect()
    }
}

fn bind(params: impl IntoParamVec) -> impl turso::IntoParams {
    turso::params_from_iter(params.into_param_vec().into_iter().map(turso::Value::from))
}

/// Mirrors rusqlite's row-index bound so `row.get::<_, T>(0)` keeps its exact
/// spelling; only `usize` indexes exist here.
pub trait RowIndex {
    fn index(self) -> usize;
}

impl RowIndex for usize {
    fn index(self) -> usize {
        self
    }
}

pub struct Row {
    inner: turso::Row,
}

impl Row {
    pub fn get<I: RowIndex, T: turso::core::types::FromValue>(&self, index: I) -> Result<T, Error> {
        Ok(self.inner.get::<T>(index.index())?)
    }
}

pub struct Rows {
    inner: turso::Rows,
}

impl Rows {
    pub fn next(&mut self) -> Result<Option<Row>, Error> {
        Ok(block_on(self.inner.next())?.map(|inner| Row { inner }))
    }
}

pub struct Statement {
    inner: turso::Statement,
}

impl Statement {
    pub fn query(&mut self, params: impl IntoParamVec) -> Result<Rows, Error> {
        Ok(Rows {
            inner: block_on(self.inner.query(bind(params)))?,
        })
    }
}

/// One open metadata database. The engine connection is kept beside its
/// database handle so the pair lives and dies together.
pub struct Connection {
    _database: turso::Database,
    inner: turso::Connection,
    multiprocess: bool,
}

/// True for turso's refusal to run multiprocess WAL in this environment — an
/// in-memory path, an IO backend without shared WAL coordination (Windows'
/// default), or a network filesystem. Deliberately does NOT match a locking
/// error: those must surface, never trigger a downgrade.
fn is_multiprocess_unsupported(error: &turso::Error) -> bool {
    error
        .to_string()
        .contains("multiprocess WAL is not supported")
}

impl fmt::Debug for Connection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("workspace_db::Connection")
    }
}

impl Connection {
    /// Opens the metadata database in MULTIPROCESS mode.
    ///
    /// This is not optional and it is not a tuning knob. Turso's default is a
    /// whole-file exclusive lock taken at open, so a second process fails with
    /// `Failed locking file. File is locked by another process` — it cannot
    /// open the store at all, let alone contend for a write. Any long-lived
    /// holder — the shelved `tensorfsd serve` was the original example — would
    /// under that default lock out a CLI `seal`, a direct-ingest writer and an
    /// out-of-band GC pass.
    ///
    /// `experimental_multiprocess_wal` swaps that whole-file lock for shared
    /// WAL coordination (turso adds `OpenFlags::NoLock` and coordinates through
    /// a side file instead) — SQLite's WAL/`-shm` design. The engine decides
    /// this AT OPEN, which is why it is a builder call: setting
    /// `journal_mode=WAL` afterwards is too late to change the lock already
    /// taken.
    ///
    /// Every opener must agree — turso rejects a legacy open against a live
    /// multiprocess database and vice versa. That is safe here because this is
    /// the ONLY place the engine is opened; `workspace.rs` funnels every
    /// `WorkspaceStore::open` through it.
    ///
    /// Some environments cannot provide it, and turso says so at open rather
    /// than degrading: Windows' default IO backend does not implement shared
    /// WAL coordination, and network filesystems are refused outright (NFS,
    /// CIFS/SMB, Ceph, Lustre, GFS2, OCFS2, AFS, Coda, NCP, 9p). There we fall
    /// back to the single-process open, because refusing to open the store at
    /// all is strictly worse than opening it single-process.
    ///
    /// That fallback is deliberately NARROW: it triggers only on turso's
    /// "multiprocess WAL is not supported" refusal, which is a deterministic
    /// property of the platform and filesystem, and NEVER on a locking error.
    /// On Linux/ext4 — production — it is unreachable, so the single-process
    /// regression cannot silently come back the way it arrived. Callers that
    /// need to know can ask [`Connection::is_multiprocess`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_string_lossy().into_owned();
        let (database, multiprocess) = match block_on(
            turso::Builder::new_local(&path)
                .experimental_multiprocess_wal(true)
                .build(),
        ) {
            Ok(database) => (database, true),
            Err(error) if is_multiprocess_unsupported(&error) => {
                (block_on(turso::Builder::new_local(&path).build())?, false)
            }
            Err(error) => return Err(error.into()),
        };
        let inner = database.connect()?;
        Ok(Self {
            _database: database,
            inner,
            multiprocess,
        })
    }

    /// Whether this store root supports concurrent access from several
    /// processes. False only where the platform or filesystem cannot support
    /// it; tests that assert cross-process behaviour should skip on false
    /// rather than fail.
    #[must_use]
    pub const fn is_multiprocess(&self) -> bool {
        self.multiprocess
    }

    pub fn pragma_update(&self, _schema: Option<()>, name: &str, value: &str) -> Result<(), Error> {
        block_on(self.inner.pragma_update(name, value))?;
        Ok(())
    }

    pub fn busy_timeout(&self, duration: std::time::Duration) -> Result<(), Error> {
        self.inner.busy_timeout(duration)?;
        Ok(())
    }

    pub fn execute_batch(&self, sql: &str) -> Result<(), Error> {
        block_on(self.inner.execute_batch(sql))?;
        Ok(())
    }

    pub fn execute(&self, sql: &str, params: impl IntoParamVec) -> Result<usize, Error> {
        let changed = block_on(self.inner.execute(sql, bind(params)))?;
        Ok(usize::try_from(changed).unwrap_or(usize::MAX))
    }

    pub fn query_row<T, F>(
        &self,
        sql: &str,
        params: impl IntoParamVec,
        mapper: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&Row) -> Result<T, Error>,
    {
        let mut rows = Rows {
            inner: block_on(self.inner.query(sql, bind(params)))?,
        };
        match rows.next()? {
            Some(row) => mapper(&row),
            None => Err(Error::QueryReturnedNoRows),
        }
    }

    pub fn prepare(&self, sql: &str) -> Result<Statement, Error> {
        Ok(Statement {
            inner: block_on(self.inner.prepare(sql))?,
        })
    }

    pub fn last_insert_rowid(&self) -> i64 {
        self.inner.last_insert_rowid()
    }

    pub fn transaction_with_behavior(
        &mut self,
        behavior: TransactionBehavior,
    ) -> Result<Transaction<'_>, Error> {
        let TransactionBehavior::Immediate = behavior;
        self.execute("BEGIN IMMEDIATE", [])?;
        Ok(Transaction {
            connection: self,
            committed: false,
        })
    }
}

/// A raw-SQL transaction that rolls back on drop unless committed, borrowing
/// the connection so helpers written against `&Connection` accept it as-is.
pub struct Transaction<'connection> {
    connection: &'connection Connection,
    committed: bool,
}

impl Transaction<'_> {
    pub fn commit(mut self) -> Result<(), Error> {
        self.connection.execute("COMMIT", [])?;
        self.committed = true;
        Ok(())
    }
}

impl Deref for Transaction<'_> {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.connection
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.connection.execute_batch("ROLLBACK");
        }
    }
}
