//! Cross-engine reopen: the escape hatch, proven rather than cited.
//!
//! Turso reads and writes SQLite's file format, so the same metadata file
//! must open under both engines with full integrity. These arms hold that
//! door open in both directions; if either ever fails, the engines' formats
//! have diverged and the swap decision must be revisited before shipping.

#![cfg(any(unix, windows))]

use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::{ByteSource, PlannerId};
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

use std::io;

struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).expect("test offsets fit");
        destination.copy_from_slice(&self.0[start..start + destination.len()]);
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

/// Builds a committed, sealed workspace via the Turso-backed store and
/// returns its snapshot id.
fn build_and_seal(root: &std::path::Path) -> SnapshotId {
    let meta = WorkspaceStore::open(root).expect("store opens");
    let payload = b"cross-engine reopen payload".repeat(64);
    let admitted = meta.store().put_bytes(&payload).expect("bytes admit");
    let hashed = plan_and_hash(&SliceSource(&payload)).expect("payload plans");
    assert_eq!(hashed.planner(), PlannerId::BlobV1);

    meta.create_workspace("main").expect("workspace creates");
    meta.commit_generation(
        "main",
        &[
            Mutation::Mkdir {
                path: "models".to_owned(),
            },
            Mutation::CreateFile {
                path: "models/weights.bin".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest: admitted.digest(),
                    length: payload.len() as u64,
                }],
            },
        ],
    )
    .expect("generation commits");
    meta.seal_snapshot("main", None).expect("snapshot seals")
}

#[test]
fn a_turso_written_database_reopens_under_rusqlite_with_full_integrity() {
    let root = tempdir("turso-to-rusqlite");
    let sealed = build_and_seal(&root);

    // The Turso store is dropped; rusqlite opens the identical file.
    let connection = rusqlite::Connection::open(root.join("metadata.sqlite3"))
        .expect("rusqlite opens the Turso-written file");
    let (name, generation): (String, i64) = connection
        .query_row("SELECT name, head_generation FROM workspaces", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .expect("workspace row reads");
    assert_eq!(name, "main");
    assert!(generation >= 1);

    let (id, blob): (Vec<u8>, Vec<u8>) = connection
        .query_row("SELECT id, blob FROM snapshots", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .expect("snapshot row reads");
    let id: [u8; 32] = id.try_into().expect("snapshot ids are 32 bytes");
    assert_eq!(SnapshotId::from_bytes(id), sealed);
    // The blob still hashes to its id under the other engine's read.
    assert_eq!(SnapshotId::of(&blob), sealed);

    let records: i64 = connection
        .query_row("SELECT COUNT(*) FROM object_maps", [], |row| row.get(0))
        .expect("object map reads");
    assert_eq!(records, 1);
}

#[test]
fn a_rusqlite_written_row_is_read_and_extended_by_the_turso_store() {
    let root = tempdir("rusqlite-to-turso");
    let sealed = build_and_seal(&root);

    // rusqlite writes into the live schema between Turso sessions.
    {
        let connection = rusqlite::Connection::open(root.join("metadata.sqlite3"))
            .expect("rusqlite opens the Turso-written file");
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .expect("rusqlite speaks the same journal");
        connection
            .execute(
                "INSERT INTO workspaces (name, head_generation, root_inode) VALUES (?1, 0, 0)",
                rusqlite::params!["sidecar"],
            )
            .expect("rusqlite inserts");
        let workspace = connection.last_insert_rowid();
        connection
            .execute(
                "INSERT INTO inodes (workspace_id, kind) VALUES (?1, 1)",
                rusqlite::params![workspace],
            )
            .expect("rusqlite inserts the root inode");
        let root_inode = connection.last_insert_rowid();
        connection
            .execute(
                "UPDATE workspaces SET root_inode = ?1 WHERE id = ?2",
                rusqlite::params![root_inode, workspace],
            )
            .expect("rusqlite links the root");
    }

    // Turso reopens the file rusqlite just wrote and works against BOTH
    // workspaces: reading the old one and committing into the new one.
    let meta = WorkspaceStore::open(&root).expect("Turso reopens the file");
    assert_eq!(
        meta.load_snapshot(&sealed)
            .expect("snapshot loads")
            .snapshot_id(),
        sealed
    );
    meta.commit_generation(
        "sidecar",
        &[Mutation::Mkdir {
            path: "from-turso".to_owned(),
        }],
    )
    .expect("Turso commits into the rusqlite-created workspace");
    assert_eq!(
        meta.head_generation("sidecar").expect("generation reads"),
        1
    );
}

fn tempdir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tensorfs-engine-reopen-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp root creates");
    path
}
