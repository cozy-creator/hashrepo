//! Two PROCESSES sharing one store root.
//!
//! The Turso swap (PR #28) landed with the engine's default whole-file
//! exclusive lock, which made the store single-process: a second process could
//! not even open it. Any long-lived holder — the shelved `tensorfsd serve`
//! was the original one — locked out the CLI `seal`, an out-of-band GC pass
//! and the direct-ingest writer under that default, every one of which is a
//! separate process.
//!
//! These arms are the executable proof that multiprocess coordination is on.
//! They deliberately assert the CAPABILITY rather than the limitation, so they
//! fail if anyone reintroduces a single-process open.

#![cfg(any(unix, windows))]
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::FileRecord;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const ROLE: &str = "TENSORFS_MP_ROLE";
const ROOT: &str = "TENSORFS_MP_ROOT";
const WORKSPACE: &str = "TENSORFS_MP_WORKSPACE";

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-mp-{name}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is sane")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("scratch root creates");
        Self(root)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One object of `fill` bytes committed as `<workspace>/model.bin`.
fn commit_one(meta: &WorkspaceStore, workspace: &str, fill: u8) {
    let bytes = vec![fill; 4096];
    let admitted = meta.store().put_bytes(&bytes).expect("object admits");
    meta.commit_generation(
        workspace,
        &[Mutation::CreateFile {
            path: "model.bin".to_owned(),
            executable: false,
            planner: PlannerId::BlobV1,
            records: vec![FileRecord::Data {
                digest: admitted.digest(),
                length: bytes.len() as u64,
            }],
        }],
    )
    .expect("generation commits");
}

fn spawn_child(root: &Path, role: &str, workspace: &str) -> std::process::Child {
    Command::new(std::env::current_exe().expect("test binary path"))
        .args(["multiprocess_child_role", "--exact", "--nocapture"])
        .env(ROLE, role)
        .env(ROOT, root)
        .env(WORKSPACE, workspace)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("child spawns")
}

fn wait_ok(child: std::process::Child, what: &str) {
    let output = child.wait_with_output().expect("child is reaped");
    assert!(
        output.status.success(),
        "{what} failed in a second process: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The child half. A no-op unless the parent set [`ROLE`], so a normal
/// `cargo test` run treats it as an empty test rather than a stray process.
#[test]
fn multiprocess_child_role() {
    let Ok(role) = std::env::var(ROLE) else {
        return;
    };
    let root = PathBuf::from(std::env::var(ROOT).expect("child needs a root"));
    let workspace = std::env::var(WORKSPACE).expect("child needs a workspace");

    let meta = WorkspaceStore::open(&root).expect("the second process opens the store");
    match role.as_str() {
        // Prove a second process can WRITE while the first holds the store.
        "write" => {
            meta.create_workspace(&workspace)
                .expect("workspace creates");
            commit_one(&meta, &workspace, 0xC2);
        }
        // Prove a second process can READ a workspace the first one committed.
        "read" => {
            let head = meta
                .head_tree(&workspace)
                .expect("the parent's workspace is visible to another process");
            assert!(
                !head.entries().is_empty(),
                "the parent's committed entry must be visible across processes"
            );
        }
        other => panic!("unknown child role {other}"),
    }
}

/// The regression arm. Under the engine's default lock this fails at
/// `WorkspaceStore::open` in the child with
/// `Failed locking file. File is locked by another process`.
#[test]
fn a_second_process_can_write_while_the_first_holds_the_store() {
    let scratch = Scratch::new("write");

    // The parent holds the store open for the whole test, exactly as a
    // long-lived server process would for its lifetime.
    let held = WorkspaceStore::open(scratch.path()).expect("the first opener succeeds");
    if !held.supports_multiprocess() {
        eprintln!(
            "skipping: this platform/filesystem cannot support shared WAL \
             coordination, so the store opened single-process"
        );
        return;
    }
    held.create_workspace("parent").expect("workspace creates");
    commit_one(&held, "parent", 0xA1);

    wait_ok(
        spawn_child(scratch.path(), "write", "child"),
        "a concurrent writer",
    );

    // The parent — still holding its original handle — sees the child's work,
    // and its own is undisturbed.
    let child_head = held
        .head_tree("child")
        .expect("the child's workspace is visible to the holder");
    assert!(
        !child_head.entries().is_empty(),
        "the child's commit must be visible to the process that held the store"
    );
    assert!(
        held.head_generation("child").expect("generation reads") >= 1,
        "the child's generation must be recorded"
    );

    let parent_head = held
        .head_tree("parent")
        .expect("the parent's own workspace survives");
    assert!(
        !parent_head.entries().is_empty(),
        "the parent's own commit survives"
    );
}

/// The read direction, which is what a CLI `seal` does against a process
/// already holding the store.
#[test]
fn a_second_process_can_read_what_the_first_committed() {
    let scratch = Scratch::new("read");

    let held = WorkspaceStore::open(scratch.path()).expect("the first opener succeeds");
    if !held.supports_multiprocess() {
        eprintln!(
            "skipping: this platform/filesystem cannot support shared WAL \
             coordination, so the store opened single-process"
        );
        return;
    }
    held.create_workspace("parent").expect("workspace creates");
    commit_one(&held, "parent", 0xB7);

    wait_ok(
        spawn_child(scratch.path(), "read", "parent"),
        "a concurrent reader",
    );
}
