//! Real-process crash coverage for the metadata layer. A SIGKILL at any
//! durability cut leaves the reopened store on the previous generation or the
//! new one — never a hybrid — and never costs a live object. The child role
//! re-invokes this test binary with env flags, one child at a time.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const ROLE_ENV: &str = "TENSORFS_WORKSPACE_CRASH_ROLE";
const ROOT_ENV: &str = "TENSORFS_WORKSPACE_CRASH_ROOT";

fn mark_ready(root: &str) {
    fs::write(Path::new(root).join("child-ready"), b"ready").expect("child marks readiness");
}

fn payload_record(store: &WorkspaceStore, bytes: &[u8]) -> FileRecord {
    let object = store.store().put_bytes(bytes).expect("child admits bytes");
    FileRecord::Data {
        digest: object.digest(),
        length: object.length(),
    }
}

/// Child dispatch: a no-op pass unless the parent set the role env.
#[test]
fn crash_child_role() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let root = std::env::var(ROOT_ENV).expect("the parent supplies the root");
    let store = WorkspaceStore::open(&root).expect("child opens the shared store");

    match role.as_str() {
        // Objects admitted, no commit: the head must never move.
        "admit-only" => {
            let record = payload_record(&store, b"admitted but never committed");
            let _ = record;
            mark_ready(&root);
            std::thread::sleep(Duration::from_secs(3600));
        }
        // An endless stream of commit generations for the parent to cut.
        "commit-loop" => {
            mark_ready(&root);
            for index in 0_u64.. {
                let record =
                    payload_record(&store, format!("generation payload {index}").as_bytes());
                store
                    .commit_generation(
                        "main",
                        &[Mutation::CreateFile {
                            path: format!("file-{index:06}.bin"),
                            executable: false,
                            planner: PlannerId::BlobV1,
                            records: vec![record],
                        }],
                    )
                    .expect("child commits");
            }
        }
        // A committed head the parent kills before any seal.
        "commit-then-hold" => {
            let record = payload_record(&store, b"committed before the kill");
            store
                .commit_generation(
                    "main",
                    &[Mutation::CreateFile {
                        path: "durable.bin".into(),
                        executable: false,
                        planner: PlannerId::BlobV1,
                        records: vec![record],
                    }],
                )
                .expect("child commits");
            mark_ready(&root);
            std::thread::sleep(Duration::from_secs(3600));
        }
        // Endless GC passes over a store with fresh orphans each round.
        "collect-loop" => {
            mark_ready(&root);
            for index in 0_u64.. {
                store
                    .store()
                    .put_bytes(format!("orphan {index}").as_bytes())
                    .expect("child seeds an orphan");
                store.collect().expect("child collects");
            }
        }
        other => panic!("unknown crash role {other:?}"),
    }
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-workspace-crash-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn spawn_child(root: &Path, role: &str) -> Child {
    let mut child = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["crash_child_role", "--exact", "--nocapture"])
        .env(ROLE_ENV, role)
        .env(ROOT_ENV, root)
        .spawn()
        .expect("child spawns");
    let ready = root.join("child-ready");
    // Liveness, not a deadline: a child that is ALIVE is working, however
    // slow the box is — wait indefinitely. A child that EXITED without
    // touching the ready marker failed; say so with its status. A wall-clock
    // bound here is a performance assertion, and this test makes none.
    while !ready.exists() {
        if let Some(status) = child.try_wait().expect("child status is readable") {
            panic!("child exited ({status}) before becoming ready");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child
}

fn kill_after(mut child: Child, delay: Duration) {
    std::thread::sleep(delay);
    child.kill().expect("SIGKILL the child");
    child.wait().expect("reap the killed child");
}

/// Every digest the reopened head tree and every snapshot references must
/// still verify: crashes may leak objects, never lose live ones.
fn assert_live_objects_verify(store: &WorkspaceStore, workspace: &str) {
    let sealed = store
        .seal_snapshot(workspace, None)
        .expect("the reopened head seals cleanly");
    let tree = store.load_snapshot(&sealed).expect("the seal round-trips");
    assert_eq!(tree.snapshot_id(), sealed);
    for (path, entry) in tree.entries() {
        if let Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, .. } = record {
                    store
                        .store()
                        .verify(digest)
                        .unwrap_or_else(|error| panic!("{path}: live object lost: {error}"));
                }
            }
        }
    }
}

#[test]
fn a_kill_before_any_commit_never_moves_the_head() {
    let root = TempRoot::new("admit");
    {
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.create_workspace("main").unwrap();
    }
    let child = spawn_child(&root.0, "admit-only");
    kill_after(child, Duration::from_millis(50));

    let store = WorkspaceStore::open(&root.0).unwrap();
    assert_eq!(store.head_generation("main").unwrap(), 0);
    assert_live_objects_verify(&store, "main");
}

#[test]
fn a_kill_inside_the_commit_stream_leaves_a_whole_generation() {
    let root = TempRoot::new("commit");
    {
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.create_workspace("main").unwrap();
    }
    let child = spawn_child(&root.0, "commit-loop");
    kill_after(child, Duration::from_millis(300));

    let store = WorkspaceStore::open(&root.0).unwrap();
    let head = store.head_generation("main").unwrap();

    // The tree at the surviving head is complete: exactly one file per
    // committed generation, every object verifiable. A torn commit would
    // surface as a missing file, a dangling digest, or a failed seal.
    let sealed = store.seal_snapshot("main", None).unwrap();
    let tree = store.load_snapshot(&sealed).unwrap();
    let files = tree
        .entries()
        .iter()
        .filter(|(_, entry)| matches!(entry, Entry::File { .. }))
        .count() as u64;
    assert_eq!(files, head, "one committed file per surviving generation");
    assert_live_objects_verify(&store, "main");
}

#[test]
fn a_kill_after_commit_preserves_the_new_generation_durably() {
    let root = TempRoot::new("durable");
    {
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.create_workspace("main").unwrap();
    }
    let child = spawn_child(&root.0, "commit-then-hold");
    kill_after(child, Duration::from_millis(50));

    let store = WorkspaceStore::open(&root.0).unwrap();
    assert_eq!(
        store.head_generation("main").unwrap(),
        1,
        "a committed generation survives the kill"
    );
    assert_live_objects_verify(&store, "main");
}

#[test]
fn a_kill_inside_gc_never_deletes_a_live_object() {
    let root = TempRoot::new("gc");
    let live_digest;
    {
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.create_workspace("main").unwrap();
        let object = store.store().put_bytes(b"the one live object").unwrap();
        live_digest = object.digest();
        store
            .commit_generation(
                "main",
                &[Mutation::CreateFile {
                    path: "live.bin".into(),
                    executable: false,
                    planner: PlannerId::BlobV1,
                    records: vec![FileRecord::Data {
                        digest: live_digest,
                        length: 19,
                    }],
                }],
            )
            .unwrap();
    }

    // Three separate kills across the mark/delete stream; the live object
    // must survive every one of them.
    for round in 0..3 {
        let child = spawn_child(&root.0, "collect-loop");
        kill_after(child, Duration::from_millis(120 + round * 90));
        let _ = fs::remove_file(root.0.join("child-ready"));

        let store = WorkspaceStore::open(&root.0).unwrap();
        assert!(
            store.store().verify(&live_digest).is_ok(),
            "round {round}: a kill inside GC lost a live object"
        );
        assert_live_objects_verify(&store, "main");
        // The interrupted pass leaves the quarantine coherent: further clean
        // passes still converge on reclaiming true orphans.
        store.collect().unwrap();
    }

    let store = WorkspaceStore::open(&root.0).unwrap();
    for _ in 0..3 {
        store.collect().unwrap();
    }
    assert!(store.store().verify(&live_digest).is_ok());
}
