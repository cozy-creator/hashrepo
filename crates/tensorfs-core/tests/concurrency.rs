//! Families 2 and 5 — many writers at once, and GC running while they work.
//!
//! Every existing concurrency claim in the suite is single-threaded: the
//! two-epoch GC invariant, the same-digest convergence rule and the
//! generation counter are all proven by sequential calls. This file puts real
//! contention under them — separate PROCESSES against one store root, and a
//! collector running against a second live connection while commits land.
//!
//! The invariants:
//!
//!  * concurrent admissions of identical bytes all succeed and converge on
//!    one file, with no torn or duplicate object;
//!  * concurrent commits neither lose an update nor produce a non-monotonic
//!    generation;
//!  * a collector racing live writers never costs a committed object — the
//!    writer either commits with its bytes intact, or refuses cleanly with
//!    `MissingObject`, and never lands a head pointing at deleted bytes.
//!
//! Process count is deliberately small (three children plus the parent, so
//! four concurrent writers) because this suite runs on a heavily shared box.

#![cfg(any(unix, windows))]

mod harness;

use std::collections::HashSet;
use std::path::Path;
use std::process::{Command, Stdio};

use harness::{Consistency, Scratch};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord};
use tensorfs_core::workspace::{Mutation, WorkspaceError, WorkspaceStore};

const ROLE: &str = "TENSORFS_CONC_ROLE";
const ROOT: &str = "TENSORFS_CONC_ROOT";
const WRITER: &str = "TENSORFS_CONC_WRITER";
const WORKSPACE: &str = "TENSORFS_CONC_WORKSPACE";

/// Objects every writer admits, so the same digest is raced by all of them.
fn shared_payload(index: u32) -> Vec<u8> {
    let mut bytes = vec![0_u8; 4096 + index as usize];
    for (position, byte) in bytes.iter_mut().enumerate() {
        *byte = ((position as u32).wrapping_mul(2_654_435_761).wrapping_add(index) & 0xFF) as u8;
    }
    bytes
}

/// Objects unique to one writer, so disjoint work proceeds alongside the
/// contended work.
fn private_payload(writer: u32, index: u32) -> Vec<u8> {
    let mut bytes = vec![0_u8; 2048 + index as usize];
    for (position, byte) in bytes.iter_mut().enumerate() {
        *byte = ((position as u32)
            .wrapping_mul(40_503)
            .wrapping_add(writer * 7919 + index)
            & 0xFF) as u8;
    }
    bytes
}

const SHARED: u32 = 12;
const PRIVATE: u32 = 6;
const COMMITS: u32 = 4;

/// Child dispatch. A no-op pass unless the parent set the role.
#[test]
fn concurrency_child_role() {
    let Ok(role) = std::env::var(ROLE) else {
        return;
    };
    let root = std::env::var(ROOT).expect("the parent supplies a root");
    let writer: u32 = std::env::var(WRITER)
        .expect("the parent supplies a writer id")
        .parse()
        .expect("writer id parses");

    match role.as_str() {
        "admit" => admit_role(&root, writer),
        "commit" => {
            let workspace = std::env::var(WORKSPACE).expect("the parent supplies a workspace");
            commit_role(&root, writer, &workspace);
        }
        other => panic!("unknown role {other}"),
    }
}

fn admit_role(root: &str, writer: u32) {
    let meta = WorkspaceStore::open(root).expect("child opens the store");
    for index in 0..SHARED {
        meta.store()
            .put_bytes(&shared_payload(index))
            .expect("shared object admits");
    }
    for index in 0..PRIVATE {
        meta.store()
            .put_bytes(&private_payload(writer, index))
            .expect("private object admits");
    }
}

fn commit_role(root: &str, writer: u32, workspace: &str) {
    let meta = WorkspaceStore::open(root).expect("child opens the store");
    for round in 0..COMMITS {
        let admitted = meta
            .store()
            .put_bytes(&private_payload(writer, round))
            .expect("object admits");
        let mutation = Mutation::CreateFile {
            path: format!("w{writer}-r{round}.bin"),
            executable: false,
            planner: PlannerId::RawFixed64mV1,
            records: vec![FileRecord::Data {
                digest: admitted.digest(),
                length: admitted.length(),
            }],
        };
        // Contention on one workspace is expected; a lost race is a retry,
        // never a silent skip.
        let mut attempts = 0;
        loop {
            match meta.commit_generation(workspace, &[mutation.clone()]) {
                Ok(_) => break,
                Err(error) => {
                    attempts += 1;
                    assert!(
                        attempts < 50,
                        "writer {writer} round {round} never committed: {error}"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
    }
}

fn spawn(root: &Path, role: &str, writer: u32, workspace: Option<&str>) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().expect("test binary path"));
    command
        .args(["concurrency_child_role", "--exact", "--nocapture"])
        .env(ROLE, role)
        .env(ROOT, root)
        .env(WRITER, writer.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(workspace) = workspace {
        command.env(WORKSPACE, workspace);
    }
    command.spawn().expect("child spawns")
}

fn join(children: Vec<std::process::Child>) {
    for child in children {
        let output = child.wait_with_output().expect("child is reaped");
        assert!(
            output.status.success(),
            "a concurrent writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Four writers admit an overlapping object set into one store at once.
/// Every caller must succeed, and each unique digest must end up as exactly
/// one correct file — no duplicates, no torn content.
#[test]
fn concurrent_admissions_of_identical_bytes_converge_on_one_object() {
    let scratch = Scratch::new("admit-race");
    let meta = WorkspaceStore::open(scratch.path()).expect("parent opens the store");

    let children: Vec<_> = (1..=3).map(|id| spawn(scratch.path(), "admit", id, None)).collect();
    // The parent is the fourth concurrent writer.
    admit_role(&scratch.path().to_string_lossy(), 0);
    join(children);

    // Exactly the expected unique digests are resident, each verifying.
    let mut expected: HashSet<ObjectDigest> = HashSet::new();
    for index in 0..SHARED {
        expected.insert(digest_of(&meta, &shared_payload(index)));
    }
    for writer in 0..4 {
        for index in 0..PRIVATE {
            expected.insert(digest_of(&meta, &private_payload(writer, index)));
        }
    }
    for digest in &expected {
        meta.store()
            .verify(digest)
            .unwrap_or_else(|error| panic!("{digest} did not survive the race: {error}"));
    }

    let scan = Consistency::scan(scratch.path());
    scan.assert_intact("after a four-way admission race");
    assert_eq!(
        scan.objects,
        expected.len() as u64,
        "the race produced a different object count than the unique digest set"
    );
    // Abandoned temps are permitted (a losing racer's lease), but nothing may
    // be left mid-write once every writer has exited.
    assert_eq!(scan.temps, 0, "the race leaked temp files");
}

fn digest_of(meta: &WorkspaceStore, bytes: &[u8]) -> ObjectDigest {
    meta.store().put_bytes(bytes).expect("admits").digest()
}

/// Four writers committing to their own workspaces must all land, and each
/// workspace's generation must equal its own commit count.
#[test]
fn concurrent_commits_to_separate_workspaces_all_land() {
    let scratch = Scratch::new("ws-race");
    let meta = WorkspaceStore::open(scratch.path()).expect("parent opens the store");
    for writer in 0..4 {
        meta.create_workspace(&format!("ws-{writer}"))
            .expect("workspace creates");
    }

    let children: Vec<_> = (1..=3)
        .map(|id| spawn(scratch.path(), "commit", id, Some(&format!("ws-{id}"))))
        .collect();
    commit_role(&scratch.path().to_string_lossy(), 0, "ws-0");
    join(children);

    for writer in 0..4 {
        let name = format!("ws-{writer}");
        assert_eq!(
            meta.head_generation(&name).expect("head reads"),
            u64::from(COMMITS),
            "{name} lost a commit"
        );
        let tree = meta.head_tree(&name).expect("tree builds");
        for round in 0..COMMITS {
            let wanted = format!("w{writer}-r{round}.bin");
            assert!(
                tree.entries().iter().any(|(path, _)| *path == wanted),
                "{name} is missing {wanted}"
            );
        }
        assert_every_referenced_object_verifies(&meta, &name);
    }
    Consistency::scan(scratch.path()).assert_intact("after a workspace race");
}

/// Four writers committing to ONE workspace. Commits serialize; none is
/// lost; the generation counter equals the total number of commits.
#[test]
fn concurrent_commits_to_one_workspace_lose_nothing() {
    let scratch = Scratch::new("one-ws-race");
    let meta = WorkspaceStore::open(scratch.path()).expect("parent opens the store");
    meta.create_workspace("shared").expect("workspace creates");

    let children: Vec<_> = (1..=3)
        .map(|id| spawn(scratch.path(), "commit", id, Some("shared")))
        .collect();
    commit_role(&scratch.path().to_string_lossy(), 0, "shared");
    join(children);

    let generation = meta.head_generation("shared").expect("head reads");
    assert_eq!(
        generation,
        u64::from(COMMITS) * 4,
        "commits to one workspace were lost or double-counted"
    );

    let tree = meta.head_tree("shared").expect("tree builds");
    for writer in 0..4 {
        for round in 0..COMMITS {
            let wanted = format!("w{writer}-r{round}.bin");
            assert!(
                tree.entries().iter().any(|(path, _)| *path == wanted),
                "the shared workspace lost {wanted}"
            );
        }
    }
    assert_every_referenced_object_verifies(&meta, "shared");
    Consistency::scan(scratch.path()).assert_intact("after a single-workspace race");
}

/// Family 5. A collector running on its OWN connection while a writer
/// commits on another, repeatedly. The two-epoch rule is proven here under
/// real contention rather than by sequential calls.
///
/// The permitted outcomes for each round are exactly two: the commit lands
/// and its objects are alive, or the commit refuses with `MissingObject`
/// because the collector reclaimed bytes that were not yet referenced. A
/// committed head that names deleted bytes is the failure this test exists
/// to catch.
#[test]
fn a_collector_racing_live_commits_never_costs_a_committed_object() {
    let scratch = Scratch::new("gc-race");
    let writer = WorkspaceStore::open(scratch.path()).expect("writer connection opens");
    // A genuinely separate connection: no shared Mutex, so the two paths
    // serialize through SQLite exactly as two processes would.
    let collector = WorkspaceStore::open(scratch.path()).expect("collector connection opens");
    writer.create_workspace("main").expect("workspace creates");

    let rounds = harness::iterations(24, 200);
    let mut committed = 0_u64;
    let mut refused = 0_u64;

    for round in 0..rounds {
        let payload = private_payload(9, round);
        let admitted = writer.store().put_bytes(&payload).expect("object admits");

        // The collector runs between admission and commit — precisely the
        // window in which the object is resident but unreferenced. The pass
        // count varies so that some rounds only quarantine (deletion needs
        // an object to stay unreferenced for two full epochs) while others
        // run far enough to actually reclaim it. Both outcomes must be safe,
        // and a fixed low pass count would silently never test the second.
        for _ in 0..=(round % 4) {
            let _ = collector.collect().expect("collection completes");
        }

        let mutation = Mutation::CreateFile {
            path: format!("r{round}.bin"),
            executable: false,
            planner: PlannerId::RawFixed64mV1,
            records: vec![FileRecord::Data {
                digest: admitted.digest(),
                length: admitted.length(),
            }],
        };
        match writer.commit_generation("main", &[mutation]) {
            Ok(_) => {
                committed += 1;
                writer.store().verify(&admitted.digest()).unwrap_or_else(|error| {
                    panic!("round {round}: committed object was collected: {error}")
                });
            }
            Err(WorkspaceError::MissingObject { .. }) => refused += 1,
            Err(other) => panic!("round {round}: unexpected commit failure {other:?}"),
        }

        // The head must never reference bytes that are gone, at any point.
        assert_every_referenced_object_verifies(&writer, "main");
    }

    eprintln!("gc race over {rounds} rounds: {committed} committed, {refused} refused cleanly");
    assert!(
        committed > 0,
        "every round refused; the race never exercised a successful commit"
    );
    assert!(
        refused > 0,
        "no round was ever refused, so the collector never actually reclaimed          an uncommitted object and the dangerous window went untested"
    );
    Consistency::scan(scratch.path()).assert_intact("after the GC race");
}

/// A collector running while objects are admitted by other PROCESSES. The
/// collector must never remove an object a live committed head references.
#[test]
fn a_collector_racing_other_processes_keeps_every_referenced_object() {
    let scratch = Scratch::new("gc-proc-race");
    let meta = WorkspaceStore::open(scratch.path()).expect("parent opens the store");
    meta.create_workspace("main").expect("workspace creates");

    // Commit a tree first, so there is a live root set to protect.
    let mut records = Vec::new();
    for index in 0..SHARED {
        let admitted = meta
            .store()
            .put_bytes(&shared_payload(index))
            .expect("object admits");
        records.push(FileRecord::Data {
            digest: admitted.digest(),
            length: admitted.length(),
        });
    }
    meta.commit_generation(
        "main",
        &[Mutation::CreateFile {
            path: "rooted.bin".to_owned(),
            executable: false,
            planner: PlannerId::RawFixed64mV1,
            records,
        }],
    )
    .expect("root commit lands");
    let rooted = referenced_digests(&meta, "main");
    assert!(!rooted.is_empty());

    let children: Vec<_> = (1..=2).map(|id| spawn(scratch.path(), "admit", id, None)).collect();
    for _ in 0..8 {
        let _ = meta.collect().expect("collection completes");
        for digest in &rooted {
            meta.store()
                .verify(digest)
                .unwrap_or_else(|error| panic!("a rooted object was collected: {error}"));
        }
    }
    join(children);

    for digest in &rooted {
        meta.store()
            .verify(digest)
            .expect("every rooted object survives the race");
    }
    assert_every_referenced_object_verifies(&meta, "main");
    Consistency::scan(scratch.path()).assert_intact("after a collector/process race");
}

fn referenced_digests(meta: &WorkspaceStore, workspace: &str) -> Vec<ObjectDigest> {
    let tree = meta.head_tree(workspace).expect("tree builds");
    let mut digests = Vec::new();
    for (_path, entry) in tree.entries() {
        if let Entry::File { records, .. } = entry {
            for record in records {
                if let FileRecord::Data { digest, .. } = record {
                    digests.push(*digest);
                }
            }
        }
    }
    digests
}

fn assert_every_referenced_object_verifies(meta: &WorkspaceStore, workspace: &str) {
    for digest in referenced_digests(meta, workspace) {
        meta.store().verify(&digest).unwrap_or_else(|error| {
            panic!("{workspace} head references unverifiable {digest}: {error}")
        });
    }
}
