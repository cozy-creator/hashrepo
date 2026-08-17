//! A crashed projection's scratch is reclaimable; a live one is untouchable
//! (issue #88, split out of #71's scrub).
//!
//! The claim under test is a LIVENESS claim, not an age one. Every case here
//! is decided by whether the scratch's advisory lease can be taken:
//!
//!  * a projection running in this process, and one running in another that
//!    is blocked mid-`fill`, both keep their scratch however long they run;
//!  * a scratch whose holder died — or which never had a lease at all — goes
//!    on the very next `scrub`, with no grace to wait out;
//!  * a `.swap-…` staged ref gets exactly the guards a store temp gets.
//!
//! No clock appears in any of it, which is the point: a wall-clock grace long
//! enough to protect a 30 GB projection is long enough to strand one forever.

#![cfg(any(unix, windows))]

use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use fs4::FileExt;
use tensorfs_core::layout::{Layout, LeaseState};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

mod harness;

use harness::{Consistency, Scratch};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// A sealed snapshot holding `files` as whole blobs, with its workspace kept
/// so the root stays live and `scrub` has no reason to touch its tree.
fn sealed(store: &WorkspaceStore, workspace: &str, files: &[(String, Vec<u8>)]) -> SnapshotId {
    store
        .create_workspace(workspace)
        .expect("workspace creates");
    let mutations: Vec<Mutation> = files
        .iter()
        .map(|(path, bytes)| {
            let object = store.store().put_bytes(bytes).expect("blob admits");
            Mutation::CreateFile {
                path: path.clone(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest: object.digest(),
                    length: object.length(),
                }],
            }
        })
        .collect();
    store
        .commit_generation(workspace, &mutations)
        .expect("commit lands");
    store.seal_snapshot(workspace, None).expect("seal lands")
}

/// A sealed snapshot of `count` files that all name ONE shared object, so the
/// fixture costs a single admission and the projection still has thousands of
/// entries to write — which is the property the race below needs.
fn sealed_wide(store: &WorkspaceStore, workspace: &str, count: u32) -> SnapshotId {
    store
        .create_workspace(workspace)
        .expect("workspace creates");
    let object = store
        .store()
        .put_bytes(b"one shard, named by every entry")
        .expect("blob admits");
    let mutations: Vec<Mutation> = (0..count)
        .map(|index| Mutation::CreateFile {
            path: format!("shard-{index:05}.bin"),
            executable: false,
            planner: PlannerId::BlobV1,
            records: vec![FileRecord::Data {
                digest: object.digest(),
                length: object.length(),
            }],
        })
        .collect();
    store
        .commit_generation(workspace, &mutations)
        .expect("commit lands");
    store.seal_snapshot(workspace, None).expect("seal lands")
}

/// The scratch shapes `Layout::project` and `Layout::set_ref` write, spelled
/// out rather than imported: the reaper's contract is with the bytes on disk,
/// so a test that asked the library for the names could not catch the library
/// changing them out from under a reaper that still looks for the old ones.
fn building_dir(root: &Path, token: &str) -> std::path::PathBuf {
    root.join("snapshots").join(format!(".building-{token}"))
}

fn lease_file(root: &Path, token: &str) -> std::path::PathBuf {
    root.join("tmp").join(format!("building-{token}.tmp"))
}

fn swap_file(root: &Path, token: &str) -> std::path::PathBuf {
    root.join("refs").join(format!(".swap-{token}"))
}

/// Plants one scratch tree with 24 bytes in it, and its lease.
fn plant_scratch(root: &Path, token: &str, with_lease: bool) {
    let scratch = building_dir(root, token);
    fs::create_dir_all(&scratch).expect("scratch plants");
    fs::write(
        scratch.join("half-projected.bin"),
        b"partially projected bytes",
    )
    .expect("scratch content plants");
    if with_lease {
        fs::write(lease_file(root, token), b"").expect("lease plants");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A projection running right now keeps its scratch, and lands correctly,
/// while a reaper hammers the same directory for its whole duration.
///
/// The gate is a COUNT — how many `.building-…` entries the reaper actually
/// saw — not an elapsed time. A run where the reaper never overlapped the
/// projection proves nothing and says so.
#[test]
fn a_live_projection_is_never_reaped_and_still_lands() {
    let scratch = Scratch::new("scratch-live");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let id = sealed_wide(&store, "live", harness::iterations(600, 20000));
    let snapshot = store.load_snapshot(&id).expect("snapshot loads");
    let layout = store.layout();

    let done = AtomicBool::new(false);
    let (reaps, examined, removed) = std::thread::scope(|scope| {
        let reaper = scope.spawn(|| {
            let (mut reaps, mut examined, mut removed) = (0_u64, 0_u64, 0_u64);
            while !done.load(Ordering::Acquire) {
                let report = layout.reap_scratch().expect("reap runs");
                reaps += 1;
                examined += report.examined;
                removed += report.trees_removed;
            }
            (reaps, examined, removed)
        });
        let stop = StopOnDrop(&done);
        let projected = layout.project(&snapshot).expect("projection lands");
        // Normal exit; the guard covers the panicking one.
        drop(stop);
        let counts = reaper.join().expect("reaper joins");
        assert!(projected.is_dir(), "the projected tree is missing");
        // Symlink or copy — the runner decides, and either is a finished
        // entry. `read_link` would have asserted the filesystem, not the
        // projection.
        assert!(
            projected.join("shard-00000.bin").exists()
                && projected.join("shard-00599.bin").exists(),
            "the projection did not finish its entries"
        );
        counts
    });

    eprintln!("{reaps} reaps overlapped the projection, examining {examined} scratch entries");
    assert_eq!(removed, 0, "a reaper removed a LIVE projection's scratch");
    assert!(
        examined > 0,
        "the reaper never saw the scratch, so it never ran inside the window \
         this test exists to cover"
    );
    assert!(
        store
            .layout()
            .tree_ids()
            .expect("trees enumerate")
            .contains(&id),
        "the projected tree must be enumerable after the race"
    );
    Consistency::scan(scratch.path()).assert_intact("after a projection/reap race");
}

/// `scrub` reaps scratch whose holder is gone — with a free lease or with no
/// lease at all — and leaves a held one, a live root's tree, and its ref.
#[test]
fn scrub_reaps_dead_scratch_and_leaves_a_held_one() {
    let scratch = Scratch::new("scratch-dead");
    let root = scratch.path();
    let store = WorkspaceStore::open(root).expect("store opens");
    let id = sealed(
        &store,
        "live",
        &[("keep.bin".to_owned(), b"the live root's bytes".to_vec())],
    );
    let live_tree = store.project_snapshot(&id).expect("projects");
    store.layout().set_ref("live", &id).expect("ref writes");

    // A crashed projector's lease is free the moment it dies; a lease removed
    // before its scratch was is the same statement with fewer files.
    plant_scratch(root, "dead-lease", true);
    plant_scratch(root, "no-lease", false);
    // A holder that is still alive. This process takes the lease on its own
    // descriptor, which is exactly what another process's would look like.
    plant_scratch(root, "held", true);
    let held = File::open(lease_file(root, "held")).expect("held lease opens");
    held.try_lock_exclusive().expect("held lease locks");
    // A `set_ref` killed between its write and its rename.
    fs::write(swap_file(root, "dead-swap"), b"0".repeat(65)).expect("staged ref plants");
    fs::write(lease_file(root, "dead-swap"), b"").expect("its lease plants");

    let report = store.scrub().expect("scrub runs");
    assert_eq!(report.scratch.examined, 4);
    assert_eq!(report.scratch.trees_removed, 2);
    assert_eq!(report.scratch.swaps_removed, 1);
    assert!(report.trees_removed.is_empty() && report.refs_removed.is_empty());

    // The evidence outlives the artifacts: after this call the names are gone
    // and only the report can say whose they were, and why they were taken.
    let mut evidence: Vec<(&str, LeaseState)> = report
        .scratch
        .reclaimed
        .iter()
        .map(|item| (item.creator.as_str(), item.lease))
        .collect();
    evidence.sort_unstable_by_key(|(creator, _)| *creator);
    assert_eq!(
        evidence,
        [
            ("dead-lease", LeaseState::Free),
            ("dead-swap", LeaseState::Free),
            ("no-lease", LeaseState::Absent),
        ],
        "a reap must record who left each artifact and why it was reclaimable"
    );
    assert!(
        report
            .scratch
            .reclaimed
            .iter()
            .all(|item| !item.path.exists()),
        "every recorded path must actually be gone"
    );

    assert!(!building_dir(root, "dead-lease").exists());
    assert!(
        !lease_file(root, "dead-lease").exists(),
        "an orphan lease is a leak"
    );
    assert!(!building_dir(root, "no-lease").exists());
    assert!(!swap_file(root, "dead-swap").exists());
    assert!(
        building_dir(root, "held").is_dir() && lease_file(root, "held").exists(),
        "a held projection's scratch was reaped"
    );
    assert!(live_tree.is_dir(), "the live root's tree was collateral");
    assert_eq!(store.layout().read_ref("live").unwrap(), Some(id));

    // Idempotent, and still bounded by the lease on a second pass.
    let again = store.scrub().expect("second scrub runs");
    assert_eq!(again.scratch.examined, 1);
    assert_eq!(again.scratch.trees_removed, 0);
    assert!(building_dir(root, "held").is_dir());

    // Dropping the lease is the only thing that changes the answer.
    drop(held);
    let after = store.scrub().expect("third scrub runs");
    assert_eq!(after.scratch.trees_removed, 1);
    assert!(!building_dir(root, "held").exists() && !lease_file(root, "held").exists());
    Consistency::scan(root).assert_intact("after reaping dead scratch");
}

// ---------------------------------------------------------------------------
// A real killed projector
// ---------------------------------------------------------------------------

/// Stops the reaper thread on drop, so a panic inside the scope below ends
/// the run rather than deadlocking it: `thread::scope` joins before it
/// propagates, and a reaper spinning on a flag its panicking partner never
/// set would never be joined. A failure must stay a failure, never a hang.
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A child killed on drop, so a failing assertion cannot leave one parked in
/// `fill` forever holding this test binary's stdout — which would hang the
/// run rather than fail it. Leaking the process a test exists to kill would
/// be this lane's own bug, reproduced in its own harness.
struct ChildGuard(std::process::Child);

impl ChildGuard {
    fn pid(&self) -> u32 {
        self.0.id()
    }

    fn kill(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

const ROLE_ENV: &str = "TENSORFS_SCRATCH_ROLE";
const ROOT_ENV: &str = "TENSORFS_SCRATCH_ROOT";
const SNAPSHOT_ENV: &str = "TENSORFS_SCRATCH_SNAPSHOT";

/// Child dispatch: a no-op pass unless the parent set the role env.
///
/// The child projects for real, without symlinks, and blocks forever inside
/// `fill` because the parent replaced the one object it must copy with a
/// writerless FIFO. Nothing here simulates the crash: the scratch tree and
/// its lease are made by the library, on the real code path.
#[test]
#[cfg(unix)]
fn scratch_child_role() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    assert_eq!(role, "project");
    let root = std::env::var(ROOT_ENV).expect("the parent supplies the store root");
    let id = std::env::var(SNAPSHOT_ENV).expect("the parent supplies the snapshot id");

    let store = tensorfs_core::store::ObjectStore::open(&root).expect("child opens the store");
    let digest = SnapshotId::parse_hex(&id).expect("the id parses");
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut store
            .open_object(&ObjectDigest::from_bytes(*digest.as_bytes()))
            .expect("the manifest object opens"),
        &mut bytes,
    )
    .expect("the manifest reads");
    let snapshot = tensorfs_core::tfm1::decode(&bytes).expect("the manifest decodes");
    let _ = Layout::without_symlinks(&store).project(&snapshot);
    unreachable!("the child must block inside fill until it is killed");
}

#[test]
#[cfg(unix)]
fn a_sigkilled_projector_leaves_scratch_the_next_scrub_reaps() {
    use std::ffi::CString;
    use std::process::Command;

    let scratch = Scratch::new("scratch-kill");
    let root = scratch.path();
    let store = WorkspaceStore::open(root).expect("store opens");
    let payload = b"the object a killed projection was copying".to_vec();
    let id = sealed(
        &store,
        "victim",
        &[("blob.bin".to_owned(), payload.clone())],
    );
    let live = sealed(
        &store,
        "bystander",
        &[("keep.bin".to_owned(), b"a bystander root".to_vec())],
    );
    let live_tree = store.project_snapshot(&live).expect("bystander projects");

    // Turn the one object the child must copy into a FIFO with no writer, so
    // `fs::copy`'s open blocks forever and the child is deterministically
    // parked inside `fill` with its scratch on disk.
    let hex = store
        .store()
        .put_bytes(&payload)
        .expect("object re-admits")
        .digest()
        .to_hex();
    let object = root
        .join("objects")
        .join("sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(&hex);
    fs::remove_file(&object).expect("object unlinks");
    let raw = CString::new(object.as_os_str().as_encoded_bytes()).expect("path is nul-free");
    assert_eq!(
        unsafe { libc::mkfifo(raw.as_ptr(), 0o644) },
        0,
        "mkfifo failed: {}",
        std::io::Error::last_os_error()
    );

    let mut child = ChildGuard(
        Command::new(std::env::current_exe().expect("test binary path"))
            .args(["scratch_child_role", "--exact", "--nocapture"])
            .env(ROLE_ENV, "project")
            .env(ROOT_ENV, root)
            .env(SNAPSHOT_ENV, hex_of(&id))
            // Nulled so a parked child can never hold this binary's stdout
            // open and turn a failure into a hang.
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("child spawns"),
    );

    // Liveness, not a deadline: an ALIVE child is working however slow the
    // box is, and one that EXITED failed. No wall-clock bound belongs here.
    let scratch_dir = loop {
        if let Some(found) = fs::read_dir(root.join("snapshots"))
            .expect("snapshots dir reads")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".building-"))
            })
        {
            break found;
        }
        if let Some(status) = child.0.try_wait().expect("child status is readable") {
            panic!("child exited ({status}) before creating its scratch");
        }
    };

    let held = store.layout().reap_scratch().expect("reap runs");
    assert_eq!(
        held.trees_removed, 0,
        "a live projector's scratch was reaped"
    );
    assert!(held.examined >= 1);
    assert!(scratch_dir.is_dir());

    let child_pid = child.pid();
    child.kill();

    // No grace: the lease died with its holder, so the next scrub takes it.
    let report = store.scrub().expect("scrub runs");
    assert_eq!(report.scratch.trees_removed, 1);
    // The dead child is named in the evidence, by the token IT wrote.
    assert_eq!(
        report
            .scratch
            .reclaimed
            .iter()
            .map(|item| item.creator.starts_with(&format!("{child_pid}-")))
            .collect::<Vec<_>>(),
        vec![true],
        "the reap must name the crashed projector that left the scratch"
    );
    assert!(
        !scratch_dir.exists(),
        "the crashed scratch survived the scrub"
    );
    assert!(
        !fs::read_dir(root.join("tmp"))
            .expect("tmp reads")
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().starts_with("building-")),
        "the crashed projection's lease is a leak"
    );
    assert!(
        live_tree.is_dir(),
        "the bystander root's tree was collateral"
    );
    assert!(store.layout().tree_ids().unwrap().contains(&live));
}

#[cfg(unix)]
fn hex_of(id: &SnapshotId) -> String {
    ObjectDigest::from_bytes(*id.as_bytes()).to_hex()
}
