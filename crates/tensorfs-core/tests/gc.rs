//! One reachability walk over the unified store, and the accounting that
//! falls out of it (`docs/mixed-cas-layout.md` §6-§7, issue #71).
//!
//! The claims, and why each is a test rather than an argument:
//!
//!  * **Both object kinds are marked.** With blobs and tensor chunks in ONE
//!    tree there is exactly one reachability story, so a snapshot whose only
//!    reference to an object is a `blob-v1` entry must keep it, and dropping
//!    that root must sweep it. Proven by deleting the root, not by inspecting
//!    a set.
//!  * **A manifest is an object at its own id**, so it is marked and swept by
//!    the same rules as the bytes it names.
//!  * **`exclusive` is the freed-if-deleted number**, asserted against the
//!    bytes a collection actually reclaims — a report that agreed with itself
//!    would prove nothing.
//!  * **Trees and refs pin nothing.** They are never walked for reachability
//!    and the scrub removes them when their root is gone; a scrub racing a
//!    deletion cannot cost a live root anything, because it removes no
//!    objects at all.

#![cfg(any(unix, windows))]

use std::collections::HashMap;
use std::fs;

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, ScrubReport, SnapshotUsage, WorkspaceStore};

mod harness;

use harness::{Consistency, Scratch};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Admits `bytes` and returns the mutation that names them as a whole blob —
/// the ONLY reference the resulting snapshot will have to that object.
fn blob(store: &WorkspaceStore, path: &str, bytes: &[u8]) -> Mutation {
    let object = store.store().put_bytes(bytes).expect("blob admits");
    Mutation::CreateFile {
        path: path.to_owned(),
        executable: false,
        planner: PlannerId::BlobV1,
        records: vec![FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        }],
    }
}

/// A workspace holding exactly the named blobs, sealed and then DELETED, so
/// the snapshot row is the only root its objects have. A live head would root
/// everything it names and make every exclusive number zero — truthfully, and
/// uselessly for a fixture about sharing.
fn sealed_only(store: &WorkspaceStore, workspace: &str, files: &[(&str, &[u8])]) -> SnapshotId {
    store
        .create_workspace(workspace)
        .expect("workspace creates");
    let mutations: Vec<Mutation> = files
        .iter()
        .map(|(path, bytes)| blob(store, path, bytes))
        .collect();
    store
        .commit_generation(workspace, &mutations)
        .expect("commit lands");
    let id = store.seal_snapshot(workspace, None).expect("seal lands");
    store
        .delete_workspace(workspace)
        .expect("workspace deletes");
    id
}

fn usage_by_id(store: &WorkspaceStore) -> HashMap<SnapshotId, SnapshotUsage> {
    store
        .usage()
        .expect("usage computes")
        .into_iter()
        .map(|entry| (entry.id, entry))
        .collect()
}

fn manifest_length(store: &WorkspaceStore, id: &SnapshotId) -> u64 {
    store
        .store()
        .verify(&ObjectDigest::from_bytes(*id.as_bytes()))
        .expect("a rooted manifest is resident")
}

/// Runs enough collection passes to carry one wave of unreferenced objects
/// all the way through quarantine, and returns the bytes reclaimed.
///
/// Four is a count, not a clock: an object quarantined at epoch `e` is deleted
/// at `e + 2`, and nothing here can cascade, so four passes always finish a
/// wave and no pass count can be short by a timing accident.
const EPOCHS_TO_RECLAIM: u32 = 4;

fn sweep(store: &WorkspaceStore) -> u64 {
    (0..EPOCHS_TO_RECLAIM)
        .map(|_| store.collect().expect("collection completes").bytes_deleted)
        .sum()
}

fn resident(store: &WorkspaceStore, digest: &ObjectDigest) -> bool {
    store.store().verify(digest).is_ok()
}

// ---------------------------------------------------------------------------
// Mark: both object kinds, and the manifest
// ---------------------------------------------------------------------------

/// A snapshot whose ONLY reference to an object is a `blob-v1` entry keeps
/// that object alive across any number of collections, and deleting the
/// snapshot sweeps it.
///
/// The red proof for this test is dropping the `FileBody::Blob` arm from
/// `manifest_objects` in `workspace.rs`: the blob is then reachable from
/// nothing, and the first pair of epochs takes it while the root still
/// exists — which is the entire hazard the unified walk exists to prevent.
#[test]
fn a_blob_entry_is_the_only_root_an_object_needs() {
    let scratch = Scratch::new("gc-blob-mark");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let payload = b"only ever named by a blob-v1 entry, never by a tensor record".to_vec();
    let digest = store.store().put_bytes(&payload).expect("admits").digest();
    let id = sealed_only(&store, "blobs", &[("weights.bin", &payload)]);
    let manifest = manifest_length(&store, &id);

    // Far past the two-epoch quarantine rule: a mark that missed the blob
    // would have reclaimed it several times over by now.
    sweep(&store);
    assert!(
        resident(&store, &digest),
        "an object a rooted snapshot names as a blob was swept: the mark does \
         not see blob-v1 entries"
    );

    store.delete_snapshot(&id).expect("snapshot deletes");
    let freed = sweep(&store);
    assert!(
        !resident(&store, &digest),
        "an object whose only root was a deleted snapshot survived collection"
    );
    assert_eq!(
        freed,
        payload.len() as u64 + manifest,
        "the reclaimed bytes must be exactly the blob and its manifest"
    );
    Consistency::scan(scratch.path()).assert_intact("after a blob-only sweep");
}

/// The manifest object survives GC for exactly as long as its root row does.
#[test]
fn a_manifest_blob_lives_and_dies_with_its_root_row() {
    let scratch = Scratch::new("gc-manifest");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let id = sealed_only(&store, "main", &[("config.json", br#"{"n":1}"#)]);
    let manifest = ObjectDigest::from_bytes(*id.as_bytes());

    let length = manifest_length(&store, &id);
    assert!(length > 0, "a manifest object must have bytes");
    sweep(&store);
    assert!(
        resident(&store, &manifest),
        "the manifest of a rooted snapshot was collected"
    );
    // Still decodable through the rooted path, not merely present on disk.
    assert_eq!(store.load_snapshot(&id).expect("loads").snapshot_id(), id);

    store.delete_snapshot(&id).expect("snapshot deletes");
    let freed = sweep(&store);
    assert!(
        !resident(&store, &manifest),
        "an unrooted manifest is an unreferenced object like any other"
    );
    assert!(
        freed >= length,
        "the sweep must have reclaimed the manifest's own {length} bytes, freed {freed}"
    );
    Consistency::scan(scratch.path()).assert_intact("after a manifest sweep");
}

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

/// The numeric fixture: two snapshots sharing one blob, with every length
/// distinct so no two numbers can be confused for each other.
///
/// The three numbers are asserted exactly, and then `exclusive` is checked
/// against reality — the bytes a real collection reclaims when one root is
/// deleted — because a report that only agreed with itself would prove
/// nothing.
#[test]
fn a_shared_blob_is_resident_to_both_roots_and_exclusive_to_neither() {
    let scratch = Scratch::new("gc-usage");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");

    let shared = vec![0xAB; 1000];
    let left_only = vec![0xCD; 17];
    let right_only = vec![0xEF; 250];
    let shared_digest = store.store().put_bytes(&shared).expect("admits").digest();

    let left = sealed_only(
        &store,
        "left",
        &[("shared.bin", &shared), ("left.bin", &left_only)],
    );
    let right = sealed_only(
        &store,
        "right",
        &[("shared.bin", &shared), ("right.bin", &right_only)],
    );
    assert_ne!(left, right, "the two snapshots must not be the same object");

    let left_manifest = manifest_length(&store, &left);
    let right_manifest = manifest_length(&store, &right);

    let usage = usage_by_id(&store);
    assert_eq!(usage.len(), 2, "both roots must be reported");
    eprintln!(
        "shared 1000 | left-only 17 | right-only 250 | manifests {left_manifest}/{right_manifest}\n\
         left  {:?}\nright {:?}",
        usage[&left], usage[&right]
    );
    assert_eq!(
        usage[&left],
        SnapshotUsage {
            id: left,
            // What a plain copy would occupy: the file bytes, manifest excluded.
            logical: 1000 + 17,
            // Real bytes this root reaches, its manifest object included.
            resident: 1000 + 17 + left_manifest,
            // The shared blob is reachable from the peer too, so it is not freed.
            exclusive: 17 + left_manifest,
        }
    );
    assert_eq!(
        usage[&right],
        SnapshotUsage {
            id: right,
            logical: 1000 + 250,
            resident: 1000 + 250 + right_manifest,
            exclusive: 250 + right_manifest,
        }
    );

    // Reality check: deleting `left` frees exactly its exclusive bytes.
    let expected = usage[&left].exclusive;
    store.delete_snapshot(&left).expect("snapshot deletes");
    let freed = sweep(&store);
    assert_eq!(
        freed, expected,
        "collection freed {freed} bytes where `exclusive` promised {expected}"
    );
    assert!(
        resident(&store, &shared_digest),
        "a blob shared with a surviving root must survive its peer's deletion"
    );

    // And the survivor now owns what it used to share.
    let usage = usage_by_id(&store);
    assert_eq!(
        usage[&right],
        SnapshotUsage {
            id: right,
            logical: 1000 + 250,
            resident: 1000 + 250 + right_manifest,
            exclusive: 1000 + 250 + right_manifest,
        }
    );
    Consistency::scan(scratch.path()).assert_intact("after shared-blob accounting");
}

/// A live workspace head is a root, so nothing a workspace still names is
/// exclusive to a snapshot — the honest answer, and the reason the fixture
/// above deletes its workspaces.
#[test]
fn a_live_head_keeps_a_snapshots_objects_off_its_exclusive_number() {
    let scratch = Scratch::new("gc-head-root");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let payload = vec![0x5A; 64];
    store.create_workspace("main").expect("workspace creates");
    store
        .commit_generation("main", &[blob(&store, "kept.bin", &payload)])
        .expect("commit lands");
    let id = store.seal_snapshot("main", None).expect("seal lands");

    let usage = usage_by_id(&store);
    assert_eq!(usage[&id].logical, 64);
    assert_eq!(usage[&id].resident, 64 + manifest_length(&store, &id));
    assert_eq!(
        usage[&id].exclusive,
        manifest_length(&store, &id),
        "the head roots the file bytes, so only the manifest is exclusive"
    );
}

// ---------------------------------------------------------------------------
// Scrub
// ---------------------------------------------------------------------------

/// The scrub removes a tree whose root row is gone and a ref that names an
/// unrooted snapshot, and touches neither for a live root.
#[test]
fn a_scrub_takes_dangling_trees_and_refs_and_leaves_live_ones() {
    let scratch = Scratch::new("gc-scrub");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let live = sealed_only(&store, "live", &[("a.bin", b"live snapshot bytes")]);
    let doomed = sealed_only(&store, "doomed", &[("b.bin", b"doomed snapshot bytes")]);

    let live_tree = store.project_snapshot(&live).expect("projects");
    let doomed_tree = store.project_snapshot(&doomed).expect("projects");
    let layout = store.layout();
    layout.set_ref("live", &live).expect("ref writes");
    layout.set_ref("doomed", &doomed).expect("ref writes");

    // A scrub with nothing to do must remove nothing.
    assert_eq!(
        store.scrub().expect("scrub runs"),
        ScrubReport::default(),
        "a scrub over a fully rooted layout removed something"
    );

    // The crash `delete_snapshot`'s ordering is designed to survive: the root
    // row is gone while the tree and the ref are still on disk. Built through
    // the real projector — `Layout::project` takes a decoded manifest and
    // never consults the root set, which is exactly why a tree pins nothing.
    let manifest = store.load_snapshot(&doomed).expect("loads");
    let _ = doomed_tree;
    store.delete_snapshot(&doomed).expect("snapshot deletes");
    let doomed_tree = layout.project(&manifest).expect("re-projects");
    layout.set_ref("doomed", &doomed).expect("ref writes");
    assert!(doomed_tree.is_dir() && layout.read_ref("doomed").unwrap() == Some(doomed));

    let report = store.scrub().expect("scrub runs");
    assert_eq!(report.trees_removed, vec![doomed]);
    assert_eq!(report.refs_removed, vec!["doomed".to_owned()]);
    assert!(
        !doomed_tree.exists(),
        "the dangling tree survived the scrub"
    );
    assert_eq!(layout.read_ref("doomed").unwrap(), None);

    assert!(live_tree.is_dir(), "the live root's tree was collateral");
    assert_eq!(layout.read_ref("live").unwrap(), Some(live));
    // Trees and refs pin nothing, so a scrub frees no object bytes.
    assert!(resident(
        &store,
        &ObjectDigest::from_bytes(*doomed.as_bytes())
    ));
    Consistency::scan(scratch.path()).assert_intact("after a scrub");
}

/// A ref whose `open` fails is left alone: a failed `open` says nothing about
/// the CONTENT, and on Windows a live ref transiently refuses to open while a
/// swap replaces it (#103). Only unrooted bytes, or bytes that are not an id,
/// make a ref dangling.
///
/// Unix-only because the refusal has to be manufactured, and a mode is the
/// portable way to manufacture one; the failure it stands in for is Windows'.
#[cfg(unix)]
#[test]
fn a_scrub_keeps_a_ref_it_could_not_open() {
    use std::os::unix::fs::PermissionsExt as _;

    let scratch = Scratch::new("gc-scrub-unreadable");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let live = sealed_only(&store, "live", &[("a.bin", b"live snapshot bytes")]);
    store.layout().set_ref("live", &live).expect("ref writes");

    let path = scratch.path().join("refs").join("live");
    let readable = fs::metadata(&path).expect("ref stats").permissions();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("ref closes");
    let report = store.scrub().expect("scrub runs");
    fs::set_permissions(&path, readable).expect("ref reopens");

    assert_eq!(
        report.refs_removed,
        Vec::<String>::new(),
        "a scrub deleted a ref it could not read"
    );
    assert_eq!(store.layout().read_ref("live").unwrap(), Some(live));
}

/// TEMPORARY (#109 probe): a live breadcrumb, flushed, so a hung run's log
/// names the operation each party was inside when the process wedged.
/// Deleted with the probe workflow before merge.
fn probe(what: &str) {
    use std::io::Write as _;
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "[probe] {what}");
    let _ = stderr.flush();
}

/// A scrub racing snapshot deletions never removes a live root's tree, its
/// ref, or any of its objects.
///
/// The scrubber runs on a SECOND connection — the shape `concurrency.rs` uses
/// for the collector race, so the two paths serialize through the database
/// exactly as two processes would — and hammers `scrub` for the whole run
/// while the main thread cycles snapshots through their entire life. Each
/// round deliberately reproduces the one interleaving the design calls out:
/// **a projection that lands after its own root was deleted**, which leaves a
/// tree the next scrub takes. That is what guarantees the scrubber has
/// something dangling to find while a live root sits beside it.
///
/// No clock anywhere. The scrubber stops on a flag the main thread sets after
/// a fixed number of rounds, and the gate on whether the race happened is a
/// COUNT of dangling trees the scrubber actually removed — not an elapsed
/// time, and not a sleep.
#[test]
fn a_scrub_racing_a_deletion_never_costs_a_live_root() {
    let scratch = Scratch::new("gc-scrub-race");
    let store = WorkspaceStore::open(scratch.path()).expect("store opens");
    let scrubber = WorkspaceStore::open(scratch.path()).expect("second connection opens");

    let survivor_bytes = b"the survivor's own bytes".to_vec();
    let survivor = sealed_only(&store, "survivor", &[("keep.bin", &survivor_bytes)]);
    let survivor_digest = store
        .store()
        .put_bytes(&survivor_bytes)
        .expect("admits")
        .digest();
    let survivor_tree = store.project_snapshot(&survivor).expect("projects");
    store
        .layout()
        .set_ref("survivor", &survivor)
        .expect("ref writes");

    let rounds = harness::iterations(12, 100);
    probe(&format!(
        "multiprocess={} rounds={rounds}",
        store.supports_multiprocess()
    ));
    // Raised when the rounds below LEAVE, panic included: a scrubber whose
    // stop flag only a healthy main thread sets turns any failure into a
    // hang, which is what #109 was.
    let stop = harness::StopFlag::new();
    let (scrubs, scrubbed_trees, deferred) = std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let mut passes = 0_u64;
            let mut trees = 0_u64;
            let mut deferred = 0_u64;
            while !stop.raised() {
                probe(&format!("s{passes}+"));
                let report = scrubber.scrub().expect("scrub runs");
                probe(&format!("s{passes}-"));
                passes += 1;
                trees += report.trees_removed.len() as u64;
                deferred += (report.trees_deferred.len() + report.refs_deferred.len()) as u64;
                assert!(
                    !report.trees_removed.contains(&survivor),
                    "a scrub removed a LIVE root's tree"
                );
                assert!(
                    !report.refs_removed.iter().any(|name| name == "survivor"),
                    "a scrub removed a LIVE root's ref"
                );
            }
            (passes, trees, deferred)
        });

        stop.racing(|| {
            for round in 0..rounds {
                probe(&format!("main: round {round} seal"));
                let doomed = sealed_only(
                    &store,
                    "doomed",
                    &[("round.bin", format!("round {round}").as_bytes())],
                );
                probe(&format!("main: round {round} load"));
                let manifest = store.load_snapshot(&doomed).expect("loads");
                probe(&format!("main: round {round} project"));
                store.project_snapshot(&doomed).expect("projects");
                probe(&format!("main: round {round} set_ref"));
                store.layout().set_ref("doomed", &doomed).expect("ref");
                probe(&format!("main: round {round} delete_snapshot"));
                store.delete_snapshot(&doomed).expect("deletes");
                // The late projection: a tree for a root that no longer exists.
                // It may also lose a rename race with the scrubber's own removal,
                // which is a cache outcome and not a claim this test makes.
                probe(&format!("main: round {round} late project"));
                let _ = store.layout().project(&manifest);
                probe(&format!("main: round {round} late set_ref"));
                let _ = store.layout().set_ref("doomed", &doomed);
                probe(&format!("main: round {round} asserts"));

                assert!(
                    survivor_tree.is_dir(),
                    "round {round}: the live root's tree was removed by a racing scrub"
                );
                assert_eq!(
                    store.layout().read_ref("survivor").unwrap(),
                    Some(survivor),
                    "round {round}: the live root's ref was removed by a racing scrub"
                );
                assert!(
                    resident(&store, &survivor_digest),
                    "round {round}: a scrub cost a live root its bytes"
                );
                probe(&format!("main: round {round} collect"));
                store.collect().expect("collection completes");
                probe(&format!("main: round {round} reload"));
                assert_eq!(
                    store.load_snapshot(&survivor).expect("loads").snapshot_id(),
                    survivor,
                    "round {round}: the live root stopped loading"
                );
                probe(&format!("main: round {round} done"));
            }
        });
        probe("main: joining the scrub thread");
        harness::join_worker(handle)
    });

    // `deferred` is zero on POSIX by construction and non-zero on Windows
    // whenever another handle was inside a doomed artifact — the outcome that
    // used to be an error, and used to end the run.
    eprintln!(
        "scrub race over {rounds} rounds: {scrubs} scrubs, {scrubbed_trees} dangling trees \
         removed, {deferred} removals deferred"
    );
    assert!(
        scrubbed_trees > 0,
        "the scrubber never removed a dangling tree, so it never ran inside \
         the window this test exists to cover"
    );
    assert!(
        survivor_tree.is_dir() && store.layout().tree_ids().unwrap().contains(&survivor),
        "the survivor's tree must still be projected at the end"
    );
    assert_eq!(store.layout().read_ref("survivor").unwrap(), Some(survivor));
    Consistency::scan(scratch.path()).assert_intact("after a scrub/deletion race");
}
