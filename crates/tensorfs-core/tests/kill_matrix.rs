//! Family 1 — SIGKILL at arbitrary points, many times, over one store.
//!
//! The existing crash coverage kills at CHOSEN cut points: before a commit,
//! inside the commit stream, after the commit, inside GC. That proves those
//! four points are safe. It does not prove the points between them are, and
//! it never asks what happens to a store that has been killed a hundred
//! times in a row.
//!
//! This matrix kills a child at a randomised delay spanning a full
//! admit → commit → seal cycle, against ONE accumulating store root, and
//! re-checks every invariant after each kill:
//!
//!  * the store reopens;
//!  * no resident object's bytes disagree with its digest name (checked by
//!    an independent on-disk rehash, not by asking the library);
//!  * the workspace head references only objects that verify;
//!  * every snapshot the child reported as sealed loads and is fully backed;
//!  * the generation advances monotonically and never regresses.
//!
//! The kill-point distribution is printed, so coverage is observable rather
//! than assumed: a matrix whose kills all land in one phase is not a matrix.
//!
//! Default run: 30 iterations, well under a minute. `TENSORFS_HEAVY=1` runs
//! 200. Any failure prints the seed for exact replay.
//!
//! The heavy run costs QUADRATIC time, not 6.7x the default: the store
//! accumulates across rounds, and each round re-verifies the whole head tree
//! and rehashes every resident object. That is the point — the invariant is
//! checked against everything the store has ever accumulated, not just the
//! latest round — but it means the 200-round variant takes tens of minutes
//! on a loaded box and belongs in a deliberate soak, never in per-PR CI.

#![cfg(any(unix, windows))]

mod harness;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use harness::{Consistency, Rng, Scratch, iterations, seed_from_env};
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

const ROLE: &str = "TENSORFS_KILL_ROLE";
const ROOT: &str = "TENSORFS_KILL_ROOT";
const ROUND: &str = "TENSORFS_KILL_ROUND";

const OBJECTS: u32 = 4;

fn phase_log(root: &Path) -> std::path::PathBuf {
    root.join("phase.log")
}

fn sealed_log(root: &Path) -> std::path::PathBuf {
    root.join("sealed.log")
}

/// Appends one phase marker. `O_APPEND` writes of this size are atomic, so
/// the parent always reads whole lines even when the child dies mid-cycle.
fn mark(root: &Path, phase: &str) {
    use std::io::Write as _;
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(phase_log(root))
    {
        let _ = writeln!(file, "{phase}");
        let _ = file.flush();
    }
}

fn payload(round: u32, index: u32) -> Vec<u8> {
    let mut bytes = vec![0_u8; 8192 + index as usize];
    for (position, byte) in bytes.iter_mut().enumerate() {
        *byte = ((position as u32)
            .wrapping_mul(2_246_822_519)
            .wrapping_add(round.wrapping_mul(97).wrapping_add(index))
            & 0xFF) as u8;
    }
    bytes
}

/// Child dispatch: one full admit → commit → seal cycle, narrating phases.
#[test]
fn kill_child_role() {
    let Ok(role) = std::env::var(ROLE) else {
        return;
    };
    assert_eq!(role, "cycle");
    let root = std::env::var(ROOT).expect("the parent supplies a root");
    let round: u32 = std::env::var(ROUND)
        .expect("the parent supplies a round")
        .parse()
        .expect("round parses");
    let root = Path::new(&root);

    mark(root, "open");
    let meta = WorkspaceStore::open(root).expect("child opens the store");
    if round == 0 {
        let _ = meta.create_workspace("main");
    }

    let mut records = Vec::new();
    for index in 0..OBJECTS {
        mark(root, &format!("admit-{index}"));
        let admitted = meta
            .store()
            .put_bytes(&payload(round, index))
            .expect("object admits");
        records.push(FileRecord::Data {
            digest: admitted.digest(),
            length: admitted.length(),
        });
    }

    mark(root, "commit");
    meta.commit_generation(
        "main",
        &[Mutation::CreateFile {
            path: format!("round-{round}.bin"),
            executable: false,
            planner: PlannerId::RawFixed64mV1,
            records,
        }],
    )
    .expect("commit lands");

    mark(root, "seal");
    let id = meta.seal_snapshot("main", None).expect("seal lands");

    // Recorded only after the seal returned, so the parent's "every sealed
    // snapshot must load" check can never be satisfied by a snapshot the
    // child never actually completed.
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(sealed_log(root))
        .expect("sealed log opens");
    writeln!(file, "{id}").expect("sealed id records");
    file.sync_all().expect("sealed log is durable");

    mark(root, "done");
}

fn spawn(root: &Path, round: u32) -> Child {
    Command::new(std::env::current_exe().expect("test binary path"))
        .args(["kill_child_role", "--exact", "--nocapture"])
        .env(ROLE, "cycle")
        .env(ROOT, root)
        .env(ROUND, round.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("child spawns")
}

/// The last phase the child announced before dying.
fn last_phase(root: &Path) -> String {
    fs::read_to_string(phase_log(root))
        .ok()
        .and_then(|log| log.lines().next_back().map(str::to_owned))
        .unwrap_or_else(|| "none".to_owned())
}

fn sealed_ids(root: &Path) -> Vec<SnapshotId> {
    fs::read_to_string(sealed_log(root))
        .unwrap_or_default()
        .lines()
        .filter_map(SnapshotId::parse_hex)
        .collect()
}

/// Every invariant that must hold after any kill, at any point.
fn assert_recovered(
    root: &Path,
    seed: u64,
    round: u32,
    phase: &str,
    previous_generation: u64,
) -> u64 {
    let context = format!("seed {seed}, round {round}, killed at {phase}");

    // An independent on-disk rehash: nothing at a digest path may disagree
    // with its name, and nothing malformed may appear under objects/.
    // `assert_examined`, not `assert_intact`: this is the matrix's primary
    // independent oracle after every round, so it must prove it actually
    // rehashed something. `calibrate` runs one uninterrupted cycle before the
    // first round, so the store is never legitimately empty here.
    Consistency::scan(root).assert_examined(1, &context);

    let meta = WorkspaceStore::open(root)
        .unwrap_or_else(|error| panic!("{context}: the store did not reopen: {error}"));

    let generation = meta
        .head_generation("main")
        .unwrap_or_else(|error| panic!("{context}: head unreadable: {error}"));
    assert!(
        generation >= previous_generation,
        "{context}: generation regressed from {previous_generation} to {generation}"
    );

    // The head is a whole generation, never a hybrid: everything it names
    // verifies against its bytes.
    let tree = meta
        .head_tree("main")
        .unwrap_or_else(|error| panic!("{context}: head tree unreadable: {error}"));
    for (path, entry) in tree.entries() {
        let Entry::File { records, .. } = entry else {
            continue;
        };
        for record in records {
            if let FileRecord::Data { digest, length } = record {
                let resident = meta.store().verify(digest).unwrap_or_else(|error| {
                    panic!("{context}: head file {path} references unverifiable {digest}: {error}")
                });
                assert_eq!(
                    resident, *length,
                    "{context}: {path} record length disagrees with resident bytes"
                );
            }
        }
    }

    // Every snapshot the child got a seal answer for must still be loadable
    // and fully backed — a sealed identity is a promise.
    for id in sealed_ids(root) {
        harness::assert_snapshot_fully_backed(&meta, &id, &context);
    }

    generation
}

/// One uninterrupted cycle, the STARTING POINT for the kill-delay range.
///
/// Round 0 is deliberate and cannot be repeated: it is the only round that
/// creates the workspace, and the child names its file `round-{round}.bin`,
/// so a second cycle at round 0 fails its own commit on the duplicate path.
/// A single sample is enough because [`adjust_ceiling`] corrects the range
/// from wherever this lands.
fn calibrate(root: &Path) -> Duration {
    let started = Instant::now();
    let mut child = spawn(root, 0);
    let status = child.wait().expect("calibration child is reaped");
    assert!(status.success(), "the calibration cycle must complete");
    started.elapsed()
}

/// Steer the kill-delay ceiling from where the last kill actually landed.
///
/// One up-front measurement cannot set this range correctly for the whole
/// run, for two independent reasons: the calibration cycle pays cold page
/// cache and a cold store, and the store ACCUMULATES, so later cycles are
/// slower than the one that was measured. A fixed ceiling derived from a
/// single sample is therefore wrong by an unknown factor on an unknown
/// machine — which is exactly how CI landed 29 of 30 kills in `done`.
///
/// So the range is a feedback loop rather than a constant. Landing in `done`
/// means the child finished before the kill and the window is too wide;
/// landing in `open` (or `none`, meaning it died before announcing anything)
/// means the window is too narrow. Anything between those is the interesting
/// middle and is left alone.
///
/// Multiplicative because the error is multiplicative: a machine an order of
/// magnitude faster needs an order-of-magnitude correction, and stepping by a
/// constant would take the whole run to get there.
fn adjust_ceiling(ceiling: u64, floor: u64, cap: u64, phase: &str) -> u64 {
    let adjusted = match phase {
        "done" => ceiling * 3 / 4,
        "open" | "none" => ceiling * 4 / 3 + 1,
        _ => ceiling,
    };
    adjusted.clamp(floor, cap)
}

/// The controller must haul an absurd starting ceiling back into range, which
/// is the macOS failure reproduced deterministically and without spawning
/// anything: there, the 8 ms floor set a window an order of magnitude wider
/// than the work, and every kill landed after the child was already done.
#[test]
fn the_kill_window_converges_from_an_absurd_starting_point() {
    let floor = 200_u64;
    let cap = 10_000_u64;

    // A window 50x too wide: every kill lands in `done`.
    let mut ceiling = cap;
    let mut steps = 0;
    while ceiling > floor * 4 && steps < 100 {
        ceiling = adjust_ceiling(ceiling, floor, cap, "done");
        steps += 1;
    }
    assert!(
        steps < 20,
        "a too-wide window took {steps} rounds to come back in range"
    );

    // And the other direction: a window so narrow every kill lands at `open`.
    let mut ceiling = floor;
    let mut steps = 0;
    while ceiling < cap / 4 && steps < 100 {
        ceiling = adjust_ceiling(ceiling, floor, cap, "open");
        steps += 1;
    }
    assert!(
        steps < 20,
        "a too-narrow window took {steps} rounds to open up"
    );

    // The bounds hold, and the interesting middle is never disturbed.
    assert_eq!(adjust_ceiling(floor, floor, cap, "done"), floor);
    assert_eq!(adjust_ceiling(cap, floor, cap, "open"), cap);
    assert_eq!(adjust_ceiling(500, floor, cap, "commit"), 500);
    assert_eq!(adjust_ceiling(500, floor, cap, "admit-2"), 500);
}

#[test]
fn a_kill_at_any_point_in_the_cycle_leaves_a_consistent_store() {
    let scratch = Scratch::new("kill-matrix");
    let root = scratch.path();
    let seed = seed_from_env(0x5EED_1257_C0FF_EE01);
    let mut rng = Rng::new(seed);

    // The kill delay is drawn uniformly from a window that STEERS ITSELF
    // toward the machine (see `adjust_ceiling`). The measured cycle only sets
    // where the search starts; the floor and cap keep it sane while it moves.
    //
    // The floor used to be 8 ms and set the ceiling outright, which inverted
    // its stated purpose on a fast box: a macOS runner finishes a cycle well
    // inside 8 ms, so delays were sampled across a window an order of
    // magnitude wider than the work and landed past completion nearly every
    // time. That is how CI put 29 of 30 kills in `done` and tripped the
    // spread assertion below. The floor now guards only a degenerate
    // measurement, so it is small.
    let cycle = calibrate(root);
    let micros = cycle.as_micros().min(u128::from(u64::MAX)) as u64;
    let floor = 200_u64;
    // Wide enough that a slowing store can still be outrun, bounded so a run
    // of `open` kills cannot let the window grow without limit.
    let cap = micros.saturating_mul(8).clamp(floor * 8, 2_000_000);
    let mut ceiling = (micros.saturating_mul(13) / 10).clamp(floor, cap);

    let rounds = iterations(30, 200);
    let mut distribution: BTreeMap<String, u32> = BTreeMap::new();
    let mut generation = 0_u64;

    for round in 1..=rounds {
        let _ = fs::remove_file(phase_log(root));

        let mut child = spawn(root, round);
        let delay = rng.range(0, ceiling.max(2));
        std::thread::sleep(Duration::from_micros(delay));
        // A child that already finished is reaped rather than killed; that
        // is a legitimate outcome of a randomised delay, not a failure.
        let _ = child.kill();
        let _ = child.wait();

        let phase = last_phase(root);
        *distribution.entry(phase.clone()).or_default() += 1;
        ceiling = adjust_ceiling(ceiling, floor, cap, &phase);
        generation = assert_recovered(root, seed, round, &phase, generation);
    }

    eprintln!("kill-point distribution over {rounds} rounds (seed {seed:#x}):");
    for (phase, count) in &distribution {
        eprintln!("  {phase:<12} {count}");
    }

    // Coverage must be real. A matrix that only ever killed during one phase
    // proves far less than its round count suggests, so the spread itself is
    // an assertion.
    assert!(
        distribution.len() >= 3,
        "kills landed in only {} distinct phases (seed {seed:#x}): {distribution:?}",
        distribution.len()
    );

    // The store survived every kill and is still usable for real work.
    let meta = WorkspaceStore::open(root).expect("final reopen");
    let id = meta
        .seal_snapshot("main", None)
        .expect("a final seal works");
    harness::assert_snapshot_fully_backed(&meta, &id, "after the whole matrix");
    Consistency::scan(root).assert_examined(1, "after the whole matrix");
}
