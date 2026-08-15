//! Family 6 — killed mid-transfer, restarted, resumed, repeatedly.
//!
//! Push and pull are long operations against a remote. In production they
//! are interrupted constantly: the pod is preempted, the container is OOM
//! killed, the operator ctrl-Cs, the node reboots. What must never happen is
//! a local store that ends up subtly wrong, or a resumed run that pays again
//! for work it already completed.
//!
//! These are REAL `SIGKILL`s of a real child process running the real
//! engine. That needs a hub a separate process can reach, which the
//! in-memory fault hub cannot be, so the transfers run against
//! [`harness::DirTransport`] — a directory-backed transport implementing the
//! same wire rules (session-scoped staging, checksum-bound grants, manifest
//! admitted strictly before the head moves).
//!
//! After every kill, and after convergence, the assertions are:
//!
//!  * the local store is intact under an independent on-disk rehash;
//!  * no snapshot is adopted whose objects are not all resident and verified;
//!  * the operation eventually converges;
//!  * once converged, repeating it transfers nothing.
//!
//! Default run: 12 interruptions per direction. `TENSORFS_HEAVY=1` runs 60.

#![cfg(any(unix, windows))]

mod harness;

use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use harness::{
    Consistency, DirTransport, Rng, Scratch, assert_snapshot_fully_backed, data_digests,
    iterations, reconstruct, sealed_workspace, seed_from_env,
};
use tensorfs_core::sync::{PushOptions, pull_snapshot, push_snapshot};
use tensorfs_core::tfm1::SnapshotId;
use tensorfs_core::workspace::WorkspaceStore;

const ROLE: &str = "TENSORFS_XFER_ROLE";
const STORE: &str = "TENSORFS_XFER_STORE";
const HUB: &str = "TENSORFS_XFER_HUB";
const SNAPSHOT: &str = "TENSORFS_XFER_SNAPSHOT";
const DONE: &str = "TENSORFS_XFER_DONE";

/// A payload with enough distinct objects that a kill can land between them.
fn corpus() -> Vec<(&'static str, Vec<Vec<u8>>)> {
    let block = |seed: u8, size: usize| -> Vec<u8> {
        (0..size)
            .map(|position| ((position as u32).wrapping_mul(31).wrapping_add(seed as u32) & 0xFF) as u8)
            .collect()
    };
    vec![
        (
            "model.bin",
            vec![
                block(1, 96 * 1024),
                block(2, 128 * 1024),
                block(3, 64 * 1024),
                block(4, 112 * 1024),
            ],
        ),
        (
            "weights/a.bin",
            vec![block(5, 80 * 1024), block(6, 96 * 1024), block(7, 48 * 1024)],
        ),
        (
            "weights/b.bin",
            vec![block(8, 72 * 1024), block(9, 88 * 1024)],
        ),
    ]
}

/// Child dispatch: one push or one pull, then a durable done marker.
#[test]
fn transfer_child_role() {
    let Ok(role) = std::env::var(ROLE) else {
        return;
    };
    let store = std::env::var(STORE).expect("the parent supplies a store");
    let hub = std::env::var(HUB).expect("the parent supplies a hub");
    let id = SnapshotId::parse_hex(&std::env::var(SNAPSHOT).expect("the parent supplies an id"))
        .expect("snapshot id parses");
    let done = std::env::var(DONE).expect("the parent supplies a done path");

    let meta = WorkspaceStore::open(&store).expect("child opens the store");
    let transport = DirTransport::new(&hub);

    match role.as_str() {
        "push" => {
            let expected = transport_head(&transport);
            push_snapshot(&meta, &transport, &id, expected.as_ref(), PushOptions::default())
                .expect("push completes");
        }
        "pull" => {
            pull_snapshot(&meta, &transport, &id).expect("pull completes");
        }
        other => panic!("unknown role {other}"),
    }
    fs::write(&done, b"done").expect("done marker writes");
}

fn transport_head(transport: &DirTransport) -> Option<SnapshotId> {
    use tensorfs_core::sync::SyncTransport as _;
    transport.head().expect("head reads")
}

fn spawn(role: &str, store: &Path, hub: &Path, id: &SnapshotId, done: &Path) -> Child {
    Command::new(std::env::current_exe().expect("test binary path"))
        .args(["transfer_child_role", "--exact", "--nocapture"])
        .env(ROLE, role)
        .env(STORE, store)
        .env(HUB, hub)
        .env(SNAPSHOT, id.to_string())
        .env(DONE, done)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("child spawns")
}

/// Times one uninterrupted run so the kill delays actually span it.
fn calibrate(role: &str, store: &Path, hub: &Path, id: &SnapshotId, done: &Path) -> Duration {
    let started = Instant::now();
    let mut child = spawn(role, store, hub, id, done);
    assert!(
        child.wait().expect("calibration child reaped").success(),
        "the calibration {role} must complete"
    );
    let elapsed = started.elapsed();
    let _ = fs::remove_file(done);
    elapsed
}

/// A push interrupted at randomised points, restarted until it converges.
#[test]
fn a_push_killed_repeatedly_still_converges_and_never_re_uploads() {
    let source = Scratch::new("push-kill-src");
    let hub = Scratch::new("push-kill-hub");
    let (meta, id) = sealed_workspace(source.path(), "publisher", &corpus());
    let done = source.path().join("done.marker");

    // Calibrate against a throwaway hub so the real one starts empty.
    let warmup = Scratch::new("push-kill-warm");
    let cycle = calibrate("push", source.path(), warmup.path(), &id, &done);

    let seed = seed_from_env(0x5EED_1257_C0FF_EE06);
    let mut rng = Rng::new(seed);
    let ceiling = ((cycle.max(Duration::from_millis(10)).as_micros() as u64) * 12 / 10).max(2);
    let attempts = iterations(12, 60);

    // Every attempt is a real interruption: the done marker is cleared each
    // round rather than breaking out on the first success, so the hub and
    // the resume path are hammered `attempts` times instead of once.
    let mut interrupted = 0_u32;
    for attempt in 1..=attempts {
        let _ = fs::remove_file(&done);
        let mut child = spawn("push", source.path(), hub.path(), &id, &done);
        std::thread::sleep(Duration::from_micros(rng.range(0, ceiling)));
        let _ = child.kill();
        let _ = child.wait();

        if !done.exists() {
            interrupted += 1;
        }

        // The publisher's own store must be untouched by an interrupted
        // push: push reads locally and writes remotely.
        Consistency::scan(source.path())
            .assert_intact(&format!("seed {seed:#x}, push attempt {attempt}"));
        assert_snapshot_fully_backed(&meta, &id, "publisher after an interrupted push");
    }

    // Whatever the kills did, a final unhindered run must converge.
    let _ = fs::remove_file(&done);
    let mut child = spawn("push", source.path(), hub.path(), &id, &done);
    assert!(
        child.wait().expect("final push reaped").success(),
        "the push never converged (seed {seed:#x})"
    );
    eprintln!("push: {interrupted}/{attempts} attempts were genuinely interrupted");
    assert!(
        interrupted > 0,
        "no push attempt was actually interrupted, so resume was never tested (seed {seed:#x})"
    );

    let transport = DirTransport::new(hub.path());
    assert_eq!(
        transport_head(&transport),
        Some(id),
        "the hub head did not reach the pushed snapshot (seed {seed:#x})"
    );

    // Converged means converged: re-pushing moves nothing.
    let report = push_snapshot(&meta, &transport, &id, Some(&id), PushOptions::default())
        .expect("a settled push succeeds");
    assert_eq!(
        report.uploaded_objects, 0,
        "a settled push re-uploaded objects the hub already holds (seed {seed:#x})"
    );
}

/// A pull interrupted at randomised points, restarted until it converges,
/// then checked byte-for-byte against the publisher's originals.
#[test]
fn a_pull_killed_repeatedly_converges_byte_exactly_without_re_downloading() {
    let source = Scratch::new("pull-kill-src");
    let hub = Scratch::new("pull-kill-hub");
    let (publisher, id) = sealed_workspace(source.path(), "publisher", &corpus());

    // Publish once, cleanly: this test attacks the pull, not the push.
    let transport = DirTransport::new(hub.path());
    push_snapshot(&publisher, &transport, &id, None, PushOptions::default()).expect("push lands");
    let originals: Vec<(String, Vec<u8>)> = corpus()
        .iter()
        .map(|(path, _)| ((*path).to_owned(), reconstruct(&publisher, &id, path)))
        .collect();
    let expected_objects = data_digests(&publisher, &id).len();

    let sink = Scratch::new("pull-kill-sink");
    let done = sink.path().join("done.marker");
    let warm_sink = Scratch::new("pull-kill-warm");
    let cycle = calibrate("pull", warm_sink.path(), hub.path(), &id, &done);

    let seed = seed_from_env(0x5EED_1257_C0FF_EE07);
    let mut rng = Rng::new(seed);
    let ceiling = ((cycle.max(Duration::from_millis(10)).as_micros() as u64) * 12 / 10).max(2);
    let attempts = iterations(12, 60);

    let mut interrupted = 0_u32;
    for attempt in 1..=attempts {
        let _ = fs::remove_file(&done);
        let mut child = spawn("pull", sink.path(), hub.path(), &id, &done);
        std::thread::sleep(Duration::from_micros(rng.range(0, ceiling)));
        let _ = child.kill();
        let _ = child.wait();

        if !done.exists() {
            interrupted += 1;
        }

        let context = format!("seed {seed:#x}, pull attempt {attempt}");
        // The partially-filled store must never be internally wrong.
        Consistency::scan(sink.path()).assert_intact(&context);

        // If a snapshot has been adopted at all, it must be COMPLETE: the
        // engine adopts only after every object is admitted and verified, so
        // a half-pulled tree may never become a loadable snapshot.
        let meta = WorkspaceStore::open(sink.path()).expect("sink reopens");
        if meta.load_snapshot(&id).is_ok() {
            assert_snapshot_fully_backed(&meta, &id, &context);
        }
    }

    let _ = fs::remove_file(&done);
    let mut child = spawn("pull", sink.path(), hub.path(), &id, &done);
    assert!(
        child.wait().expect("final pull reaped").success(),
        "the pull never converged (seed {seed:#x})"
    );
    eprintln!("pull: {interrupted}/{attempts} attempts were genuinely interrupted");
    assert!(
        interrupted > 0,
        "no pull attempt was actually interrupted, so resume was never tested (seed {seed:#x})"
    );

    let meta = WorkspaceStore::open(sink.path()).expect("sink reopens");
    assert_snapshot_fully_backed(&meta, &id, "after a repeatedly-killed pull");
    for (path, bytes) in &originals {
        assert_eq!(
            &reconstruct(&meta, &id, path),
            bytes,
            "{path} is not byte-exact after {attempts} interruptions (seed {seed:#x})"
        );
    }
    assert_eq!(
        data_digests(&meta, &id).len(),
        expected_objects,
        "the pulled snapshot names a different object set"
    );

    // Nothing already verified is paid for twice.
    let report = pull_snapshot(&meta, &transport, &id).expect("a settled pull succeeds");
    assert_eq!(
        report.fetched_objects, 0,
        "a settled pull re-downloaded objects it already held (seed {seed:#x})"
    );
    assert_eq!(report.skipped_local_resident as usize, expected_objects);
    Consistency::scan(sink.path()).assert_intact("after a repeatedly-killed pull");
}
