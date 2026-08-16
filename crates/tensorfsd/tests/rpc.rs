//! Control-plane integration: a real spawned daemon, a real Unix socket, and
//! real mounts behind the eight-method protocol.

#![cfg(target_os = "linux")]

use std::cell::RefCell;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use serde_json::{Value, json};
use tensorfsd::rpc::ACCEPT_POLL;

const FRAME_BOUND: u32 = 1024 * 1024;

/// How often a client that is waiting stops to ask whether the daemon is still
/// alive.
///
/// This is a polling TICK, not a bound on how long an answer may take: every
/// expiry re-asks the only question that matters — is the daemon's process
/// still running? — and keeps waiting while it is. No value of this constant
/// can turn a slow-but-healthy daemon into a failure, which is the whole
/// difference between it and the wall-clock bound it replaced.
const LIVENESS_POLL: Duration = Duration::from_millis(100);

fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        eprintln!("skipping: /dev/fuse is not available");
        return false;
    }
    if Command::new("fusermount3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: fusermount3 is not available");
        return false;
    }
    true
}

/// The daemon under test. Dropping it SIGKILLs and reaps the child, so no
/// test path leaks a daemon or, through it, a mount.
///
/// The child is the harness's health oracle, so it sits behind a `RefCell`:
/// every place that would otherwise consult a clock instead asks this process
/// whether it is still running.
struct Daemon {
    child: RefCell<Child>,
    root: PathBuf,
    socket: PathBuf,
}

impl Daemon {
    fn spawn(root: &Path) -> Self {
        let socket = root.join("control.sock");
        let child = Command::new(env!("CARGO_BIN_EXE_tensorfsd"))
            .args([
                "serve",
                "--store",
                root.join("store").to_str().expect("utf-8 path"),
                "--socket",
                socket.to_str().expect("utf-8 path"),
                "--mounts",
                root.join("mnt").to_str().expect("utf-8 path"),
            ])
            .spawn()
            .expect("the daemon binary spawns");
        let daemon = Self {
            child: RefCell::new(child),
            root: root.to_path_buf(),
            socket,
        };
        // A daemon that is still starting is making progress, however loaded
        // the box is; a daemon that has EXITED will never create the socket,
        // and that is observable directly rather than inferred from a clock.
        while !daemon.socket.exists() {
            daemon.assert_alive("the control socket to appear");
            std::thread::sleep(LIVENESS_POLL);
        }
        daemon
    }

    /// The child's exit status, or `None` while it is still running.
    fn exited(&self) -> Option<ExitStatus> {
        self.child
            .borrow_mut()
            .try_wait()
            .expect("the daemon's status reads")
    }

    /// Fail the test if the daemon has died, naming the exit. A daemon that is
    /// alive is working — the harness has no business guessing how long that
    /// is allowed to take.
    fn assert_alive(&self, awaited: &str) {
        if let Some(status) = self.exited() {
            panic!("the daemon exited ({status}) while the test waited for {awaited}");
        }
    }

    fn connect(&self) -> Client<'_> {
        let stream = UnixStream::connect(&self.socket).expect("the control socket accepts");
        // The read timeout is this client's polling tick, NOT a verdict: when
        // it expires, `Client::read` asks whether the daemon is alive and, if
        // it is, waits again.
        //
        // What that buys is a gate on the real condition. A daemon that DIED
        // fails the test at the next tick, naming the exit — faster and
        // clearer than any bound. A daemon that is merely slow is never
        // accused of anything, which is the failure the bound this replaced
        // actually produced: at 10s `receive` expired mid-mount under load and
        // the test read it as a daemon that never answered. A daemon that is
        // alive but has stopped answering hangs, deliberately — that is a
        // livelock, and the CI job timeout reports it as the hang it is
        // instead of dressing it up as an assertion about milliseconds.
        stream
            .set_read_timeout(Some(LIVENESS_POLL))
            .expect("the read timeout installs");
        Client {
            stream,
            daemon: self,
        }
    }

    fn shutdown(self) {
        let pid = self.child.borrow().id();
        assert_eq!(
            Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .expect("kill runs")
                .code(),
            Some(0)
        );
        let status = self.child.borrow_mut().wait().expect("the daemon exits");
        assert!(status.success(), "graceful shutdown exits cleanly");
        assert_no_mounts_under(&self.root);
        // Forget the child so Drop does not double-reap.
        std::mem::forget(self);
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let child = self.child.get_mut();
        let _ = child.kill();
        let _ = child.wait();
        // A SIGKILLed daemon cannot unmount; sweep so nothing leaks.
        if let Ok(entries) = std::fs::read_dir(self.root.join("mnt")) {
            for entry in entries.flatten() {
                let _ = Command::new("fusermount3")
                    .arg("-u")
                    .arg(entry.path())
                    .output();
            }
        }
    }
}

/// A control connection together with the daemon that owns it.
///
/// The pairing is the point: when a read produces nothing, the honest question
/// is not "how long has this taken?" but "is the daemon still there?", and only
/// a client that can see the child process can ask it.
struct Client<'a> {
    stream: UnixStream,
    daemon: &'a Daemon,
}

impl Client<'_> {
    /// Report a failed read against what the daemon is actually doing.
    ///
    /// The read has already failed, so the test fails either way; this only
    /// decides which message it carries. `try_wait` can lose a moment's race
    /// with the child's own exit path — the kernel closes the socket before
    /// the status is reapable — so an unresolved status gets one tick of
    /// grace. That changes the wording, never the verdict.
    fn diagnose(&self, error: &io::Error, context: &str) -> ! {
        if self.daemon.exited().is_none() {
            std::thread::sleep(LIVENESS_POLL);
        }
        match self.daemon.exited() {
            Some(status) => panic!("the daemon exited ({status}) while {context}: {error}"),
            None => panic!("the daemon is still alive, but {context} failed: {error}"),
        }
    }
}

impl Read for Client<'_> {
    /// An expired read timeout is not an answer, so it is not treated as one:
    /// it is a chance to check the daemon's pulse and go back to waiting. A
    /// slow-but-healthy daemon is therefore waited on indefinitely, and a dead
    /// one fails the test at the next tick.
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.stream.read(buffer) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    self.daemon.assert_alive("a response");
                }
                outcome => return outcome,
            }
        }
    }
}

impl Write for Client<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.stream.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

/// Whether `mountpoint` is in the kernel's mount table right now. Release
/// unmounts before it answers, so this is an exact observation, not a poll.
fn is_mounted(mountpoint: &str) -> bool {
    std::fs::read_to_string("/proc/mounts")
        .expect("mount table reads")
        .lines()
        .any(|line| line.split(' ').nth(1) == Some(mountpoint))
}

fn assert_no_mounts_under(root: &Path) {
    let table = std::fs::read_to_string("/proc/mounts").expect("mount table reads");
    let prefix = root.display().to_string();
    assert!(
        !table.lines().any(|line| line.contains(&prefix)),
        "no mount under {prefix} may survive: {table}"
    );
}

fn send(stream: &mut Client, value: &Value) {
    let bytes = serde_json::to_vec(value).expect("request serializes");
    let length = u32::try_from(bytes.len()).expect("request fits a frame");
    // A dead daemon breaks the pipe under the writer as readily as it starves
    // the reader, so the write side reports the same diagnosis rather than a
    // bare EPIPE.
    if let Err(error) = stream
        .write_all(&length.to_le_bytes())
        .and_then(|()| stream.write_all(&bytes))
    {
        stream.diagnose(&error, "writing a request frame");
    }
}

fn receive(stream: &mut Client) -> Value {
    let mut length = [0_u8; 4];
    if let Err(error) = stream.read_exact(&mut length) {
        stream.diagnose(&error, "waiting for a response length");
    }
    let length = u32::from_le_bytes(length);
    assert!(
        length > 0 && length <= FRAME_BOUND,
        "bounded response frame"
    );
    let mut bytes = vec![0_u8; length as usize];
    if let Err(error) = stream.read_exact(&mut bytes) {
        stream.diagnose(&error, "waiting for a response body");
    }
    serde_json::from_slice(&bytes).expect("response parses")
}

fn call(stream: &mut Client, id: u64, method: &str, params: Value) -> Value {
    send(
        stream,
        &json!({"id": id, "method": method, "params": params}),
    );
    let response = receive(stream);
    assert_eq!(response["id"], json!(id), "response correlates: {response}");
    response
}

fn ok(stream: &mut Client, id: u64, method: &str, params: Value) -> Value {
    let response = call(stream, id, method, params);
    assert!(
        response.get("error").is_none(),
        "{method} should succeed: {response}"
    );
    response["ok"].clone()
}

fn error_code(stream: &mut Client, id: u64, method: &str, params: Value) -> String {
    let response = call(stream, id, method, params);
    response["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("{method} should refuse: {response}"))
        .to_owned()
}

#[test]
fn the_control_socket_serves_the_full_workspace_lifecycle() {
    if !fuse_available() {
        return;
    }
    let root = tempdir("rpc-lifecycle");
    let daemon = Daemon::spawn(&root);

    let mode = std::fs::metadata(&daemon.socket)
        .expect("socket metadata reads")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "the control socket is mode 0600");

    let mut stream = daemon.connect();
    let hello = ok(&mut stream, 1, "hello", json!({}));
    assert_eq!(hello["protocol"], json!(1));
    assert_eq!(hello["server"], json!("tensorfsd"));

    let status = ok(&mut stream, 2, "status", json!({}));
    assert_eq!(status["mounts"], json!([]));

    let opened = ok(
        &mut stream,
        3,
        "create_workspace",
        json!({"workspace": "ws"}),
    );
    let workspace_lease = opened["lease"].as_u64().expect("a lease id");
    let workspace_mount = opened["mountpoint"].as_str().expect("a mountpoint");

    let payload = b"bytes written through the daemon's own mount".to_vec();
    let file = Path::new(workspace_mount).join("model.bin");
    std::fs::write(&file, &payload).expect("the workspace mount accepts writes");
    let handle = std::fs::File::open(&file).expect("the file reopens");
    handle.sync_all().expect("fsync composes and commits");
    drop(handle);

    let sealed = ok(
        &mut stream,
        4,
        "commit_workspace",
        json!({"workspace": "ws"}),
    );
    let snapshot = sealed["snapshot"]
        .as_str()
        .expect("a snapshot id")
        .to_owned();
    assert_eq!(snapshot.len(), 64);

    let viewed = ok(
        &mut stream,
        5,
        "open_snapshot",
        json!({"snapshot": snapshot}),
    );
    let snapshot_lease = viewed["lease"].as_u64().expect("a lease id");
    let snapshot_mount = viewed["mountpoint"].as_str().expect("a mountpoint");
    assert_eq!(
        std::fs::read(Path::new(snapshot_mount).join("model.bin")).expect("snapshot serves"),
        payload,
        "the sealed snapshot serves the committed bytes byte-exactly"
    );

    let status = ok(&mut stream, 6, "status", json!({}));
    assert_eq!(status["mounts"].as_array().expect("a mounts list").len(), 2);

    assert_eq!(
        error_code(
            &mut stream,
            7,
            "delete_workspace",
            json!({"workspace": "ws"})
        ),
        "workspace-mounted",
        "a mounted workspace refuses deletion"
    );
    assert_eq!(
        error_code(
            &mut stream,
            8,
            "push_snapshot",
            json!({"snapshot": snapshot})
        ),
        "unconfigured",
        "push_snapshot refuses honestly when the daemon has no sync target"
    );
    assert_eq!(
        error_code(&mut stream, 9, "mount_everything", json!({})),
        "unknown-method"
    );
    assert_eq!(
        error_code(&mut stream, 10, "release", json!({"lease": 999})),
        "unknown-lease"
    );

    ok(&mut stream, 11, "release", json!({"lease": snapshot_lease}));
    assert!(
        !Path::new(snapshot_mount).exists(),
        "releasing the lease unmounts and removes the mountpoint"
    );
    ok(
        &mut stream,
        12,
        "release",
        json!({"lease": workspace_lease}),
    );
    ok(
        &mut stream,
        13,
        "delete_workspace",
        json!({"workspace": "ws"}),
    );
    assert_eq!(
        error_code(
            &mut stream,
            14,
            "commit_workspace",
            json!({"workspace": "ws"})
        ),
        "unknown-workspace"
    );

    drop(stream);
    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_oversized_frame_is_refused_before_allocation() {
    if !fuse_available() {
        return;
    }
    let root = tempdir("rpc-oversized");
    let daemon = Daemon::spawn(&root);

    let mut stream = daemon.connect();
    stream
        .write_all(&(FRAME_BOUND + 1).to_le_bytes())
        .expect("the length prefix writes");
    let refusal = receive(&mut stream);
    assert_eq!(refusal["error"]["code"], json!("frame-too-large"));
    let mut rest = Vec::new();
    assert_eq!(
        stream.read_to_end(&mut rest).expect("the daemon hangs up"),
        0,
        "a framing violation ends the connection"
    );

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// Splits `bytes` at `at`, with a pause longer than the daemon's poll window
/// between the halves, so the frame is genuinely in flight across an idle tick.
///
/// This is the condition `fill_frame_bytes` exists for and the one no test
/// reached: every other request in this file is one `write_all`, which the
/// kernel delivers in a single `read`, so the retry path never runs.
fn write_across_a_poll_tick(stream: &mut Client, bytes: &[u8], at: usize) {
    // The daemon's accept loop polls on the same ACCEPT_POLL tick, so a split
    // written immediately after connect can land entirely BEFORE the
    // connection is accepted — both halves then arrive in one read and the
    // test silently proves nothing. One completed round trip guarantees the
    // per-connection thread exists and is blocked in its read loop.
    let warmup = ok(stream, 0, "hello", json!({}));
    assert!(warmup.is_object(), "the warm-up round trip answers");

    stream
        .write_all(&bytes[..at])
        .and_then(|()| stream.flush())
        .expect("the first half writes");
    // This sleep FORCES the condition under test; it does not wait for one to
    // be met. The daemon's read loop must see a WouldBlock between the halves,
    // so the pause has to outlast its poll window — derived from the daemon's
    // own ACCEPT_POLL so the two can never drift apart, and overshot so the
    // tick definitely expires. A longer pause only makes the condition surer;
    // nothing here fails because something took too long.
    std::thread::sleep(ACCEPT_POLL + ACCEPT_POLL / 2);
    stream
        .write_all(&bytes[at..])
        .and_then(|()| stream.flush())
        .expect("the second half writes");
}

fn framed(value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).expect("request serializes");
    let length = u32::try_from(body.len()).expect("request fits a frame");
    let mut frame = length.to_le_bytes().to_vec();
    frame.extend_from_slice(&body);
    frame
}

/// A request body that arrives in two reads, straddling the poll window, must
/// reassemble byte-exactly.
///
/// This is the regression that shipped once already: `read_exact` cannot be
/// retried after `WouldBlock` — it leaves the buffer unspecified and the
/// stream advanced — so the body loop tracks a fill offset and reads into
/// `buffer[filled..]`. Reading into `buffer` from the start instead silently
/// reassembles the frame out of order, and the corruption is invisible until
/// the JSON fails to parse. Nothing forced that path before this test.
///
/// No mount, so no FUSE dependency: this exercises the socket and the framing
/// only.
#[test]
fn a_request_body_split_across_the_poll_window_reassembles_exactly() {
    let root = tempdir("rpc-split-body");
    let daemon = Daemon::spawn(&root);
    let mut stream = daemon.connect();

    let frame = framed(&json!({"id": 7, "method": "hello", "params": {}}));
    // Split inside the BODY: the length prefix lands whole in the first write.
    assert!(frame.len() > 8, "the body must be long enough to split");
    write_across_a_poll_tick(&mut stream, &frame, 4 + (frame.len() - 4) / 2);

    let response = receive(&mut stream);
    assert_eq!(response["id"], json!(7), "response correlates: {response}");
    assert!(
        response.get("error").is_none(),
        "a body that merely arrived in two pieces is a VALID request, not a \
         malformed one — the framing reassembled it wrong: {response}"
    );

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// The same, one layer earlier: a length prefix that arrives in two reads.
///
/// The idle refusal is only correct while NOTHING of the frame has arrived —
/// hence `!frame_started && filled == 0`. Dropping the `filled == 0` half
/// makes a half-read length prefix report "idle", and the caller loops and
/// reads the prefix's remaining bytes as if they began a new frame, so a
/// perfectly ordinary request is misparsed as a garbage length.
#[test]
fn a_length_prefix_split_across_the_poll_window_is_not_mistaken_for_idle() {
    let root = tempdir("rpc-split-prefix");
    let daemon = Daemon::spawn(&root);
    let mut stream = daemon.connect();

    let frame = framed(&json!({"id": 11, "method": "hello", "params": {}}));
    // Split INSIDE the 4-byte length prefix.
    write_across_a_poll_tick(&mut stream, &frame, 2);

    let response = receive(&mut stream);
    assert_eq!(response["id"], json!(11), "response correlates: {response}");
    assert!(
        response.get("error").is_none(),
        "a prefix that merely arrived in two pieces is not an idle tick: {response}"
    );

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// A frame of exactly the bound is legal — the refusal says `1..=BOUND`.
///
/// `an_oversized_frame_is_refused_before_allocation` only ever sends
/// `BOUND + 1`, so an off-by-one that turned the ceiling into `>=` would
/// refuse the largest legal frame with nothing going red.
#[test]
fn a_frame_of_exactly_the_bound_is_accepted_not_refused() {
    let root = tempdir("rpc-exact-bound");
    let daemon = Daemon::spawn(&root);
    let mut stream = daemon.connect();

    // Pad an unknown method out to exactly the bound. Unknown-method is a
    // dispatch answer, which proves the frame was read whole and parsed —
    // exactly what is under test — while keeping the payload trivial.
    let skeleton = json!({"id": 13, "method": "nosuchmethod", "params": {"pad": ""}});
    let padding = FRAME_BOUND as usize - serde_json::to_vec(&skeleton).unwrap().len();
    let request = json!({
        "id": 13, "method": "nosuchmethod", "params": {"pad": "a".repeat(padding)}
    });
    let frame = framed(&request);
    assert_eq!(
        frame.len() - 4,
        FRAME_BOUND as usize,
        "the request body must be exactly the bound"
    );
    stream.write_all(&frame).expect("the frame writes");

    let response = receive(&mut stream);
    assert_eq!(
        response["error"]["code"],
        json!("unknown-method"),
        "the largest LEGAL frame must reach dispatch, not be refused for its \
         size: {}",
        response["error"]
    );

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// A zero-length frame is a framing violation, and must be refused as one.
///
/// Without the `length == 0` clause it falls through to a zero-byte body that
/// fails JSON parsing instead, downgrading a typed protocol refusal into a
/// generic `malformed-frame`.
#[test]
fn a_zero_length_frame_is_refused_as_a_framing_violation() {
    let root = tempdir("rpc-zero-frame");
    let daemon = Daemon::spawn(&root);
    let mut stream = daemon.connect();

    stream
        .write_all(&0_u32.to_le_bytes())
        .expect("the length prefix writes");

    let refusal = receive(&mut stream);
    assert_eq!(
        refusal["error"]["code"],
        json!("frame-too-large"),
        "an empty frame is refused by the framing layer, not by the JSON \
         parser downstream: {refusal}"
    );

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_closed_connection_releases_every_mount_it_opened() {
    if !fuse_available() {
        return;
    }
    let root = tempdir("rpc-hangup");
    let daemon = Daemon::spawn(&root);

    let mut first = daemon.connect();
    let opened = ok(
        &mut first,
        1,
        "create_workspace",
        json!({"workspace": "gone"}),
    );
    let mountpoint = opened["mountpoint"]
        .as_str()
        .expect("a mountpoint")
        .to_owned();
    assert!(Path::new(&mountpoint).exists());
    drop(first);

    let mut second = daemon.connect();
    // Both conditions belong in the SAME poll. The teardown detaches entries
    // under the state mutex and drops them after releasing it — deliberately,
    // so a slow unmount cannot hold every other connection hostage — so an
    // empty mount table does not yet imply an unmounted mountpoint. Asserting
    // the mountpoint immediately after the table went empty raced that
    // ordering and failed under load.
    //
    // The loop has no deadline, and needs none: every `status` round trip is
    // itself gated on the daemon being alive, so a daemon that dies fails here
    // immediately and says so. One that lives but never reaps hangs — which is
    // the honest report of "never reaped", where a clock would instead accuse
    // a merely slow unmount of leaking.
    loop {
        let status = ok(&mut second, 1, "status", json!({}));
        let reaped = status["mounts"]
            .as_array()
            .expect("a mounts list")
            .is_empty()
            && !Path::new(&mountpoint).exists();
        if reaped {
            break;
        }
        std::thread::sleep(LIVENESS_POLL);
    }

    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// Leases are connection-bound: only the connection that opened a mount may
/// release it.
///
/// The lifecycle test above releases lease 999, which no connection holds and
/// no daemon ever issued. That refusal comes from `detach` missing the entry,
/// so it is produced with or without the ownership check and proves nothing
/// about ownership. The condition that separates the two is a lease that DOES
/// exist and belongs to somebody else, which needs a real mount and a second
/// live connection.
///
/// Nothing here is timed. The owner's `create_workspace` round trip has
/// already returned before the intruder is even connected, so the lease is in
/// the daemon's table when the foreign release arrives; the refusal cannot be
/// a race with an unfinished mount.
#[test]
fn one_connection_cannot_release_another_connections_lease() {
    if !fuse_available() {
        return;
    }
    let root = tempdir("rpc-foreign-release");
    let daemon = Daemon::spawn(&root);

    let mut owner = daemon.connect();
    let opened = ok(
        &mut owner,
        1,
        "create_workspace",
        json!({"workspace": "held"}),
    );
    let lease = opened["lease"].as_u64().expect("a lease id");
    let mountpoint = opened["mountpoint"]
        .as_str()
        .expect("a mountpoint")
        .to_owned();
    assert!(is_mounted(&mountpoint), "the owner's mount is live");

    let mut intruder = daemon.connect();
    // Prove the intruder's connection is up and dispatching before it tries
    // the foreign lease, so a refusal can never be an artefact of a
    // half-established connection.
    assert_eq!(
        ok(&mut intruder, 1, "status", json!({}))["mounts"]
            .as_array()
            .expect("a mounts list")
            .len(),
        1,
        "the intruder sees the owner's mount"
    );
    assert_eq!(
        error_code(&mut intruder, 2, "release", json!({"lease": lease})),
        "unknown-lease",
        "a foreign connection must not be able to release this lease"
    );

    // The refusal must also be a no-op: refusing after tearing the mount down
    // would be the same bug with a nicer error code.
    assert!(
        is_mounted(&mountpoint),
        "a refused release must leave the mount standing"
    );
    assert_eq!(
        ok(&mut intruder, 3, "status", json!({}))["mounts"]
            .as_array()
            .expect("a mounts list")
            .len(),
        1,
        "the lease survives the foreign release attempt"
    );

    // And the real owner is unaffected by the attempt.
    ok(&mut owner, 2, "release", json!({"lease": lease}));
    assert!(
        !is_mounted(&mountpoint),
        "the owner's own release still unmounts"
    );
    assert!(
        !Path::new(&mountpoint).exists(),
        "the owner's own release still removes the mountpoint"
    );

    drop(intruder);
    drop(owner);
    daemon.shutdown();
    let _ = std::fs::remove_dir_all(&root);
}

/// The harness's own red: a daemon that dies with a request outstanding must
/// fail the test at once, naming the exit.
///
/// This is the guard on the gate that replaced a wall-clock bound. The bound
/// was a bad instrument in both directions — it accused a slow-but-healthy
/// daemon of never answering, and it made a dead one wait out the full bound
/// before saying anything. Liveness answers the real question, but only while
/// this stays true: if the diagnosis ever degrades to a bare I/O error, the
/// suite silently loses its ability to tell "dead" from "busy", and nothing
/// else in this file would notice.
///
/// No mount, so no FUSE dependency.
#[test]
fn a_dead_daemon_fails_the_read_at_once_and_names_the_exit() {
    let root = tempdir("rpc-dead-daemon");
    let daemon = Daemon::spawn(&root);
    let mut stream = daemon.connect();
    // One completed round trip, so the connection is genuinely established and
    // the failure below cannot be an artefact of a half-open socket.
    ok(&mut stream, 1, "hello", json!({}));

    let pid = daemon.child.borrow().id();
    assert_eq!(
        Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .expect("kill runs")
            .code(),
        Some(0)
    );

    // Either half of the round trip may be the one that notices: a dead peer
    // breaks the pipe under the write, and ends the read at EOF. Both must
    // arrive at the same diagnosis, so both are under the guard.
    let failure = std::panic::catch_unwind(AssertUnwindSafe(|| {
        send(
            &mut stream,
            &json!({"id": 2, "method": "status", "params": {}}),
        );
        receive(&mut stream)
    }))
    .expect_err("a response that can never arrive must fail the test");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| String::from("<non-string panic>"));
    assert!(
        message.contains("the daemon exited"),
        "the failure must name the daemon's death, not merely a failed read: {message}"
    );
    assert!(
        message.contains("signal: 9"),
        "the failure must name the exit itself, so the reader never has to \
         guess whether the daemon died or was slow: {message}"
    );

    drop(stream);
    let _ = std::fs::remove_dir_all(&root);
}

fn tempdir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tensorfsd-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is sane")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("the test root creates");
    root
}
