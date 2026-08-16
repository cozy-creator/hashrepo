//! Control-plane integration: a real spawned daemon, a real Unix socket, and
//! real mounts behind the eight-method protocol.

#![cfg(target_os = "linux")]

use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const FRAME_BOUND: u32 = 1024 * 1024;

/// How long a client waits for a response before calling the daemon wedged.
/// See `Daemon::connect` — an anti-hang bound, deliberately not a latency gate.
const RESPONSE_BOUND: Duration = Duration::from_secs(120);

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
struct Daemon {
    child: Child,
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
        let deadline = Instant::now() + Duration::from_secs(10);
        while !socket.exists() {
            assert!(
                Instant::now() < deadline,
                "the control socket never appeared"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        Self {
            child,
            root: root.to_path_buf(),
            socket,
        }
    }

    fn connect(&self) -> UnixStream {
        let stream = UnixStream::connect(&self.socket).expect("the control socket accepts");
        // A daemon that stops answering must fail the test, never hang it.
        //
        // The bound is generous ON PURPOSE. It exists to convert a wedged
        // daemon into a failure, not to assert how fast a mount is: several
        // of these requests mount a real filesystem, and this suite shares a
        // machine whose wall-clock for the same work swings by more than an
        // order of magnitude under load. At 10s that made the harness itself
        // flaky — `receive` timed out mid-mount and the test read it as a
        // daemon that never answered. A timeout tight enough to measure
        // performance is a performance assertion, and this is not one.
        stream
            .set_read_timeout(Some(RESPONSE_BOUND))
            .expect("the read timeout installs");
        stream
    }

    fn shutdown(mut self) {
        let pid = self.child.id();
        assert_eq!(
            Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .expect("kill runs")
                .code(),
            Some(0)
        );
        let status = self.child.wait().expect("the daemon exits");
        assert!(status.success(), "graceful shutdown exits cleanly");
        assert_no_mounts_under(&self.root);
        // Forget the child so Drop does not double-reap.
        std::mem::forget(self);
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

fn assert_no_mounts_under(root: &Path) {
    let table = std::fs::read_to_string("/proc/mounts").expect("mount table reads");
    let prefix = root.display().to_string();
    assert!(
        !table.lines().any(|line| line.contains(&prefix)),
        "no mount under {prefix} may survive: {table}"
    );
}

fn send(stream: &mut UnixStream, value: &Value) {
    let bytes = serde_json::to_vec(value).expect("request serializes");
    let length = u32::try_from(bytes.len()).expect("request fits a frame");
    stream
        .write_all(&length.to_le_bytes())
        .and_then(|()| stream.write_all(&bytes))
        .expect("request frame writes");
}

fn receive(stream: &mut UnixStream) -> Value {
    let mut length = [0_u8; 4];
    stream
        .read_exact(&mut length)
        .expect("response length reads");
    let length = u32::from_le_bytes(length);
    assert!(
        length > 0 && length <= FRAME_BOUND,
        "bounded response frame"
    );
    let mut bytes = vec![0_u8; length as usize];
    stream.read_exact(&mut bytes).expect("response body reads");
    serde_json::from_slice(&bytes).expect("response parses")
}

fn call(stream: &mut UnixStream, id: u64, method: &str, params: Value) -> Value {
    send(
        stream,
        &json!({"id": id, "method": method, "params": params}),
    );
    let response = receive(stream);
    assert_eq!(response["id"], json!(id), "response correlates: {response}");
    response
}

fn ok(stream: &mut UnixStream, id: u64, method: &str, params: Value) -> Value {
    let response = call(stream, id, method, params);
    assert!(
        response.get("error").is_none(),
        "{method} should succeed: {response}"
    );
    response["ok"].clone()
}

fn error_code(stream: &mut UnixStream, id: u64, method: &str, params: Value) -> String {
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
fn write_across_a_poll_tick(stream: &mut UnixStream, bytes: &[u8], at: usize) {
    // The daemon's accept loop polls on the same 200ms tick, so a split
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
    // ACCEPT_POLL is 200ms; overshoot so the daemon definitely sees WouldBlock.
    std::thread::sleep(Duration::from_millis(350));
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
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = ok(&mut second, 1, "status", json!({}));
        if status["mounts"]
            .as_array()
            .expect("a mounts list")
            .is_empty()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the daemon must reap a hung-up connection's mounts: {status}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        !Path::new(&mountpoint).exists(),
        "the abandoned mountpoint is gone"
    );

    daemon.shutdown();
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
