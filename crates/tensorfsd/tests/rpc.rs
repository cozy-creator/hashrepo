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
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
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
        "unimplemented",
        "push_snapshot refuses honestly until pgw#1258/th#1960"
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
