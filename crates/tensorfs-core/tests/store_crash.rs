//! Real-process crash coverage: a writer killed mid-admission must leave the
//! store publishable-object-free, its abandoned temp must survive the grace
//! and then be reclaimed, and a live writer in another process must never be
//! collected. The child role re-invokes this test binary with an env flag.

#![cfg(any(unix, windows))]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use tensorfs_core::store::{ObjectStore, TempCollection};

const ROLE_ENV: &str = "TENSORFS_CRASH_ROLE";
const ROOT_ENV: &str = "TENSORFS_CRASH_ROOT";

/// Child dispatch: a no-op pass unless the parent set the role env.
#[test]
fn crash_child_role() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    let root = std::env::var(ROOT_ENV).expect("the parent supplies the store root");
    assert_eq!(role, "hold");

    let store = ObjectStore::open(&root).expect("child opens the shared store");
    let mut writer = store.writer().expect("child starts an admission");
    writer
        .write_all(b"bytes a crashed writer will abandon")
        .expect("child writes");
    fs::write(Path::new(&root).join("child-ready"), b"ready").expect("child marks readiness");

    // Hold the lease mid-admission until the parent kills this process.
    std::thread::sleep(Duration::from_secs(3600));
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-crash-{name}-{}", std::process::id()));
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

fn spawn_holding_child(root: &Path) -> Child {
    let mut child = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["crash_child_role", "--exact", "--nocapture"])
        .env(ROLE_ENV, "hold")
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

fn object_files(root: &Path) -> usize {
    fn walk(dir: &Path, count: &mut usize) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, count);
            } else {
                *count += 1;
            }
        }
    }
    let mut count = 0;
    walk(&root.join("objects"), &mut count);
    count
}

fn temp_files(root: &Path) -> usize {
    fs::read_dir(root.join("tmp"))
        .map(|entries| entries.filter_map(Result::ok).count())
        .unwrap_or(0)
}

#[test]
fn a_sigkilled_admission_leaves_no_object_and_a_grace_protected_temp() {
    let root = TempRoot::new("kill");
    let mut child = spawn_holding_child(&root.0);

    child.kill().expect("SIGKILL the mid-admission writer");
    child.wait().expect("reap the killed writer");

    // The store never saw a partial object: install is one hard link of a
    // complete verified temp, and that link never happened.
    assert_eq!(object_files(&root.0), 0);
    assert_eq!(
        temp_files(&root.0),
        1,
        "the abandoned temp survives the kill"
    );

    let store = ObjectStore::open(&root.0).unwrap();

    // Inside the grace the fresh abandon is examined but protected.
    let early = store
        .collect_abandoned_temps(Duration::from_secs(3600))
        .unwrap();
    assert_eq!(
        early,
        TempCollection {
            examined: 1,
            deleted: 0,
            bytes_deleted: 0
        }
    );

    std::thread::sleep(Duration::from_millis(200));
    let late = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        late,
        TempCollection {
            examined: 1,
            deleted: 1,
            bytes_deleted: 35
        }
    );
    assert_eq!(temp_files(&root.0), 0);
}

#[test]
fn a_live_writer_in_another_process_is_never_collected() {
    let root = TempRoot::new("live");
    let mut child = spawn_holding_child(&root.0);

    let store = ObjectStore::open(&root.0).unwrap();
    std::thread::sleep(Duration::from_millis(200));

    // Past the grace but leased by the live child: retained.
    let held = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        held,
        TempCollection {
            examined: 1,
            deleted: 0,
            bytes_deleted: 0
        },
        "another process's lease must retain its temp"
    );
    assert_eq!(temp_files(&root.0), 1);

    child.kill().expect("SIGKILL the live writer");
    child.wait().expect("reap the killed writer");
    std::thread::sleep(Duration::from_millis(200));

    let reclaimed = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        reclaimed,
        TempCollection {
            examined: 1,
            deleted: 1,
            bytes_deleted: 35
        }
    );
    assert_eq!(temp_files(&root.0), 0);
}
