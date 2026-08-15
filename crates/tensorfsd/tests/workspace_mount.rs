#![cfg(target_os = "linux")]

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Mutex;
use std::time::Duration;

use tensorfs_core::tfm1::{Entry, FileRecord};
use tensorfs_core::workspace::WorkspaceStore;
use tensorfsd::mount_workspace;

/// Real mounts are a shared-kernel resource; the machine budget is two at a
/// time, so the mount-bearing tests serialize on one lock.
static MOUNT_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    // A poisoned lock only means another test failed; these tests share
    // nothing but the two-mount machine budget.
    MOUNT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const MIB: u64 = 1024 * 1024;

fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        eprintln!("skipping: /dev/fuse is not available");
        return false;
    }
    if process::Command::new("fusermount3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: fusermount3 is not available");
        return false;
    }
    true
}

fn unique_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tensorfsd-{label}-{}-{:x}",
        process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .expect("clock is sane")
            .as_nanos()
    ));
    fs::create_dir_all(&path).expect("test directory creates");
    path
}

fn mounted_here(mountpoint: &Path) -> bool {
    let mounts = fs::read_to_string("/proc/self/mounts").expect("mount table reads");
    let needle = mountpoint.to_str().expect("test paths are UTF-8");
    mounts
        .lines()
        .any(|line| line.split_whitespace().nth(1) == Some(needle))
}

fn pattern(seed: u8, length: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; length];
    let mut state = u64::from(seed) | 1;
    for chunk in bytes.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        for (index, byte) in chunk.iter_mut().enumerate() {
            *byte = (state >> (index * 8)) as u8;
        }
    }
    bytes
}

fn count_store_objects(root: &Path) -> usize {
    let mut count = 0;
    let namespace = root.join("objects").join("sha256");
    for first in fs::read_dir(&namespace).expect("object namespace lists") {
        let first = first.expect("object fanout entry reads");
        if !first.path().is_dir() {
            continue;
        }
        for second in fs::read_dir(first.path()).expect("object fanout lists") {
            let second = second.expect("object fanout entry reads");
            count += fs::read_dir(second.path())
                .expect("object leaf lists")
                .count();
        }
    }
    count
}

fn file_digests(store: &WorkspaceStore, workspace: &str, path: &str) -> Vec<String> {
    let tree = store.head_tree(workspace).expect("head tree builds");
    for (entry_path, entry) in tree.entries() {
        if entry_path == path {
            if let Entry::File { records, .. } = entry {
                return records
                    .iter()
                    .map(|record| match record {
                        FileRecord::Data { digest, .. } => digest.to_string(),
                        FileRecord::Hole { length } => format!("hole:{length}"),
                    })
                    .collect();
            }
        }
    }
    panic!("{path} is not a committed file");
}

fn proc_io(pid: u32) -> (u64, u64) {
    let raw = fs::read_to_string(format!("/proc/{pid}/io")).expect("daemon io stats read");
    let field = |name: &str| {
        raw.lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .expect("io stat field parses")
    };
    (field("read_bytes:"), field("write_bytes:"))
}

struct BinMount {
    child: process::Child,
    mountpoint: PathBuf,
}

impl BinMount {
    fn spawn(root: &Path, workspace: &str, mountpoint: &Path) -> Self {
        let mut child = process::Command::new(env!("CARGO_BIN_EXE_tensorfsd"))
            .args([
                "mount-workspace",
                "--store",
                root.to_str().expect("test paths are UTF-8"),
                "--workspace",
                workspace,
                mountpoint.to_str().expect("test paths are UTF-8"),
            ])
            .spawn()
            .expect("daemon spawns");
        for _ in 0..100 {
            if mounted_here(mountpoint) {
                return Self {
                    child,
                    mountpoint: mountpoint.to_path_buf(),
                };
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("daemon did not mount in time");
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn sigterm_and_wait(mut self) {
        let _ = process::Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
        let _ = self.child.wait();
        assert!(
            !mounted_here(&self.mountpoint),
            "a terminated daemon must leave no mount behind"
        );
    }

    fn sigkill_and_reap(mut self) {
        let _ = process::Command::new("kill")
            .args(["-KILL", &self.child.id().to_string()])
            .status();
        let _ = self.child.wait();
        // SIGKILL leaves a disconnected FUSE endpoint; reap it explicitly the
        // way an operator would.
        let _ = process::Command::new("fusermount3")
            .args(["-u", "-z"])
            .arg(&self.mountpoint)
            .status();
        for _ in 0..50 {
            if !mounted_here(&self.mountpoint) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("stale mount survived cleanup");
    }
}

#[test]
fn ordinary_writes_survive_fsync_and_remount_byte_exact() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("ws-root");
    let mountpoint = unique_dir("ws-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace mounts");
    let base = mount.mountpoint();

    fs::create_dir(base.join("models")).expect("mkdir works");
    let payload = pattern(7, (2 * MIB) as usize);
    let file_path = base.join("models").join("weights.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&file_path)
        .expect("create works");
    file.write_all_at(&payload, 0).expect("write works");

    // The dirty overlay must serve reads BEFORE any fsync composes it.
    let mut before_fsync = vec![0_u8; payload.len()];
    file.read_exact_at(&mut before_fsync, 0)
        .expect("pre-fsync read works");
    assert_eq!(before_fsync, payload, "reads merge the newest dirty ranges");

    // An overlapping rewrite wins over the older dirty bytes.
    let overwrite = pattern(9, 4096);
    file.write_all_at(&overwrite, 1024).expect("rewrite works");
    let mut merged = vec![0_u8; 4096];
    file.read_exact_at(&mut merged, 1024)
        .expect("post-rewrite read works");
    assert_eq!(merged, overwrite, "the newest dirty range wins");

    file.sync_all().expect("fsync works");
    drop(file);

    fs::write(base.join("config.json"), b"{\"kind\":\"e2e\"}\n").expect("small write works");
    std::os::unix::fs::symlink("models/weights.bin", base.join("latest")).expect("symlink works");
    fs::hard_link(&file_path, base.join("models").join("alias.bin")).expect("hardlink works");
    fs::set_permissions(&file_path, fs::Permissions::from_mode(0o755)).expect("chmod works");

    let grown = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(base.join("sparse.bin"))
        .expect("sparse file creates");
    grown.write_all_at(b"head", 0).expect("head write works");
    grown.set_len(3 * MIB).expect("truncate-grow works");
    grown.sync_all().expect("sparse fsync works");
    drop(grown);

    let shrunk_path = base.join("shrunk.bin");
    let shrunk = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&shrunk_path)
        .expect("shrunk file creates");
    shrunk
        .write_all_at(&pattern(11, MIB as usize), 0)
        .expect("shrunk write works");
    shrunk.set_len(1000).expect("truncate-shrink works");
    shrunk.sync_all().expect("shrunk fsync works");
    drop(shrunk);

    fs::rename(base.join("config.json"), base.join("config-final.json")).expect("rename works");

    let file_ino = fs::metadata(&file_path).expect("metadata reads").ino();
    let alias_meta = fs::metadata(base.join("models").join("alias.bin")).expect("alias reads");
    assert_eq!(alias_meta.ino(), file_ino, "a hardlink shares its inode");
    assert_eq!(alias_meta.nlink(), 2, "a hardlink group counts two links");

    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");

    // Everything committed must survive a completely fresh mount.
    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let base = mount.mountpoint();
    let mut expected = payload.clone();
    expected[1024..1024 + 4096].copy_from_slice(&overwrite);
    assert_eq!(
        fs::read(base.join("models").join("weights.bin")).expect("weights read"),
        expected,
        "committed bytes are byte-exact after remount"
    );
    assert_eq!(
        fs::read(base.join("models").join("alias.bin")).expect("alias read"),
        expected,
        "the hardlink serves the same bytes"
    );
    let mode = fs::metadata(base.join("models").join("weights.bin"))
        .expect("metadata reads")
        .permissions()
        .mode();
    assert_eq!(mode & 0o111, 0o111, "the executable bit survived");
    assert_eq!(
        fs::read_link(base.join("latest")).expect("readlink works"),
        PathBuf::from("models/weights.bin"),
        "the symlink target survived"
    );
    assert_eq!(
        fs::read(base.join("config-final.json")).expect("renamed read"),
        b"{\"kind\":\"e2e\"}\n",
        "the renamed file survived under its new name"
    );
    let sparse = fs::read(base.join("sparse.bin")).expect("sparse read");
    assert_eq!(sparse.len() as u64, 3 * MIB, "truncate-grow set the size");
    assert_eq!(&sparse[..4], b"head", "the written head survived");
    assert!(
        sparse[4..].iter().all(|byte| *byte == 0),
        "grown bytes read as zeros"
    );
    let shrunk = fs::read(base.join("shrunk.bin")).expect("shrunk read");
    assert_eq!(
        shrunk,
        pattern(11, MIB as usize)[..1000],
        "shrink kept the prefix"
    );

    mount.unmount();
    assert!(
        !mounted_here(&mountpoint),
        "final unmount leaves a clean table"
    );
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn an_8_kib_overwrite_recomposes_exactly_the_touched_object() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("cow-root");
    let mountpoint = unique_dir("cow-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    // Phase 1: a three-object file written through a real mount.
    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let big_path = mountpoint.join("big.bin");
    let big = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&big_path)
        .expect("big file creates");
    let chunk = pattern(3, (8 * MIB) as usize);
    for index in 0..20 {
        big.write_all_at(&chunk, index * 8 * MIB)
            .expect("bulk write works");
    }
    big.sync_all().expect("bulk fsync works");
    drop(big);
    daemon.sigterm_and_wait();

    let meta = WorkspaceStore::open(&root).expect("store reopens");
    let before = file_digests(&meta, "main", "big.bin");
    assert_eq!(before.len(), 3, "160 MiB composes to three grid objects");
    let objects_before = count_store_objects(&root);
    drop(meta);

    // Phase 2: an 8 KiB overwrite inside the second object, with the daemon's
    // real I/O accounted, so a whole-file recompose cannot hide.
    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let (read_before, write_before) = proc_io(daemon.pid());
    let big = OpenOptions::new()
        .write(true)
        .open(mountpoint.join("big.bin"))
        .expect("big file reopens");
    big.write_all_at(&pattern(5, 8192), 70 * MIB)
        .expect("overwrite works");
    big.sync_all().expect("overwrite fsync works");
    drop(big);
    let (read_after, write_after) = proc_io(daemon.pid());
    daemon.sigterm_and_wait();

    let read_delta = read_after - read_before;
    let write_delta = write_after - write_before;
    assert!(
        read_delta < 96 * MIB,
        "composing one 64 MiB slot reads one slot, not the file (read {read_delta} bytes)"
    );
    assert!(
        write_delta < 112 * MIB,
        "composing one 64 MiB slot writes one object, not the file (wrote {write_delta} bytes)"
    );

    let meta = WorkspaceStore::open(&root).expect("store reopens again");
    let after = file_digests(&meta, "main", "big.bin");
    assert_eq!(after.len(), 3, "the record count is unchanged");
    assert_eq!(before[0], after[0], "the first object digest is untouched");
    assert_eq!(before[2], after[2], "the third object digest is untouched");
    assert_ne!(
        before[1], after[1],
        "exactly the overwritten object changed"
    );
    assert_eq!(
        count_store_objects(&root),
        objects_before + 1,
        "exactly one new object was admitted"
    );
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn an_unlinked_open_file_keeps_serving_until_close() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("unlink-root");
    let mountpoint = unique_dir("unlink-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace mounts");
    let base = mount.mountpoint();
    let payload = pattern(13, (MIB / 2) as usize);
    let path = base.join("doomed.bin");
    fs::write(&path, &payload).expect("write works");

    let mut handle = File::open(&path).expect("open works");
    fs::remove_file(&path).expect("unlink works");
    assert!(
        fs::metadata(&path).is_err(),
        "the name is gone the moment unlink returns"
    );

    let mut served = Vec::new();
    handle
        .read_to_end(&mut served)
        .expect("the open handle still reads");
    assert_eq!(served, payload, "unlink-open serves the exact bytes");
    drop(handle);

    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");

    let meta = WorkspaceStore::open(&root).expect("store reopens");
    let tree = meta.head_tree("main").expect("head tree builds");
    assert!(
        tree.entries().iter().all(|(path, _)| path != "doomed.bin"),
        "the committed tree no longer names the unlinked file"
    );
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn dirty_bytes_die_with_the_daemon_and_committed_generations_do_not() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("crash-root");
    let mountpoint = unique_dir("crash-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let daemon = BinMount::spawn(&root, "main", &mountpoint);

    // Durable arm: fsync returns only after the compose committed.
    let durable = pattern(17, (2 * MIB) as usize);
    let durable_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(mountpoint.join("durable.bin"))
        .expect("durable file creates");
    durable_file
        .write_all_at(&durable, 0)
        .expect("durable write works");
    durable_file.sync_all().expect("durable fsync works");

    // Lost arm: dirty bytes with no fsync and the handle deliberately open.
    let lost_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(mountpoint.join("lost.bin"))
        .expect("lost file creates");
    lost_file
        .write_all_at(&pattern(19, MIB as usize), 0)
        .expect("lost write works");

    daemon.sigkill_and_reap();
    drop(lost_file);
    drop(durable_file);

    // Recovery exposes exactly the committed generations: the fsynced bytes,
    // and the created-but-never-flushed file at its committed (empty) state —
    // never a hybrid.
    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let base = mount.mountpoint();
    assert_eq!(
        fs::read(base.join("durable.bin")).expect("durable read"),
        durable,
        "fsynced bytes survive the SIGKILL"
    );
    let lost = fs::metadata(base.join("lost.bin")).expect("lost metadata reads");
    assert_eq!(
        lost.len(),
        0,
        "un-fsynced dirty bytes die with the daemon; the committed create remains"
    );
    mount.unmount();
    assert!(
        !mounted_here(&mountpoint),
        "final unmount leaves a clean table"
    );
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}
