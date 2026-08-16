#![cfg(target_os = "linux")]

use std::ffi::{CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

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

/// One directory entry exactly as `getdents64` delivered it. `std::fs::read_dir`
/// filters `.` and `..` out and never exposes `d_ino`/`d_type`, so the entries
/// the mount actually serves are only visible through the raw call.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Dirent {
    name: String,
    ino: u64,
    kind: u8,
}

fn raw_dirents(path: &Path) -> Vec<Dirent> {
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("test paths have no NUL");
    // SAFETY: `c_path` is a NUL-terminated path, and every pointer below is
    // used only while the stream is open.
    let stream = unsafe { libc::opendir(c_path.as_ptr()) };
    assert!(
        !stream.is_null(),
        "opendir({}) failed: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    let mut entries = Vec::new();
    loop {
        // A NULL return means end-of-stream OR failure; only errno separates
        // them, so it must start clear.
        unsafe { *libc::__errno_location() = 0 };
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            let error = std::io::Error::last_os_error();
            assert_eq!(
                error.raw_os_error(),
                Some(0),
                "readdir({}) failed: {error}",
                path.display()
            );
            break;
        }
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }
            .to_str()
            .expect("test names are UTF-8")
            .to_owned();
        entries.push(Dirent {
            name,
            ino: unsafe { (*entry).d_ino },
            kind: unsafe { (*entry).d_type },
        });
    }
    unsafe { libc::closedir(stream) };
    entries
}

/// The listed names, sorted, including `.` and `..`.
fn dirent_names(entries: &[Dirent]) -> Vec<String> {
    let mut names: Vec<String> = entries.iter().map(|entry| entry.name.clone()).collect();
    names.sort();
    names
}

fn dirent<'a>(entries: &'a [Dirent], name: &str) -> &'a Dirent {
    let mut found = entries.iter().filter(|entry| entry.name == name);
    let entry = found
        .next()
        .unwrap_or_else(|| panic!("{name} is listed, got {:?}", dirent_names(entries)));
    assert!(found.next().is_none(), "{name} is listed exactly once");
    entry
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
        if entry_path == path
            && let Entry::File { records, .. } = entry
        {
            return records
                .iter()
                .map(|record| match record {
                    FileRecord::Data { digest, .. } => digest.to_string(),
                    FileRecord::Hole { length } => format!("hole:{length}"),
                })
                .collect();
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

/// The daemon's cumulative major faults — page faults that had to go to the
/// storage layer. Executable text paging lands here, and `read_bytes` counts
/// those same bytes, so this separates "the workload read the file" from "the
/// process read itself".
fn proc_majflt(pid: u32) -> u64 {
    let raw = fs::read_to_string(format!("/proc/{pid}/stat")).expect("daemon stat reads");
    // Field 12 (1-based) is majflt. The comm field is parenthesised and may
    // contain spaces, so split after the closing paren rather than on it.
    let tail = raw.rsplit_once(')').expect("stat has a comm field").1;
    tail.split_whitespace()
        .nth(9)
        .and_then(|value| value.parse::<u64>().ok())
        .expect("majflt parses")
}

/// Pull the daemon's executable into the page cache before a measured window.
///
/// `read_bytes` counts bytes fetched from the storage layer, and the daemon
/// demand-pages its own text: `BinMount::spawn` returns as soon as the mount
/// appears, so the compose path has never executed and none of its pages are
/// resident. Those faults would otherwise land inside the measurement and are
/// indistinguishable from the workload's own reads. Reading the file populates
/// the shared per-inode page cache, so the daemon's later text faults are
/// minor and cost no block I/O.
///
/// This does not weaken the bound — it removes a term that was never part of
/// what the bound is about.
fn prefault_daemon_image() {
    let image = Path::new(env!("CARGO_BIN_EXE_tensorfsd"));
    let bytes = fs::read(image).expect("daemon image reads");
    // Defeat any optimisation that would elide the read entirely.
    assert!(!bytes.is_empty(), "the daemon image is not empty");
}

struct BinMount {
    child: process::Child,
    mountpoint: PathBuf,
    /// Set once the daemon has been signalled and its endpoint dealt with, so
    /// the panic-path cleanup in `Drop` stands down.
    reaped: bool,
}

impl Drop for BinMount {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        // A test that panics mid-way — the normal state of affairs while
        // red-proving — would otherwise leave a live daemon and a mount entry
        // behind on a machine several lanes share. An in-process
        // `mount_workspace` unmounts itself when its session drops; a spawned
        // daemon has nothing that would.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = process::Command::new("fusermount3")
            .args(["-u", "-z"])
            .arg(&self.mountpoint)
            .status();
    }
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
                    reaped: false,
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
        self.reaped = true;
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
        self.reaped = true;
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
    // The daemon has only run the mount path so far; its compose text is not
    // resident and would fault in on the storage layer inside the window below.
    prefault_daemon_image();
    let majflt_before = proc_majflt(daemon.pid());
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
    let majflt_delta = proc_majflt(daemon.pid()) - majflt_before;
    daemon.sigterm_and_wait();

    let read_delta = read_after - read_before;
    let write_delta = write_after - write_before;
    println!(
        "8 KiB overwrite: read {} KiB, wrote {} KiB, {majflt_delta} major faults",
        read_delta / 1024,
        write_delta / 1024,
    );
    // A gross ceiling on block reads, and deliberately NOT the proof that only
    // one slot was composed. Measured 2026-08-15, both with a warm cache and
    // with every object evicted first: a single-slot compose and a forced
    // whole-file recompose read the SAME volume (0 KiB warm, 96 MiB cold in
    // both cases). Read volume here is invariant under the regression, so it
    // cannot discriminate it — the write bound below is what does, and it was
    // red-proved against exactly that mutation. The exact composition of the
    // cold-cache 96 MiB was not run to ground; see the PR body.
    assert!(
        read_delta < 96 * MIB,
        "the overwrite path must not explode block reads \
         (read {read_delta} bytes across {majflt_delta} major faults; a non-zero fault \
          count means the daemon paged itself in and the prefault did not hold)"
    );
    // THE discriminating assertion: a whole-file recompose writes 160 MiB here
    // against one object's 64 MiB, and forcing every slot to compose fails
    // exactly this line.
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
fn readdir_serves_exactly_the_live_namespace_on_a_writable_mount() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("readdir-root");
    let mountpoint = unique_dir("readdir-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace mounts");
    let base = mount.mountpoint();
    let root_ino = fs::metadata(base).expect("root metadata reads").ino();

    // An empty writable root still serves the two synthetic entries and
    // nothing else.
    let listing = raw_dirents(base);
    assert_eq!(
        dirent_names(&listing),
        [".", ".."],
        "an empty root is empty"
    );
    assert_eq!(dirent(&listing, ".").ino, root_ino, "`.` names itself");
    assert_eq!(
        dirent(&listing, "..").ino,
        root_ino,
        "the mount root is its own parent"
    );

    fs::create_dir(base.join("models")).expect("mkdir works");
    fs::write(base.join("notes.txt"), b"hello").expect("write works");
    std::os::unix::fs::symlink("notes.txt", base.join("latest")).expect("symlink works");
    fs::hard_link(base.join("notes.txt"), base.join("alias.txt")).expect("hardlink works");

    // Every created name appears, once, with an honest type and inode.
    let listing = raw_dirents(base);
    assert_eq!(
        dirent_names(&listing),
        [".", "..", "alias.txt", "latest", "models", "notes.txt"],
        "the listing is exactly the live namespace"
    );
    assert_eq!(dirent(&listing, "models").kind, libc::DT_DIR);
    assert_eq!(dirent(&listing, "notes.txt").kind, libc::DT_REG);
    assert_eq!(dirent(&listing, "alias.txt").kind, libc::DT_REG);
    assert_eq!(dirent(&listing, "latest").kind, libc::DT_LNK);
    for name in ["alias.txt", "latest", "models", "notes.txt"] {
        let meta = fs::symlink_metadata(base.join(name)).expect("entry stats");
        assert_eq!(
            dirent(&listing, name).ino,
            meta.ino(),
            "the listed inode of {name} is the one stat reports"
        );
    }
    assert_eq!(
        dirent(&listing, "alias.txt").ino,
        dirent(&listing, "notes.txt").ino,
        "both links of a hardlink group list the same inode"
    );

    // A subdirectory names its real parent through `..`.
    let models = base.join("models");
    fs::create_dir(models.join("v1")).expect("nested mkdir works");
    let nested = raw_dirents(&models);
    assert_eq!(dirent_names(&nested), [".", "..", "v1"]);
    assert_eq!(
        dirent(&nested, ".").ino,
        fs::metadata(&models).expect("subdir stats").ino(),
        "`.` names the directory itself"
    );
    assert_eq!(
        dirent(&nested, "..").ino,
        root_ino,
        "`..` names the parent directory, not the directory itself"
    );

    // Entries disappear the moment their name does.
    fs::remove_file(base.join("alias.txt")).expect("unlink works");
    fs::remove_dir(models.join("v1")).expect("rmdir works");
    assert_eq!(
        dirent_names(&raw_dirents(base)),
        [".", "..", "latest", "models", "notes.txt"],
        "an unlinked name leaves the listing at once"
    );
    assert_eq!(dirent_names(&raw_dirents(&models)), [".", ".."]);

    // An unlinked-but-open file is not a directory entry, and is still a
    // readable file.
    let payload = pattern(23, 4096);
    fs::write(base.join("doomed.bin"), &payload).expect("doomed write works");
    let mut handle = File::open(base.join("doomed.bin")).expect("doomed opens");
    fs::remove_file(base.join("doomed.bin")).expect("doomed unlinks");
    let listing = dirent_names(&raw_dirents(base));
    assert!(
        !listing.iter().any(|name| name == "doomed.bin"),
        "an unlinked-open file is never listed, got {listing:?}"
    );
    let mut served = Vec::new();
    handle
        .read_to_end(&mut served)
        .expect("the open handle still reads");
    assert_eq!(served, payload, "the unlisted file still serves its bytes");
    drop(handle);

    // A directory far larger than one readdir reply lists completely: the
    // kernel re-issues the call at the last offset the mount handed back, and
    // every entry must survive that resume exactly once. The names are long
    // deliberately — a 24-byte `fuse_dirent` header plus a 246-byte name puts
    // 162 entries at ~44 KiB against the ~32 KiB reply this kernel asks for,
    // so the listing takes two calls. Measured 2026-08-16: the first reply
    // fills at entry 118 and the second resumes there. Short names do not
    // force the split — 202 entries of ~80 bytes arrive in one reply and the
    // resume path is never entered.
    let bulk = base.join("bulk");
    fs::create_dir(&bulk).expect("bulk mkdir works");
    let mut expected: Vec<String> = (0..160)
        .map(|index| format!("entry-{index:04}-{}", "p".repeat(240)))
        .collect();
    for name in &expected {
        fs::write(bulk.join(name), b"x").expect("bulk write works");
    }
    expected.push("..".to_owned());
    expected.push(".".to_owned());
    expected.sort();
    // The padding is identical across names, so a failure prints only the
    // distinguishing prefix instead of 44 KiB of `p`.
    let stems = |names: &[String]| -> Vec<String> {
        names
            .iter()
            .map(|name| name.chars().take(11).collect())
            .collect()
    };
    let expected_stems = stems(&expected);
    assert_eq!(
        stems(&dirent_names(&raw_dirents(&bulk))),
        expected_stems,
        "a multi-reply directory lists every entry exactly once"
    );

    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");

    // The namespace a fresh mount rebuilds lists the same way.
    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let base = mount.mountpoint();
    assert_eq!(
        dirent_names(&raw_dirents(base)),
        [".", "..", "bulk", "latest", "models", "notes.txt"],
        "a remounted namespace lists exactly what was committed"
    );
    assert_eq!(
        stems(&dirent_names(&raw_dirents(&base.join("bulk")))),
        expected_stems,
        "the large directory survives a remount entry for entry"
    );
    let nested = raw_dirents(&base.join("models"));
    assert_eq!(
        dirent(&nested, "..").ino,
        fs::metadata(base).expect("root stats").ino(),
        "`..` still names the parent after a remount"
    );
    mount.unmount();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn rmdir_removes_empty_directories_and_refuses_populated_ones() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("rmdir-root");
    let mountpoint = unique_dir("rmdir-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace mounts");
    let base = mount.mountpoint();

    // The simple case: an empty directory goes away.
    fs::create_dir(base.join("empty")).expect("mkdir works");
    fs::remove_dir(base.join("empty")).expect("an empty directory removes");
    assert!(
        fs::metadata(base.join("empty")).is_err(),
        "the removed name is gone the moment rmdir returns"
    );

    // A directory holding a file refuses, and loses nothing.
    fs::create_dir(base.join("full")).expect("mkdir works");
    let kept = pattern(31, 8192);
    fs::write(base.join("full").join("keep.bin"), &kept).expect("child write works");
    let refused = fs::remove_dir(base.join("full")).expect_err("a populated directory refuses");
    assert_eq!(
        refused.raw_os_error(),
        Some(libc::ENOTEMPTY),
        "a directory holding a file refuses with ENOTEMPTY"
    );
    assert_eq!(
        fs::read(base.join("full").join("keep.bin")).expect("child reads"),
        kept,
        "a refused rmdir touches nothing"
    );

    // A directory holding only a subdirectory refuses just the same.
    fs::create_dir(base.join("full").join("nested")).expect("nested mkdir works");
    assert_eq!(
        fs::remove_dir(base.join("full"))
            .expect_err("a directory holding two children refuses")
            .raw_os_error(),
        Some(libc::ENOTEMPTY)
    );
    fs::remove_file(base.join("full").join("keep.bin")).expect("child unlinks");
    assert_eq!(
        fs::remove_dir(base.join("full"))
            .expect_err("a directory holding a subdirectory refuses")
            .raw_os_error(),
        Some(libc::ENOTEMPTY),
        "a subdirectory alone is still not empty"
    );
    fs::remove_dir(base.join("full").join("nested")).expect("nested rmdir works");
    fs::remove_dir(base.join("full")).expect("an emptied directory removes");

    // Names that are not empty directories.
    assert_eq!(
        fs::remove_dir(base.join("never"))
            .expect_err("a missing name refuses")
            .raw_os_error(),
        Some(libc::ENOENT)
    );
    fs::write(base.join("plain.bin"), b"x").expect("plain write works");
    assert_eq!(
        fs::remove_dir(base.join("plain.bin"))
            .expect_err("a regular file refuses rmdir")
            .raw_os_error(),
        Some(libc::ENOTDIR)
    );

    // POSIX: unlink removes the entry, not the file. A directory whose only
    // child was unlinked while open holds no entries, so rmdir must succeed —
    // and the surviving handle outlives the directory it was named in.
    fs::create_dir(base.join("holding")).expect("holding mkdir works");
    let doomed = pattern(37, (MIB / 4) as usize);
    fs::write(base.join("holding").join("doomed.bin"), &doomed).expect("doomed write works");
    let mut handle = File::open(base.join("holding").join("doomed.bin")).expect("doomed opens");
    fs::remove_file(base.join("holding").join("doomed.bin")).expect("doomed unlinks");
    fs::remove_dir(base.join("holding"))
        .expect("a directory whose only entry was unlinked is empty and removes");
    let mut served = Vec::new();
    handle
        .read_to_end(&mut served)
        .expect("the open handle still reads");
    assert_eq!(
        served, doomed,
        "an unlinked-open file outlives the directory that named it"
    );
    drop(handle);

    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");

    // Every removal was its own durable generation.
    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let base = mount.mountpoint();
    assert_eq!(
        dirent_names(&raw_dirents(base)),
        [".", "..", "plain.bin"],
        "only the names that survived rmdir come back"
    );
    mount.unmount();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

/// One append record: self-describing, so a reader can tell which writer wrote
/// it, in what order, and whether it arrived whole.
const APPEND_RECORD: usize = 1024;
const APPEND_RECORDS: u32 = 1024;

fn append_record(writer: u8, seq: u32) -> Vec<u8> {
    let mut record = vec![0_u8; APPEND_RECORD];
    record[0] = writer;
    record[1..5].copy_from_slice(&seq.to_le_bytes());
    for (index, byte) in record[5..].iter_mut().enumerate() {
        *byte = (index as u8)
            .wrapping_mul(31)
            .wrapping_add(writer)
            .wrapping_add(seq as u8);
    }
    record
}

#[test]
fn two_o_append_writers_never_overwrite_each_other() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("append-root");
    let mountpoint = unique_dir("append-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    // A real daemon process: the appenders are then genuinely foreign to the
    // filesystem serving them.
    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let path = mountpoint.join("appended.log");
    drop(File::create(&path).expect("the log creates"));

    // Two independent O_APPEND descriptors, released together and each writing
    // its own records. Nothing coordinates their offsets: every write lands
    // wherever the append contract says the end of the file is at that instant.
    let barrier = Barrier::new(2);
    let spans: [(Instant, Instant); 2] = std::thread::scope(|scope| {
        let writers = [1_u8, 2_u8].map(|writer| {
            let barrier = &barrier;
            let path = path.as_path();
            scope.spawn(move || {
                let mut file = OpenOptions::new()
                    .append(true)
                    .open(path)
                    .expect("the log opens for append");
                barrier.wait();
                let started = Instant::now();
                for seq in 0..APPEND_RECORDS {
                    file.write_all(&append_record(writer, seq))
                        .expect("append write works");
                }
                file.sync_all().expect("append fsync works");
                (started, Instant::now())
            })
        });
        writers.map(|writer| writer.join().expect("append writers do not panic"))
    });
    assert!(
        spans[0].0 < spans[1].1 && spans[1].0 < spans[0].1,
        "the two writers were in flight at the same time: {spans:?}"
    );

    let total = 2 * APPEND_RECORDS as usize * APPEND_RECORD;
    let bytes = fs::read(&path).expect("the log reads");
    assert_eq!(
        bytes.len(),
        total,
        "the file is exactly the sum of both writers' payloads; anything shorter \
         means one writer's appends landed on top of the other's"
    );

    // Every record must be whole, in its own writer's order, and present once.
    // A torn or interleaved write breaks the record alignment and the header
    // check below fails on the very first misaligned byte.
    let mut order: Vec<u8> = Vec::with_capacity(2 * APPEND_RECORDS as usize);
    let mut seen: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for (index, chunk) in bytes.chunks(APPEND_RECORD).enumerate() {
        let writer = chunk[0];
        assert!(
            writer == 1 || writer == 2,
            "record {index} starts with a writer tag, got {writer}"
        );
        let seq = u32::from_le_bytes(chunk[1..5].try_into().expect("four header bytes"));
        assert_eq!(
            chunk,
            append_record(writer, seq).as_slice(),
            "record {index} (writer {writer}, seq {seq}) arrived whole and unmodified"
        );
        order.push(writer);
        seen[usize::from(writer) - 1].push(seq);
    }
    for (index, sequence) in seen.iter().enumerate() {
        assert_eq!(
            sequence,
            &(0..APPEND_RECORDS).collect::<Vec<u32>>(),
            "writer {} landed every record exactly once, in the order it wrote them",
            index + 1
        );
    }
    // Evidence that the two writers were interleaved in the file itself, not
    // merely overlapping in wall-clock time.
    let transitions = order.windows(2).filter(|pair| pair[0] != pair[1]).count();
    println!("O_APPEND: {transitions} writer transitions across {total} bytes");
    assert!(
        transitions > 0,
        "the writers' records interleave; a run with zero transitions means the \
         test serialized them and proved nothing"
    );

    daemon.sigterm_and_wait();

    // The same bytes, from the committed generation.
    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let committed = fs::read(mount.mountpoint().join("appended.log")).expect("the log rereads");
    assert_eq!(
        committed, bytes,
        "the fsynced appends survive a remount byte for byte"
    );
    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn fsyncdir_succeeds_and_the_namespace_it_acknowledged_survives_a_sigkill() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("fsyncdir-root");
    let mountpoint = unique_dir("fsyncdir-mnt");
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);

    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let models = mountpoint.join("models");
    fs::create_dir(&models).expect("mkdir works");
    fs::create_dir(models.join("nested")).expect("nested mkdir works");
    std::os::unix::fs::symlink("weights.bin", models.join("latest")).expect("symlink works");
    let payload = pattern(41, (MIB / 2) as usize);
    let weights = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(models.join("weights.bin"))
        .expect("weights create works");
    weights
        .write_all_at(&payload, 0)
        .expect("weights write works");
    weights.sync_all().expect("weights fsync works");
    drop(weights);

    // fsync and fdatasync on a directory descriptor are FSYNCDIR and
    // FSYNCDIR(datasync) — the only way to reach the operation from userspace.
    let handle = File::open(&models).expect("the directory opens");
    handle.sync_all().expect("fsyncdir succeeds");
    handle.sync_data().expect("fsyncdir with datasync succeeds");
    drop(handle);
    let handle = File::open(&mountpoint).expect("the mount root opens");
    handle
        .sync_all()
        .expect("fsyncdir on the mount root succeeds");
    drop(handle);

    // Whatever fsyncdir acknowledged must be on disk: no orderly shutdown runs
    // after this point.
    daemon.sigkill_and_reap();

    let mount = mount_workspace(&root, "main", &mountpoint).expect("workspace remounts");
    let base = mount.mountpoint();
    assert_eq!(
        dirent_names(&raw_dirents(base)),
        [".", "..", "models"],
        "the root fsyncdir acknowledged survived the kill"
    );
    assert_eq!(
        dirent_names(&raw_dirents(&base.join("models"))),
        [".", "..", "latest", "nested", "weights.bin"],
        "every entry fsyncdir acknowledged survived the kill"
    );
    assert_eq!(
        fs::read(base.join("models").join("weights.bin")).expect("weights read"),
        payload,
        "the fsynced file content survived alongside its directory entry"
    );
    assert_eq!(
        fs::read_link(base.join("models").join("latest")).expect("readlink works"),
        PathBuf::from("weights.bin"),
        "the symlink survived"
    );
    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves a clean table");
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
