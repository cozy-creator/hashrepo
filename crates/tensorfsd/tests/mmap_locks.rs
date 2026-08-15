//! Real-mount proofs for the mmap serving path and POSIX advisory locks.
//!
//! Every mount is served by a SPAWNED `tensorfsd` process, never this test
//! process: a process that serves a mount must never touch it (the
//! same-process writeback deadlock the pgw#1256 bench hit live). Mapped and
//! locked access below therefore always crosses a process boundary to the
//! daemon, and lock contention additionally crosses a second boundary to
//! `python3` helper processes.

#![cfg(target_os = "linux")]

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Mutex;
use std::time::Duration;

use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::FileRecord;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

static MOUNT_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
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
    if process::Command::new("python3")
        .args(["-c", "import fcntl"])
        .output()
        .is_err()
    {
        eprintln!("skipping: python3 with fcntl is not available");
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

/// A python helper that acquires locks through the mount and holds them until
/// killed, so contention genuinely crosses process boundaries.
struct LockHolder {
    child: process::Child,
}

impl LockHolder {
    /// `ops` are `(start, len, exclusive)` rows all taken before signalling
    /// readiness through `sentinel`.
    fn hold(file: &Path, sentinel: &Path, ops: &[(u64, u64, bool)]) -> Self {
        let rows = ops
            .iter()
            .map(|(start, len, exclusive)| format!("({start},{len},{})", u8::from(*exclusive)))
            .collect::<Vec<_>>()
            .join(",");
        let code = format!(
            "import fcntl, pathlib, time\n\
             fd = open('{file}', 'r+b')\n\
             for start, length, exclusive in [{rows}]:\n\
             \x20   kind = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH\n\
             \x20   fcntl.lockf(fd, kind | fcntl.LOCK_NB, length, start)\n\
             pathlib.Path('{sentinel}').touch()\n\
             time.sleep(600)\n",
            file = file.display(),
            sentinel = sentinel.display(),
        );
        let mut child = process::Command::new("python3")
            .args(["-c", &code])
            .spawn()
            .expect("lock holder spawns");
        for _ in 0..100 {
            if sentinel.exists() {
                return Self { child };
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("lock holder did not signal readiness");
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One-shot lock attempt from a separate process: exit 0 granted, 3 refused.
fn try_lock(file: &Path, start: u64, len: u64, exclusive: bool) -> bool {
    let kind = if exclusive { "LOCK_EX" } else { "LOCK_SH" };
    let code = format!(
        "import fcntl, sys\n\
         fd = open('{file}', 'r+b')\n\
         try:\n\
         \x20   fcntl.lockf(fd, fcntl.{kind} | fcntl.LOCK_NB, {len}, {start})\n\
         except OSError:\n\
         \x20   sys.exit(3)\n",
        file = file.display(),
    );
    let status = process::Command::new("python3")
        .args(["-c", &code])
        .status()
        .expect("lock attempt runs");
    match status.code() {
        Some(0) => true,
        Some(3) => false,
        other => panic!("lock attempt died unexpectedly: {other:?}"),
    }
}

/// F_GETLK from a separate process; returns `(l_type, l_start, l_len, l_pid)`.
fn get_lock(file: &Path, start: u64, len: u64, exclusive: bool) -> (i32, u64, u64, u32) {
    let l_type = if exclusive { "F_WRLCK" } else { "F_RDLCK" };
    let code = format!(
        "import fcntl, struct\n\
         fd = open('{file}', 'r+b')\n\
         probe = struct.pack('hhqqi4x', fcntl.{l_type}, 0, {start}, {len}, 0)\n\
         result = fcntl.fcntl(fd, fcntl.F_GETLK, probe)\n\
         l_type, _, l_start, l_len, l_pid = struct.unpack('hhqqi4x', result)\n\
         print(l_type, l_start, l_len, l_pid)\n",
        file = file.display(),
    );
    let output = process::Command::new("python3")
        .args(["-c", &code])
        .output()
        .expect("getlk runs");
    assert!(
        output.status.success(),
        "getlk failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("getlk output is UTF-8");
    let mut fields = text.split_whitespace();
    let mut next = || fields.next().expect("getlk field present");
    (
        next().parse().expect("l_type parses"),
        next().parse().expect("l_start parses"),
        next().parse().expect("l_len parses"),
        next().parse().expect("l_pid parses"),
    )
}

#[test]
fn mapped_reads_serve_a_sparse_multi_record_file_byte_exact() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("mm-sparse-root");
    let mountpoint = unique_dir("mm-sparse-mnt");

    // Committed shape: [1 MiB data][2 MiB hole][64 KiB + 3 bytes data], the
    // tail deliberately not page-aligned.
    let head = pattern(3, MIB as usize);
    let tail = pattern(9, (64 * 1024 + 3) as usize);
    {
        let meta = WorkspaceStore::open(&root).expect("store opens");
        meta.create_workspace("main").expect("workspace creates");
        let head_object = meta.store().put_bytes(&head).expect("head admits");
        let tail_object = meta.store().put_bytes(&tail).expect("tail admits");
        meta.commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "sparse.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![
                    FileRecord::Data {
                        digest: head_object.digest(),
                        length: head.len() as u64,
                    },
                    FileRecord::Hole { length: 2 * MIB },
                    FileRecord::Data {
                        digest: tail_object.digest(),
                        length: tail.len() as u64,
                    },
                ],
            }],
        )
        .expect("file commits");
    }

    let mount = BinMount::spawn(&root, "main", &mountpoint);
    let file = fs::File::open(mountpoint.join("sparse.bin")).expect("file opens through mount");
    let logical = MIB + 2 * MIB + tail.len() as u64;
    assert_eq!(file.metadata().expect("stat works").len(), logical);

    // The map itself is the read path: page faults become FUSE reads served
    // by the daemon process.
    let map = unsafe_free_map(&file);
    assert_eq!(map.len() as u64, logical);
    assert_eq!(&map[..head.len()], &head[..], "head bytes are byte-exact");
    let hole = &map[head.len()..(MIB + 2 * MIB) as usize];
    assert!(
        hole.iter().all(|byte| *byte == 0),
        "every faulted hole page reads as zeros"
    );
    assert_eq!(
        &map[(MIB + 2 * MIB) as usize..],
        &tail[..],
        "unaligned tail bytes are byte-exact"
    );
    drop(map);
    drop(file);
    mount.sigterm_and_wait();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

/// `memmap2::Mmap::map` is `unsafe` only because of foreign truncation UB;
/// these tests own every accessor of the file, so the obligation is met by
/// construction. Isolated here so the test bodies stay `unsafe`-free.
fn unsafe_free_map(file: &fs::File) -> memmap2::Mmap {
    // SAFETY: no other process truncates the mapped file while the map lives;
    // the daemon never mutates committed content underneath a mount.
    unsafe { memmap2::Mmap::map(file).expect("mmap succeeds on a page-cache mount") }
}

fn unsafe_free_map_mut(file: &fs::File) -> memmap2::MmapMut {
    // SAFETY: same single-accessor argument as `unsafe_free_map`.
    unsafe { memmap2::MmapMut::map_mut(file).expect("writable mmap succeeds") }
}

#[test]
fn shared_writable_mappings_survive_msync_and_a_daemon_kill() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("mm-msync-root");
    let mountpoint = unique_dir("mm-msync-mnt");
    {
        let meta = WorkspaceStore::open(&root).expect("store opens");
        meta.create_workspace("main").expect("workspace creates");
    }

    let mount = BinMount::spawn(&root, "main", &mountpoint);
    let file_path = mountpoint.join("mapped.bin");
    let base = pattern(21, (2 * MIB) as usize);
    {
        let mut file = fs::File::create(&file_path).expect("create works");
        file.write_all(&base).expect("write works");
        file.sync_all().expect("baseline fsync works");
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&file_path)
        .expect("rw open works");
    let mut map = unsafe_free_map_mut(&file);
    map[..64].copy_from_slice(&[0xA5; 64]);
    map[4096 - 32..4096 + 32].copy_from_slice(&[0x5A; 64]);
    let last = map.len() - 8;
    map[last..].copy_from_slice(&[0xC3; 8]);
    // msync(MS_SYNC): the kernel writes the dirty pages back through FUSE and
    // then issues fsync on the mapped file — the guaranteed durability cut.
    map.flush().expect("msync succeeds");

    // SIGKILL the daemon while the mapping and descriptor are still open:
    // no RELEASE is ever delivered, so its best-effort compose cannot mask
    // the msync path — the msync'd bytes must already be composed and
    // committed, and every undirtied byte of daemon state dies here.
    mount.sigkill_and_reap();
    drop(map);
    drop(file);

    let mount = BinMount::spawn(&root, "main", &mountpoint);
    let mut resurrected = Vec::new();
    fs::File::open(mountpoint.join("mapped.bin"))
        .expect("file reopens")
        .read_to_end(&mut resurrected)
        .expect("read works");
    assert_eq!(resurrected.len(), base.len());
    assert_eq!(&resurrected[..64], &[0xA5; 64], "page-zero edit survived");
    assert_eq!(
        &resurrected[4096 - 32..4096 + 32],
        &[0x5A; 64],
        "page-boundary edit survived"
    );
    assert_eq!(
        &resurrected[last..],
        &[0xC3; 8],
        "final-bytes edit survived"
    );
    assert_eq!(
        &resurrected[64..4096 - 32],
        &base[64..4096 - 32],
        "unmapped-edit region is untouched"
    );
    mount.sigterm_and_wait();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn mapped_reads_stay_coherent_with_descriptor_writes() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("mm-coherence-root");
    let mountpoint = unique_dir("mm-coherence-mnt");
    {
        let meta = WorkspaceStore::open(&root).expect("store opens");
        meta.create_workspace("main").expect("workspace creates");
    }

    let mount = BinMount::spawn(&root, "main", &mountpoint);
    let file_path = mountpoint.join("coherent.bin");
    fs::write(&file_path, pattern(33, 64 * 1024)).expect("seed write works");
    let file = fs::File::open(&file_path).expect("ro open works");
    let map = unsafe_free_map(&file);
    assert_ne!(&map[8192..8192 + 16], b"COHERENCE-PINNED");

    // The contract: one mount, one page cache. A write through any other
    // descriptor of this mount is visible to live mappings immediately,
    // without a remap and before any fsync.
    let writer = OpenOptions::new()
        .write(true)
        .open(&file_path)
        .expect("rw open works");
    writer
        .write_all_at(b"COHERENCE-PINNED", 8192)
        .expect("pwrite works");
    assert_eq!(
        &map[8192..8192 + 16],
        b"COHERENCE-PINNED",
        "mapped readers observe descriptor writes through the shared page cache"
    );

    // And the inverse: a mapped write is visible to read(2) immediately.
    drop(map);
    let map_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&file_path)
        .expect("rw open works");
    let mut map = unsafe_free_map_mut(&map_file);
    map[0..8].copy_from_slice(b"MAPPED!!");
    let mut through_fd = [0_u8; 8];
    let mut reader = fs::File::open(&file_path).expect("ro open works");
    reader.seek(SeekFrom::Start(0)).expect("seek works");
    reader.read_exact(&mut through_fd).expect("read works");
    assert_eq!(
        &through_fd, b"MAPPED!!",
        "descriptor reads observe mapped writes through the shared page cache"
    );

    drop(map);
    drop(map_file);
    drop(reader);
    drop(writer);
    drop(file);
    mount.sigterm_and_wait();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn posix_locks_conflict_split_and_release_across_processes() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let root = unique_dir("lk-root");
    let mountpoint = unique_dir("lk-mnt");
    let scratch = unique_dir("lk-scratch");
    {
        let meta = WorkspaceStore::open(&root).expect("store opens");
        meta.create_workspace("main").expect("workspace creates");
    }

    let mount = BinMount::spawn(&root, "main", &mountpoint);
    let file = mountpoint.join("locked.bin");
    fs::write(&file, vec![0_u8; 16 * 1024]).expect("seed write works");

    // Exclusive conflicts, range precision, and getlk reporting.
    let holder = LockHolder::hold(&file, &scratch.join("ready-1"), &[(0, 4096, true)]);
    assert!(!try_lock(&file, 0, 4096, true), "wrlck vs wrlck refuses");
    assert!(!try_lock(&file, 100, 100, false), "rdlck vs wrlck refuses");
    assert!(
        try_lock(&file, 4096, 4096, true),
        "an adjacent range is free: lock ranges are byte-precise"
    );
    let (l_type, l_start, l_len, l_pid) = get_lock(&file, 0, 4096, true);
    assert_eq!(l_type, libc::F_WRLCK, "getlk reports the holder's type");
    assert_eq!(
        (l_start, l_len),
        (0, 4096),
        "getlk reports the exact extent"
    );
    assert_eq!(l_pid, holder.pid(), "getlk names the holding process");

    // Closing releases: kill the holder and the range frees.
    holder.kill();
    assert!(
        try_lock(&file, 0, 4096, true),
        "a dead holder's locks are released on close"
    );

    // Shared locks share; exclusive still refuses.
    let holder = LockHolder::hold(&file, &scratch.join("ready-2"), &[(8192, 4096, false)]);
    assert!(
        try_lock(&file, 8192, 4096, false),
        "rdlck shares with rdlck"
    );
    assert!(!try_lock(&file, 9000, 100, true), "wrlck vs rdlck refuses");
    holder.kill();

    // Unlock-middle splits: the freed middle is grantable, both edges refuse.
    let split_sentinel = scratch.join("ready-3");
    let split_code = format!(
        "import fcntl, pathlib, time\n\
         fd = open('{file}', 'r+b')\n\
         fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 10000, 0)\n\
         fcntl.lockf(fd, fcntl.LOCK_UN, 2000, 4000)\n\
         pathlib.Path('{sentinel}').touch()\n\
         time.sleep(600)\n",
        file = file.display(),
        sentinel = split_sentinel.display(),
    );
    let mut splitter = process::Command::new("python3")
        .args(["-c", &split_code])
        .spawn()
        .expect("splitter spawns");
    for _ in 0..100 {
        if split_sentinel.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(split_sentinel.exists(), "splitter signalled readiness");
    assert!(
        try_lock(&file, 4000, 2000, true),
        "the unlocked middle of a split lock is free"
    );
    assert!(
        !try_lock(&file, 0, 100, true),
        "the left fragment still holds"
    );
    assert!(
        !try_lock(&file, 9000, 500, true),
        "the right fragment still holds"
    );
    let _ = splitter.kill();
    let _ = splitter.wait();

    mount.sigterm_and_wait();
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
    fs::remove_dir_all(&scratch).ok();
}
