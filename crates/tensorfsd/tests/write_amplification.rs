//! What a write through the mount actually costs the disk.
//!
//! Byte amplification is the daemon's `/proc/<pid>/io` `write_bytes` delta
//! divided by the logical bytes the application wrote, measured around the
//! write and its fsync with the daemon still alive — the same method the
//! 8 KiB-overwrite arm in `workspace_mount.rs` uses, so the numbers are
//! comparable across the change that introduced in-memory slot assembly.
//!
//! Three arms:
//!
//! * A sequential writer fills each 64 MiB grid slot in order, so every slot
//!   completes in RAM and is admitted exactly once. It must approach 1×.
//! * A post-fsync SIGKILL, because slots admitted straight out of RAM are the
//!   surface this change adds and durability across a crash is the claim that
//!   matters most.
//! * A sparse writer touches the far end of many slots and completes none of
//!   them, so the overlay must degrade to the spill file rather than grow. Its
//!   amplification is high and that is correct: a 64 MiB grid object costs
//!   64 MiB to admit no matter how few of its bytes changed. What is under test
//!   is that it stays byte-exact and that both the pre-flush and post-flush
//!   resident set stay bounded — the budget is load-bearing, not decorative.
//!
//! The spill arm was two until this file was trimmed for wall-clock; they drove
//! the identical access pattern and differed only in slot count and in which
//! RSS sample they asserted, so one arm now carries every assertion from both.

#![cfg(target_os = "linux")]

use std::fs::{self, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Mutex;
use std::time::Duration;

use tensorfs_core::workspace::WorkspaceStore;

/// Real mounts are a shared-kernel resource; these arms serialize with each
/// other exactly as the other mount-bearing suites do.
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

/// The daemon's process-lifetime resident high-water mark, in bytes.
fn proc_vmhwm(pid: u32) -> u64 {
    let raw = fs::read_to_string(format!("/proc/{pid}/status")).expect("daemon status reads");
    raw.lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .expect("VmHWM parses")
}

struct BinMount {
    child: process::Child,
    mountpoint: PathBuf,
    /// Set once the daemon has been signalled and its endpoint dealt with, so
    /// the panic-path cleanup in `Drop` stands down.
    reaped: bool,
}

impl Drop for BinMount {
    /// A panicking arm skips its explicit teardown. Without this guard the
    /// daemon outlives the run holding the harness's inherited stdout pipe, so
    /// cargo waits on it forever and a plain assertion failure presents as a
    /// hung suite plus a stale mount.
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = process::Command::new("fusermount3")
            .args(["-u", "-z"])
            .arg(&self.mountpoint)
            .status();
    }
}

impl BinMount {
    /// Spawns the real daemon binary, optionally overriding the assembly
    /// budget so an arm can force the overflow path deterministically.
    fn spawn(root: &Path, workspace: &str, mountpoint: &Path, budget: Option<u64>) -> Self {
        let mut command = process::Command::new(env!("CARGO_BIN_EXE_tensorfsd"));
        command.args([
            "mount-workspace",
            "--store",
            root.to_str().expect("test paths are UTF-8"),
            "--workspace",
            workspace,
            mountpoint.to_str().expect("test paths are UTF-8"),
        ]);
        if let Some(budget) = budget {
            command.env("TENSORFS_ASSEMBLY_BUDGET_BYTES", budget.to_string());
        }
        // Null stdio: an inherited pipe would keep the test harness's output
        // stream open for as long as a leaked daemon lives.
        command
            .stdout(process::Stdio::null())
            .stderr(process::Stdio::null());
        let mut child = command.spawn().expect("daemon spawns");
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
}

fn fresh_workspace(label: &str) -> (PathBuf, PathBuf) {
    let root = unique_dir(&format!("{label}-root"));
    let mountpoint = unique_dir(&format!("{label}-mnt"));
    let meta = WorkspaceStore::open(&root).expect("store opens");
    meta.create_workspace("main").expect("workspace creates");
    drop(meta);
    (root, mountpoint)
}

fn report(arm: &str, logical: u64, written: u64, read: u64) -> f64 {
    let amplification = written as f64 / logical as f64;
    println!(
        "{arm}: logical {} KiB, daemon wrote {} KiB, read {} KiB -> {amplification:.2}x write amplification",
        logical / 1024,
        written / 1024,
        read / 1024,
    );
    amplification
}

/// The headline arm. A sequential writer fills grid slots in order, so each
/// completes in RAM, is hashed and admitted straight out of memory, and its
/// buffer is freed — one write per byte, with no staging copy behind it.
#[test]
fn a_sequential_write_moves_each_byte_to_disk_about_once() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let (root, mountpoint) = fresh_workspace("amp-seq");
    let daemon = BinMount::spawn(&root, "main", &mountpoint, None);

    // Exactly two grid slots. What this arm proves is that a slot which
    // completes in RAM is admitted once and never staged; two completions
    // demonstrate that as well as three, and the payload is the arm's whole
    // cost. Deliberately a whole number of slots: a partial tail would add the
    // compose path and blur the one-write-per-byte claim.
    let logical = 128 * MIB;
    let block = pattern(11, MIB as usize);
    let path = mountpoint.join("model.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .expect("create works");

    let (read_before, write_before) = proc_io(daemon.pid());
    let mut offset = 0_u64;
    while offset < logical {
        file.write_all_at(&block, offset).expect("write works");
        offset += MIB;
    }
    file.sync_all().expect("fsync works");
    let (read_after, write_after) = proc_io(daemon.pid());

    // Byte-exactness first: a cheap write that loses bytes is worthless.
    // One probe at each end of both slots, including the last full block.
    let mut sample = vec![0_u8; block.len()];
    for at in [0_u64, 63 * MIB, 64 * MIB, logical - MIB] {
        file.read_exact_at(&mut sample, at)
            .expect("read back works");
        assert_eq!(sample, block, "bytes at {at} survive assembly");
    }
    drop(file);

    let peak_rss = proc_vmhwm(daemon.pid());
    daemon.sigterm_and_wait();

    let amplification = report(
        "sequential",
        logical,
        write_after - write_before,
        read_after - read_before,
    );
    println!("sequential: daemon peak RSS {} MiB", peak_rss / MIB);

    assert!(
        amplification < 1.5,
        "a sequential write must move each byte to disk about once, not stage it first \
         (measured {amplification:.2}x)"
    );
    // Assembly holds one slot at a time, so the daemon must not scale with
    // the file: the payload cannot become the daemon.
    assert!(
        peak_rss < 200 * MIB,
        "in-memory assembly must stay bounded (peak RSS {} MiB)",
        peak_rss / MIB
    );
}

/// The crash arm for the new surface. A slot admitted straight out of RAM is
/// named by a generation the daemon commits at fsync; nothing about it is
/// still in memory afterwards. So a SIGKILL immediately after that fsync must
/// leave every byte readable on a fresh mount — the object was durable before
/// the generation that names it existed, or the file would not be readable at
/// all.
///
/// Deliberately larger than one grid slot: below 64 MiB nothing ever completes
/// in RAM and this arm would test the old path by accident.
#[test]
fn eagerly_admitted_slots_survive_a_post_fsync_kill() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let (root, mountpoint) = fresh_workspace("amp-crash");
    let daemon = BinMount::spawn(&root, "main", &mountpoint, None);

    // Two full slots plus a partial one: two are admitted from RAM during the
    // write, the third composes at fsync.
    let logical = 140 * MIB;
    let block = pattern(41, MIB as usize);
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(mountpoint.join("crash.bin"))
        .expect("create works");
    let mut offset = 0_u64;
    while offset < logical {
        file.write_all_at(&block, offset).expect("write works");
        offset += MIB;
    }
    file.sync_all().expect("fsync works");
    drop(file);
    daemon.sigkill_and_reap();

    // A fresh daemon over the same store: nothing in the killed process's
    // memory can help now.
    let remounted = unique_dir("amp-crash-remnt");
    let daemon = BinMount::spawn(&root, "main", &remounted, None);
    let path = remounted.join("crash.bin");
    let metadata = fs::metadata(&path).expect("the fsynced file survives");
    assert_eq!(metadata.len(), logical, "the whole file survives the kill");

    let survivor = OpenOptions::new()
        .read(true)
        .open(&path)
        .expect("survivor opens");
    let mut sample = vec![0_u8; block.len()];
    // One probe inside each admitted slot and one inside the composed tail.
    for at in [0_u64, 63 * MIB, 64 * MIB, 127 * MIB, 128 * MIB, 139 * MIB] {
        survivor
            .read_exact_at(&mut sample, at)
            .expect("every slot reads back after the kill");
        assert_eq!(sample, block, "bytes at {at} survive the post-fsync kill");
    }
    drop(survivor);
    daemon.sigterm_and_wait();
    println!(
        "post-fsync kill: {} MiB byte-exact after remount",
        logical / MIB
    );
}

/// The degradation-and-budget arm. A sparse writer touches the far end of many
/// slots and completes none of them, so the overlay must fall back to the spill
/// file rather than grow. High amplification here is correct — a 64 MiB grid
/// object costs 64 MiB to admit however few of its bytes changed — so what is
/// under test is that the path stays byte-exact and that memory stays bounded,
/// not that it is cheap.
///
/// This arm was two: one proving the spill path stays exact and one proving the
/// budget forces it. They drove the *identical* access pattern — an 8 KiB stamp
/// 60 MiB into each slot, one-slot ceiling — and differed only in slot count and
/// in which RSS sample they asserted, so the pair paid for two daemons and two
/// rounds of composing to test one mechanism. Merged, with every assertion from
/// both kept: the six-slot count is what the pre-flush bound needs to be
/// discriminating, and the descending write order is the more adversarial of the
/// two, so nothing arrives where the next byte is expected.
#[test]
fn an_out_of_order_writer_spills_instead_of_growing_and_stays_exact() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let (root, mountpoint) = fresh_workspace("amp-sparse");
    // One slot's worth of ceiling: the first slot may assemble in RAM, every
    // later one must overflow. Left at the 512 MiB default these six incomplete
    // slots would all fit, and the arm would silently stop testing the spill.
    let budget = 64 * MIB;
    let daemon = BinMount::spawn(&root, "main", &mountpoint, Some(budget));

    let slots = 6_u64;
    // A DISTINCT stamp per slot. One shared pattern made every slot's bytes
    // interchangeable, so a spill read that addressed the wrong slot returned
    // the right bytes by coincidence and the readback below could not see it —
    // the spill holds its bytes at their LOGICAL offsets, and nothing proved
    // the read path agreed.
    let stamps: Vec<Vec<u8>> = (0..slots)
        .map(|slot| pattern(31 + slot as u8, 8192))
        .collect();
    let path = mountpoint.join("wide.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .expect("create works");
    file.set_len(slots * 64 * MIB).expect("truncate works");

    let (read_before, write_before) = proc_io(daemon.pid());
    // Descending slot order, each write landing 60 MiB into its own slot:
    // nothing completes, and nothing arrives where the next byte is expected.
    // Held in RAM, every one would zero-extend a buffer to 60 MiB — 360 MiB
    // for 48 KiB of data, which is what the pre-flush bound below rules out.
    for slot in (0..slots).rev() {
        let at = slot * 64 * MIB + 60 * MIB;
        file.write_all_at(&stamps[slot as usize], at)
            .expect("far-offset write works");
    }
    let peak_before_flush = proc_vmhwm(daemon.pid());
    file.sync_all().expect("fsync works");
    let (read_after, write_after) = proc_io(daemon.pid());

    drop(file);
    let peak_rss = proc_vmhwm(daemon.pid());
    daemon.sigterm_and_wait();

    // Byte-exactness is asserted on a FRESH daemon, never through the mount
    // that just did the writing. Reading back through the writing mount is
    // served out of the kernel page cache — those pages are clean and present
    // — so it proves the kernel remembers what we wrote and says nothing at
    // all about the spill. The spill holds its bytes at their LOGICAL offsets;
    // making the compose read them at slot-relative offsets instead gives
    // every slot the first slot's bytes, and the in-mount readback stayed
    // green through all of it.
    let remounted = unique_dir("amp-sparse-remnt");
    let daemon = BinMount::spawn(&root, "main", &remounted, None);
    let survivor = OpenOptions::new()
        .read(true)
        .open(remounted.join("wide.bin"))
        .expect("survivor opens");
    let mut sample = vec![0_u8; stamps[0].len()];
    for slot in 0..slots {
        let at = slot * 64 * MIB + 60 * MIB;
        survivor
            .read_exact_at(&mut sample, at)
            .expect("sparse read back works");
        assert_eq!(
            sample, stamps[slot as usize],
            "slot {slot}'s own stamp at {at} survives the spill path"
        );
    }
    // The untouched bytes of a sparse file are still holes reading as zero.
    let mut gap = vec![0xAA_u8; 4096];
    survivor
        .read_exact_at(&mut gap, 8 * MIB)
        .expect("gap reads");
    assert!(
        gap.iter().all(|byte| *byte == 0),
        "untouched bytes are zero"
    );
    drop(survivor);
    daemon.sigterm_and_wait();

    // No amplification bound is asserted: admitting six 64 MiB grid objects for
    // 48 KiB of change is inherent to content addressing on a fixed grid, and
    // pretending otherwise would be a bound on the grid, not on this code.
    report(
        "sparse/out-of-order",
        slots * stamps[0].len() as u64,
        write_after - write_before,
        read_after - read_before,
    );
    println!(
        "budget: ceiling {} MiB, peak RSS before flush {} MiB, after {} MiB",
        budget / MIB,
        peak_before_flush / MIB,
        peak_rss / MIB
    );
    // Composed slots ride the bounded compose pool, so a few hundred MiB after
    // the flush is legitimate; what must not happen is the overlay holding them.
    assert!(
        peak_rss < 320 * MIB,
        "the overflow path must hold slots on disk, not in the daemon (peak RSS {} MiB)",
        peak_rss / MIB
    );
    // Assert the PRE-FLUSH peak: at that point assembly is the only thing
    // holding slot buffers, so the number is about the budget and nothing
    // else. After the flush, composing legitimately holds COMPOSE_WORKERS
    // slot buffers, which floors the measurement near 256 MiB and would
    // drown out exactly what this arm tests.
    assert!(
        peak_before_flush < 200 * MIB,
        "the assembly budget must force the spill path \
         (pre-flush peak RSS {} MiB, ceiling {} MiB)",
        peak_before_flush / MIB,
        budget / MIB
    );
}
