//! What a write through the mount actually costs the disk.
//!
//! Byte amplification is the daemon's `/proc/<pid>/io` `write_bytes` delta
//! divided by the logical bytes the application wrote, measured around the
//! write and its fsync with the daemon still alive — the same method the
//! 8 KiB-overwrite arm in `workspace_mount.rs` uses, so the numbers are
//! comparable across the change that introduced in-memory slot assembly.
//!
//! Two arms, because they are supposed to differ:
//!
//! * A sequential writer fills each 64 MiB grid slot in order, so every slot
//!   completes in RAM and is admitted exactly once. It must approach 1×.
//! * A sparse writer touches the far end of many slots and completes none of
//!   them, so the overlay degrades to the spill file. Its amplification is
//!   high and that is correct: a 64 MiB grid object costs 64 MiB to admit no
//!   matter how few of its bytes changed.
//!
//! The third arm is the budget itself: a writer whose resident slots would
//! exceed the ceiling must spill rather than grow, and the daemon's peak RSS
//! is the evidence.

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
        let mut child = command.spawn().expect("daemon spawns");
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
        "{arm}: logical {} MiB, daemon wrote {} MiB, read {} MiB -> {amplification:.2}x write amplification",
        logical / MIB,
        written / MIB,
        read / MIB,
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

    let logical = 512 * MIB;
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
    let mut sample = vec![0_u8; block.len()];
    for at in [0_u64, 64 * MIB, 200 * MIB, logical - MIB] {
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
    // the file: 512 MiB of payload cannot become 512 MiB of daemon.
    assert!(
        peak_rss < 320 * MIB,
        "in-memory assembly must stay bounded (peak RSS {} MiB)",
        peak_rss / MIB
    );
}

/// The degradation arm. A sparse writer touches the far end of many slots and
/// completes none of them, so the overlay falls back to the spill file. High
/// amplification here is correct — the grid object is 64 MiB regardless — and
/// the claim under test is that it stays correct and bounded, not that it is
/// cheap.
#[test]
fn an_out_of_order_writer_degrades_to_the_spill_and_stays_exact() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let (root, mountpoint) = fresh_workspace("amp-sparse");
    let daemon = BinMount::spawn(&root, "main", &mountpoint, None);

    let slots = 6_u64;
    let size = slots * 64 * MIB;
    let stamp = pattern(23, 8192);
    let path = mountpoint.join("sparse.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .expect("create works");
    file.set_len(size).expect("truncate works");

    let (read_before, write_before) = proc_io(daemon.pid());
    // Descending slot order, each write landing near the far end of its slot:
    // nothing completes, and nothing arrives where the next byte is expected.
    for slot in (0..slots).rev() {
        let at = slot * 64 * MIB + 60 * MIB;
        file.write_all_at(&stamp, at).expect("sparse write works");
    }
    file.sync_all().expect("fsync works");
    let (read_after, write_after) = proc_io(daemon.pid());

    let mut sample = vec![0_u8; stamp.len()];
    for slot in 0..slots {
        let at = slot * 64 * MIB + 60 * MIB;
        file.read_exact_at(&mut sample, at)
            .expect("sparse read back works");
        assert_eq!(sample, stamp, "the stamp at {at} survives the spill path");
    }
    // The untouched bytes of a sparse file are still holes reading as zero.
    let mut gap = vec![0xAA_u8; 4096];
    file.read_exact_at(&mut gap, 8 * MIB).expect("gap reads");
    assert!(
        gap.iter().all(|byte| *byte == 0),
        "untouched bytes are zero"
    );
    drop(file);

    let peak_rss = proc_vmhwm(daemon.pid());
    daemon.sigterm_and_wait();

    let logical = slots * stamp.len() as u64;
    report(
        "sparse/out-of-order",
        logical,
        write_after - write_before,
        read_after - read_before,
    );
    println!(
        "sparse/out-of-order: daemon peak RSS {} MiB",
        peak_rss / MIB
    );

    // No amplification bound is asserted: admitting six 64 MiB grid objects
    // for 48 KiB of change is inherent to content addressing on a fixed grid,
    // and pretending otherwise would be a bound on the grid, not on this code.
    assert!(
        peak_rss < 512 * MIB,
        "even the degraded path must not scale with the file (peak RSS {} MiB)",
        peak_rss / MIB
    );
}

/// The budget is load-bearing, not decorative. A writer whose resident slots
/// would exceed the ceiling must spill instead of growing, and the daemon's
/// peak RSS is the evidence: without the check, the same access pattern
/// balloons past three quarters of a gigabyte.
#[test]
fn a_writer_past_the_budget_spills_instead_of_growing_memory() {
    let _serial = serial();
    if !fuse_available() {
        return;
    }
    let (root, mountpoint) = fresh_workspace("amp-budget");
    // One slot's worth of ceiling: the first slot may assemble in RAM, every
    // later one must overflow.
    let budget = 64 * MIB;
    let daemon = BinMount::spawn(&root, "main", &mountpoint, Some(budget));

    let slots = 12_u64;
    let stamp = pattern(31, 8192);
    let path = mountpoint.join("wide.bin");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .expect("create works");
    file.set_len(slots * 64 * MIB).expect("truncate works");

    // Each write lands 60 MiB into its own slot. Held in RAM, every one of
    // them would zero-extend a buffer to 60 MiB: 720 MiB for 96 KiB of data.
    for slot in 0..slots {
        let at = slot * 64 * MIB + 60 * MIB;
        file.write_all_at(&stamp, at)
            .expect("far-offset write works");
    }
    let peak_before_flush = proc_vmhwm(daemon.pid());
    file.sync_all().expect("fsync works");

    let mut sample = vec![0_u8; stamp.len()];
    for slot in 0..slots {
        let at = slot * 64 * MIB + 60 * MIB;
        file.read_exact_at(&mut sample, at)
            .expect("read back works");
        assert_eq!(
            sample, stamp,
            "the stamp at {at} survives the overflow path"
        );
    }
    drop(file);
    let peak_rss = proc_vmhwm(daemon.pid());
    daemon.sigterm_and_wait();

    println!(
        "budget: ceiling {} MiB, peak RSS before flush {} MiB, after {} MiB",
        budget / MIB,
        peak_before_flush / MIB,
        peak_rss / MIB
    );
    // Composing a slot at flush legitimately holds one slot-sized buffer on
    // top of the ceiling, so the bound is generous — but nowhere near the
    // 720 MiB an unbudgeted overlay would reach.
    assert!(
        peak_rss < 320 * MIB,
        "the assembly budget must force the spill path (peak RSS {} MiB, ceiling {} MiB)",
        peak_rss / MIB,
        budget / MIB
    );
}
