//! Measures the direct (non-FUSE) ingest lane.
//!
//! Two arms, one binary, so before/after is produced under one set of
//! conditions rather than compared across runs:
//!
//!   double <store_root> <native_path>
//!       What this repo's benchmarks measured until now: `plan_and_hash`
//!       hashes every byte, then `put_bytes` hashes every byte AGAIN inside
//!       the writer. Two full SHA-256 passes for one admission, whole file
//!       resident. No production path does this.
//!
//!   single <store_root> <native_path>
//!       What production does: plan for boundaries only, then read, hash and
//!       admit each region exactly once, up to `ingest_concurrency()` at a
//!       time, streaming from the file.
//!
//! Worker count is set by `TENSORFS_ASSEMBLY_BUDGET_BYTES` (budget divided by
//! the 64 MiB object ceiling, clamped by core count), so the harness varies
//! concurrency from outside without this binary calling the unsafe
//! `set_var`.
//!
//! Wall-clock is REPORTED, never asserted: this box swings more than 10x with
//! load. The amplification numbers are counter ratios and are the honest part.
//!
//! `store` and `workspace` are gated to `any(unix, windows)` in `lib.rs`, so
//! every item here carries the same gate and the CI-exact `--all-targets`
//! wasm32 check still sees a target that compiles.

#[cfg(any(unix, windows))]
use std::env;
#[cfg(any(unix, windows))]
use std::path::Path;
#[cfg(any(unix, windows))]
use std::time::Instant;

#[cfg(any(unix, windows))]
use tensorfs_core::object::plan_and_hash;
#[cfg(any(unix, windows))]
use tensorfs_core::planner::{ByteSource, plan};
#[cfg(any(unix, windows))]
use tensorfs_core::source::FileByteSource;
#[cfg(any(unix, windows))]
use tensorfs_core::store::{AdmittedObject, ObjectStore, ingest_concurrency};

#[cfg(any(unix, windows))]
struct Slice<'a>(&'a [u8]);

#[cfg(any(unix, windows))]
impl ByteSource for Slice<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> std::io::Result<()> {
        let start = usize::try_from(offset).expect("offset fits");
        destination.copy_from_slice(&self.0[start..start + destination.len()]);
        Ok(())
    }

    fn check_unchanged(&self) -> std::io::Result<()> {
        Ok(())
    }
}

/// This process's cumulative bytes actually read from / written to storage.
/// Amplification is a ratio of these counters, never of stopwatches.
#[cfg(any(unix, windows))]
fn io_counters() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let mut read = 0;
    let mut write = 0;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("read_bytes:") {
            read = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("write_bytes:") {
            write = rest.trim().parse().unwrap_or(0);
        }
    }
    (read, write)
}

#[cfg(any(unix, windows))]
fn one_minute_load() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

#[cfg(any(unix, windows))]
fn report(mode: &str, workers: usize, objects: usize, bytes: u64, elapsed: f64, io: (u64, u64)) {
    println!("MODE={mode}");
    println!("WORKERS={workers}");
    println!("OBJECTS={objects}");
    println!("LOGICAL_BYTES={bytes}");
    println!("WALL_S={elapsed:.4}");
    println!("MIB_S={:.0}", bytes as f64 / elapsed / 1_048_576.0);
    println!("PROC_READ_BYTES={}", io.0);
    println!("PROC_WRITE_BYTES={}", io.1);
    println!("WRITE_AMPLIFICATION={:.4}", io.1 as f64 / bytes as f64);
}

/// Production's shape: boundaries first, then one read + one hash + one
/// install per region, bounded-concurrent, streaming.
#[cfg(any(unix, windows))]
fn single(store_root: &Path, native: &Path) {
    let store = ObjectStore::open(store_root).expect("store opens");
    let source = FileByteSource::open(native).expect("source opens");
    let planned = plan(&source).expect("source plans");

    let (read0, write0) = io_counters();
    let start = Instant::now();
    let admitted = store
        .admit_regions(&source, planned.regions())
        .expect("regions admit");
    let elapsed = start.elapsed().as_secs_f64();
    let (read1, write1) = io_counters();

    let bytes: u64 = admitted.iter().map(AdmittedObject::length).sum();
    report(
        "single-pass",
        ingest_concurrency(),
        admitted.len(),
        bytes,
        elapsed,
        (read1.saturating_sub(read0), write1.saturating_sub(write0)),
    );
}

/// The historical harness composition, kept only to produce the "before"
/// number in the same process and the same conditions as the "after".
#[cfg(any(unix, windows))]
fn double(store_root: &Path, native: &Path) {
    let store = ObjectStore::open(store_root).expect("store opens");

    let (read0, write0) = io_counters();
    let start = Instant::now();
    let bytes = std::fs::read(native).expect("source reads");
    let hashed = plan_and_hash(&Slice(&bytes)).expect("source plans and hashes");
    let mut offset = 0_usize;
    let mut admitted_bytes = 0_u64;
    for object in hashed.objects() {
        let length = usize::try_from(object.length()).expect("length fits");
        store
            .put_bytes(&bytes[offset..offset + length])
            .expect("object admits");
        admitted_bytes += object.length();
        offset += length;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let (read1, write1) = io_counters();

    report(
        "double-hash",
        1,
        hashed.objects().len(),
        admitted_bytes,
        elapsed,
        (read1.saturating_sub(read0), write1.saturating_sub(write0)),
    );
}

#[cfg(any(unix, windows))]
fn main() {
    let args: Vec<String> = env::args().collect();
    println!("LOAD_PRE={:.2}", one_minute_load());
    match args.get(1).map(String::as_str) {
        Some("single") => single(Path::new(&args[2]), Path::new(&args[3])),
        Some("double") => double(Path::new(&args[2]), Path::new(&args[3])),
        _ => {
            eprintln!("usage: ingest_bench single <store_root> <native_path>");
            eprintln!("       ingest_bench double <store_root> <native_path>");
            std::process::exit(2);
        }
    }
    println!("LOAD_POST={:.2}", one_minute_load());
}

#[cfg(not(any(unix, windows)))]
fn main() {
    eprintln!("ingest_bench: the write arms need a filesystem target");
}
