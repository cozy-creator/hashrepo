//! Fixture builder and write-arm driver for the load-controlled pod benchmark.
//!
//! Subcommands:
//!   setup <store_root> <native_path> <size_mib>
//!       Writes the native fixture, ingests the identical bytes into the CAS,
//!       commits and seals a snapshot. Prints the snapshot id and every
//!       backing object path so the reader can evict them for a true-cold run.
//!
//!   direct-ingest <store_root> <native_path>
//!       Times the bypass write lane: read the file, plan, hash and admit
//!       every object into the CAS with no mount in the path. Reports
//!       throughput and this process's own read/write byte counters.
//!
//! The read arms live in bench.py; only the write arms need library access.

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::{ByteSource, PlannerId};
use tensorfs_core::tfm1::FileRecord;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

struct Slice<'a>(&'a [u8]);

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
/// The amplification claim is a ratio of counters, not of stopwatches.
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

/// Deterministic, poorly-compressible payload so no layer can cheat by
/// collapsing runs of identical bytes.
fn payload(size_mib: usize) -> Vec<u8> {
    let block: Vec<u8> = (0..(1usize << 20))
        .map(|i| ((i * 2_654_435_761) >> 13) as u8)
        .collect();
    let mut out = Vec::with_capacity(size_mib << 20);
    for round in 0..size_mib {
        let mut chunk = block.clone();
        chunk[0] = round as u8;
        out.extend_from_slice(&chunk);
    }
    out
}

fn object_path(store_root: &Path, digest: &str) -> PathBuf {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    store_root
        .join("objects/sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(hex)
}

fn setup(store_root: &Path, native: &Path, size_mib: usize) {
    let bytes = payload(size_mib);
    std::fs::write(native, &bytes).expect("native fixture written");

    let meta = WorkspaceStore::open(store_root).expect("store opens");
    meta.create_workspace("main").expect("workspace created");

    let hashed = plan_and_hash(&Slice(&bytes)).expect("fixture plans");
    let mut records = Vec::new();
    let mut paths = Vec::new();
    let mut offset = 0_usize;
    for object in hashed.objects() {
        let length = usize::try_from(object.length()).expect("length fits");
        let admitted = meta
            .store()
            .put_bytes(&bytes[offset..offset + length])
            .expect("object admits");
        assert_eq!(admitted.digest(), object.digest(), "admission is exact");
        paths.push(object_path(store_root, &object.digest().to_string()));
        records.push(FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        });
        offset += length;
    }
    assert_eq!(offset, bytes.len(), "records cover the file");

    meta.commit_generation(
        "main",
        &[Mutation::CreateFile {
            path: "model.bin".to_owned(),
            executable: false,
            planner: PlannerId::RawFixed64mV1,
            records,
        }],
    )
    .expect("generation commits");
    let id = meta.seal_snapshot("main", None).expect("snapshot seals");

    println!("SNAPSHOT={id}");
    println!("SIZE_BYTES={}", bytes.len());
    for path in &paths {
        println!("OBJ={}", path.display());
    }
}

fn direct_ingest(store_root: &Path, native: &Path) {
    let meta = WorkspaceStore::open(store_root).expect("store opens");
    let (read0, write0) = io_counters();
    let start = Instant::now();

    let bytes = std::fs::read(native).expect("source reads");
    let hashed = plan_and_hash(&Slice(&bytes)).expect("plans");
    let mut offset = 0_usize;
    let mut admitted_bytes = 0_u64;
    for object in hashed.objects() {
        let length = usize::try_from(object.length()).expect("length fits");
        meta.store()
            .put_bytes(&bytes[offset..offset + length])
            .expect("admits");
        admitted_bytes += object.length();
        offset += length;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let (read1, write1) = io_counters();
    println!("LOGICAL_BYTES={admitted_bytes}");
    println!("WALL_S={elapsed:.4}");
    println!("MIB_S={:.0}", admitted_bytes as f64 / elapsed / 1048576.0);
    println!("PROC_READ_BYTES={}", read1.saturating_sub(read0));
    println!("PROC_WRITE_BYTES={}", write1.saturating_sub(write0));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("setup") => setup(
            Path::new(&args[2]),
            Path::new(&args[3]),
            args[4].parse().expect("size_mib"),
        ),
        Some("direct-ingest") => direct_ingest(Path::new(&args[2]), Path::new(&args[3])),
        _ => {
            eprintln!("usage: podbench setup <store> <native> <size_mib>");
            eprintln!("       podbench direct-ingest <store> <native>");
            std::process::exit(2);
        }
    }
}
