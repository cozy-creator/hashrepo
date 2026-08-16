//! Concurrency proofs for `ObjectStore::admit_regions`.
//!
//! Every assertion here is on identity, ordering or file count. There is no
//! wall-clock assertion anywhere: the development box swings more than tenfold
//! with load, so a timing gate would rot into a flake. Throughput lives in
//! `examples/ingest_bench.rs`, which reports rather than asserts.

#![cfg(any(unix, windows))]

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::{ByteSource, plan};
use tensorfs_core::store::{ObjectStore, ingest_concurrency};

struct Slice<'a>(&'a [u8]);

impl ByteSource for Slice<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).expect("offset fits");
        destination.copy_from_slice(&self.0[start..start + destination.len()]);
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

/// A source that refuses one chosen region, so failure ordering is testable
/// without a filesystem fault. `reads` counts how many reads were served, which
/// is how "the other workers still ran" is observed.
struct FailingAt<'a> {
    inner: Slice<'a>,
    fail_from: u64,
    reads: AtomicUsize,
}

impl ByteSource for FailingAt<'_> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if offset == self.fail_from {
            return Err(io::Error::other("planted read failure"));
        }
        self.inner.read_exact_at(offset, destination)
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

/// A deterministic safetensors file with many small tensors, so the planner
/// yields many independent regions without a gigabyte of payload. Sixteen
/// small objects exercise the same fan-out as sixteen 64 MiB ones.
fn safetensors(tensors: &[(&str, usize, u8)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_usize;
    for (index, (name, length, _)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{name}":{{"dtype":"U8","shape":[{length}],"data_offsets":[{offset},{}]}}"#,
            offset + length
        ));
        offset += length;
    }
    header.push('}');

    let mut file = Vec::with_capacity(8 + header.len() + offset);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(*fill, *length));
    }
    file
}

fn many_tensor_fixture() -> Vec<u8> {
    let tensors: Vec<(String, usize, u8)> = (0..16)
        .map(|index| (format!("blk{index}.weight"), 64 * 1024 + index, index as u8))
        .collect();
    let borrowed: Vec<(&str, usize, u8)> = tensors
        .iter()
        .map(|(name, length, fill)| (name.as_str(), *length, *fill))
        .collect();
    safetensors(&borrowed)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tensorfs-concurrent-{}-{name}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("scratch root");
    path
}

#[test]
fn concurrent_admission_reproduces_the_serial_digests_in_region_order() {
    let bytes = many_tensor_fixture();
    let root = scratch("order");
    let store = ObjectStore::open(&root).expect("store opens");

    let planned = plan(&Slice(&bytes)).expect("fixture plans");
    assert!(
        planned.regions().len() >= 8,
        "the fixture must actually fan out: {} regions",
        planned.regions().len()
    );

    let admitted = store
        .admit_regions(&Slice(&bytes), planned.regions())
        .expect("regions admit");

    // The serial hasher is the oracle: concurrency may not change one digest.
    let serial = plan_and_hash(&Slice(&bytes)).expect("fixture plans and hashes");
    assert_eq!(
        admitted.len(),
        serial.objects().len(),
        "one admitted object per planned region"
    );
    for (index, (got, want)) in admitted.iter().zip(serial.objects()).enumerate() {
        assert_eq!(
            got.digest(),
            want.digest(),
            "region {index} digest must match the serial plan"
        );
        assert_eq!(got.length(), want.length(), "region {index} length");
    }

    // Every admitted object is resident and rehashes to its own name.
    for object in &admitted {
        let verified = store.verify(&object.digest()).expect("object verifies");
        assert_eq!(verified, object.length(), "verified length");
    }

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn racing_writers_of_one_digest_all_succeed_and_install_one_file() {
    let root = scratch("race");
    let store = Arc::new(ObjectStore::open(&root).expect("store opens"));
    let payload = Arc::new(vec![0xA5_u8; 3 * 1024 * 1024]);

    let workers = 8;
    let digests: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let store = Arc::clone(&store);
                let payload = Arc::clone(&payload);
                scope.spawn(move || {
                    store
                        .put_bytes(&payload)
                        .expect("racing admission succeeds")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker joins"))
            .collect()
    });

    let first = digests[0].digest();
    for admitted in &digests {
        assert_eq!(
            admitted.digest(),
            first,
            "identical bytes must converge on one digest"
        );
    }

    // No-clobber install means exactly one file exists at that digest path,
    // and no temp survived the race.
    let verified = store.verify(&first).expect("the single object verifies");
    assert_eq!(verified, payload.len() as u64);
    let temps = std::fs::read_dir(root.join("tmp"))
        .expect("tmp dir")
        .filter_map(Result::ok)
        .count();
    assert_eq!(temps, 0, "a racing admission must leave no temp behind");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_failing_region_reports_in_region_order_not_completion_order() {
    let bytes = many_tensor_fixture();
    let root = scratch("failorder");
    let store = ObjectStore::open(&root).expect("store opens");
    let planned = plan(&Slice(&bytes)).expect("fixture plans");

    // Fail an early region. Later regions are still attempted concurrently,
    // so the reported error must be chosen by region index, not by whichever
    // worker happened to finish first.
    let target = planned.regions()[1].offset();
    let source = FailingAt {
        inner: Slice(&bytes),
        fail_from: target,
        reads: AtomicUsize::new(0),
    };

    let error = store
        .admit_regions(&source, planned.regions())
        .expect_err("the planted failure must surface");
    assert!(
        format!("{error}").contains("I/O"),
        "the read failure must surface as a store I/O error, got: {error}"
    );
    assert!(
        source.reads.load(Ordering::Relaxed) > 1,
        "other regions must still have been attempted concurrently"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn admitting_no_regions_touches_nothing() {
    let root = scratch("empty");
    let store = ObjectStore::open(&root).expect("store opens");
    let admitted = store
        .admit_regions(&Slice(b""), &[])
        .expect("an empty region list is not an error");
    assert!(admitted.is_empty(), "no regions, no objects");
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn concurrency_is_at_least_one_and_never_exceeds_the_core_count() {
    let workers = ingest_concurrency();
    assert!(workers >= 1, "there is always at least one worker");
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    assert!(
        workers <= cores,
        "SHA-256 is compute-bound: {workers} workers on {cores} cores would only add switches"
    );
}
