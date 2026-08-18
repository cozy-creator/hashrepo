//! #115: the streamed store→memory read engine, buffered and O_DIRECT.
//!
//! Byte-equality across the two arms is the whole claim: `Direct` changes how
//! bytes travel (page cache bypassed, alignment handled), never which bytes
//! arrive — including a multi-object (>64 MiB) tensor and a `Hole`-bearing
//! sparse file, which zero-fills and never skips.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::store::ObjectStore;
use tensorfs_core::stream::{ReadMode, StreamReader};
use tensorfs_core::tfm1::FileRecord;

const MIB: usize = 1024 * 1024;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-stream-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// One safetensors file: a 96 MiB tensor (two objects: 64 + 32 MiB) and a
/// small one, with cheap deterministic content.
fn fixture() -> Vec<u8> {
    let big = 96 * MIB;
    let small = 1024;
    let header = format!(
        "{{\"big\":{{\"dtype\":\"U8\",\"shape\":[{big}],\"data_offsets\":[0,{big}]}},\
         \"small\":{{\"dtype\":\"U8\",\"shape\":[{small}],\"data_offsets\":[{big},{}]}}}}",
        big + small
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    let start = file.len();
    file.resize(start + big + small, 0);
    for (index, chunk) in file[start..].chunks_mut(8).enumerate() {
        let word = (index as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(97);
        chunk.copy_from_slice(&word.to_le_bytes()[..chunk.len()]);
    }
    file
}

/// Ingests the fixture and punches one 32 MiB hole: the big tensor's second
/// object is replaced by a hole of the same length. Returns the records and
/// the EXPECTED logical bytes (that range zeroed).
fn committed(store: &ObjectStore) -> (Vec<FileRecord>, Vec<u8>, u64) {
    let mut bytes = fixture();
    let planned = tensorfs_core::planner::plan(bytes.as_slice()).expect("plans");
    let admitted = store
        .admit_regions(bytes.as_slice(), planned.regions())
        .expect("admits");
    assert!(admitted.len() >= 4, "header, 64 MiB, 32 MiB, small");

    let mut records: Vec<FileRecord> = admitted
        .iter()
        .map(|object| FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        })
        .collect();

    // The big tensor's second object is records[2]: header, 64 MiB, 32 MiB.
    let hole_start: u64 = records[..2].iter().map(record_length).sum();
    let hole_length = record_length(&records[2]);
    assert_eq!(hole_length, 32 * MIB as u64);
    records[2] = FileRecord::Hole {
        length: hole_length,
    };
    let from = hole_start as usize;
    bytes[from..from + hole_length as usize].fill(0);
    (records, bytes, hole_start)
}

const fn record_length(record: &FileRecord) -> u64 {
    match record {
        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
    }
}

fn read(reader: &StreamReader<&ObjectStore>, offset: u64, length: usize) -> Vec<u8> {
    let mut buffer = vec![0_u8; length];
    reader
        .read_exact_at(offset, &mut buffer)
        .expect("range reads");
    buffer
}

#[test]
fn buffered_and_direct_arms_return_identical_bytes_holes_included() {
    let root = TempRoot::new("equality");
    let store = ObjectStore::open(root.path()).expect("store opens");
    let (records, expected, hole_start) = committed(&store);

    let buffered = StreamReader::new(&store, &records, ReadMode::Buffered).expect("opens");
    assert_eq!(buffered.length(), expected.len() as u64);

    // Ranges that cross every seam: whole file, object boundary, hole
    // boundary (zero-fill, never skip), unaligned slivers, the tail.
    let probes: Vec<(u64, usize)> = vec![
        (0, expected.len()),
        (0, 8 + 100),
        (hole_start - 3, 4096 + 7),
        (hole_start + MIB as u64 + 13, 5),
        (expected.len() as u64 - 1500, 1500),
    ];

    for (offset, length) in &probes {
        assert_eq!(
            read(&buffered, *offset, *length),
            expected[*offset as usize..*offset as usize + *length],
            "buffered [{offset}, +{length})"
        );
    }

    // Every hole byte is a zero, and it was never skipped: the read after
    // the hole is the source's own bytes again.
    let over_hole = read(&buffered, hole_start, 32 * MIB);
    assert!(over_hole.iter().all(|byte| *byte == 0));

    #[cfg(target_os = "linux")]
    {
        let direct = StreamReader::new(&store, &records, ReadMode::Direct).expect("opens");
        for (offset, length) in &probes {
            assert_eq!(
                read(&direct, *offset, *length),
                read(&buffered, *offset, *length),
                "direct == buffered at [{offset}, +{length})"
            );
        }
        // An aligned destination exercises the straight-into-the-buffer fast
        // path over the multi-object tensor; equality is the proof it landed.
        let tensor_start = 8 + header_len(&expected);
        let mut aligned = AlignedVec::new(96 * MIB);
        direct
            .read_exact_at(tensor_start as u64, aligned.slice_mut())
            .expect("aligned read");
        assert_eq!(
            aligned.slice_mut(),
            &expected[tensor_start..tensor_start + 96 * MIB]
        );
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Direct is a Linux capability; elsewhere it refuses at construction.
        assert!(StreamReader::new(&store, &records, ReadMode::Direct).is_err());
    }

    // Past-the-end refuses, never truncates.
    let mut sliver = [0_u8; 2];
    assert!(
        buffered
            .read_exact_at(expected.len() as u64 - 1, &mut sliver)
            .is_err()
    );
}

fn header_len(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize
}

/// A page-aligned destination, so the test can drive the fast path.
#[cfg(target_os = "linux")]
struct AlignedVec {
    raw: Vec<u8>,
    start: usize,
    length: usize,
}

#[cfg(target_os = "linux")]
impl AlignedVec {
    fn new(length: usize) -> Self {
        let raw = vec![0_u8; length + 4096];
        let shift = raw.as_ptr() as usize % 4096;
        let start = (4096 - shift) % 4096;
        Self { raw, start, length }
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.length]
    }
}
