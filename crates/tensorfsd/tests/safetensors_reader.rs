//! The product claim, proved with the real library: `safetensors.safe_open`
//! opens a MOUNTED PATH directly — no TensorFS wrapper, no byte API, no
//! export step — and gets byte-identical tensors back.
//!
//! The reader is a genuine separate `safetensors` process (the same
//! `python3`-helper pattern `mmap_locks.rs` uses); it mmaps the file it is
//! handed, so every tensor it returns was faulted in through FUSE by the
//! spawned daemon. The file itself is produced by an ordinary `std::fs`
//! writer through the same mount.
//!
//! The seal cycle is the point: written live, the file commits under the raw
//! grid as one object. `seal_snapshot` re-plans it through the tensor-aware
//! planner and re-boundaries it into a header object plus one object per
//! tensor. The library must still read exactly the same bytes afterwards —
//! from the read-only snapshot mount and from the re-boundaried workspace
//! mount alike. That property is why the planner exists.
//!
//! Prerequisites: `/dev/fuse`, `fusermount3`, and a `python3` that can
//! `import safetensors` and `numpy`. Each is named loudly on skip. The
//! reader is a prerequisite of THIS test, not of the `tensorfs` Python
//! package — it stays out of `pyproject.toml` (numpy's stubs need a newer
//! `python_version` than the package's typing gate targets) and out of the
//! project virtualenv. Give it its own:
//!
//! ```text
//! python3 -m venv /tmp/safetensors-reader
//! /tmp/safetensors-reader/bin/pip install safetensors numpy
//! TENSORFS_TEST_PYTHON=/tmp/safetensors-reader/bin/python3 cargo test -p tensorfsd
//! ```
//!
//! `numpy` is the framework because it is the only `safe_open` backend that
//! does not pull a multi-GB deep-learning runtime. The PyTorch and Diffusers
//! half of the claim is NOT proved here.

#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Mutex;
use std::time::Duration;

use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord};
use tensorfs_core::workspace::WorkspaceStore;
use tensorfsd::mount_snapshot;

static MOUNT_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    MOUNT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const MODEL: &str = "model.safetensors";

/// The interpreter that must carry `safetensors`, nameable because it is
/// deliberately not the project's own virtualenv.
fn python() -> String {
    std::env::var("TENSORFS_TEST_PYTHON").unwrap_or_else(|_| "python3".to_owned())
}

/// Loud, prerequisite-naming skip: a silently passing test here would assert
/// nothing at all about the claim it exists to prove.
fn prerequisites_available() -> bool {
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
    let interpreter = python();
    match process::Command::new(&interpreter)
        .args(["-c", "import safetensors, numpy"])
        .output()
    {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            eprintln!(
                "skipping: `{interpreter}` cannot import safetensors and numpy \
                 (install them, or set TENSORFS_TEST_PYTHON): {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
            false
        }
        Err(error) => {
            eprintln!("skipping: `{interpreter}` is not runnable: {error}");
            false
        }
    }
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

/// The daemon that serves a workspace mount, spawned as its own process: a
/// process serving a mount must never read through it.
struct BinMount {
    child: process::Child,
    mountpoint: PathBuf,
    done: bool,
}

impl Drop for BinMount {
    fn drop(&mut self) {
        if self.done {
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
            .stdout(process::Stdio::null())
            .stderr(process::Stdio::null())
            .spawn()
            .expect("daemon spawns");
        for _ in 0..100 {
            if mounted_here(mountpoint) {
                return Self {
                    child,
                    mountpoint: mountpoint.to_path_buf(),
                    done: false,
                };
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("daemon did not mount in time");
    }

    fn sigterm_and_wait(mut self) {
        self.done = true;
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

/// One tensor of the fixture: its declared safetensors dtype and shape, and
/// the exact body bytes an ordinary writer puts on the mount.
struct Tensor {
    name: &'static str,
    dtype: &'static str,
    numpy_dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// A small but structurally genuine checkpoint: several dtypes, a 2-D matrix
/// whose rows a lazy reader can slice, a 1-D bias, and `__metadata__`.
///
/// Small on purpose. The tensor-aware planner has no minimum region size, so
/// every tensor becomes its own object at a total cost of under 1 MiB of I/O;
/// nothing about the boundary claim needs a real checkpoint's gigabytes.
fn fixture_tensors() -> Vec<Tensor> {
    vec![
        Tensor {
            name: "block.0.weight",
            dtype: "F32",
            numpy_dtype: "float32",
            shape: vec![256, 512],
            bytes: pattern(11, 256 * 512 * 4),
        },
        Tensor {
            name: "block.0.bias",
            dtype: "F32",
            numpy_dtype: "float32",
            shape: vec![512],
            bytes: pattern(29, 512 * 4),
        },
        Tensor {
            name: "block.1.weight",
            dtype: "F16",
            numpy_dtype: "float16",
            shape: vec![384, 512],
            bytes: pattern(47, 384 * 512 * 2),
        },
        Tensor {
            name: "embedding.ids",
            dtype: "I64",
            numpy_dtype: "int64",
            shape: vec![1024],
            bytes: pattern(83, 1024 * 8),
        },
        Tensor {
            name: "quant.scales",
            dtype: "U8",
            numpy_dtype: "uint8",
            shape: vec![65536],
            bytes: pattern(151, 65536),
        },
    ]
}

/// The tensor whose rows the reader also fetches lazily, and the row window
/// it takes — a partial, offset read of a single tensor's object.
const SLICED: &str = "block.0.weight";
const SLICE_ROWS: std::ops::Range<usize> = 1..3;
const SLICE_ROW_BYTES: usize = 512 * 4;

/// Serializes the fixture into real safetensors container bytes and reports
/// each tensor's byte span within the file, so the re-boundaried records can
/// be checked against the tensor grid the library itself sees.
fn encode(tensors: &[Tensor]) -> (Vec<u8>, u64, Vec<(String, u64)>) {
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({ "format": "pt", "producer": "tensorfs-integration-test" }),
    );
    let mut cursor = 0_u64;
    let mut spans = Vec::new();
    for tensor in tensors {
        let length = tensor.bytes.len() as u64;
        header.insert(
            tensor.name.to_owned(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, cursor + length],
            }),
        );
        spans.push((tensor.name.to_owned(), length));
        cursor += length;
    }

    let mut encoded =
        serde_json::to_vec(&serde_json::Value::Object(header)).expect("header serializes");
    // The eight-byte alignment convention every real writer follows; the pad
    // lives inside the declared header length.
    encoded.resize(encoded.len().next_multiple_of(8), b' ');

    let mut file = Vec::with_capacity(8 + encoded.len() + cursor as usize);
    file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    file.extend_from_slice(&encoded);
    for tensor in tensors {
        file.extend_from_slice(&tensor.bytes);
    }
    (file, 8 + encoded.len() as u64, spans)
}

/// The reader: a real `safetensors` process that is handed nothing but a
/// filesystem path. It mmaps that path, materializes every tensor and one
/// lazy row slice, and writes the raw bytes back out for comparison.
const READER: &str = r#"
import json, sys
from safetensors import safe_open

path, blob_path, slice_path, sliced, lo, hi = sys.argv[1:7]
rows = []
with safe_open(path, framework="numpy") as handle:
    metadata = handle.metadata()
    with open(blob_path, "wb") as blob:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            raw = tensor.tobytes()
            blob.write(raw)
            rows.append({
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "length": len(raw),
            })
    view = handle.get_slice(sliced)
    window = view[int(lo):int(hi)]
    with open(slice_path, "wb") as blob:
        blob.write(window.tobytes())
    slice_report = {"dtype": view.get_dtype(), "shape": list(view.get_shape())}
print(json.dumps({"metadata": metadata, "tensors": rows, "slice": slice_report}))
"#;

/// What one reader run observed: every tensor's bytes keyed by name, plus the
/// dtype/shape the library reported and the lazy slice it fetched.
struct Observed {
    tensors: BTreeMap<String, (String, Vec<u64>, Vec<u8>)>,
    slice: Vec<u8>,
    slice_dtype: String,
    slice_shape: Vec<u64>,
    metadata: BTreeMap<String, String>,
}

/// Runs the reader against `path` — which is always inside a mount — with its
/// scratch output landing outside it.
fn read_with_safetensors(path: &Path, scratch: &Path) -> Observed {
    let blob = scratch.join("tensors.bin");
    let slice_blob = scratch.join("slice.bin");
    let output = process::Command::new(python())
        .arg("-c")
        .arg(READER)
        .arg(path)
        .arg(&blob)
        .arg(&slice_blob)
        .arg(SLICED)
        .arg(SLICE_ROWS.start.to_string())
        .arg(SLICE_ROWS.end.to_string())
        .output()
        .expect("the safetensors reader runs");
    assert!(
        output.status.success(),
        "safe_open failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("the reader's report parses");
    let bytes = fs::read(&blob).expect("the reader's tensor blob reads");
    let mut tensors = BTreeMap::new();
    let mut cursor = 0_usize;
    for row in report["tensors"]
        .as_array()
        .expect("the report lists tensors")
    {
        let name = row["name"].as_str().expect("a tensor name").to_owned();
        let dtype = row["dtype"].as_str().expect("a numpy dtype").to_owned();
        let shape = row["shape"]
            .as_array()
            .expect("a shape")
            .iter()
            .map(|value| value.as_u64().expect("a dimension"))
            .collect();
        let length = row["length"].as_u64().expect("a byte length") as usize;
        tensors.insert(
            name,
            (dtype, shape, bytes[cursor..cursor + length].to_vec()),
        );
        cursor += length;
    }
    assert_eq!(cursor, bytes.len(), "the blob holds exactly the tensors");

    let metadata = report["metadata"]
        .as_object()
        .expect("the header carries __metadata__")
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                value
                    .as_str()
                    .expect("metadata values are strings")
                    .to_owned(),
            )
        })
        .collect();

    Observed {
        tensors,
        slice: fs::read(&slice_blob).expect("the reader's slice blob reads"),
        slice_dtype: report["slice"]["dtype"]
            .as_str()
            .expect("a slice dtype")
            .to_owned(),
        slice_shape: report["slice"]["shape"]
            .as_array()
            .expect("a slice shape")
            .iter()
            .map(|value| value.as_u64().expect("a dimension"))
            .collect(),
        metadata,
    }
}

/// Asserts the reader saw exactly the fixture: every tensor byte-for-byte,
/// with the dtype and shape the writer declared.
fn assert_matches_fixture(observed: &Observed, tensors: &[Tensor], stage: &str) {
    assert_eq!(
        observed.tensors.len(),
        tensors.len(),
        "{stage}: safe_open enumerated every tensor"
    );
    assert_eq!(
        observed.metadata.get("producer").map(String::as_str),
        Some("tensorfs-integration-test"),
        "{stage}: __metadata__ survives"
    );
    for tensor in tensors {
        let (dtype, shape, bytes) = observed
            .tensors
            .get(tensor.name)
            .unwrap_or_else(|| panic!("{stage}: safe_open returned {}", tensor.name));
        assert_eq!(dtype, tensor.numpy_dtype, "{stage}: {} dtype", tensor.name);
        assert_eq!(shape, &tensor.shape, "{stage}: {} shape", tensor.name);
        assert_eq!(
            bytes.len(),
            tensor.bytes.len(),
            "{stage}: {} byte length",
            tensor.name
        );
        assert!(
            bytes == &tensor.bytes,
            "{stage}: {} bytes differ from what the ordinary writer wrote",
            tensor.name
        );
    }

    let sliced = tensors
        .iter()
        .find(|tensor| tensor.name == SLICED)
        .expect("the sliced tensor is in the fixture");
    assert_eq!(observed.slice_dtype, sliced.dtype, "{stage}: slice dtype");
    assert_eq!(observed.slice_shape, sliced.shape, "{stage}: slice shape");
    let start = SLICE_ROWS.start * SLICE_ROW_BYTES;
    let end = SLICE_ROWS.end * SLICE_ROW_BYTES;
    assert!(
        observed.slice == sliced.bytes[start..end],
        "{stage}: the lazy row slice differs from the written rows"
    );
}

/// The committed shape of `MODEL`: its planner and the length of each data
/// record, which is the object grid a reader's page faults are served from.
fn committed_grid(root: &Path) -> (PlannerId, Vec<u64>) {
    let store = WorkspaceStore::open(root).expect("store reopens");
    let tree = store.head_tree("main").expect("head tree builds");
    for (path, entry) in tree.entries() {
        if path == MODEL
            && let Entry::File {
                planner, records, ..
            } = entry
        {
            return (
                *planner,
                records
                    .iter()
                    .map(|record| match record {
                        FileRecord::Data { length, .. } => *length,
                        FileRecord::Hole { length } => *length,
                    })
                    .collect(),
            );
        }
    }
    panic!("{MODEL} is not a committed file");
}

#[test]
fn a_real_safetensors_reader_opens_a_mounted_path_across_the_seal_reboundary() {
    let _serial = serial();
    if !prerequisites_available() {
        return;
    }
    let root = unique_dir("st-root");
    let mountpoint = unique_dir("st-mnt");
    let snap_mountpoint = unique_dir("st-snap-mnt");
    let scratch = unique_dir("st-scratch");
    {
        let store = WorkspaceStore::open(&root).expect("store opens");
        store.create_workspace("main").expect("workspace creates");
    }

    let tensors = fixture_tensors();
    let (encoded, header_end, spans) = encode(&tensors);

    // Arm one: an ordinary writer puts a real checkpoint on the mount with
    // nothing but `std::fs`, and a real safetensors process opens that path.
    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let model = mountpoint.join(MODEL);
    {
        let mut handle = fs::File::create(&model).expect("create works through the mount");
        handle.write_all(&encoded).expect("write works");
        handle.sync_all().expect("fsync works");
    }
    let live = read_with_safetensors(&model, &scratch);
    assert_matches_fixture(&live, &tensors, "live workspace mount");
    daemon.sigterm_and_wait();

    // Written live, the file carries the raw grid: one object under the
    // 64 MiB ceiling, with no idea a tensor is a tensor.
    let (planner, grid) = committed_grid(&root);
    assert_eq!(
        planner,
        PlannerId::RawFixed64mV1,
        "a live write commits under the raw grid"
    );
    assert_eq!(
        grid,
        vec![encoded.len() as u64],
        "the raw grid is one whole-file object"
    );

    // Sealing re-plans through the tensor-aware planner: the header and each
    // tensor become their own object.
    let snapshot = {
        let store = WorkspaceStore::open(&root).expect("store reopens");
        store.seal_snapshot("main", None).expect("snapshot seals")
    };
    let (planner, grid) = committed_grid(&root);
    assert_eq!(
        planner,
        PlannerId::SafetensorsV1,
        "sealing recovers the tensor-aware planner"
    );
    let mut expected_grid = vec![header_end];
    expected_grid.extend(spans.iter().map(|(_, length)| *length));
    assert_eq!(
        grid, expected_grid,
        "every object boundary lands exactly on a tensor boundary the library declared"
    );

    // Arm two: the same real reader, the same path shape, now served from a
    // read-only snapshot mount whose bytes are composed out of one object per
    // tensor rather than one object per file.
    let snapshot_mount =
        mount_snapshot(&root, &snapshot, &snap_mountpoint).expect("snapshot mounts");
    let sealed = read_with_safetensors(&snap_mountpoint.join(MODEL), &scratch);
    assert_matches_fixture(&sealed, &tensors, "sealed snapshot mount");
    snapshot_mount.unmount();
    assert!(!mounted_here(&snap_mountpoint));

    // Arm three: the writable mount serves the re-boundaried records too, so
    // the seal did not quietly move the working tree off the library's path.
    let daemon = BinMount::spawn(&root, "main", &mountpoint);
    let reboundaried = read_with_safetensors(&model, &scratch);
    assert_matches_fixture(&reboundaried, &tensors, "re-boundaried workspace mount");
    daemon.sigterm_and_wait();

    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
    fs::remove_dir_all(&snap_mountpoint).ok();
    fs::remove_dir_all(&scratch).ok();
}
