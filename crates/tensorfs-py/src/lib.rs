//! `tensorfs._tensorfs` — the compiled half of the `tensorfs` distribution.
//!
//! Three surfaces, chosen because they are the ones a direct tensor read
//! needs and the ones Python cannot do at native speed:
//!
//! 1. **The CAS** (`ObjectStore`): admit, read, verify. Hashing and the
//!    crash-safe install stay in Rust.
//! 2. **The formats** (`decode_snapshot`, `decode_pack` / `encode_pack`): one
//!    decoder for TFM1 and TFP1, so Python never re-implements a parser whose
//!    refusals are conformance-tested in Rust.
//! 3. **The byte-source path** (`RecordsReader`): a committed file's ordered
//!    records read as a bounded random-access byte source, plus the canonical
//!    planner over that same source. Together they are exactly "give me the
//!    bytes of one tensor without materializing the file".
//!
//! `#![forbid(unsafe_code)]` is deliberately absent here, unlike every other
//! crate in this workspace: the PyO3 attribute macros expand to `unsafe impl`s
//! in this crate. No hand-written `unsafe` appears below.

use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tensorfs_core::layout::{self, Layout, STUB_MAGIC};
use tensorfs_core::object::{self, ObjectDigest};
use tensorfs_core::planner::{self, ByteSource, PlannerId, RegionKind};
use tensorfs_core::source::FileByteSource;
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{self, Entry, FileRecord as CoreFileRecord, SnapshotId};
use tensorfs_core::tfp1;
use tensorfs_core::workspace_source::RecordsSource;

create_exception!(
    "tensorfs._tensorfs",
    TensorfsError,
    PyException,
    "Base class for every error raised by the compiled TensorFS extension."
);
create_exception!(
    "tensorfs._tensorfs",
    StoreError,
    TensorfsError,
    "The object store refused an admission, a read, or a verification."
);
create_exception!(
    "tensorfs._tensorfs",
    FormatError,
    TensorfsError,
    "A TFM1 or TFP1 byte string is not a valid canonical encoding."
);
create_exception!(
    "tensorfs._tensorfs",
    PlanError,
    TensorfsError,
    "The canonical planner refused a source."
);
create_exception!(
    "tensorfs._tensorfs",
    LayoutError,
    TensorfsError,
    "A snapshot tree or ref could not be projected, read, or removed."
);

/// Flattens an error and its whole `source()` chain into one message. The
/// chain carries the actionable half: "object store read failed" alone names
/// neither the path nor the errno.
fn describe(error: &dyn Error) -> String {
    let mut text = error.to_string();
    let mut next = error.source();
    while let Some(inner) = next {
        text.push_str(": ");
        text.push_str(&inner.to_string());
        next = inner.source();
    }
    text
}

fn store_error(error: tensorfs_core::store::StoreError) -> PyErr {
    StoreError::new_err(describe(&error))
}

fn layout_error(error: layout::LayoutError) -> PyErr {
    LayoutError::new_err(describe(&error))
}

fn plan_error(error: planner::PlanError) -> PyErr {
    PlanError::new_err(describe(&error))
}

fn tfm1_error(error: tfm1::Tfm1Error) -> PyErr {
    FormatError::new_err(describe(&error))
}

fn tfp1_error(error: tfp1::Tfp1Error) -> PyErr {
    FormatError::new_err(describe(&error))
}

fn io_error(error: std::io::Error) -> PyErr {
    PyErr::from(error)
}

/// Digests cross this boundary as bare lowercase 64-character hex, which is
/// exactly `tensorfs.CASRef.digest` and exactly the store's on-disk name. The
/// algorithm-tagged spelling `sha256:<hex>` — `str(CASRef)`, and the Rust
/// `Display` — is accepted on input and never produced on output.
fn hex_of(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn parse_hex32(text: &str) -> PyResult<[u8; 32]> {
    let digits = text.strip_prefix("sha256:").unwrap_or(text);
    if digits.len() != 64 {
        return Err(TensorfsError::new_err(format!(
            "digest must be 64 hexadecimal characters, got {}",
            digits.len()
        )));
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in digits.as_bytes().chunks_exact(2).enumerate() {
        let pair = std::str::from_utf8(pair)
            .map_err(|_| TensorfsError::new_err("digest is not ASCII hexadecimal"))?;
        if pair.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return Err(TensorfsError::new_err(
                "digest must be lowercase hexadecimal",
            ));
        }
        bytes[index] = u8::from_str_radix(pair, 16)
            .map_err(|_| TensorfsError::new_err("digest is not ASCII hexadecimal"))?;
    }
    Ok(bytes)
}

fn parse_digest(text: &str) -> PyResult<ObjectDigest> {
    parse_hex32(text).map(ObjectDigest::from_bytes)
}

const fn region_kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Header => "header",
        RegionKind::Tensor => "tensor",
        RegionKind::Blob => "blob",
    }
}

const fn planner_name(planner: PlannerId) -> &'static str {
    planner.as_str()
}

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

/// One object as the store admitted it.
#[pyclass(
    frozen,
    get_all,
    module = "tensorfs._tensorfs",
    name = "AdmittedObject"
)]
pub struct PyAdmittedObject {
    /// Bare lowercase 64-character hex.
    digest: String,
    length: u64,
    /// True when an identical object was already resident and was re-verified
    /// rather than rewritten.
    preexisting: bool,
}

#[pymethods]
impl PyAdmittedObject {
    fn __repr__(&self) -> String {
        format!(
            "AdmittedObject(digest='{}', length={}, preexisting={})",
            self.digest,
            self.length,
            if self.preexisting { "True" } else { "False" }
        )
    }
}

/// The result of one abandoned-temporary sweep.
#[pyclass(
    frozen,
    get_all,
    module = "tensorfs._tensorfs",
    name = "TempCollection"
)]
pub struct PyTempCollection {
    examined: u64,
    deleted: u64,
    bytes_deleted: u64,
}

#[pymethods]
impl PyTempCollection {
    fn __repr__(&self) -> String {
        format!(
            "TempCollection(examined={}, deleted={}, bytes_deleted={})",
            self.examined, self.deleted, self.bytes_deleted
        )
    }
}

/// One planned byte range of a source, before hashing.
#[pyclass(frozen, get_all, module = "tensorfs._tensorfs", name = "Region")]
pub struct PyRegion {
    offset: u64,
    length: u64,
    /// `"header"`, `"tensor"` or `"blob"`.
    kind: String,
}

#[pymethods]
impl PyRegion {
    fn __repr__(&self) -> String {
        format!(
            "Region(offset={}, length={}, kind='{}')",
            self.offset, self.length, self.kind
        )
    }
}

/// One planned range together with the digest of its bytes.
#[pyclass(frozen, get_all, module = "tensorfs._tensorfs", name = "PlannedObject")]
pub struct PyPlannedObject {
    digest: String,
    length: u64,
    kind: String,
}

#[pymethods]
impl PyPlannedObject {
    fn __repr__(&self) -> String {
        format!(
            "PlannedObject(digest='{}', length={}, kind='{}')",
            self.digest, self.length, self.kind
        )
    }
}

/// The canonical planner's decomposition of one source.
///
/// For a safetensors or GGUF file the `"tensor"` regions ARE the tensor byte
/// ranges, subdivided at `MAX_OBJECT_SIZE`. That is what makes a direct tensor
/// read possible without a second parser in Python.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "Plan")]
pub struct PyPlan {
    planner: String,
    file_size: u64,
    regions: Vec<Py<PyRegion>>,
}

#[pymethods]
impl PyPlan {
    /// `"safetensors-v1"`, `"gguf-v1"` or `"blob-v1"`.
    #[getter]
    fn planner(&self) -> &str {
        &self.planner
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.file_size
    }

    #[getter]
    fn regions(&self, py: Python<'_>) -> Vec<Py<PyRegion>> {
        self.regions.iter().map(|item| item.clone_ref(py)).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Plan(planner='{}', file_size={}, regions={})",
            self.planner,
            self.file_size,
            self.regions.len()
        )
    }
}

fn plan_into_python(py: Python<'_>, plan: &planner::Plan) -> PyResult<PyPlan> {
    let regions = plan
        .regions()
        .iter()
        .map(|region| {
            Py::new(
                py,
                PyRegion {
                    offset: region.offset(),
                    length: region.length(),
                    kind: region_kind_name(region.kind()).to_owned(),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyPlan {
        planner: planner_name(plan.planner()).to_owned(),
        file_size: plan.file_size(),
        regions,
    })
}

/// A plan whose every object has been hashed from one stable source snapshot.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "HashedPlan")]
pub struct PyHashedPlan {
    planner: String,
    file_size: u64,
    objects: Vec<Py<PyPlannedObject>>,
}

#[pymethods]
impl PyHashedPlan {
    #[getter]
    fn planner(&self) -> &str {
        &self.planner
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.file_size
    }

    #[getter]
    fn objects(&self, py: Python<'_>) -> Vec<Py<PyPlannedObject>> {
        self.objects.iter().map(|item| item.clone_ref(py)).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "HashedPlan(planner='{}', file_size={}, objects={})",
            self.planner,
            self.file_size,
            self.objects.len()
        )
    }
}

fn hashed_plan_into_python(py: Python<'_>, plan: &object::HashedPlan) -> PyResult<PyHashedPlan> {
    let objects = plan
        .objects()
        .iter()
        .map(|planned| {
            let digest = planned.digest();
            Py::new(
                py,
                PyPlannedObject {
                    digest: hex_of(digest.as_bytes()),
                    length: planned.length(),
                    kind: region_kind_name(planned.kind()).to_owned(),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyHashedPlan {
        planner: planner_name(plan.planner()).to_owned(),
        file_size: plan.file_size(),
        objects,
    })
}

/// One record of a committed file: either a stored object or a hole.
// `from_py_object` is explicit rather than inherited from `Clone`: PyO3 is
// making that derive opt-in, and `RecordsReader.__new__` really does take a
// sequence of these by value.
#[pyclass(
    frozen,
    from_py_object,
    module = "tensorfs._tensorfs",
    name = "FileRecord"
)]
#[derive(Clone)]
pub struct PyFileRecord {
    kind: String,
    digest: Option<String>,
    length: u64,
}

#[pymethods]
impl PyFileRecord {
    #[new]
    #[pyo3(signature = (kind, length, digest = None))]
    fn new(kind: &str, length: u64, digest: Option<&str>) -> PyResult<Self> {
        match (kind, digest) {
            ("data", Some(digest)) => {
                let parsed = parse_digest(digest)?;
                Ok(Self {
                    kind: "data".to_owned(),
                    digest: Some(hex_of(parsed.as_bytes())),
                    length,
                })
            }
            ("data", None) => Err(TensorfsError::new_err("a data record needs a digest")),
            ("hole", None) => Ok(Self {
                kind: "hole".to_owned(),
                digest: None,
                length,
            }),
            ("hole", Some(_)) => Err(TensorfsError::new_err("a hole record has no digest")),
            (other, _) => Err(TensorfsError::new_err(format!(
                "record kind must be 'data' or 'hole', got '{other}'"
            ))),
        }
    }

    /// `"data"` or `"hole"`.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// The object digest for a `"data"` record; `None` for a hole.
    #[getter]
    fn digest(&self) -> Option<&str> {
        self.digest.as_deref()
    }

    #[getter]
    fn length(&self) -> u64 {
        self.length
    }

    #[staticmethod]
    fn data(digest: &str, length: u64) -> PyResult<Self> {
        Self::new("data", length, Some(digest))
    }

    #[staticmethod]
    fn hole(length: u64) -> PyResult<Self> {
        Self::new("hole", length, None)
    }

    fn __repr__(&self) -> String {
        match &self.digest {
            Some(digest) => format!(
                "FileRecord(kind='data', digest='{digest}', length={})",
                self.length
            ),
            None => format!("FileRecord(kind='hole', length={})", self.length),
        }
    }
}

impl PyFileRecord {
    fn to_core(&self) -> PyResult<CoreFileRecord> {
        match &self.digest {
            Some(digest) => Ok(CoreFileRecord::Data {
                digest: parse_digest(digest)?,
                length: self.length,
            }),
            None => Ok(CoreFileRecord::Hole {
                length: self.length,
            }),
        }
    }

    fn from_core(record: &CoreFileRecord) -> Self {
        match record {
            CoreFileRecord::Data { digest, length } => Self {
                kind: "data".to_owned(),
                digest: Some(hex_of(digest.as_bytes())),
                length: *length,
            },
            CoreFileRecord::Hole { length } => Self {
                kind: "hole".to_owned(),
                digest: None,
                length: *length,
            },
        }
    }
}

/// One path in a decoded snapshot.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "SnapshotEntry")]
pub struct PySnapshotEntry {
    path: String,
    kind: String,
    executable: Option<bool>,
    planner: Option<String>,
    logical_size: Option<u64>,
    records: Option<Vec<Py<PyFileRecord>>>,
    digest: Option<String>,
    body_sha256: Option<String>,
    target: Option<String>,
    ordinal: Option<u64>,
}

#[pymethods]
impl PySnapshotEntry {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// `"directory"`, `"file"`, `"symlink"` or `"hardlink"`.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    #[getter]
    fn executable(&self) -> Option<bool> {
        self.executable
    }

    #[getter]
    fn planner(&self) -> Option<&str> {
        self.planner.as_deref()
    }

    #[getter]
    fn logical_size(&self) -> Option<u64> {
        self.logical_size
    }

    #[getter]
    fn records(&self, py: Python<'_>) -> Option<Vec<Py<PyFileRecord>>> {
        self.records
            .as_ref()
            .map(|records| records.iter().map(|item| item.clone_ref(py)).collect())
    }

    /// The whole-file SHA-256 (bare hex) of a `blob-v1` entry, else `None`.
    /// Tensor entries have no whole-file hash — their identity is the record
    /// list.
    #[getter]
    fn digest(&self) -> Option<&str> {
        self.digest.as_deref()
    }

    /// A file entry's TFM1 body digest (bare hex): SHA-256 over the entry's
    /// canonical body encoding. This is the value a pointer stub carries, and
    /// the only file-level identity a tensor container has.
    #[getter]
    fn body_sha256(&self) -> Option<&str> {
        self.body_sha256.as_deref()
    }

    #[getter]
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    #[getter]
    fn ordinal(&self) -> Option<u64> {
        self.ordinal
    }

    fn __repr__(&self) -> String {
        format!("SnapshotEntry(path='{}', kind='{}')", self.path, self.kind)
    }
}

/// One decoded, validated TFM1 snapshot.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "Snapshot")]
pub struct PySnapshot {
    inner: tfm1::Snapshot,
    entries: Vec<Py<PySnapshotEntry>>,
}

#[pymethods]
impl PySnapshot {
    /// The parent snapshot id as bare hex, or `None` for a root snapshot.
    #[getter]
    fn parent(&self) -> Option<String> {
        self.inner.parent().map(|id| hex_of(id.as_bytes()))
    }

    /// This snapshot's own id: the SHA-256 of its canonical encoding.
    #[getter]
    fn snapshot_id(&self) -> String {
        let id = self.inner.snapshot_id();
        hex_of(id.as_bytes())
    }

    #[getter]
    fn entries(&self, py: Python<'_>) -> Vec<Py<PySnapshotEntry>> {
        self.entries.iter().map(|item| item.clone_ref(py)).collect()
    }

    /// Re-encodes to the canonical bytes. Decoding then re-encoding is
    /// byte-identical, which is what makes `snapshot_id` an identity.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    /// The ordered records of one file entry, or `None` when the path is
    /// absent or is not a file. Feed the result to `RecordsReader`.
    fn file_records(&self, py: Python<'_>, path: &str) -> PyResult<Option<Vec<Py<PyFileRecord>>>> {
        for (entry_path, entry) in self.inner.entries() {
            if entry_path != path {
                continue;
            }
            let Entry::File { body, .. } = entry else {
                return Ok(None);
            };
            return body
                .records()
                .iter()
                .map(|record| Py::new(py, PyFileRecord::from_core(record)))
                .collect::<PyResult<Vec<_>>>()
                .map(Some);
        }
        Ok(None)
    }

    /// Resolves a hardlink ordinal to its carrier path.
    fn ordinal_path(&self, ordinal: u64) -> Option<String> {
        self.inner.ordinal_path(ordinal).map(ToOwned::to_owned)
    }

    fn __repr__(&self) -> String {
        format!(
            "Snapshot(snapshot_id='{}', entries={})",
            self.snapshot_id(),
            self.entries.len()
        )
    }
}

/// One object carried inside a TFP1 pack.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "PackObject")]
pub struct PyPackObject {
    digest: String,
    bytes: Vec<u8>,
}

#[pymethods]
impl PyPackObject {
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }

    #[getter]
    fn length(&self) -> usize {
        self.bytes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PackObject(digest='{}', length={})",
            self.digest,
            self.bytes.len()
        )
    }
}

// ---------------------------------------------------------------------------
// The object store
// ---------------------------------------------------------------------------

/// Crash-safe immutable object storage under one root directory.
///
/// Every method releases the GIL for the duration of its I/O and hashing, so
/// a concurrent Python thread is not blocked behind an ingest.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "ObjectStore")]
pub struct PyObjectStore {
    inner: Arc<ObjectStore>,
    root: PathBuf,
}

#[pymethods]
impl PyObjectStore {
    #[new]
    fn new(py: Python<'_>, root: PathBuf) -> PyResult<Self> {
        let opened = {
            let root = root.clone();
            py.detach(move || ObjectStore::open(&root))
                .map_err(store_error)?
        };
        Ok(Self {
            inner: Arc::new(opened),
            root,
        })
    }

    #[getter]
    fn root(&self) -> PathBuf {
        self.root.clone()
    }

    /// Admits one byte string and returns its verified identity.
    ///
    /// `data` is copied out of the Python buffer before the GIL is released;
    /// that copy is small beside the SHA-256 pass and the fsync it guards.
    fn put_bytes(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<PyAdmittedObject> {
        let store = Arc::clone(&self.inner);
        let admitted = py
            .detach(move || store.put_bytes(&data))
            .map_err(store_error)?;
        let digest = admitted.digest();
        Ok(PyAdmittedObject {
            digest: hex_of(digest.as_bytes()),
            length: admitted.length(),
            preexisting: admitted.preexisting(),
        })
    }

    /// Plans one file with the canonical planner and admits every region,
    /// reading and hashing each byte exactly once.
    ///
    /// Returns `(plan, objects)`, which correspond positionally: `objects[i]`
    /// is the admitted form of `plan.regions[i]`.
    fn admit_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
    ) -> PyResult<(PyPlan, Vec<PyAdmittedObject>)> {
        let store = Arc::clone(&self.inner);
        let (plan, admitted) = py.detach(move || -> PyResult<_> {
            let source = FileByteSource::open(&path).map_err(io_error)?;
            let plan = planner::plan(&source).map_err(plan_error)?;
            let admitted = store
                .admit_regions(&source, plan.regions())
                .map_err(store_error)?;
            Ok((plan, admitted))
        })?;
        let objects = admitted
            .iter()
            .map(|item| {
                let digest = item.digest();
                PyAdmittedObject {
                    digest: hex_of(digest.as_bytes()),
                    length: item.length(),
                    preexisting: item.preexisting(),
                }
            })
            .collect();
        Ok((plan_into_python(py, &plan)?, objects))
    }

    /// Reads one resident object whole. Objects are bounded to
    /// `MAX_OBJECT_SIZE`, so this read cannot be unbounded.
    fn read_object<'py>(&self, py: Python<'py>, digest: &str) -> PyResult<Bound<'py, PyBytes>> {
        let digest = parse_digest(digest)?;
        let store = Arc::clone(&self.inner);
        let bytes = py.detach(move || -> PyResult<Vec<u8>> {
            use std::io::Read as _;
            let mut file = store.open_object(&digest).map_err(store_error)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).map_err(io_error)?;
            Ok(bytes)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Reads `length` bytes of one resident object, starting at `offset`.
    fn read_object_range<'py>(
        &self,
        py: Python<'py>,
        digest: &str,
        offset: u64,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let digest = parse_digest(digest)?;
        let store = Arc::clone(&self.inner);
        let bytes = py.detach(move || -> PyResult<Vec<u8>> {
            use std::io::{Read as _, Seek as _, SeekFrom};
            let mut file = store.open_object(&digest).map_err(store_error)?;
            file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| io_error(std::io::Error::from(std::io::ErrorKind::OutOfMemory)))?;
            bytes.resize(length, 0);
            file.read_exact(&mut bytes).map_err(io_error)?;
            Ok(bytes)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Rehashes one resident object and returns its length. This is the
    /// explicit scrub: the read path does not re-verify content.
    fn verify(&self, py: Python<'_>, digest: &str) -> PyResult<u64> {
        let digest = parse_digest(digest)?;
        let store = Arc::clone(&self.inner);
        py.detach(move || store.verify(&digest))
            .map_err(store_error)
    }

    /// True when the object is resident and structurally openable. It does
    /// NOT re-verify content — call `verify` for that.
    fn contains(&self, py: Python<'_>, digest: &str) -> PyResult<bool> {
        let digest = parse_digest(digest)?;
        let store = Arc::clone(&self.inner);
        Ok(py.detach(move || store.open_object(&digest)).is_ok())
    }

    /// Deletes temporary files abandoned longer ago than `grace_seconds`.
    ///
    /// The grace is the CALLER's, deliberately: this crate ships no timeout
    /// constant of its own.
    fn collect_abandoned_temps(
        &self,
        py: Python<'_>,
        grace_seconds: f64,
    ) -> PyResult<PyTempCollection> {
        if !grace_seconds.is_finite() || grace_seconds < 0.0 {
            return Err(TensorfsError::new_err(
                "grace_seconds must be a finite, non-negative number of seconds",
            ));
        }
        let grace = Duration::from_secs_f64(grace_seconds);
        let store = Arc::clone(&self.inner);
        let collected = py
            .detach(move || store.collect_abandoned_temps(grace))
            .map_err(store_error)?;
        Ok(PyTempCollection {
            examined: collected.examined,
            deleted: collected.deleted,
            bytes_deleted: collected.bytes_deleted,
        })
    }

    /// Whether this filesystem let the store create a symlink at open. False
    /// means the projection writes copies instead: dedup lost, correctness
    /// kept.
    #[getter]
    fn supports_symlinks(&self) -> bool {
        self.inner.supports_symlinks()
    }

    /// Projects one snapshot as `snapshots/<id>/…` and returns the tree path.
    ///
    /// Zero bytes are copied: directories are directories, blob-planner files
    /// are relative symlinks into `objects/`, and tensor-planner files — which
    /// have no single inode to point at — are `TFSSTUB1` pointer stubs
    /// (`spec/v1/TFSSTUB1.md`). The tree is disposable and pins nothing.
    fn project_snapshot(&self, py: Python<'_>, snapshot: &PySnapshot) -> PyResult<PathBuf> {
        let store = Arc::clone(&self.inner);
        let decoded = snapshot.inner.clone();
        py.detach(move || Layout::new(&store).project(&decoded))
            .map_err(layout_error)
    }

    /// Removes one projected tree. A tree is cache, not evidence, so this
    /// frees nothing the manifest cannot rebuild.
    fn remove_snapshot_tree(&self, py: Python<'_>, snapshot_id: &str) -> PyResult<bool> {
        let id = SnapshotId::from_bytes(parse_hex32(snapshot_id)?);
        let store = Arc::clone(&self.inner);
        py.detach(move || Layout::new(&store).remove_tree(&id))
            .map_err(layout_error)
    }

    /// Points `refs/<name>` at one snapshot, replacing any previous value by
    /// one `rename(2)`: a concurrent reader sees the old id or the new one and
    /// never a gap.
    fn set_ref(&self, py: Python<'_>, name: &str, snapshot_id: &str) -> PyResult<()> {
        let id = SnapshotId::from_bytes(parse_hex32(snapshot_id)?);
        let store = Arc::clone(&self.inner);
        let name = name.to_owned();
        py.detach(move || Layout::new(&store).set_ref(&name, &id))
            .map_err(layout_error)
    }

    fn read_ref(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        let store = Arc::clone(&self.inner);
        let name = name.to_owned();
        let found = py
            .detach(move || Layout::new(&store).read_ref(&name))
            .map_err(layout_error)?;
        Ok(found.map(|id| hex_of(id.as_bytes())))
    }

    fn remove_ref(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let store = Arc::clone(&self.inner);
        let name = name.to_owned();
        py.detach(move || Layout::new(&store).remove_ref(&name))
            .map_err(layout_error)
    }

    fn __repr__(&self) -> String {
        format!("ObjectStore(root={:?})", self.root)
    }
}

// ---------------------------------------------------------------------------
// The byte-source path
// ---------------------------------------------------------------------------

/// A committed file's ordered records read as one bounded random-access byte
/// source, with nothing materialized.
///
/// This is the foundation of a direct tensor read: `plan()` gives the tensor
/// byte ranges of a safetensors or GGUF file, and `read_at()` returns exactly
/// those bytes, assembled from the immutable objects that back them.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "RecordsReader")]
pub struct PyRecordsReader {
    source: RecordsSource<Arc<ObjectStore>>,
}

#[pymethods]
impl PyRecordsReader {
    #[new]
    fn new(store: &PyObjectStore, records: Vec<PyFileRecord>) -> PyResult<Self> {
        let records = records
            .iter()
            .map(PyFileRecord::to_core)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            source: RecordsSource::new(Arc::clone(&store.inner), &records),
        })
    }

    /// The committed logical length of the file, holes included.
    #[getter]
    fn length(&self) -> u64 {
        self.source.len()
    }

    /// Reads `length` bytes starting at `offset`. Holes read as zeros. A
    /// range past the committed length is refused, never truncated.
    fn read_at<'py>(
        &self,
        py: Python<'py>,
        offset: u64,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| -> PyResult<Vec<u8>> {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| io_error(std::io::Error::from(std::io::ErrorKind::OutOfMemory)))?;
            bytes.resize(length, 0);
            self.source
                .read_exact_at(offset, &mut bytes)
                .map_err(io_error)?;
            Ok(bytes)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Runs the canonical planner over the committed bytes without
    /// materializing the file. For a safetensors or GGUF file the `"tensor"`
    /// regions are the tensor byte ranges.
    fn plan(&self, py: Python<'_>) -> PyResult<PyPlan> {
        let plan = py
            .detach(|| planner::plan(&self.source))
            .map_err(plan_error)?;
        plan_into_python(py, &plan)
    }

    fn __repr__(&self) -> String {
        format!("RecordsReader(length={})", self.source.len())
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Decodes canonical TFM1 bytes.
#[pyfunction]
fn decode_snapshot(py: Python<'_>, data: &[u8]) -> PyResult<PySnapshot> {
    let snapshot = tfm1::decode(data).map_err(tfm1_error)?;
    let entries = snapshot
        .entries()
        .iter()
        .map(|(path, entry)| {
            let converted = match entry {
                Entry::Directory => PySnapshotEntry {
                    path: path.clone(),
                    kind: "directory".to_owned(),
                    executable: None,
                    planner: None,
                    logical_size: None,
                    records: None,
                    digest: None,
                    body_sha256: None,
                    target: None,
                    ordinal: None,
                },
                Entry::File { executable, body } => PySnapshotEntry {
                    path: path.clone(),
                    kind: "file".to_owned(),
                    executable: Some(*executable),
                    planner: Some(planner_name(body.planner_id()).to_owned()),
                    logical_size: Some(body.logical_size()),
                    // The effective record run: a tensor body's records, or a
                    // nonempty blob's single whole-file record. Feed it to
                    // `RecordsReader` unchanged.
                    records: Some(
                        body.records()
                            .iter()
                            .map(|record| Py::new(py, PyFileRecord::from_core(record)))
                            .collect::<PyResult<Vec<_>>>()?,
                    ),
                    digest: match body {
                        tfm1::FileBody::Blob { digest, .. } => Some(hex_of(digest.as_bytes())),
                        tfm1::FileBody::Tensor { .. } => None,
                    },
                    body_sha256: Some(hex_of(body.body_sha256().as_bytes())),
                    target: None,
                    ordinal: None,
                },
                Entry::Symlink { target } => PySnapshotEntry {
                    path: path.clone(),
                    kind: "symlink".to_owned(),
                    executable: None,
                    planner: None,
                    logical_size: None,
                    records: None,
                    digest: None,
                    body_sha256: None,
                    target: Some(target.clone()),
                    ordinal: None,
                },
                Entry::Hardlink { ordinal } => PySnapshotEntry {
                    path: path.clone(),
                    kind: "hardlink".to_owned(),
                    executable: None,
                    planner: None,
                    logical_size: None,
                    records: None,
                    digest: None,
                    body_sha256: None,
                    target: None,
                    ordinal: Some(*ordinal),
                },
            };
            Py::new(py, converted)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PySnapshot {
        inner: snapshot,
        entries,
    })
}

/// Renders one pointer stub. `body_sha256` is a file entry's TFM1 body digest
/// and `size` its logical size; both come from the manifest, so rendering a
/// stub reads no tensor bytes. The bytes are the contract — see
/// `spec/v1/TFSSTUB1.md` and `spec/v1/tfsstub1-vectors/`.
#[pyfunction]
fn stub_bytes<'py>(py: Python<'py>, body_sha256: &str, size: u64) -> PyResult<Bound<'py, PyBytes>> {
    let digest = parse_digest(body_sha256)?;
    Ok(PyBytes::new(py, &layout::stub_bytes(&digest, size)))
}

/// Reads one pointer stub as `(body_sha256, size)`, or `None` when the bytes
/// are not a stub. The magic decides in eight bytes.
#[pyfunction]
fn parse_stub(data: &[u8]) -> Option<(String, u64)> {
    layout::parse_stub(data).map(|stub| (hex_of(stub.body_sha256.as_bytes()), stub.size))
}

/// The snapshot id of canonical TFM1 bytes, without decoding them.
#[pyfunction]
fn snapshot_id_of(data: &[u8]) -> String {
    let id = SnapshotId::of(data);
    hex_of(id.as_bytes())
}

/// Decodes a TFP1 pack and verifies every object's digest against its bytes.
#[pyfunction]
fn decode_pack(py: Python<'_>, data: Vec<u8>) -> PyResult<Vec<PyPackObject>> {
    py.detach(move || {
        let pack = tfp1::decode(&data).map_err(tfp1_error)?;
        pack.verify_objects().map_err(tfp1_error)?;
        Ok(pack
            .objects()
            .map(|object| {
                let digest = object.digest();
                PyPackObject {
                    digest: hex_of(digest.as_bytes()),
                    bytes: object.bytes().to_vec(),
                }
            })
            .collect())
    })
}

/// Encodes `(digest, bytes)` pairs into one canonical TFP1 pack.
#[pyfunction]
fn encode_pack<'py>(
    py: Python<'py>,
    objects: Vec<(String, Vec<u8>)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let parsed = objects
        .iter()
        .map(|(digest, bytes)| Ok((parse_digest(digest)?, bytes.as_slice())))
        .collect::<PyResult<Vec<_>>>()?;
    let encoded = py.detach(|| tfp1::encode(&parsed)).map_err(tfp1_error)?;
    Ok(PyBytes::new(py, &encoded))
}

/// Runs the canonical planner over an in-memory byte string.
#[pyfunction]
fn plan_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PyPlan> {
    let plan = planner::plan(data).map_err(plan_error)?;
    plan_into_python(py, &plan)
}

/// Plans and hashes an in-memory byte string.
#[pyfunction]
fn plan_and_hash_bytes(py: Python<'_>, data: Vec<u8>) -> PyResult<PyHashedPlan> {
    let plan = py
        .detach(move || object::plan_and_hash(data.as_slice()))
        .map_err(plan_error)?;
    hashed_plan_into_python(py, &plan)
}

/// Runs the canonical planner over a file, reading only bounded headers.
#[pyfunction]
fn plan_file(py: Python<'_>, path: PathBuf) -> PyResult<PyPlan> {
    let plan = py.detach(move || -> PyResult<_> {
        let source = FileByteSource::open(&path).map_err(io_error)?;
        planner::plan(&source).map_err(plan_error)
    })?;
    plan_into_python(py, &plan)
}

/// Plans and hashes a file. Every byte is read and hashed exactly once.
#[pyfunction]
fn plan_and_hash_file(py: Python<'_>, path: PathBuf) -> PyResult<PyHashedPlan> {
    let plan = py.detach(move || -> PyResult<_> {
        let source = FileByteSource::open(&path).map_err(io_error)?;
        object::plan_and_hash(&source).map_err(plan_error)
    })?;
    hashed_plan_into_python(py, &plan)
}

/// How many objects the store hashes and installs concurrently.
#[pyfunction]
fn ingest_concurrency() -> usize {
    tensorfs_core::store::ingest_concurrency()
}

/// The compiled half of the `tensorfs` distribution.
#[pymodule]
#[pyo3(name = "_tensorfs")]
fn tensorfs_extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();

    module.add("__version__", env!("CARGO_PKG_VERSION"))?;

    module.add("TensorfsError", py.get_type::<TensorfsError>())?;
    module.add("StoreError", py.get_type::<StoreError>())?;
    module.add("FormatError", py.get_type::<FormatError>())?;
    module.add("PlanError", py.get_type::<PlanError>())?;
    module.add("LayoutError", py.get_type::<LayoutError>())?;
    module.add("STUB_MAGIC", PyBytes::new(py, STUB_MAGIC))?;

    module.add_class::<PyAdmittedObject>()?;
    module.add_class::<PyTempCollection>()?;
    module.add_class::<PyRegion>()?;
    module.add_class::<PyPlan>()?;
    module.add_class::<PyPlannedObject>()?;
    module.add_class::<PyHashedPlan>()?;
    module.add_class::<PyFileRecord>()?;
    module.add_class::<PySnapshotEntry>()?;
    module.add_class::<PySnapshot>()?;
    module.add_class::<PyPackObject>()?;
    module.add_class::<PyObjectStore>()?;
    module.add_class::<PyRecordsReader>()?;

    module.add_function(wrap_pyfunction!(decode_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(stub_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(parse_stub, module)?)?;
    module.add_function(wrap_pyfunction!(snapshot_id_of, module)?)?;
    module.add_function(wrap_pyfunction!(decode_pack, module)?)?;
    module.add_function(wrap_pyfunction!(encode_pack, module)?)?;
    module.add_function(wrap_pyfunction!(plan_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(plan_and_hash_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(plan_file, module)?)?;
    module.add_function(wrap_pyfunction!(plan_and_hash_file, module)?)?;
    module.add_function(wrap_pyfunction!(ingest_concurrency, module)?)?;

    module.add("MAX_OBJECT_SIZE", planner::MAX_OBJECT_SIZE)?;
    module.add("MAX_PACK_PAYLOAD", tfp1::MAX_PACK_PAYLOAD)?;
    module.add("MAX_PACK_OBJECTS", tfp1::MAX_PACK_OBJECTS)?;
    module.add("MAX_FILE_RECORDS", tfm1::MAX_FILE_RECORDS)?;
    module.add("MAX_PATH_BYTES", tfm1::MAX_PATH_BYTES)?;
    module.add("MAX_SYMLINK_TARGET_BYTES", tfm1::MAX_SYMLINK_TARGET_BYTES)?;

    Ok(())
}
