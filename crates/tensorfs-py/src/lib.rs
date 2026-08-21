//! `tensorfs._tensorfs` — the compiled half of the `tensorfs` distribution.
//!
//! Four surfaces, chosen because they are the ones a direct tensor read
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
//! 4. **The fill path** (`TensorStreamReader.fill_host_into`,
//!    `CudaFillClient`): source records plus plain destination address data
//!    drive tensorfs's one layout plan/transform. No torch type crosses it.
//!
//! `#![forbid(unsafe_code)]` is deliberately absent here, unlike every other
//! crate in this workspace: the PyO3 attribute macros expand to `unsafe impl`s
//! in this crate. The hand-written `unsafe` is confined to memory-map/buffer
//! protocol export and the checked borrowed destination span; both are raw
//! pointer contracts with CPython and cannot be expressed in safe Rust.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::{c_int, c_void};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tensorfs_core::compose;
use tensorfs_core::contract;
use tensorfs_core::layout::{self, Layout, Removal, STUB_MAGIC};
use tensorfs_core::layout_fill::{self, ChunkedSource, FillSink, TensorFill};
use tensorfs_core::layout_morphism;
use tensorfs_core::object::{self, ObjectDigest};
use tensorfs_core::planner::{self, ByteSource, PlannerId, RegionKind};
use tensorfs_core::source::FileByteSource;
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{self, Entry, FileRecord as CoreFileRecord, SnapshotId};
use tensorfs_core::tfp1;
use tensorfs_core::workspace_source::RecordsSource;
use tensorfs_cuda::CudaSink;

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

fn fill_error(error: layout_morphism::LayoutError) -> PyErr {
    LayoutError::new_err(error.to_string())
}

fn compose_error(error: compose::ComposeError) -> PyErr {
    match error {
        compose::ComposeError::Store(error) => store_error(error),
        compose::ComposeError::Plan(error) => plan_error(error),
        other => TensorfsError::new_err(describe(&other)),
    }
}

fn contract_error(error: contract::ContractError) -> PyErr {
    PyValueError::new_err(describe(&error))
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

fn parse_planner(name: &str) -> PyResult<PlannerId> {
    match name {
        "safetensors-v1" => Ok(PlannerId::SafetensorsV1),
        "gguf-v1" => Ok(PlannerId::GgufV1),
        "blob-v1" => Ok(PlannerId::BlobV1),
        other => Err(TensorfsError::new_err(format!("unknown planner '{other}'"))),
    }
}

const fn record_length(record: &CoreFileRecord) -> u64 {
    match record {
        CoreFileRecord::Data { length, .. } | CoreFileRecord::Hole { length } => *length,
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
    contract: String,
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

    /// The layout contract that directed these boundaries, `name@version`,
    /// or `"none"` for the plain per-tensor grid.
    #[getter]
    fn contract(&self) -> &str {
        &self.contract
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
            "Plan(planner='{}', contract='{}', file_size={}, regions={})",
            self.planner,
            self.contract,
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
        contract: plan.contract().to_string(),
        file_size: plan.file_size(),
        regions,
    })
}

/// A plan whose every object has been hashed from one stable source snapshot.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "HashedPlan")]
pub struct PyHashedPlan {
    contract: String,
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

    /// The layout contract that directed these boundaries, `name@version`,
    /// or `"none"`.
    #[getter]
    fn contract(&self) -> &str {
        &self.contract
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
        contract: plan.contract().to_string(),
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
// The zero-copy export: one CAS object, mapped, handed out as a `Py_buffer`
// ---------------------------------------------------------------------------

/// One immutable CAS object, memory-mapped and exported through the C buffer
/// protocol.
///
/// This is where "direct tensor reads are zero-copy" stops being a property of
/// the facade's `mmap` module and becomes a property of the extension. A
/// `memoryview` over one of these addresses the mapped pages themselves: no
/// intermediate `bytes`, no copy at the language boundary, and one resident
/// object per tensor piece rather than a whole shard.
///
/// The export is READ-ONLY and refuses `PyBUF_WRITABLE`. CAS objects are named
/// by the hash of their bytes, so a writable view would be a view that can
/// invalidate its own name.
///
/// Ownership is the buffer protocol's, not the caller's: `Py_buffer` holds a
/// strong reference to this object, so the mapping outlives every view taken
/// of it even if the reader that opened it is closed first.
///
/// Requires the buffer protocol under the Limited API, which is why this
/// distribution's abi3 floor is 3.11 -- `Py_buffer`, `Py_bf_getbuffer` and
/// `PyBuffer_FillInfo` entered CPython's stable ABI in 3.11 (bpo-45459), and
/// PyO3 gates its own buffer-protocol tests on exactly `any(not(Py_LIMITED_API),
/// Py_3_11)`.
/// `weakref` is required, not cosmetic: the reader keeps its mappings in a
/// `WeakValueDictionary` so a mapping lives exactly as long as someone is
/// reading through it. Without that, a conversion that touches every tensor of
/// an 8 GiB shard ends with the whole shard resident: measured at 8036.7 MiB
/// peak RSS with a strong cache against 310.1 MiB with a weak one.
#[pyclass(frozen, weakref, module = "tensorfs._tensorfs", name = "MappedObject")]
pub struct PyMappedObject {
    /// `None` for a zero-length object: a zero-length mapping is refused by
    /// `mmap(2)` itself, and an empty export is the honest answer.
    map: Option<memmap2::Mmap>,
    path: PathBuf,
}

impl PyMappedObject {
    fn bytes(&self) -> &[u8] {
        self.map.as_deref().unwrap_or(&[])
    }
}

#[pymethods]
impl PyMappedObject {
    /// Maps one object file read-only. The caller holds whatever store lock
    /// keeps collection from unlinking it between resolution and open; once
    /// the mapping exists the bytes are pinned by the inode.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let opened = path.clone();
        let map = py
            .detach(move || {
                let file = std::fs::File::open(&opened)?;
                let length = file.metadata()?.len();
                if length == 0 {
                    return Ok(None);
                }
                // SAFETY: CAS objects are immutable once installed -- they are
                // named by their own digest and never opened for writing --
                // so the mapping cannot be mutated underneath a live view by
                // this library. A caller that edits the store out of band is
                // outside the contract, exactly as it is for `mmap.mmap`.
                let map = unsafe { memmap2::Mmap::map(&file) }?;
                Ok::<_, std::io::Error>(Some(map))
            })
            .map_err(io_error)?;
        Ok(Self { map, path })
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    #[getter]
    fn length(&self) -> usize {
        self.bytes().len()
    }

    fn __len__(&self) -> usize {
        self.bytes().len()
    }

    /// Exports the mapping itself. Nothing is copied, and nothing is read
    /// until the consumer touches a page.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("Py_buffer is null"));
        }
        if flags & pyo3::ffi::PyBUF_WRITABLE == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err(
                "a CAS object is immutable; its buffer cannot be exported writable",
            ));
        }
        let bytes = slf.get().bytes();
        let length = pyo3::ffi::Py_ssize_t::try_from(bytes.len())
            .map_err(|_| PyBufferError::new_err("mapping is larger than Py_ssize_t"))?;
        // SAFETY: `view` is non-null and CPython owns it; `bytes` points into
        // a mapping owned by `slf`, and `PyBuffer_FillInfo` takes a strong
        // reference to `slf` for the buffer's lifetime.
        let filled = unsafe {
            pyo3::ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                bytes.as_ptr() as *mut c_void,
                length,
                1,
                flags,
            )
        };
        if filled != 0 {
            return Err(PyErr::take(slf.py())
                .unwrap_or_else(|| PyBufferError::new_err("Py_buffer could not be filled")));
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MappedObject(path={:?}, length={})",
            self.path,
            self.bytes().len()
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
    #[pyo3(signature = (path, seams = None))]
    fn admit_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        seams: Option<&str>,
    ) -> PyResult<(PyPlan, Vec<PyAdmittedObject>)> {
        let store = Arc::clone(&self.inner);
        let registry = parse_seams(seams)?;
        let (plan, admitted) = py.detach(move || -> PyResult<_> {
            let source = FileByteSource::open(&path).map_err(io_error)?;
            let plan = plan_under(&source, registry.as_ref())?;
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
    ///
    /// `False` means this call did not take the name: nothing was there, or —
    /// on Windows only — the rename that takes it away was refused
    /// transiently, which leaves the tree for the next scrub.
    fn remove_snapshot_tree(&self, py: Python<'_>, snapshot_id: &str) -> PyResult<bool> {
        let id = SnapshotId::from_bytes(parse_hex32(snapshot_id)?);
        let store = Arc::clone(&self.inner);
        py.detach(move || Layout::new(&store).remove_tree(&id))
            .map(Removal::taken)
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
            .map(Removal::taken)
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
// The streamed store→VRAM read path (#115)
// ---------------------------------------------------------------------------

/// One tensor of a streamed container: name and geometry, no bytes.
#[pyclass(frozen, get_all, module = "tensorfs._tensorfs", name = "StreamTensor")]
pub struct PyStreamTensor {
    name: String,
    /// The container's own dtype spelling: safetensors' (`"BF16"`) or the
    /// ggml type name for GGUF.
    dtype: String,
    /// Logical shape, outermost axis first.
    shape: Vec<u64>,
    /// Offset of the tensor's first byte within the logical file.
    offset: u64,
    /// The tensor's own extent in bytes.
    nbytes: u64,
}

#[pymethods]
impl PyStreamTensor {
    fn __repr__(&self) -> String {
        format!(
            "StreamTensor(name='{}', dtype='{}', offset={}, nbytes={})",
            self.name, self.dtype, self.offset, self.nbytes
        )
    }
}

/// The fill's measured facts, returned unchanged across the Python seam.
#[pyclass(frozen, get_all, module = "tensorfs._tensorfs", name = "FillStats")]
pub struct PyFillStats {
    source_bytes: u64,
    destination_bytes: u64,
    padding_bytes: u64,
    runs: u64,
    chunks: u64,
}

impl From<layout_fill::FillStats> for PyFillStats {
    fn from(stats: layout_fill::FillStats) -> Self {
        Self {
            source_bytes: stats.source_bytes,
            destination_bytes: stats.destination_bytes,
            padding_bytes: stats.padding_bytes,
            runs: stats.runs,
            chunks: stats.chunks,
        }
    }
}

enum SourceSegment {
    Data {
        map: memmap2::Mmap,
        start: usize,
        end: usize,
    },
    Hole(Vec<u8>),
}

impl SourceSegment {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Data { map, start, end } => &map[*start..*end],
            Self::Hole(bytes) => bytes,
        }
    }
}

/// Maps exactly one tensor's source pieces out of the committed record run.
/// The mappings own their file descriptors' inodes for the duration of the
/// fill; no tensor-sized source buffer is assembled first.
fn source_segments(
    store: &ObjectStore,
    records: &[CoreFileRecord],
    offset: u64,
    length: u64,
) -> PyResult<Vec<SourceSegment>> {
    let finish = offset
        .checked_add(length)
        .ok_or_else(|| LayoutError::new_err("tensor source range overflows u64"))?;
    let mut position = 0_u64;
    let mut segments = Vec::new();
    let mut covered = 0_u64;
    for record in records {
        let record_end = position + record_length(record);
        if record_end <= offset {
            position = record_end;
            continue;
        }
        if position >= finish {
            break;
        }
        let from = offset.max(position);
        let to = finish.min(record_end);
        let within = usize::try_from(from - position)
            .map_err(|_| LayoutError::new_err("tensor source offset does not fit usize"))?;
        let count = usize::try_from(to - from)
            .map_err(|_| LayoutError::new_err("tensor source length does not fit usize"))?;
        match record {
            CoreFileRecord::Hole { .. } => segments.push(SourceSegment::Hole(vec![0; count])),
            CoreFileRecord::Data { digest, .. } => {
                let file = store.open_object(digest).map_err(store_error)?;
                // SAFETY: store objects are immutable after admission. The
                // mapping owns the inode while SourceSegment is alive.
                let map = unsafe { memmap2::Mmap::map(&file) }.map_err(io_error)?;
                let end = within
                    .checked_add(count)
                    .ok_or_else(|| LayoutError::new_err("tensor source slice overflows usize"))?;
                if end > map.len() {
                    return Err(LayoutError::new_err(format!(
                        "object {} holds {} bytes, tensor fill asks for byte {}",
                        digest,
                        map.len(),
                        end,
                    )));
                }
                segments.push(SourceSegment::Data {
                    map,
                    start: within,
                    end,
                });
            }
        }
        covered += to - from;
        position = record_end;
    }
    if covered != length {
        return Err(LayoutError::new_err(format!(
            "tensor source range holds {covered} bytes, fill requires {length}"
        )));
    }
    Ok(segments)
}

struct BufferSink {
    address: usize,
    capacity: usize,
}

impl FillSink for BufferSink {
    fn staging(
        &mut self,
        bytes: usize,
        device_offset: u64,
    ) -> Result<&mut [u8], layout_morphism::LayoutError> {
        let at =
            usize::try_from(device_offset).map_err(|_| layout_morphism::LayoutError::Buffer {
                got: self.capacity,
                want: usize::MAX,
            })?;
        let end = at
            .checked_add(bytes)
            .ok_or(layout_morphism::LayoutError::Buffer {
                got: self.capacity,
                want: usize::MAX,
            })?;
        if end > self.capacity {
            return Err(layout_morphism::LayoutError::Buffer {
                got: self.capacity,
                want: end,
            });
        }
        // SAFETY: the Py_buffer retained by fill_host_into keeps this writable
        // contiguous allocation alive for the whole call, and the bounds were
        // checked against its reported capacity above.
        Ok(unsafe { std::slice::from_raw_parts_mut((self.address + at) as *mut u8, bytes) })
    }

    fn commit(&mut self) -> Result<(), layout_morphism::LayoutError> {
        Ok(())
    }
}

fn fill_tensor<S: FillSink>(
    reader: &PyTensorStreamReader,
    name: &str,
    destination_offset: u64,
    layout: &str,
    sink: &mut S,
) -> PyResult<PyFillStats> {
    let tensor = reader
        .tensors
        .iter()
        .map(Py::get)
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| {
            TensorfsError::new_err(format!("this container holds no tensor named {name:?}"))
        })?;
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |total, dim| total.checked_mul(*dim))
        .ok_or_else(|| LayoutError::new_err(format!("{name:?}: logical shape overflows u64")))?;
    if elements == 0 || tensor.nbytes % elements != 0 {
        return Err(LayoutError::new_err(format!(
            "{name:?}: {} bytes do not divide over shape {:?}",
            tensor.nbytes, tensor.shape,
        )));
    }
    let element_bytes = usize::try_from(tensor.nbytes / elements)
        .map_err(|_| LayoutError::new_err("element size does not fit usize"))?;
    let segments = source_segments(
        reader.store.as_ref(),
        &reader.records,
        tensor.offset,
        tensor.nbytes,
    )?;
    let slices = segments
        .iter()
        .map(SourceSegment::bytes)
        .collect::<Vec<_>>();
    let source = ChunkedSource::new(&slices);
    let arrangement = layout_morphism::arrangement(layout).map_err(fill_error)?;
    let stats = layout_fill::fill(
        &source,
        &TensorFill {
            layout: arrangement,
            shape: tensor.shape.clone(),
            element_bytes,
            device_offset: destination_offset,
        },
        sink,
    )
    .map_err(fill_error)?;
    Ok(stats.into())
}

/// One reusable pinned staging allocation for a whole destination map.
#[pyclass(module = "tensorfs._tensorfs", name = "CudaFillClient")]
pub struct PyCudaFillClient {
    sink: Mutex<CudaSink>,
}

#[pymethods]
impl PyCudaFillClient {
    #[new]
    #[pyo3(signature = (staging_bytes, device = 0))]
    fn new(staging_bytes: usize, device: i32) -> PyResult<Self> {
        if staging_bytes == 0 {
            return Err(LayoutError::new_err("CUDA fill staging must hold bytes"));
        }
        tensorfs_cuda::bind(device).map_err(fill_error)?;
        let sink = CudaSink::new(0, 0, staging_bytes).map_err(fill_error)?;
        Ok(Self {
            sink: Mutex::new(sink),
        })
    }

    #[pyo3(signature = (reader, name, destination_ptr, destination_bytes, destination_offset = 0, layout = "torch.contiguous@1"))]
    fn fill(
        &self,
        reader: &PyTensorStreamReader,
        name: &str,
        destination_ptr: u64,
        destination_bytes: usize,
        destination_offset: u64,
        layout: &str,
    ) -> PyResult<PyFillStats> {
        if destination_ptr == 0 {
            return Err(LayoutError::new_err(
                "CUDA fill destination pointer is null",
            ));
        }
        let mut sink = self
            .sink
            .lock()
            .map_err(|_| LayoutError::new_err("CUDA fill client was poisoned by a prior panic"))?;
        sink.retarget(destination_ptr, destination_bytes)
            .map_err(fill_error)?;
        fill_tensor(reader, name, destination_offset, layout, &mut *sink)
    }
}

/// A committed tensor container's streamed read surface: FILE-ORDER tensor
/// iteration and GIL-released bulk `readinto` into caller-owned buffers.
///
/// `tensors` is ordered by ascending file offset — an API CONTRACT, not an
/// accident of the header. For canonically packaged snapshots record order is
/// contract memory order, so walking `tensors` and reading each one is the
/// sequential access pattern the load-order ruling measured at ~10x.
///
/// The caller's buffer is typically CUDA-pinned host memory; this class sees
/// a writable buffer-protocol object and nothing more (the torch boundary
/// holds: no torch, no CUDA in this crate).
///
/// `direct=True` opens every object `O_DIRECT` (Linux): page cache bypassed,
/// whole aligned blocks landing straight in the caller's aligned buffer,
/// tails via an aligned over-read. The DEFAULT stays buffered — the page
/// cache is what makes warm loads and cross-checkpoint chunk sharing nearly
/// free, and the extra memcpy pipelines behind disk DMA.
#[pyclass(frozen, module = "tensorfs._tensorfs", name = "TensorStreamReader")]
pub struct PyTensorStreamReader {
    store: Arc<ObjectStore>,
    records: Vec<CoreFileRecord>,
    reader: tensorfs_core::stream::StreamReader<Arc<ObjectStore>>,
    format: String,
    tensors: Vec<Py<PyStreamTensor>>,
}

#[pymethods]
impl PyTensorStreamReader {
    #[new]
    #[pyo3(signature = (store, records, direct = false))]
    fn new(
        py: Python<'_>,
        store: &PyObjectStore,
        records: Vec<PyFileRecord>,
        direct: bool,
    ) -> PyResult<Self> {
        let records = records
            .iter()
            .map(PyFileRecord::to_core)
            .collect::<PyResult<Vec<_>>>()?;
        let mode = if direct {
            tensorfs_core::stream::ReadMode::Direct
        } else {
            tensorfs_core::stream::ReadMode::Buffered
        };
        let reader =
            tensorfs_core::stream::StreamReader::new(Arc::clone(&store.inner), &records, mode)
                .map_err(io_error)?;
        let inventory = py
            .detach(|| planner::inventory(&reader))
            .map_err(plan_error)?
            .ok_or_else(|| TensorfsError::new_err("these records are not a tensor container"))?;
        // ASCENDING FILE OFFSET, whatever order the header spelled them in:
        // the ordered iteration is the API contract.
        let mut tensors: Vec<&planner::InventoryTensor> = inventory.tensors().iter().collect();
        tensors.sort_by_key(|tensor| tensor.offset());
        let tensors = tensors
            .into_iter()
            .map(|tensor| {
                Py::new(
                    py,
                    PyStreamTensor {
                        name: tensor.name().to_owned(),
                        dtype: tensor.dtype().to_owned(),
                        shape: tensor.shape().to_vec(),
                        offset: tensor.offset(),
                        nbytes: tensor.length(),
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            store: Arc::clone(&store.inner),
            records,
            reader,
            format: match inventory.format() {
                planner::TensorFormat::SafetensorsV1 => "safetensors-v1".to_owned(),
                planner::TensorFormat::GgufV1 => "gguf-v1".to_owned(),
            },
            tensors,
        })
    }

    /// Every tensor, in ASCENDING FILE-OFFSET order (= record order =
    /// contract memory order for canonical packagings). This ordering is the
    /// contract; safetensors header key order is not offset order in general.
    #[getter]
    fn tensors(&self, py: Python<'_>) -> Vec<Py<PyStreamTensor>> {
        self.tensors.iter().map(|item| item.clone_ref(py)).collect()
    }

    /// `"safetensors-v1"` or `"gguf-v1"`.
    #[getter]
    fn format(&self) -> &str {
        &self.format
    }

    /// The committed logical length of the container, holes included.
    #[getter]
    fn length(&self) -> u64 {
        self.reader.length()
    }

    #[getter]
    fn direct(&self) -> bool {
        self.reader.mode() == tensorfs_core::stream::ReadMode::Direct
    }

    /// Copies `[offset, offset + length)` into `buffer`, which must be a
    /// writable C-contiguous buffer of at least `length` bytes. The copy runs
    /// in Rust WITH THE GIL RELEASED, so a second Python thread — the
    /// caller's H2D pipeline — makes progress underneath it. Holes read as
    /// zeros. Returns the byte count written.
    fn readinto(
        &self,
        py: Python<'_>,
        offset: u64,
        length: u64,
        buffer: pyo3::buffer::PyBuffer<u8>,
    ) -> PyResult<u64> {
        let destination = writable_span(&buffer, length)?;
        py.detach(move || {
            // SAFETY: `buffer` holds a live Py_buffer for the duration of
            // this call, so the exporter keeps the memory valid and unmoved;
            // `writable_span` checked writability, contiguity and capacity.
            let slice =
                unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, length as usize) };
            self.reader.read_exact_at(offset, slice)
        })
        .map_err(io_error)?;
        Ok(length)
    }

    /// `readinto` addressed by tensor name: exactly that tensor's bytes,
    /// into the front of `buffer`. Returns the tensor's `nbytes`.
    fn read_tensor_into(
        &self,
        py: Python<'_>,
        name: &str,
        buffer: pyo3::buffer::PyBuffer<u8>,
    ) -> PyResult<u64> {
        let found = self
            .tensors
            .iter()
            .map(|tensor| tensor.get())
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                TensorfsError::new_err(format!("this container holds no tensor named {name:?}"))
            })?;
        self.readinto(py, found.offset, found.nbytes, buffer)
    }

    /// Fill one tensor into a caller-owned host buffer through tensorfs's one
    /// plan/transform implementation. The buffer protocol is the whole seam:
    /// no numpy or torch type crosses it.
    #[pyo3(signature = (name, buffer, destination_offset = 0, layout = "torch.contiguous@1"))]
    fn fill_host_into(
        &self,
        name: &str,
        buffer: pyo3::buffer::PyBuffer<u8>,
        destination_offset: u64,
        layout: &str,
    ) -> PyResult<PyFillStats> {
        let address = writable_span(&buffer, 0)?;
        let mut sink = BufferSink {
            address,
            capacity: buffer.len_bytes(),
        };
        fill_tensor(self, name, destination_offset, layout, &mut sink)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorStreamReader(format='{}', tensors={}, length={}, direct={})",
            self.format,
            self.tensors.len(),
            self.reader.length(),
            if self.direct() { "True" } else { "False" }
        )
    }
}

/// Checks a destination buffer and returns its base address: writable,
/// C-contiguous, and at least `length` bytes.
fn writable_span(buffer: &pyo3::buffer::PyBuffer<u8>, length: u64) -> PyResult<usize> {
    if buffer.readonly() {
        return Err(TensorfsError::new_err(
            "the destination buffer is read-only",
        ));
    }
    if !buffer.is_c_contiguous() {
        return Err(TensorfsError::new_err(
            "the destination buffer must be C-contiguous",
        ));
    }
    let capacity = buffer.len_bytes() as u64;
    if capacity < length {
        return Err(TensorfsError::new_err(format!(
            "the destination holds {capacity} bytes, the read needs {length}"
        )));
    }
    Ok(buffer.buf_ptr() as usize)
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
///
/// With a registry, the file is first IDENTIFIED from its header inventory
/// and then planned under the contract that won — fused tensors cut at their
/// declared seams before the 64 MiB grid. Without one, the plain per-tensor
/// grid, which is `contract:none`.
#[pyfunction]
#[pyo3(signature = (path, seams = None))]
fn plan_file(py: Python<'_>, path: PathBuf, seams: Option<&str>) -> PyResult<PyPlan> {
    let registry = parse_seams(seams)?;
    let plan = py.detach(move || -> PyResult<_> {
        let source = FileByteSource::open(&path).map_err(io_error)?;
        plan_under(&source, registry.as_ref())
    })?;
    plan_into_python(py, &plan)
}

/// Plans and hashes a file. Every byte is read and hashed exactly once.
#[pyfunction]
#[pyo3(signature = (path, seams = None))]
fn plan_and_hash_file(
    py: Python<'_>,
    path: PathBuf,
    seams: Option<&str>,
) -> PyResult<PyHashedPlan> {
    let registry = parse_seams(seams)?;
    let plan = py.detach(move || -> PyResult<_> {
        let source = FileByteSource::open(&path).map_err(io_error)?;
        let plan = plan_under(&source, registry.as_ref())?;
        object::hash_planned(&source, &plan).map_err(plan_error)
    })?;
    hashed_plan_into_python(py, &plan)
}

/// Plans one file, cutting at the caller's declared seams when it has any.
///
/// v1 IDENTIFIED the file first — it scored a registry of library contracts
/// against the header and let the winner direct the cuts. That was a second
/// implementation of "which checkpoint is this", and it is gone: a caller that
/// knows the layout says so, and a caller that does not gets plain chunking.
fn plan_under<S: planner::ByteSource + ?Sized>(
    source: &S,
    seams: Option<&contract::Contract>,
) -> PyResult<planner::Plan> {
    match seams {
        None => planner::plan(source).map_err(plan_error),
        Some(seams) => planner::plan_with(source, Some(seams)).map_err(plan_error),
    }
}

/// Parses an optional seam document.
fn parse_seams(document: Option<&str>) -> PyResult<Option<contract::Contract>> {
    document
        .map(|text| contract::Contract::parse(text).map_err(contract_error))
        .transpose()
}

/// One validated contract document's identity and read-side fields, exactly
/// as the Rust parser sees them. This is the validation gate behind
/// `tensorfs.Contract`: construction serializes to a v1 document and runs THE
/// validator, so a malformed contract refuses at author-module import time
/// with the Rust message, never at deploy.
#[pyclass(frozen, get_all, module = "tensorfs._tensorfs", name = "ContractInfo")]
pub struct PyContractInfo {
    /// The library name, or `None` for an anonymous custom.
    name: Option<String>,
    version: Option<u32>,
    description: String,
    /// The optional top-level load dtype, torch spelling.
    dtype: Option<String>,
    /// Bare lowercase hex SHA-256 of the canonical rendering.
    digest: String,
    /// `name@version` for library documents, `sha256:<hex>` for customs.
    stamp: String,
}

#[pymethods]
impl PyContractInfo {
    fn __repr__(&self) -> String {
        format!("ContractInfo(stamp='{}')", self.stamp)
    }
}

/// Parses and validates one contract document, returning its identity.
#[pyfunction]
fn contract_info(document: &str) -> PyResult<PyContractInfo> {
    let parsed = contract::Contract::parse(document).map_err(contract_error)?;
    Ok(PyContractInfo {
        name: parsed.name().map(str::to_owned),
        version: (parsed.version() > 0).then_some(parsed.version()),
        description: parsed.description().to_owned(),
        dtype: parsed.dtype().map(str::to_owned),
        digest: hex_of(&parsed.digest()),
        stamp: parsed.stamp().to_string(),
    })
}

/// A contract argument: a plain string, or a `tensorfs.Contract`-shaped object
/// carrying the wanted text as an attribute (`document` where a full document
/// travels, `stamp` where only the identity does).
fn text_or_attr(any: &Bound<'_, PyAny>, attribute: &str) -> PyResult<String> {
    if let Ok(text) = any.extract::<String>() {
        return Ok(text);
    }
    any.getattr(attribute)?.extract::<String>()
}

// THE BIND VERDICT AND THE CONTRACT REGISTRY ARE DELETED (tensorfs#151).
//
// `contract_verdict` was a pyo3 port of `verdict.go` + `match.go`, added so a
// Python caller would not write a THIRD copy of the pattern matcher. v2 removes
// the reason rather than the copy: the verdict is computed ONCE, in Go, by the
// hub, and workers consume it binding-carried. A worker that recomputed an
// admit locally would be a second authority on whether a checkpoint may serve.
//
// What a worker still needs is IDENTITY FACTS — a quant rule's declared dtype
// and capability floor, a topology's key set — and those are DATA, read from
// the vendored `spec/v2/` documents. See `python/src/tensorfs/layout2.py`.

#[pyfunction]
fn ingest_concurrency() -> usize {
    tensorfs_core::store::ingest_concurrency()
}

/// Re-keys one committed tensor container, returning the composed record list.
///
/// `names` maps every tensor in the source to the name it keeps. Only the
/// rewritten header is admitted; every tensor record is inherited verbatim, so
/// nothing is re-chunked, re-hashed or re-uploaded.
#[pyfunction]
#[pyo3(signature = (store, planner, records, names, contract = None))]
fn rekey(
    py: Python<'_>,
    store: &PyObjectStore,
    planner: &str,
    records: Vec<PyFileRecord>,
    names: BTreeMap<String, String>,
    contract: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PyFileRecord>> {
    compose_records(
        py,
        store,
        planner,
        records,
        &names,
        contract,
        compose::rekey,
    )
}

/// Composes a trimmed container: exactly the tensors `names` maps survive.
///
/// The omitted tensors' records are omitted with them, so a sync of the
/// result never asks for their objects. Pass identity pairs to trim without
/// renaming.
#[pyfunction]
#[pyo3(signature = (store, planner, records, names, contract = None))]
fn subset(
    py: Python<'_>,
    store: &PyObjectStore,
    planner: &str,
    records: Vec<PyFileRecord>,
    names: BTreeMap<String, String>,
    contract: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PyFileRecord>> {
    compose_records(
        py,
        store,
        planner,
        records,
        &names,
        contract,
        compose::subset,
    )
}

/// Derives one committed container in a different layout contract.
///
/// Returns the derived record list, the target stamp, and how many data
/// objects were admitted: zero for the run-preserving majority (rename,
/// reshape, outer-axis fuse/split), and one tensor's worth for each tensor
/// the two contracts declare re-arranged.
#[pyfunction]
fn derive(
    py: Python<'_>,
    store: &PyObjectStore,
    planner: &str,
    records: Vec<PyFileRecord>,
    source_contract: &Bound<'_, PyAny>,
    target_contract: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyFileRecord>, String)> {
    let format = parse_planner(planner)?
        .tensor_format()
        .ok_or_else(|| TensorfsError::new_err("only a tensor container can be recomposed"))?;
    let records = records
        .iter()
        .map(PyFileRecord::to_core)
        .collect::<PyResult<Vec<_>>>()?;
    let source_document = text_or_attr(source_contract, "document")?;
    let target_document = text_or_attr(target_contract, "document")?;
    let source = contract::Contract::parse(&source_document).map_err(contract_error)?;
    let target = contract::Contract::parse(&target_document).map_err(contract_error)?;
    let body = tfm1::FileBody::Tensor {
        format,
        contract: source.stamp(),
        logical_size: records.iter().map(record_length).sum(),
        records,
    };
    let composed = py
        .detach(|| compose::derive(store.inner.as_ref(), &body, &source, &target))
        .map_err(compose_error)?;
    let stamp = match &composed {
        tfm1::FileBody::Tensor { contract, .. } => contract.to_string(),
        tfm1::FileBody::Blob { .. } => contract::Stamp::None.to_string(),
    };
    Ok((
        composed
            .records()
            .iter()
            .map(PyFileRecord::from_core)
            .collect(),
        stamp,
    ))
}

/// Re-chunks a committed tensor container under a layout contract, returning
/// the new record list and the stamp that explains its boundaries.
///
/// This is the `contract:none` -> contract upgrade: every object the contract
/// does not move is inherited by digest, so only the seam-affected tensors are
/// read and admitted.
#[pyfunction]
#[pyo3(signature = (store, planner, records, seams = None))]
fn adopt(
    py: Python<'_>,
    store: &PyObjectStore,
    planner: &str,
    records: Vec<PyFileRecord>,
    seams: Option<&str>,
) -> PyResult<(Vec<PyFileRecord>, String)> {
    let format = parse_planner(planner)?
        .tensor_format()
        .ok_or_else(|| TensorfsError::new_err("only a tensor container can be recomposed"))?;
    let records = records
        .iter()
        .map(PyFileRecord::to_core)
        .collect::<PyResult<Vec<_>>>()?;
    let body = tfm1::FileBody::Tensor {
        format,
        contract: contract::Stamp::None,
        logical_size: records.iter().map(record_length).sum(),
        records,
    };
    let registry = parse_seams(seams)?;
    let composed = py.detach(|| -> PyResult<tfm1::FileBody> {
        compose::adopt(store.inner.as_ref(), &body, registry.as_ref()).map_err(compose_error)
    })?;
    let stamp = match &composed {
        tfm1::FileBody::Tensor { contract, .. } => contract.to_string(),
        tfm1::FileBody::Blob { .. } => contract::Stamp::None.to_string(),
    };
    Ok((
        composed
            .records()
            .iter()
            .map(PyFileRecord::from_core)
            .collect(),
        stamp,
    ))
}

/// A composition over one committed tensor body: `compose::rekey` or
/// `compose::subset`.
type Composition = fn(
    &ObjectStore,
    &tfm1::FileBody,
    &BTreeMap<String, String>,
) -> Result<tfm1::FileBody, compose::ComposeError>;

fn compose_records(
    py: Python<'_>,
    store: &PyObjectStore,
    planner: &str,
    records: Vec<PyFileRecord>,
    names: &BTreeMap<String, String>,
    contract: Option<&Bound<'_, PyAny>>,
    composition: Composition,
) -> PyResult<Vec<PyFileRecord>> {
    let format = parse_planner(planner)?
        .tensor_format()
        .ok_or_else(|| TensorfsError::new_err("only a tensor container can be recomposed"))?;
    let records = records
        .iter()
        .map(PyFileRecord::to_core)
        .collect::<PyResult<Vec<_>>>()?;
    let contract = match contract {
        None => contract::Stamp::None,
        // A stamp string, or a Contract object whose `.stamp` spells one.
        Some(any) => {
            let text = text_or_attr(any, "stamp")?;
            contract::Stamp::parse(&text).map_err(contract_error)?
        }
    };
    let body = tfm1::FileBody::Tensor {
        format,
        contract,
        logical_size: records.iter().map(record_length).sum(),
        records,
    };
    let composed = py
        .detach(|| composition(&store.inner, &body, names))
        .map_err(compose_error)?;
    Ok(composed
        .records()
        .iter()
        .map(PyFileRecord::from_core)
        .collect())
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
    module.add_class::<PyMappedObject>()?;
    module.add_class::<PyObjectStore>()?;
    module.add_class::<PyRecordsReader>()?;
    module.add_class::<PyContractInfo>()?;
    module.add_class::<PyStreamTensor>()?;
    module.add_class::<PyFillStats>()?;
    module.add_class::<PyCudaFillClient>()?;
    module.add_class::<PyTensorStreamReader>()?;

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
    module.add_function(wrap_pyfunction!(contract_info, module)?)?;
    module.add_function(wrap_pyfunction!(rekey, module)?)?;
    module.add_function(wrap_pyfunction!(subset, module)?)?;
    module.add_function(wrap_pyfunction!(adopt, module)?)?;
    module.add_function(wrap_pyfunction!(derive, module)?)?;

    module.add("MAX_OBJECT_SIZE", planner::MAX_OBJECT_SIZE)?;
    module.add("MAX_PACK_PAYLOAD", tfp1::MAX_PACK_PAYLOAD)?;
    module.add("MAX_PACK_OBJECTS", tfp1::MAX_PACK_OBJECTS)?;
    module.add("MAX_FILE_RECORDS", tfm1::MAX_FILE_RECORDS)?;
    module.add("MAX_PATH_BYTES", tfm1::MAX_PATH_BYTES)?;
    module.add("MAX_SYMLINK_TARGET_BYTES", tfm1::MAX_SYMLINK_TARGET_BYTES)?;

    Ok(())
}
