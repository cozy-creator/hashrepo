//! The GPU leg of the fill path: pinned staging, one contiguous H2D, into a
//! destination pointer THE CALLER OWNS.
//!
//! A SEPARATE CRATE on purpose. `tensorfs-core` is `#![forbid(unsafe_code)]`,
//! and the crate that decides which byte goes where keeps that proof: every
//! `unsafe` in this program is in this file. The driver is resolved by `dlopen`
//! at first use — the same discipline varena's `cuda.rs` uses — so this crate
//! builds with no CUDA toolkit present and a box with no card gets a typed,
//! recoverable error rather than a link failure.
//!
//! IT IS THE REFERENCE, NOT THE FLEET'S SINK. varena's refill plane already
//! owns the production byte-motion mechanics (io_uring on the disk leg,
//! budgeted pinned slabs, async H2D on a copy stream) and is the `FillSink`
//! a worker uses. This crate exists so tensorfs can PROVE its own device leg
//! on its own card tests — both backends byte-equal — instead of asserting it.
//!
//! THIS MODULE NEVER ALLOCATES DEVICE MEMORY. `cuMemAlloc` is not called and
//! not wrapped. varena owns the address space; a destination pointer arrives
//! from the caller and this code writes to it. The symbols are exposed on
//! [`Driver`] so a test can play varena and allocate its own scratch — in the
//! test, deliberately, not in the crate.

use std::ffi::{c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::Library;

use tensorfs_core::layout_fill::FillSink;
use tensorfs_core::layout_morphism::LayoutError;

/// A device pointer, as the driver spells it.
pub type CUdeviceptr = u64;

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
/// `cuMemHostAlloc` flag: the allocation is page-locked and portable across
/// contexts. Pinned is the point — a pageable staging buffer makes the H2D
/// synchronous and halves the bandwidth.
const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x01;

/// The driver symbols this crate uses, resolved once.
///
/// Deliberately short. Every symbol here is on the path from a staged host
/// buffer to a caller's device pointer; nothing here reserves, backs, frees or
/// sizes device memory, because none of that is tensorfs's to decide.
pub struct Driver {
    _library: Library,
    pub init: unsafe extern "C" fn(c_uint) -> CUresult,
    pub device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CUresult,
    pub primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CUresult,
    pub ctx_set_current: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub mem_free_host: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    /// Device-side allocation, exposed for CALLERS AND TESTS ONLY. The fill
    /// path does not call it: varena owns the address space.
    pub mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    /// As above — a test that plays varena has to clean up after itself.
    pub mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    /// As above — reading the device back is how a test byte-compares.
    pub memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

fn driver_error(detail: String) -> LayoutError {
    LayoutError::Record {
        handle: "cuda".into(),
        detail,
    }
}

/// The CUDA driver, resolved on first use.
///
/// Absence is a TYPED ERROR and never a panic: a box with no card is a normal
/// box, and the CPU backend is a complete answer on it.
pub fn driver() -> Result<&'static Driver, LayoutError> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    match DRIVER.get_or_init(load) {
        Ok(driver) => Ok(driver),
        Err(detail) => Err(driver_error(detail.clone())),
    }
}

fn load() -> Result<Driver, String> {
    unsafe {
        let library = Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|error| format!("libcuda.so.1 did not load: {error}"))?;
        macro_rules! symbol {
            ($name:literal) => {
                *library
                    .get($name)
                    .map_err(|error| format!("{}: {error}", stringify!($name)))?
            };
        }
        let driver = Driver {
            init: symbol!(b"cuInit\0"),
            device_get: symbol!(b"cuDeviceGet\0"),
            primary_ctx_retain: symbol!(b"cuDevicePrimaryCtxRetain\0"),
            ctx_set_current: symbol!(b"cuCtxSetCurrent\0"),
            mem_host_alloc: symbol!(b"cuMemHostAlloc\0"),
            mem_free_host: symbol!(b"cuMemFreeHost\0"),
            memcpy_htod: symbol!(b"cuMemcpyHtoD_v2\0"),
            mem_alloc: symbol!(b"cuMemAlloc_v2\0"),
            mem_free: symbol!(b"cuMemFree_v2\0"),
            memcpy_dtoh: symbol!(b"cuMemcpyDtoH_v2\0"),
            _library: library,
        };
        Ok(driver)
    }
}

/// Bind the calling thread to `device`'s primary context.
///
/// The primary context is RETAINED, never created: a worker process shares one
/// context with torch and everything else in it, and a second context would
/// mean the pointers varena hands over are not addressable here.
pub fn bind(device: c_int) -> Result<(), LayoutError> {
    let cuda = driver()?;
    unsafe {
        check(cuda, (cuda.init)(0), "cuInit")?;
        let mut ordinal: c_int = 0;
        check(cuda, (cuda.device_get)(&mut ordinal, device), "cuDeviceGet")?;
        let mut context: *mut c_void = std::ptr::null_mut();
        check(
            cuda,
            (cuda.primary_ctx_retain)(&mut context, ordinal),
            "cuDevicePrimaryCtxRetain",
        )?;
        check(cuda, (cuda.ctx_set_current)(context), "cuCtxSetCurrent")
    }
}

fn check(_cuda: &Driver, status: CUresult, call: &str) -> Result<(), LayoutError> {
    if status == CUDA_SUCCESS {
        return Ok(());
    }
    Err(driver_error(format!("{call} returned {status}")))
}

/// The GPU fill sink: a pinned host staging buffer, and one contiguous H2D per
/// tensor into `base + device_offset`.
///
/// `base` comes from the caller. On this fleet that is a varena reservation's
/// `base_ptr()`, and the offsets are varena's too. This type does not know how
/// the address was obtained and must not: an address space it could allocate
/// from is an address space two owners are paging.
pub struct CudaSink {
    base: CUdeviceptr,
    destination_bytes: usize,
    staging: *mut c_void,
    staging_bytes: usize,
    staged: usize,
    /// Byte offset of the staged tensor within `base`.
    offset: u64,
    /// Bytes moved to the device so far — the number a bench reads.
    pub bytes_to_device: u64,
}

unsafe impl Send for CudaSink {}

impl CudaSink {
    /// Allocate `staging_bytes` of PINNED host memory and target `base`.
    ///
    /// Host memory, not device memory: the pinned staging buffer is the thing
    /// tensorfs is allowed to own, because it is the buffer the transform
    /// writes into on its way past.
    pub fn new(
        base: CUdeviceptr,
        destination_bytes: usize,
        staging_bytes: usize,
    ) -> Result<Self, LayoutError> {
        let cuda = driver()?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(
                cuda,
                (cuda.mem_host_alloc)(&mut pointer, staging_bytes, CU_MEMHOSTALLOC_PORTABLE),
                "cuMemHostAlloc",
            )?;
        }
        Ok(Self {
            base,
            destination_bytes,
            staging: pointer,
            staging_bytes,
            staged: 0,
            offset: 0,
            bytes_to_device: 0,
        })
    }

    /// The pinned staging capacity, in bytes.
    pub fn capacity(&self) -> usize {
        self.staging_bytes
    }

    /// Point the reusable staging allocation at a new caller-owned address.
    ///
    /// Retargeting moves no bytes and allocates nothing. A client can fill a
    /// whole destination map through one pinned slab while varena remains the
    /// sole owner of every destination address.
    pub fn retarget(
        &mut self,
        base: CUdeviceptr,
        destination_bytes: usize,
    ) -> Result<(), LayoutError> {
        if self.staged != 0 {
            return Err(LayoutError::Record {
                handle: "cuda".into(),
                detail: "cannot retarget while a staged transfer is pending".into(),
            });
        }
        self.base = base;
        self.destination_bytes = destination_bytes;
        Ok(())
    }
}

impl FillSink for CudaSink {
    fn staging(&mut self, bytes: usize, device_offset: u64) -> Result<&mut [u8], LayoutError> {
        if bytes > self.staging_bytes {
            // A tensor bigger than the staging buffer is a REFUSAL, not a
            // silent split: splitting it here would put a second chunking
            // policy in a crate that already has one, and the caller sized this
            // buffer against a budget it owns.
            return Err(LayoutError::Buffer {
                got: self.staging_bytes,
                want: bytes,
            });
        }
        let end = usize::try_from(device_offset)
            .ok()
            .and_then(|offset| offset.checked_add(bytes))
            .ok_or(LayoutError::Buffer {
                got: self.destination_bytes,
                want: usize::MAX,
            })?;
        if end > self.destination_bytes {
            return Err(LayoutError::Buffer {
                got: self.destination_bytes,
                want: end,
            });
        }
        self.staged = bytes;
        self.offset = device_offset;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.staging as *mut u8, bytes) })
    }

    fn commit(&mut self) -> Result<(), LayoutError> {
        if self.staged == 0 {
            return Ok(());
        }
        let cuda = driver()?;
        unsafe {
            check(
                cuda,
                (cuda.memcpy_htod)(self.base + self.offset, self.staging, self.staged),
                "cuMemcpyHtoD",
            )?;
        }
        self.bytes_to_device += self.staged as u64;
        self.staged = 0;
        Ok(())
    }
}

impl Drop for CudaSink {
    fn drop(&mut self) {
        if self.staging.is_null() {
            return;
        }
        if let Ok(cuda) = driver() {
            unsafe {
                let _ = (cuda.mem_free_host)(self.staging);
            }
        }
        self.staging = std::ptr::null_mut();
    }
}
