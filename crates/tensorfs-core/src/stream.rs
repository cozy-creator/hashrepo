//! The streamed store→memory read engine: a committed tensor container's
//! records read sequentially into caller-owned buffers, buffered or O_DIRECT.
//!
//! This is the storage half of the no-fill serving ruling (2026-08-19): a
//! serving pytorch endpoint never materializes tensor files — it streams
//! tensor bytes from the chunk store straight into (typically CUDA-pinned)
//! host memory and hands them to the H2D pipeline. tensorfs yields bytes and
//! geometry; it neither knows nor cares what the destination memory is.
//!
//! Two read modes:
//!
//! - **Buffered** (the default): ordinary page-cached reads. The extra
//!   page-cache→buffer memcpy (~10+ GB/s) pipelines behind disk DMA
//!   (~2-3.5 GB/s NVMe), and the page cache is what makes warm loads and
//!   cross-checkpoint chunk sharing nearly free.
//! - **Direct** (`O_DIRECT`, Linux only): page cache bypassed. Whole-object
//!   reads land straight in the caller's buffer when it is 4096-aligned
//!   (CUDA-pinned memory is page-aligned); tails and unaligned destinations
//!   go through a bounded aligned bounce buffer. An unsupported filesystem
//!   surfaces the kernel's refusal — never a silent fallback to buffered.

use std::borrow::Borrow;
use std::io::{self, Read, Seek, SeekFrom};

use crate::object::ObjectDigest;
use crate::planner::ByteSource;
use crate::store::ObjectStore;
use crate::tfm1::FileRecord;

/// The alignment `Direct` mode requires of file offsets and satisfies for
/// destination pointers: one page, which also satisfies every block device's
/// logical block size.
pub const DIRECT_ALIGNMENT: usize = 4096;

/// The bounce buffer bound for unaligned `Direct` reads: large enough to
/// amortize syscalls, small enough to sit in cache comfortably.
const BOUNCE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadMode {
    Buffered,
    Direct,
}

/// A committed file's ordered records, readable at arbitrary offsets into
/// caller-owned memory. Holes read as zeros — zero-fill, never skip.
pub struct StreamReader<S> {
    store: S,
    /// `(start_offset, record)` in file order.
    segments: Vec<(u64, FileRecord)>,
    length: u64,
    mode: ReadMode,
}

const fn record_length(record: &FileRecord) -> u64 {
    match record {
        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
    }
}

impl<S: Borrow<ObjectStore>> StreamReader<S> {
    /// Builds a reader over one committed record run. `Direct` refuses off
    /// Linux at construction — a mode the platform cannot honor is a caller
    /// error, not a silent downgrade.
    pub fn new(store: S, records: &[FileRecord], mode: ReadMode) -> io::Result<Self> {
        if mode == ReadMode::Direct && !cfg!(target_os = "linux") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "O_DIRECT reads are a Linux capability; open buffered instead",
            ));
        }
        let mut segments = Vec::with_capacity(records.len());
        let mut position = 0_u64;
        for record in records {
            segments.push((position, record.clone()));
            position += record_length(record);
        }
        Ok(Self {
            store,
            segments,
            length: position,
            mode,
        })
    }

    #[must_use]
    pub const fn mode(&self) -> ReadMode {
        self.mode
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// Reads exactly `destination.len()` bytes starting at `offset`. Holes
    /// read as zeros; a range past the committed length refuses, never
    /// truncates.
    pub fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let length = u64::try_from(destination.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "destination length does not fit in a file offset",
            )
        })?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "range exceeds the committed length",
            ));
        }
        for (start, record) in &self.segments {
            let segment_end = start + record_length(record);
            if segment_end <= offset {
                continue;
            }
            if *start >= end {
                break;
            }
            let from = offset.max(*start);
            let to = end.min(segment_end);
            let slice = &mut destination[(from - offset) as usize..(to - offset) as usize];
            match record {
                FileRecord::Hole { .. } => slice.fill(0),
                FileRecord::Data { digest, .. } => self.read_object(digest, from - start, slice)?,
            }
        }
        Ok(())
    }

    fn read_object(
        &self,
        digest: &ObjectDigest,
        within: u64,
        destination: &mut [u8],
    ) -> io::Result<()> {
        match self.mode {
            ReadMode::Buffered => {
                let mut file = self
                    .store
                    .borrow()
                    .open_object(digest)
                    .map_err(io::Error::other)?;
                file.seek(SeekFrom::Start(within))?;
                file.read_exact(destination)
            }
            ReadMode::Direct => direct_read(self.store.borrow(), digest, within, destination),
        }
    }
}

/// The reader is also a planner byte source, so the header inventory that
/// names the tensors is recovered through the very same read path.
///
/// Stability: the records are a value snapshot of committed metadata over
/// immutable CAS objects, so the no-op `check_unchanged` answer is truthful.
impl<S: Borrow<ObjectStore>> ByteSource for StreamReader<S> {
    fn len(&self) -> u64 {
        self.length
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        Self::read_exact_at(self, offset, destination)
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
fn direct_read(
    _store: &ObjectStore,
    _digest: &ObjectDigest,
    _within: u64,
    _destination: &mut [u8],
) -> io::Result<()> {
    // Unreachable: `StreamReader::new` refuses Direct off Linux.
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

#[cfg(target_os = "linux")]
fn direct_read(
    store: &ObjectStore,
    digest: &ObjectDigest,
    within: u64,
    destination: &mut [u8],
) -> io::Result<()> {
    let file = store.open_object_direct(digest).map_err(io::Error::other)?;
    let align = DIRECT_ALIGNMENT as u64;

    // Fast path: aligned destination, aligned file offset. Whole aligned
    // blocks land straight in the caller's buffer; only the final partial
    // block goes through a one-block aligned over-read.
    if within.is_multiple_of(align)
        && (destination.as_ptr() as usize).is_multiple_of(DIRECT_ALIGNMENT)
    {
        let body = destination.len() - destination.len() % DIRECT_ALIGNMENT;
        read_full_at(&file, &mut destination[..body], within)?;
        let tail = destination.len() - body;
        if tail > 0 {
            let mut scratch = Aligned::new(DIRECT_ALIGNMENT);
            let got = read_up_to_at(&file, scratch.slice_mut(), within + body as u64)?;
            if got < tail {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            destination[body..].copy_from_slice(&scratch.slice_mut()[..tail]);
        }
        return Ok(());
    }

    // General path: a bounded aligned bounce buffer. Still O_DIRECT — the
    // page cache stays bypassed — at the cost of one memcpy.
    let mut scratch = Aligned::new(BOUNCE_BYTES);
    let mut cursor = within;
    let mut written = 0_usize;
    while written < destination.len() {
        let start = cursor - cursor % align;
        let skip = (cursor - start) as usize;
        let want = (destination.len() - written).min(BOUNCE_BYTES - skip);
        let need = skip + want;
        let read_len = need.next_multiple_of(DIRECT_ALIGNMENT).min(BOUNCE_BYTES);
        let got = read_up_to_at(&file, &mut scratch.slice_mut()[..read_len], start)?;
        if got < need {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        destination[written..written + want]
            .copy_from_slice(&scratch.slice_mut()[skip..skip + want]);
        written += want;
        cursor += want as u64;
    }
    Ok(())
}

/// Fills `destination` exactly, from `offset`, looping over short reads.
#[cfg(target_os = "linux")]
fn read_full_at(file: &std::fs::File, destination: &mut [u8], offset: u64) -> io::Result<()> {
    let got = read_up_to_at(file, destination, offset)?;
    if got < destination.len() {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    Ok(())
}

/// Reads as much of `destination` as the file holds from `offset`, looping
/// over short reads, and reports how much arrived. Short only at end of file.
#[cfg(target_os = "linux")]
fn read_up_to_at(file: &std::fs::File, destination: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;

    let mut filled = 0_usize;
    while filled < destination.len() {
        match file.read_at(&mut destination[filled..], offset + filled as u64) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// A page-aligned scratch buffer, built without unsafe: over-allocate one
/// alignment unit and start at the first aligned byte.
#[cfg(target_os = "linux")]
struct Aligned {
    raw: Vec<u8>,
    start: usize,
    length: usize,
}

#[cfg(target_os = "linux")]
impl Aligned {
    fn new(length: usize) -> Self {
        let raw = vec![0_u8; length + DIRECT_ALIGNMENT];
        let shift = raw.as_ptr() as usize % DIRECT_ALIGNMENT;
        let start = (DIRECT_ALIGNMENT - shift) % DIRECT_ALIGNMENT;
        Self { raw, start, length }
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.length]
    }
}
