//! A committed file's ordered records presented as one planner byte source.

use std::io::{self, Read, Seek, SeekFrom};

use crate::object::ObjectDigest;
use crate::planner::ByteSource;
use crate::store::ObjectStore;
use crate::tfm1::FileRecord;

/// Presents one committed file — ordered `Data`/`Hole` records over the
/// immutable object store — as a bounded random-access byte source, so the
/// closed planner registry can re-derive semantic boundaries from bounded
/// headers without materializing the file.
///
/// Stability argument for `check_unchanged`: the records are a value snapshot
/// of committed metadata, and every object they name is an immutable CAS
/// entry the committed head pins as a GC root, so the viewed bytes cannot
/// change for this source's lifetime. The no-op answer is therefore truthful,
/// not optimistic.
pub(crate) struct RecordsSource<'store> {
    store: &'store ObjectStore,
    /// `(start_offset, record)` in file order.
    segments: Vec<(u64, FileRecord)>,
    length: u64,
}

impl<'store> RecordsSource<'store> {
    pub(crate) fn new(store: &'store ObjectStore, records: &[FileRecord]) -> Self {
        let mut segments = Vec::with_capacity(records.len());
        let mut position = 0_u64;
        for record in records {
            segments.push((position, record.clone()));
            position += record_length(record);
        }
        Self {
            store,
            segments,
            length: position,
        }
    }

    fn read_object(
        &self,
        digest: &ObjectDigest,
        within: u64,
        destination: &mut [u8],
    ) -> io::Result<()> {
        let mut file = self.store.open_object(digest).map_err(io::Error::other)?;
        file.seek(SeekFrom::Start(within))?;
        file.read_exact(destination)
    }
}

const fn record_length(record: &FileRecord) -> u64 {
    match record {
        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
    }
}

impl ByteSource for RecordsSource<'_> {
    fn len(&self) -> u64 {
        self.length
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
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

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}
