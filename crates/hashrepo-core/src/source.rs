use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(not(any(unix, windows)))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(not(any(unix, windows)))]
use std::sync::Mutex;

use crate::planner::ByteSource;

/// A bounded, random-access byte source backed by an open regular file.
///
/// The file length is captured when the source is constructed. Reads use the
/// open handle rather than reopening its path, so replacing the path cannot
/// redirect an existing source to different bytes. Callers must still treat a
/// file being mutated through another handle as an I/O error boundary: a
/// truncation is reported as [`io::ErrorKind::UnexpectedEof`].
#[derive(Debug)]
pub struct FileByteSource {
    #[cfg(any(unix, windows))]
    file: File,
    #[cfg(not(any(unix, windows)))]
    file: Mutex<File>,
    length: u64,
}

impl FileByteSource {
    /// Opens `path` and captures its current length without reading its
    /// contents.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_file(File::open(path)?)
    }

    /// Creates a source from an already-open regular file and captures its
    /// current length without reading its contents.
    pub fn from_file(file: File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "byte source must be a regular file",
            ));
        }

        #[cfg(any(unix, windows))]
        {
            Ok(Self {
                file,
                length: metadata.len(),
            })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Ok(Self {
                file: Mutex::new(file),
                length: metadata.len(),
            })
        }
    }

    /// Returns the file length captured when this source was constructed.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.length
    }

    /// Returns whether the captured file length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[cfg(unix)]
    fn read_at(&self, destination: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(&self.file, destination, offset)
    }

    #[cfg(windows)]
    fn read_at(&self, destination: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(&self.file, destination, offset)
    }

    #[cfg(not(any(unix, windows)))]
    fn read_at(&self, destination: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        file.seek(SeekFrom::Start(offset))?;
        file.read(destination)
    }
}

impl ByteSource for FileByteSource {
    fn len(&self) -> u64 {
        self.len()
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
                "range exceeds the captured file length",
            ));
        }

        read_exact_with(destination, offset, |buffer, position| {
            self.read_at(buffer, position)
        })
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

fn read_exact_with(
    mut destination: &mut [u8],
    mut offset: u64,
    mut read_at: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
) -> io::Result<()> {
    while !destination.is_empty() {
        match read_at(destination, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "file was truncated during a positional read",
                ));
            }
            Ok(read) if read <= destination.len() => {
                offset = offset
                    .checked_add(u64::try_from(read).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "read length is too large")
                    })?)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
                destination = &mut destination[read..];
            }
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "positional read exceeded its destination",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> io::Result<Self> {
            loop {
                let nonce = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "hashrepo-file-byte-source-{}-{nonce}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Ok(Self(path)),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(error),
                }
            }
        }

        fn join(&self, path: impl AsRef<Path>) -> PathBuf {
            self.0.join(path)
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn positional_read_does_not_move_the_file_cursor() -> io::Result<()> {
        let directory = TempDirectory::new()?;
        let path = directory.join("source");
        fs::write(&path, b"0123456789")?;
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(9))?;
        let mut source = FileByteSource::from_file(file)?;

        let mut bytes = [0_u8; 4];
        source.read_exact_at(2, &mut bytes)?;

        assert_eq!(&bytes, b"2345");
        assert_eq!(source.file.stream_position()?, 9);
        Ok(())
    }

    #[test]
    fn concurrent_disjoint_reads_do_not_interfere() -> io::Result<()> {
        let directory = TempDirectory::new()?;
        let path = directory.join("source");
        let contents = (0_u32..4096).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        fs::write(&path, &contents)?;
        let source = Arc::new(FileByteSource::open(path)?);
        let barrier = Arc::new(Barrier::new(8));

        let readers = (0..8)
            .map(|reader| {
                let source = Arc::clone(&source);
                let barrier = Arc::clone(&barrier);
                let expected = contents[reader * 2048..(reader + 1) * 2048].to_vec();
                thread::spawn(move || -> io::Result<()> {
                    barrier.wait();
                    for _ in 0..128 {
                        let mut actual = vec![0_u8; expected.len()];
                        source.read_exact_at((reader * 2048) as u64, &mut actual)?;
                        assert_eq!(actual, expected);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();

        for reader in readers {
            reader.join().expect("reader thread panicked")?;
        }
        Ok(())
    }

    #[test]
    fn captured_length_does_not_grow_and_truncation_is_reported() -> io::Result<()> {
        let directory = TempDirectory::new()?;
        let path = directory.join("source");
        fs::write(&path, b"abcdefghij")?;
        let source = FileByteSource::open(&path)?;

        OpenOptions::new()
            .append(true)
            .open(&path)?
            .write_all(b"klmnop")?;
        assert_eq!(source.len(), 10);
        assert_eq!(
            source.read_exact_at(10, &mut [0_u8; 1]).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );

        OpenOptions::new().write(true).open(&path)?.set_len(4)?;
        let error = source.read_exact_at(2, &mut [0_u8; 6]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(source.len(), 10);
        Ok(())
    }

    #[test]
    fn replacing_the_path_does_not_redirect_an_open_source() -> io::Result<()> {
        let directory = TempDirectory::new()?;
        let path = directory.join("source");
        let original_path = directory.join("original");
        fs::write(&path, b"first")?;
        let source = FileByteSource::open(&path)?;

        fs::rename(&path, original_path)?;
        fs::write(&path, b"other")?;

        let mut bytes = [0_u8; 5];
        source.read_exact_at(0, &mut bytes)?;
        assert_eq!(&bytes, b"first");
        Ok(())
    }

    #[test]
    fn rejects_overflow_and_out_of_bounds_ranges() -> io::Result<()> {
        let directory = TempDirectory::new()?;
        let path = directory.join("source");
        fs::write(&path, b"1234")?;
        let source = FileByteSource::open(path)?;

        let overflow = source.read_exact_at(u64::MAX, &mut [0_u8; 2]).unwrap_err();
        assert_eq!(overflow.kind(), io::ErrorKind::InvalidInput);
        let out_of_bounds = source.read_exact_at(3, &mut [0_u8; 2]).unwrap_err();
        assert_eq!(out_of_bounds.kind(), io::ErrorKind::UnexpectedEof);
        source.read_exact_at(4, &mut [])?;
        assert_eq!(
            source.read_exact_at(5, &mut []).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        Ok(())
    }

    #[test]
    fn propagates_non_interrupted_read_errors() {
        let error = read_exact_with(&mut [0_u8; 1], 0, |_, _| {
            Err(io::Error::from_raw_os_error(5))
        })
        .unwrap_err();
        assert_eq!(error.raw_os_error(), Some(5));
    }

    #[test]
    fn retries_interrupted_and_partial_reads() -> io::Result<()> {
        let mut calls = 0;
        let mut destination = [0_u8; 4];
        read_exact_with(&mut destination, 11, |buffer, offset| {
            calls += 1;
            match calls {
                1 => Err(io::Error::from(io::ErrorKind::Interrupted)),
                2 => {
                    assert_eq!(offset, 11);
                    buffer[..2].copy_from_slice(b"ab");
                    Ok(2)
                }
                3 => {
                    assert_eq!(offset, 13);
                    buffer.copy_from_slice(b"cd");
                    Ok(2)
                }
                _ => unreachable!(),
            }
        })?;
        assert_eq!(&destination, b"abcd");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn reads_a_sparse_multi_gibibyte_offset_without_allocating_the_file() -> io::Result<()> {
        use std::os::unix::fs::MetadataExt;

        const OFFSET: u64 = 5 * 1024 * 1024 * 1024 + 123;

        let directory = TempDirectory::new()?;
        let path = directory.join("sparse");
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        file.seek(SeekFrom::Start(OFFSET))?;
        file.write_all(b"hashrepo")?;
        let metadata = file.metadata()?;
        assert_eq!(metadata.len(), OFFSET + 8);
        assert!(metadata.blocks() * 512 < 1024 * 1024);
        let mut source = FileByteSource::from_file(file)?;

        let mut bytes = [0_u8; 8];
        source.read_exact_at(OFFSET, &mut bytes)?;

        assert_eq!(&bytes, b"hashrepo");
        assert_eq!(source.file.stream_position()?, OFFSET + 8);
        Ok(())
    }
}
