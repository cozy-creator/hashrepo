//! TFP1: the canonical upload-staging envelope.
//!
//! One pack carries one or more whole missing TFM1 objects behind a sorted
//! `(digest, payload_offset, length)` index so a receiver can stream-verify
//! and admit each object without buffering more than one. TFP1 bounds upload
//! request count and nothing else: it never participates in snapshot or file
//! identity and is never retained as a local or remote storage format.

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::object::ObjectDigest;
use crate::planner::MAX_OBJECT_SIZE;

/// One pack's uncompressed payload bound, shared with the object bound.
pub const MAX_PACK_PAYLOAD: u64 = MAX_OBJECT_SIZE;
/// Mirrors the planner's bounded object cardinality; keeps index overhead
/// separately bounded from the payload.
pub const MAX_PACK_OBJECTS: usize = 1_000_000;

const MAGIC: [u8; 4] = *b"TFP1";
const ROW_BYTES: u64 = 32 + 8 + 8;

#[derive(Debug, Error)]
pub enum Tfp1Error {
    #[error("pack bytes are truncated")]
    Truncated,
    #[error("pack magic is not TFP1")]
    BadMagic,
    #[error("a pack must carry at least one object")]
    ZeroObjects,
    #[error("pack exceeds bounded object cardinality")]
    ObjectLimit,
    #[error("declared count exceeds the remaining input")]
    CountExceedsInput,
    #[error("index rows are not in strict ascending digest order")]
    IndexOrder,
    #[error("index declares the same digest twice")]
    DuplicateDigest,
    #[error("index declares a zero-length object")]
    ZeroLengthObject,
    #[error("index declares an object above the maximum object size")]
    ObjectTooLarge,
    #[error("pack payload exceeds the maximum pack size")]
    PayloadTooLarge,
    #[error("index offset does not tile the payload in index order")]
    OffsetMismatch,
    #[error("pack has trailing bytes")]
    TrailingBytes,
    #[error("object bytes do not hash to their indexed digest")]
    DigestMismatch,
}

impl Tfp1Error {
    /// Stable kebab-case label shared with the language-neutral vectors.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Truncated => "truncated",
            Self::BadMagic => "bad-magic",
            Self::ZeroObjects => "zero-objects",
            Self::ObjectLimit => "object-limit",
            Self::CountExceedsInput => "count-exceeds-input",
            Self::IndexOrder => "index-order",
            Self::DuplicateDigest => "duplicate-digest",
            Self::ZeroLengthObject => "zero-length-object",
            Self::ObjectTooLarge => "object-too-large",
            Self::PayloadTooLarge => "payload-too-large",
            Self::OffsetMismatch => "offset-mismatch",
            Self::TrailingBytes => "trailing-bytes",
            Self::DigestMismatch => "digest-mismatch",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexRow {
    digest: ObjectDigest,
    offset: u64,
    length: u64,
}

/// One whole object viewed inside a parsed pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackObject<'pack> {
    digest: ObjectDigest,
    bytes: &'pack [u8],
}

impl<'pack> PackObject<'pack> {
    #[must_use]
    pub const fn digest(&self) -> ObjectDigest {
        self.digest
    }

    #[must_use]
    pub const fn bytes(&self) -> &'pack [u8] {
        self.bytes
    }
}

/// One structurally validated pack over borrowed envelope bytes.
#[derive(Debug, Eq, PartialEq)]
pub struct Pack<'bytes> {
    rows: Vec<IndexRow>,
    payload: &'bytes [u8],
}

impl<'bytes> Pack<'bytes> {
    /// Validates the complete structure without hashing any payload byte.
    /// A receiver that wants verified objects calls [`Pack::verify_objects`]
    /// (or [`decode`], which does both).
    pub fn parse(bytes: &'bytes [u8]) -> Result<Self, Tfp1Error> {
        let mut reader = Reader { bytes, position: 0 };
        if reader.take(4)? != MAGIC {
            return Err(Tfp1Error::BadMagic);
        }
        let count = reader.take_u64()?;
        if count == 0 {
            return Err(Tfp1Error::ZeroObjects);
        }
        if count > MAX_PACK_OBJECTS as u64 {
            return Err(Tfp1Error::ObjectLimit);
        }
        if count > reader.remaining() / ROW_BYTES {
            return Err(Tfp1Error::CountExceedsInput);
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(count as usize)
            .map_err(|_| Tfp1Error::CountExceedsInput)?;

        let mut total = 0_u64;
        for _ in 0..count {
            let digest =
                ObjectDigest::from_bytes(reader.take(32)?.try_into().expect("32 bytes were taken"));
            let offset = reader.take_u64()?;
            let length = reader.take_u64()?;
            if length == 0 {
                return Err(Tfp1Error::ZeroLengthObject);
            }
            if length > MAX_OBJECT_SIZE {
                return Err(Tfp1Error::ObjectTooLarge);
            }
            if offset != total {
                return Err(Tfp1Error::OffsetMismatch);
            }
            if let Some(previous) = rows.last() {
                let previous: &IndexRow = previous;
                match previous.digest.as_bytes().cmp(digest.as_bytes()) {
                    std::cmp::Ordering::Less => {}
                    std::cmp::Ordering::Equal => return Err(Tfp1Error::DuplicateDigest),
                    std::cmp::Ordering::Greater => return Err(Tfp1Error::IndexOrder),
                }
            }
            total = total
                .checked_add(length)
                .ok_or(Tfp1Error::PayloadTooLarge)?;
            if total > MAX_PACK_PAYLOAD {
                return Err(Tfp1Error::PayloadTooLarge);
            }
            rows.push(IndexRow {
                digest,
                offset,
                length,
            });
        }

        let remaining = reader.remaining();
        if remaining < total {
            return Err(Tfp1Error::Truncated);
        }
        if remaining > total {
            return Err(Tfp1Error::TrailingBytes);
        }
        let payload = &bytes[reader.position..];
        Ok(Self { rows, payload })
    }

    #[must_use]
    pub fn object_count(&self) -> usize {
        self.rows.len()
    }

    /// Walks the index and hashes each object's payload bytes in turn, so a
    /// receiver never holds more than one object's hash state.
    pub fn verify_objects(&self) -> Result<(), Tfp1Error> {
        for object in self.objects() {
            let mut hasher = Sha256::new();
            hasher.update(object.bytes);
            let actual = ObjectDigest::from_bytes(hasher.finalize().into());
            if actual != object.digest {
                return Err(Tfp1Error::DigestMismatch);
            }
        }
        Ok(())
    }

    /// The whole objects in canonical index order. Slices borrow the envelope.
    pub fn objects(&self) -> impl Iterator<Item = PackObject<'bytes>> + '_ {
        let payload = self.payload;
        self.rows.iter().map(move |row| PackObject {
            digest: row.digest,
            bytes: &payload[row.offset as usize..(row.offset + row.length) as usize],
        })
    }
}

/// Strictly decodes and fully verifies one canonical pack: every structural
/// rule and every object digest.
pub fn decode(bytes: &[u8]) -> Result<Pack<'_>, Tfp1Error> {
    let pack = Pack::parse(bytes)?;
    pack.verify_objects()?;
    Ok(pack)
}

/// Builds one canonical pack from whole objects. The builder sorts the index,
/// refuses duplicates and bounds violations, and verifies every provided
/// digest against its bytes before emitting.
pub fn encode(objects: &[(ObjectDigest, &[u8])]) -> Result<Vec<u8>, Tfp1Error> {
    if objects.is_empty() {
        return Err(Tfp1Error::ZeroObjects);
    }
    if objects.len() > MAX_PACK_OBJECTS {
        return Err(Tfp1Error::ObjectLimit);
    }
    let mut total = 0_u64;
    for (digest, bytes) in objects {
        if bytes.is_empty() {
            return Err(Tfp1Error::ZeroLengthObject);
        }
        if bytes.len() as u64 > MAX_OBJECT_SIZE {
            return Err(Tfp1Error::ObjectTooLarge);
        }
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or(Tfp1Error::PayloadTooLarge)?;
        if total > MAX_PACK_PAYLOAD {
            return Err(Tfp1Error::PayloadTooLarge);
        }
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        if ObjectDigest::from_bytes(hasher.finalize().into()) != *digest {
            return Err(Tfp1Error::DigestMismatch);
        }
    }

    let mut sorted: Vec<&(ObjectDigest, &[u8])> = objects.iter().collect();
    sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    for pair in sorted.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(Tfp1Error::DuplicateDigest);
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    let mut offset = 0_u64;
    for (digest, object) in &sorted {
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&(object.len() as u64).to_le_bytes());
        offset += object.len() as u64;
    }
    for (_, object) in &sorted {
        bytes.extend_from_slice(object);
    }
    Ok(bytes)
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], Tfp1Error> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(Tfp1Error::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(Tfp1Error::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn take_u64(&mut self) -> Result<u64, Tfp1Error> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes were taken"),
        ))
    }

    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.position) as u64
    }
}
