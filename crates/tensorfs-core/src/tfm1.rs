//! TFM1: the canonical binary snapshot manifest.
//!
//! One byte encoding is canonical. The builder is the sole writer and the
//! decoder refuses every deviation, so `decode(bytes)` followed by
//! `Snapshot::to_bytes` reproduces the input exactly. `SnapshotId` is the
//! SHA-256 of the canonical bytes; nothing platform-derived can enter them.
//!
//! File bodies come in exactly two shapes: tensor containers
//! (`safetensors-v1`, `gguf-v1`) carry an ordered record list on the 64 MiB
//! tensor grid, and everything else is ONE unchunked `blob-v1` whose body is
//! its logical size and whole-file SHA-256. A chunked blob, a blob with a
//! hole, or a multi-record blob is unrepresentable, not refused.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

use sha2::{Digest, Sha256};
use thiserror::Error;
use unicode_normalization::is_nfc;

use crate::contract::{MAX_CONTRACT_NAME_BYTES, Stamp};
use crate::object::ObjectDigest;
use crate::planner::{MAX_OBJECT_SIZE, PlannerId, TensorFormat};

pub const MAX_PATH_BYTES: usize = 4096;
pub const MAX_SYMLINK_TARGET_BYTES: usize = 4096;
/// Mirrors the planner's bounded object cardinality for one regular file.
pub const MAX_FILE_RECORDS: usize = 1_000_000;

/// SHA-256 of the empty byte string: the one canonical digest of an empty
/// blob, so every file body has the same shape and zero-size bodies cannot
/// smuggle a second identity.
pub const EMPTY_BLOB_DIGEST: ObjectDigest = ObjectDigest::from_bytes([
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
    0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
]);

const MAGIC: [u8; 4] = *b"TFM1";
const KIND_DIRECTORY: u8 = 1;
const KIND_FILE: u8 = 2;
const KIND_SYMLINK: u8 = 3;
const KIND_HARDLINK: u8 = 4;
const RECORD_DATA: u8 = 1;
const RECORD_HOLE: u8 = 2;
const PLANNER_SAFETENSORS: u8 = 1;
const PLANNER_GGUF: u8 = 2;
const PLANNER_BLOB: u8 = 4;
/// Minimum encoded sizes used to bound declared counts before allocation.
const MIN_ENTRY_BYTES: u64 = 4 + 1 + 1;
const MIN_RECORD_BYTES: u64 = 1 + 8;

const WINDOWS_RESERVED: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct SnapshotId([u8; 32]);

impl SnapshotId {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The identity of one canonical TFM1 encoding: exactly its SHA-256.
    #[must_use]
    pub fn of(canonical_bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(canonical_bytes);
        Self(hasher.finalize().into())
    }

    #[must_use]
    pub fn parse_hex(text: &str) -> Option<Self> {
        if text.len() != 64 {
            return None;
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
            let digits = std::str::from_utf8(pair).ok()?;
            if digits.bytes().any(|byte| byte.is_ascii_uppercase()) {
                return None;
            }
            bytes[index] = u8::from_str_radix(digits, 16).ok()?;
        }
        Some(Self(bytes))
    }
}

impl fmt::Debug for SnapshotId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for SnapshotId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FileRecord {
    Data { digest: ObjectDigest, length: u64 },
    Hole { length: u64 },
}

impl FileRecord {
    const fn length(&self) -> u64 {
        match self {
            Self::Data { length, .. } | Self::Hole { length } => *length,
        }
    }
}

/// One regular file's canonical content description.
///
/// Tensor containers carry their ordered record list; everything else is one
/// whole blob named by its own SHA-256. The invalid combinations — a chunked
/// blob, a record list without a tensor format — cannot be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FileBody {
    Tensor {
        format: TensorFormat,
        /// Which layout contract directed this file's chunking. Part of the
        /// canonical bytes, so identity is self-describing: the same file
        /// chunked under two contracts is two snapshots, and reading one tells
        /// you which layout it is without consulting any registry.
        contract: Stamp,
        logical_size: u64,
        records: Vec<FileRecord>,
    },
    Blob {
        logical_size: u64,
        digest: ObjectDigest,
    },
}

impl FileBody {
    #[must_use]
    pub const fn logical_size(&self) -> u64 {
        match self {
            Self::Tensor { logical_size, .. } | Self::Blob { logical_size, .. } => *logical_size,
        }
    }

    #[must_use]
    pub const fn planner_id(&self) -> PlannerId {
        match self {
            Self::Tensor { format, .. } => format.planner_id(),
            Self::Blob { .. } => PlannerId::BlobV1,
        }
    }

    /// This file's identity: SHA-256 over the body's canonical encoding —
    /// exactly the bytes the manifest carries for it, planner tag included.
    ///
    /// TFM1 deliberately has no whole-file hash for a tensor container (its
    /// identity IS its record list), so this is the only file-level digest
    /// derivable from a manifest entry without reading the file's bytes. A
    /// pointer stub carries it for that reason.
    #[must_use]
    pub fn body_sha256(&self) -> ObjectDigest {
        let mut bytes = Vec::new();
        emit_body(&mut bytes, self);
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        ObjectDigest::from_bytes(hasher.finalize().into())
    }

    /// The body as an ordered record run. A tensor body is its records; a
    /// nonempty blob is one whole-file data record; an empty blob is none.
    /// This is the one read-path vocabulary, so record consumers need no blob
    /// special case.
    #[must_use]
    pub fn records(&self) -> Cow<'_, [FileRecord]> {
        match self {
            Self::Tensor { records, .. } => Cow::Borrowed(records.as_slice()),
            Self::Blob {
                logical_size: 0, ..
            } => Cow::Owned(Vec::new()),
            Self::Blob {
                logical_size,
                digest,
            } => Cow::Owned(vec![FileRecord::Data {
                digest: *digest,
                length: *logical_size,
            }]),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Entry {
    Directory,
    File { executable: bool, body: FileBody },
    Symlink { target: String },
    Hardlink { ordinal: u64 },
}

/// One entry of the committed-but-unsealed workspace head tree.
///
/// The mutable workspace stages every file as a record run (the COW mount
/// composes on the 64 MiB grid), a shape the canonical manifest deliberately
/// cannot express for non-tensor files. This is that staging vocabulary: the
/// same tree facts as [`Entry`], with file content still in record form.
/// Sealing is what canonicalizes it into TFM1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TreeEntry {
    Directory,
    File {
        executable: bool,
        planner: PlannerId,
        contract: Stamp,
        records: Vec<FileRecord>,
    },
    Symlink {
        target: String,
    },
    Hardlink {
        ordinal: u64,
    },
}

/// The resolved, validated workspace head tree: every path, ordering,
/// hardlink and topology rule of TFM1 holds, but file bodies stay in staging
/// record form. Produced only by [`SnapshotBuilder::finish_tree`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeadTree {
    entries: Vec<(String, TreeEntry)>,
}

impl HeadTree {
    #[must_use]
    pub fn entries(&self) -> &[(String, TreeEntry)] {
        &self.entries
    }

    /// Resolves a hardlink ordinal to the carrier path, in ordinal order.
    #[must_use]
    pub fn ordinal_path(&self, ordinal: u64) -> Option<&str> {
        self.entries
            .iter()
            .filter(|(_, entry)| matches!(entry, TreeEntry::File { .. }))
            .nth(usize::try_from(ordinal).ok()?)
            .map(|(path, _)| path.as_str())
    }
}

#[derive(Debug, Error)]
pub enum Tfm1Error {
    #[error("manifest bytes are truncated")]
    Truncated,
    #[error("manifest magic is not TFM1")]
    BadMagic,
    #[error("parent flag must be 0 or 1")]
    ParentFlag,
    #[error("declared count exceeds the remaining input")]
    CountExceedsInput,
    #[error("manifest has trailing bytes")]
    TrailingBytes,
    #[error("entries are not in strict canonical path order")]
    EntryOrder,
    #[error("path appears more than once")]
    DuplicatePath,
    #[error("path is not valid UTF-8")]
    PathUtf8,
    #[error("path is empty or exceeds the bounded length")]
    PathLength,
    #[error("path is not NFC-normalized")]
    PathNotNfc,
    #[error("path is absolute")]
    PathAbsolute,
    #[error("path has an empty segment")]
    PathEmptySegment,
    #[error("path has a dot or dot-dot segment")]
    PathDotSegment,
    #[error("path contains a forbidden character")]
    PathForbiddenCharacter,
    #[error("path segment ends with a dot or a space")]
    PathTrailingDotSpace,
    #[error("path segment is a Windows-reserved device name")]
    PathWindowsReserved,
    #[error("paths collide under case folding")]
    CaseFoldCollision,
    #[error("entry parent is missing or is not a directory")]
    MissingParentDirectory,
    #[error("entry kind tag is unknown")]
    UnknownEntryKind,
    #[error("executable flag must be 0 or 1")]
    ExecutableFlag,
    #[error("planner tag is unknown")]
    UnknownPlanner,
    #[error("contract name is empty, too long, or outside the handle charset")]
    ContractName,
    #[error("contract version disagrees with the presence of a contract name")]
    ContractVersion,
    #[error("file record tag is unknown")]
    UnknownRecordTag,
    #[error("file record has zero length")]
    ZeroLengthRecord,
    #[error("file has adjacent hole records")]
    AdjacentHoles,
    #[error("data record exceeds the maximum object size")]
    DataTooLarge,
    #[error("file exceeds bounded record cardinality")]
    RecordLimit,
    #[error("record lengths overflow or do not sum to the logical size")]
    LengthSumMismatch,
    #[error("an empty blob must carry the digest of the empty byte string")]
    EmptyBlobDigest,
    #[error("a blob body is exactly one whole-file object or empty")]
    BlobRecords,
    #[error("symlink target is empty, too long, or contains forbidden bytes")]
    SymlinkTarget,
    #[error("hardlink ordinal does not name a previously assigned file")]
    HardlinkOrdinal,
    #[error("hardlink target is not a regular file entry")]
    HardlinkTarget,
    #[error("manifest could not reserve bounded working memory")]
    ResourceExhausted,
}

impl Tfm1Error {
    /// Stable kebab-case label shared with the language-neutral vectors.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Truncated => "truncated",
            Self::BadMagic => "bad-magic",
            Self::ParentFlag => "parent-flag",
            Self::CountExceedsInput => "count-exceeds-input",
            Self::TrailingBytes => "trailing-bytes",
            Self::EntryOrder => "entry-order",
            Self::DuplicatePath => "duplicate-path",
            Self::PathUtf8 => "path-utf8",
            Self::PathLength => "path-length",
            Self::PathNotNfc => "path-not-nfc",
            Self::PathAbsolute => "path-absolute",
            Self::PathEmptySegment => "path-empty-segment",
            Self::PathDotSegment => "path-dot-segment",
            Self::PathForbiddenCharacter => "path-forbidden-character",
            Self::PathTrailingDotSpace => "path-trailing-dot-space",
            Self::PathWindowsReserved => "path-windows-reserved",
            Self::CaseFoldCollision => "case-fold-collision",
            Self::MissingParentDirectory => "missing-parent-directory",
            Self::UnknownEntryKind => "unknown-entry-kind",
            Self::ExecutableFlag => "executable-flag",
            Self::UnknownPlanner => "unknown-planner",
            Self::ContractName => "contract-name",
            Self::ContractVersion => "contract-version",
            Self::UnknownRecordTag => "unknown-record-tag",
            Self::ZeroLengthRecord => "zero-length-record",
            Self::AdjacentHoles => "adjacent-holes",
            Self::DataTooLarge => "data-too-large",
            Self::RecordLimit => "record-limit",
            Self::LengthSumMismatch => "length-sum-mismatch",
            Self::EmptyBlobDigest => "empty-blob-digest",
            Self::BlobRecords => "blob-records",
            Self::SymlinkTarget => "symlink-target",
            Self::HardlinkOrdinal => "hardlink-ordinal",
            Self::HardlinkTarget => "hardlink-target",
            Self::ResourceExhausted => "resource-exhausted",
        }
    }
}

const fn planner_tag(body: &FileBody) -> u8 {
    match body {
        FileBody::Tensor {
            format: TensorFormat::SafetensorsV1,
            ..
        } => PLANNER_SAFETENSORS,
        FileBody::Tensor {
            format: TensorFormat::GgufV1,
            ..
        } => PLANNER_GGUF,
        FileBody::Blob { .. } => PLANNER_BLOB,
    }
}

pub(crate) fn validate_path(path: &str) -> Result<(), Tfm1Error> {
    if path.is_empty() || path.len() > MAX_PATH_BYTES {
        return Err(Tfm1Error::PathLength);
    }
    if !is_nfc(path) {
        return Err(Tfm1Error::PathNotNfc);
    }
    if path.starts_with('/') {
        return Err(Tfm1Error::PathAbsolute);
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            return Err(Tfm1Error::PathEmptySegment);
        }
        if segment == "." || segment == ".." {
            return Err(Tfm1Error::PathDotSegment);
        }
        for character in segment.chars() {
            if character.is_control()
                || matches!(character, '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
            {
                return Err(Tfm1Error::PathForbiddenCharacter);
            }
        }
        if segment.ends_with('.') || segment.ends_with(' ') {
            return Err(Tfm1Error::PathTrailingDotSpace);
        }
        let stem = segment
            .split('.')
            .next()
            .expect("split yields a first part");
        if stem.len() <= 4
            && WINDOWS_RESERVED
                .iter()
                .any(|reserved| stem.eq_ignore_ascii_case(reserved))
        {
            return Err(Tfm1Error::PathWindowsReserved);
        }
    }
    Ok(())
}

pub(crate) fn validate_symlink_target(target: &str) -> Result<(), Tfm1Error> {
    if target.is_empty()
        || target.len() > MAX_SYMLINK_TARGET_BYTES
        || target.chars().any(char::is_control)
    {
        return Err(Tfm1Error::SymlinkTarget);
    }
    Ok(())
}

/// The structural rules every record run obeys regardless of planner: bounded
/// cardinality, no zero-length records, no adjacent holes, and an exact
/// non-overflowing length sum. This is the staging-domain rule — the 64 MiB
/// data-record cap belongs only to tensor bodies.
pub(crate) fn validate_record_run(
    records: &[FileRecord],
    logical_size: u64,
) -> Result<(), Tfm1Error> {
    if records.len() > MAX_FILE_RECORDS {
        return Err(Tfm1Error::RecordLimit);
    }
    let mut sum = 0_u64;
    let mut previous_was_hole = false;
    for record in records {
        if record.length() == 0 {
            return Err(Tfm1Error::ZeroLengthRecord);
        }
        match record {
            FileRecord::Data { .. } => previous_was_hole = false,
            FileRecord::Hole { .. } => {
                if previous_was_hole {
                    return Err(Tfm1Error::AdjacentHoles);
                }
                previous_was_hole = true;
            }
        }
        sum = sum
            .checked_add(record.length())
            .ok_or(Tfm1Error::LengthSumMismatch)?;
    }
    if sum != logical_size {
        return Err(Tfm1Error::LengthSumMismatch);
    }
    Ok(())
}

/// The tensor-body rule: the structural run rules plus the tensor chunk grid
/// cap on every data record.
pub(crate) fn validate_records(records: &[FileRecord], logical_size: u64) -> Result<(), Tfm1Error> {
    validate_record_run(records, logical_size)?;
    for record in records {
        if let FileRecord::Data { length, .. } = record
            && *length > MAX_OBJECT_SIZE
        {
            return Err(Tfm1Error::DataTooLarge);
        }
    }
    Ok(())
}

pub(crate) fn case_fold(path: &str) -> String {
    path.chars().flat_map(char::to_lowercase).collect()
}

/// Validates the whole-tree facts every canonical manifest must satisfy:
/// case-fold uniqueness and explicit parent directories for every entry.
fn validate_tree<E>(
    entries: &[(String, E)],
    is_directory: impl Fn(&E) -> bool,
) -> Result<(), Tfm1Error> {
    let mut folded = HashSet::new();
    for (path, _) in entries {
        if !folded.insert(case_fold(path)) {
            return Err(Tfm1Error::CaseFoldCollision);
        }
    }
    let directories: HashSet<&str> = entries
        .iter()
        .filter(|(_, entry)| is_directory(entry))
        .map(|(path, _)| path.as_str())
        .collect();
    for (path, _) in entries {
        if let Some((parent, _)) = path.rsplit_once('/')
            && !directories.contains(parent)
        {
            return Err(Tfm1Error::MissingParentDirectory);
        }
    }
    Ok(())
}

/// One file body's canonical encoding: the planner tag and the body per
/// planner. Shared by the manifest writer and [`FileBody::body_sha256`], so a
/// stub's digest is provably over the same bytes the manifest carries.
fn emit_body(bytes: &mut Vec<u8>, body: &FileBody) {
    bytes.push(planner_tag(body));
    match body {
        FileBody::Tensor {
            contract,
            logical_size,
            records,
            ..
        } => {
            bytes.extend_from_slice(&logical_size.to_le_bytes());
            let name = contract.name().unwrap_or_default();
            bytes.push(name.len() as u8);
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&contract.version().to_le_bytes());
            bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
            for record in records {
                match record {
                    FileRecord::Data { digest, length } => {
                        bytes.push(RECORD_DATA);
                        bytes.extend_from_slice(digest.as_bytes());
                        bytes.extend_from_slice(&length.to_le_bytes());
                    }
                    FileRecord::Hole { length } => {
                        bytes.push(RECORD_HOLE);
                        bytes.extend_from_slice(&length.to_le_bytes());
                    }
                }
            }
        }
        FileBody::Blob {
            logical_size,
            digest,
        } => {
            bytes.extend_from_slice(&logical_size.to_le_bytes());
            bytes.extend_from_slice(digest.as_bytes());
        }
    }
}

fn emit(parent: Option<&SnapshotId>, entries: &[(String, Entry)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    match parent {
        None => bytes.push(0),
        Some(parent) => {
            bytes.push(1);
            bytes.extend_from_slice(parent.as_bytes());
        }
    }
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (path, entry) in entries {
        bytes.extend_from_slice(&(path.len() as u32).to_le_bytes());
        bytes.extend_from_slice(path.as_bytes());
        match entry {
            Entry::Directory => bytes.push(KIND_DIRECTORY),
            Entry::File { executable, body } => {
                bytes.push(KIND_FILE);
                bytes.push(u8::from(*executable));
                emit_body(&mut bytes, body);
            }
            Entry::Symlink { target } => {
                bytes.push(KIND_SYMLINK);
                bytes.extend_from_slice(&(target.len() as u32).to_le_bytes());
                bytes.extend_from_slice(target.as_bytes());
            }
            Entry::Hardlink { ordinal } => {
                bytes.push(KIND_HARDLINK);
                bytes.extend_from_slice(&ordinal.to_le_bytes());
            }
        }
    }
    bytes
}

/// One decoded, validated snapshot. Values exist only through the builder or
/// `decode`, so re-encoding always reproduces the canonical bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    parent: Option<SnapshotId>,
    entries: Vec<(String, Entry)>,
}

impl Snapshot {
    #[must_use]
    pub const fn parent(&self) -> Option<&SnapshotId> {
        self.parent.as_ref()
    }

    #[must_use]
    pub fn entries(&self) -> &[(String, Entry)] {
        &self.entries
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        emit(self.parent.as_ref(), &self.entries)
    }

    #[must_use]
    pub fn snapshot_id(&self) -> SnapshotId {
        SnapshotId::of(&self.to_bytes())
    }

    /// Resolves a hardlink ordinal to the carrier path, in ordinal order.
    #[must_use]
    pub fn ordinal_path(&self, ordinal: u64) -> Option<&str> {
        self.entries
            .iter()
            .filter(|(_, entry)| matches!(entry, Entry::File { .. }))
            .nth(usize::try_from(ordinal).ok()?)
            .map(|(path, _)| path.as_str())
    }
}

enum Pending {
    Directory,
    File {
        executable: bool,
        planner: PlannerId,
        contract: Stamp,
        records: Vec<FileRecord>,
    },
    Symlink {
        target: String,
    },
    HardlinkTo {
        target_path: String,
    },
}

/// The sole TFM1 writer. Inputs are pure tree facts; no host identity, inode
/// number, timestamp, or random value can reach the canonical bytes.
#[derive(Default)]
pub struct SnapshotBuilder {
    parent: Option<SnapshotId>,
    pending: Vec<(String, Pending)>,
}

impl SnapshotBuilder {
    #[must_use]
    pub fn new(parent: Option<SnapshotId>) -> Self {
        Self {
            parent,
            pending: Vec::new(),
        }
    }

    pub fn directory(&mut self, path: impl Into<String>) -> &mut Self {
        self.pending.push((path.into(), Pending::Directory));
        self
    }

    /// Declares one regular file in the staging vocabulary: a planner and an
    /// ordered record run. The logical size is derived from the records; it
    /// cannot disagree. [`finish`] canonicalizes the body per planner — a
    /// `blob-v1` file must be empty or exactly one whole-file data record.
    ///
    /// [`finish`]: Self::finish
    pub fn file(
        &mut self,
        path: impl Into<String>,
        executable: bool,
        planner: PlannerId,
        records: Vec<FileRecord>,
    ) -> &mut Self {
        self.file_under(path, executable, planner, Stamp::None, records)
    }

    /// The same declaration, stamped with the layout contract that directed
    /// this file's chunking. A tensor file planned under a contract MUST be
    /// declared this way: the stamp is part of the snapshot's identity, and a
    /// missing one would claim the plain grid produced these boundaries.
    pub fn file_under(
        &mut self,
        path: impl Into<String>,
        executable: bool,
        planner: PlannerId,
        contract: Stamp,
        records: Vec<FileRecord>,
    ) -> &mut Self {
        self.pending.push((
            path.into(),
            Pending::File {
                executable,
                planner,
                contract,
                records,
            },
        ));
        self
    }

    pub fn symlink(&mut self, path: impl Into<String>, target: impl Into<String>) -> &mut Self {
        self.pending.push((
            path.into(),
            Pending::Symlink {
                target: target.into(),
            },
        ));
        self
    }

    /// Declares `path` as a hardlink of the regular file at `target_path`.
    /// The canonical carrier is the group's first sorted path, so the builder
    /// may re-root the file body onto `path`.
    pub fn hardlink(
        &mut self,
        path: impl Into<String>,
        target_path: impl Into<String>,
    ) -> &mut Self {
        self.pending.push((
            path.into(),
            Pending::HardlinkTo {
                target_path: target_path.into(),
            },
        ));
        self
    }

    /// Resolves the declared entries into the validated tree: path and
    /// symlink rules, duplicate refusal, hardlink re-rooting onto the first
    /// sorted path, ordinal assignment, and whole-tree topology. File bodies
    /// stay in staging record form.
    fn resolve(self) -> Result<Vec<(String, TreeEntry)>, Tfm1Error> {
        let mut entries: BTreeMap<String, Pending> = BTreeMap::new();
        for (path, pending) in self.pending {
            validate_path(&path)?;
            if let Pending::Symlink { target } = &pending {
                validate_symlink_target(target)?;
            }
            if entries.insert(path, pending).is_some() {
                return Err(Tfm1Error::DuplicatePath);
            }
        }

        // Re-root every hardlink group onto its first sorted path so object
        // ordinals depend only on the tree, never on declaration order.
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (path, pending) in &entries {
            if let Pending::HardlinkTo { target_path } = pending {
                match entries.get(target_path) {
                    Some(Pending::File { .. }) => groups
                        .entry(target_path.clone())
                        .or_default()
                        .push(path.clone()),
                    _ => return Err(Tfm1Error::HardlinkTarget),
                }
            }
        }
        for (file_path, link_paths) in groups {
            let carrier = link_paths
                .iter()
                .chain(std::iter::once(&file_path))
                .min()
                .expect("a link group is never empty")
                .clone();
            if carrier != file_path {
                let body = entries
                    .remove(&file_path)
                    .expect("the group target is present");
                entries.insert(
                    file_path,
                    Pending::HardlinkTo {
                        target_path: carrier.clone(),
                    },
                );
                for link in link_paths {
                    entries.insert(
                        link.clone(),
                        Pending::HardlinkTo {
                            target_path: carrier.clone(),
                        },
                    );
                }
                entries.insert(carrier, body);
            }
        }

        let mut resolved = Vec::with_capacity(entries.len());
        let mut ordinals: BTreeMap<String, u64> = BTreeMap::new();
        for (path, pending) in entries {
            let entry = match pending {
                Pending::Directory => TreeEntry::Directory,
                Pending::File {
                    executable,
                    planner,
                    contract,
                    records,
                } => {
                    let mut logical_size = 0_u64;
                    for record in &records {
                        if record.length() == 0 {
                            return Err(Tfm1Error::ZeroLengthRecord);
                        }
                        logical_size = logical_size
                            .checked_add(record.length())
                            .ok_or(Tfm1Error::LengthSumMismatch)?;
                    }
                    validate_record_run(&records, logical_size)?;
                    ordinals.insert(path.clone(), ordinals.len() as u64);
                    TreeEntry::File {
                        executable,
                        planner,
                        contract,
                        records,
                    }
                }
                Pending::Symlink { target } => TreeEntry::Symlink { target },
                Pending::HardlinkTo { target_path } => {
                    let ordinal = *ordinals
                        .get(&target_path)
                        .ok_or(Tfm1Error::HardlinkOrdinal)?;
                    TreeEntry::Hardlink { ordinal }
                }
            };
            resolved.push((path, entry));
        }
        validate_tree(&resolved, |entry| matches!(entry, TreeEntry::Directory))?;
        Ok(resolved)
    }

    /// Finishes as the committed-head staging tree, without canonicalizing
    /// file bodies. This is the mount's tree-read seam: a mid-write file may
    /// hold a record shape the canonical manifest cannot express.
    pub fn finish_tree(self) -> Result<HeadTree, Tfm1Error> {
        Ok(HeadTree {
            entries: self.resolve()?,
        })
    }

    /// Finishes as one canonical snapshot value, canonicalizing every file
    /// body: tensor planners keep their validated record list; a `blob-v1`
    /// file must be empty or exactly one whole-file data record, which
    /// becomes the recordless blob body.
    pub fn finish(self) -> Result<Snapshot, Tfm1Error> {
        let parent = self.parent;
        let resolved = self.resolve()?;
        let mut entries = Vec::with_capacity(resolved.len());
        for (path, entry) in resolved {
            let entry = match entry {
                TreeEntry::Directory => Entry::Directory,
                TreeEntry::File {
                    executable,
                    planner,
                    contract,
                    records,
                } => Entry::File {
                    executable,
                    body: canonical_body(planner, contract, records)?,
                },
                TreeEntry::Symlink { target } => Entry::Symlink { target },
                TreeEntry::Hardlink { ordinal } => Entry::Hardlink { ordinal },
            };
            entries.push((path, entry));
        }
        Ok(Snapshot { parent, entries })
    }
}

fn canonical_body(
    planner: PlannerId,
    contract: Stamp,
    records: Vec<FileRecord>,
) -> Result<FileBody, Tfm1Error> {
    match planner.tensor_format() {
        Some(format) => {
            let logical_size = records.iter().map(FileRecord::length).sum();
            validate_records(&records, logical_size)?;
            Ok(FileBody::Tensor {
                format,
                contract,
                logical_size,
                records,
            })
        }
        None => match records.as_slice() {
            [] => Ok(FileBody::Blob {
                logical_size: 0,
                digest: EMPTY_BLOB_DIGEST,
            }),
            [FileRecord::Data { digest, length }] => Ok(FileBody::Blob {
                logical_size: *length,
                digest: *digest,
            }),
            _ => Err(Tfm1Error::BlobRecords),
        },
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], Tfm1Error> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(Tfm1Error::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(Tfm1Error::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn take_u8(&mut self) -> Result<u8, Tfm1Error> {
        Ok(self.take(1)?[0])
    }

    fn take_u32(&mut self) -> Result<u32, Tfm1Error> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes were taken"),
        ))
    }

    fn take_u64(&mut self) -> Result<u64, Tfm1Error> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes were taken"),
        ))
    }

    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.position) as u64
    }
}

/// Strictly decodes one canonical TFM1 manifest. Every structural, path,
/// record, ordering, and topology rule refuses; the decoder never consults a
/// planner and reconstructs solely from the body facts. Planner byte 3 — the
/// retired `raw-fixed-64m-v1` grid — refuses as an unknown planner.
pub fn decode(bytes: &[u8]) -> Result<Snapshot, Tfm1Error> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(4)? != MAGIC {
        return Err(Tfm1Error::BadMagic);
    }
    let parent = match reader.take_u8()? {
        0 => None,
        1 => Some(SnapshotId::from_bytes(
            reader.take(32)?.try_into().expect("32 bytes were taken"),
        )),
        _ => return Err(Tfm1Error::ParentFlag),
    };

    let entry_count = reader.take_u64()?;
    if entry_count > reader.remaining() / MIN_ENTRY_BYTES {
        return Err(Tfm1Error::CountExceedsInput);
    }
    let mut entries: Vec<(String, Entry)> = Vec::new();
    entries
        .try_reserve_exact(usize::try_from(entry_count).map_err(|_| Tfm1Error::CountExceedsInput)?)
        .map_err(|_| Tfm1Error::ResourceExhausted)?;

    let mut assigned_ordinals = 0_u64;
    for _ in 0..entry_count {
        let path_length = usize::try_from(reader.take_u32()?).expect("u32 fits in usize");
        if path_length == 0 || path_length > MAX_PATH_BYTES {
            return Err(Tfm1Error::PathLength);
        }
        let path =
            std::str::from_utf8(reader.take(path_length)?).map_err(|_| Tfm1Error::PathUtf8)?;
        validate_path(path)?;
        if let Some((previous, _)) = entries.last() {
            match previous.as_str().cmp(path) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => return Err(Tfm1Error::DuplicatePath),
                std::cmp::Ordering::Greater => return Err(Tfm1Error::EntryOrder),
            }
        }

        let entry = match reader.take_u8()? {
            KIND_DIRECTORY => Entry::Directory,
            KIND_FILE => {
                let executable = match reader.take_u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err(Tfm1Error::ExecutableFlag),
                };
                let body = match reader.take_u8()? {
                    PLANNER_SAFETENSORS => {
                        decode_tensor_body(&mut reader, TensorFormat::SafetensorsV1)?
                    }
                    PLANNER_GGUF => decode_tensor_body(&mut reader, TensorFormat::GgufV1)?,
                    PLANNER_BLOB => {
                        let logical_size = reader.take_u64()?;
                        let digest = ObjectDigest::from_bytes(
                            reader.take(32)?.try_into().expect("32 bytes were taken"),
                        );
                        if logical_size == 0 && digest != EMPTY_BLOB_DIGEST {
                            return Err(Tfm1Error::EmptyBlobDigest);
                        }
                        FileBody::Blob {
                            logical_size,
                            digest,
                        }
                    }
                    _ => return Err(Tfm1Error::UnknownPlanner),
                };
                assigned_ordinals += 1;
                Entry::File { executable, body }
            }
            KIND_SYMLINK => {
                let target_length = usize::try_from(reader.take_u32()?).expect("u32 fits in usize");
                if target_length > MAX_SYMLINK_TARGET_BYTES {
                    return Err(Tfm1Error::SymlinkTarget);
                }
                let target = std::str::from_utf8(reader.take(target_length)?)
                    .map_err(|_| Tfm1Error::SymlinkTarget)?;
                validate_symlink_target(target)?;
                Entry::Symlink {
                    target: target.to_owned(),
                }
            }
            KIND_HARDLINK => {
                let ordinal = reader.take_u64()?;
                if ordinal >= assigned_ordinals {
                    return Err(Tfm1Error::HardlinkOrdinal);
                }
                Entry::Hardlink { ordinal }
            }
            _ => return Err(Tfm1Error::UnknownEntryKind),
        };
        entries.push((path.to_owned(), entry));
    }
    if reader.remaining() != 0 {
        return Err(Tfm1Error::TrailingBytes);
    }
    validate_tree(&entries, |entry| matches!(entry, Entry::Directory))?;
    Ok(Snapshot { parent, entries })
}

fn decode_tensor_body(
    reader: &mut Reader<'_>,
    format: TensorFormat,
) -> Result<FileBody, Tfm1Error> {
    let logical_size = reader.take_u64()?;
    let contract = decode_stamp(reader)?;
    let record_count = reader.take_u64()?;
    if record_count > MAX_FILE_RECORDS as u64 {
        return Err(Tfm1Error::RecordLimit);
    }
    if record_count > reader.remaining() / MIN_RECORD_BYTES {
        return Err(Tfm1Error::CountExceedsInput);
    }
    let mut records = Vec::new();
    records
        .try_reserve_exact(usize::try_from(record_count).expect("bounded record count"))
        .map_err(|_| Tfm1Error::ResourceExhausted)?;
    for _ in 0..record_count {
        let record = match reader.take_u8()? {
            RECORD_DATA => FileRecord::Data {
                digest: ObjectDigest::from_bytes(
                    reader.take(32)?.try_into().expect("32 bytes were taken"),
                ),
                length: reader.take_u64()?,
            },
            RECORD_HOLE => FileRecord::Hole {
                length: reader.take_u64()?,
            },
            _ => return Err(Tfm1Error::UnknownRecordTag),
        };
        records.push(record);
    }
    validate_records(&records, logical_size)?;
    Ok(FileBody::Tensor {
        format,
        contract,
        logical_size,
        records,
    })
}

/// The contract stamp: a bounded name and its version, or the canonical
/// absent stamp (empty name, version zero). Every other combination refuses —
/// a nameless version and a version-less name are both unreadable claims.
fn decode_stamp(reader: &mut Reader<'_>) -> Result<Stamp, Tfm1Error> {
    let length = usize::from(reader.take_u8()?);
    if length > MAX_CONTRACT_NAME_BYTES {
        return Err(Tfm1Error::ContractName);
    }
    let name = std::str::from_utf8(reader.take(length)?).map_err(|_| Tfm1Error::ContractName)?;
    let version = reader.take_u32()?;
    if name.is_empty() {
        if version != 0 {
            return Err(Tfm1Error::ContractVersion);
        }
        return Ok(Stamp::None);
    }
    if version == 0 {
        return Err(Tfm1Error::ContractVersion);
    }
    Stamp::named(name, version).map_err(|_| Tfm1Error::ContractName)
}
