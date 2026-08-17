//! Compose a new tensor container out of a committed one's own objects.
//!
//! A re-key is a header rewrite and nothing else. The container's header is
//! its own region; every tensor's chunk grid is cut relative to that tensor's
//! own start, so a header of a different length cannot move a tensor against
//! its grid; and TFM1 identity is the record list, so a file whose records
//! already exist costs no hashing. Renaming every key therefore costs one new
//! header object and inherits every tensor record verbatim — nothing is
//! re-chunked, re-hashed, re-written or re-uploaded.
//!
//! The result is an ordinary TFM1 file body. It carries no program, no
//! provenance edge and no new grammar: a composed snapshot is indistinguishable
//! from an ingested one, which is what makes it readable by foreign tools,
//! collectable by the ordinary mark walk and pullable by ordinary missing-object
//! computation.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use thiserror::Error;

use crate::planner::{
    ByteSource as _, MAX_OBJECT_SIZE, PlanError, TensorFormat, gguf, safetensors,
};
use crate::store::{ObjectStore, StoreError};
use crate::tfm1::{FileBody, FileRecord};
use crate::workspace_source::RecordsSource;

#[derive(Debug, Error)]
pub enum ComposeError {
    #[error("only a tensor container can be recomposed")]
    NotATensorContainer,
    #[error("the committed bytes are not a readable {0} container")]
    Unreadable(&'static str),
    #[error("the source container holds no tensor named {0:?}")]
    UnknownTensor(String),
    #[error("the composition does not name source tensor {0:?}")]
    UnnamedTensor(String),
    #[error("two source tensors would both be named {0:?}")]
    DuplicateName(String),
    #[error("{0:?} is not a usable tensor name in this container")]
    UnusableName(String),
    #[error("tensor {0:?} does not occupy whole objects, so it cannot be inherited")]
    NotObjectAligned(String),
    #[error("the composed header does not fit the format's bounds")]
    HeaderTooLarge,
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// What a composition does with a source tensor the caller did not name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Unnamed {
    Refuse,
    Drop,
}

/// Re-keys one committed tensor container: every tensor survives under the
/// name `names` maps it to, and every tensor's objects are inherited verbatim.
///
/// `names` must name every tensor in the source exactly once — a re-key is a
/// renaming, and a silent drop would be a different operation with a different
/// meaning for the bytes a sync must move.
pub fn rekey(
    store: &ObjectStore,
    body: &FileBody,
    names: &BTreeMap<String, String>,
) -> Result<FileBody, ComposeError> {
    compose(store, body, names, Unnamed::Refuse)
}

/// Composes a trimmed container: exactly the tensors `names` maps survive, in
/// the source's own data order, and the rest are omitted.
///
/// Omitting a tensor omits its records, which is the whole mechanism: a sync
/// of the subset computes its missing objects from the record list it has, so
/// the trimmed bytes are never requested and never downloaded. Trimming and
/// renaming are one operation because they are one header rewrite; pass
/// identity pairs to trim without renaming.
pub fn subset(
    store: &ObjectStore,
    body: &FileBody,
    names: &BTreeMap<String, String>,
) -> Result<FileBody, ComposeError> {
    compose(store, body, names, Unnamed::Drop)
}

fn compose(
    store: &ObjectStore,
    body: &FileBody,
    names: &BTreeMap<String, String>,
    unnamed: Unnamed,
) -> Result<FileBody, ComposeError> {
    let FileBody::Tensor {
        format, records, ..
    } = body
    else {
        return Err(ComposeError::NotATensorContainer);
    };
    let source = RecordsSource::new(store, records);
    let composed = match format {
        TensorFormat::SafetensorsV1 => {
            compose_safetensors(store, records, &source, names, unnamed)?
        }
        TensorFormat::GgufV1 => compose_gguf(store, records, &source, names, unnamed)?,
    };
    Ok(FileBody::Tensor {
        format: *format,
        logical_size: composed.iter().map(record_length).sum(),
        records: composed,
    })
}

const fn record_length(record: &FileRecord) -> u64 {
    match record {
        FileRecord::Data { length, .. } | FileRecord::Hole { length } => *length,
    }
}

/// Resolves the source tensors that survive, in the source's own data order,
/// to the names they keep. No two may collide, and a name the source does not
/// hold is always a refusal — a trim that silently keeps nothing would be a
/// typo that looks like a working composition.
fn select<'a, T>(
    tensors: &'a [T],
    name_of: impl Fn(&'a T) -> &'a str,
    names: &'a BTreeMap<String, String>,
    unnamed: Unnamed,
) -> Result<Vec<(&'a str, &'a T)>, ComposeError> {
    let mut kept = Vec::with_capacity(tensors.len());
    let mut taken = HashSet::new();
    let mut sources = HashSet::new();
    for tensor in tensors {
        let old = name_of(tensor);
        sources.insert(old);
        let new = match (names.get(old), unnamed) {
            (Some(new), _) => new,
            (None, Unnamed::Drop) => continue,
            (None, Unnamed::Refuse) => {
                return Err(ComposeError::UnnamedTensor(old.to_owned()));
            }
        };
        if !taken.insert(new.as_str()) {
            return Err(ComposeError::DuplicateName(new.clone()));
        }
        kept.push((new.as_str(), tensor));
    }
    for old in names.keys() {
        if !sources.contains(old.as_str()) {
            return Err(ComposeError::UnknownTensor(old.clone()));
        }
    }
    Ok(kept)
}

/// The record run covering exactly `[start, start + length)`, or a refusal.
///
/// A tensor that begins or ends inside a record has no digest of its own, so
/// it cannot be carried by reference — its bytes would have to be rewritten.
/// The seal planner never produces such a grid, but the manifest wire allows
/// one, so the refusal is real rather than an assertion.
fn inherit(records: &[FileRecord], start: u64, length: u64) -> Option<Vec<FileRecord>> {
    let mut taken = Vec::new();
    let mut position = 0_u64;
    let end = start.checked_add(length)?;
    for record in records {
        let record_end = position + record_length(record);
        if record_end <= start {
            position = record_end;
            continue;
        }
        if position >= end {
            break;
        }
        if position < start || record_end > end {
            return None;
        }
        taken.push(record.clone());
        position = record_end;
    }
    (taken.iter().map(record_length).sum::<u64>() == length).then_some(taken)
}

/// Admits one new header region and returns its records on the seal grid.
fn admit(store: &ObjectStore, bytes: &[u8]) -> Result<Vec<FileRecord>, ComposeError> {
    let grid = usize::try_from(MAX_OBJECT_SIZE).map_err(|_| ComposeError::HeaderTooLarge)?;
    let mut records = Vec::new();
    for piece in bytes.chunks(grid) {
        let admitted = store.put_bytes(piece)?;
        records.push(FileRecord::Data {
            digest: admitted.digest(),
            length: admitted.length(),
        });
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

const SAFETENSORS_METADATA_KEY: &str = "__metadata__";

fn compose_safetensors(
    store: &ObjectStore,
    records: &[FileRecord],
    source: &RecordsSource<&ObjectStore>,
    names: &BTreeMap<String, String>,
    unnamed: Unnamed,
) -> Result<Vec<FileRecord>, ComposeError> {
    let layout =
        safetensors::read_layout(source)?.ok_or(ComposeError::Unreadable("safetensors"))?;
    let kept = select(
        &layout.tensors,
        |tensor| tensor.name.as_str(),
        names,
        unnamed,
    )?;
    for (name, _) in &kept {
        if name.is_empty() || *name == SAFETENSORS_METADATA_KEY {
            return Err(ComposeError::UnusableName((*name).to_owned()));
        }
    }

    // `data_offsets` are relative to the data section, so recomputing them for
    // the new header length can never shift a tensor against its own grid.
    let mut header = String::from("{");
    let mut cursor = 0_u64;
    for (index, (name, tensor)) in kept.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let quoted = serde_json::to_string(name).expect("a string always serializes");
        let dtype = serde_json::to_string(&tensor.dtype).expect("a string always serializes");
        write!(header, "{quoted}:{{\"dtype\":{dtype},\"shape\":[").expect("writing to a String");
        for (axis, dimension) in tensor.shape.iter().enumerate() {
            if axis > 0 {
                header.push(',');
            }
            write!(header, "{dimension}").expect("writing to a String");
        }
        let length = tensor.end - tensor.start;
        write!(
            header,
            "],\"data_offsets\":[{cursor},{}]}}",
            cursor + length
        )
        .expect("writing to a String");
        cursor += length;
    }
    header.push('}');

    let mut bytes = header.into_bytes();
    // The reference writer pads the header to an 8-byte boundary so tensor
    // data lands aligned; trailing whitespace is header JSON either way.
    bytes.resize(bytes.len().next_multiple_of(8), b' ');
    let length = u64::try_from(bytes.len()).map_err(|_| ComposeError::HeaderTooLarge)?;
    let mut file = length.to_le_bytes().to_vec();
    file.append(&mut bytes);

    let mut composed = admit(store, &file)?;
    for (_, tensor) in kept {
        composed.extend(
            inherit(
                records,
                layout.header_end + tensor.start,
                tensor.end - tensor.start,
            )
            .ok_or_else(|| ComposeError::NotObjectAligned(tensor.name.clone()))?,
        );
    }
    Ok(composed)
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

fn compose_gguf(
    store: &ObjectStore,
    records: &[FileRecord],
    source: &RecordsSource<&ObjectStore>,
    names: &BTreeMap<String, String>,
    unnamed: Unnamed,
) -> Result<Vec<FileRecord>, ComposeError> {
    let layout = gguf::read_layout(source)?.ok_or(ComposeError::Unreadable("gguf"))?;
    let kept = select(
        &layout.tensors,
        |tensor| tensor.name.as_str(),
        names,
        unnamed,
    )?;
    for (name, _) in &kept {
        if name.len() as u64 > gguf::MAX_TENSOR_NAME_LEN || name.as_bytes().contains(&0) {
            return Err(ComposeError::UnusableName((*name).to_owned()));
        }
    }

    // Tensor names live in the directory, so a re-key leaves the metadata
    // block byte-identical and it is inherited whole. Dropping a tensor is the
    // one thing that reaches it: the tensor count sits in its fixed prefix, so
    // a trim rewrites that domain and nothing else in it.
    let mut composed = if kept.len() == layout.tensors.len() {
        inherit(records, 0, layout.directory_start)
            .ok_or_else(|| ComposeError::NotObjectAligned("the metadata block".to_owned()))?
    } else {
        let span =
            usize::try_from(layout.directory_start).map_err(|_| ComposeError::HeaderTooLarge)?;
        let mut metadata = vec![0_u8; span];
        source
            .read_exact_at(0, &mut metadata)
            .map_err(|error| ComposeError::Plan(PlanError::Read(error)))?;
        metadata[8..16].copy_from_slice(&(kept.len() as u64).to_le_bytes());
        admit(store, &metadata)?
    };

    let mut directory = Vec::new();
    let mut offset = 0_u64;
    for (name, tensor) in &kept {
        directory.extend_from_slice(&(name.len() as u64).to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
        directory.extend_from_slice(&tensor.dimension_count.to_le_bytes());
        for axis in 0..tensor.dimension_count as usize {
            directory.extend_from_slice(&tensor.dimensions[axis].to_le_bytes());
        }
        directory.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        directory.extend_from_slice(&offset.to_le_bytes());
        offset +=
            gguf::align_up(tensor.length, layout.alignment).ok_or(ComposeError::HeaderTooLarge)?;
    }
    let directory_end = layout
        .directory_start
        .checked_add(directory.len() as u64)
        .ok_or(ComposeError::HeaderTooLarge)?;
    let data_start = if kept.is_empty() {
        directory_end
    } else {
        gguf::align_up(directory_end, layout.alignment).ok_or(ComposeError::HeaderTooLarge)?
    };
    composed.extend(admit(store, &directory)?);
    if data_start > directory_end {
        // Alignment padding is its own object, never glued to tensor data.
        // That isolation is what lets a GGUF share tensor objects with its
        // safetensors twin, and it makes this padding shareable in turn.
        let padding = vec![0_u8; (data_start - directory_end) as usize];
        composed.extend(admit(store, &padding)?);
    }

    for (_, tensor) in kept {
        let start = layout.data_start + tensor.offset;
        let padded =
            gguf::align_up(tensor.length, layout.alignment).ok_or(ComposeError::HeaderTooLarge)?;
        composed.extend(
            inherit(records, start, padded)
                .ok_or_else(|| ComposeError::NotObjectAligned(tensor.name.clone()))?,
        );
    }
    Ok(composed)
}
