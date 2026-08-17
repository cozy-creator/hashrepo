use std::collections::HashSet;
use std::io;

use super::{ByteSource, Plan, PlanError, PlannerId, Region, RegionKind, append_split_region};

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_METADATA_COUNT: u64 = 1_000_000;
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_SYMBOL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_VALUE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_INSPECTION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 16_000_000;
const MAX_ARRAY_DEPTH: u32 = 16;
const MAX_METADATA_KEY_LEN: u64 = u16::MAX as u64;
pub(crate) const MAX_TENSOR_NAME_LEN: u64 = 63;

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

const GENERAL_ALIGNMENT: &[u8] = b"general.alignment";

#[derive(Debug)]
enum ParseFailure {
    Invalid,
    Read(io::Error),
    ResourceExhausted,
}

impl From<io::Error> for ParseFailure {
    fn from(error: io::Error) -> Self {
        Self::Read(error)
    }
}

type ParseResult<T> = Result<T, ParseFailure>;

/// One tensor as the directory declares it. `offset` is relative to the
/// aligned start of the data section and `length` is the unpadded extent;
/// composition needs the name and geometry the plan discards.
#[derive(Clone, Debug)]
pub(crate) struct TensorInfo {
    pub(crate) name: String,
    pub(crate) dimensions: [u64; 4],
    pub(crate) dimension_count: u32,
    pub(crate) ggml_type: u32,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

/// A fully validated GGUF file: the three header domains the planner keeps
/// apart — metadata, directory, pre-data padding — and every tensor.
pub(crate) struct Layout {
    pub(crate) alignment: u64,
    pub(crate) directory_start: u64,
    pub(crate) directory_end: u64,
    pub(crate) data_start: u64,
    pub(crate) tensors: Vec<TensorInfo>,
}

struct Reader<'a, S: ?Sized> {
    source: &'a S,
    position: u64,
    length: u64,
    limit: Option<u64>,
}

impl<'a, S: ByteSource + ?Sized> Reader<'a, S> {
    fn new(source: &'a S) -> Self {
        Self {
            source,
            position: 0,
            length: source.len(),
            limit: None,
        }
    }

    fn remaining(&self) -> u64 {
        self.length - self.position
    }

    fn ensure(&self, length: u64) -> ParseResult<()> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(ParseFailure::Invalid)?;
        if end > self.length || self.limit.is_some_and(|limit| end > limit) {
            return Err(ParseFailure::Invalid);
        }
        Ok(())
    }

    fn read<const N: usize>(&mut self) -> ParseResult<[u8; N]> {
        self.ensure(N as u64)?;
        let mut bytes = [0_u8; N];
        self.source.read_exact_at(self.position, &mut bytes)?;
        self.position += N as u64;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> ParseResult<u8> {
        Ok(self.read::<1>()?[0])
    }

    fn read_u32(&mut self) -> ParseResult<u32> {
        Ok(u32::from_le_bytes(self.read()?))
    }

    fn read_u64(&mut self) -> ParseResult<u64> {
        Ok(u64::from_le_bytes(self.read()?))
    }

    fn skip(&mut self, length: u64) -> ParseResult<()> {
        self.ensure(length)?;
        self.position += length;
        Ok(())
    }

    fn limit_next(&mut self, length: u64) -> ParseResult<()> {
        self.limit = Some(
            self.position
                .checked_add(length)
                .ok_or(ParseFailure::Invalid)?,
        );
        Ok(())
    }

    fn clear_limit(&mut self) {
        self.limit = None;
    }

    fn read_bounded_bytes(&mut self, max_length: u64) -> ParseResult<Vec<u8>> {
        let length = self.read_u64()?;
        if length > max_length {
            return Err(ParseFailure::Invalid);
        }
        self.ensure(length)?;
        let length = usize::try_from(length).map_err(|_| ParseFailure::Invalid)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| ParseFailure::ResourceExhausted)?;
        bytes.resize(length, 0);
        self.source.read_exact_at(self.position, &mut bytes)?;
        self.position += length as u64;
        Ok(bytes)
    }

    fn skip_utf8_string(&mut self, budget: &mut MetadataBudget) -> ParseResult<()> {
        let length = self.read_u64()?;
        budget.charge_bytes(length)?;
        self.ensure(length)?;

        // Metadata strings can be large, so validate them in bounded memory rather
        // than allocating from their untrusted length prefix.
        let mut remaining = length;
        let mut pending = 0_usize;
        let mut buffer = [0_u8; 8196];
        while remaining > 0 {
            let chunk = remaining.min(8192) as usize;
            self.source
                .read_exact_at(self.position, &mut buffer[pending..pending + chunk])?;
            self.position += chunk as u64;
            remaining -= chunk as u64;

            let used = pending + chunk;
            match std::str::from_utf8(&buffer[..used]) {
                Ok(_) => pending = 0,
                Err(error) if error.error_len().is_none() => {
                    let valid = error.valid_up_to();
                    let incomplete = used - valid;
                    if incomplete > 3 {
                        return Err(ParseFailure::Invalid);
                    }
                    buffer.copy_within(valid..used, 0);
                    pending = incomplete;
                }
                Err(_) => return Err(ParseFailure::Invalid),
            }
        }
        if pending != 0 {
            return Err(ParseFailure::Invalid);
        }
        Ok(())
    }

    fn check_bool_bytes(&mut self, count: u64) -> ParseResult<()> {
        self.ensure(count)?;
        let mut remaining = count;
        let mut buffer = [0_u8; 8192];
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            self.source
                .read_exact_at(self.position, &mut buffer[..chunk])?;
            if buffer[..chunk].iter().any(|value| *value > 1) {
                return Err(ParseFailure::Invalid);
            }
            self.position += chunk as u64;
            remaining -= chunk as u64;
        }
        Ok(())
    }
}

struct MetadataBudget {
    array_elements: u64,
    value_bytes: u64,
}

impl MetadataBudget {
    fn charge(&mut self, count: u64) -> ParseResult<()> {
        self.array_elements = self
            .array_elements
            .checked_add(count)
            .ok_or(ParseFailure::Invalid)?;
        if self.array_elements > MAX_ARRAY_ELEMENTS {
            return Err(ParseFailure::Invalid);
        }
        Ok(())
    }

    fn charge_bytes(&mut self, count: u64) -> ParseResult<()> {
        self.value_bytes = self
            .value_bytes
            .checked_add(count)
            .ok_or(ParseFailure::Invalid)?;
        if self.value_bytes > MAX_METADATA_VALUE_BYTES {
            return Err(ParseFailure::Invalid);
        }
        Ok(())
    }
}

pub(crate) fn try_plan<S: ByteSource + ?Sized>(source: &S) -> Result<Option<Plan>, PlanError> {
    lift(parse(source).and_then(|layout| plan_layout(source.len(), &layout)))
}

/// The whole structural proof, shared by the planner and by composition.
pub(crate) fn read_layout<S: ByteSource + ?Sized>(source: &S) -> Result<Option<Layout>, PlanError> {
    lift(parse(source))
}

fn lift<T>(result: ParseResult<T>) -> Result<Option<T>, PlanError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(ParseFailure::Invalid) => Ok(None),
        Err(ParseFailure::Read(error)) => Err(PlanError::Read(error)),
        Err(ParseFailure::ResourceExhausted) => Err(PlanError::ResourceExhausted),
    }
}

fn parse<S: ByteSource + ?Sized>(source: &S) -> ParseResult<Layout> {
    let mut reader = Reader::new(source);
    if reader.read::<4>()? != MAGIC {
        return Err(ParseFailure::Invalid);
    }

    // GGUF v3 added big-endian support without an endian marker. Reading the
    // version as little-endian therefore rejects a big-endian v2/v3 header, as
    // well as older and future structural versions.
    if !matches!(reader.read_u32()?, 2 | 3) {
        return Err(ParseFailure::Invalid);
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    validate_counts(metadata_count, tensor_count, reader.remaining())?;

    let mut metadata_keys = HashSet::new();
    let mut tensor_names = HashSet::new();
    let mut symbol_bytes = 0_u64;
    let mut budget = MetadataBudget {
        array_elements: 0,
        value_bytes: 0,
    };
    let mut alignment = DEFAULT_ALIGNMENT;
    reader.limit_next(MAX_METADATA_INSPECTION_BYTES)?;

    for _ in 0..metadata_count {
        let key = reader.read_bounded_bytes(MAX_METADATA_KEY_LEN)?;
        validate_key(&key)?;
        charge_symbol_bytes(&mut symbol_bytes, key.len())?;
        if !metadata_keys.insert(key.clone()) {
            return Err(ParseFailure::Invalid);
        }

        let value_type = reader.read_u32()?;
        if key == GENERAL_ALIGNMENT {
            if value_type != GGUF_TYPE_UINT32 {
                return Err(ParseFailure::Invalid);
            }
            budget.charge_bytes(4)?;
            alignment = reader.read_u32()? as u64;
        } else {
            parse_metadata_value(&mut reader, value_type, 0, &mut budget)?;
        }
    }
    reader.clear_limit();

    if alignment < 8 || !alignment.is_power_of_two() {
        return Err(ParseFailure::Invalid);
    }

    let directory_start = reader.position;
    let mut tensors = Vec::new();
    let tensor_capacity = usize::try_from(tensor_count).map_err(|_| ParseFailure::Invalid)?;
    tensors
        .try_reserve_exact(tensor_capacity)
        .map_err(|_| ParseFailure::ResourceExhausted)?;
    let mut expected_offset = 0_u64;

    for _ in 0..tensor_count {
        let name = reader.read_bounded_bytes(MAX_TENSOR_NAME_LEN)?;
        validate_name(&name)?;
        charge_symbol_bytes(&mut symbol_bytes, name.len())?;
        if !tensor_names.insert(name.clone()) {
            return Err(ParseFailure::Invalid);
        }
        let name = String::from_utf8(name).map_err(|_| ParseFailure::Invalid)?;

        let dimension_count = reader.read_u32()?;
        if dimension_count > 4 {
            return Err(ParseFailure::Invalid);
        }
        let mut dimensions = [1_u64; 4];
        for dimension in dimensions.iter_mut().take(dimension_count as usize) {
            *dimension = reader.read_u64()?;
            if *dimension > i64::MAX as u64 {
                return Err(ParseFailure::Invalid);
            }
        }

        let ggml_type = reader.read_u32()?;
        let (block_elements, block_bytes) = ggml_layout(ggml_type)?;
        let length = tensor_length(dimensions, block_elements, block_bytes)?;
        let offset = reader.read_u64()?;
        if offset != expected_offset {
            return Err(ParseFailure::Invalid);
        }
        expected_offset = expected_offset
            .checked_add(align_up(length, alignment).ok_or(ParseFailure::Invalid)?)
            .ok_or(ParseFailure::Invalid)?;
        tensors.push(TensorInfo {
            name,
            dimensions,
            dimension_count,
            ggml_type,
            offset,
            length,
        });
    }

    let directory_end = reader.position;
    let data_start = if tensors.is_empty() {
        directory_end
    } else {
        align_up(directory_end, alignment).ok_or(ParseFailure::Invalid)?
    };
    let expected_file_size = data_start
        .checked_add(expected_offset)
        .ok_or(ParseFailure::Invalid)?;
    if expected_file_size != reader.length {
        return Err(ParseFailure::Invalid);
    }

    Ok(Layout {
        alignment,
        directory_start,
        directory_end,
        data_start,
        tensors,
    })
}

/// The region shape a validated layout produces: metadata, directory and
/// pre-data padding as separate header domains, then every tensor's unpadded
/// extent with its own trailing padding kept out of it. Isolating padding is
/// what lets a GGUF share tensor objects with a safetensors twin.
fn plan_layout(file_size: u64, layout: &Layout) -> ParseResult<Plan> {
    let mut regions = Vec::new();
    append_domain(&mut regions, 0, layout.directory_start, RegionKind::Header)?;
    append_domain(
        &mut regions,
        layout.directory_start,
        layout.directory_end - layout.directory_start,
        RegionKind::Header,
    )?;
    append_domain(
        &mut regions,
        layout.directory_end,
        layout.data_start - layout.directory_end,
        RegionKind::Header,
    )?;

    for tensor in &layout.tensors {
        let tensor_start = layout
            .data_start
            .checked_add(tensor.offset)
            .ok_or(ParseFailure::Invalid)?;
        append_domain(
            &mut regions,
            tensor_start,
            tensor.length,
            RegionKind::Tensor,
        )?;
        let padded_length =
            align_up(tensor.length, layout.alignment).ok_or(ParseFailure::Invalid)?;
        append_domain(
            &mut regions,
            tensor_start + tensor.length,
            padded_length - tensor.length,
            RegionKind::Header,
        )?;
    }

    Ok(Plan {
        planner: PlannerId::GgufV1,
        file_size,
        regions,
    })
}

fn validate_counts(metadata_count: u64, tensor_count: u64, remaining: u64) -> ParseResult<()> {
    if metadata_count > MAX_METADATA_COUNT || tensor_count > MAX_TENSOR_COUNT {
        return Err(ParseFailure::Invalid);
    }
    // Minimum encodings: empty-key u8 metadata is 13 bytes; a zero-dimension,
    // empty-name tensor directory entry is 24 bytes.
    let minimum = metadata_count
        .checked_mul(13)
        .and_then(|bytes| {
            tensor_count
                .checked_mul(24)
                .and_then(|more| bytes.checked_add(more))
        })
        .ok_or(ParseFailure::Invalid)?;
    if minimum > remaining {
        return Err(ParseFailure::Invalid);
    }
    Ok(())
}

fn validate_key(key: &[u8]) -> ParseResult<()> {
    if key.is_empty()
        || key
            .split(|byte| *byte == b'.')
            .any(|segment| !is_lower_snake_case(segment))
    {
        return Err(ParseFailure::Invalid);
    }
    Ok(())
}

fn is_lower_snake_case(segment: &[u8]) -> bool {
    segment.first().is_some_and(u8::is_ascii_lowercase)
        && segment
            .last()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && segment.windows(2).all(|pair| pair != b"__")
        && segment
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
}

fn validate_name(name: &[u8]) -> ParseResult<()> {
    if name.contains(&0) || std::str::from_utf8(name).is_err() {
        return Err(ParseFailure::Invalid);
    }
    Ok(())
}

fn charge_symbol_bytes(total: &mut u64, length: usize) -> ParseResult<()> {
    *total = total
        .checked_add(length as u64)
        .ok_or(ParseFailure::Invalid)?;
    if *total > MAX_SYMBOL_BYTES {
        return Err(ParseFailure::Invalid);
    }
    Ok(())
}

fn parse_metadata_value<S: ByteSource + ?Sized>(
    reader: &mut Reader<'_, S>,
    value_type: u32,
    depth: u32,
    budget: &mut MetadataBudget,
) -> ParseResult<()> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 => {
            budget.charge_bytes(1)?;
            reader.skip(1)
        }
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
            budget.charge_bytes(2)?;
            reader.skip(2)
        }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
            budget.charge_bytes(4)?;
            reader.skip(4)
        }
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
            budget.charge_bytes(8)?;
            reader.skip(8)
        }
        GGUF_TYPE_BOOL => {
            budget.charge_bytes(1)?;
            if reader.read_u8()? > 1 {
                return Err(ParseFailure::Invalid);
            }
            Ok(())
        }
        GGUF_TYPE_STRING => reader.skip_utf8_string(budget),
        GGUF_TYPE_ARRAY => parse_metadata_array(reader, depth, budget),
        _ => Err(ParseFailure::Invalid),
    }
}

fn parse_metadata_array<S: ByteSource + ?Sized>(
    reader: &mut Reader<'_, S>,
    depth: u32,
    budget: &mut MetadataBudget,
) -> ParseResult<()> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(ParseFailure::Invalid);
    }
    let element_type = reader.read_u32()?;
    if element_type > GGUF_TYPE_FLOAT64 {
        return Err(ParseFailure::Invalid);
    }
    let count = reader.read_u64()?;
    budget.charge(count)?;

    let fixed_size = match element_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 => Some(1_u64),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => Some(2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => Some(4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => Some(8),
        _ => None,
    };
    if let Some(size) = fixed_size {
        let bytes = count.checked_mul(size).ok_or(ParseFailure::Invalid)?;
        budget.charge_bytes(bytes)?;
        return reader.skip(bytes);
    }
    if element_type == GGUF_TYPE_BOOL {
        budget.charge_bytes(count)?;
        return reader.check_bool_bytes(count);
    }

    for _ in 0..count {
        parse_metadata_value(reader, element_type, depth + 1, budget)?;
    }
    Ok(())
}

fn tensor_length(dimensions: [u64; 4], block_elements: u64, block_bytes: u64) -> ParseResult<u64> {
    if !dimensions[0].is_multiple_of(block_elements) {
        return Err(ParseFailure::Invalid);
    }
    let element_count = dimensions
        .into_iter()
        .try_fold(1_u64, |product, dimension| {
            product.checked_mul(dimension).ok_or(ParseFailure::Invalid)
        })?;
    if element_count >= i64::MAX as u64 {
        return Err(ParseFailure::Invalid);
    }
    (dimensions[0] / block_elements)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(dimensions[1]))
        .and_then(|plane| plane.checked_mul(dimensions[2]))
        .and_then(|cube| cube.checked_mul(dimensions[3]))
        .ok_or(ParseFailure::Invalid)
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
}

fn append_domain(
    regions: &mut Vec<Region>,
    offset: u64,
    length: u64,
    kind: RegionKind,
) -> ParseResult<()> {
    match append_split_region(regions, offset, length, kind) {
        Ok(()) => Ok(()),
        Err(PlanError::ObjectLimit) => Err(ParseFailure::Invalid),
        Err(PlanError::ResourceExhausted) => Err(ParseFailure::ResourceExhausted),
        Err(_) => Err(ParseFailure::Invalid),
    }
}

// Reviewed and pinned against ggml-org/llama.cpp at
// 7e4c0a96880dae4fc4268ad441f8a6446bd5460a:
//   ggml/include/ggml.h (enum ggml_type)
//   ggml/src/ggml.c (type_traits)
//   gguf-py/gguf/constants.py (GGML_QUANT_SIZES)
// The tuple is (elements per encoded block, bytes per encoded block). Removed
// IDs 4, 5, 31..=33, and 36..=38 deliberately have no entry.
fn ggml_layout(ggml_type: u32) -> ParseResult<(u64, u64)> {
    let layout = match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 40),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        42 => (64, 18),   // Q2_0
        _ => return Err(ParseFailure::Invalid),
    };
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::MAX_OBJECT_SIZE;

    #[derive(Clone)]
    struct TensorSpec<'a> {
        name: &'a str,
        dimensions: &'a [u64],
        ggml_type: u32,
        offset: Option<u64>,
    }

    struct SparseFixture {
        prefix: Vec<u8>,
        length: u64,
    }

    struct SliceFixture<'a>(&'a [u8]);

    impl ByteSource for SliceFixture<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
            self.0.read_exact_at(offset, destination)
        }

        fn check_unchanged(&self) -> io::Result<()> {
            Ok(())
        }
    }

    impl ByteSource for SparseFixture {
        fn len(&self) -> u64 {
            self.length
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
            let end = offset
                .checked_add(destination.len() as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "overflow"))?;
            if end > self.length {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
            }
            destination.fill(0);
            if offset < self.prefix.len() as u64 {
                let available = (self.prefix.len() as u64 - offset).min(destination.len() as u64);
                destination[..available as usize]
                    .copy_from_slice(&self.prefix[offset as usize..(offset + available) as usize]);
            }
            Ok(())
        }

        fn check_unchanged(&self) -> io::Result<()> {
            Ok(())
        }
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value);
    }

    fn metadata(key: &str, value_type: u32, value: Vec<u8>) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_string(&mut bytes, key.as_bytes());
        push_u32(&mut bytes, value_type);
        bytes.extend(value);
        bytes
    }

    fn u32_value(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn string_value(value: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_string(&mut bytes, value);
        bytes
    }

    fn array_value(element_type: u32, values: Vec<u8>, count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, element_type);
        push_u64(&mut bytes, count);
        bytes.extend(values);
        bytes
    }

    fn build_prefix(
        version: u32,
        metadata_entries: &[Vec<u8>],
        tensors: &[TensorSpec<'_>],
        alignment: u64,
    ) -> (Vec<u8>, u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        push_u32(&mut bytes, version);
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, metadata_entries.len() as u64);
        for entry in metadata_entries {
            bytes.extend_from_slice(entry);
        }

        let mut expected_offset = 0_u64;
        for tensor in tensors {
            push_string(&mut bytes, tensor.name.as_bytes());
            push_u32(&mut bytes, tensor.dimensions.len() as u32);
            for dimension in tensor.dimensions {
                push_u64(&mut bytes, *dimension);
            }
            push_u32(&mut bytes, tensor.ggml_type);
            push_u64(&mut bytes, tensor.offset.unwrap_or(expected_offset));

            let mut dimensions = [1_u64; 4];
            for (destination, source) in dimensions.iter_mut().zip(tensor.dimensions) {
                *destination = *source;
            }
            let (block_elements, block_bytes) = ggml_layout(tensor.ggml_type).unwrap_or((1, 1));
            let length = tensor_length(dimensions, block_elements, block_bytes).unwrap_or(0);
            expected_offset += align_up(length, alignment).unwrap();
        }

        let data_start = if tensors.is_empty() {
            bytes.len() as u64
        } else {
            align_up(bytes.len() as u64, alignment).unwrap()
        };
        bytes.resize(data_start as usize, 0);
        (bytes, data_start + expected_offset)
    }

    fn build(
        version: u32,
        metadata_entries: &[Vec<u8>],
        tensors: &[TensorSpec<'_>],
        alignment: u64,
    ) -> Vec<u8> {
        let (mut bytes, length) = build_prefix(version, metadata_entries, tensors, alignment);
        bytes.resize(length as usize, 0);
        bytes
    }

    fn tensor<'a>(name: &'a str, dimensions: &'a [u64], ggml_type: u32) -> TensorSpec<'a> {
        TensorSpec {
            name,
            dimensions,
            ggml_type,
            offset: None,
        }
    }

    fn assert_raw_fallback(bytes: &[u8]) {
        assert!(try_plan(&SliceFixture(bytes)).unwrap().is_none());
    }

    #[test]
    fn plans_v2_and_v3_with_separate_semantic_domains() {
        let entries = vec![
            metadata("general.alignment", GGUF_TYPE_UINT32, u32_value(64)),
            metadata("general.name", GGUF_TYPE_STRING, string_value(b"fixture")),
        ];
        let tensors = vec![tensor("dense", &[3, 2], 0), tensor("quant", &[256], 10)];

        for version in [2, 3] {
            let bytes = build(version, &entries, &tensors, 64);
            let plan = try_plan(&SliceFixture(&bytes)).unwrap().unwrap();
            assert_eq!(plan.planner, PlannerId::GgufV1);
            assert_eq!(plan.file_size, bytes.len() as u64);
            plan.validate().unwrap();

            let tensor_regions: Vec<_> = plan
                .regions
                .iter()
                .filter(|region| region.kind == RegionKind::Tensor)
                .map(|region| region.length)
                .collect();
            assert_eq!(tensor_regions, [24, 84]);
            assert!(plan.regions.windows(2).any(
                |pair| pair[0].kind == RegionKind::Header && pair[1].kind == RegionKind::Header
            ));
        }
    }

    #[test]
    fn parses_all_metadata_scalar_and_array_encodings() {
        let mut entries = Vec::new();
        for (index, (value_type, width)) in [
            (GGUF_TYPE_UINT8, 1),
            (GGUF_TYPE_INT8, 1),
            (GGUF_TYPE_UINT16, 2),
            (GGUF_TYPE_INT16, 2),
            (GGUF_TYPE_UINT32, 4),
            (GGUF_TYPE_INT32, 4),
            (GGUF_TYPE_FLOAT32, 4),
            (GGUF_TYPE_UINT64, 8),
            (GGUF_TYPE_INT64, 8),
            (GGUF_TYPE_FLOAT64, 8),
        ]
        .into_iter()
        .enumerate()
        {
            entries.push(metadata(
                &format!("scalar.value_{index}"),
                value_type,
                vec![0; width],
            ));
            entries.push(metadata(
                &format!("array.value_{index}"),
                GGUF_TYPE_ARRAY,
                array_value(value_type, vec![0; width * 2], 2),
            ));
        }
        entries.push(metadata("bool", GGUF_TYPE_BOOL, vec![1]));
        entries.push(metadata(
            "bool_array",
            GGUF_TYPE_ARRAY,
            array_value(GGUF_TYPE_BOOL, vec![0, 1], 2),
        ));
        entries.push(metadata(
            "strings",
            GGUF_TYPE_ARRAY,
            array_value(
                GGUF_TYPE_STRING,
                [string_value(b"alpha"), string_value("beta".as_bytes())].concat(),
                2,
            ),
        ));
        entries.push(metadata(
            "nested",
            GGUF_TYPE_ARRAY,
            array_value(
                GGUF_TYPE_ARRAY,
                array_value(GGUF_TYPE_UINT8, vec![1, 2], 2),
                1,
            ),
        ));

        let bytes = build(3, &entries, &[], DEFAULT_ALIGNMENT);
        let plan = try_plan(&SliceFixture(&bytes)).unwrap().unwrap();
        assert_eq!(plan.planner, PlannerId::GgufV1);
        plan.validate().unwrap();
    }

    #[test]
    fn metadata_inspection_has_an_exact_total_byte_budget() {
        const KEY: &[u8] = b"general.description";

        fn sparse_string(length: u64) -> SparseFixture {
            let mut prefix = Vec::new();
            prefix.extend_from_slice(&MAGIC);
            push_u32(&mut prefix, 3);
            push_u64(&mut prefix, 0);
            push_u64(&mut prefix, 1);
            push_string(&mut prefix, KEY);
            push_u32(&mut prefix, GGUF_TYPE_STRING);
            push_u64(&mut prefix, length);
            SparseFixture {
                length: prefix.len() as u64 + length,
                prefix,
            }
        }

        let encoded_overhead = 8 + KEY.len() as u64 + 4 + 8;
        let exact_value_length = MAX_METADATA_INSPECTION_BYTES - encoded_overhead;

        assert!(
            try_plan(&sparse_string(exact_value_length))
                .unwrap()
                .is_some()
        );
        assert!(
            try_plan(&sparse_string(exact_value_length + 1))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn malformed_keys_small_alignment_and_int64_max_geometry_fall_back() {
        for key in ["Bad Key", "_", "a_", "a__b", "1abc", "a.1b"] {
            let bad_key = metadata(key, GGUF_TYPE_UINT8, vec![0]);
            assert_raw_fallback(&build(3, &[bad_key], &[], DEFAULT_ALIGNMENT));
        }

        let small_alignment = metadata("general.alignment", GGUF_TYPE_UINT32, u32_value(4));
        assert_raw_fallback(&build(3, &[small_alignment], &[], 4));

        let too_many_elements = [i64::MAX as u64];
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("impossible", &too_many_elements, 24)],
            DEFAULT_ALIGNMENT,
        ));
    }

    #[test]
    fn pinned_ggml_table_covers_dense_and_quantized_types() {
        let expected = [
            (0, (1, 4)),
            (1, (1, 2)),
            (2, (32, 18)),
            (3, (32, 20)),
            (6, (32, 22)),
            (7, (32, 24)),
            (8, (32, 34)),
            (9, (32, 40)),
            (10, (256, 84)),
            (11, (256, 110)),
            (12, (256, 144)),
            (13, (256, 176)),
            (14, (256, 210)),
            (15, (256, 292)),
            (16, (256, 66)),
            (17, (256, 74)),
            (18, (256, 98)),
            (19, (256, 50)),
            (20, (32, 18)),
            (21, (256, 110)),
            (22, (256, 82)),
            (23, (256, 136)),
            (24, (1, 1)),
            (25, (1, 2)),
            (26, (1, 4)),
            (27, (1, 8)),
            (28, (1, 8)),
            (29, (256, 56)),
            (30, (1, 2)),
            (34, (256, 54)),
            (35, (256, 66)),
            (39, (32, 17)),
            (40, (64, 36)),
            (41, (128, 18)),
            (42, (64, 18)),
        ];
        for ggml_type in 0..=43 {
            let pinned = expected
                .iter()
                .find_map(|(candidate, layout)| (*candidate == ggml_type).then_some(*layout));
            assert_eq!(ggml_layout(ggml_type).ok(), pinned, "GGML type {ggml_type}");
        }
        assert!(ggml_layout(u32::MAX).is_err());
    }

    #[test]
    fn plans_representative_legacy_k_iq_tq_and_float4_tensors() {
        let tensors = [
            tensor("f16", &[7], 1),
            tensor("q4_0", &[32], 2),
            tensor("q2_k", &[256], 10),
            tensor("iq2_xs", &[256], 17),
            tensor("tq1_0", &[256], 34),
            tensor("nvfp4", &[64], 40),
            tensor("q1_0", &[128], 41),
        ];
        let bytes = build(3, &[], &tensors, DEFAULT_ALIGNMENT);
        let plan = try_plan(&SliceFixture(&bytes)).unwrap().unwrap();
        let lengths: Vec<_> = plan
            .regions
            .iter()
            .filter(|region| region.kind == RegionKind::Tensor)
            .map(|region| region.length)
            .collect();
        assert_eq!(lengths, [14, 18, 84, 74, 54, 36, 18]);
        plan.validate().unwrap();
    }

    #[test]
    fn tensor_chunks_are_tensor_relative_at_the_64_mib_boundary() {
        let exact = tensor("exact", &[MAX_OBJECT_SIZE / 4], 0);
        let over = tensor("over", &[MAX_OBJECT_SIZE / 4 + 1], 0);
        let tensors = [exact, over];
        let (prefix, length) = build_prefix(3, &[], &tensors, DEFAULT_ALIGNMENT);
        let source = SparseFixture { prefix, length };

        let plan = try_plan(&source).unwrap().unwrap();
        let tensor_regions: Vec<_> = plan
            .regions
            .iter()
            .filter(|region| region.kind == RegionKind::Tensor)
            .map(|region| region.length)
            .collect();
        assert_eq!(tensor_regions, [MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, 4]);
        plan.validate().unwrap();
    }

    #[test]
    fn rejects_wrong_magic_versions_and_big_endian() {
        assert_raw_fallback(b"GGU");
        assert_raw_fallback(b"GGUX");

        for version in [0, 1, 4, u32::MAX] {
            let bytes = build(version, &[], &[], DEFAULT_ALIGNMENT);
            assert_raw_fallback(&bytes);
        }

        let mut big_endian = build(3, &[], &[], DEFAULT_ALIGNMENT);
        big_endian[4..8].copy_from_slice(&3_u32.to_be_bytes());
        assert_raw_fallback(&big_endian);
    }

    #[test]
    fn rejects_duplicate_metadata_keys_and_tensor_names() {
        let duplicate_keys = vec![
            metadata("same", GGUF_TYPE_UINT8, vec![0]),
            metadata("same", GGUF_TYPE_UINT8, vec![1]),
        ];
        assert_raw_fallback(&build(3, &duplicate_keys, &[], DEFAULT_ALIGNMENT));

        let duplicate_tensors = [tensor("same", &[1], 0), tensor("same", &[1], 0)];
        assert_raw_fallback(&build(3, &[], &duplicate_tensors, DEFAULT_ALIGNMENT));
    }

    #[test]
    fn rejects_bad_alignment_type_value_and_invalid_metadata() {
        for alignment in [0, 3, 48] {
            let entries = vec![metadata(
                "general.alignment",
                GGUF_TYPE_UINT32,
                u32_value(alignment),
            )];
            assert_raw_fallback(&build(3, &entries, &[], DEFAULT_ALIGNMENT));
        }

        let wrong_type = vec![metadata(
            "general.alignment",
            GGUF_TYPE_UINT64,
            64_u64.to_le_bytes().to_vec(),
        )];
        assert_raw_fallback(&build(3, &wrong_type, &[], DEFAULT_ALIGNMENT));

        let invalid_bool = vec![metadata("bad", GGUF_TYPE_BOOL, vec![2])];
        assert_raw_fallback(&build(3, &invalid_bool, &[], DEFAULT_ALIGNMENT));

        let invalid_utf8 = vec![metadata("bad", GGUF_TYPE_STRING, string_value(&[0xff]))];
        assert_raw_fallback(&build(3, &invalid_utf8, &[], DEFAULT_ALIGNMENT));
    }

    #[test]
    fn rejects_untrusted_counts_lengths_and_nesting_before_allocation() {
        for (tensor_count, metadata_count) in [
            (MAX_TENSOR_COUNT + 1, 0),
            (0, MAX_METADATA_COUNT + 1),
            (u64::MAX, u64::MAX),
        ] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&MAGIC);
            push_u32(&mut bytes, 3);
            push_u64(&mut bytes, tensor_count);
            push_u64(&mut bytes, metadata_count);
            assert_raw_fallback(&bytes);
        }

        let mut oversized_key = Vec::new();
        oversized_key.extend_from_slice(&MAGIC);
        push_u32(&mut oversized_key, 3);
        push_u64(&mut oversized_key, 0);
        push_u64(&mut oversized_key, 1);
        push_u64(&mut oversized_key, u64::MAX);
        oversized_key.extend_from_slice(&[0; 5]);
        assert_raw_fallback(&oversized_key);

        let oversized_array = vec![metadata(
            "too_many",
            GGUF_TYPE_ARRAY,
            array_value(GGUF_TYPE_UINT8, Vec::new(), MAX_ARRAY_ELEMENTS + 1),
        )];
        assert_raw_fallback(&build(3, &oversized_array, &[], DEFAULT_ALIGNMENT));

        let oversized_name = "n".repeat((MAX_TENSOR_NAME_LEN + 1) as usize);
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor(&oversized_name, &[1], 0)],
            DEFAULT_ALIGNMENT,
        ));

        let mut nested = array_value(GGUF_TYPE_UINT8, Vec::new(), 0);
        for _ in 0..=MAX_ARRAY_DEPTH {
            nested = array_value(GGUF_TYPE_ARRAY, nested, 1);
        }
        let excessive_nesting = vec![metadata("nested", GGUF_TYPE_ARRAY, nested)];
        assert_raw_fallback(&build(3, &excessive_nesting, &[], DEFAULT_ALIGNMENT));
    }

    #[test]
    fn rejects_dimensions_types_divisibility_overflow_and_gaps() {
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("five-d", &[1, 1, 1, 1, 1], 0)],
            DEFAULT_ALIGNMENT,
        ));
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("removed", &[32], 4)],
            DEFAULT_ALIGNMENT,
        ));
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("future", &[1], 43)],
            DEFAULT_ALIGNMENT,
        ));
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("partial-block", &[31], 2)],
            DEFAULT_ALIGNMENT,
        ));
        assert_raw_fallback(&build(
            3,
            &[],
            &[tensor("overflow", &[i64::MAX as u64, 2], 0)],
            DEFAULT_ALIGNMENT,
        ));

        let tensors = [
            tensor("first", &[1], 0),
            TensorSpec {
                name: "gap",
                dimensions: &[1],
                ggml_type: 0,
                offset: Some(64),
            },
        ];
        assert_raw_fallback(&build(3, &[], &tensors, DEFAULT_ALIGNMENT));
    }

    #[test]
    fn rejects_trailing_bytes_and_every_truncation_point() {
        let tensors = [tensor("dense", &[8, 2], 0), tensor("quant", &[256], 12)];
        let valid = build(3, &[], &tensors, DEFAULT_ALIGNMENT);

        let mut trailing = valid.clone();
        trailing.push(0);
        assert_raw_fallback(&trailing);

        for length in 0..valid.len() {
            assert_raw_fallback(&valid[..length]);
        }
        assert!(try_plan(&SliceFixture(&valid)).unwrap().is_some());
    }

    #[test]
    fn accepts_empty_tensors_without_zero_length_regions() {
        let tensors = [tensor("empty", &[0], 0), tensor("data", &[1], 0)];
        let bytes = build(3, &[], &tensors, DEFAULT_ALIGNMENT);
        let plan = try_plan(&SliceFixture(&bytes)).unwrap().unwrap();
        assert!(plan.regions.iter().all(|region| region.length > 0));
        assert_eq!(
            plan.regions
                .iter()
                .filter(|region| region.kind == RegionKind::Tensor)
                .count(),
            1
        );
        plan.validate().unwrap();
    }
}
