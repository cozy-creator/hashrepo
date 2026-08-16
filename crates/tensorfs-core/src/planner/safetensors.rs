use std::collections::HashSet;
use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use super::{
    ByteSource, MAX_OBJECT_COUNT, Plan, PlanError, PlannerId, RegionKind, append_split_region,
};

const PREFIX_SIZE: u64 = 8;
const MAX_HEADER_SIZE: u64 = 100_000_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorDescriptor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct Header(Vec<(String, TensorDescriptor)>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut names = HashSet::new();
                let mut tensors = Vec::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name.clone()) {
                        return Err(de::Error::custom("duplicate safetensors header key"));
                    }
                    if name == "__metadata__" {
                        map.next_value::<Option<Metadata>>()?;
                    } else {
                        if name.is_empty() {
                            return Err(de::Error::custom("empty safetensors tensor name"));
                        }
                        if tensors.len() >= MAX_OBJECT_COUNT {
                            return Err(de::Error::custom(
                                "safetensors tensor count exceeds object limit",
                            ));
                        }
                        tensors.push((name, map.next_value::<TensorDescriptor>()?));
                    }
                }
                Ok(Header(tensors))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct Metadata;

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MetadataVisitor;

        impl<'de> Visitor<'de> for MetadataVisitor {
            type Value = Metadata;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors string metadata object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key) {
                        return Err(de::Error::custom("duplicate safetensors metadata key"));
                    }
                    map.next_value::<String>()?;
                }
                Ok(Metadata)
            }
        }

        deserializer.deserialize_map(MetadataVisitor)
    }
}

#[derive(Clone, Copy, Debug)]
struct TensorSpan {
    start: u64,
    end: u64,
}

/// Returns a tensor-aligned plan only after proving the complete safetensors
/// structure from its bounded JSON header. Tensor body bytes are never read.
pub(crate) fn try_plan<S: ByteSource + ?Sized>(source: &S) -> Result<Option<Plan>, PlanError> {
    let file_size = source.len();
    if file_size < PREFIX_SIZE + 2 {
        return Ok(None);
    }

    let mut prefix = [0_u8; PREFIX_SIZE as usize];
    source.read_exact_at(0, &mut prefix)?;
    let header_size = u64::from_le_bytes(prefix);
    if !(2..=MAX_HEADER_SIZE).contains(&header_size) {
        return Ok(None);
    }
    let header_end = PREFIX_SIZE
        .checked_add(header_size)
        .expect("the bounded safetensors header cannot overflow u64");
    if header_end > file_size {
        return Ok(None);
    }

    let header_buffer_size = match usize::try_from(header_size) {
        Ok(size) => size,
        Err(_) => return Ok(None),
    };
    let mut header_bytes = Vec::new();
    header_bytes
        .try_reserve_exact(header_buffer_size)
        .map_err(|_| PlanError::ResourceExhausted)?;
    header_bytes.resize(header_buffer_size, 0);
    source.read_exact_at(PREFIX_SIZE, &mut header_bytes)?;
    if header_bytes.first() != Some(&b'{') {
        return Ok(None);
    }

    let mut deserializer = serde_json::Deserializer::from_slice(&header_bytes);
    let header = match Header::deserialize(&mut deserializer) {
        Ok(header) if deserializer.end().is_ok() => header,
        Ok(_) | Err(_) => return Ok(None),
    };

    let data_size = file_size - header_end;
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(header.0.len())
        .map_err(|_| PlanError::ResourceExhausted)?;
    for (_name, tensor) in header.0 {
        let [start, end] = tensor.data_offsets;
        if end < start || tensor_byte_size(&tensor).is_none_or(|size| size != end - start) {
            return Ok(None);
        }
        spans.push(TensorSpan { start, end });
    }
    spans.sort_unstable_by_key(|span| (span.start, span.end));

    let mut cursor = 0_u64;
    for span in &spans {
        if span.start != cursor || span.end > data_size {
            return Ok(None);
        }
        cursor = span.end;
    }
    if cursor != data_size {
        return Ok(None);
    }

    let mut regions = Vec::new();
    match append_split_region(&mut regions, 0, header_end, RegionKind::Header) {
        Ok(()) => {}
        Err(PlanError::ObjectLimit) => return Ok(None),
        Err(error) => return Err(error),
    }
    for span in spans {
        let length = span.end - span.start;
        if length == 0 {
            continue;
        }
        match append_split_region(
            &mut regions,
            header_end + span.start,
            length,
            RegionKind::Tensor,
        ) {
            Ok(()) => {}
            Err(PlanError::ObjectLimit) => return Ok(None),
            Err(error) => return Err(error),
        }
    }

    Ok(Some(Plan {
        planner: PlannerId::SafetensorsV1,
        file_size,
        regions,
    }))
}

fn tensor_byte_size(tensor: &TensorDescriptor) -> Option<u64> {
    let bits_per_element = dtype_bits(&tensor.dtype)?;
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))?;
    let bits = elements.checked_mul(bits_per_element)?;
    if bits % 8 != 0 {
        return None;
    }
    Some(bits / 8)
}

fn dtype_bits(dtype: &str) -> Option<u64> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E5M2FNUZ"
        | "F8_E4M3FNUZ" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "I64" | "U64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}
