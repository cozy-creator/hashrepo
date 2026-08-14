use std::io;

use thiserror::Error;

pub mod gguf;
pub mod safetensors;

pub const MAX_OBJECT_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannerId {
    SafetensorsV1,
    GgufV1,
    RawFixed64mV1,
}

impl PlannerId {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SafetensorsV1 => "safetensors-v1",
            Self::GgufV1 => "gguf-v1",
            Self::RawFixed64mV1 => "raw-fixed-64m-v1",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionKind {
    Header,
    Tensor,
    Raw,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Region {
    pub offset: u64,
    pub length: u64,
    pub kind: RegionKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub planner: PlannerId,
    pub file_size: u64,
    pub regions: Vec<Region>,
}

impl Plan {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.file_size == 0 {
            if self.regions.is_empty() {
                return Ok(());
            }
            return Err(PlanError::InvalidCoverage);
        }

        let mut expected_offset = 0_u64;
        for region in &self.regions {
            if region.offset != expected_offset
                || region.length == 0
                || region.length > MAX_OBJECT_SIZE
            {
                return Err(PlanError::InvalidCoverage);
            }
            expected_offset = expected_offset
                .checked_add(region.length)
                .ok_or(PlanError::InvalidCoverage)?;
        }
        if expected_offset != self.file_size {
            return Err(PlanError::InvalidCoverage);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("source read failed")]
    Read(#[from] io::Error),
    #[error("planner produced invalid file coverage")]
    InvalidCoverage,
    #[error("plan source length mismatch: expected {expected}, got {actual}")]
    SourceLengthMismatch { expected: u64, actual: u64 },
}

pub trait ByteSource {
    fn len(&self) -> u64;
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ByteSource for [u8] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset is too large"))?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        let source = self
            .get(start..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "range is truncated"))?;
        destination.copy_from_slice(source);
        Ok(())
    }
}

pub fn plan(source: &dyn ByteSource) -> Result<Plan, PlanError> {
    if let Some(plan) = gguf::try_plan(source)? {
        plan.validate()?;
        return Ok(plan);
    }
    if let Some(plan) = safetensors::try_plan(source)? {
        plan.validate()?;
        return Ok(plan);
    }
    let plan = raw_plan(source.len());
    plan.validate()?;
    Ok(plan)
}

#[must_use]
pub fn raw_plan(file_size: u64) -> Plan {
    Plan {
        planner: PlannerId::RawFixed64mV1,
        file_size,
        regions: split_region(0, file_size, RegionKind::Raw),
    }
}

#[must_use]
pub(crate) fn split_region(offset: u64, length: u64, kind: RegionKind) -> Vec<Region> {
    let mut regions = Vec::new();
    let mut cursor = offset;
    let mut remaining = length;
    while remaining > 0 {
        let part = remaining.min(MAX_OBJECT_SIZE);
        regions.push(Region {
            offset: cursor,
            length: part,
            kind,
        });
        cursor += part;
        remaining -= part;
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_plan_uses_bounded_contiguous_regions() {
        let plan = raw_plan(MAX_OBJECT_SIZE + 1);
        assert_eq!(plan.planner, PlannerId::RawFixed64mV1);
        assert_eq!(
            plan.regions,
            vec![
                Region {
                    offset: 0,
                    length: MAX_OBJECT_SIZE,
                    kind: RegionKind::Raw,
                },
                Region {
                    offset: MAX_OBJECT_SIZE,
                    length: 1,
                    kind: RegionKind::Raw,
                },
            ]
        );
        plan.validate().unwrap();
    }
}
