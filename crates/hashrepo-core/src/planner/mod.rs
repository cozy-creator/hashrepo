use std::io;

use thiserror::Error;

mod gguf;
mod safetensors;

pub const MAX_OBJECT_SIZE: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_OBJECT_COUNT: usize = 1_000_000;

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
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) kind: RegionKind,
}

impl Region {
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    #[must_use]
    pub const fn kind(&self) -> RegionKind {
        self.kind
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub(crate) planner: PlannerId,
    pub(crate) file_size: u64,
    pub(crate) regions: Vec<Region>,
}

impl Plan {
    #[must_use]
    pub const fn planner(&self) -> PlannerId {
        self.planner
    }

    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub(crate) fn validate(&self) -> Result<(), PlanError> {
        if self.regions.len() > MAX_OBJECT_COUNT {
            return Err(PlanError::ObjectLimit);
        }
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
    #[error("source changed during planning or hashing")]
    SourceChanged(#[source] io::Error),
    #[error("planner produced invalid file coverage")]
    InvalidCoverage,
    #[error("planner exceeds bounded object cardinality")]
    ObjectLimit,
    #[error("planner could not reserve bounded working memory")]
    ResourceExhausted,
    #[error("plan source length mismatch: expected {expected}, got {actual}")]
    SourceLengthMismatch { expected: u64, actual: u64 },
}

pub trait ByteSource {
    fn len(&self) -> u64;
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()>;
    /// Refuses if this source no longer represents the immutable snapshot
    /// captured when it was constructed.
    fn check_unchanged(&self) -> io::Result<()>;

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

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

pub fn plan<S: ByteSource + ?Sized>(source: &S) -> Result<Plan, PlanError> {
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    let result = plan_once(source);
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    result
}

fn plan_once<S: ByteSource + ?Sized>(source: &S) -> Result<Plan, PlanError> {
    if source.len() < 10 {
        let plan = raw_plan(source.len())?;
        plan.validate()?;
        return Ok(plan);
    }
    if source.len() >= 24 {
        if let Some(plan) = gguf::try_plan(source)? {
            plan.validate()?;
            return Ok(plan);
        }
    }
    if let Some(plan) = safetensors::try_plan(source)? {
        plan.validate()?;
        return Ok(plan);
    }
    let plan = raw_plan(source.len())?;
    plan.validate()?;
    Ok(plan)
}

pub(crate) fn raw_plan(file_size: u64) -> Result<Plan, PlanError> {
    let mut regions = Vec::new();
    append_split_region(&mut regions, 0, file_size, RegionKind::Raw)?;
    Ok(Plan {
        planner: PlannerId::RawFixed64mV1,
        file_size,
        regions,
    })
}

pub(crate) fn append_split_region(
    regions: &mut Vec<Region>,
    offset: u64,
    length: u64,
    kind: RegionKind,
) -> Result<(), PlanError> {
    let object_count = if length == 0 {
        0
    } else {
        usize::try_from(length.div_ceil(MAX_OBJECT_SIZE)).map_err(|_| PlanError::ObjectLimit)?
    };
    let new_count = regions
        .len()
        .checked_add(object_count)
        .filter(|count| *count <= MAX_OBJECT_COUNT)
        .ok_or(PlanError::ObjectLimit)?;
    regions
        .try_reserve_exact(new_count - regions.len())
        .map_err(|_| PlanError::ResourceExhausted)?;

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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_plan_uses_bounded_contiguous_regions() {
        let plan = raw_plan(MAX_OBJECT_SIZE + 1).unwrap();
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

    #[test]
    fn coverage_validation_refuses_every_partition_escape() {
        let invalid = [
            vec![Region {
                offset: 1,
                length: 3,
                kind: RegionKind::Raw,
            }],
            vec![
                Region {
                    offset: 0,
                    length: 2,
                    kind: RegionKind::Raw,
                },
                Region {
                    offset: 3,
                    length: 1,
                    kind: RegionKind::Raw,
                },
            ],
            vec![
                Region {
                    offset: 0,
                    length: 3,
                    kind: RegionKind::Raw,
                },
                Region {
                    offset: 2,
                    length: 2,
                    kind: RegionKind::Raw,
                },
            ],
            vec![Region {
                offset: 0,
                length: 0,
                kind: RegionKind::Raw,
            }],
            vec![Region {
                offset: 0,
                length: MAX_OBJECT_SIZE + 1,
                kind: RegionKind::Raw,
            }],
            vec![Region {
                offset: 0,
                length: 3,
                kind: RegionKind::Raw,
            }],
        ];

        for regions in invalid {
            let plan = Plan {
                planner: PlannerId::RawFixed64mV1,
                file_size: 4,
                regions,
            };
            assert_eq!(
                plan.validate().unwrap_err().to_string(),
                "planner produced invalid file coverage"
            );
        }
    }

    #[test]
    fn coverage_validation_refuses_unbounded_object_cardinality() {
        let regions = (0..=MAX_OBJECT_COUNT)
            .map(|offset| Region {
                offset: offset as u64,
                length: 1,
                kind: RegionKind::Raw,
            })
            .collect();
        let plan = Plan {
            planner: PlannerId::RawFixed64mV1,
            file_size: MAX_OBJECT_COUNT as u64 + 1,
            regions,
        };

        assert!(matches!(plan.validate(), Err(PlanError::ObjectLimit)));
    }

    struct HugeSource;

    impl ByteSource for HugeSource {
        fn len(&self) -> u64 {
            (MAX_OBJECT_COUNT as u64 + 1) * MAX_OBJECT_SIZE
        }

        fn read_exact_at(&self, _offset: u64, destination: &mut [u8]) -> io::Result<()> {
            destination.fill(0);
            Ok(())
        }

        fn check_unchanged(&self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn raw_planning_refuses_cardinality_before_allocating_regions() {
        assert!(matches!(plan(&HugeSource), Err(PlanError::ObjectLimit)));
    }

    #[test]
    fn empty_file_has_one_canonical_empty_partition() {
        let empty = raw_plan(0).unwrap();
        assert!(empty.regions.is_empty());
        empty.validate().unwrap();

        let noncanonical = Plan {
            planner: PlannerId::RawFixed64mV1,
            file_size: 0,
            regions: vec![Region {
                offset: 0,
                length: 0,
                kind: RegionKind::Raw,
            }],
        };
        assert!(matches!(
            noncanonical.validate(),
            Err(PlanError::InvalidCoverage)
        ));
    }
}
