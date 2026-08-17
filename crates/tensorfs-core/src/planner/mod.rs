use std::io;

use thiserror::Error;

use crate::contract::{Contract, Stamp};

pub(crate) mod gguf;
pub(crate) mod safetensors;

/// The tensor chunk grid constant: the bound on one tensor-planned object.
/// It is NOT a store admission cap — a blob is one object of any size.
pub const MAX_OBJECT_SIZE: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_OBJECT_COUNT: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannerId {
    SafetensorsV1,
    GgufV1,
    BlobV1,
}

impl PlannerId {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SafetensorsV1 => "safetensors-v1",
            Self::GgufV1 => "gguf-v1",
            Self::BlobV1 => "blob-v1",
        }
    }

    #[must_use]
    pub const fn tensor_format(self) -> Option<TensorFormat> {
        match self {
            Self::SafetensorsV1 => Some(TensorFormat::SafetensorsV1),
            Self::GgufV1 => Some(TensorFormat::GgufV1),
            Self::BlobV1 => None,
        }
    }
}

/// The tensor container formats: exactly the planners whose file bodies carry
/// a record list. `blob-v1` is structurally excluded, so "a chunked blob" is
/// unrepresentable rather than refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorFormat {
    SafetensorsV1,
    GgufV1,
}

impl TensorFormat {
    #[must_use]
    pub const fn planner_id(self) -> PlannerId {
        match self {
            Self::SafetensorsV1 => PlannerId::SafetensorsV1,
            Self::GgufV1 => PlannerId::GgufV1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionKind {
    Header,
    Tensor,
    Blob,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    /// The contract that directed these boundaries, carried from planning to
    /// the manifest so the stamp cannot drift from the cuts it explains.
    pub(crate) contract: Stamp,
    pub(crate) file_size: u64,
    pub(crate) regions: Vec<Region>,
}

impl Plan {
    #[must_use]
    pub const fn planner(&self) -> PlannerId {
        self.planner
    }

    #[must_use]
    pub const fn contract(&self) -> &Stamp {
        &self.contract
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
        // A blob is ONE region covering the whole file, of any size; the
        // 64 MiB grid constant applies only to tensor-planned regions. The
        // scope (is this a tensor container?) decides the shape — the number
        // never enforces a preference.
        let blob = self.planner == PlannerId::BlobV1;
        if blob && self.regions.len() > 1 {
            return Err(PlanError::InvalidCoverage);
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
                || (!blob && region.length > MAX_OBJECT_SIZE)
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
    plan_with(source, None)
}

/// Plans one file under a declared layout contract.
///
/// The contract adds cut points inside fused tensors — nothing else. Byte
/// ORDER is untouched (the load-order ruling: chunk boundaries may move, bytes
/// may not), coverage is still exact, and a contract that does not describe
/// this file simply changes nothing. Passing `None` is the plain per-tensor
/// grid, which is what `contract:none` means.
pub fn plan_with<S: ByteSource + ?Sized>(
    source: &S,
    contract: Option<&Contract>,
) -> Result<Plan, PlanError> {
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    let result = plan_once(source, contract);
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    result
}

fn plan_once<S: ByteSource + ?Sized>(
    source: &S,
    contract: Option<&Contract>,
) -> Result<Plan, PlanError> {
    if source.len() < 10 {
        let plan = blob_plan(source.len());
        plan.validate()?;
        return Ok(plan);
    }
    if source.len() >= 24
        && let Some(plan) = gguf::try_plan(source, contract)?
    {
        plan.validate()?;
        return Ok(plan);
    }
    if let Some(plan) = safetensors::try_plan(source, contract)? {
        plan.validate()?;
        return Ok(plan);
    }
    let plan = blob_plan(source.len());
    plan.validate()?;
    Ok(plan)
}

/// One tensor as a container's header declares it, in logical (row-major)
/// order regardless of the container's own axis convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryTensor {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

impl InventoryTensor {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The canonical dtype name: safetensors' own spelling, and the ggml type
    /// name for GGUF, so one contract vocabulary covers both containers.
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Logical shape, outermost axis first. GGUF's `ne` order is reversed
    /// here, so "axis 0" means the same thing in every container.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Offset of this tensor's first byte within the file.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// The tensor's own extent, excluding any container padding.
    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }
}

/// A tensor container's header inventory: everything contract identification
/// is allowed to look at, and nothing else. No tensor byte is read to build
/// one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInventory {
    pub(crate) format: TensorFormat,
    pub(crate) tensors: Vec<InventoryTensor>,
}

impl TensorInventory {
    #[must_use]
    pub const fn format(&self) -> TensorFormat {
        self.format
    }

    #[must_use]
    pub fn tensors(&self) -> &[InventoryTensor] {
        &self.tensors
    }
}

/// Reads one container's header inventory, or `None` when the file is not a
/// tensor container.
pub fn inventory<S: ByteSource + ?Sized>(source: &S) -> Result<Option<TensorInventory>, PlanError> {
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    if source.len() >= 24
        && let Some(layout) = gguf::read_layout(source)?
    {
        return Ok(Some(gguf::inventory(&layout)));
    }
    if let Some(layout) = safetensors::read_layout(source)? {
        return Ok(Some(safetensors::inventory(&layout)));
    }
    Ok(None)
}

/// Appends one tensor's regions, cut at contract-declared seams first and then
/// on the 64 MiB grid within each part.
///
/// The grid inside a part starts at the part's own start, exactly as it would
/// if that part were a standalone tensor in the split packaging — which is the
/// whole mechanism: the fused file's objects become the split file's objects.
pub(crate) fn append_tensor_regions(
    regions: &mut Vec<Region>,
    offset: u64,
    length: u64,
    seams: &[u64],
    kind: RegionKind,
) -> Result<(), PlanError> {
    let mut cursor = 0_u64;
    for seam in seams {
        if *seam <= cursor || *seam >= length {
            return Err(PlanError::InvalidCoverage);
        }
        append_split_region(regions, offset + cursor, seam - cursor, kind)?;
        cursor = *seam;
    }
    append_split_region(regions, offset + cursor, length - cursor, kind)
}

/// The whole-blob plan for every non-tensor file: one unchunked region of any
/// size (none when empty). There is no raw grid and no fallback splitting.
pub(crate) fn blob_plan(file_size: u64) -> Plan {
    let regions = if file_size == 0 {
        Vec::new()
    } else {
        vec![Region {
            offset: 0,
            length: file_size,
            kind: RegionKind::Blob,
        }]
    };
    Plan {
        planner: PlannerId::BlobV1,
        contract: Stamp::None,
        file_size,
        regions,
    }
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
    fn a_blob_plan_is_one_region_of_any_size() {
        let plan = blob_plan(5 * (MAX_OBJECT_SIZE + 1));
        assert_eq!(plan.planner, PlannerId::BlobV1);
        assert_eq!(
            plan.regions,
            vec![Region {
                offset: 0,
                length: 5 * (MAX_OBJECT_SIZE + 1),
                kind: RegionKind::Blob,
            }]
        );
        plan.validate().unwrap();
    }

    #[test]
    fn a_blob_plan_never_splits_at_the_tensor_grid() {
        let plan = blob_plan(MAX_OBJECT_SIZE + 1);
        assert_eq!(plan.regions.len(), 1, "the raw grid is dead");
        plan.validate().unwrap();
    }

    #[test]
    fn a_multi_region_blob_plan_refuses() {
        let plan = Plan {
            contract: Stamp::None,
            planner: PlannerId::BlobV1,
            file_size: 4,
            regions: vec![
                Region {
                    offset: 0,
                    length: 2,
                    kind: RegionKind::Blob,
                },
                Region {
                    offset: 2,
                    length: 2,
                    kind: RegionKind::Blob,
                },
            ],
        };
        assert!(matches!(plan.validate(), Err(PlanError::InvalidCoverage)));
    }

    #[test]
    fn tensor_regions_keep_the_grid_cap_and_blob_regions_do_not() {
        let oversized = |planner| Plan {
            contract: Stamp::None,
            planner,
            file_size: MAX_OBJECT_SIZE + 1,
            regions: vec![Region {
                offset: 0,
                length: MAX_OBJECT_SIZE + 1,
                kind: RegionKind::Tensor,
            }],
        };
        assert!(matches!(
            oversized(PlannerId::SafetensorsV1).validate(),
            Err(PlanError::InvalidCoverage)
        ));
        oversized(PlannerId::BlobV1).validate().unwrap();
    }

    #[test]
    fn coverage_validation_refuses_every_partition_escape() {
        let invalid = [
            vec![Region {
                offset: 1,
                length: 3,
                kind: RegionKind::Tensor,
            }],
            vec![
                Region {
                    offset: 0,
                    length: 2,
                    kind: RegionKind::Tensor,
                },
                Region {
                    offset: 3,
                    length: 1,
                    kind: RegionKind::Tensor,
                },
            ],
            vec![
                Region {
                    offset: 0,
                    length: 3,
                    kind: RegionKind::Tensor,
                },
                Region {
                    offset: 2,
                    length: 2,
                    kind: RegionKind::Tensor,
                },
            ],
            vec![Region {
                offset: 0,
                length: 0,
                kind: RegionKind::Tensor,
            }],
            vec![Region {
                offset: 0,
                length: MAX_OBJECT_SIZE + 1,
                kind: RegionKind::Tensor,
            }],
            vec![Region {
                offset: 0,
                length: 3,
                kind: RegionKind::Tensor,
            }],
        ];

        for regions in invalid {
            let plan = Plan {
                contract: Stamp::None,
                planner: PlannerId::SafetensorsV1,
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
                kind: RegionKind::Tensor,
            })
            .collect();
        let plan = Plan {
            contract: Stamp::None,
            planner: PlannerId::SafetensorsV1,
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
    fn a_huge_non_tensor_source_plans_as_one_blob_not_a_grid() {
        let plan = plan(&HugeSource).unwrap();
        assert_eq!(plan.planner, PlannerId::BlobV1);
        assert_eq!(plan.regions.len(), 1);
        assert_eq!(plan.regions[0].length, HugeSource.len());
    }

    #[test]
    fn empty_file_has_one_canonical_empty_partition() {
        let empty = blob_plan(0);
        assert!(empty.regions.is_empty());
        empty.validate().unwrap();

        let noncanonical = Plan {
            contract: Stamp::None,
            planner: PlannerId::BlobV1,
            file_size: 0,
            regions: vec![Region {
                offset: 0,
                length: 0,
                kind: RegionKind::Blob,
            }],
        };
        assert!(matches!(
            noncanonical.validate(),
            Err(PlanError::InvalidCoverage)
        ));
    }
}
