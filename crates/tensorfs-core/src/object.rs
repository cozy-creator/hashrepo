use std::fmt;

use sha2::{Digest, Sha256};

use crate::planner::{self, ByteSource, Plan, PlanError, PlannerId, RegionKind};

const HASH_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ObjectDigest([u8; 32]);

impl ObjectDigest {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for ObjectDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for ObjectDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedObject {
    digest: ObjectDigest,
    length: u64,
    kind: RegionKind,
}

impl PlannedObject {
    #[must_use]
    pub const fn digest(&self) -> ObjectDigest {
        self.digest
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
pub struct HashedPlan {
    planner: PlannerId,
    file_size: u64,
    objects: Vec<PlannedObject>,
}

impl HashedPlan {
    #[must_use]
    pub const fn planner(&self) -> PlannerId {
        self.planner
    }

    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    #[must_use]
    pub fn objects(&self) -> &[PlannedObject] {
        &self.objects
    }
}

/// Selects the sole canonical planner and hashes each resulting object from one
/// stable source snapshot. Callers cannot supply a planner ID or boundaries.
pub fn plan_and_hash<S: ByteSource + ?Sized>(source: &S) -> Result<HashedPlan, PlanError> {
    let plan = planner::plan(source)?;
    hash_plan(source, &plan)
}

pub(crate) fn hash_plan<S: ByteSource + ?Sized>(
    source: &S,
    plan: &Plan,
) -> Result<HashedPlan, PlanError> {
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    plan.validate()?;
    if source.len() != plan.file_size {
        return Err(PlanError::SourceLengthMismatch {
            expected: plan.file_size,
            actual: source.len(),
        });
    }

    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(HASH_BUFFER_SIZE)
        .map_err(|_| PlanError::ResourceExhausted)?;
    buffer.resize(HASH_BUFFER_SIZE, 0);
    let mut objects = Vec::new();
    objects
        .try_reserve_exact(plan.regions.len())
        .map_err(|_| PlanError::ResourceExhausted)?;
    for region in &plan.regions {
        let mut hasher = Sha256::new();
        let mut cursor = region.offset;
        let mut remaining = region.length;
        while remaining > 0 {
            let read_length = usize::try_from(remaining.min(HASH_BUFFER_SIZE as u64))
                .expect("bounded hash buffer length fits usize");
            source.read_exact_at(cursor, &mut buffer[..read_length])?;
            hasher.update(&buffer[..read_length]);
            cursor += read_length as u64;
            remaining -= read_length as u64;
        }
        objects.push(PlannedObject {
            digest: ObjectDigest::from_bytes(hasher.finalize().into()),
            length: region.length,
            kind: region.kind,
        });
    }

    let hashed = HashedPlan {
        planner: plan.planner,
        file_size: plan.file_size,
        objects,
    };
    source.check_unchanged().map_err(PlanError::SourceChanged)?;
    Ok(hashed)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::io;

    use super::*;
    use crate::planner::{Plan, Region};

    #[test]
    fn hashes_only_the_bytes_in_each_planned_object() {
        let source = b"sameother-same".as_slice();
        let plan = Plan {
            planner: PlannerId::SafetensorsV1,
            file_size: source.len() as u64,
            regions: vec![
                Region {
                    offset: 0,
                    length: 4,
                    kind: RegionKind::Tensor,
                },
                Region {
                    offset: 4,
                    length: 6,
                    kind: RegionKind::Header,
                },
                Region {
                    offset: 10,
                    length: 4,
                    kind: RegionKind::Tensor,
                },
            ],
        };

        let hashed = hash_plan(source, &plan).unwrap();
        assert_eq!(hashed.objects[0].digest, hashed.objects[2].digest);
        assert_ne!(hashed.objects[0].digest, hashed.objects[1].digest);
        assert_eq!(
            hashed.objects[0].digest.to_string(),
            "sha256:0967115f2813a3541eaef77de9d9d5773f1c0c04314b0bbfe4ff3b3b1c55b5d5"
        );
    }

    #[test]
    fn refuses_a_plan_for_a_different_source_length() {
        let source = b"bytes".as_slice();
        let plan = Plan {
            planner: PlannerId::RawFixed64mV1,
            file_size: 4,
            regions: vec![Region {
                offset: 0,
                length: 4,
                kind: RegionKind::Raw,
            }],
        };

        assert!(matches!(
            hash_plan(source, &plan),
            Err(PlanError::SourceLengthMismatch {
                expected: 4,
                actual: 5
            })
        ));
    }

    struct BrokenSource;

    impl ByteSource for BrokenSource {
        fn len(&self) -> u64 {
            4
        }

        fn read_exact_at(&self, _offset: u64, _destination: &mut [u8]) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "broken"))
        }

        fn check_unchanged(&self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn propagates_source_read_failure_without_emitting_an_object() {
        let plan = Plan {
            planner: PlannerId::RawFixed64mV1,
            file_size: 4,
            regions: vec![Region {
                offset: 0,
                length: 4,
                kind: RegionKind::Raw,
            }],
        };

        assert!(matches!(
            hash_plan(&BrokenSource, &plan),
            Err(PlanError::Read(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    struct ChangingSource {
        bytes: [u8; 3],
        generation: Cell<u64>,
    }

    impl ByteSource for ChangingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
            let start = offset as usize;
            destination.copy_from_slice(&self.bytes[start..start + destination.len()]);
            self.generation.set(1);
            Ok(())
        }

        fn check_unchanged(&self) -> io::Result<()> {
            if self.generation.get() != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "source generation changed",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn combined_planning_and_hashing_refuses_a_torn_source_generation() {
        let source = ChangingSource {
            bytes: *b"abc",
            generation: Cell::new(0),
        };

        assert!(matches!(
            plan_and_hash(&source),
            Err(PlanError::SourceChanged(error)) if error.kind() == io::ErrorKind::InvalidData
        ));
    }
}
