use std::fmt;

use sha2::{Digest, Sha256};

use crate::planner::{ByteSource, Plan, PlanError, PlannerId, RegionKind};

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
    pub digest: ObjectDigest,
    pub length: u64,
    pub kind: RegionKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashedPlan {
    pub planner: PlannerId,
    pub file_size: u64,
    pub objects: Vec<PlannedObject>,
}

pub fn hash_plan<S: ByteSource + ?Sized>(source: &S, plan: &Plan) -> Result<HashedPlan, PlanError> {
    plan.validate()?;
    if source.len() != plan.file_size {
        return Err(PlanError::SourceLengthMismatch {
            expected: plan.file_size,
            actual: source.len(),
        });
    }

    let mut buffer = vec![0_u8; HASH_BUFFER_SIZE];
    let mut objects = Vec::with_capacity(plan.regions.len());
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

    Ok(HashedPlan {
        planner: plan.planner,
        file_size: plan.file_size,
        objects,
    })
}

#[cfg(test)]
mod tests {
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
}
