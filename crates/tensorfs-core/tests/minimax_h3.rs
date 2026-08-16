//! Opt-in proof over the pinned official MiniMax-H3 transformer.
//!
//! Point `TENSORFS_MINIMAX_H3_DIR` at a directory holding the 14 shards of
//! `MiniMaxAI/MiniMax-H3` at revision
//! `42ed227ee7df40d41602854ae760620d6eb651fe`. The corpus is about 62 GiB, so
//! this is never a CI gate; without the variable the test reports skip and
//! passes.
//!
//! The reduction is built as a `ByteSource` over the real shard bytes rather
//! than a second materialized copy, so the surviving tensor digests are hashed
//! from genuine payload bytes at *different absolute offsets*. That is exactly
//! the invariant under test: tensor identity is offset-, shard- and
//! order-independent.

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use tensorfs_core::object::plan_and_hash;
use tensorfs_core::planner::{ByteSource, MAX_OBJECT_SIZE, PlannerId, RegionKind};
use tensorfs_core::source::FileByteSource;

const ENV_DIR: &str = "TENSORFS_MINIMAX_H3_DIR";
const SHARD_COUNT: usize = 14;
const ADALN_MARKER: &str = "adaln_proj.linear.";

const FULL_TENSORS: usize = 638;
const FULL_BYTES: u64 = 66_280_430_080;
const FULL_REFS: usize = 1_508;
const ADALN_TENSORS: usize = 100;
const ADALN_BYTES: u64 = 26_020_915_200;
const ADALN_REFS: usize = 450;
const REDUCED_TENSORS: usize = 538;
const REDUCED_BYTES: u64 = 40_259_514_880;
const REDUCED_REFS: usize = 1_058;

#[derive(Clone, Debug)]
struct Tensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

impl Tensor {
    fn length(&self) -> u64 {
        self.end - self.start
    }

    fn refs(&self) -> usize {
        if self.length() == 0 {
            0
        } else {
            usize::try_from(self.length().div_ceil(MAX_OBJECT_SIZE)).expect("bounded ref count")
        }
    }
}

fn corpus_dir() -> Option<PathBuf> {
    let raw = env::var_os(ENV_DIR)?;
    let path = PathBuf::from(raw);
    path.is_dir().then_some(path)
}

fn shard_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!(
        "diffusion_pytorch_model-{index:05}-of-{SHARD_COUNT:05}.safetensors"
    ))
}

/// Reads only the 8-byte prefix and JSON header, never a payload byte.
fn read_header(path: &Path) -> io::Result<(u64, Vec<Tensor>)> {
    let mut file = File::open(path)?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)?;
    let header_size = u64::from_le_bytes(prefix);
    let mut raw = vec![0_u8; usize::try_from(header_size).expect("bounded header")];
    file.read_exact(&mut raw)?;

    let parsed: serde_json::Value = serde_json::from_slice(&raw).expect("shard header is JSON");
    let object = parsed.as_object().expect("shard header is an object");

    let mut tensors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let entry = value.as_object().expect("tensor descriptor is an object");
        let offsets = entry["data_offsets"]
            .as_array()
            .expect("data_offsets is an array");
        tensors.push(Tensor {
            name: name.clone(),
            dtype: entry["dtype"]
                .as_str()
                .expect("dtype is a string")
                .to_owned(),
            shape: entry["shape"]
                .as_array()
                .expect("shape is an array")
                .iter()
                .map(|dimension| dimension.as_u64().expect("dimension is u64"))
                .collect(),
            start: offsets[0].as_u64().expect("start is u64"),
            end: offsets[1].as_u64().expect("end is u64"),
        });
    }
    tensors.sort_by_key(|tensor| (tensor.start, tensor.end));
    Ok((8 + header_size, tensors))
}

/// Serializes a safetensors header for `tensors`, repacked contiguously from
/// zero. Declaration order follows ascending original offset.
fn build_header(tensors: &[Tensor]) -> Vec<u8> {
    let mut json = String::from("{");
    let mut offset = 0_u64;
    for (index, tensor) in tensors.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let end = offset + tensor.length();
        json.push_str(&serde_json::to_string(&tensor.name).expect("name serializes"));
        json.push_str(&format!(
            r#":{{"dtype":"{}","shape":[{shape}],"data_offsets":[{offset},{end}]}}"#,
            tensor.dtype
        ));
        offset = end;
    }
    json.push('}');

    let bytes = json.into_bytes();
    let mut framed = Vec::with_capacity(8 + bytes.len());
    framed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    framed.extend_from_slice(&bytes);
    framed
}

struct Segment {
    reduced_start: u64,
    length: u64,
    file_offset: u64,
}

/// A shard with a tensor family removed, presented as ordinary file bytes
/// without materializing a second copy.
struct ReducedShard {
    header: Vec<u8>,
    segments: Vec<Segment>,
    total_len: u64,
    file: FileByteSource,
}

impl ReducedShard {
    fn new(path: &Path, header_end: u64, survivors: &[Tensor]) -> io::Result<Self> {
        let header = build_header(survivors);
        let mut segments = Vec::with_capacity(survivors.len());
        let mut reduced_start = 0_u64;
        for tensor in survivors {
            let length = tensor.length();
            if length == 0 {
                continue;
            }
            segments.push(Segment {
                reduced_start,
                length,
                file_offset: header_end + tensor.start,
            });
            reduced_start += length;
        }
        Ok(Self {
            total_len: header.len() as u64 + reduced_start,
            header,
            segments,
            file: FileByteSource::open(path)?,
        })
    }
}

impl ByteSource for ReducedShard {
    fn len(&self) -> u64 {
        self.total_len
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let header_len = self.header.len() as u64;
        let mut written = 0_usize;
        let mut position = offset;

        while written < destination.len() {
            if position < header_len {
                let start = usize::try_from(position).expect("header offset fits");
                let take = (self.header.len() - start).min(destination.len() - written);
                destination[written..written + take]
                    .copy_from_slice(&self.header[start..start + take]);
                written += take;
                position += take as u64;
                continue;
            }

            let payload = position - header_len;
            let segment = self
                .segments
                .iter()
                .find(|segment| {
                    payload >= segment.reduced_start
                        && payload < segment.reduced_start + segment.length
                })
                .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
            let within = payload - segment.reduced_start;
            let available = usize::try_from(segment.length - within).unwrap_or(usize::MAX);
            let take = available.min(destination.len() - written);
            self.file.read_exact_at(
                segment.file_offset + within,
                &mut destination[written..written + take],
            )?;
            written += take;
            position += take as u64;
        }
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        self.file.check_unchanged()
    }
}

/// Walks a plan's tensor objects in emission order and attributes each to the
/// tensor that produced it. The planner emits header regions first, then
/// tensors in ascending start order, so the mapping is exact rather than
/// inferred from digests.
fn digests_by_tensor(
    plan: &tensorfs_core::object::HashedPlan,
    tensors: &[Tensor],
) -> BTreeMap<String, Vec<String>> {
    let mut objects = plan
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor);

    let mut mapped = BTreeMap::new();
    for tensor in tensors {
        let mut digests = Vec::new();
        for _ in 0..tensor.refs() {
            let object = objects
                .next()
                .expect("plan emits one object per tensor subdivision");
            digests.push(object.digest().to_string());
        }
        if !digests.is_empty() {
            mapped.insert(tensor.name.clone(), digests);
        }
    }
    assert!(
        objects.next().is_none(),
        "every tensor object must be attributed"
    );
    mapped
}

#[test]
fn pinned_minimax_h3_reduction_preserves_every_surviving_tensor_object() {
    let Some(dir) = corpus_dir() else {
        eprintln!("skipping: set {ENV_DIR} to the pinned 14-shard MiniMax-H3 corpus");
        return;
    };

    let mut full_tensors = 0_usize;
    let mut full_bytes = 0_u64;
    let mut full_refs = 0_usize;
    let mut adaln_tensors = 0_usize;
    let mut adaln_bytes = 0_u64;
    let mut adaln_refs = 0_usize;
    let mut reduced_refs = 0_usize;
    let mut reduced_tensors = 0_usize;
    let mut reduced_bytes = 0_u64;
    let mut header_objects = 0_usize;
    let mut preserved = 0_usize;

    for index in 1..=SHARD_COUNT {
        let path = shard_path(&dir, index);
        let (header_end, tensors) = read_header(&path).expect("shard header reads");

        let full_source = FileByteSource::open(&path).expect("shard opens");
        let full_plan = plan_and_hash(&full_source).expect("shard plans and hashes");
        assert_eq!(
            full_plan.planner(),
            PlannerId::SafetensorsV1,
            "{path:?} must plan semantically"
        );

        let survivors = tensors
            .iter()
            .filter(|tensor| !tensor.name.contains(ADALN_MARKER))
            .cloned()
            .collect::<Vec<_>>();
        let removed = tensors.len() - survivors.len();

        full_tensors += tensors.len();
        full_bytes += tensors.iter().map(Tensor::length).sum::<u64>();
        full_refs += tensors.iter().map(Tensor::refs).sum::<usize>();
        adaln_tensors += removed;
        adaln_bytes += tensors
            .iter()
            .filter(|tensor| tensor.name.contains(ADALN_MARKER))
            .map(Tensor::length)
            .sum::<u64>();
        adaln_refs += tensors
            .iter()
            .filter(|tensor| tensor.name.contains(ADALN_MARKER))
            .map(Tensor::refs)
            .sum::<usize>();
        header_objects += full_plan
            .objects()
            .iter()
            .filter(|object| object.kind() == RegionKind::Header)
            .count();

        let reduced_source =
            ReducedShard::new(&path, header_end, &survivors).expect("reduction opens");
        let reduced_plan = plan_and_hash(&reduced_source).expect("reduction plans and hashes");
        assert_eq!(
            reduced_plan.planner(),
            PlannerId::SafetensorsV1,
            "{path:?} reduction must plan semantically"
        );

        reduced_tensors += survivors.len();
        reduced_bytes += survivors.iter().map(Tensor::length).sum::<u64>();
        reduced_refs += survivors.iter().map(Tensor::refs).sum::<usize>();

        let full_map = digests_by_tensor(&full_plan, &tensors);
        let reduced_map = digests_by_tensor(&reduced_plan, &survivors);

        for tensor in &survivors {
            let Some(expected) = full_map.get(&tensor.name) else {
                continue;
            };
            let actual = reduced_map
                .get(&tensor.name)
                .unwrap_or_else(|| panic!("{} missing from the reduction", tensor.name));
            assert_eq!(
                expected, actual,
                "{} must keep byte-identical objects after the reduction",
                tensor.name
            );
            preserved += actual.len();
        }

        for tensor in &tensors {
            if tensor.name.contains(ADALN_MARKER) {
                assert!(
                    !reduced_map.contains_key(&tensor.name),
                    "{} must be absent from the reduction",
                    tensor.name
                );
            }
        }
    }

    assert_eq!(full_tensors, FULL_TENSORS);
    assert_eq!(full_bytes, FULL_BYTES);
    assert_eq!(full_refs, FULL_REFS);
    assert_eq!(header_objects, SHARD_COUNT);
    assert_eq!(adaln_tensors, ADALN_TENSORS);
    assert_eq!(adaln_bytes, ADALN_BYTES);
    assert_eq!(adaln_refs, ADALN_REFS);
    assert_eq!(reduced_tensors, REDUCED_TENSORS);
    assert_eq!(reduced_bytes, REDUCED_BYTES);
    assert_eq!(reduced_refs, REDUCED_REFS);

    // The intersection claim: every one of the surviving references, and no
    // fewer, is byte-identical across the two layouts.
    assert_eq!(
        preserved, REDUCED_REFS,
        "the full and reduced plans must share exactly {REDUCED_REFS} tensor references"
    );
}
