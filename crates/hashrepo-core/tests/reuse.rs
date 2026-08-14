use std::collections::HashSet;
use std::io;

use hashrepo_core::object::{HashedPlan, plan_and_hash};
use hashrepo_core::planner::{ByteSource, MAX_OBJECT_SIZE, PlannerId, RegionKind, plan};

struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset too large"))?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        destination.copy_from_slice(
            self.0
                .get(start..end)
                .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?,
        );
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

fn safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, bytes)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let end = offset + bytes.len() as u64;
        header.push_str(&serde_json::to_string(name).expect("tensor name serializes"));
        header.push_str(&format!(
            r#":{{"dtype":"U8","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            bytes.len()
        ));
        offset = end;
    }
    header.push('}');

    let mut file = Vec::with_capacity(8 + header.len() + offset as usize);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, bytes) in tensors {
        file.extend_from_slice(bytes);
    }
    file
}

fn hashed(bytes: &[u8]) -> HashedPlan {
    let source = SliceSource(bytes);
    let hashed = plan_and_hash(&source).expect("valid fixture plans and hashes");
    assert_eq!(hashed.planner(), PlannerId::SafetensorsV1);
    hashed
}

#[test]
fn borrowed_slices_pass_directly_through_the_closed_registry() {
    let bytes = b"ordinary raw bytes".as_slice();
    let planned = plan(bytes).unwrap();

    assert_eq!(planned.planner(), PlannerId::RawFixed64mV1);
}

fn tensor_digests(hashed: &HashedPlan) -> Vec<String> {
    hashed
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .map(|object| object.digest().to_string())
        .collect()
}

fn changed_object_indexes(before: &HashedPlan, after: &HashedPlan) -> HashSet<usize> {
    assert_eq!(before.objects().len(), after.objects().len());
    before
        .objects()
        .iter()
        .zip(after.objects())
        .enumerate()
        .filter_map(|(index, (left, right))| (left.digest() != right.digest()).then_some(index))
        .collect()
}

#[test]
fn insertion_deletion_and_reordering_preserve_every_surviving_tensor_object() {
    let a = b"attention-weights".as_slice();
    let b = b"adaln-weights".as_slice();
    let c = b"mlp-weights".as_slice();
    let inserted = b"new-tensor".as_slice();
    let parent = hashed(&safetensors(&[("a", a), ("b", b), ("c", c)]));
    let child = hashed(&safetensors(&[("inserted", inserted), ("c", c), ("a", a)]));

    let parent_objects = tensor_digests(&parent).into_iter().collect::<HashSet<_>>();
    let child_objects = tensor_digests(&child).into_iter().collect::<HashSet<_>>();
    let expected = hashed(&safetensors(&[("a", a), ("c", c)]));
    let expected_shared = tensor_digests(&expected)
        .into_iter()
        .collect::<HashSet<_>>();

    assert_eq!(
        parent_objects
            .intersection(&child_objects)
            .cloned()
            .collect::<HashSet<_>>(),
        expected_shared
    );
}

#[test]
fn removing_one_small_tensor_does_not_repack_any_surviving_tensor() {
    let payloads = (0_u8..30).map(|value| vec![value; 20]).collect::<Vec<_>>();
    let names = (0..30)
        .map(|index| format!("tensor-{index:02}"))
        .collect::<Vec<_>>();
    let parent_rows = names
        .iter()
        .zip(&payloads)
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect::<Vec<_>>();
    let child_rows = parent_rows.iter().copied().skip(1).collect::<Vec<_>>();

    let parent = tensor_digests(&hashed(&safetensors(&parent_rows)));
    let child = tensor_digests(&hashed(&safetensors(&child_rows)));

    assert_eq!(parent.len(), 30);
    assert_eq!(child.len(), 29);
    assert_eq!(parent[1..], child);
}

#[test]
fn actual_serialized_bytes_not_training_declarations_determine_reuse() {
    let attention = b"attention".as_slice();
    let mlp = b"mlp-weights".as_slice();
    let norm = b"norm".as_slice();
    let bias = b"bias".as_slice();
    let base = hashed(&safetensors(&[
        ("block.attention", attention),
        ("block.mlp", mlp),
        ("block.norm", norm),
        ("block.bias", bias),
    ]));

    // A selected or trainable tensor that serializes byte-identically changes
    // neither its object nor the header.
    let selected_but_identical = hashed(&safetensors(&[
        ("block.attention", attention),
        ("block.mlp", mlp),
        ("block.norm", norm),
        ("block.bias", bias),
    ]));
    assert!(changed_object_indexes(&base, &selected_but_identical).is_empty());

    let changed_bias = b"BIAS".as_slice();
    let bias_only = hashed(&safetensors(&[
        ("block.attention", attention),
        ("block.mlp", mlp),
        ("block.norm", norm),
        ("block.bias", changed_bias),
    ]));
    assert_eq!(
        changed_object_indexes(&base, &bias_only),
        HashSet::from([4])
    );

    let dense = hashed(&safetensors(&[
        ("block.attention", b"ATTENTION"),
        ("block.mlp", b"MLP-WEIGHTS"),
        ("block.norm", b"NORM"),
        ("block.bias", b"BIAS"),
    ]));
    assert_eq!(
        changed_object_indexes(&base, &dense),
        HashSet::from([1, 2, 3, 4])
    );
}

#[test]
fn tensor_rename_changes_only_the_header_identity() {
    let payload = b"shared tensor bytes".as_slice();
    let before = hashed(&safetensors(&[("old.name", payload)]));
    let after = hashed(&safetensors(&[("new.name", payload)]));

    assert_eq!(changed_object_indexes(&before, &after), HashSet::from([0]));
}

struct PatternTensorSource {
    prefix_and_header: Vec<u8>,
    tensor_length: u64,
    mutations: Vec<(u64, u8)>,
}

impl PatternTensorSource {
    fn new(tensor_length: u64, mutations: Vec<(u64, u8)>) -> Self {
        let header = format!(
            r#"{{"weight":{{"dtype":"U8","shape":[{tensor_length}],"data_offsets":[0,{tensor_length}]}}}}"#
        );
        let mut prefix_and_header = Vec::with_capacity(8 + header.len());
        prefix_and_header.extend_from_slice(&(header.len() as u64).to_le_bytes());
        prefix_and_header.extend_from_slice(header.as_bytes());
        Self {
            prefix_and_header,
            tensor_length,
            mutations,
        }
    }
}

impl ByteSource for PatternTensorSource {
    fn len(&self) -> u64 {
        self.prefix_and_header.len() as u64 + self.tensor_length
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let header_end = self.prefix_and_header.len() as u64;
        let end = offset
            .checked_add(destination.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.len() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        if offset < header_end {
            if end > header_end {
                return Err(io::Error::other("test read unexpectedly crossed header"));
            }
            let start = offset as usize;
            destination.copy_from_slice(&self.prefix_and_header[start..start + destination.len()]);
            return Ok(());
        }

        destination.fill(0x5a);
        let tensor_start = offset - header_end;
        let tensor_end = tensor_start + destination.len() as u64;
        for (mutation_offset, value) in &self.mutations {
            if (tensor_start..tensor_end).contains(mutation_offset) {
                destination[(mutation_offset - tensor_start) as usize] = *value;
            }
        }
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

fn hash_pattern(source: &PatternTensorSource) -> HashedPlan {
    plan_and_hash(source).expect("pattern source plans and hashes")
}

#[test]
fn localized_and_boundary_crossing_updates_replace_exactly_intersected_objects() {
    let tensor_length = MAX_OBJECT_SIZE + 1;
    let base = hash_pattern(&PatternTensorSource::new(tensor_length, vec![]));
    let localized = hash_pattern(&PatternTensorSource::new(
        tensor_length,
        vec![(MAX_OBJECT_SIZE, 0xa5)],
    ));
    let boundary_crossing = hash_pattern(&PatternTensorSource::new(
        tensor_length,
        vec![(MAX_OBJECT_SIZE - 1, 0xa5), (MAX_OBJECT_SIZE, 0xa5)],
    ));

    assert_eq!(
        changed_object_indexes(&base, &localized),
        HashSet::from([2])
    );
    assert_eq!(
        changed_object_indexes(&base, &boundary_crossing),
        HashSet::from([1, 2])
    );
}

#[test]
fn eight_object_tensor_uses_tensor_relative_64_mib_boundaries() {
    let tensor_length = 7 * MAX_OBJECT_SIZE + 50_000_000;
    let source = PatternTensorSource::new(tensor_length, vec![]);
    let planned = plan(&source).expect("large tensor plans");
    let tensor_regions = planned
        .regions()
        .iter()
        .filter(|region| region.kind() == RegionKind::Tensor)
        .collect::<Vec<_>>();

    assert_eq!(tensor_regions.len(), 8);
    assert!(
        tensor_regions[..7]
            .iter()
            .all(|region| region.length() == MAX_OBJECT_SIZE)
    );
    assert_eq!(tensor_regions[7].length(), 50_000_000);
}

#[test]
fn minimax_h3_adaln_accounting_matches_the_pinned_launch_shape() {
    const FULL_TENSORS: u64 = 638;
    const FULL_BYTES: u64 = 66_280_430_080;
    const FULL_REFS: u64 = 1_508;
    const ADALN_WEIGHTS: u64 = 50;
    const ADALN_WEIGHT_BYTES: u64 = 520_224_768;
    const ADALN_BIASES: u64 = 50;
    const ADALN_BIAS_BYTES: u64 = 193_536;

    let adaln_bytes = ADALN_WEIGHTS * ADALN_WEIGHT_BYTES + ADALN_BIASES * ADALN_BIAS_BYTES;
    let weight_refs = ADALN_WEIGHT_BYTES.div_ceil(MAX_OBJECT_SIZE);
    let removed_refs = ADALN_WEIGHTS * weight_refs + ADALN_BIASES;

    assert_eq!(adaln_bytes, 26_020_915_200);
    assert_eq!(weight_refs, 8);
    assert_eq!(FULL_TENSORS - ADALN_WEIGHTS - ADALN_BIASES, 538);
    assert_eq!(FULL_BYTES - adaln_bytes, 40_259_514_880);
    assert_eq!(FULL_REFS - removed_refs, 1_058);
}
