use std::collections::HashSet;
use std::io;

use hashrepo_core::object::{HashedPlan, plan_and_hash};
use hashrepo_core::planner::{ByteSource, MAX_OBJECT_SIZE, PlannerId, RegionKind, plan};

const MIB: u64 = 1024 * 1024;

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

fn tensor_digest_set(hashed: &HashedPlan) -> HashSet<String> {
    tensor_digests(hashed).into_iter().collect()
}

fn header_digests(hashed: &HashedPlan) -> Vec<String> {
    hashed
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Header)
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
fn masked_or_selected_updates_only_change_reuse_when_serialized_bytes_change() {
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

    // A selected tensor whose optimizer update is masked, or otherwise
    // serializes byte-identically, changes neither its object nor the header.
    let masked_or_selected_but_identical = hashed(&safetensors(&[
        ("block.attention", attention),
        ("block.mlp", mlp),
        ("block.norm", norm),
        ("block.bias", bias),
    ]));
    assert_eq!(base, masked_or_selected_but_identical);

    let changed_bias = b"bIas".as_slice();
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
fn standalone_composite_and_merged_lora_follow_serialized_tensor_semantics() {
    let base_weight = b"base-weight-v1".as_slice();
    let merged_weight = b"base-weight-v2".as_slice();
    let norm = b"shared-norm".as_slice();
    let lora_a = b"low-rank-a".as_slice();
    let lora_b = b"low-rank-b".as_slice();

    let standalone = hashed(&safetensors(&[
        ("block.lora_A.weight", lora_a),
        ("block.lora_B.weight", lora_b),
    ]));
    let composite = hashed(&safetensors(&[
        ("block.weight", base_weight),
        ("block.norm", norm),
        ("block.lora_A.weight", lora_a),
        ("block.lora_B.weight", lora_b),
    ]));
    let merged = hashed(&safetensors(&[
        ("block.weight", merged_weight),
        ("block.norm", norm),
    ]));

    let standalone_tensors = tensor_digests(&standalone);
    let composite_tensors = tensor_digests(&composite);
    let merged_tensors = tensor_digests(&merged);

    // A composite file stores the adapter tensors verbatim, so it reuses the
    // standalone adapter objects even though the file header is different.
    assert_eq!(standalone_tensors, composite_tensors[2..]);
    // Merging changes the serialized base weight. HashRepo does not infer that
    // the new weight was produced from the LoRA delta, but still reuses norm.
    assert_ne!(merged_tensors[0], composite_tensors[0]);
    assert_eq!(merged_tensors[1], composite_tensors[1]);
    assert!(tensor_digest_set(&standalone).is_disjoint(&tensor_digest_set(&merged)));
}

#[test]
fn norm_only_and_embedding_only_updates_replace_only_the_named_tensor_objects() {
    let attention = b"attention-v1".as_slice();
    let base = hashed(&safetensors(&[
        ("model.embed_tokens.weight", b"tok0tok1tok2"),
        ("model.layers.0.attention.weight", attention),
        ("model.layers.0.norm.weight", b"norm-v1!"),
    ]));
    let norm_only = hashed(&safetensors(&[
        ("model.embed_tokens.weight", b"tok0tok1tok2"),
        ("model.layers.0.attention.weight", attention),
        ("model.layers.0.norm.weight", b"NORM-v2!"),
    ]));
    let embedding_only = hashed(&safetensors(&[
        ("model.embed_tokens.weight", b"tok0TOK1tok2"),
        ("model.layers.0.attention.weight", attention),
        ("model.layers.0.norm.weight", b"norm-v1!"),
    ]));

    assert_eq!(
        changed_object_indexes(&base, &norm_only),
        HashSet::from([3])
    );
    assert_eq!(
        changed_object_indexes(&base, &embedding_only),
        HashSet::from([1])
    );
}

#[test]
fn expert_per_tensor_moe_updates_replace_only_the_changed_expert() {
    let base = hashed(&safetensors(&[
        ("model.gate.weight", b"router-v1"),
        ("model.experts.0.weight", b"expert-000"),
        ("model.experts.1.weight", b"expert-111"),
        ("model.experts.2.weight", b"expert-222"),
    ]));
    let expert_one_only = hashed(&safetensors(&[
        ("model.gate.weight", b"router-v1"),
        ("model.experts.0.weight", b"expert-000"),
        ("model.experts.1.weight", b"EXPERT-111"),
        ("model.experts.2.weight", b"expert-222"),
    ]));

    assert_eq!(
        changed_object_indexes(&base, &expert_one_only),
        HashSet::from([3])
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
fn tensor_chunking_covers_every_64_mib_boundary_length() {
    let cases = [
        (0, Vec::new()),
        (1, vec![1]),
        (MAX_OBJECT_SIZE - 1, vec![MAX_OBJECT_SIZE - 1]),
        (MAX_OBJECT_SIZE, vec![MAX_OBJECT_SIZE]),
        (MAX_OBJECT_SIZE + 1, vec![MAX_OBJECT_SIZE, 1]),
    ];

    for (tensor_length, expected_lengths) in cases {
        let source = PatternTensorSource::new(tensor_length, vec![]);
        let planned = plan(&source).expect("boundary fixture plans");
        let actual_lengths = planned
            .regions()
            .iter()
            .filter(|region| region.kind() == RegionKind::Tensor)
            .map(|region| region.length())
            .collect::<Vec<_>>();

        assert_eq!(planned.planner(), PlannerId::SafetensorsV1);
        assert_eq!(
            actual_lengths, expected_lengths,
            "tensor length {tensor_length}"
        );
    }
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
fn one_byte_mutation_inside_an_eight_object_tensor_replaces_one_object() {
    let tensor_length = 7 * MAX_OBJECT_SIZE + 1;
    let mutation_offset = 3 * MAX_OBJECT_SIZE + 17;
    let changed = hash_pattern(&PatternTensorSource::new(
        tensor_length,
        vec![(mutation_offset, 0xa5)],
    ));
    let tensor_objects = changed
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .collect::<Vec<_>>();

    assert_eq!(tensor_objects.len(), 8);
    assert!(
        tensor_objects[..7]
            .iter()
            .all(|object| object.length() == MAX_OBJECT_SIZE)
    );
    assert_eq!(tensor_objects[7].length(), 1);

    // Every unmodified full-size object contains the same generated bytes and
    // therefore has the same digest. Only the object containing the one-byte
    // mutation loses that reusable identity; the eighth tail remains separate.
    let reusable_digest = tensor_objects[0].digest();
    for index in [0, 1, 2, 4, 5, 6] {
        assert_eq!(tensor_objects[index].digest(), reusable_digest);
    }
    assert_ne!(tensor_objects[3].digest(), reusable_digest);
}

#[test]
fn packed_moe_update_replaces_every_64_mib_object_intersecting_the_expert() {
    const EXPERT_BYTES: u64 = 40 * MIB;
    let tensor_length = 3 * EXPERT_BYTES;
    let base = hash_pattern(&PatternTensorSource::new(tensor_length, vec![]));

    // Packed expert 1 occupies [40 MiB, 80 MiB), crossing the 64 MiB object
    // boundary. Representative changed bytes on both sides replace both and
    // only those intersecting objects.
    let expert_one = hash_pattern(&PatternTensorSource::new(
        tensor_length,
        vec![(EXPERT_BYTES, 0xa5), (2 * EXPERT_BYTES - 1, 0xa5)],
    ));
    let tensor_lengths = base
        .objects()
        .iter()
        .filter(|object| object.kind() == RegionKind::Tensor)
        .map(|object| object.length())
        .collect::<Vec<_>>();

    assert_eq!(tensor_lengths, vec![MAX_OBJECT_SIZE, 56 * MIB]);
    assert_eq!(
        changed_object_indexes(&base, &expert_one),
        HashSet::from([1, 2])
    );
}

#[test]
fn cross_shard_updates_and_tensor_moves_preserve_unaffected_objects() {
    let shard_zero = hashed(&safetensors(&[
        ("model.layers.0.weight", b"layer-zero"),
        ("model.layers.1.weight", b"layer-one!"),
    ]));
    let shard_one = hashed(&safetensors(&[
        ("model.layers.2.weight", b"layer-two!"),
        ("model.norm.weight", b"final-norm"),
    ]));

    let revised_shard_zero = hashed(&safetensors(&[
        ("model.layers.0.weight", b"layer-zero"),
        ("model.layers.1.weight", b"LAYER-ONE!"),
    ]));
    let unchanged_shard_one = hashed(&safetensors(&[
        ("model.layers.2.weight", b"layer-two!"),
        ("model.norm.weight", b"final-norm"),
    ]));

    assert_eq!(
        changed_object_indexes(&shard_zero, &revised_shard_zero),
        HashSet::from([2])
    );
    assert_eq!(shard_one, unchanged_shard_one);

    let moved_left = hashed(&safetensors(&[("model.layers.0.weight", b"layer-zero")]));
    let moved_right = hashed(&safetensors(&[
        ("model.layers.1.weight", b"layer-one!"),
        ("model.layers.2.weight", b"layer-two!"),
        ("model.norm.weight", b"final-norm"),
    ]));
    let before_move = tensor_digest_set(&shard_zero)
        .union(&tensor_digest_set(&shard_one))
        .cloned()
        .collect::<HashSet<_>>();
    let after_move = tensor_digest_set(&moved_left)
        .union(&tensor_digest_set(&moved_right))
        .cloned()
        .collect::<HashSet<_>>();

    assert_eq!(before_move, after_move);
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

/// The negative control for every reuse arm above. Full dense fine-tuning
/// changes essentially every parameter tensor, so the honest answer is zero
/// tensor-object reuse. An unchanged architecture — identical tensor names,
/// order and shapes, hence a byte-identical header — must not be able to
/// manufacture a reuse claim.
#[test]
fn full_dense_fine_tuning_reuses_no_tensor_object_despite_an_identical_architecture() {
    let names = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
    ];
    let lengths = [4_096_usize, 4_096, 8_192, 8_192, 256];

    let base_bytes: Vec<Vec<u8>> = lengths
        .iter()
        .enumerate()
        .map(|(index, length)| vec![0x10 + index as u8; *length])
        .collect();
    let tuned_bytes: Vec<Vec<u8>> = lengths
        .iter()
        .enumerate()
        .map(|(index, length)| vec![0xA0 + index as u8; *length])
        .collect();

    let base_rows: Vec<(&str, &[u8])> = names
        .iter()
        .zip(&base_bytes)
        .map(|(name, bytes)| (*name, bytes.as_slice()))
        .collect();
    let tuned_rows: Vec<(&str, &[u8])> = names
        .iter()
        .zip(&tuned_bytes)
        .map(|(name, bytes)| (*name, bytes.as_slice()))
        .collect();

    let base = safetensors(&base_rows);
    let tuned = safetensors(&tuned_rows);

    let before = hashed(&base);
    let after = hashed(&tuned);

    // Same names, same order, same shapes: the architecture is byte-identical.
    assert_eq!(header_digests(&before), header_digests(&after));

    // Every serialized tensor byte changed, so nothing may be reused. Matching
    // names, shapes and architecture cannot rescue a single object.
    assert_eq!(tensor_digests(&before).len(), names.len());
    assert_eq!(tensor_digests(&after).len(), names.len());
    assert!(tensor_digest_set(&before).is_disjoint(&tensor_digest_set(&after)));
    assert_eq!(
        changed_object_indexes(&before, &after).len(),
        names.len(),
        "every tensor object must differ under a full dense update"
    );

    // The control proves the assertion above is not vacuous: this harness does
    // observe reuse when the serialized bytes genuinely repeat.
    assert_eq!(
        tensor_digest_set(&before),
        tensor_digest_set(&hashed(&base))
    );
}
