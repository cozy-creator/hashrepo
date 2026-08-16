use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::{MAX_OBJECT_SIZE, PlannerId};
use tensorfs_core::tfm1::{
    Entry, FileRecord, MAX_FILE_RECORDS, Snapshot, SnapshotBuilder, SnapshotId, Tfm1Error, decode,
};

fn digest(seed: u8) -> ObjectDigest {
    ObjectDigest::from_bytes([seed; 32])
}

fn data(seed: u8, length: u64) -> FileRecord {
    FileRecord::Data {
        digest: digest(seed),
        length,
    }
}

fn small_file(builder: &mut SnapshotBuilder, path: &str, seed: u8) {
    builder.file(path, false, PlannerId::BlobV1, vec![data(seed, 8)]);
}

/// Assembles manifest bytes by hand, mirroring the documented grammar. This
/// keeps the wire format pinned independently of the implementation.
#[derive(Default)]
struct Raw(Vec<u8>);

impl Raw {
    fn magic(mut self) -> Self {
        self.0.extend_from_slice(b"TFM1");
        self
    }

    fn no_parent(mut self) -> Self {
        self.0.push(0);
        self
    }

    fn entry_count(mut self, count: u64) -> Self {
        self.0.extend_from_slice(&count.to_le_bytes());
        self
    }

    fn path(mut self, path: &str) -> Self {
        self.0.extend_from_slice(&(path.len() as u32).to_le_bytes());
        self.0.extend_from_slice(path.as_bytes());
        self
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self.0.extend_from_slice(bytes);
        self
    }

    fn u64(mut self, value: u64) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }
}

fn decoded(builder: SnapshotBuilder) -> Snapshot {
    builder.finish().expect("fixture tree is valid")
}

fn reason(error: Tfm1Error) -> &'static str {
    error.reason()
}

#[test]
fn the_empty_snapshot_has_one_canonical_encoding() {
    let empty = decoded(SnapshotBuilder::new(None));
    let raw = Raw::default().magic().no_parent().entry_count(0);

    assert_eq!(empty.to_bytes(), raw.0);
    assert_eq!(decode(&empty.to_bytes()).unwrap(), empty);
}

#[test]
fn the_hand_assembled_grammar_matches_the_builder_byte_for_byte() {
    let mut builder = SnapshotBuilder::new(None);
    builder.file(
        "model.safetensors",
        true,
        PlannerId::SafetensorsV1,
        vec![data(0xAA, 10), FileRecord::Hole { length: 5 }],
    );
    builder.file(
        "video.webm",
        false,
        PlannerId::BlobV1,
        vec![data(0xBB, 5 * 1024 * 1024 * 1024)],
    );
    let snapshot = decoded(builder);

    let raw = Raw::default()
        .magic()
        .no_parent()
        .entry_count(2)
        .path("model.safetensors")
        .bytes(&[2, 1, 1]) // file, executable, safetensors-v1
        .u64(15) // logical size
        .u64(2) // record count
        .bytes(&[1]) // data tag
        .bytes(&[0xAA; 32])
        .u64(10)
        .bytes(&[2]) // hole tag
        .u64(5)
        .path("video.webm")
        .bytes(&[2, 0, 4]) // file, not executable, blob-v1
        .u64(5 * 1024 * 1024 * 1024) // logical size: far beyond one record
        .bytes(&[0xBB; 32]); // the whole-file digest; no record list exists

    assert_eq!(snapshot.to_bytes(), raw.0);
}

#[test]
fn identical_trees_built_separately_produce_the_same_id() {
    let build = || {
        let mut builder = SnapshotBuilder::new(None);
        builder.directory("weights");
        small_file(&mut builder, "weights/a.bin", 1);
        builder.symlink("latest", "weights/a.bin");
        decoded(builder)
    };

    assert_eq!(build().to_bytes(), build().to_bytes());
    assert_eq!(build().snapshot_id(), build().snapshot_id());
}

#[test]
fn a_changed_parent_is_a_different_history_even_with_equal_tree_bytes() {
    let tree = |parent: Option<SnapshotId>| {
        let mut builder = SnapshotBuilder::new(parent);
        small_file(&mut builder, "a", 1);
        decoded(builder)
    };

    let orphan = tree(None);
    let child = tree(Some(SnapshotId::of(b"other history")));
    assert_eq!(orphan.entries(), child.entries());
    assert_ne!(orphan.snapshot_id(), child.snapshot_id());
    assert_eq!(
        decode(&child.to_bytes()).unwrap().parent(),
        child.parent(),
        "the explicit parent survives decoding"
    );
}

#[test]
fn renaming_a_file_changes_identity_but_never_the_referenced_digests() {
    let with_name = |name: &str| {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            name,
            false,
            PlannerId::SafetensorsV1,
            vec![data(7, 100), data(8, 200)],
        );
        decoded(builder)
    };

    let before = with_name("original.bin");
    let after = with_name("renamed.bin");
    assert_ne!(before.snapshot_id(), after.snapshot_id());

    let digests = |snapshot: &Snapshot| {
        snapshot
            .entries()
            .iter()
            .filter_map(|(_, entry)| match entry {
                Entry::File { body, .. } => Some(body.records().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(digests(&before), digests(&after));
}

#[test]
fn hardlink_groups_reroot_to_the_first_sorted_path() {
    let mut builder = SnapshotBuilder::new(None);
    small_file(&mut builder, "z-target", 3);
    builder.hardlink("a-link", "z-target");
    let snapshot = decoded(builder);

    let entries: Vec<(&str, &Entry)> = snapshot
        .entries()
        .iter()
        .map(|(path, entry)| (path.as_str(), entry))
        .collect();
    assert!(matches!(entries[0], ("a-link", Entry::File { .. })));
    assert!(matches!(
        entries[1],
        ("z-target", Entry::Hardlink { ordinal: 0 })
    ));
    assert_eq!(snapshot.ordinal_path(0), Some("a-link"));
}

#[test]
fn redirecting_a_hardlink_changes_the_snapshot_id() {
    let linked_to = |target: &str| {
        let mut builder = SnapshotBuilder::new(None);
        small_file(&mut builder, "file-one", 1);
        small_file(&mut builder, "file-two", 2);
        builder.hardlink("zz-link", target);
        decoded(builder)
    };

    assert_ne!(
        linked_to("file-one").snapshot_id(),
        linked_to("file-two").snapshot_id()
    );
}

#[test]
fn hardlinks_only_target_regular_files_and_never_chain() {
    let mut to_directory = SnapshotBuilder::new(None);
    to_directory.directory("d");
    to_directory.hardlink("link", "d");
    assert!(matches!(
        to_directory.finish(),
        Err(Tfm1Error::HardlinkTarget)
    ));

    let mut chained = SnapshotBuilder::new(None);
    small_file(&mut chained, "file", 1);
    chained.hardlink("link-one", "file");
    chained.hardlink("link-two", "link-one");
    assert!(matches!(chained.finish(), Err(Tfm1Error::HardlinkTarget)));

    let mut dangling = SnapshotBuilder::new(None);
    dangling.hardlink("link", "missing");
    assert!(matches!(dangling.finish(), Err(Tfm1Error::HardlinkTarget)));
}

#[test]
fn every_path_rule_refuses_at_the_builder() {
    let refusals: &[(&str, &str)] = &[
        ("/absolute", "path-absolute"),
        ("a//b", "path-empty-segment"),
        ("a/", "path-empty-segment"),
        ("a/./b", "path-dot-segment"),
        ("a/../b", "path-dot-segment"),
        ("back\\slash", "path-forbidden-character"),
        ("colon:name", "path-forbidden-character"),
        ("quest?ion", "path-forbidden-character"),
        ("control\u{7}", "path-forbidden-character"),
        ("trailing.", "path-trailing-dot-space"),
        ("trailing ", "path-trailing-dot-space"),
        ("CON", "path-windows-reserved"),
        ("nested/lpt1.txt", "path-windows-reserved"),
        ("com9.tar.gz", "path-windows-reserved"),
        ("cafe\u{301}", "path-not-nfc"),
    ];
    for (path, expected) in refusals {
        let mut builder = SnapshotBuilder::new(None);
        builder.directory(*path);
        let error = builder.finish().expect_err(path);
        assert_eq!(reason(error), *expected, "path {path:?}");
    }

    let mut too_long = SnapshotBuilder::new(None);
    too_long.directory("a".repeat(4097));
    assert_eq!(reason(too_long.finish().unwrap_err()), "path-length");

    let mut empty = SnapshotBuilder::new(None);
    empty.directory("");
    assert_eq!(reason(empty.finish().unwrap_err()), "path-length");
}

#[test]
fn case_fold_collisions_and_missing_parents_refuse() {
    let mut collision = SnapshotBuilder::new(None);
    collision.directory("Weights");
    collision.directory("weights");
    assert_eq!(
        reason(collision.finish().unwrap_err()),
        "case-fold-collision"
    );

    let mut orphan = SnapshotBuilder::new(None);
    small_file(&mut orphan, "missing/child.bin", 1);
    assert_eq!(
        reason(orphan.finish().unwrap_err()),
        "missing-parent-directory"
    );

    let mut through_symlink = SnapshotBuilder::new(None);
    through_symlink.symlink("indirect", "elsewhere");
    small_file(&mut through_symlink, "indirect/child.bin", 1);
    assert_eq!(
        reason(through_symlink.finish().unwrap_err()),
        "missing-parent-directory"
    );

    let mut duplicate = SnapshotBuilder::new(None);
    duplicate.directory("twice");
    duplicate.directory("twice");
    assert_eq!(reason(duplicate.finish().unwrap_err()), "duplicate-path");
}

#[test]
fn record_rules_refuse_at_the_builder() {
    let refuse = |records: Vec<FileRecord>, expected: &str| {
        let mut builder = SnapshotBuilder::new(None);
        builder.file("f", false, PlannerId::SafetensorsV1, records);
        assert_eq!(reason(builder.finish().unwrap_err()), expected);
    };

    refuse(vec![data(1, 0)], "zero-length-record");
    refuse(vec![FileRecord::Hole { length: 0 }], "zero-length-record");
    refuse(
        vec![
            FileRecord::Hole { length: 1 },
            FileRecord::Hole { length: 1 },
        ],
        "adjacent-holes",
    );
    refuse(vec![data(1, MAX_OBJECT_SIZE + 1)], "data-too-large");
    refuse(
        vec![
            FileRecord::Hole { length: u64::MAX },
            data(1, MAX_OBJECT_SIZE),
        ],
        "length-sum-mismatch",
    );

    let mut over_limit = SnapshotBuilder::new(None);
    over_limit.file(
        "f",
        false,
        PlannerId::SafetensorsV1,
        (0..=MAX_FILE_RECORDS)
            .map(|_| data(1, 1))
            .collect::<Vec<_>>(),
    );
    assert_eq!(reason(over_limit.finish().unwrap_err()), "record-limit");
}

#[test]
fn a_blob_body_is_exactly_one_whole_file_object_or_empty() {
    // The canonical forms round-trip…
    let mut canonical = SnapshotBuilder::new(None);
    canonical.file("big.bin", false, PlannerId::BlobV1, vec![data(9, u64::MAX)]);
    canonical.file("empty.bin", false, PlannerId::BlobV1, Vec::new());
    let snapshot = decoded(canonical);
    match &snapshot.entries()[1].1 {
        Entry::File { body, .. } => {
            assert_eq!(body.planner_id(), PlannerId::BlobV1);
            assert_eq!(body.logical_size(), 0);
            assert_eq!(
                *body,
                tensorfs_core::tfm1::FileBody::Blob {
                    logical_size: 0,
                    digest: tensorfs_core::tfm1::EMPTY_BLOB_DIGEST,
                }
            );
        }
        _ => panic!("expected a file entry"),
    }

    // …and every other record shape refuses: a chunked blob and a hole in a
    // blob are unrepresentable, so the staging vocabulary cannot smuggle one
    // through the builder.
    let refuse = |records: Vec<FileRecord>| {
        let mut builder = SnapshotBuilder::new(None);
        builder.file("f", false, PlannerId::BlobV1, records);
        assert_eq!(reason(builder.finish().unwrap_err()), "blob-records");
    };
    refuse(vec![data(1, 4), data(2, 4)]);
    refuse(vec![FileRecord::Hole { length: 4 }]);
    refuse(vec![data(1, 4), FileRecord::Hole { length: 4 }]);
}

#[test]
fn symlink_target_rules_refuse() {
    for target in ["", "control\u{1}char", &"t".repeat(4097)] {
        let mut builder = SnapshotBuilder::new(None);
        builder.symlink("link", target);
        assert_eq!(reason(builder.finish().unwrap_err()), "symlink-target");
    }
}

#[test]
fn an_empty_file_and_an_all_hole_file_are_both_canonical() {
    let mut builder = SnapshotBuilder::new(None);
    builder.file("empty", false, PlannerId::BlobV1, Vec::new());
    builder.file(
        "sparse.safetensors",
        false,
        PlannerId::SafetensorsV1,
        vec![FileRecord::Hole {
            length: 10 * MAX_OBJECT_SIZE,
        }],
    );
    let snapshot = decoded(builder);
    assert_eq!(snapshot, decode(&snapshot.to_bytes()).unwrap());
}

#[test]
fn decode_refuses_structural_corruption_at_every_cut() {
    let valid = {
        let mut builder = SnapshotBuilder::new(None);
        small_file(&mut builder, "a", 1);
        decoded(builder).to_bytes()
    };

    for cut in 0..valid.len() {
        assert!(
            decode(&valid[..cut]).is_err(),
            "a prefix cut at {cut} must refuse"
        );
    }

    let mut trailing = valid.clone();
    trailing.push(0);
    assert_eq!(reason(decode(&trailing).unwrap_err()), "trailing-bytes");

    let mut magic = valid.clone();
    magic[0] = b'X';
    assert_eq!(reason(decode(&magic).unwrap_err()), "bad-magic");

    let mut flag = valid;
    flag[4] = 2;
    assert_eq!(reason(decode(&flag).unwrap_err()), "parent-flag");
}

#[test]
fn decode_refuses_disorder_duplicates_and_forward_hardlinks() {
    let two = |first: &str, second: &str| {
        Raw::default()
            .magic()
            .no_parent()
            .entry_count(2)
            .path(first)
            .bytes(&[1])
            .path(second)
            .bytes(&[1])
            .0
    };
    assert_eq!(reason(decode(&two("b", "a")).unwrap_err()), "entry-order");
    assert_eq!(
        reason(decode(&two("a", "a")).unwrap_err()),
        "duplicate-path"
    );

    let forward_link = Raw::default()
        .magic()
        .no_parent()
        .entry_count(1)
        .path("link")
        .bytes(&[4])
        .u64(0)
        .0;
    assert_eq!(
        reason(decode(&forward_link).unwrap_err()),
        "hardlink-ordinal"
    );
}

#[test]
fn decode_bounds_declared_counts_before_allocating() {
    let absurd_entries = Raw::default().magic().no_parent().entry_count(u64::MAX).0;
    assert_eq!(
        reason(decode(&absurd_entries).unwrap_err()),
        "count-exceeds-input"
    );

    let absurd_records = Raw::default()
        .magic()
        .no_parent()
        .entry_count(1)
        .path("f")
        .bytes(&[2, 0, 1]) // file, not executable, safetensors-v1
        .u64(0)
        .u64(u64::MAX)
        .0;
    assert_eq!(reason(decode(&absurd_records).unwrap_err()), "record-limit");

    let claimed_but_missing = Raw::default()
        .magic()
        .no_parent()
        .entry_count(1)
        .path("f")
        .bytes(&[2, 0, 1])
        .u64(9)
        .u64(1)
        .0;
    assert_eq!(
        reason(decode(&claimed_but_missing).unwrap_err()),
        "count-exceeds-input"
    );
}

#[test]
fn decode_refuses_unknown_tags_and_invalid_flags() {
    let entry_kind = Raw::default()
        .magic()
        .no_parent()
        .entry_count(1)
        .path("x")
        .bytes(&[9])
        .0;
    assert_eq!(
        reason(decode(&entry_kind).unwrap_err()),
        "unknown-entry-kind"
    );

    let file = |body: &[u8]| {
        Raw::default()
            .magic()
            .no_parent()
            .entry_count(1)
            .path("f")
            .bytes(body)
            .0
    };
    assert_eq!(
        reason(
            decode(&file(&[
                2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]))
            .unwrap_err()
        ),
        "executable-flag"
    );
    assert_eq!(
        reason(
            decode(&file(&[
                2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]))
            .unwrap_err()
        ),
        "unknown-planner"
    );

    let record_tag = Raw::default()
        .magic()
        .no_parent()
        .entry_count(1)
        .path("f")
        .bytes(&[2, 0, 1])
        .u64(1)
        .u64(1)
        .bytes(&[9])
        .u64(1)
        .0;
    assert_eq!(
        reason(decode(&record_tag).unwrap_err()),
        "unknown-record-tag"
    );
}

#[test]
fn the_bounded_record_cardinality_is_exactly_the_planner_limit() {
    let mut builder = SnapshotBuilder::new(None);
    builder.file(
        "many",
        false,
        PlannerId::SafetensorsV1,
        (0..MAX_FILE_RECORDS).map(|_| data(1, 1)).collect(),
    );
    let snapshot = builder.finish().expect("the exact limit is valid");
    let round_trip = decode(&snapshot.to_bytes()).expect("the exact limit decodes");
    assert_eq!(round_trip, snapshot);
}
