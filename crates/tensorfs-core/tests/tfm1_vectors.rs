//! The language-neutral TFM1 conformance corpus.
//!
//! Golden fixtures are the builder's canonical bytes; refusal fixtures are
//! hand-assembled deviations. Setting `TENSORFS_WRITE_TFM1_VECTORS=1`
//! regenerates the committed corpus in place before asserting it, which is
//! only legitimate while the pre-launch format may still be replaced.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::{MAX_OBJECT_SIZE, PlannerId};
use tensorfs_core::tfm1::{FileRecord, SnapshotBuilder, SnapshotId, decode};

const CORPUS_DIR: &str = "../../spec/v1/tfm1-vectors";
const REGEN_ENV: &str = "TENSORFS_WRITE_TFM1_VECTORS";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    #[serde(rename = "$schema")]
    schema: String,
    format: u32,
    hash: String,
    max_object_size: u64,
    fixture_encoding: String,
    golden: Vec<Golden>,
    refusals: Vec<Refusal>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Golden {
    name: String,
    fixture: String,
    snapshot_id: String,
    description: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Refusal {
    name: String,
    fixture: String,
    reason: String,
}

fn digest(seed: u8) -> ObjectDigest {
    ObjectDigest::from_bytes([seed; 32])
}

fn data(seed: u8, length: u64) -> FileRecord {
    FileRecord::Data {
        digest: digest(seed),
        length,
    }
}

fn golden_bytes() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    let empty = SnapshotBuilder::new(None)
        .finish()
        .expect("the empty snapshot is valid")
        .to_bytes();
    let empty_id = SnapshotId::of(&empty);

    let single_file = {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            "model.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![data(0x11, 18)],
        );
        builder.finish().expect("valid").to_bytes()
    };

    let tree = {
        let mut builder = SnapshotBuilder::new(None);
        builder.directory("weights");
        builder.directory("weights/text-encoder");
        builder.file(
            "weights/model.safetensors",
            false,
            PlannerId::SafetensorsV1,
            vec![data(0x21, 96), data(0x22, 1024)],
        );
        builder.file(
            "weights/text-encoder/encoder.safetensors",
            false,
            PlannerId::SafetensorsV1,
            vec![data(0x23, 64), data(0x24, 512)],
        );
        builder.file(
            "run.sh",
            true,
            PlannerId::RawFixed64mV1,
            vec![data(0x25, 5)],
        );
        builder.finish().expect("valid").to_bytes()
    };

    let sparse = {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            "sparse.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![
                FileRecord::Hole { length: 4096 },
                data(0x31, 100),
                FileRecord::Hole { length: 1 << 40 },
            ],
        );
        builder.finish().expect("valid").to_bytes()
    };

    let hardlinks = {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            "z-original.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![data(0x41, 33)],
        );
        builder.hardlink("m-alias.bin", "z-original.bin");
        builder.hardlink("a-alias.bin", "z-original.bin");
        builder.file(
            "b-solo.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![data(0x42, 7)],
        );
        builder.finish().expect("valid").to_bytes()
    };

    let symlink = {
        let mut builder = SnapshotBuilder::new(None);
        builder.directory("snapshots");
        builder.file(
            "snapshots/current.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![data(0x51, 9)],
        );
        builder.symlink("latest", "snapshots/current.bin");
        builder.finish().expect("valid").to_bytes()
    };

    let empty_directory = {
        let mut builder = SnapshotBuilder::new(None);
        builder.directory("empty");
        builder.finish().expect("valid").to_bytes()
    };

    let boundary = {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            "boundary.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![
                data(0x61, 1),
                data(0x62, MAX_OBJECT_SIZE - 1),
                data(0x63, MAX_OBJECT_SIZE),
                FileRecord::Hole { length: 1 },
            ],
        );
        builder.file("zero.bin", false, PlannerId::RawFixed64mV1, Vec::new());
        builder.finish().expect("valid").to_bytes()
    };

    let with_parent = {
        let mut builder = SnapshotBuilder::new(Some(empty_id));
        builder.file(
            "child.bin",
            false,
            PlannerId::RawFixed64mV1,
            vec![data(0x71, 3)],
        );
        builder.finish().expect("valid").to_bytes()
    };

    let gguf_provenance = {
        let mut builder = SnapshotBuilder::new(None);
        builder.file(
            "model.gguf",
            false,
            PlannerId::GgufV1,
            vec![data(0x81, 48), data(0x82, 2048)],
        );
        builder.finish().expect("valid").to_bytes()
    };

    vec![
        (
            "empty-snapshot",
            "no parent and no entries; the smallest canonical manifest",
            empty,
        ),
        (
            "single-file",
            "one raw-planned regular file with one data record",
            single_file,
        ),
        (
            "tree",
            "nested explicit directories, safetensors provenance, and an executable file",
            tree,
        ),
        (
            "sparse-file",
            "hole, data, and a beyond-object-size trailing hole",
            sparse,
        ),
        (
            "hardlink-group",
            "a three-path link group re-rooted onto its first sorted path",
            hardlinks,
        ),
        (
            "symlink",
            "a symlink whose target is stored bytes, not a resolved path",
            symlink,
        ),
        (
            "empty-directory",
            "an explicit directory with no children",
            empty_directory,
        ),
        (
            "boundary-objects",
            "data record lengths 1, 64 MiB minus one, and exactly 64 MiB, plus an empty file",
            boundary,
        ),
        (
            "with-parent",
            "names the empty snapshot as its explicit parent history",
            with_parent,
        ),
        (
            "gguf-provenance",
            "a gguf-v1 planned file pinning the remaining planner tag",
            gguf_provenance,
        ),
    ]
}

/// Hand-assembled deviations from the documented grammar.
struct Raw(Vec<u8>);

impl Raw {
    fn manifest() -> Self {
        let mut bytes = Vec::from(*b"TFM1");
        bytes.push(0);
        Self(bytes)
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

    fn directory(self, path: &str) -> Self {
        self.path(path).bytes(&[1])
    }

    fn raw_file(self, path: &str, logical_size: u64, records: &[(u8, u64)]) -> Self {
        let mut raw = self
            .path(path)
            .bytes(&[2, 0, 3])
            .u64(logical_size)
            .u64(records.len() as u64);
        for (tag, length) in records {
            raw.0.push(*tag);
            if *tag == 1 {
                raw.0.extend_from_slice(&[0x99; 32]);
            }
            raw.0.extend_from_slice(&length.to_le_bytes());
        }
        raw
    }
}

fn refusal_bytes() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    let mut cases = Vec::new();
    let valid = Raw::manifest().entry_count(1).raw_file("a", 4, &[(1, 4)]).0;

    let mut bad_magic = valid.clone();
    bad_magic[0] = b'X';
    cases.push(("bad-magic", "bad-magic", bad_magic));

    cases.push(("truncated", "truncated", valid[..valid.len() - 3].to_vec()));

    let mut trailing = valid.clone();
    trailing.push(0);
    cases.push(("trailing-bytes", "trailing-bytes", trailing));

    let mut parent_flag = valid;
    parent_flag[4] = 2;
    cases.push(("parent-flag", "parent-flag", parent_flag));

    cases.push((
        "entry-order",
        "entry-order",
        Raw::manifest()
            .entry_count(2)
            .directory("b")
            .directory("a")
            .0,
    ));
    cases.push((
        "duplicate-path",
        "duplicate-path",
        Raw::manifest()
            .entry_count(2)
            .directory("a")
            .directory("a")
            .0,
    ));
    cases.push((
        "case-fold-collision",
        "case-fold-collision",
        Raw::manifest()
            .entry_count(2)
            .directory("Weights")
            .directory("weights")
            .0,
    ));
    cases.push((
        "missing-parent-directory",
        "missing-parent-directory",
        Raw::manifest()
            .entry_count(1)
            .raw_file("a/b", 1, &[(1, 1)])
            .0,
    ));

    for (name, path) in [
        ("path-absolute", "/rooted"),
        ("path-dot-segment", "a/../b"),
        ("path-empty-segment", "a//b"),
        ("path-forbidden-character", "back\\slash"),
        ("path-trailing-dot-space", "name."),
        ("path-windows-reserved", "COM1.log"),
        ("path-not-nfc", "cafe\u{301}"),
    ] {
        cases.push((name, name, Raw::manifest().entry_count(1).directory(path).0));
    }

    let mut invalid_utf8 = Raw::manifest().entry_count(1).0;
    invalid_utf8.extend_from_slice(&2_u32.to_le_bytes());
    invalid_utf8.extend_from_slice(&[0xFF, 0xFE, 1]);
    cases.push(("path-utf8", "path-utf8", invalid_utf8));

    // Padded past the minimum-entry-size pre-check so the refusal is the
    // zero path length itself, not the count bound.
    let mut zero_length_path = Raw::manifest().entry_count(1).0;
    zero_length_path.extend_from_slice(&0_u32.to_le_bytes());
    zero_length_path.extend_from_slice(&[1, 0]);
    cases.push(("path-length", "path-length", zero_length_path));

    cases.push((
        "unknown-entry-kind",
        "unknown-entry-kind",
        Raw::manifest().entry_count(1).path("x").bytes(&[9]).0,
    ));
    cases.push((
        "executable-flag",
        "executable-flag",
        Raw::manifest()
            .entry_count(1)
            .path("f")
            .bytes(&[2, 2, 3])
            .u64(0)
            .u64(0)
            .0,
    ));
    cases.push((
        "unknown-planner",
        "unknown-planner",
        Raw::manifest()
            .entry_count(1)
            .path("f")
            .bytes(&[2, 0, 9])
            .u64(0)
            .u64(0)
            .0,
    ));
    cases.push((
        "unknown-record-tag",
        "unknown-record-tag",
        Raw::manifest().entry_count(1).raw_file("f", 1, &[(9, 1)]).0,
    ));
    cases.push((
        "zero-length-record",
        "zero-length-record",
        Raw::manifest().entry_count(1).raw_file("f", 1, &[(1, 0)]).0,
    ));
    cases.push((
        "adjacent-holes",
        "adjacent-holes",
        Raw::manifest()
            .entry_count(1)
            .raw_file("f", 2, &[(2, 1), (2, 1)])
            .0,
    ));
    cases.push((
        "data-too-large",
        "data-too-large",
        Raw::manifest()
            .entry_count(1)
            .raw_file("f", MAX_OBJECT_SIZE + 1, &[(1, MAX_OBJECT_SIZE + 1)])
            .0,
    ));
    cases.push((
        "length-sum-mismatch",
        "length-sum-mismatch",
        Raw::manifest().entry_count(1).raw_file("f", 5, &[(1, 4)]).0,
    ));
    // The record sum overflows u64, and the declared logical size is the
    // value a WRAPPING sum would land on: u64::MAX + 64 MiB wraps to
    // 64 MiB - 1. A decoder that adds without checking accepts this manifest
    // outright, so the fixture is refused only by the overflow guard itself.
    cases.push((
        "length-sum-overflow",
        "length-sum-mismatch",
        Raw::manifest()
            .entry_count(1)
            .raw_file(
                "f",
                MAX_OBJECT_SIZE - 1,
                &[(2, u64::MAX), (1, MAX_OBJECT_SIZE)],
            )
            .0,
    ));
    cases.push((
        "record-limit",
        "record-limit",
        Raw::manifest()
            .entry_count(1)
            .path("f")
            .bytes(&[2, 0, 3])
            .u64(0)
            .u64(1_000_001)
            .0,
    ));
    cases.push((
        "count-exceeds-input",
        "count-exceeds-input",
        Raw::manifest().entry_count(u64::MAX).0,
    ));
    cases.push((
        "hardlink-ordinal",
        "hardlink-ordinal",
        Raw::manifest()
            .entry_count(1)
            .path("link")
            .bytes(&[4])
            .u64(0)
            .0,
    ));
    cases.push((
        "symlink-target",
        "symlink-target",
        Raw::manifest()
            .entry_count(1)
            .path("link")
            .bytes(&[3])
            .bytes(&0_u32.to_le_bytes())
            .0,
    ));

    cases
}

fn encode_fixture(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2 + 1);
    for byte in bytes {
        write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex.push('\n');
    hex
}

fn decode_fixture(encoded: &str) -> Vec<u8> {
    let mut lines = encoded.lines();
    let hex = lines.next().expect("fixture must have one hex line");
    assert!(lines.next().is_none(), "fixture has extra lines");
    assert!(encoded.ends_with('\n'), "fixture must end in a line feed");
    assert_eq!(hex.len() % 2, 0, "fixture has an odd hex digit count");
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).expect("ASCII hex is UTF-8");
            u8::from_str_radix(digits, 16).expect("validated hexadecimal pair")
        })
        .collect()
}

fn hex_id(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut hex = String::new();
    for byte in hasher.finalize() {
        write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex
}

fn regenerate(corpus_dir: &Path) {
    let fixtures = corpus_dir.join("fixtures");
    fs::create_dir_all(&fixtures).expect("fixture directory is writable");
    let mut corpus = Corpus {
        schema: "tfm1-vectors.schema.json".to_owned(),
        format: 1,
        hash: "sha256".to_owned(),
        max_object_size: MAX_OBJECT_SIZE,
        fixture_encoding: "lowercase-hex-lf-v1".to_owned(),
        golden: Vec::new(),
        refusals: Vec::new(),
    };
    for (name, description, bytes) in golden_bytes() {
        let fixture = format!("fixtures/{name}.hex");
        fs::write(corpus_dir.join(&fixture), encode_fixture(&bytes)).expect("fixture is writable");
        corpus.golden.push(Golden {
            name: name.to_owned(),
            fixture,
            snapshot_id: hex_id(&bytes),
            description: description.to_owned(),
        });
    }
    for (name, reason, bytes) in refusal_bytes() {
        let fixture = format!("fixtures/refusal-{name}.hex");
        fs::write(corpus_dir.join(&fixture), encode_fixture(&bytes)).expect("fixture is writable");
        corpus.refusals.push(Refusal {
            name: name.to_owned(),
            fixture,
            reason: reason.to_owned(),
        });
    }
    let mut serialized = serde_json::to_string_pretty(&corpus).expect("the corpus serializes");
    serialized.push('\n');
    fs::write(corpus_dir.join("tfm1-vectors.json"), serialized).expect("index is writable");
}

#[test]
fn shared_tfm1_vectors_match_the_canonical_encoder_and_decoder() {
    let corpus_dir = Path::new(CORPUS_DIR);
    if std::env::var_os(REGEN_ENV).is_some() {
        regenerate(corpus_dir);
    }
    let corpus: Corpus = serde_json::from_str(
        &fs::read_to_string(corpus_dir.join("tfm1-vectors.json"))
            .expect("the committed corpus exists"),
    )
    .expect("the committed corpus parses strictly");
    assert_eq!(corpus.max_object_size, MAX_OBJECT_SIZE);

    let golden = golden_bytes();
    assert_eq!(corpus.golden.len(), golden.len());
    for ((name, _description, bytes), row) in golden.iter().zip(&corpus.golden) {
        assert_eq!(&row.name, name);
        let fixture = decode_fixture(
            &fs::read_to_string(corpus_dir.join(&row.fixture)).expect("fixture exists"),
        );
        assert_eq!(&fixture, bytes, "{name}: builder bytes drifted");
        assert_eq!(
            hex_id(&fixture),
            row.snapshot_id,
            "{name}: identity drifted"
        );

        let snapshot = decode(&fixture).expect(name);
        assert_eq!(snapshot.to_bytes(), fixture, "{name}: re-encode drifted");
        assert_eq!(
            snapshot.snapshot_id(),
            SnapshotId::parse_hex(&row.snapshot_id).expect("index ids are lowercase hex"),
        );
    }

    let refusals = refusal_bytes();
    assert_eq!(corpus.refusals.len(), refusals.len());
    for ((name, reason, bytes), row) in refusals.iter().zip(&corpus.refusals) {
        assert_eq!(&row.name, name);
        assert_eq!(&row.reason, reason);
        let fixture = decode_fixture(
            &fs::read_to_string(corpus_dir.join(&row.fixture)).expect("fixture exists"),
        );
        assert_eq!(&fixture, bytes, "{name}: refusal bytes drifted");
        let error = decode(&fixture).expect_err(name);
        assert_eq!(error.reason(), row.reason, "{name}: reason drifted");
    }
}

#[test]
fn every_single_byte_mutation_of_a_golden_fixture_refuses_or_changes_the_id() {
    let (_, _, bytes) = &golden_bytes()[1];
    for index in 0..bytes.len() {
        let mut mutated = bytes.clone();
        mutated[index] ^= 0x01;
        match decode(&mutated) {
            Err(_) => {}
            Ok(snapshot) => {
                assert_eq!(snapshot.to_bytes(), mutated, "canonical re-encode");
                assert_ne!(
                    SnapshotId::of(&mutated),
                    SnapshotId::of(bytes),
                    "byte {index}: an accepted mutation must change the identity"
                );
            }
        }
    }
}
