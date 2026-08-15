//! Language-neutral TFP1 conformance corpus.
//!
//! `TENSORFS_WRITE_TFP1_VECTORS=1 cargo test --test tfp1_vectors` regenerates
//! the committed corpus in place before asserting it, which is how the
//! fixtures were authored; an ordinary run asserts the committed bytes only.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::MAX_OBJECT_SIZE;
use tensorfs_core::tfp1::{decode, encode};

const CORPUS_DIR: &str = "../../spec/v1/tfp1-vectors";
const REGEN_ENV: &str = "TENSORFS_WRITE_TFP1_VECTORS";

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
    /// SHA-256 of the fixture bytes. A drift pin for the corpus itself,
    /// deliberately not called an id: TFP1 carries no identity.
    fixture_sha256: String,
    description: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Refusal {
    name: String,
    fixture: String,
    reason: String,
}

fn digest_of(bytes: &[u8]) -> ObjectDigest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    ObjectDigest::from_bytes(hasher.finalize().into())
}

fn built(objects: &[&[u8]]) -> Vec<u8> {
    let rows: Vec<(ObjectDigest, &[u8])> = objects
        .iter()
        .map(|bytes| (digest_of(bytes), *bytes))
        .collect();
    encode(&rows).expect("golden fixtures encode")
}

fn golden_bytes() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    vec![
        (
            "single-object",
            "one small object; the smallest canonical pack",
            built(&[b"tensorfs staging bytes"]),
        ),
        (
            "three-objects",
            "three objects whose canonical digest order differs from any size order",
            built(&[b"x", b"attention-weights", b"norm"]),
        ),
        (
            "eight-objects",
            "a deeper index walk with mixed small object sizes",
            built(&[
                b"a",
                b"bb",
                b"ccc",
                b"dddd",
                b"eeeee",
                b"ffffff",
                b"ggggggg",
                b"hhhhhhhh",
            ]),
        ),
    ]
}

/// A raw envelope assembler for refusal fixtures the canonical builder can
/// never produce.
struct Raw(Vec<u8>);

impl Raw {
    fn pack() -> Self {
        Self(b"TFP1".to_vec())
    }

    fn magic(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    fn u64(mut self, value: u64) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self.0.extend_from_slice(bytes);
        self
    }

    fn row(self, digest: &ObjectDigest, offset: u64, length: u64) -> Self {
        self.bytes(&digest.as_bytes()[..]).u64(offset).u64(length)
    }
}

fn refusal_bytes() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    let one = digest_of(b"one");
    let two = digest_of(b"two");
    let (low, high) = if one.as_bytes() < two.as_bytes() {
        (one, two)
    } else {
        (two, one)
    };

    let mut cases: Vec<(&'static str, &'static str, Vec<u8>)> = Vec::new();

    cases.push(("bad-magic", "bad-magic", Raw::magic(b"TFPX").u64(1).0));
    cases.push(("zero-objects", "zero-objects", Raw::pack().u64(0).0));
    cases.push(("object-limit", "object-limit", Raw::pack().u64(1_000_001).0));
    cases.push((
        "count-exceeds-input",
        "count-exceeds-input",
        Raw::pack().u64(2).row(&low, 0, 1).0,
    ));
    // The count bound makes an in-index truncation unreachable: any count the
    // bound admits has all of its rows present. Truncation therefore only
    // exists in the header and the payload.
    cases.push((
        "truncated-header",
        "truncated",
        Raw::pack().bytes(&1_u64.to_le_bytes()[..4]).0,
    ));
    cases.push((
        "truncated-payload",
        "truncated",
        Raw::pack()
            .u64(1)
            .row(&digest_of(b"gone"), 0, 4)
            .bytes(b"go")
            .0,
    ));
    cases.push((
        "index-order",
        "index-order",
        Raw::pack().u64(2).row(&high, 0, 1).row(&low, 1, 1).0,
    ));
    cases.push((
        "duplicate-digest",
        "duplicate-digest",
        Raw::pack().u64(2).row(&low, 0, 1).row(&low, 1, 1).0,
    ));
    cases.push((
        "zero-length-object",
        "zero-length-object",
        Raw::pack().u64(1).row(&low, 0, 0).0,
    ));
    cases.push((
        "object-too-large",
        "object-too-large",
        Raw::pack().u64(1).row(&low, 0, MAX_OBJECT_SIZE + 1).0,
    ));
    cases.push((
        "payload-too-large",
        "payload-too-large",
        Raw::pack()
            .u64(2)
            .row(&low, 0, MAX_OBJECT_SIZE)
            .row(&high, MAX_OBJECT_SIZE, MAX_OBJECT_SIZE)
            .0,
    ));
    cases.push((
        "offset-mismatch-start",
        "offset-mismatch",
        Raw::pack()
            .u64(1)
            .row(&digest_of(b"off"), 7, 3)
            .bytes(b"off")
            .0,
    ));
    cases.push((
        "offset-mismatch-gap",
        "offset-mismatch",
        Raw::pack().u64(2).row(&low, 0, 1).row(&high, 2, 1).0,
    ));
    cases.push(("trailing-bytes", "trailing-bytes", {
        let mut bytes = built(&[b"sealed"]);
        bytes.push(0);
        bytes
    }));
    cases.push((
        "digest-mismatch",
        "digest-mismatch",
        Raw::pack()
            .u64(1)
            .row(&digest_of(b"good"), 0, 4)
            .bytes(b"goud")
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
        schema: "tfp1-vectors.schema.json".to_owned(),
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
            fixture_sha256: hex_id(&bytes),
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
    fs::write(corpus_dir.join("tfp1-vectors.json"), serialized).expect("index is writable");
}

#[test]
fn shared_tfp1_vectors_match_the_canonical_encoder_and_decoder() {
    let corpus_dir = Path::new(CORPUS_DIR);
    if std::env::var_os(REGEN_ENV).is_some() {
        regenerate(corpus_dir);
    }
    let corpus: Corpus = serde_json::from_str(
        &fs::read_to_string(corpus_dir.join("tfp1-vectors.json"))
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
        assert_eq!(hex_id(&fixture), row.fixture_sha256, "{name}: pin drifted");

        let pack = decode(&fixture).expect(name);
        let decoded: Vec<(ObjectDigest, &[u8])> = pack
            .objects()
            .map(|object| (object.digest(), object.bytes()))
            .collect();
        assert_eq!(
            encode(&decoded).expect(name),
            fixture,
            "{name}: re-encode drifted"
        );
    }

    let refusals = refusal_bytes();
    assert_eq!(corpus.refusals.len(), refusals.len());
    for ((name, reason, bytes), row) in refusals.iter().zip(&corpus.refusals) {
        assert_eq!(row.name, *name);
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
fn every_single_byte_mutation_of_a_golden_pack_refuses_or_changes_its_content() {
    let (_, _, bytes) = &golden_bytes()[0];
    for index in 0..bytes.len() {
        let mut mutated = bytes.clone();
        mutated[index] ^= 0x01;
        match decode(&mutated) {
            Err(_) => {}
            Ok(pack) => {
                let decoded: Vec<(ObjectDigest, Vec<u8>)> = pack
                    .objects()
                    .map(|object| (object.digest(), object.bytes().to_vec()))
                    .collect();
                let rows: Vec<(ObjectDigest, &[u8])> = decoded
                    .iter()
                    .map(|(digest, bytes)| (*digest, bytes.as_slice()))
                    .collect();
                assert_eq!(
                    encode(&rows).expect("accepted mutations stay canonical"),
                    mutated,
                    "byte {index}: an accepted mutation must still be canonical"
                );
                assert_ne!(&mutated, bytes, "byte {index}: mutation must differ");
            }
        }
    }
}
