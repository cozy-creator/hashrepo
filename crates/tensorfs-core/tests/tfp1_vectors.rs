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
    generated_golden: Vec<GeneratedGolden>,
    refusals: Vec<Refusal>,
}

/// A golden pack too large to commit as hex. The corpus carries the recipe
/// and the envelope's own SHA-256; every consumer builds the bytes itself and
/// must land on that exact pin. A committed fixture would be 128 MiB of hex.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneratedGolden {
    name: String,
    objects: Vec<GeneratedObject>,
    pack_sha256: String,
    description: String,
}

/// One object body as `byte[i] = (ramp_seed + i) mod 256` — position-sensitive
/// (unlike a constant fill, any shift or reorder changes the digest) and
/// trivially identical in every language.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneratedObject {
    ramp_seed: u8,
    length: u64,
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

/// The one deterministic byte ramp both languages generate from the corpus.
fn ramp(seed: u8, length: u64) -> Vec<u8> {
    (0..length)
        .map(|index| seed.wrapping_add(index as u8))
        .collect()
}

fn generated_pack(objects: &[(u8, u64)]) -> Vec<u8> {
    let bodies: Vec<Vec<u8>> = objects
        .iter()
        .map(|(seed, length)| ramp(*seed, *length))
        .collect();
    let rows: Vec<(ObjectDigest, &[u8])> = bodies
        .iter()
        .map(|body| (digest_of(body), body.as_slice()))
        .collect();
    encode(&rows).expect("generated goldens encode")
}

/// One generated-golden recipe: name, description, and the ramp objects.
type GeneratedRecipe = (&'static str, &'static str, Vec<(u8, u64)>);

/// Goldens whose bytes are generated rather than committed. The 64 MiB
/// boundary pack is a real golden — decoded, verified and re-encoded — it
/// simply cannot live in git as 128 MiB of hexadecimal.
fn generated_golden_recipes() -> Vec<GeneratedRecipe> {
    vec![(
        "full-payload-single-object",
        "one object at the 64 MiB bound; the payload cap equals the object cap, so this pack is exactly full",
        vec![(0xa5, MAX_OBJECT_SIZE)],
    )]
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
    // Overlap, the gap's mirror image: the second row starts two bytes INSIDE
    // the first. The payload is the eight bytes both rows actually address and
    // both digests are the real hashes of the overlapping slices, so a decoder
    // that dropped the offset rule would accept this pack outright rather than
    // trip over a length or a digest. Only the tiling rule refuses it.
    cases.push(("offset-overlap", "offset-mismatch", {
        let payload = b"overlap!";
        let front = digest_of(&payload[0..4]);
        let back = digest_of(&payload[2..6]);
        assert!(
            front.as_bytes() < back.as_bytes(),
            "the overlap fixture must stay in canonical index order"
        );
        Raw::pack()
            .u64(2)
            .row(&front, 0, 4)
            .row(&back, 2, 4)
            .bytes(payload)
            .0
    }));
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
        generated_golden: Vec::new(),
        refusals: Vec::new(),
    };
    for (name, description, objects) in generated_golden_recipes() {
        corpus.generated_golden.push(GeneratedGolden {
            name: name.to_owned(),
            pack_sha256: hex_id(&generated_pack(&objects)),
            objects: objects
                .into_iter()
                .map(|(ramp_seed, length)| GeneratedObject { ramp_seed, length })
                .collect(),
            description: description.to_owned(),
        });
    }
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

/// The corpus's generated goldens: built from the committed recipe, pinned by
/// the envelope's own digest, and put through the same decode / verify /
/// re-encode contract every committed golden faces.
///
/// A full per-byte mutation sweep is not run here — 67 million decodes of a
/// 64 MiB pack is not a test, it is an outage. The pin already fails on any
/// single changed byte; the sweep below is the structural sample that proves
/// the pin is not the only thing guarding the pack.
#[test]
fn generated_golden_packs_hold_their_pin_at_the_payload_bound() {
    let corpus: Corpus = serde_json::from_str(
        &fs::read_to_string(Path::new(CORPUS_DIR).join("tfp1-vectors.json"))
            .expect("the committed corpus exists"),
    )
    .expect("the committed corpus parses strictly");

    let recipes = generated_golden_recipes();
    assert_eq!(corpus.generated_golden.len(), recipes.len());
    for ((name, _description, objects), row) in recipes.iter().zip(&corpus.generated_golden) {
        assert_eq!(&row.name, name);
        let declared: Vec<(u8, u64)> = row
            .objects
            .iter()
            .map(|object| (object.ramp_seed, object.length))
            .collect();
        assert_eq!(&declared, objects, "{name}: the committed recipe drifted");

        let pack_bytes = generated_pack(objects);
        assert_eq!(hex_id(&pack_bytes), row.pack_sha256, "{name}: pin drifted");

        let payload: u64 = objects.iter().map(|(_, length)| *length).sum();
        assert_eq!(
            pack_bytes.len() as u64,
            12 + 48 * objects.len() as u64 + payload,
            "{name}: the envelope is magic + count + rows + whole objects"
        );

        let pack = decode(&pack_bytes).expect(name);
        assert_eq!(pack.object_count(), objects.len());
        let decoded: Vec<(ObjectDigest, &[u8])> = pack
            .objects()
            .map(|object| (object.digest(), object.bytes()))
            .collect();
        assert_eq!(
            decoded
                .iter()
                .map(|(_, bytes)| bytes.len() as u64)
                .sum::<u64>(),
            payload,
        );
        assert_eq!(
            encode(&decoded).expect(name),
            pack_bytes,
            "{name}: re-encode drifted"
        );
    }

    // The boundary claim itself: exactly one object, exactly the payload cap.
    let (_, _, boundary) = &recipes[0];
    assert_eq!(boundary.len(), 1);
    assert_eq!(boundary[0].1, MAX_OBJECT_SIZE);

    // Structural sample, each mutation applied and reverted in place so the
    // sweep never holds a second copy of a 64 MiB pack.
    let mut full = generated_pack(boundary);
    let last = full.len() - 1;
    for (index, expected) in [(12, "digest-mismatch"), (last, "digest-mismatch")] {
        full[index] ^= 0x01;
        assert_eq!(
            decode(&full).expect_err("a mutated boundary pack").reason(),
            expected,
            "byte {index}"
        );
        full[index] ^= 0x01;
    }

    full.push(0);
    assert_eq!(
        decode(&full)
            .expect_err("a full pack has no room left")
            .reason(),
        "trailing-bytes"
    );
    full.pop();

    full.pop();
    assert_eq!(
        decode(&full)
            .expect_err("one byte short of the bound")
            .reason(),
        "truncated"
    );

    // The row's own length field, one byte past the bound: refused on the
    // declaration, before a single payload byte is read.
    let mut over_row = full[..60].to_vec();
    over_row[52..60].copy_from_slice(&(MAX_OBJECT_SIZE + 1).to_le_bytes());
    assert_eq!(
        decode(&over_row)
            .expect_err("one byte over the bound")
            .reason(),
        "object-too-large"
    );
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
