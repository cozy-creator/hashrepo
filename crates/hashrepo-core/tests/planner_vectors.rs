use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Component, Path};

use hashrepo_core::object::plan_and_hash;
use hashrepo_core::planner::{ByteSource, MAX_OBJECT_SIZE, PlannerId, RegionKind, plan};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const CORPUS_PATH: &str = "../../spec/v1/planner-vectors/planner-vectors.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    #[serde(rename = "$schema")]
    schema: String,
    format: u32,
    hash: String,
    max_object_size: u64,
    fixture_encoding: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    fixture: String,
    source_sha256: String,
    #[serde(default)]
    zero_tail: u64,
    classification: Classification,
    expected: Expected,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Classification {
    Semantic,
    Raw,
    Fallback,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    planner: String,
    file_size: u64,
    objects: Vec<ExpectedObject>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedObject {
    offset: u64,
    length: u64,
    kind: String,
    digest: String,
}

struct FixtureSource {
    prefix: Vec<u8>,
    zero_tail: u64,
}

impl ByteSource for FixtureSource {
    fn len(&self) -> u64 {
        self.prefix.len() as u64 + self.zero_tail
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> io::Result<()> {
        let end = offset
            .checked_add(destination.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.len() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }

        destination.fill(0);
        let prefix_length = self.prefix.len() as u64;
        if offset < prefix_length {
            let copied = (prefix_length - offset).min(destination.len() as u64) as usize;
            destination[..copied]
                .copy_from_slice(&self.prefix[offset as usize..offset as usize + copied]);
        }
        Ok(())
    }

    fn check_unchanged(&self) -> io::Result<()> {
        Ok(())
    }
}

fn tagged_sha256(source: &impl ByteSource) -> String {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut offset = 0_u64;
    while offset < source.len() {
        let length = (source.len() - offset).min(buffer.len() as u64) as usize;
        source
            .read_exact_at(offset, &mut buffer[..length])
            .expect("fixture source is immutable and complete");
        hasher.update(&buffer[..length]);
        offset += length as u64;
    }

    let mut tagged = String::from("sha256:");
    for byte in hasher.finalize() {
        write!(tagged, "{byte:02x}").expect("writing to a String cannot fail");
    }
    tagged
}

fn decode_fixture(encoded: &str) -> Vec<u8> {
    let mut lines = encoded.lines();
    let hex = lines
        .next()
        .expect("planner fixture must have one hex line");
    assert!(lines.next().is_none(), "planner fixture has extra lines");
    assert!(
        encoded.ends_with('\n'),
        "planner fixture must end in a line terminator"
    );
    assert_eq!(hex.len() % 2, 0, "fixture has an odd hex digit count");
    assert!(
        hex.bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "fixture is not lowercase hexadecimal"
    );

    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).expect("ASCII hex is UTF-8");
            u8::from_str_radix(digits, 16).expect("validated hexadecimal pair")
        })
        .collect()
}

#[test]
fn fixture_decoder_accepts_one_lf_or_crlf_terminated_hex_line() {
    assert_eq!(decode_fixture("00ff\n"), [0x00, 0xff]);
    assert_eq!(decode_fixture("00ff\r\n"), [0x00, 0xff]);
}

const fn kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Header => "header",
        RegionKind::Tensor => "tensor",
        RegionKind::Raw => "raw",
    }
}

#[test]
fn shared_planner_vectors_match_the_closed_automatic_registry() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let corpus_path = crate_dir.join(CORPUS_PATH);
    let corpus_dir = corpus_path
        .parent()
        .expect("corpus path has a parent directory");
    let corpus: Corpus =
        serde_json::from_slice(&fs::read(&corpus_path).expect("read shared planner vector corpus"))
            .expect("parse shared planner vector corpus");

    assert_eq!(corpus.schema, "planner-vectors.schema.json");
    assert_eq!(corpus.format, 1);
    assert_eq!(corpus.hash, "sha256");
    assert_eq!(corpus.max_object_size, MAX_OBJECT_SIZE);
    assert_eq!(corpus.fixture_encoding, "lowercase-hex-lf-v1");
    assert!(!corpus.cases.is_empty());

    let mut names = HashSet::new();
    for case in corpus.cases {
        assert!(names.insert(case.name.clone()), "duplicate case name");

        let relative = Path::new(&case.fixture);
        assert!(!relative.is_absolute(), "fixture path must be relative");
        assert!(
            relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
            "fixture path must not traverse"
        );
        assert_eq!(relative.parent(), Some(Path::new("fixtures")));

        let source = FixtureSource {
            prefix: decode_fixture(
                &fs::read_to_string(corpus_dir.join(relative)).expect("read planner fixture"),
            ),
            zero_tail: case.zero_tail,
        };
        assert_eq!(tagged_sha256(&source), case.source_sha256, "{}", case.name);

        let planned = plan(&source).expect("immutable fixture must produce a plan");
        let hashed = plan_and_hash(&source).expect("immutable fixture must plan and hash");

        assert_eq!(
            planned.planner().as_str(),
            case.expected.planner,
            "{}",
            case.name
        );
        assert_eq!(hashed.planner(), planned.planner(), "{}", case.name);
        assert_eq!(
            planned.file_size(),
            case.expected.file_size,
            "{}",
            case.name
        );
        assert_eq!(hashed.file_size(), planned.file_size(), "{}", case.name);
        assert_eq!(
            planned.regions().len(),
            case.expected.objects.len(),
            "{}",
            case.name
        );
        assert_eq!(
            hashed.objects().len(),
            case.expected.objects.len(),
            "{}",
            case.name
        );

        match case.classification {
            Classification::Semantic => {
                assert_ne!(planned.planner(), PlannerId::RawFixed64mV1, "{}", case.name);
            }
            Classification::Raw | Classification::Fallback => {
                assert_eq!(planned.planner(), PlannerId::RawFixed64mV1, "{}", case.name);
            }
        }

        for ((region, object), expected) in planned
            .regions()
            .iter()
            .zip(hashed.objects())
            .zip(&case.expected.objects)
        {
            assert_eq!(region.offset(), expected.offset, "{}", case.name);
            assert_eq!(region.length(), expected.length, "{}", case.name);
            assert_eq!(kind_name(region.kind()), expected.kind, "{}", case.name);
            assert_eq!(object.length(), expected.length, "{}", case.name);
            assert_eq!(kind_name(object.kind()), expected.kind, "{}", case.name);
            assert_eq!(
                object.digest().to_string(),
                expected.digest,
                "{}",
                case.name
            );
        }
    }
}
