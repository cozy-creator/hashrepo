//! The pointer-stub format as a contract.
//!
//! Stubs are the one projection artifact a foreign consumer parses, so their
//! BYTES are pinned: key order, absence of whitespace, the trailing line
//! feed. `TENSORFS_WRITE_TFSSTUB1_VECTORS=1` regenerates the committed corpus
//! in place before asserting it, which is legitimate only while the
//! pre-launch format may still be replaced.
//!
//! The corpus is plain ASCII, one stub per fixture file, so the Python suite
//! reads the same files and checks the same bytes
//! (`python/tests/test_pointer_stubs.py`).

#![cfg(any(unix, windows))]

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tensorfs_core::layout::{STUB_MAGIC, parse_stub, stub_bytes};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotBuilder};

const CORPUS_DIR: &str = "../../spec/v1/tfsstub1-vectors";
const REGEN_ENV: &str = "TENSORFS_WRITE_TFSSTUB1_VECTORS";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    format: u32,
    magic: String,
    hash: String,
    fixture_encoding: String,
    golden: Vec<Golden>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Golden {
    name: String,
    fixture: String,
    body_sha256: String,
    size: u64,
    description: String,
}

fn digest_of(seed: &[u8]) -> ObjectDigest {
    ObjectDigest::from_bytes(Sha256::digest(seed).into())
}

/// The body digest of a real safetensors entry, so the corpus fences the TFM1
/// body encoding itself and not only the JSON assembly.
fn real_safetensors_body_digest() -> ObjectDigest {
    let mut builder = SnapshotBuilder::new(None);
    builder.file(
        "model.safetensors",
        false,
        PlannerId::SafetensorsV1,
        vec![
            FileRecord::Data {
                digest: digest_of(b"tfsstub1-vector-header"),
                length: 96,
            },
            FileRecord::Data {
                digest: digest_of(b"tfsstub1-vector-tensor"),
                length: 4096,
            },
        ],
    );
    let snapshot = builder.finish().expect("the fixture entry is valid");
    match &snapshot.entries()[0].1 {
        Entry::File { body, .. } => body.body_sha256(),
        other => panic!("expected a file entry, got {other:?}"),
    }
}

/// Every golden case, with a distinct digest AND a distinct size, so no
/// assertion can pass by matching the wrong row.
fn cases() -> Vec<(&'static str, &'static str, ObjectDigest, u64)> {
    vec![
        (
            "zero-size",
            "a tensor container with no bytes: the size field is 0, not absent",
            digest_of(b""),
            0,
        ),
        (
            "small",
            "the common case: a few kibibytes of weights behind ~128 bytes of stub",
            digest_of(b"tfsstub1-vector-small"),
            4096,
        ),
        (
            "four-gigabyte",
            "the doc's example size; stat sees the stub, this field sees the truth",
            digest_of(b"tfsstub1-vector-4gb"),
            4_000_000_000,
        ),
        (
            "max-u64-size",
            "u64::MAX: the size is a JSON number a consumer must not read as a float",
            digest_of(b"tfsstub1-vector-max"),
            u64::MAX,
        ),
        (
            "real-safetensors-body",
            "the body digest of an actual two-record safetensors entry",
            real_safetensors_body_digest(),
            4192,
        ),
    ]
}

fn regenerate(corpus_dir: &Path) {
    let fixtures = corpus_dir.join("fixtures");
    let _ = fs::remove_dir_all(&fixtures);
    fs::create_dir_all(&fixtures).expect("fixture directory is writable");
    let mut corpus = Corpus {
        format: 1,
        magic: "TFSSTUB1".to_owned(),
        hash: "sha256".to_owned(),
        fixture_encoding: "raw-ascii-v1".to_owned(),
        golden: Vec::new(),
    };
    for (name, description, digest, size) in cases() {
        let fixture = format!("fixtures/{name}.stub");
        fs::write(corpus_dir.join(&fixture), stub_bytes(&digest, size))
            .expect("fixture is writable");
        corpus.golden.push(Golden {
            name: name.to_owned(),
            fixture,
            body_sha256: digest.to_hex(),
            size,
            description: description.to_owned(),
        });
    }
    let mut serialized = serde_json::to_string_pretty(&corpus).expect("the corpus serializes");
    serialized.push('\n');
    fs::write(corpus_dir.join("tfsstub1-vectors.json"), serialized).expect("index is writable");
}

#[test]
fn the_committed_stub_corpus_matches_the_renderer_and_the_parser() {
    let corpus_dir = Path::new(CORPUS_DIR);
    if std::env::var_os(REGEN_ENV).is_some() {
        regenerate(corpus_dir);
    }
    let corpus: Corpus = serde_json::from_str(
        &fs::read_to_string(corpus_dir.join("tfsstub1-vectors.json")).expect("the index reads"),
    )
    .expect("the index parses");
    assert_eq!(corpus.magic.as_bytes(), STUB_MAGIC);
    assert_eq!(corpus.golden.len(), cases().len(), "the corpus is stale");

    let mut seen_bytes: Vec<Vec<u8>> = Vec::new();
    for (golden, (name, _, digest, size)) in corpus.golden.iter().zip(cases()) {
        assert_eq!(golden.name, name, "corpus order drifted");
        assert_eq!(golden.body_sha256, digest.to_hex(), "{name}");
        assert_eq!(golden.size, size, "{name}");

        let committed = fs::read(corpus_dir.join(&golden.fixture))
            .unwrap_or_else(|_| panic!("{name}: fixture reads"));
        assert_eq!(
            committed,
            stub_bytes(&digest, size),
            "{name}: the renderer no longer produces the committed bytes"
        );
        assert!(committed.starts_with(STUB_MAGIC), "{name}");
        assert!(committed.ends_with(b"\n"), "{name}: one trailing line feed");
        assert_eq!(
            committed.iter().filter(|byte| **byte == b'\n').count(),
            1,
            "{name}: a stub is exactly one line"
        );

        let parsed = parse_stub(&committed).unwrap_or_else(|| panic!("{name}: parses"));
        assert_eq!(parsed.body_sha256, digest, "{name}");
        assert_eq!(parsed.size, size, "{name}");

        assert!(
            !seen_bytes.contains(&committed),
            "{name}: two fixtures are byte-identical, so neither proves anything"
        );
        seen_bytes.push(committed);
    }
}

/// Corrupting either field changes the bytes and the parse — the round-trip
/// is load-bearing, not a tautology over one value.
#[test]
fn corrupting_either_stub_field_is_visible() {
    let digest = digest_of(b"round-trip");
    let honest = stub_bytes(&digest, 4096);
    let parsed = parse_stub(&honest).expect("the honest stub parses");

    let wrong_digest = stub_bytes(&digest_of(b"a different body"), 4096);
    assert_ne!(wrong_digest, honest);
    assert_ne!(
        parse_stub(&wrong_digest).expect("still a stub").body_sha256,
        parsed.body_sha256
    );

    let wrong_size = stub_bytes(&digest, 4097);
    assert_ne!(wrong_size, honest);
    assert_eq!(parse_stub(&wrong_size).expect("still a stub").size, 4097);
}

/// The parser refuses what is not a stub, so "is this a stub?" never answers
/// yes for a weights file or a truncated line.
#[test]
fn the_parser_refuses_everything_that_is_not_a_stub() {
    let digest = digest_of(b"refusals");
    let honest = String::from_utf8(stub_bytes(&digest, 7)).expect("stubs are ASCII");
    for (name, bytes) in [
        ("empty", Vec::new()),
        (
            "a safetensors header",
            vec![0x40, 0, 0, 0, 0, 0, 0, 0, b'{'],
        ),
        ("gguf magic", b"GGUF\x03\x00\x00\x00".to_vec()),
        ("magic only", STUB_MAGIC.to_vec()),
        (
            "no space",
            honest.replace("TFSSTUB1 ", "TFSSTUB1").into_bytes(),
        ),
        (
            "truncated json",
            honest.as_bytes()[..honest.len() - 4].to_vec(),
        ),
        (
            "an unknown field",
            honest
                .replace(",\"read\"", ",\"why\":1,\"read\"")
                .into_bytes(),
        ),
        (
            "another reader",
            honest
                .replace("\"tensorfs\"", "\"someone-else\"")
                .into_bytes(),
        ),
        (
            "an uppercase digest",
            honest
                .replace(&digest.to_hex(), &digest.to_hex().to_uppercase())
                .into_bytes(),
        ),
    ] {
        assert!(
            parse_stub(&bytes).is_none(),
            "{name} must not parse as a pointer stub"
        );
    }
}

/// The magic is a GREP FENCE: `TFSSTUB1` appears only where the format is
/// defined, documented, pinned, or asserted. A second in-tree speller of the
/// magic is how a "contract" quietly forks.
#[test]
fn the_stub_magic_is_unique_in_tree() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels under the repository root")
        .to_path_buf();
    let allowed = [
        "crates/tensorfs-core/src/layout.rs",
        "crates/tensorfs-core/tests/projection.rs",
        "crates/tensorfs-core/tests/tfsstub1_vectors.rs",
        "crates/tensorfs-py/src/lib.rs",
        "docs/mixed-cas-layout.md",
        "python/src/tensorfs/_tensorfs.pyi",
        "python/tests/test_pointer_stubs.py",
        "README.md",
        "spec/v1/TFSSTUB1.md",
        "spec/v1/tfsstub1-vectors",
    ];

    let mut offenders = Vec::new();
    let mut examined = 0_u64;
    let mut stack = vec![root.clone()];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            examined += 1;
            if !text.contains("TFSSTUB1") {
                continue;
            }
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if !allowed.iter().any(|prefix| relative.starts_with(prefix)) {
                offenders.push(relative);
            }
        }
    }

    assert!(
        examined > 50,
        "the fence read {examined} files; it cannot vouch for a tree it never walked"
    );
    offenders.sort();
    assert!(
        offenders.is_empty(),
        "TFSSTUB1 escaped its definition sites: {offenders:?}\n\
         The magic has one renderer (`layout::stub_bytes`) and one spec \
         (`spec/v1/TFSSTUB1.md`). A second speller forks the contract."
    );
}
