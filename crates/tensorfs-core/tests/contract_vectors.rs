//! The language-neutral contract conformance corpus (#114).
//!
//! Rust authors `spec/v1/contract-vectors/contract-vectors.json`; Go (and any
//! later implementation) parses the same documents and must agree on accept /
//! refuse verdicts, refusal labels, digests and stamps. That agreement — not
//! discipline — is the cross-language sync mechanism for contract documents.
//!
//! Regenerate with `TENSORFS_WRITE_CONTRACT_VECTORS=1 cargo test --test
//! contract_vectors`, legitimate only while the pre-launch format may be
//! replaced in place.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tensorfs_core::contract::{BUILTIN, Contract};

#[derive(Debug, Deserialize, Serialize)]
struct Corpus {
    format: String,
    golden: Vec<Golden>,
    refusals: Vec<Refusal>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Golden {
    name: String,
    /// Library documents by file (relative to `spec/v1/`); customs inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<String>,
    /// Bare lowercase hex SHA-256 of the canonical rendering.
    digest: String,
    /// `name@version` or `sha256:<hex>`.
    stamp: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Refusal {
    name: String,
    document: String,
    /// The stable kebab-case `ContractError::reason` label.
    reason: String,
}

fn spec_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/v1")
        .canonicalize()
        .expect("the spec directory exists")
}

fn corpus_path() -> PathBuf {
    spec_root().join("contract-vectors/contract-vectors.json")
}

fn hex_of(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// A custom exercising every canonical-rendering branch the library does not:
/// nameless identity, top-level dtype, a permute view mixing literal / axis /
/// divided-axis / auto, optional tensors, dtype constraints, and named sets.
const CUSTOM_RICH: &str = r#"{
    "format": "tensorfs-contract-v1",
    "description": "vector fixture: every declaration branch",
    "dtype": "bfloat16",
    "tensors": [
        {"role": "blocks.{i}.attn.qkv", "pattern": "blocks.{i}.attn.qkv_proj.weight",
         "rank": 2, "dtypes": ["BF16", "F16"],
         "fusion": {"axis": 0, "parts": [{"role": "q", "share": 2}, {"role": "k", "share": 1},
                                         {"role": "v", "share": 1}]}},
        {"role": "blocks.{i}.rope", "pattern": "blocks.{i}.rope.weight", "required": false,
         "permute": {"view": [4, "shape[0]/2", "auto", "shape[1]"], "axes": [0, 2, 1, 3]}}
    ],
    "sets": {"adaln": ["blocks.{i}.adaln.weight", "norm_out.weight"], "rope": ["blocks.{i}.rope.weight"]}
}"#;

/// The interleaved shapes: head-major groups on the fused side, the sole
/// empty-role part on the split side.
const CUSTOM_INTERLEAVED: &str = r#"{
    "format": "tensorfs-contract-v1",
    "tensors": [
        {"role": "b.{i}.qkv", "pattern": "b.{i}.qkv.weight", "rank": 2,
         "fusion": {"axis": 0, "groups": 56,
                    "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1},
                              {"role": "v", "share": 1}]}},
        {"role": "b.{i}.qkv#q", "pattern": "b.{i}.q.weight", "rank": 2, "required": false,
         "fusion": {"axis": 0, "groups": 56, "parts": [{"role": "", "share": 1}]}}
    ]
}"#;

fn golden_documents() -> Vec<(String, Option<String>, String)> {
    // (case name, file reference, document text)
    let mut cases: Vec<(String, Option<String>, String)> = BUILTIN
        .iter()
        .map(|(file, document)| {
            (
                file.trim_end_matches(".json").to_owned(),
                Some(format!("contracts/{file}")),
                (*document).to_owned(),
            )
        })
        .collect();
    cases.push(("custom-rich".to_owned(), None, CUSTOM_RICH.to_owned()));
    cases.push((
        "custom-interleaved".to_owned(),
        None,
        CUSTOM_INTERLEAVED.to_owned(),
    ));
    cases
}

fn refusal_documents() -> Vec<(&'static str, String, &'static str)> {
    const MINIMAL: &str = r#"{
        "format": "tensorfs-contract-v1",
        "tensors": [{"role": "a.b", "pattern": "a.b"}]
    }"#;
    let with = |replacement: &str| MINIMAL.replace(r#""tensors""#, replacement);
    vec![
        ("not-json", "{".to_owned(), "json"),
        (
            "unknown-field",
            MINIMAL.replace(r#""format""#, r#""unknown": 1, "format""#),
            "json",
        ),
        (
            "wrong-format",
            MINIMAL.replace("tensorfs-contract-v1", "tensorfs-contract-v2"),
            "format",
        ),
        (
            "uppercase-name",
            with(r#""name": "Sdxl.Fused", "version": 1, "tensors""#),
            "name",
        ),
        (
            "zero-version",
            with(r#""name": "sdxl.fused", "version": 0, "tensors""#),
            "version",
        ),
        (
            "name-without-version",
            with(r#""name": "sdxl.fused", "tensors""#),
            "identity",
        ),
        (
            "version-without-name",
            with(r#""version": 1, "tensors""#),
            "identity",
        ),
        (
            "uppercase-dtype",
            with(r#""dtype": "BF16", "tensors""#),
            "dtype",
        ),
        (
            "no-tensors",
            MINIMAL.replace(r#"[{"role": "a.b", "pattern": "a.b"}]"#, "[]"),
            "no-tensors",
        ),
        (
            "adjacent-holes",
            MINIMAL.replace("a.b", "a.{i}{i}"),
            "pattern",
        ),
        (
            "role-hole-mismatch",
            MINIMAL.replace(r#""role": "a.b""#, r#""role": "a.{i}""#),
            "role-holes",
        ),
        (
            "duplicate-pattern",
            MINIMAL.replace(
                r#"[{"role": "a.b", "pattern": "a.b"}]"#,
                r#"[{"role": "a.b", "pattern": "a.b"}, {"role": "a.c", "pattern": "a.b"}]"#,
            ),
            "duplicate",
        ),
        (
            "inner-axis-fusion",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "fusion": {"axis": 1, "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1}]}"#,
            ),
            "fusion",
        ),
        (
            "one-part-fusion",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "fusion": {"axis": 0, "parts": [{"role": "q", "share": 1}]}"#,
            ),
            "fusion",
        ),
        (
            "zero-share",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "fusion": {"axis": 0, "parts": [{"role": "q", "share": 0}, {"role": "k", "share": 1}]}"#,
            ),
            "fusion",
        ),
        (
            "identity-permute",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "permute": {"view": [2, "auto"], "axes": [0, 1]}"#,
            ),
            "permute",
        ),
        (
            "axes-not-a-permutation",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "permute": {"view": [2, "auto"], "axes": [1, 1]}"#,
            ),
            "permute",
        ),
        (
            "permute-inside-fusion",
            MINIMAL.replace(
                r#""pattern": "a.b""#,
                r#""pattern": "a.b", "fusion": {"axis": 0, "parts": [{"role": "q", "share": 1}, {"role": "k", "share": 1}]}, "permute": {"view": [2, "auto"], "axes": [1, 0]}"#,
            ),
            "permute",
        ),
        (
            "empty-set",
            MINIMAL.replace(
                r#"}]"#,
                r#"}], "sets": {"adaln": []}"#,
            ),
            "set",
        ),
    ]
}

fn build_corpus() -> Corpus {
    let golden = golden_documents()
        .into_iter()
        .map(|(name, file, document)| {
            let parsed = Contract::parse(&document)
                .unwrap_or_else(|error| panic!("golden {name} refuses: {error}"));
            Golden {
                name,
                document: file.is_none().then_some(document),
                file,
                digest: hex_of(parsed.digest()),
                stamp: parsed.stamp().to_string(),
            }
        })
        .collect();
    let refusals = refusal_documents()
        .into_iter()
        .map(|(name, document, reason)| {
            let error = Contract::parse(&document)
                .expect_err(&format!("refusal {name} unexpectedly parses"));
            assert_eq!(error.reason(), reason, "refusal {name} label");
            Refusal {
                name: name.to_owned(),
                document,
                reason: reason.to_owned(),
            }
        })
        .collect();
    Corpus {
        format: "tensorfs-contract-vectors-v1".to_owned(),
        golden,
        refusals,
    }
}

#[test]
fn shared_contract_vectors_match_the_canonical_validator() {
    let corpus = build_corpus();
    if std::env::var_os("TENSORFS_WRITE_CONTRACT_VECTORS").is_some() {
        fs::create_dir_all(corpus_path().parent().unwrap()).expect("directory");
        let mut rendered = serde_json::to_string_pretty(&corpus).expect("serializes");
        rendered.push('\n');
        fs::write(corpus_path(), rendered).expect("writes");
    }

    let committed: Corpus = serde_json::from_str(
        &fs::read_to_string(corpus_path()).expect("the committed corpus exists"),
    )
    .expect("the committed corpus parses");
    assert_eq!(committed.format, "tensorfs-contract-vectors-v1");
    assert_eq!(committed.golden.len(), corpus.golden.len());
    assert_eq!(committed.refusals.len(), corpus.refusals.len());

    // Verify the COMMITTED corpus against the live validator: file references
    // resolve against spec/v1, digests and stamps agree, refusals refuse with
    // the same labels.
    let mut stamps = BTreeMap::new();
    for case in &committed.golden {
        let document = match (&case.file, &case.document) {
            (Some(file), None) => fs::read_to_string(spec_root().join(file))
                .unwrap_or_else(|_| panic!("{file} exists")),
            (None, Some(inline)) => inline.clone(),
            other => panic!("{}: exactly one of file/document, got {other:?}", case.name),
        };
        let parsed = Contract::parse(&document)
            .unwrap_or_else(|error| panic!("golden {} refuses: {error}", case.name));
        assert_eq!(hex_of(parsed.digest()), case.digest, "{} digest", case.name);
        assert_eq!(
            parsed.stamp().to_string(),
            case.stamp,
            "{} stamp",
            case.name
        );
        stamps.insert(case.stamp.clone(), case.name.clone());
    }
    // Every builtin appears, so the corpus cannot silently trail the library.
    for (file, document) in BUILTIN {
        let parsed = Contract::parse(document).expect("builtin parses");
        assert!(
            stamps.contains_key(&parsed.stamp().to_string()),
            "builtin {file} is missing from the corpus; regenerate it"
        );
    }
    for case in &committed.refusals {
        let error = Contract::parse(&case.document)
            .expect_err(&format!("refusal {} unexpectedly parses", case.name));
        assert_eq!(error.reason(), case.reason, "refusal {} label", case.name);
    }
}
