//! AUTO-RATIFICATION, and what it does and does not prove.
//!
//! `ratify` proves a record is a WELL-FORMED map: every source element written
//! exactly once, padding zero, and the bytes that come back through the inverse
//! are the bytes that went in. It cannot prove a well-formed map is the RIGHT
//! one — any bijection round-trips — so the records that describe somebody
//! else's kernel are additionally checked here against an ORACLE written from
//! that kernel's own source, independently of the JSON. Both halves matter and
//! neither substitutes for the other.

use std::collections::BTreeMap;

use tensorfs_core::layout_morphism::{
    LayoutClass, LayoutMorphism, arrangement, catalog, probe_shapes, ratify,
};

fn record(handle: &str) -> &'static LayoutMorphism {
    arrangement(handle).unwrap_or_else(|error| panic!("{handle}: {error}"))
}

/// A shipped record, decoded to JSON so a test can change ONE index and watch
/// the guard fire. The digest is dropped: the Go loader checks it, and here it
/// would refuse every mutation before the guard under test ever ran.
fn mutable(handle: &str) -> serde_json::Value {
    let mut value = serde_json::to_value(SerializableRecord::from(record(handle))).unwrap();
    value.as_object_mut().unwrap().remove("digest");
    value
}

/// The serde shape of a record, for round-tripping one back out to JSON.
#[derive(serde::Serialize)]
struct SerializableRecord {
    format: String,
    name: String,
    version: u32,
    class: &'static str,
    description: String,
    rank: usize,
    sub_axes: Vec<serde_json::Value>,
    permutation: Vec<usize>,
    candidate: bool,
    provenance: Vec<String>,
}

impl From<&LayoutMorphism> for SerializableRecord {
    fn from(source: &LayoutMorphism) -> Self {
        Self {
            format: source.format.clone(),
            name: source.name.clone(),
            version: source.version,
            class: match source.class {
                LayoutClass::Inductor => "inductor",
                LayoutClass::EndpointDeclared => "endpoint-declared",
            },
            description: source.description.clone(),
            rank: source.rank,
            sub_axes: source
                .sub_axes
                .iter()
                .map(|sub| serde_json::json!({"axis": sub.axis, "extent": sub.extent}))
                .collect(),
            permutation: source.permutation.clone(),
            candidate: source.candidate,
            provenance: source.provenance.clone(),
        }
    }
}

fn reparse(value: &serde_json::Value) -> LayoutMorphism {
    serde_json::from_value(value.clone()).expect("a mutated record must still decode")
}

#[test]
fn every_catalogued_arrangement_auto_ratifies() {
    let records = catalog().expect("the vendored catalog must load");
    assert!(
        records.len() >= 7,
        "the v1 vocabulary is closed and has seven entries; the catalog carries {}",
        records.len()
    );
    let mut padded = 0;
    let mut exact = 0;
    for (handle, record) in records {
        let report = ratify(record).unwrap_or_else(|error| panic!("{handle}: {error}"));
        assert!(
            !report.probes.is_empty(),
            "{handle} proved nothing: no probe shape reached it"
        );
        for probe in &report.probes {
            if probe.padded {
                padded += 1;
            } else {
                exact += 1;
            }
        }
        println!("{handle}: {} probe shape(s) ratified", report.probes.len());
    }
    // Both halves of the language have to be exercised, or the padded arm of
    // the verifier is unproven code that happens to compile.
    assert!(exact > 0, "no exact arrangement was ratified");
    assert!(
        padded > 0,
        "no PADDED arrangement was ratified, so round-trip-on-the-image was \
         never the thing under test"
    );
}

#[test]
fn a_permutation_that_is_not_a_bijection_is_refused() {
    let mut document = mutable("torch.channels_last-2d@1");
    // The shipped record ratifies.
    ratify(&reparse(&document)).expect("the shipped record must ratify");
    // ONE index: [0, 2, 3, 1] -> [0, 2, 3, 3].
    document["permutation"] = serde_json::json!([0, 2, 3, 3]);
    let error = ratify(&reparse(&document)).expect_err("a non-bijection was ratified");
    assert!(
        error.to_string().contains("bijection"),
        "the refusal does not name what is wrong: {error}"
    );
}

#[test]
fn a_factorization_that_does_not_reach_its_axis_proves_nothing() {
    let mut document = mutable("cublas.blockscale-128x4@1");
    ratify(&reparse(&document)).expect("the shipped record must ratify");
    // ONE extent: the 32-row factor becomes 16, so the axis addresses half its
    // rows. Every probe shape is then refused — and a record that refuses every
    // probe is a record nothing was proved about, which is a failure and not a
    // pass. This is the arm that catches "the guard never ran".
    document["sub_axes"][2]["extent"] = serde_json::json!("16");
    let error = ratify(&reparse(&document)).expect_err("a short factorization was ratified");
    assert!(
        error.to_string().contains("nothing was proved"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn an_unratified_candidate_never_derives() {
    let mut document = mutable("torch.channels_last-2d@1");
    document["candidate"] = serde_json::json!(true);
    let wish = reparse(&document);
    // It still VERIFIES — a candidate is well-formed data, not garbage.
    ratify(&wish).expect("a candidate must still auto-verify");
    // And it still refuses to derive.
    let error = wish
        .ensure_ratified()
        .expect_err("an unratified candidate was allowed to derive");
    assert!(error.to_string().contains("candidate"), "{error}");
    // Nothing in the shipped catalog is a candidate today; if that changes it
    // is a deliberate act and this line is where it is noticed.
    for (handle, record) in catalog().unwrap() {
        assert!(record.ensure_ratified().is_ok(), "{handle} ships unratified");
    }
}

/// The oracle for `torch.channels_last-2d@1`, written from torch's definition
/// of the memory format rather than from the record: NCHW logical, NHWC in
/// storage. If the record's permutation drifted, these two disagree.
#[test]
fn channels_last_matches_an_independent_nhwc_oracle() {
    let shape = [2u64, 3, 4, 5];
    let plan = record("torch.channels_last-2d@1").plan(&shape).unwrap();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let count = (n * c * h * w) as usize;

    let source: Vec<u8> = (0..count as u32).flat_map(|at| at.to_le_bytes()).collect();
    let mut destination = vec![0u8; count * 4];
    plan.apply(&source, &mut destination, 4).unwrap();

    let mut expected = vec![0u8; count * 4];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let to = (((ni * h + hi) * w + wi) * c + ci) as usize;
                    let from = (((ni * c + ci) * h + hi) * w + wi) as usize;
                    expected[to * 4..to * 4 + 4]
                        .copy_from_slice(&source[from * 4..from * 4 + 4]);
                }
            }
        }
    }
    assert_eq!(destination, expected, "the record is not torch's channels_last");
}

/// The oracle for `cublas.blockscale-128x4@1`, written from python-gen-worker
/// `nvfp4_quant.to_blocked_scales`' own reshape/permute chain:
///
///   padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
///        .reshape(-1, 4, 32, 4).transpose(1, 2)
///
/// which puts destination index `(rb*ncb + cb)*512 + b32*16 + a4*4 + c4` at
/// source row `rb*128 + a4*32 + b32`, column `cb*4 + c4`. The record is a
/// transcription of that chain; this test is a second transcription of it,
/// and a transcription slip makes the two disagree.
#[test]
fn the_blocked_scale_record_matches_the_packer_it_was_transcribed_from() {
    // Deliberately RAGGED: 100 rows and 3 block-columns both round up, so the
    // padding path is what is compared, not just the aligned interior.
    let (rows, cols) = (100u64, 3u64);
    let plan = record("cublas.blockscale-128x4@1")
        .plan(&[rows, cols])
        .unwrap();
    let count = (rows * cols) as usize;
    let source: Vec<u8> = (0..count as u32)
        .flat_map(|at| (at + 1).to_le_bytes())
        .collect();
    let mut destination = vec![0u8; plan.dest_elements() as usize * 4];
    plan.apply(&source, &mut destination, 4).unwrap();

    let nrb = rows.div_ceil(128);
    let ncb = cols.div_ceil(4);
    let mut expected = vec![0u8; (nrb * 128 * ncb * 4) as usize * 4];
    for rb in 0..nrb {
        for cb in 0..ncb {
            for a4 in 0..4u64 {
                for b32 in 0..32u64 {
                    for c4 in 0..4u64 {
                        let to = ((rb * ncb + cb) * 512 + b32 * 16 + a4 * 4 + c4) as usize;
                        let row = rb * 128 + a4 * 32 + b32;
                        let col = cb * 4 + c4;
                        if row < rows && col < cols {
                            let from = (row * cols + col) as usize;
                            expected[to * 4..to * 4 + 4]
                                .copy_from_slice(&source[from * 4..from * 4 + 4]);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        destination.len(),
        expected.len(),
        "the record and the packer disagree on the destination size"
    );
    assert_eq!(
        destination, expected,
        "the record does not reproduce to_blocked_scales"
    );
}

/// THE RED ARM OF BOTH ORACLES: a record that is still a perfect bijection —
/// so `ratify` accepts it, and must — but is the WRONG permutation. This is
/// exactly the failure auto-ratification cannot see and the reason these
/// records carry a cited source and an oracle test.
#[test]
fn a_bijective_but_wrong_permutation_survives_ratification_and_fails_the_oracle() {
    let mut document = mutable("cublas.blockscale-128x4@1");
    // Swap the two 4-extent sub-axes' storage positions: [0,3,2,1,4] ->
    // [0,3,2,4,1]. Still a bijection; still round-trips; different bytes.
    document["permutation"] = serde_json::json!([0, 3, 2, 4, 1]);
    let wrong = reparse(&document);
    ratify(&wrong).expect(
        "a wrong-but-bijective permutation MUST still auto-ratify — if this \
         fails, the claim that ratification proves correctness is being made \
         somewhere it should not be",
    );

    let (rows, cols) = (100u64, 3u64);
    let plan = wrong.plan(&[rows, cols]).unwrap();
    let right = record("cublas.blockscale-128x4@1").plan(&[rows, cols]).unwrap();
    let count = (rows * cols) as usize;
    let source: Vec<u8> = (0..count as u32)
        .flat_map(|at| (at + 1).to_le_bytes())
        .collect();
    let mut mangled = vec![0u8; plan.dest_elements() as usize * 4];
    let mut correct = vec![0u8; right.dest_elements() as usize * 4];
    plan.apply(&source, &mut mangled, 4).unwrap();
    right.apply(&source, &mut correct, 4).unwrap();
    assert_ne!(
        mangled, correct,
        "two different permutations produced the same bytes, so the oracle \
         tests above are not measuring the permutation"
    );
}

/// The cross-LANGUAGE guard. Go decides with these records and this crate moves
/// the bytes with them, which is two evaluators of one shape language. They are
/// held together by banked vectors: this test writes them, the Go suite checks
/// them, and a drift in either evaluator fails on one side or the other.
#[test]
fn plan_vectors_agree_with_the_bank() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/v2/vectors/layout-plans.json"
    );
    let mut plans = Vec::new();
    for (handle, record) in catalog().unwrap() {
        for shape in probe_shapes(record) {
            let Ok(plan) = record.plan(&shape) else {
                continue;
            };
            plans.push(serde_json::json!({
                "layout": handle,
                "shape": shape,
                "source_elements": plan.source_elements(),
                "dest_elements": plan.dest_elements(),
                "padded": plan.padded(),
                "storage_extents": plan.storage_extents(),
            }));
        }
    }
    let banked = serde_json::json!({
        "format": "tensorfs-layout-plan-vectors-v1",
        "note": "Generated by crates/tensorfs-core/tests/layout_morphism.rs. \
                 Checked by the Go suite against its own evaluator: these are the \
                 two implementations of one shape language agreeing in writing.",
        "plans": plans,
    });
    let rendered = serde_json::to_string_pretty(&banked).unwrap() + "\n";
    if std::env::var("TENSORFS_REBANK").is_ok() {
        std::fs::write(path, &rendered).expect("rebank layout-plans.json");
        return;
    }
    let existing = std::fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("{path}: {error}. Run with TENSORFS_REBANK=1 to write it.")
    });
    let left: serde_json::Value = serde_json::from_str(&existing).unwrap();
    assert_eq!(
        left, banked,
        "the banked plan vectors and this crate's evaluator disagree"
    );
}

/// Every record's probe set is derived from the record, so this is a guard on
/// the DERIVATION: a vocabulary entry whose alignment is misread would silently
/// be probed at a shape that proves less.
#[test]
fn probe_shapes_are_derived_and_cover_both_arms() {
    let mut shapes = BTreeMap::new();
    for (handle, record) in catalog().unwrap() {
        let probes = probe_shapes(record);
        assert!(!probes.is_empty(), "{handle} has no probe shape");
        for probe in &probes {
            if record.rank != 0 {
                assert_eq!(
                    probe.len(),
                    record.rank,
                    "{handle}: a rank-{} record was probed at {probe:?}",
                    record.rank
                );
            }
        }
        shapes.insert(handle.clone(), probes);
    }
    // The exact-division records must refuse their ragged probe; the padded
    // ones must accept it. If every record accepted every probe, the "skip a
    // refused shape" path in `ratify` would be dead code.
    let exact = record("nunchaku.micro-scale@1");
    let ragged = shapes["nunchaku.micro-scale@1"].last().unwrap();
    assert!(
        exact.plan(ragged).is_err(),
        "an exact factorization accepted a ragged shape {ragged:?}"
    );
    let padded = record("cublas.blockscale-128x4@1");
    let ragged = shapes["cublas.blockscale-128x4@1"].last().unwrap();
    assert!(
        padded.plan(ragged).is_ok(),
        "a padded factorization refused a ragged shape {ragged:?}"
    );
}
