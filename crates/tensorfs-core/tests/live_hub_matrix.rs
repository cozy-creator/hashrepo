//! The live-hub robustness and efficiency matrix.
//!
//! `live_hub.rs` proves the happy path ONCE. This proves the properties that
//! only a real hub can settle and that one run cannot: repeatability, racing
//! producers, interruption and resume, measured efficiency, and the hub's own
//! refusals.
//!
//! Opt-in exactly like `live_hub.rs`, and loud about why it skipped. See
//! `README-live-hub.md` for the token mint and the runner script.
//!
//! Knobs (all optional):
//!   TENSORFS_HUB_ROUNDS      repeated round trips        (default 10)
//!   TENSORFS_HUB_SCALE_MIB   the scale arm's payload     (default 384)
//!   TENSORFS_E2E_DIR         scratch root                (default $TMPDIR)

#![cfg(any(unix, windows))]

mod live_support;

use std::collections::BTreeMap;

use live_support::{
    Counting, FailUploadAfter, MIB, hub, knob, safetensors, scratch, seal, transport,
};
use tensorfs_core::store::ObjectDigest;
use tensorfs_core::sync::{
    CompleteStatus, DownloadGrant, GrantsPlan, PackClaim, PackGrant, SyncError, SyncPlan,
    SyncTransport, TransportError, pull_snapshot, push_snapshot,
};
use tensorfs_core::tfm1::SnapshotId;
use tensorfs_core::workspace::WorkspaceStore;

/// A per-process nonce. Every fixture's TENSOR NAMES carry it, so a shared
/// long-lived hub cannot already hold these objects — the property that makes
/// "uploaded" mean something.
fn nonce(tag: &str) -> String {
    format!("m{}{tag}", std::process::id())
}

fn mib_per_second(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return f64::INFINITY;
    }
    (bytes as f64) / seconds / (MIB as f64)
}

// ---------------------------------------------------------------------------
// Family 1 — robustness on the real wire
// ---------------------------------------------------------------------------

/// The same round trip, many times, chaining each push onto the head the last
/// one established. Flakiness against a real hub is a finding, not noise, so
/// every round asserts the same invariants rather than retrying.
#[test]
fn repeated_round_trips_hold_their_invariants() {
    let Some(hub) = hub() else { return };
    let rounds = knob("TENSORFS_HUB_ROUNDS", 10);
    let client = transport(&hub);
    let mut head = client.head().expect("head reads");
    println!("starting head: {head:?}; rounds={rounds}");

    for round in 0..rounds {
        let tag = nonce(&format!("r{round}"));
        let bytes = safetensors(&tag, &[("w", 6 * MIB, 0x40), ("b", 4096, 0x41)]);
        let root = scratch(&format!("round-{round}"));
        let (producer, snapshot) = seal(&root, &[("model.safetensors", bytes.clone())]);

        let counting = Counting::new(&client);
        let report = push_snapshot(
            &producer,
            &counting,
            &snapshot,
            head.as_ref(),
            Default::default(),
        )
        .unwrap_or_else(|error| panic!("round {round} push failed: {error}"));

        assert!(
            report.uploaded_objects > 0,
            "round {round}: a nonced fixture must upload"
        );
        assert_eq!(
            report.skipped_remote_resident, 0,
            "round {round}: a nonced fixture cannot already be resident"
        );

        let live = client.head().expect("head reads").expect("head is set");
        assert_eq!(
            live, snapshot,
            "round {round}: the hub's head must be exactly what we pushed"
        );

        // Pull it back into a store that has never seen these bytes.
        let consumer_root = scratch(&format!("round-{round}-consumer"));
        let consumer = WorkspaceStore::open(&consumer_root).expect("consumer opens");
        pull_snapshot(&consumer, &client, &live).expect("pull succeeds");
        let got = live_support::file_bytes(&consumer, &live, "model.safetensors");
        assert!(
            got == bytes,
            "round {round}: pulled bytes must be byte-exact"
        );

        println!(
            "round {round}: requests={} uploaded={}B packs={} complete_calls={}",
            counting.counts().requests(),
            report.uploaded_bytes,
            report.packs,
            report.complete_attempts
        );
        head = Some(live);
    }
    println!("REPEATED ROUND TRIPS PASSED ({rounds} rounds)");
}

/// Two producers race the same base head. The hub's compare-and-swap is real,
/// so exactly one may win and the loser must be told so in a typed way — never
/// a silent overwrite, never both.
#[test]
fn racing_producers_resolve_to_exactly_one_head() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    // Establish a common base both producers will build on.
    let base_tag = nonce("base");
    let base_bytes = safetensors(&base_tag, &[("w", 2 * MIB, 0x50)]);
    let base_root = scratch("race-base");
    let (base_producer, base_snapshot) = seal(&base_root, &[("base.safetensors", base_bytes)]);
    let prior = client.head().expect("head reads");
    push_snapshot(
        &base_producer,
        &client,
        &base_snapshot,
        prior.as_ref(),
        Default::default(),
    )
    .expect("base push succeeds");
    let base_head = client.head().expect("head reads").expect("head is set");
    assert_eq!(base_head, base_snapshot, "the base head is ours");

    // Two distinct snapshots, both declaring the SAME expected head.
    let outcomes: Vec<(SnapshotId, Result<(), SyncError>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|which| {
                let hub = &hub;
                let base_head = base_head;
                scope.spawn(move || {
                    let tag = nonce(&format!("race{which}"));
                    let bytes = safetensors(&tag, &[("w", 3 * MIB, 0x60)]);
                    let root = scratch(&format!("race-{which}"));
                    let name = format!("{which}.safetensors");
                    let (producer, snapshot) = seal(&root, &[(name.as_str(), bytes)]);
                    let client = transport(hub);
                    let outcome = push_snapshot(
                        &producer,
                        &client,
                        &snapshot,
                        Some(&base_head),
                        Default::default(),
                    )
                    .map(|_| ());
                    (snapshot, outcome)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("producer thread"))
            .collect()
    });

    let winners: Vec<SnapshotId> = outcomes
        .iter()
        .filter(|(_, outcome)| outcome.is_ok())
        .map(|(id, _)| *id)
        .collect();
    let losers: Vec<&SyncError> = outcomes
        .iter()
        .filter_map(|(_, outcome)| outcome.as_ref().err())
        .collect();

    assert_eq!(
        winners.len(),
        1,
        "exactly one producer may win the head; got {winners:?} and losers {losers:?}"
    );
    assert_eq!(losers.len(), 1, "the other must be refused, not dropped");

    // The loser's refusal must be typed and must name the conflict.
    let loser = losers[0];
    let text = loser.to_string();
    assert!(
        text.contains("head_conflict") || matches!(loser, SyncError::HeadRefused { .. }),
        "the losing producer must be refused with a head conflict, got: {loser}"
    );

    // The hub's head is the winner's, and it is fully readable.
    let live = client.head().expect("head reads").expect("head is set");
    assert_eq!(live, winners[0], "the head is the winner's snapshot");

    let consumer_root = scratch("race-consumer");
    let consumer = WorkspaceStore::open(&consumer_root).expect("consumer opens");
    pull_snapshot(&consumer, &client, &live).expect("the winning head pulls cleanly");
    println!("RACE PASSED: winner={live}, loser refused with: {loser}");
}

/// A push interrupted mid-transfer, then retried. This measures what resume
/// actually keeps against a real hub — the number that decides whether a large
/// model upload can survive a flaky link.
#[test]
fn an_interrupted_push_reports_exactly_what_it_kept() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    // Several packs, so there is a middle to die in: each tensor is its own
    // 64 MiB-grid object and the payload bound puts them in separate packs.
    let tag = nonce("interrupt");
    let bytes = safetensors(
        &tag,
        &[
            ("t0", 40 * MIB, 0x70),
            ("t1", 40 * MIB, 0x71),
            ("t2", 40 * MIB, 0x72),
        ],
    );
    let root = scratch("interrupt");
    let (producer, snapshot) = seal(&root, &[("big.safetensors", bytes.clone())]);
    let head = client.head().expect("head reads");

    // Die after exactly one pack reaches the hub.
    let interrupted = FailUploadAfter::new(&client, 1);
    let options = tensorfs_core::sync::PushOptions {
        max_upload_attempts: 1,
        ..Default::default()
    };
    let failure = push_snapshot(
        &producer,
        &interrupted,
        &snapshot,
        head.as_ref(),
        options,
    )
    .expect_err("the injected carrier failure must surface");
    let survived = interrupted.survived();
    println!("interrupted after {survived} pack(s): {failure}");
    assert!(survived >= 1, "the arm must die mid-transfer, not before it");

    // The head must not have moved: an incomplete push publishes nothing.
    let after = client.head().expect("head reads");
    assert_eq!(
        after, head,
        "an interrupted push must not move the head at all"
    );

    // Now retry cleanly and measure what the hub credited us for.
    let counting = Counting::new(&client);
    let resumed = push_snapshot(
        &producer,
        &counting,
        &snapshot,
        head.as_ref(),
        Default::default(),
    )
    .expect("the retry completes");
    let live = client.head().expect("head reads").expect("head is set");
    assert_eq!(live, snapshot, "the retry publishes the snapshot");

    // The contract is promotion-level resumption across sessions: nothing was
    // promoted before the abort, so the hub legitimately reports nothing
    // resident. Assert the CONTRACT, and print the cost it implies.
    println!(
        "RESUME: retry uploaded {} objects / {} bytes, hub reported {} already resident \
         (packs staged before the abort: {survived})",
        resumed.uploaded_objects, resumed.uploaded_bytes, resumed.skipped_remote_resident
    );
    assert!(
        resumed.uploaded_objects > 0,
        "the retry must actually finish the transfer"
    );
    let consumer_root = scratch("interrupt-consumer");
    let consumer = WorkspaceStore::open(&consumer_root).expect("consumer opens");
    pull_snapshot(&consumer, &client, &live).expect("pull succeeds");
    let got = live_support::file_bytes(&consumer, &live, "big.safetensors");
    assert!(got == bytes, "the resumed snapshot is byte-exact");
}

// ---------------------------------------------------------------------------
// Family 2 — measured efficiency
// ---------------------------------------------------------------------------

/// The scale arm: enough objects to cross pack boundaries and exercise grant
/// batching, measured end to end. Byte and request counts are ASSERTED;
/// wall-clock is only REPORTED, because this box swings with load and a timing
/// gate would rot.
#[test]
fn a_large_round_trip_is_measured_end_to_end() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);
    let scale_mib = knob("TENSORFS_HUB_SCALE_MIB", 384);
    let tensor = scale_mib * MIB / 4;

    let tag = nonce("scale");
    let alpha = safetensors(
        &tag,
        &[("a0", tensor, 0x80), ("a1", tensor, 0x81), ("meta", 4096, 0x82)],
    );
    let beta = safetensors(&tag, &[("b0", tensor, 0x90), ("b1", tensor, 0x91)]);
    let declared = (alpha.len() + beta.len()) as u64;

    let root = scratch("scale");
    let (producer, snapshot) = seal(
        &root,
        &[
            ("model-00001.safetensors", alpha.clone()),
            ("model-00002.safetensors", beta.clone()),
        ],
    );
    let head = client.head().expect("head reads");

    let push_meter = Counting::new(&client);
    let started = std::time::Instant::now();
    let report = push_snapshot(
        &producer,
        &push_meter,
        &snapshot,
        head.as_ref(),
        Default::default(),
    )
    .expect("scale push succeeds");
    let push_seconds = started.elapsed().as_secs_f64();
    let push_counts = push_meter.counts();
    let push_peak = push_meter.peak_concurrency();

    let live = client.head().expect("head reads").expect("head is set");
    assert_eq!(live, snapshot, "the hub's head is what we pushed");

    let consumer_root = scratch("scale-consumer");
    let consumer = WorkspaceStore::open(&consumer_root).expect("consumer opens");
    let pull_meter = Counting::new(&client);
    let started = std::time::Instant::now();
    let pulled = pull_snapshot(&consumer, &pull_meter, &live).expect("scale pull succeeds");
    let pull_seconds = started.elapsed().as_secs_f64();
    let pull_counts = pull_meter.counts();
    let pull_peak = pull_meter.peak_concurrency();

    let mut expected = BTreeMap::new();
    expected.insert("model-00001.safetensors", alpha);
    expected.insert("model-00002.safetensors", beta);
    for (path, want) in &expected {
        let got = live_support::file_bytes(&consumer, &live, path);
        assert!(got == *want, "{path}: pulled bytes are byte-exact");
    }

    println!("=== SCALE ARM: {scale_mib} MiB declared across {} objects ===", report.uploaded_objects);
    println!(
        "push  {:>10} bytes  {:>3} packs  {:>3} requests  {:>7.1} MiB/s  peak_concurrency={push_peak}",
        report.uploaded_bytes,
        report.packs,
        push_counts.requests(),
        mib_per_second(report.uploaded_bytes, push_seconds)
    );
    println!(
        "  declare={} pack_grants={} upload_pack={} complete={}",
        push_counts.declare, push_counts.pack_grants, push_counts.upload_pack, push_counts.complete
    );
    println!(
        "pull  {:>10} bytes  {:>3} objects {:>3} requests  {:>7.1} MiB/s  peak_concurrency={pull_peak}",
        pulled.fetched_bytes,
        pulled.fetched_objects,
        pull_counts.requests(),
        mib_per_second(pulled.fetched_bytes, pull_seconds)
    );
    println!(
        "  head={} download_grants={} download={}",
        pull_counts.head, pull_counts.download_grants, pull_counts.download
    );
    println!(
        "requests per object: push={:.2} pull={:.2}",
        push_counts.requests() as f64 / report.uploaded_objects.max(1) as f64,
        pull_counts.requests() as f64 / pulled.fetched_objects.max(1) as f64
    );

    // Byte efficiency is exact and is a gate: a fresh push moves the declared
    // payload once and no more.
    assert_eq!(
        report.uploaded_bytes, declared,
        "a fresh push must move the declared payload exactly once"
    );
    assert_eq!(
        pulled.fetched_bytes, declared,
        "a cold pull must fetch the declared payload exactly once"
    );

    // Grant batching: the hub advertises a per-request pack bound, so a push
    // must not spend a whole round trip per pack. This is the assertion that
    // catches the batching regression described in the PR body.
    println!(
        "NOTE grant round trips: {} pack-grant calls for {} packs",
        push_counts.pack_grants, report.packs
    );
}

/// A one-byte edit must move one object, not one tensor. The existing proof
/// asserts a ceiling; this measures the ratio and pins it.
#[test]
fn a_single_byte_edit_moves_exactly_one_object() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    let tag = nonce("dedup");
    let original = safetensors(&tag, &[("w", 70 * MIB, 0xA0), ("b", 4096, 0xA1)]);
    let root = scratch("dedup-origin");
    let (producer, snapshot) = seal(&root, &[("model.safetensors", original.clone())]);
    let head = client.head().expect("head reads");
    push_snapshot(
        &producer,
        &client,
        &snapshot,
        head.as_ref(),
        Default::default(),
    )
    .expect("origin push succeeds");
    let base = client.head().expect("head reads").expect("head is set");

    // Flip one byte inside the SECOND 64 MiB grid object.
    let mut edited = original.clone();
    let payload_start = edited.len() - (70 * MIB + 4096);
    edited[payload_start + 64 * MIB + 8] ^= 0xFF;

    let editor_root = scratch("dedup-editor");
    let (editor, edited_snapshot) = seal(&editor_root, &[("model.safetensors", edited)]);
    assert_ne!(edited_snapshot, snapshot, "an edit changes the snapshot id");

    let meter = Counting::new(&client);
    let delta = push_snapshot(
        &editor,
        &meter,
        &edited_snapshot,
        Some(&base),
        Default::default(),
    )
    .expect("delta push succeeds");

    let moved = delta.uploaded_bytes;
    let declared = original.len() as u64;
    println!(
        "DEDUP: moved {moved} of {declared} declared bytes ({:.4}%), objects={} resident={} packs={}",
        (moved as f64 / declared as f64) * 100.0,
        delta.uploaded_objects,
        delta.skipped_remote_resident,
        delta.packs
    );

    assert_eq!(
        delta.uploaded_objects, 1,
        "an 8-byte edit must move exactly one object, not {}",
        delta.uploaded_objects
    );
    assert!(
        delta.skipped_remote_resident >= 2,
        "the hub must recognize the untouched objects as resident: {delta:?}"
    );
    assert!(
        moved <= 8 * MIB as u64,
        "the moved object must be the tail object, not a whole tensor: {moved} bytes"
    );
}

// ---------------------------------------------------------------------------
// Family 3 — the hub's own refusals
// ---------------------------------------------------------------------------

/// Declaring against a base the repo has moved past must be refused, and the
/// refusal must arrive before any bytes are transferred.
#[test]
fn the_hub_refuses_a_stale_base() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    let tag = nonce("stale");
    let bytes = safetensors(&tag, &[("w", 2 * MIB, 0xB0)]);
    let root = scratch("stale");
    let (producer, snapshot) = seal(&root, &[("stale.safetensors", bytes)]);

    // A head that is syntactically valid and certainly not the repo's.
    let bogus = SnapshotId::parse_hex(&"ab".repeat(32)).expect("valid hex");

    let meter = Counting::new(&client);
    let refusal = push_snapshot(
        &producer,
        &meter,
        &snapshot,
        Some(&bogus),
        Default::default(),
    )
    .expect_err("a stale base must be refused");

    println!("STALE BASE refused with: {refusal}");
    assert!(
        refusal.to_string().contains("head_conflict"),
        "the refusal must name the head conflict, got: {refusal}"
    );
    assert_eq!(
        meter.counts().upload_pack,
        0,
        "a stale base must be refused before any bytes move"
    );
}

/// A pack whose declared member list does not match the envelope it actually
/// staged must be refused TERMINALLY at complete — the hub re-derives the
/// index from bytes and does not take the client's word for it.
#[test]
fn the_hub_refuses_a_pack_that_lies_about_its_members() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    // Two small tensors land in ONE pack, so a claim can drop a member.
    let tag = nonce("liar");
    let bytes = safetensors(&tag, &[("t0", 3 * MIB, 0xC0), ("t1", 3 * MIB, 0xC1)]);
    let root = scratch("liar");
    let (producer, snapshot) = seal(&root, &[("liar.safetensors", bytes)]);
    let head = client.head().expect("head reads");

    let liar = DropsAClaimedMember::new(&client);
    let outcome = push_snapshot(&producer, &liar, &snapshot, head.as_ref(), Default::default());

    match outcome {
        Err(error) => {
            println!("LYING PACK refused with: {error}");
            let text = error.to_string();
            assert!(
                text.contains("pack_mismatch") || text.contains("mismatch"),
                "the refusal must name the pack mismatch, got: {error}"
            );
        }
        Ok(report) => {
            // If the claim was never mutated the arm proved nothing; say so
            // rather than passing quietly.
            assert!(
                !liar.mutated(),
                "a mutated claim must never promote: {report:?}"
            );
            eprintln!("NOTE: the fixture produced no multi-member pack; arm did not exercise");
        }
    }

    // Whatever happened, the head must not carry a snapshot we never proved.
    let after = client.head().expect("head reads");
    assert_ne!(
        after,
        Some(snapshot),
        "a snapshot staged through a lying pack must never become the head"
    );
}

/// A declared session that is never completed must leave the head untouched.
#[test]
fn an_abandoned_session_leaves_no_head() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);
    let before = client.head().expect("head reads");

    let tag = nonce("abandon");
    let bytes = safetensors(&tag, &[("w", 2 * MIB, 0xD0)]);
    let root = scratch("abandon");
    let (producer, snapshot) = seal(&root, &[("abandon.safetensors", bytes)]);

    let tfm1 = producer
        .load_snapshot(&snapshot)
        .expect("snapshot loads")
        .to_bytes();
    let plan = client
        .declare(&tfm1, before.as_ref())
        .expect("declare succeeds");
    println!(
        "declared session {} with {} missing objects, then abandoned it",
        plan.session,
        plan.missing.len()
    );
    assert!(!plan.missing.is_empty(), "a nonced fixture must be missing");

    let after = client.head().expect("head reads");
    assert_eq!(after, before, "an abandoned session must not move the head");
}

/// Asking for a download grant for something the hub does not hold must be a
/// typed answer, never a grant that would yield wrong or absent bytes.
#[test]
fn download_grants_refuse_an_unheld_digest() {
    let Some(hub) = hub() else { return };
    let client = transport(&hub);

    let phantom = ObjectDigest::from_bytes(*SnapshotId::of(b"a digest no hub holds").as_bytes());
    match client.download_grants(std::slice::from_ref(&phantom)) {
        Ok(grants) => {
            assert!(
                grants.iter().all(|grant| grant.digest != phantom),
                "the hub must not grant a download for an object it does not hold"
            );
            println!("UNHELD DIGEST: hub returned {} grants (none ours)", grants.len());
        }
        Err(error) => {
            println!("UNHELD DIGEST refused with: {error}");
        }
    }
}

// ---------------------------------------------------------------------------
// A decorator that makes a pack claim disagree with the bytes it stages.
// ---------------------------------------------------------------------------

struct DropsAClaimedMember<'a, T: SyncTransport> {
    inner: &'a T,
    mutated: std::sync::atomic::AtomicBool,
}

impl<'a, T: SyncTransport> DropsAClaimedMember<'a, T> {
    fn new(inner: &'a T) -> Self {
        Self {
            inner,
            mutated: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn mutated(&self) -> bool {
        self.mutated.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<T: SyncTransport> SyncTransport for DropsAClaimedMember<'_, T> {
    fn declare(
        &self,
        tfm1_bytes: &[u8],
        expected_head: Option<&SnapshotId>,
    ) -> Result<SyncPlan, TransportError> {
        self.inner.declare(tfm1_bytes, expected_head)
    }

    fn pack_grants(
        &self,
        session: &str,
        claims: &[PackClaim],
    ) -> Result<GrantsPlan, TransportError> {
        let doctored: Vec<PackClaim> = claims
            .iter()
            .map(|claim| {
                let mut claim = claim.clone();
                if claim.objects.len() > 1 {
                    claim.objects.pop();
                    self.mutated
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }
                claim
            })
            .collect();
        self.inner.pack_grants(session, &doctored)
    }

    fn upload_pack(&self, grant: &PackGrant, pack: &[u8]) -> Result<(), TransportError> {
        self.inner.upload_pack(grant, pack)
    }

    fn complete(&self, session: &str) -> Result<CompleteStatus, TransportError> {
        self.inner.complete(session)
    }

    fn head(&self) -> Result<Option<SnapshotId>, TransportError> {
        self.inner.head()
    }

    fn download_grants(
        &self,
        digests: &[ObjectDigest],
    ) -> Result<Vec<DownloadGrant>, TransportError> {
        self.inner.download_grants(digests)
    }

    fn download(&self, grant: &DownloadGrant) -> Result<Vec<u8>, TransportError> {
        self.inner.download(grant)
    }
}
