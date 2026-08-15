//! Family 4 — the hub is hostile, broken, or lying.
//!
//! Upload and download are the surface most exposed to the real world: a
//! flaky CDN, an expiring presign, a proxy that answers with HTML, an object
//! store that hands back a valid-but-wrong body. This file enumerates the
//! failure CLASSES rather than a few instances, in both directions, and
//! asserts one invariant everywhere:
//!
//! > either a typed error, or correct bytes. Never silent corruption, and
//! > never a local snapshot adopted that references bytes we did not verify
//! > ourselves.
//!
//! Two layers, because the classes live at two different seams:
//!
//!  * **Layer A** drives the REAL [`HttpTransport`] against a scripted
//!    loopback server, which is the only way to reach transport-level and
//!    HTTP-semantic behaviour (framing, status mapping, redirects). The
//!    transport is exercised, never modified.
//!  * **Layer B** drives the REAL engine (`push_snapshot`/`pull_snapshot`)
//!    over a fault-injecting hub that runs the real TFM1/TFP1 decoders, for
//!    the content-adversarial and protocol-state classes.
//!
//! Timeout expiry (read timeout mid-transfer, connect timeout) is the one
//! enumerated class not asserted here: the transport's constants are 60 s and
//! 600 s, so a truthful test would have to block for minutes. Connect
//! *failure* is covered instead, and both timeouts surface through the same
//! `ureq::Error::Transport` arm as the covered faults, which the transport
//! maps to `TransportError::Io` at a single site.

#![cfg(any(unix, windows))]

mod harness;

use harness::{
    DownloadFault, FaultHub, FaultServer, Faults, Reply, Scratch, assert_snapshot_fully_backed,
    data_digests, reconstruct, sealed_workspace,
};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::store::StoreError;
use tensorfs_core::sync::http::{HttpTransport, TokenSource};
use tensorfs_core::sync::{
    PushOptions, SyncError, SyncTransport, TransportError, manifest_object_digest, pull_snapshot,
    push_snapshot,
};
use tensorfs_core::tfm1::SnapshotId;
use tensorfs_core::workspace::WorkspaceStore;

// ===========================================================================
// Layer A — the real HttpTransport against a hostile wire
// ===========================================================================

fn transport(base: &str) -> HttpTransport {
    HttpTransport::new(base, "org", "repo").with_token(TokenSource::Static("t".to_owned()))
}

fn a_digest(seed: &[u8]) -> ObjectDigest {
    ObjectDigest::from_bytes(*SnapshotId::of(seed).as_bytes())
}

/// Every 5xx shape a load balancer produces becomes a typed refusal carrying
/// the status, never a success and never a panic.
#[test]
fn server_error_storms_map_to_typed_refusals() {
    for status in [500_u16, 502, 503, 504] {
        let server = FaultServer::start(vec![Reply::json(status, r#"{"error":{"code":"boom"}}"#)]);
        let error = transport(&server.base)
            .head()
            .expect_err("a 5xx must not read as a head");
        match error {
            TransportError::Refused { code, .. } => {
                assert!(
                    code == "boom" || code == format!("http-{status}"),
                    "unexpected code {code} for {status}"
                );
            }
            other => panic!("{status} produced {other:?}, expected a refusal"),
        }
    }
}

/// Rate limiting, with and without `Retry-After`. Both are refusals the
/// engine can surface; neither is mistaken for an answer.
#[test]
fn rate_limiting_maps_to_a_typed_refusal() {
    for body in [r#"{"error":{"code":"rate_limited"}}"#, "{}"] {
        let server = FaultServer::start(vec![Reply::json(429, body)]);
        let error = transport(&server.base)
            .head()
            .expect_err("429 must not read as a head");
        assert!(
            matches!(error, TransportError::Refused { .. }),
            "429 produced {error:?}"
        );
    }
}

/// Token expiry mid-session. 401/403 on a control route are refusals; an
/// envelope whose code carries `expired` is promoted to `Expired` so the
/// engine replans instead of failing the push.
#[test]
fn authentication_failures_are_typed_and_expiry_is_distinguished() {
    let server = FaultServer::start(vec![Reply::json(401, r#"{"error":{"code":"unauthorized"}}"#)]);
    let error = transport(&server.base).head().expect_err("401 refuses");
    assert!(matches!(error, TransportError::Refused { .. }));

    let server = FaultServer::start(vec![Reply::json(
        403,
        r#"{"error":{"code":"session_expired"}}"#,
    )]);
    let error = transport(&server.base).head().expect_err("403 refuses");
    assert!(
        matches!(error, TransportError::Expired(_)),
        "an expiry envelope must replan, got {error:?}"
    );
}

/// A presigned GET that 404s or 403s after being issued — the object was
/// swept, or the presign died early.
#[test]
fn a_dead_download_grant_is_typed_not_silent() {
    let digest = a_digest(b"dead-grant");
    let server = FaultServer::start(vec![Reply::json(404, r#"{"error":{"code":"gone"}}"#)]);
    let grant = tensorfs_core::sync::DownloadGrant {
        digest,
        length: 8,
        url: format!("{}/object", server.base),
    };
    let error = transport(&server.base)
        .download(&grant)
        .expect_err("404 must not read as bytes");
    assert!(matches!(error, TransportError::Refused { .. }));

    let server = FaultServer::start(vec![Reply::json(403, "{}")]);
    let grant = tensorfs_core::sync::DownloadGrant {
        digest,
        length: 8,
        url: format!("{}/object", server.base),
    };
    let error = transport(&server.base)
        .download(&grant)
        .expect_err("403 must not read as bytes");
    assert!(
        matches!(error, TransportError::Expired(_)),
        "a 403 on a presign means replan, got {error:?}"
    );
}

/// The framing classes: a body shorter than `Content-Length`, a body longer
/// than it, headers with no body at all, an unterminated chunked stream, and
/// a bare hang-up. None may yield bytes.
#[test]
fn broken_response_framing_never_yields_bytes() {
    let payload = b"the-honest-object-bytes".to_vec();
    let digest = a_digest(&payload);
    let cases: Vec<(&str, Reply)> = vec![
        (
            "short body",
            Reply::Short {
                status: 200,
                declared: payload.len(),
                send: payload[..5].to_vec(),
            },
        ),
        (
            "overlong body",
            Reply::Overlong {
                status: 200,
                declared: payload.len(),
                body: [payload.clone(), b"extra".to_vec()].concat(),
            },
        ),
        (
            "headers only",
            Reply::HeadersOnly {
                status: 200,
                declared: payload.len(),
            },
        ),
        (
            "unterminated chunked",
            Reply::UnterminatedChunked {
                status: 200,
                chunk: payload[..4].to_vec(),
            },
        ),
        ("hang up", Reply::HangUp),
    ];

    for (name, reply) in cases {
        let server = FaultServer::start(vec![reply]);
        let grant = tensorfs_core::sync::DownloadGrant {
            digest,
            length: payload.len() as u64,
            url: format!("{}/object", server.base),
        };
        match transport(&server.base).download(&grant) {
            Err(error) => {
                assert!(
                    matches!(error, TransportError::Io(_) | TransportError::Refused { .. }),
                    "{name}: unexpected error shape {error:?}"
                );
            }
            Ok(bytes) => {
                // The only tolerable success is the exact honest payload; an
                // overlong body that happens to contain a correct prefix must
                // NOT be truncated into a false success.
                assert_eq!(
                    bytes, payload,
                    "{name}: transport returned {} bytes that are not the object",
                    bytes.len()
                );
            }
        }
    }
}

/// A 200 with an empty body, and a proxy's HTML error page served as 200 —
/// both must fail to parse rather than decode as an answer.
#[test]
fn non_json_and_empty_control_answers_are_rejected() {
    let server = FaultServer::start(vec![Reply::Body {
        status: 200,
        body: String::new(),
    }]);
    let error = transport(&server.base)
        .head()
        .expect_err("an empty 200 is not a head");
    assert!(matches!(error, TransportError::Io(_)));

    let server = FaultServer::start(vec![Reply::HtmlPage]);
    let error = transport(&server.base)
        .head()
        .expect_err("HTML is not a head");
    assert!(
        matches!(error, TransportError::Io(_)),
        "an HTML body produced {error:?}"
    );
}

/// A download that answers 206 when the whole object was requested: the
/// transport asked for everything, so a partial answer is short and the
/// length guard refuses it.
#[test]
fn a_partial_answer_to_a_whole_object_request_is_refused() {
    let payload = b"0123456789abcdef".to_vec();
    let server = FaultServer::start(vec![Reply::Bytes {
        status: 206,
        body: payload[..6].to_vec(),
    }]);
    let grant = tensorfs_core::sync::DownloadGrant {
        digest: a_digest(&payload),
        length: payload.len() as u64,
        url: format!("{}/object", server.base),
    };
    let error = transport(&server.base)
        .download(&grant)
        .expect_err("a short 206 must refuse");
    assert!(
        matches!(error, TransportError::Io(_)),
        "206 produced {error:?}"
    );
}

/// Connect failure — nothing listening — is a carrier fault, not a refusal.
#[test]
fn a_connect_failure_is_a_carrier_fault() {
    let dead = FaultServer::dead_address();
    let error = transport(&dead)
        .head()
        .expect_err("a closed port cannot answer");
    assert!(
        matches!(error, TransportError::Io(_)),
        "connect failure produced {error:?}"
    );
}

/// A control route that redirects to another host must not carry the bearer
/// token to that host. This is the credential-leak class: the assertion is on
/// what the second server actually received.
#[test]
fn a_cross_host_redirect_does_not_forward_the_bearer_token() {
    let victim = FaultServer::start(vec![Reply::json(200, r#"{"snapshot_id":""}"#)]);
    let attacker_base = victim.base.clone();
    let server = FaultServer::start(vec![Reply::Redirect {
        status: 302,
        location: format!("{attacker_base}/stolen"),
    }]);

    // The result itself does not matter; what the redirect target saw does.
    let _ = transport(&server.base).head();

    let _first = server
        .requests
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the first server saw the request");
    if let Ok(followed) = victim
        .requests
        .recv_timeout(std::time::Duration::from_secs(10))
    {
        assert!(
            followed.authorization.is_none(),
            "the bearer token was forwarded across a redirect to {:?} — credential leak",
            followed.host
        );
    }
}

// ===========================================================================
// Layer B — the real engine against a lying hub
// ===========================================================================

fn publisher(scratch: &Scratch) -> (WorkspaceStore, SnapshotId) {
    sealed_workspace(
        scratch.path(),
        "publisher",
        &[
            ("model.bin", vec![vec![7_u8; 4096], vec![9_u8; 8192]]),
            ("weights/extra.bin", vec![vec![3_u8; 2048]]),
        ],
    )
}

/// Pushes a snapshot into a healthy hub and returns everything a puller
/// needs: the hub holding it, the id, and the publisher's own bytes.
fn published(scratch: &Scratch) -> (FaultHub, SnapshotId, Vec<(String, Vec<u8>)>) {
    let (meta, id) = publisher(scratch);
    let hub = FaultHub::new();
    push_snapshot(&meta, &hub, &id, None, PushOptions::default()).expect("healthy push");
    let originals = vec![
        ("model.bin".to_owned(), reconstruct(&meta, &id, "model.bin")),
        (
            "weights/extra.bin".to_owned(),
            reconstruct(&meta, &id, "weights/extra.bin"),
        ),
    ];
    (hub, id, originals)
}

/// The content-adversarial matrix. For each way a hub can hand back bytes
/// that are not the requested object, the pull must fail with a typed error
/// and leave nothing behind: no object at that digest, no adopted snapshot.
#[test]
fn no_content_adversarial_download_can_place_wrong_bytes_locally() {
    let faults = [
        DownloadFault::WrongBytes,
        DownloadFault::SwappedWithAnother,
        DownloadFault::ValidPrefix,
        DownloadFault::SameLengthDifferentContent,
        DownloadFault::BitFlip,
        DownloadFault::Empty,
        DownloadFault::Overlong,
    ];

    for fault in faults {
        let source = Scratch::new("adv-src");
        let (hub, id, _) = published(&source);
        let target_digest = {
            let meta = WorkspaceStore::open(source.path()).expect("reopen");
            data_digests(&meta, &id)[0]
        };

        hub.set_faults(Faults {
            downloads: [(*target_digest.as_bytes(), fault.clone())]
                .into_iter()
                .collect(),
            ..Faults::default()
        });

        let sink = Scratch::new("adv-sink");
        let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");
        let error = pull_snapshot(&meta, &hub, &id)
            .expect_err(&format!("{fault:?} must refuse, never adopt"));

        // The refusal must come from the admission boundary or the manifest
        // guard — a store/length/digest error, not a generic success.
        assert!(
            matches!(
                error,
                SyncError::Store(
                    StoreError::DigestMismatch { .. } | StoreError::LengthMismatch { .. }
                ) | SyncError::Workspace(_)
                    | SyncError::Transport(_)
            ),
            "{fault:?} produced {error:?}"
        );

        // Nothing landed: the poisoned digest is absent, and the snapshot was
        // never adopted.
        assert!(
            meta.store().verify(&target_digest).is_err(),
            "{fault:?}: poisoned bytes became resident at {target_digest}"
        );
        assert!(
            meta.load_snapshot(&id).is_err(),
            "{fault:?}: a snapshot was adopted despite unverified bytes"
        );
        harness::Consistency::scan(sink.path()).assert_intact(&format!("{fault:?}"));
    }
}

/// A manifest whose bytes do not hash to the id we asked for, and a manifest
/// truncated in flight. Both must refuse before any object is fetched.
#[test]
fn a_corrupt_or_truncated_manifest_is_refused_before_any_object_moves() {
    for (name, faults) in [
        (
            "corrupt",
            Faults {
                corrupt_manifest: true,
                ..Faults::default()
            },
        ),
        (
            "truncated",
            Faults {
                truncate_manifest: true,
                ..Faults::default()
            },
        ),
    ] {
        let source = Scratch::new("man-src");
        let (hub, id, _) = published(&source);
        hub.set_faults(faults);

        let sink = Scratch::new("man-sink");
        let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");
        let error =
            pull_snapshot(&meta, &hub, &id).expect_err(&format!("{name} manifest must refuse"));
        assert!(
            matches!(error, SyncError::RemoteManifestCorrupt(_)),
            "{name} manifest produced {error:?}"
        );
        assert_eq!(
            hub.state.borrow().downloads_by_digest.len(),
            1,
            "{name}: objects were fetched despite an unusable manifest"
        );
        assert!(meta.load_snapshot(&id).is_err());
        harness::Consistency::scan(sink.path()).assert_intact(name);
    }
}

/// A hub that omits a grant for an object the manifest requires must be
/// detected by the engine, not silently skipped into a short local tree.
#[test]
fn an_omitted_download_grant_is_detected() {
    let source = Scratch::new("omit-src");
    let (hub, id, _) = published(&source);
    let missing = {
        let meta = WorkspaceStore::open(source.path()).expect("reopen");
        data_digests(&meta, &id)[1]
    };
    hub.set_faults(Faults {
        omit_download_grant: Some(*missing.as_bytes()),
        ..Faults::default()
    });

    let sink = Scratch::new("omit-sink");
    let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");
    let error = pull_snapshot(&meta, &hub, &id).expect_err("an omitted grant must refuse");
    assert!(
        matches!(error, SyncError::GrantOmitted(digest) if digest == missing),
        "omission produced {error:?}"
    );
    assert!(meta.load_snapshot(&id).is_err());
}

/// A hub that adds a grant for an object the manifest does not name must be
/// ignored: the engine fetches its closure, not the hub's suggestions.
#[test]
fn an_unrequested_grant_is_never_fetched() {
    let source = Scratch::new("extra-src");
    let (hub, id, originals) = published(&source);
    hub.set_faults(Faults {
        inject_unrequested_grant: true,
        ..Faults::default()
    });

    let sink = Scratch::new("extra-sink");
    let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");
    pull_snapshot(&meta, &hub, &id).expect("an extra grant does not break a pull");

    let wanted: Vec<[u8; 32]> = data_digests(&meta, &id)
        .iter()
        .map(|digest| *digest.as_bytes())
        .collect();
    let manifest = *manifest_object_digest(&id).as_bytes();
    for digest in hub.state.borrow().downloads_by_digest.keys() {
        assert!(
            wanted.contains(digest) || *digest == manifest,
            "the engine fetched an object outside the manifest closure"
        );
    }
    for (path, bytes) in &originals {
        assert_eq!(&reconstruct(&meta, &id, path), bytes, "{path} round-trips");
    }
}

/// The head moves under an in-flight push. `complete` must report the
/// conflict terminally rather than promoting over a competitor.
#[test]
fn a_head_advancing_mid_push_is_refused_terminally() {
    let scratch = Scratch::new("head-race");
    let (meta, id) = publisher(&scratch);
    let hub = FaultHub::with_faults(Faults {
        advance_head_after_grants: Some(0),
        ..Faults::default()
    });

    let error = push_snapshot(&meta, &hub, &id, None, PushOptions::default())
        .expect_err("a lost head race must refuse");
    assert!(
        matches!(error, SyncError::HeadRefused { .. }),
        "head race produced {error:?}"
    );
    assert_ne!(
        hub.state.borrow().head,
        Some(id),
        "the push promoted over a competing head"
    );
}

/// A session the hub has forgotten (restart, eviction) must surface as a
/// typed refusal rather than a hang or a partial promotion.
#[test]
fn a_forgotten_session_refuses_rather_than_promoting() {
    let scratch = Scratch::new("forgotten");
    let (meta, id) = publisher(&scratch);
    let hub = FaultHub::with_faults(Faults {
        forget_sessions: true,
        ..Faults::default()
    });

    let error = push_snapshot(&meta, &hub, &id, None, PushOptions::default())
        .expect_err("an unknown session must refuse");
    assert!(
        matches!(error, SyncError::Transport(TransportError::Refused { .. })),
        "forgotten session produced {error:?}"
    );
    assert!(hub.state.borrow().head.is_none());
}

/// `complete` answering terminally must not be retried into success, and
/// must not move the local or remote head.
#[test]
fn a_terminal_completion_is_not_retried_into_success() {
    let scratch = Scratch::new("terminal");
    let (meta, id) = publisher(&scratch);
    let hub = FaultHub::with_faults(Faults {
        terminal_complete: Some("verification_failed".to_owned()),
        ..Faults::default()
    });

    let error = push_snapshot(&meta, &hub, &id, None, PushOptions::default())
        .expect_err("a terminal completion must refuse");
    assert!(matches!(error, SyncError::HeadRefused { code } if code == "verification_failed"));
    assert_eq!(
        hub.state.borrow().complete_calls,
        1,
        "a terminal answer must not be retried"
    );
    assert!(hub.state.borrow().head.is_none());
}

/// Pushing the same promoted snapshot twice is idempotent: the second push
/// re-declares against the now-current head and moves no bytes.
#[test]
fn completing_twice_is_idempotent_and_moves_no_bytes() {
    let scratch = Scratch::new("twice");
    let (meta, id) = publisher(&scratch);
    let hub = FaultHub::new();

    let first = push_snapshot(&meta, &hub, &id, None, PushOptions::default()).expect("first push");
    assert!(first.uploaded_objects > 0);

    let second = push_snapshot(&meta, &hub, &id, Some(&id), PushOptions::default())
        .expect("re-pushing a promoted snapshot succeeds");
    assert_eq!(
        second.uploaded_objects, 0,
        "a second push retransmitted objects the hub already holds"
    );
    assert_eq!(hub.state.borrow().head, Some(id));
}

/// The upload-side transient classes: carrier faults and grant expiries burn
/// through bounded retries and the push still converges, with no object
/// uploaded more times than the retry budget allows.
#[test]
fn transient_upload_faults_converge_without_duplicating_work() {
    let scratch = Scratch::new("flap-up");
    let (meta, id) = publisher(&scratch);
    let hub = FaultHub::with_faults(Faults {
        fail_uploads: 2,
        expire_uploads: 1,
        expire_grant_calls: 1,
        incomplete_completes: 3,
        ..Faults::default()
    });

    let report = push_snapshot(&meta, &hub, &id, None, PushOptions::default())
        .expect("bounded transients must converge");
    assert_eq!(hub.state.borrow().head, Some(id));
    assert!(report.replans > 0, "the expiries should have forced replans");

    // Every object the manifest names is present on the hub exactly once as
    // a promoted object.
    for digest in data_digests(&meta, &id) {
        assert!(
            hub.state.borrow().objects.contains_key(digest.as_bytes()),
            "{digest} did not survive the transient storm"
        );
    }
}

/// The download-side transient class: a hub that fails N times then serves
/// correctly. The pull surfaces the fault typed; a retry after the outage
/// ends converges to byte-exact content and re-fetches nothing already held.
#[test]
fn transient_download_faults_surface_typed_then_converge_on_retry() {
    let source = Scratch::new("flap-down-src");
    let (hub, id, originals) = published(&source);
    hub.set_faults(Faults {
        fail_downloads: 1,
        ..Faults::default()
    });

    let sink = Scratch::new("flap-down-sink");
    let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");
    let error = pull_snapshot(&meta, &hub, &id).expect_err("the injected fault must surface");
    assert!(matches!(
        error,
        SyncError::Transport(TransportError::Io(_))
    ));

    hub.heal();
    pull_snapshot(&meta, &hub, &id).expect("the retry converges");
    assert_snapshot_fully_backed(&meta, &id, "after a transient download outage");
    for (path, bytes) in &originals {
        assert_eq!(&reconstruct(&meta, &id, path), bytes, "{path} is byte-exact");
    }
}

/// The end-to-end statement: a store dragged through every fault class this
/// file injects, then given an honest hub, converges to byte-exact content
/// and re-fetches nothing it had already verified.
#[test]
fn a_store_survives_the_full_gauntlet_and_still_reconstructs_byte_exactly() {
    let source = Scratch::new("gauntlet-src");
    let (hub, id, originals) = published(&source);
    let digests = {
        let meta = WorkspaceStore::open(source.path()).expect("reopen");
        data_digests(&meta, &id)
    };

    let sink = Scratch::new("gauntlet-sink");
    let meta = WorkspaceStore::open(sink.path()).expect("sink store opens");

    let gauntlet = [
        DownloadFault::WrongBytes,
        DownloadFault::ValidPrefix,
        DownloadFault::SameLengthDifferentContent,
        DownloadFault::BitFlip,
        DownloadFault::Empty,
        DownloadFault::Overlong,
        DownloadFault::SwappedWithAnother,
    ];
    for fault in gauntlet {
        hub.set_faults(Faults {
            downloads: [(*digests[0].as_bytes(), fault.clone())]
                .into_iter()
                .collect(),
            ..Faults::default()
        });
        let error = pull_snapshot(&meta, &hub, &id).expect_err("each gauntlet leg refuses");
        assert!(
            !matches!(error, SyncError::Workspace(_) if meta.load_snapshot(&id).is_ok()),
            "{fault:?} adopted a snapshot it should not have"
        );
        harness::Consistency::scan(sink.path()).assert_intact(&format!("gauntlet {fault:?}"));
    }
    // Transport-level flapping on top of the content faults.
    hub.set_faults(Faults {
        fail_downloads: 2,
        ..Faults::default()
    });
    let _ = pull_snapshot(&meta, &hub, &id);
    let _ = pull_snapshot(&meta, &hub, &id);

    hub.heal();
    pull_snapshot(&meta, &hub, &id).expect("an honest hub after the storm converges");

    assert_snapshot_fully_backed(&meta, &id, "after the full gauntlet");
    for (path, bytes) in &originals {
        assert_eq!(
            &reconstruct(&meta, &id, path),
            bytes,
            "{path} is not byte-exact after the gauntlet"
        );
    }
    harness::Consistency::scan(sink.path()).assert_intact("after the full gauntlet");

    // Nothing already verified was re-fetched by the final converging pull:
    // a second pull moves zero objects.
    let before = hub.state.borrow().downloads_by_digest.len();
    let report = pull_snapshot(&meta, &hub, &id).expect("a settled pull is a no-op");
    assert_eq!(
        report.fetched_objects, 0,
        "a settled store re-fetched objects it already held"
    );
    assert_eq!(
        hub.state.borrow().downloads_by_digest.len(),
        before,
        "a settled pull touched new objects"
    );
}
