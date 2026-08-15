//! Wire-shape fixtures for the th#1960 HTTP adapter, served over a real
//! loopback socket: every request the adapter emits is recorded and asserted
//! against the landed tensorhub spellings (routes, field names, tagged
//! digests, the claim's checksum binding), and every response it parses is a
//! canned body derived from the Go handler's types. This is the drift fence
//! until the live E2E runs against a deployed hub.

#![cfg(any(unix, windows))]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

use base64::Engine as _;
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::sync::http::{HttpTransport, TokenSource};
use tensorfs_core::sync::{
    CompleteStatus, DownloadGrant, PackClaim, PackGrant, SyncTransport, TransportError,
};
use tensorfs_core::tfm1::SnapshotId;

/// One scripted HTTP exchange: the canned answer, plus the recorded request.
struct Recorded {
    method: String,
    path: String,
    authorization: Option<String>,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// Serves `answers` sequentially on a loopback socket; returns the base URL
/// and a receiver yielding one `Recorded` per request, in order.
fn loopback(answers: Vec<(u16, String)>) -> (String, mpsc::Receiver<Recorded>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback binds");
    let address = listener.local_addr().expect("bound address");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        for (status, answer) in answers {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (head_end, mut have) = loop {
                let read = stream.read(&mut chunk).expect("request reads");
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(position) = find_blank_line(&buffer) {
                    break (position, buffer.len());
                }
                if read == 0 {
                    return;
                }
            };
            let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
            let mut lines = head.split("\r\n");
            let request_line = lines.next().unwrap_or_default();
            let mut parts = request_line.split(' ');
            let method = parts.next().unwrap_or_default().to_owned();
            let path = parts.next().unwrap_or_default().to_owned();
            let mut authorization = None;
            let mut headers = Vec::new();
            let mut content_length = 0_usize;
            for line in lines {
                let Some((name, value)) = line.split_once(':') else {
                    continue;
                };
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim().to_owned();
                if name == "authorization" {
                    authorization = Some(value.clone());
                }
                if name == "content-length" {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((name, value));
            }
            let body_start = head_end + 4;
            while have < body_start + content_length {
                let read = stream.read(&mut chunk).expect("body reads");
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
                have += read;
            }
            let body = buffer[body_start.min(buffer.len())..].to_vec();
            sender
                .send(Recorded {
                    method,
                    path,
                    authorization,
                    headers,
                    body,
                })
                .ok();
            let response = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}",
                answer.len(),
            );
            stream.write_all(response.as_bytes()).expect("reply writes");
        }
    });
    (format!("http://{address}"), receiver)
}

fn find_blank_line(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn transport(base: &str) -> HttpTransport {
    HttpTransport::new(base, "cozy", "demo").with_token(TokenSource::Static("tok".to_owned()))
}

fn digest_of(seed: &[u8]) -> ObjectDigest {
    use sha2::Digest as _;
    ObjectDigest::from_bytes(sha2::Sha256::digest(seed).into())
}

#[test]
fn declare_emits_and_parses_the_landed_wire() {
    let id = SnapshotId::of(b"manifest bytes");
    let have = digest_of(b"have");
    let miss = digest_of(b"miss");
    let canned = format!(
        r#"{{"snapshot_id":"{id}","session_id":"sess-1","stage":"uploading",
            "expires_at":"2026-08-15T12:00:00Z",
            "have":["{have}"],
            "staged_packs":[{{"sha256":"aa","size_bytes":1,"objects":2,"staged":true}}],
            "missing":[{{"digest":"{miss}","size_bytes":4096}}],
            "max_pack_payload":67108864,"max_packs_per_request":16,
            "declared_bytes":8192,"distinct_objects":2,"resident_objects":1}}"#,
    );
    let (base, recorded) = loopback(vec![(201, canned)]);

    let plan = transport(&base)
        .declare(b"manifest bytes", None)
        .expect("declare parses");
    assert_eq!(plan.snapshot_id, id);
    assert_eq!(plan.session, "sess-1");
    assert_eq!(plan.have, vec![have]);
    assert_eq!(plan.missing, vec![(miss, 4096)]);
    assert_eq!(plan.max_pack_payload, 67_108_864);
    assert_eq!(plan.max_packs_per_request, 16);
    assert!(plan.staged_packs[0].staged);

    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/v1/repos/cozy/demo/snapshot-sync");
    assert_eq!(request.authorization.as_deref(), Some("Bearer tok"));
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(body["expected_head"], "");
    let round_trip = base64::engine::general_purpose::STANDARD
        .decode(body["tfm1_base64"].as_str().expect("base64 field"))
        .expect("base64 decodes");
    assert_eq!(round_trip, b"manifest bytes");
}

#[test]
fn pack_grant_claims_bind_the_envelope_checksum_on_the_wire() {
    let member = digest_of(b"member");
    let canned = format!(
        r#"{{"grants":[{{"pack_sha256":"{sha}","staging_key":"snapshots/staging/sess-1/{sha}.tfp1",
             "put_url":"http://example.invalid/put","headers":{{"x-amz-checksum-sha256":"{sha}"}},
             "expires_at":"2026-08-15T12:00:00Z"}}],
            "staged_packs":[],"missing":[{{"digest":"{member}","size_bytes":4096}}]}}"#,
        sha = "ab".repeat(32),
    );
    let (base, recorded) = loopback(vec![(200, canned)]);

    let claim = PackClaim {
        sha256: "ab".repeat(32),
        size_bytes: 12 + 48 + 4096,
        objects: vec![member],
    };
    let granted = transport(&base)
        .pack_grants("sess-1", std::slice::from_ref(&claim))
        .expect("grants parse");
    assert_eq!(granted.grants.len(), 1);
    assert_eq!(granted.grants[0].pack_sha256, "ab".repeat(32));
    assert_eq!(granted.missing, vec![(member, 4096)]);

    let request = recorded.recv().expect("request recorded");
    assert_eq!(
        request.path,
        "/api/v1/repos/cozy/demo/snapshot-sync/sess-1/pack-grants"
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    let pack = &body["packs"][0];
    // The checksum binding IS the wire contract: the claim carries the
    // envelope's own sha256 and its exact arithmetic size.
    assert_eq!(pack["sha256"], serde_json::json!("ab".repeat(32)));
    assert_eq!(pack["size_bytes"], serde_json::json!(12 + 48 + 4096));
    assert_eq!(
        pack["objects"][0],
        serde_json::json!(member.to_string()),
        "members ride as tagged refs"
    );
}

#[test]
fn uploads_replay_grant_headers_and_map_403_to_expired() {
    let (base, recorded) = loopback(vec![(200, String::new()), (403, String::new())]);
    let grant = |url: &str| PackGrant {
        pack_sha256: "cd".repeat(32),
        staging_key: "snapshots/staging/s/p.tfp1".to_owned(),
        url: url.to_owned(),
        headers: vec![("x-amz-checksum-sha256".to_owned(), "cd".repeat(32))],
    };
    let client = transport(&base);

    client
        .upload_pack(&grant(&format!("{base}/put-1")), b"PACKBYTES")
        .expect("upload succeeds");
    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.method, "PUT");
    assert_eq!(request.body, b"PACKBYTES");
    assert!(
        request
            .headers
            .iter()
            .any(|(name, value)| name == "x-amz-checksum-sha256" && *value == "cd".repeat(32)),
        "grant headers are replayed verbatim"
    );
    assert!(
        request.authorization.is_none(),
        "presigned uploads carry no bearer"
    );

    let expired = client
        .upload_pack(&grant(&format!("{base}/put-2")), b"PACKBYTES")
        .expect_err("403 maps to expiry");
    assert!(matches!(expired, TransportError::Expired(_)));
}

#[test]
fn complete_maps_the_three_terminal_shapes() {
    let id = SnapshotId::of(b"m");
    let promoted = format!(
        r#"{{"stage":"promoted","snapshot_id":"{id}","head":"{id}","promoted":2,"total":2}}"#
    );
    let retryable = format!(
        r#"{{"stage":"promoting","snapshot_id":"{id}","promoted":1,"total":2,
            "failure":{{"code":"promote_incomplete","retryable":true,"message":"promoted 1 of 2"}}}}"#
    );
    let terminal = format!(
        r#"{{"stage":"promoting","snapshot_id":"{id}","promoted":1,"total":2,
            "failure":{{"code":"pack_mismatch","retryable":false,"message":"forged","findings":["digest-mismatch"]}}}}"#
    );
    let (base, recorded) = loopback(vec![(200, promoted), (409, retryable), (422, terminal)]);
    let client = transport(&base);

    assert_eq!(
        client.complete("sess-1").expect("promoted parses"),
        CompleteStatus::Promoted
    );
    let request = recorded.recv().expect("request recorded");
    assert_eq!(
        request.path,
        "/api/v1/repos/cozy/demo/snapshot-sync/sess-1/complete"
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(
        body,
        serde_json::json!({}),
        "complete sends an empty object"
    );

    assert_eq!(
        client.complete("sess-1").expect("retryable parses"),
        CompleteStatus::Incomplete {
            code: "promote_incomplete".to_owned()
        }
    );
    assert_eq!(
        client.complete("sess-1").expect("terminal parses"),
        CompleteStatus::Failed {
            code: "pack_mismatch".to_owned()
        }
    );
}

#[test]
fn head_and_download_grants_speak_the_landed_shapes() {
    let id = SnapshotId::of(b"head");
    let known = digest_of(b"known");
    let unknown = digest_of(b"unknown");
    let answers = vec![
        (200, r#"{"snapshot_id":""}"#.to_owned()),
        (200, format!(r#"{{"snapshot_id":"{id}"}}"#)),
        (
            200,
            format!(
                r#"{{"grants":[{{"digest":"{known}","size_bytes":9,
                     "get_url":"http://example.invalid/get","expires_at":"2026-08-15T12:00:00Z"}}],
                    "unknown":["{unknown}"]}}"#
            ),
        ),
    ];
    let (base, recorded) = loopback(answers);
    let client = transport(&base);

    assert_eq!(client.head().expect("empty head parses"), None);
    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, "/api/v1/repos/cozy/demo/snapshot-sync/head");

    assert_eq!(client.head().expect("head parses"), Some(id));
    recorded.recv().expect("request recorded");

    let grants = client
        .download_grants(&[known, unknown])
        .expect("grants parse");
    assert_eq!(grants.len(), 1, "unknown digests are omitted, not invented");
    assert_eq!(grants[0].digest, known);
    assert_eq!(grants[0].length, 9);
    let request = recorded.recv().expect("request recorded");
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(
        body["digests"],
        serde_json::json!([known.to_string(), unknown.to_string()]),
        "digests ride as tagged refs"
    );
}

#[test]
fn downloads_enforce_the_granted_length_and_map_403_to_expired() {
    let (base, _recorded) = loopback(vec![
        (200, "NINEBYTES".to_owned()),
        (200, "SHORT".to_owned()),
        (403, String::new()),
    ]);
    let client = transport(&base);
    let grant = |url: &str| DownloadGrant {
        digest: digest_of(b"x"),
        length: 9,
        url: url.to_owned(),
    };

    let bytes = client
        .download(&grant(&format!("{base}/get-1")))
        .expect("exact length succeeds");
    assert_eq!(bytes, b"NINEBYTES");

    let short = client
        .download(&grant(&format!("{base}/get-2")))
        .expect_err("length lies refuse");
    assert!(matches!(short, TransportError::Io(_)));

    let expired = client
        .download(&grant(&format!("{base}/get-3")))
        .expect_err("403 maps to expiry");
    assert!(matches!(expired, TransportError::Expired(_)));
}

#[test]
fn error_envelopes_map_to_refusals_and_expiry() {
    let conflict = r#"{"error":{"code":"head_conflict","message":"the repo head is x, not y"}}"#;
    let expired = r#"{"error":{"code":"session_expired","message":"declare again"}}"#;
    let (base, _recorded) = loopback(vec![(409, conflict.to_owned()), (410, expired.to_owned())]);
    let client = transport(&base);

    let refused = client
        .declare(b"m", None)
        .expect_err("conflict envelope refuses");
    assert!(matches!(
        refused,
        TransportError::Refused { code, .. } if code == "head_conflict"
    ));

    let lease = client
        .declare(b"m", None)
        .expect_err("expired envelope replans");
    assert!(matches!(lease, TransportError::Expired(_)));
}
