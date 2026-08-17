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
    CompleteStatus, DownloadGrant, PackClaim, PackGrant, Progress, ProgressSink, SyncTransport,
    TransportError,
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
        .upload_pack(
            &grant(&format!("{base}/put-1")),
            b"PACKBYTES",
            ProgressSink::silent(),
        )
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
        .upload_pack(
            &grant(&format!("{base}/put-2")),
            b"PACKBYTES",
            ProgressSink::silent(),
        )
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

    let mut bytes = Vec::new();
    let written = client
        .download(
            &grant(&format!("{base}/get-1")),
            &mut bytes,
            ProgressSink::silent(),
        )
        .expect("exact length succeeds");
    assert_eq!(bytes, b"NINEBYTES");
    assert_eq!(written, 9);

    let mut partial = Vec::new();
    let short = client
        .download(
            &grant(&format!("{base}/get-2")),
            &mut partial,
            ProgressSink::silent(),
        )
        .expect_err("length lies refuse");
    assert!(matches!(short, TransportError::Io(_)));

    let mut nothing = Vec::new();
    let expired = client
        .download(
            &grant(&format!("{base}/get-3")),
            &mut nothing,
            ProgressSink::silent(),
        )
        .expect_err("403 maps to expiry");
    assert!(matches!(expired, TransportError::Expired(_)));
    assert!(
        nothing.is_empty(),
        "an expired grant must not have written anything into the sink"
    );
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

/// A download must be visible WHILE it is in flight, not only when it returns.
///
/// pgw#1287: a 31.6 GB pull that reports nothing until its last byte lands is
/// indistinguishable from a dead one, and a consumer watching for liveness
/// condemns a pod that is downloading correctly. So this asserts the timing
/// directly rather than the totals: the server refuses to write the rest of
/// the body until the client has REPORTED the first piece, which a transport
/// that reports only on return can never do. The bounded wait exists so a
/// regression fails the test instead of hanging the suite — a test bailout,
/// not a health judgement.
#[test]
fn a_download_reports_bytes_while_the_object_is_still_arriving() {
    const PIECE: usize = 256 * 1024;
    let body = vec![0x5A_u8; 4 * PIECE];
    let served = body.clone();

    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback binds");
    let address = listener.local_addr().expect("bound address");
    let (reported, first_report) = mpsc::channel::<()>();
    let (stalled, saw_stall) = mpsc::channel::<()>();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("the client connects");
        let mut chunk = [0_u8; 4096];
        let mut head = Vec::new();
        while find_blank_line(&head).is_none() {
            let read = stream.read(&mut chunk).expect("request reads");
            if read == 0 {
                return;
            }
            head.extend_from_slice(&chunk[..read]);
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            served.len()
        );
        stream.write_all(response.as_bytes()).expect("head writes");
        stream
            .write_all(&served[..PIECE])
            .expect("first piece writes");
        stream.flush().expect("first piece flushes");
        // The whole point: nothing more moves until the client has reported
        // what already arrived.
        if first_report
            .recv_timeout(std::time::Duration::from_secs(10))
            .is_err()
        {
            stalled.send(()).ok();
            return;
        }
        stream.write_all(&served[PIECE..]).expect("the rest writes");
    });

    let client = transport(&format!("http://{address}"));
    let grant = DownloadGrant {
        digest: digest_of(b"streamed"),
        length: body.len() as u64,
        url: format!("http://{address}/objects/streamed"),
    };
    let beats = std::cell::RefCell::new(Vec::new());
    let sink = |event: Progress| {
        if let Progress::Bytes(moved) = event {
            beats.borrow_mut().push(moved);
            reported.send(()).ok();
        }
    };

    let mut fetched_bytes = Vec::new();
    let fetched = client.download(&grant, &mut fetched_bytes, ProgressSink::new(&sink));
    server.join().expect("the server thread finishes");
    assert!(
        saw_stall.try_recv().is_err(),
        "the download reported nothing while the object was still arriving — \
         the whole transfer is invisible until it ends"
    );
    let fetched = fetched.expect("the download completes");
    assert_eq!(fetched, body.len() as u64);
    assert_eq!(fetched_bytes, body);

    let beats = beats.borrow().clone();
    assert!(
        beats.len() >= 2,
        "a {}-byte object reported {} time(s); sub-object progress is what \
         makes a long transfer observable",
        body.len(),
        beats.len()
    );
    assert_eq!(
        beats.iter().sum::<u64>(),
        body.len() as u64,
        "every byte that moved must be reported exactly once"
    );
}

/// The upload reports its bytes as they leave, and the wire framing is
/// unchanged by that: a presigned signature covers a `content-length` body,
/// never a chunked one.
#[test]
fn an_upload_reports_its_bytes_without_changing_the_framing() {
    let (base, recorded) = loopback(vec![(200, String::new())]);
    let pack = vec![0xC3_u8; 40_000];
    let grant = PackGrant {
        pack_sha256: "ab".repeat(32),
        staging_key: "snapshots/staging/s/p.tfp1".to_owned(),
        url: format!("{base}/put-streamed"),
        headers: vec![("x-amz-checksum-sha256".to_owned(), "ab".repeat(32))],
    };
    let moved = std::cell::Cell::new(0_u64);
    let beats = std::cell::Cell::new(0_usize);
    let sink = |event: Progress| {
        if let Progress::Bytes(bytes) = event {
            moved.set(moved.get() + bytes);
            beats.set(beats.get() + 1);
        }
    };

    transport(&base)
        .upload_pack(&grant, &pack, ProgressSink::new(&sink))
        .expect("upload succeeds");

    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.body, pack, "the exact pack bytes reach the store");
    assert!(
        request
            .headers
            .iter()
            .any(|(name, value)| name == "content-length" && value == "40000"),
        "a presigned PUT states its length"
    );
    assert!(
        !request
            .headers
            .iter()
            .any(|(name, value)| name == "transfer-encoding" && value.contains("chunked")),
        "chunked framing would break the presigned signature"
    );
    assert_eq!(moved.get(), pack.len() as u64);
    assert!(
        beats.get() >= 2,
        "a streamed upload reports as it goes; {} report(s) is a single \
         end-of-transfer beat",
        beats.get()
    );
}

// ---------------------------------------------------------------------------
// The blob lane (th#2064)
// ---------------------------------------------------------------------------

/// `missing_blobs` on the declare answer, in the spelling the hub emits.
///
/// It is ADDITIVE: a hub with nothing above the pack bound answers `[]`, and
/// an older hub omits the field entirely. Both must parse as an empty blob
/// lane rather than as a failure.
#[test]
fn declare_parses_the_additive_blob_lane() {
    let id = SnapshotId::of(b"manifest bytes");
    let big = digest_of(b"a dataset video");
    let small = digest_of(b"config");
    let canned = format!(
        r#"{{"snapshot_id":"{id}","session_id":"sess-1","stage":"uploading",
            "have":[],"staged_packs":[],
            "missing":[{{"digest":"{small}","size_bytes":4096}}],
            "missing_blobs":[{{"digest":"{big}","size_bytes":69206016}}],
            "max_pack_payload":67108864,"max_packs_per_request":16}}"#,
    );
    let without = format!(
        r#"{{"snapshot_id":"{id}","session_id":"sess-2","stage":"uploading",
            "have":[],"staged_packs":[],
            "missing":[{{"digest":"{small}","size_bytes":4096}}],
            "max_pack_payload":67108864,"max_packs_per_request":16}}"#,
    );
    let (base, _recorded) = loopback(vec![(201, canned), (201, without)]);
    let client = transport(&base);

    let plan = client.declare(b"manifest bytes", None).expect("parses");
    assert_eq!(plan.missing, vec![(small, 4096)]);
    assert_eq!(plan.missing_blobs, vec![(big, 69_206_016)]);

    let plan = client.declare(b"manifest bytes", None).expect("parses");
    assert!(
        plan.missing_blobs.is_empty(),
        "an omitted field is an empty lane, not a parse failure"
    );
    assert_eq!(plan.missing, vec![(small, 4096)]);
}

/// The blob-grant request and answer, in the landed spellings: tagged digests
/// out, `upload_id` / `part_size` / `parts[].put_url` / `uploaded_parts` back.
#[test]
fn blob_grants_emit_and_parse_the_landed_wire() {
    let big = digest_of(b"a dataset video");
    let canned = format!(
        r#"{{"grants":[{{"digest":"{big}","length":69206016,
             "staging_key":"snapshots/staging/sess-1/{big}.blob",
             "upload_id":"r2-upload-1","part_size":67108864,
             "parts":[
               {{"part_number":1,"size_bytes":67108864,"put_url":"http://example.invalid/p1",
                 "headers":{{"x-amz-part-number":"1"}}}},
               {{"part_number":2,"size_bytes":2097152,"put_url":"http://example.invalid/p2",
                 "headers":{{}}}}],
             "uploaded_parts":[1],
             "expires_at":"2026-08-17T12:00:00Z"}}]}}"#,
    );
    let (base, recorded) = loopback(vec![(200, canned)]);

    let grants = transport(&base)
        .blob_grants("sess-1", &[big])
        .expect("blob grants parse");
    assert_eq!(grants.len(), 1);
    let grant = &grants[0];
    assert_eq!(grant.digest, big);
    assert_eq!(grant.length, 69_206_016);
    assert_eq!(grant.upload_id, "r2-upload-1");
    assert_eq!(grant.part_size, 67_108_864);
    assert_eq!(grant.uploaded_parts, vec![1]);
    assert_eq!(grant.parts.len(), 2);
    assert_eq!(grant.parts[1].part_number, 2);
    assert_eq!(grant.parts[1].size_bytes, 2_097_152);
    assert_eq!(grant.parts[0].url, "http://example.invalid/p1");
    assert_eq!(
        grant.parts[0].headers,
        vec![("x-amz-part-number".to_owned(), "1".to_owned())]
    );
    // The parts must cover the object exactly, which is what the engine
    // checks before it moves a byte.
    assert_eq!(
        grant.parts.iter().map(|part| part.size_bytes).sum::<u64>(),
        grant.length
    );

    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/api/v1/repos/cozy/demo/snapshot-sync/sess-1/blob-grants"
    );
    assert_eq!(request.authorization.as_deref(), Some("Bearer tok"));
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(body["digests"][0], big.to_string());
}

/// A part PUT replays the granted headers verbatim, states its length, and
/// returns the store's etag — which the hub needs to complete the upload, so
/// a 200 without one is a failure rather than an empty string.
#[test]
fn a_part_put_replays_its_grant_and_returns_the_etag() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback binds");
    let address = listener.local_addr().expect("bound address");
    let server = thread::spawn(move || {
        let mut seen = Vec::new();
        for reply in ["\"etag-1\"", ""] {
            let (mut stream, _) = listener.accept().expect("accepts");
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            loop {
                let read = stream.read(&mut chunk).expect("reads");
                buffer.extend_from_slice(&chunk[..read]);
                if read == 0 || buffer.len() >= 6 && find_blank_line(&buffer).is_some() {
                    let head_end = find_blank_line(&buffer).unwrap_or(buffer.len());
                    if buffer.len() >= head_end + 4 + 6 {
                        break;
                    }
                }
                if read == 0 {
                    break;
                }
            }
            seen.push(String::from_utf8_lossy(&buffer).into_owned());
            let etag = if reply.is_empty() {
                String::new()
            } else {
                format!("etag: {reply}\r\n")
            };
            let response =
                format!("HTTP/1.1 200 OK\r\n{etag}content-length: 0\r\nconnection: close\r\n\r\n");
            stream.write_all(response.as_bytes()).expect("replies");
        }
        seen
    });

    let client = transport(&format!("http://{address}"));
    let part = tensorfs_core::sync::BlobPart {
        part_number: 7,
        size_bytes: 6,
        url: format!("http://{address}/part-7"),
        headers: vec![("x-amz-part-number".to_owned(), "7".to_owned())],
    };
    let etag = client
        .upload_blob_part(&part, b"SIXSIX", ProgressSink::silent())
        .expect("the part lands");
    assert_eq!(
        etag, "\"etag-1\"",
        "the etag is returned verbatim, quotes included"
    );

    let missing = client
        .upload_blob_part(&part, b"SIXSIX", ProgressSink::silent())
        .expect_err("a part with no etag cannot be completed");
    assert!(matches!(missing, TransportError::Io(_)), "got {missing:?}");

    let seen = server.join().expect("server thread finishes");
    assert!(
        seen[0].contains("PUT /part-7"),
        "wrong request: {}",
        seen[0]
    );
    assert!(
        seen[0]
            .to_ascii_lowercase()
            .contains("x-amz-part-number: 7"),
        "the granted headers are replayed verbatim: {}",
        seen[0]
    );
    assert!(
        seen[0].to_ascii_lowercase().contains("content-length: 6"),
        "a presigned PUT must state its length, not switch to chunked framing: {}",
        seen[0]
    );
    assert!(
        seen[0].ends_with("SIXSIX"),
        "the exact part bytes must be the body: {}",
        seen[0]
    );
}

/// Reporting part etags: the route, the tagged digest, and the part rows.
#[test]
fn blob_part_etags_are_reported_in_the_landed_spelling() {
    let big = digest_of(b"a dataset video");
    let (base, recorded) = loopback(vec![(200, "{}".to_owned())]);

    transport(&base)
        .report_blob_parts(
            "sess-1",
            &big,
            &[
                tensorfs_core::sync::BlobPartReport {
                    part_number: 1,
                    etag: "\"one\"".to_owned(),
                },
                tensorfs_core::sync::BlobPartReport {
                    part_number: 2,
                    etag: "\"two\"".to_owned(),
                },
            ],
        )
        .expect("the report lands");

    let request = recorded.recv().expect("request recorded");
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/api/v1/repos/cozy/demo/snapshot-sync/sess-1/blob-parts"
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(body["digest"], big.to_string());
    assert_eq!(body["parts"][0]["part_number"], 1);
    assert_eq!(body["parts"][1]["etag"], "\"two\"");
}

/// A blob grant whose lease ran out surfaces as `Expired`, so the engine
/// re-asks rather than replaying a URL the store has stopped honouring.
#[test]
fn an_expired_blob_grant_surfaces_as_expiry() {
    let expired = r#"{"error":{"code":"session_expired","message":"declare again"}}"#;
    let (base, _recorded) = loopback(vec![(410, expired.to_owned())]);
    let error = transport(&base)
        .blob_grants("sess-1", &[digest_of(b"big")])
        .expect_err("an expired lease must not read as grants");
    assert!(matches!(error, TransportError::Expired(_)), "got {error:?}");
}
