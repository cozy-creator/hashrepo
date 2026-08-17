//! The multipart blob lane (tensorfs#79) against the hub wire tensorhub
//! th#2064 landed.
//!
//! An object larger than a TFP1 pack payload cannot ride a pack, so the
//! remote partitions its answer into a pack lane and a blob lane, opens one
//! multipart upload per blob, presigns its parts, and stream-hashes the
//! assembled object exactly once at `complete`. **Parts are transport and
//! never enter identity.**
//!
//! The hub in `harness::FaultHub` models that wire honestly: part bytes land
//! in a store that computes no digest of its own and are keyed by upload id,
//! so adopting the same upload on a re-grant is genuinely what makes a resume
//! find its parts.
//!
//! Most tests declare a small pack-payload bound so a few kibibytes ride the
//! blob lane. The engine's behaviour does not depend on the constant — it
//! uses whatever the remote declares — and one test at the end pays for a
//! genuinely oversized object to prove the small bound is a fixture choice
//! and not an assumption.

#![cfg(any(unix, windows))]

mod harness;

use harness::{FaultHub, Faults, Scratch, sha256_hex};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::sync::{
    ProgressSink, PushOptions, PushReport, SyncError, pull_snapshot, push_snapshot,
};
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::tfp1;
use tensorfs_core::workspace::{Mutation, WorkspaceStore};

/// The bound most tests declare: anything above it is a blob.
const SMALL_BOUND: u64 = 4096;

fn faults(bound: u64) -> Faults {
    Faults {
        pack_payload_bound: bound,
        ..Faults::default()
    }
}

/// A workspace holding one small pack-lane file and one file above `bound`,
/// with contents that differ from each other so nothing can pass by crossing
/// them.
fn two_lane_workspace(root: &std::path::Path, blob_len: usize) -> (WorkspaceStore, SnapshotId) {
    let meta = WorkspaceStore::open(root).expect("workspace store opens");
    meta.create_workspace("publisher")
        .expect("workspace creates");

    let small = b"{\"model_type\":\"llama\"}".to_vec();
    let blob: Vec<u8> = (0..blob_len).map(|index| (index % 251) as u8).collect();
    assert_ne!(small, blob);

    let small_object = meta.store().put_bytes(&small).expect("small admits");
    let blob_object = meta.store().put_bytes(&blob).expect("blob admits");
    meta.commit_generation(
        "publisher",
        &[
            Mutation::Mkdir {
                path: "clips".to_owned(),
            },
            Mutation::CreateFile {
                path: "config.json".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest: small_object.digest(),
                    length: small_object.length(),
                }],
            },
            Mutation::CreateFile {
                path: "clips/train.webm".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest: blob_object.digest(),
                    length: blob_object.length(),
                }],
            },
        ],
    )
    .expect("both files commit");
    let id = meta.seal_snapshot("publisher", None).expect("seals");
    (meta, id)
}

fn blob_digest(meta: &WorkspaceStore, id: &SnapshotId, path: &str) -> (ObjectDigest, u64) {
    let snapshot = meta.load_snapshot(id).expect("snapshot loads");
    for (entry_path, entry) in snapshot.entries() {
        if entry_path != path {
            continue;
        }
        if let tensorfs_core::tfm1::Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, length } = record {
                    return (*digest, *length);
                }
            }
        }
    }
    panic!("{path} has no data record");
}

fn push(meta: &WorkspaceStore, hub: &FaultHub, id: &SnapshotId) -> Result<PushReport, SyncError> {
    push_snapshot(
        meta,
        hub,
        id,
        None,
        PushOptions::default(),
        ProgressSink::silent(),
    )
}

// ---------------------------------------------------------------------------
// The lane itself
// ---------------------------------------------------------------------------

/// A snapshot with one object in each lane syncs end to end, and each object
/// travelled by its own lane: the small one inside a TFP1 pack, the large one
/// as multipart parts.
///
/// Red proof: remove the `push_blob_lane` call from `push_snapshot` and the
/// blob never leaves — `complete` answers `upload_incomplete` until the
/// attempt budget is spent.
#[test]
fn a_snapshot_with_both_lanes_syncs_and_each_object_takes_its_own_lane() {
    let root = Scratch::new("blob-both-lanes");
    let (meta, id) = two_lane_workspace(root.path(), 10 * SMALL_BOUND as usize + 17);
    let (blob, blob_len) = blob_digest(&meta, &id, "clips/train.webm");
    let (small, _) = blob_digest(&meta, &id, "config.json");

    let hub = FaultHub::with_faults(faults(SMALL_BOUND));
    let report = push(&meta, &hub, &id).expect("both lanes complete");

    assert_eq!(report.blobs, 1, "exactly one object rode the blob lane");
    assert_eq!(
        report.blob_parts,
        blob_len.div_ceil(SMALL_BOUND),
        "every part of the blob was PUT"
    );
    assert_eq!(report.packs, 1, "the small object still rode a pack");
    assert_eq!(report.uploaded_objects, 2);

    let state = hub.state.borrow();
    assert_eq!(state.head, Some(id), "the head advanced to this snapshot");
    // The blob arrived by parts and the small object by a pack: neither used
    // the other's mechanism.
    assert_eq!(
        state.blob_part_puts.get(blob.as_bytes()).copied(),
        Some(blob_len.div_ceil(SMALL_BOUND) as u32),
        "the blob's parts went to the multipart store"
    );
    assert_eq!(
        state.blob_part_puts.get(small.as_bytes()).copied(),
        None,
        "the small object must never have opened a multipart upload"
    );
    assert!(
        state.uploads_by_digest.contains_key(small.as_bytes()),
        "the small object was carried by a pack"
    );
    assert!(
        !state.uploads_by_digest.contains_key(blob.as_bytes()),
        "the blob must never have been packed"
    );

    // And the bytes the hub now holds are the bytes we sent.
    let held = state.objects.get(blob.as_bytes()).expect("blob promoted");
    assert_eq!(held.len() as u64, blob_len);
    assert_eq!(sha256_hex(held), blob.to_hex());
}

/// The blob a push sent comes back byte-exact through a pull into a fresh
/// store — the round trip is what proves parts never entered identity.
#[test]
fn a_blob_round_trips_byte_exactly_through_a_pull() {
    let source = Scratch::new("blob-rt-source");
    let (meta, id) = two_lane_workspace(source.path(), 5 * SMALL_BOUND as usize + 3);
    let (blob, blob_len) = blob_digest(&meta, &id, "clips/train.webm");
    let original = {
        use std::io::Read as _;
        let mut file = meta.store().open_object(&blob).expect("blob opens");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("blob reads");
        bytes
    };

    let hub = FaultHub::with_faults(faults(SMALL_BOUND));
    push(&meta, &hub, &id).expect("push completes");

    let sink = Scratch::new("blob-rt-sink");
    let puller = WorkspaceStore::open(sink.path()).expect("sink store opens");
    let report = pull_snapshot(&puller, &hub, &id, ProgressSink::silent()).expect("pull completes");
    assert_eq!(
        report.fetched_objects, 2,
        "both data objects were fetched; the manifest is not one of them"
    );

    let mut fetched = Vec::new();
    {
        use std::io::Read as _;
        let mut file = puller.store().open_object(&blob).expect("blob resident");
        file.read_to_end(&mut fetched).expect("blob reads");
    }
    assert_eq!(fetched, original);
    assert_eq!(fetched.len() as u64, blob_len);
}

// ---------------------------------------------------------------------------
// Identity: parts are transport
// ---------------------------------------------------------------------------

/// A corrupted part is invisible to the store — it issues an etag for
/// whatever arrived — and is caught only by the admission-time stream hash.
/// The refusal is terminal, nothing is promoted, and the head does not move.
///
/// Red proof: drop the digest comparison in the hub's `complete` and the
/// corrupt blob promotes with the declared digest naming bytes that are not
/// it — the exact poisoning the stream hash exists to stop.
#[test]
fn a_corrupted_part_refuses_terminally_and_promotes_nothing() {
    let root = Scratch::new("blob-corrupt-part");
    let (meta, id) = two_lane_workspace(root.path(), 3 * SMALL_BOUND as usize);
    let (blob, _) = blob_digest(&meta, &id, "clips/train.webm");

    let hub = FaultHub::with_faults(Faults {
        pack_payload_bound: SMALL_BOUND,
        // The last part, so the corruption survives every earlier part
        // landing correctly.
        corrupt_blob_part: Some(3),
        ..Faults::default()
    });

    match push(&meta, &hub, &id) {
        Err(SyncError::HeadRefused { code }) => {
            assert_eq!(code, "blob_digest_mismatch", "the refusal must name itself");
        }
        other => panic!("expected a terminal blob_digest_mismatch, got {other:?}"),
    }

    let state = hub.state.borrow();
    assert!(
        !state.objects.contains_key(blob.as_bytes()),
        "a blob whose parts do not hash to it must never be promoted"
    );
    assert_eq!(state.head, None, "the head must not have moved");
}

// ---------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------

/// Re-granting adopts the SAME upload and reports the parts already landed,
/// so a resumed push re-sends only what is missing.
///
/// The grant lease runs out after two parts have landed. That is a REPLAN,
/// not a failure: the engine re-asks, the remote adopts the same upload and
/// reports the two parts it already holds, and only the remaining two are
/// sent. Counting part PUTs is what makes this a resume claim rather than an
/// "it eventually worked" claim.
///
/// Red proof: set `forget_blob_upload_id`, which answers the re-grant with a
/// fresh upload id — every part is then re-sent and the total below is 6
/// instead of 4, which is the defect th#2064 found in its own resume promise.
#[test]
fn a_resumed_blob_adopts_the_live_upload_and_re_sends_only_missing_parts() {
    let parts = 4_u64;
    let landed_before_expiry = 2_u32;

    let mut totals = Vec::new();
    for forget in [false, true] {
        let root = Scratch::new(if forget {
            "blob-resume-forgetful"
        } else {
            "blob-resume"
        });
        let (meta, id) = two_lane_workspace(root.path(), (parts * SMALL_BOUND) as usize);
        let (blob, _) = blob_digest(&meta, &id, "clips/train.webm");
        let hub = FaultHub::with_faults(Faults {
            pack_payload_bound: SMALL_BOUND,
            expire_blob_parts_after: Some(landed_before_expiry),
            forget_blob_upload_id: forget,
            ..Faults::default()
        });

        let report = push(&meta, &hub, &id).expect("the replan finishes the blob");
        let total = hub
            .state
            .borrow()
            .blob_part_puts
            .get(blob.as_bytes())
            .copied()
            .expect("parts were counted");
        assert_eq!(
            u64::from(total),
            report.blob_parts,
            "the report and the hub must agree on how many parts were sent"
        );
        assert_eq!(hub.state.borrow().head, Some(id), "the blob still promotes");
        totals.push(u64::from(total));
    }

    assert_eq!(
        totals[0], parts,
        "adopting the live upload must send each part exactly once"
    );
    assert_eq!(
        totals[1],
        parts + u64::from(landed_before_expiry),
        "a fresh upload id orphans the landed parts and re-sends them, which is \
         precisely the defect the adoption rule exists to prevent"
    );
    assert!(
        totals[1] > totals[0],
        "the red arm must actually cost more than the honest one"
    );
}

/// A part PUT reset by the carrier is retried; a push does not die because a
/// residential uplink dropped one connection. Measured need, not a preference
/// (th#2064's standing-stack proof lost a 64 MiB part exactly this way).
///
/// Red proof: remove the retry loop in `push_blob_lane` and the push fails
/// with the injected carrier fault.
#[test]
fn a_reset_part_upload_is_retried_rather_than_killing_the_push() {
    let root = Scratch::new("blob-part-retry");
    let (meta, id) = two_lane_workspace(root.path(), 2 * SMALL_BOUND as usize);
    let hub = FaultHub::with_faults(Faults {
        pack_payload_bound: SMALL_BOUND,
        fail_blob_parts: 2,
        ..Faults::default()
    });

    let report = push(&meta, &hub, &id).expect("the push survives two resets");
    assert_eq!(report.blobs, 1);
    assert_eq!(hub.state.borrow().head, Some(id));
}

/// The retry budget is bounded: a carrier that never recovers fails the push
/// with a typed error naming the object, rather than looping forever.
#[test]
fn part_retries_are_bounded_and_the_refusal_names_the_object() {
    let root = Scratch::new("blob-part-exhausted");
    let (meta, id) = two_lane_workspace(root.path(), 2 * SMALL_BOUND as usize);
    let (blob, _) = blob_digest(&meta, &id, "clips/train.webm");
    let hub = FaultHub::with_faults(Faults {
        pack_payload_bound: SMALL_BOUND,
        fail_blob_parts: u32::MAX,
        ..Faults::default()
    });

    match push(&meta, &hub, &id) {
        Err(SyncError::BlobPartAttemptsExhausted { digest, attempts }) => {
            assert_eq!(digest, blob);
            assert_eq!(attempts, PushOptions::default().max_upload_attempts);
        }
        other => panic!("expected a bounded part-retry refusal, got {other:?}"),
    }
    assert_eq!(hub.state.borrow().head, None);
}

// ---------------------------------------------------------------------------
// The lanes cannot be confused
// ---------------------------------------------------------------------------

/// A remote that puts a pack-lane object in the blob lane is refused, typed
/// and naming the object. The client does not "fix" the partition: a remote
/// that partitions wrongly is broken, and opening a multipart upload for an
/// object the pack lane carries is billed state nothing will reclaim.
#[test]
fn a_remote_that_partitions_into_the_wrong_lane_is_refused() {
    let root = Scratch::new("blob-lane-mismatch");
    let (meta, id) = two_lane_workspace(root.path(), 2 * SMALL_BOUND as usize);
    let hub = FaultHub::with_faults(Faults {
        pack_payload_bound: SMALL_BOUND,
        blob_lane_everything: true,
        ..Faults::default()
    });

    match push(&meta, &hub, &id) {
        Err(SyncError::BlobLaneMismatch {
            length,
            limit,
            lane,
            ..
        }) => {
            assert_eq!(lane, "blob");
            assert_eq!(limit, SMALL_BOUND);
            assert!(length <= SMALL_BOUND, "the misplaced object is a small one");
        }
        other => panic!("expected a lane refusal, got {other:?}"),
    }
    assert!(hub.state.borrow().objects.is_empty());
}

/// A remote that omits a grant for a blob it just listed is refused by name,
/// rather than the push silently completing with the blob still missing.
#[test]
fn an_omitted_blob_grant_is_refused_by_name() {
    let root = Scratch::new("blob-grant-omitted");
    let (meta, id) = two_lane_workspace(root.path(), 2 * SMALL_BOUND as usize);
    let (blob, _) = blob_digest(&meta, &id, "clips/train.webm");
    let hub = FaultHub::with_faults(Faults {
        pack_payload_bound: SMALL_BOUND,
        omit_blob_grant: Some(*blob.as_bytes()),
        ..Faults::default()
    });

    match push(&meta, &hub, &id) {
        Err(SyncError::BlobGrantOmitted(digest)) => assert_eq!(digest, blob),
        other => panic!("expected an omitted-grant refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The empty file, which neither lane can carry
// ---------------------------------------------------------------------------

/// A zero-length file never reaches either lane, because a `blob-v1` entry of
/// logical size 0 carries NO data record at all — so the client's closure
/// does not contain the empty digest and cannot offer it.
///
/// This is the client half of th#1338: TFP1 refuses a zero-length member and
/// a multipart upload needs a part, so a client that DID offer it would stall
/// the snapshot on `upload_incomplete` forever. One ordinary `__init__.py` is
/// enough to reach it.
#[test]
fn an_empty_file_is_offered_to_neither_lane() {
    let root = Scratch::new("blob-empty-file");
    let meta = WorkspaceStore::open(root.path()).expect("workspace store opens");
    meta.create_workspace("publisher")
        .expect("workspace creates");
    let real = meta.store().put_bytes(b"not empty").expect("object admits");
    meta.commit_generation(
        "publisher",
        &[
            Mutation::Mkdir {
                path: "pkg".to_owned(),
            },
            Mutation::CreateFile {
                path: "pkg/__init__.py".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: Vec::new(),
            },
            Mutation::CreateFile {
                path: "pkg/model.py".to_owned(),
                executable: false,
                planner: PlannerId::BlobV1,
                records: vec![FileRecord::Data {
                    digest: real.digest(),
                    length: real.length(),
                }],
            },
        ],
    )
    .expect("the empty file commits");
    let id = meta.seal_snapshot("publisher", None).expect("seals");

    // The empty blob's digest is pinned by the format, and it must not appear
    // in anything the client offers the remote.
    let empty = ObjectDigest::from_bytes(*tensorfs_core::tfm1::EMPTY_BLOB_DIGEST.as_bytes());
    let snapshot = meta.load_snapshot(&id).expect("loads");
    let mut offered = Vec::new();
    for (_path, entry) in snapshot.entries() {
        if let tensorfs_core::tfm1::Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, .. } = record {
                    offered.push(*digest);
                }
            }
        }
    }
    assert!(
        !offered.contains(&empty),
        "the empty digest must not be offered to either lane"
    );

    let hub = FaultHub::with_faults(faults(SMALL_BOUND));
    let report = push(&meta, &hub, &id).expect("a snapshot with an empty file syncs");
    assert_eq!(report.blobs, 0);
    assert_eq!(hub.state.borrow().head, Some(id));

    // And it comes back: the empty file is a manifest fact, not an object to
    // move.
    let sink = Scratch::new("blob-empty-sink");
    let puller = WorkspaceStore::open(sink.path()).expect("sink store opens");
    pull_snapshot(&puller, &hub, &id, ProgressSink::silent()).expect("pull completes");
    let pulled = puller.load_snapshot(&id).expect("pulled snapshot loads");
    let empty_entry = pulled
        .entries()
        .iter()
        .find(|(path, _)| path == "pkg/__init__.py")
        .expect("the empty file survived the round trip");
    match &empty_entry.1 {
        tensorfs_core::tfm1::Entry::File { body, .. } => {
            assert_eq!(body.logical_size(), 0);
            assert!(body.records().is_empty());
        }
        other => panic!("expected a file entry, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The bound is the remote's, not ours
// ---------------------------------------------------------------------------

/// One genuinely oversized object, against the wire's own 64 MiB bound, with
/// no lowered fixture bound anywhere.
///
/// Every test above declares a small bound so the suite stays fast. This one
/// pays for a real one, so "the small bound is a fixture choice" is a
/// measured claim rather than an assurance: the object crosses the actual
/// pack payload limit, takes two real parts, and promotes.
#[test]
fn an_object_above_the_real_pack_payload_bound_rides_the_blob_lane() {
    let root = Scratch::new("blob-real-bound");
    let over = tfp1::MAX_PACK_PAYLOAD as usize + 4096;
    let (meta, id) = two_lane_workspace(root.path(), over);
    let (blob, blob_len) = blob_digest(&meta, &id, "clips/train.webm");
    assert!(blob_len > tfp1::MAX_PACK_PAYLOAD);

    // No `pack_payload_bound`: the hub declares the wire's own 64 MiB.
    let hub = FaultHub::default();
    let report = push(&meta, &hub, &id).expect("the oversized blob syncs");

    assert_eq!(report.blobs, 1);
    assert_eq!(
        report.blob_parts, 2,
        "64 MiB parts, so 64 MiB + 4 KiB is two"
    );
    assert_eq!(report.packs, 1, "the small file still rode a pack");
    let state = hub.state.borrow();
    assert_eq!(state.head, Some(id));
    assert_eq!(
        state
            .objects
            .get(blob.as_bytes())
            .expect("the blob promoted")
            .len() as u64,
        blob_len
    );
}
