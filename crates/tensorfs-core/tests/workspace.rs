//! Behavioral coverage for the transactional workspace metadata layer: the
//! generation commit boundary, snapshot seal/round-trip identity, hardlink
//! and truncate semantics, leases, and the two-epoch GC protocol.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::{Path, PathBuf};

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::store::StoreError;
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::workspace::{GcReport, Mutation, WorkspaceError, WorkspaceStore};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-workspace-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn admitted(store: &WorkspaceStore, bytes: &[u8]) -> FileRecord {
    let object = store.store().put_bytes(bytes).unwrap();
    FileRecord::Data {
        digest: object.digest(),
        length: object.length(),
    }
}

fn data(record: &FileRecord) -> (ObjectDigest, u64) {
    match record {
        FileRecord::Data { digest, length } => (*digest, *length),
        FileRecord::Hole { .. } => panic!("expected a data record"),
    }
}

#[test]
fn a_generation_commit_is_atomic_and_ordered() {
    let root = TempRoot::new("commit");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();
    assert_eq!(store.head_generation("main").unwrap(), 0);

    let weights = admitted(&store, b"model weights");
    let generation = store
        .commit_generation(
            "main",
            &[
                Mutation::Mkdir {
                    path: "model".into(),
                },
                Mutation::CreateFile {
                    path: "model/weights.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![weights],
                },
            ],
        )
        .unwrap();
    assert_eq!(generation, 1);
    assert_eq!(store.head_generation("main").unwrap(), 1);

    // A failing batch moves nothing: the missing parent refuses the whole
    // commit even though the first mutation alone would have succeeded.
    let error = store
        .commit_generation(
            "main",
            &[
                Mutation::Mkdir {
                    path: "docs".into(),
                },
                Mutation::CreateFile {
                    path: "absent/child.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![],
                },
            ],
        )
        .unwrap_err();
    assert!(matches!(error, WorkspaceError::Missing(_)));
    assert_eq!(store.head_generation("main").unwrap(), 1);
    let sealed = store.seal_snapshot("main", None).unwrap();
    let tree = store.load_snapshot(&sealed).unwrap();
    assert!(
        !tree.entries().iter().any(|(path, _)| path == "docs"),
        "a refused batch must leave no partial mutation behind"
    );
}

#[test]
fn a_commit_refuses_unresident_and_corrupt_object_references() {
    let root = TempRoot::new("residency");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();

    let phantom = FileRecord::Data {
        digest: ObjectDigest::from_bytes([0xAB; 32]),
        length: 4,
    };
    let error = store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "phantom.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![phantom],
            }],
        )
        .unwrap_err();
    assert!(matches!(error, WorkspaceError::MissingObject { .. }));
    assert_eq!(store.head_generation("main").unwrap(), 0);
}

#[test]
fn sealed_snapshots_round_trip_identity_and_rebuild_equal_trees() {
    let root = TempRoot::new("seal");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();

    let shared = admitted(&store, b"shared layer bytes");
    let unique = admitted(&store, b"unique head bytes");
    store
        .commit_generation(
            "main",
            &[
                Mutation::Mkdir {
                    path: "model".into(),
                },
                Mutation::CreateFile {
                    path: "model/base.bin".into(),
                    executable: false,
                    planner: PlannerId::SafetensorsV1,
                    records: vec![shared.clone(), unique],
                },
                Mutation::Hardlink {
                    path: "model/alias.bin".into(),
                    target: "model/base.bin".into(),
                },
                Mutation::Symlink {
                    path: "latest".into(),
                    target: "model/base.bin".into(),
                },
                Mutation::Mkdir {
                    path: "empty".into(),
                },
            ],
        )
        .unwrap();

    let first = store.seal_snapshot("main", None).unwrap();
    let decoded = store.load_snapshot(&first).unwrap();
    assert_eq!(decoded.snapshot_id(), first);

    // Rebuilding a workspace from the stored blob alone reproduces the exact
    // identity: the snapshot, not the SQLite rows, carries the tree.
    store
        .create_workspace_from_snapshot("rebuilt", &first)
        .unwrap();
    let second = store.seal_snapshot("rebuilt", None).unwrap();
    assert_eq!(first, second);

    // Same facts committed to a completely fresh database: same identity.
    let other_root = TempRoot::new("seal-b");
    let other = WorkspaceStore::open(&other_root.0).unwrap();
    other.store().put_bytes(b"shared layer bytes").unwrap();
    other.store().put_bytes(b"unique head bytes").unwrap();
    let blob = store.load_snapshot(&first).unwrap().to_bytes();
    let reimported = tensorfs_core::tfm1::decode(&blob).unwrap();
    assert_eq!(reimported.snapshot_id(), first);
}

#[test]
fn renames_move_dirents_without_touching_object_identity() {
    let root = TempRoot::new("rename");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();
    let payload = admitted(&store, b"stable payload");
    let (digest, _) = data(&payload);

    store
        .commit_generation(
            "main",
            &[
                Mutation::Mkdir { path: "a".into() },
                Mutation::Mkdir { path: "b".into() },
                Mutation::CreateFile {
                    path: "a/file.bin".into(),
                    executable: true,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![payload],
                },
            ],
        )
        .unwrap();
    let before = store.seal_snapshot("main", None).unwrap();

    store
        .commit_generation(
            "main",
            &[Mutation::Rename {
                from: "a/file.bin".into(),
                to: "b/renamed.bin".into(),
            }],
        )
        .unwrap();
    let after = store.seal_snapshot("main", None).unwrap();

    assert_ne!(before, after, "a rename changes the manifest identity");
    let tree = store.load_snapshot(&after).unwrap();
    let entry = tree
        .entries()
        .iter()
        .find(|(path, _)| path == "b/renamed.bin")
        .expect("the renamed file exists");
    match &entry.1 {
        tensorfs_core::tfm1::Entry::File { records, .. } => match &records[0] {
            FileRecord::Data { digest: kept, .. } => {
                assert_eq!(*kept, digest, "renames never move object digests")
            }
            FileRecord::Hole { .. } => panic!("expected data"),
        },
        _ => panic!("expected a file entry"),
    }

    let error = store
        .commit_generation(
            "main",
            &[Mutation::Rename {
                from: "b".into(),
                to: "b/inside".into(),
            }],
        )
        .unwrap_err();
    assert!(matches!(error, WorkspaceError::RenameIntoSelf(_)));
}

#[test]
fn truncate_cuts_at_record_boundaries_shrinks_holes_and_refuses_mid_data() {
    let root = TempRoot::new("truncate");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();
    let first = admitted(&store, b"0123456789");
    let second = admitted(&store, b"abcdefghij");

    store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "sparse.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![
                    first.clone(),
                    FileRecord::Hole { length: 30 },
                    second.clone(),
                ],
            }],
        )
        .unwrap();

    // Boundary cut drops the tail records exactly.
    store
        .commit_generation(
            "main",
            &[Mutation::Truncate {
                path: "sparse.bin".into(),
                length: 40,
            }],
        )
        .unwrap();
    // A cut inside the hole shrinks it.
    store
        .commit_generation(
            "main",
            &[Mutation::Truncate {
                path: "sparse.bin".into(),
                length: 15,
            }],
        )
        .unwrap();
    // Extension grows a trailing hole rather than fabricating bytes.
    store
        .commit_generation(
            "main",
            &[Mutation::Truncate {
                path: "sparse.bin".into(),
                length: 100,
            }],
        )
        .unwrap();
    // A cut inside a data record cannot invent a shorter object.
    let error = store
        .commit_generation(
            "main",
            &[Mutation::Truncate {
                path: "sparse.bin".into(),
                length: 5,
            }],
        )
        .unwrap_err();
    assert!(matches!(error, WorkspaceError::MidRecordTruncate { .. }));

    let sealed = store.seal_snapshot("main", None).unwrap();
    let tree = store.load_snapshot(&sealed).unwrap();
    match &tree.entries()[0].1 {
        tensorfs_core::tfm1::Entry::File {
            logical_size,
            records,
            ..
        } => {
            assert_eq!(*logical_size, 100);
            assert_eq!(
                records,
                &vec![first, FileRecord::Hole { length: 90 }],
                "10 data bytes, then one merged 90-byte hole"
            );
        }
        _ => panic!("expected a file entry"),
    }
}

#[test]
fn sibling_case_fold_collisions_refuse_at_the_metadata_layer() {
    let root = TempRoot::new("fold");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();
    store
        .commit_generation(
            "main",
            &[Mutation::Mkdir {
                path: "Weights".into(),
            }],
        )
        .unwrap();
    let error = store
        .commit_generation(
            "main",
            &[Mutation::Mkdir {
                path: "weights".into(),
            }],
        )
        .unwrap_err();
    assert!(matches!(error, WorkspaceError::CaseFoldCollision(_)));
}

fn quiet_collect(store: &WorkspaceStore) -> GcReport {
    store.collect().unwrap()
}

#[test]
fn gc_needs_two_full_epochs_and_rescues_rereferenced_objects() {
    let root = TempRoot::new("gc");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();

    let keep = admitted(&store, b"kept forever");
    let (keep_digest, _) = data(&keep);
    let orphan = store.store().put_bytes(b"never referenced").unwrap();
    store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "kept.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![keep],
            }],
        )
        .unwrap();

    // Epoch 1 marks the orphan; nothing is deleted.
    let first = quiet_collect(&store);
    assert_eq!(first.newly_quarantined, 1);
    assert_eq!(first.deleted, 0);
    // Epoch 2: still only one epoch old; retained.
    let second = quiet_collect(&store);
    assert_eq!(second.deleted, 0);
    assert!(store.store().verify(&orphan.digest()).is_ok());
    // Epoch 3: two full epochs quarantined; deleted.
    let third = quiet_collect(&store);
    assert_eq!(third.deleted, 1);
    assert!(
        matches!(
            store.store().verify(&orphan.digest()),
            Err(StoreError::Missing { .. })
        ),
        "the orphan must be UNLINKED by GC, not merely unreadable"
    );
    assert!(store.store().verify(&keep_digest).is_ok());

    // Rescue: an object quarantined once but referenced before deletion
    // leaves quarantine untouched by later epochs.
    let rescue = store.store().put_bytes(b"rescued bytes").unwrap();
    quiet_collect(&store);
    store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "rescued.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![FileRecord::Data {
                    digest: rescue.digest(),
                    length: 13,
                }],
            }],
        )
        .unwrap();
    let after_rescue = quiet_collect(&store);
    assert_eq!(after_rescue.rescued, 1);
    quiet_collect(&store);
    quiet_collect(&store);
    assert!(store.store().verify(&rescue.digest()).is_ok());
}

#[test]
fn snapshot_and_lease_roots_pin_objects_against_collection() {
    let root = TempRoot::new("roots");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("main").unwrap();

    // A snapshot pins its digests even after the workspace forgets them.
    let sealed_bytes = admitted(&store, b"snapshot pinned");
    let (sealed_digest, _) = data(&sealed_bytes);
    store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "pinned.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![sealed_bytes],
            }],
        )
        .unwrap();
    let snapshot = store.seal_snapshot("main", None).unwrap();
    store
        .commit_generation(
            "main",
            &[Mutation::Unlink {
                path: "pinned.bin".into(),
            }],
        )
        .unwrap();
    for _ in 0..3 {
        quiet_collect(&store);
    }
    assert!(
        store.store().verify(&sealed_digest).is_ok(),
        "snapshot roots pin objects"
    );

    // An unlinked-open lease pins the orphaned inode's map the same way.
    let leased_bytes = admitted(&store, b"lease pinned");
    let (leased_digest, _) = data(&leased_bytes);
    store
        .commit_generation(
            "main",
            &[Mutation::CreateFile {
                path: "open.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![leased_bytes],
            }],
        )
        .unwrap();
    let lease = store
        .acquire_unlinked_lease("main", "open.bin", "test-holder")
        .unwrap();
    store
        .commit_generation(
            "main",
            &[Mutation::Unlink {
                path: "open.bin".into(),
            }],
        )
        .unwrap();
    for _ in 0..3 {
        quiet_collect(&store);
    }
    assert!(
        store.store().verify(&leased_digest).is_ok(),
        "an unlinked-open lease pins the object map"
    );

    // Releasing the last lease releases the pin; deletion then follows the
    // ordinary two-epoch protocol. The snapshot still pins its own digest.
    store.release_lease(lease).unwrap();
    store.delete_snapshot(&snapshot).unwrap();
    for _ in 0..3 {
        quiet_collect(&store);
    }
    assert!(
        matches!(
            store.store().verify(&leased_digest),
            Err(StoreError::Missing { .. })
        ),
        "the leased object must be UNLINKED once the lease is gone"
    );
    assert!(
        matches!(
            store.store().verify(&sealed_digest),
            Err(StoreError::Missing { .. })
        ),
        "the sealed object must be UNLINKED once the snapshot is gone"
    );
}

#[test]
fn deleting_a_workspace_unreferences_its_objects() {
    let root = TempRoot::new("delete");
    let store = WorkspaceStore::open(&root.0).unwrap();
    store.create_workspace("doomed").unwrap();
    let payload = admitted(&store, b"doomed payload");
    let (digest, _) = data(&payload);
    store
        .commit_generation(
            "doomed",
            &[Mutation::CreateFile {
                path: "payload.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![payload],
            }],
        )
        .unwrap();
    store.delete_workspace("doomed").unwrap();
    assert!(matches!(
        store.head_generation("doomed").unwrap_err(),
        WorkspaceError::UnknownWorkspace(_)
    ));
    for _ in 0..3 {
        quiet_collect(&store);
    }
    assert!(
        matches!(
            store.store().verify(&digest),
            Err(StoreError::Missing { .. })
        ),
        "the object must be UNLINKED by GC, not merely unreadable"
    );
}

// ---------------------------------------------------------------------------
// Corrupt-generation isolation
// ---------------------------------------------------------------------------

/// Damages one stored snapshot blob in place through a SECOND engine, so the
/// corruption is a fact on disk rather than a state the library was asked to
/// enter, and returns the bytes now resident.
type Corruption = fn(&[u8], &ObjectDigest) -> Vec<u8>;

fn corrupt_stored_snapshot(
    root: &Path,
    id: &SnapshotId,
    shape: Corruption,
    named: &ObjectDigest,
) -> Vec<u8> {
    let connection =
        rusqlite::Connection::open(root.join("metadata.sqlite3")).expect("rusqlite opens");
    let honest: Vec<u8> = connection
        .query_row(
            "SELECT blob FROM snapshots WHERE id = ?1",
            rusqlite::params![id.as_bytes().as_slice()],
            |row| row.get(0),
        )
        .expect("the snapshot row reads");
    assert_eq!(
        SnapshotId::of(&honest),
        *id,
        "the fixture's own bytes must hash to their id before corruption"
    );
    let damaged = shape(&honest, named);
    assert_ne!(damaged, honest, "the corruption must change something");
    assert_ne!(
        SnapshotId::of(&damaged),
        *id,
        "the damaged bytes must stop hashing to their id, or nothing is under test"
    );
    connection
        .execute(
            "UPDATE snapshots SET blob = ?1 WHERE id = ?2",
            rusqlite::params![damaged, id.as_bytes().as_slice()],
        )
        .expect("the corruption writes");
    damaged
}

/// Flips the low bit of the last byte of `named`'s digest where it appears in
/// the manifest.
///
/// This is the DANGEROUS shape, and the reason this arm leads with it: the
/// result is still a grammatically perfect TFM1 manifest that decodes into a
/// plausible tree — it simply names an object nobody ever admitted. A store
/// that dropped its id re-check would hand that tree back as fact, and would
/// stop counting the real object as a GC root.
fn flip_a_named_digest(honest: &[u8], named: &ObjectDigest) -> Vec<u8> {
    let needle = named.as_bytes();
    let at = honest
        .windows(32)
        .position(|window| window == needle.as_slice())
        .expect("a sealed manifest carries the digests it names");
    let mut damaged = honest.to_vec();
    damaged[at + 31] ^= 0x01;
    damaged
}

fn garbage_of_the_same_length(honest: &[u8], _named: &ObjectDigest) -> Vec<u8> {
    honest
        .iter()
        .enumerate()
        .map(|(index, byte)| byte ^ 0xA5_u8.wrapping_add(index as u8))
        .collect()
}

fn scalar(root: &Path, sql: &str) -> i64 {
    let connection =
        rusqlite::Connection::open(root.join("metadata.sqlite3")).expect("rusqlite opens");
    connection
        .query_row(sql, [], |row| row.get(0))
        .expect("the scalar reads")
}

/// A corrupt stored generation is isolated: every reader refuses with a typed
/// error, the pass that meets it while holding the metadata WRITE transaction
/// rolls that transaction back whole, and the last known-good snapshot — and
/// the live head — are untouched.
///
/// All three `SnapshotCorrupt` raise sites are exercised against the same
/// damaged blob: `load_snapshot`, `acquire_pending_sync_lease`, and `collect`.
/// `collect` is the interesting one. It reads snapshot blobs INSIDE the
/// `BEGIN IMMEDIATE` that also advances the GC epoch, marks quarantine and
/// unlinks bytes, so a refusal there must abort the pass whole rather than
/// leave a half-collected store — a snapshot that can no longer be read must
/// never stop rooting the objects it names.
#[test]
fn a_corrupt_generation_is_isolated_and_the_last_known_good_snapshot_survives() {
    let shapes: [(&str, Corruption); 2] = [
        ("a flipped bit inside a named digest", flip_a_named_digest),
        ("whole-blob garbage", garbage_of_the_same_length),
    ];
    for (index, (shape_name, shape)) in shapes.into_iter().enumerate() {
        let root = TempRoot::new(&format!("corrupt-generation-{index}"));
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.create_workspace("main").unwrap();

        // Generation 1, sealed: the last known-good snapshot.
        let early = admitted(&store, b"the known-good generation's bytes");
        let (early_digest, _) = data(&early);
        store
            .commit_generation(
                "main",
                &[Mutation::CreateFile {
                    path: "early.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![early],
                }],
            )
            .unwrap();
        let good = store.seal_snapshot("main", None).unwrap();

        // Generation 2, sealed: the one about to be damaged on disk.
        let late = admitted(&store, b"the doomed generation's bytes");
        let (late_digest, _) = data(&late);
        store
            .commit_generation(
                "main",
                &[Mutation::CreateFile {
                    path: "late.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![late],
                }],
            )
            .unwrap();
        let doomed = store.seal_snapshot("main", None).unwrap();
        assert_ne!(good, doomed);
        let head_before = store.head_generation("main").unwrap();
        drop(store);

        let epoch_before = scalar(&root.0, "SELECT epoch FROM gc_state WHERE id = 1");
        let damaged = corrupt_stored_snapshot(&root.0, &doomed, shape, &late_digest);

        let store = WorkspaceStore::open(&root.0).unwrap();

        // Site 1 — the plain reader.
        let error = store
            .load_snapshot(&doomed)
            .expect_err("a corrupt snapshot must never load");
        assert!(
            matches!(error, WorkspaceError::SnapshotCorrupt),
            "{shape_name}: load_snapshot produced {error:?}"
        );

        // Site 2 — the transfer pin. It must refuse BEFORE inserting the
        // lease, so a refused pin leaves no root behind.
        let error = store
            .acquire_pending_sync_lease(&doomed, "test-holder")
            .expect_err("a corrupt snapshot must never be pinned for transfer");
        assert!(
            matches!(error, WorkspaceError::SnapshotCorrupt),
            "{shape_name}: acquire_pending_sync_lease produced {error:?}"
        );

        // Site 3 — GC, which meets the corruption holding the write
        // transaction.
        let error = store
            .collect()
            .expect_err("GC must refuse a mark pass whose roots it cannot read");
        assert!(
            matches!(error, WorkspaceError::SnapshotCorrupt),
            "{shape_name}: collect produced {error:?}"
        );

        // The last known-good snapshot still loads and still names resident
        // bytes.
        assert_eq!(
            store
                .load_snapshot(&good)
                .expect("known-good loads")
                .snapshot_id(),
            good,
            "{shape_name}"
        );
        assert!(
            store.store().verify(&early_digest).is_ok(),
            "{shape_name}: the known-good generation's bytes were collected"
        );
        // The corrupt generation's bytes are resident too: a pass that refused
        // must not have quarantined or deleted anything.
        assert!(
            store.store().verify(&late_digest).is_ok(),
            "{shape_name}: a refused GC pass still touched the object store"
        );

        // The live head is untouched by any of the three refusals.
        assert_eq!(
            store.head_generation("main").unwrap(),
            head_before,
            "{shape_name}: the workspace head moved"
        );
        let tree = store.head_tree("main").unwrap();
        let mut paths: Vec<&str> = tree
            .entries()
            .iter()
            .map(|(path, _)| path.as_str())
            .collect();
        paths.sort_unstable();
        assert_eq!(paths, ["early.bin", "late.bin"], "{shape_name}");
        drop(store);

        // GC's write transaction rolled back WHOLE: the epoch did not advance
        // and nothing entered quarantine. The refused pin left no lease and no
        // pinned digest.
        assert_eq!(
            scalar(&root.0, "SELECT epoch FROM gc_state WHERE id = 1"),
            epoch_before,
            "{shape_name}: the refused GC pass still advanced the epoch"
        );
        for (table, complaint) in [
            ("gc_quarantine", "the refused GC pass still quarantined"),
            ("leases", "the refused pin still inserted a lease"),
            ("lease_objects", "the refused pin still pinned digests"),
        ] {
            assert_eq!(
                scalar(&root.0, &format!("SELECT COUNT(*) FROM {table}")),
                0,
                "{shape_name}: {complaint}"
            );
        }
        // The damaged bytes are still there: a refusal reports, it does not
        // repair, and it does not destroy the evidence.
        let connection =
            rusqlite::Connection::open(root.0.join("metadata.sqlite3")).expect("rusqlite opens");
        let resident: Vec<u8> = connection
            .query_row(
                "SELECT blob FROM snapshots WHERE id = ?1",
                rusqlite::params![doomed.as_bytes().as_slice()],
                |row| row.get(0),
            )
            .expect("the damaged row is still there");
        assert_eq!(
            resident, damaged,
            "{shape_name}: the evidence was rewritten"
        );
        drop(connection);

        // The operator recovery path: dropping the unreadable generation
        // un-wedges GC, and the surviving snapshot still roots its own bytes.
        let store = WorkspaceStore::open(&root.0).unwrap();
        store.delete_snapshot(&doomed).unwrap();
        for _ in 0..3 {
            store
                .collect()
                .expect("GC runs again once the unreadable generation is gone");
        }
        assert!(
            store.store().verify(&early_digest).is_ok(),
            "{shape_name}: the known-good snapshot must still root its bytes"
        );
        assert_eq!(
            store.load_snapshot(&good).unwrap().snapshot_id(),
            good,
            "{shape_name}"
        );
    }
}
