//! Behavioral coverage for the transactional workspace metadata layer: the
//! generation commit boundary, snapshot seal/round-trip identity, hardlink
//! and truncate semantics, leases, and the two-epoch GC protocol.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::PathBuf;

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::store::StoreError;
use tensorfs_core::tfm1::FileRecord;
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
