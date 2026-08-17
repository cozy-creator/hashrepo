//! The projected layout: 0444 objects, snapshot symlink trees, pointer
//! stubs, atomic refs, manifests as blobs at their own id, and deletion that
//! takes the tree and only the tree.
//!
//! Every assertion here is made through a NAIVE consumer wherever one exists
//! — `std::fs::read` through the projected path, `OpenOptions` writing at it,
//! `read_link` on a ref — because the claim is about what an ordinary tool
//! sees, not about what our own reader can reconstruct.

#![cfg(any(unix, windows))]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs_core::layout::{Layout, Removal, STUB_MAGIC, stub_bytes};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::planner::PlannerId;
use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceError, WorkspaceStore};

mod harness;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-projection-{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
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

/// A minimal well-formed safetensors file, so the tree has a real
/// tensor-planner entry to project as a stub.
fn safetensors(tensors: &[(&str, u64, u8)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, (name, length, _)) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let end = offset + length;
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{length}],\"data_offsets\":[{offset},{end}]}}"
        ));
        offset = end;
    }
    header.push('}');
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for (_, length, fill) in tensors {
        file.extend(std::iter::repeat_n(*fill, *length as usize));
    }
    file
}

fn blob(store: &WorkspaceStore, path: &str, bytes: &[u8]) -> Mutation {
    let object = store.store().put_bytes(bytes).unwrap();
    Mutation::CreateFile {
        path: path.to_owned(),
        executable: false,
        planner: PlannerId::BlobV1,
        records: vec![FileRecord::Data {
            digest: object.digest(),
            length: object.length(),
        }],
    }
}

/// Four files whose bytes all differ from each other, three of them at
/// different depths, so a projection that crossed two entries or resolved a
/// link at the wrong depth cannot pass by accident.
struct Fixture {
    config: Vec<u8>,
    vocab: Vec<u8>,
    clip: Vec<u8>,
    weights: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            config: br#"{"model_type":"llama","hidden_size":4096}"#.to_vec(),
            vocab: b"<pad>\n<unk>\nhello\nworld\n".to_vec(),
            clip: b"RIFF....WEBPonly-in-the-clips-directory".to_vec(),
            weights: safetensors(&[("blk.0.attn", 4096, 0xA1), ("blk.0.norm", 512, 0xC3)]),
        };
        let all = [
            &fixture.config,
            &fixture.vocab,
            &fixture.clip,
            &fixture.weights,
        ];
        for (index, left) in all.iter().enumerate() {
            for right in all.iter().skip(index + 1) {
                assert_ne!(left, right, "fixture files must differ from each other");
            }
        }
        fixture
    }
}

/// Seals a workspace holding the fixture tree and returns its snapshot id.
fn sealed(store: &WorkspaceStore, fixture: &Fixture) -> SnapshotId {
    store.create_workspace("main").unwrap();
    let mutations = vec![
        Mutation::Mkdir {
            path: "clips".into(),
        },
        Mutation::Mkdir {
            path: "clips/train".into(),
        },
        Mutation::Mkdir {
            path: "tokenizer".into(),
        },
        blob(store, "config.json", &fixture.config),
        blob(store, "tokenizer/vocab.txt", &fixture.vocab),
        blob(store, "clips/train/v.webm", &fixture.clip),
        blob(store, "model.safetensors", &fixture.weights),
    ];
    store.commit_generation("main", &mutations).unwrap();
    store.seal_snapshot("main", None).unwrap()
}

fn skip_without_symlinks(store: &WorkspaceStore) -> bool {
    if store.store().supports_symlinks() {
        return false;
    }
    eprintln!("skipped: this filesystem has no symlinks, the projection copies instead");
    true
}

// ---------------------------------------------------------------------------
// Immutability
// ---------------------------------------------------------------------------

/// Objects install 0444, and the consequence a consumer actually meets: an
/// ordinary `open("ab")` through a projected snapshot symlink is refused by
/// the operating system.
///
/// Red proof: drop the `set_read_only` call in `ObjectWriter::install` and the
/// append below succeeds, taking both assertions with it.
#[test]
#[cfg(unix)]
fn a_write_through_a_projected_symlink_is_refused_by_the_object_mode() {
    use std::os::unix::fs::PermissionsExt as _;

    let root = TempRoot::new("immutable");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    if skip_without_symlinks(&store) {
        return;
    }
    let tree = store.project_snapshot(&id).unwrap();

    let projected = tree.join("config.json");
    let resident = fs::canonicalize(&projected).unwrap();
    assert_eq!(
        fs::metadata(&resident).unwrap().permissions().mode() & 0o777,
        0o444,
        "an admitted object must install read-only"
    );

    let error = fs::OpenOptions::new()
        .append(true)
        .open(&projected)
        .expect_err("appending through the symlink must be refused");
    assert_eq!(
        error.kind(),
        std::io::ErrorKind::PermissionDenied,
        "expected EACCES, got {error:?}"
    );
    assert_eq!(
        fs::read(&projected).unwrap(),
        fixture.config,
        "the refused write must not have reached the bytes"
    );
}

// ---------------------------------------------------------------------------
// The tree builder
// ---------------------------------------------------------------------------

/// A projected tree is directories, relative symlinks and stubs — and a naive
/// `read` through a nested symlink returns the blob's bytes byte for byte.
#[test]
fn a_projected_tree_serves_every_blob_byte_identically_through_naive_reads() {
    let root = TempRoot::new("tree");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let tree = store.project_snapshot(&id).unwrap();

    assert!(tree.join("clips").join("train").is_dir());
    assert!(tree.join("tokenizer").is_dir());

    for (path, expected) in [
        ("config.json", &fixture.config),
        ("tokenizer/vocab.txt", &fixture.vocab),
        ("clips/train/v.webm", &fixture.clip),
    ] {
        let projected = tree.join(path);
        assert_eq!(
            &fs::read(&projected).unwrap(),
            expected,
            "{path}: a naive read through the projection must be byte-identical"
        );
    }

    // The tensor container is a stub, not a symlink and not absence.
    let stub = tree.join("model.safetensors");
    let bytes = fs::read(&stub).unwrap();
    assert!(
        bytes.starts_with(STUB_MAGIC),
        "a tensor-planner file projects as a pointer stub"
    );
    assert_ne!(
        bytes, fixture.weights,
        "a stub must never be the tensor bytes"
    );

    // Zero bytes copied: every non-directory entry in the tree together is
    // smaller than the single file the stub stands for.
    let projected_bytes: u64 = walk(&tree)
        .iter()
        .map(|path| fs::symlink_metadata(path).unwrap())
        .filter(|metadata| !metadata.is_dir())
        .map(|metadata| metadata.len())
        .sum();
    assert!(
        projected_bytes < fixture.weights.len() as u64,
        "the projection copied bytes: {projected_bytes} of them"
    );
}

/// Symlink targets are RELATIVE at every depth, so the store relocates as a
/// unit. Four levels deep is the case the design prototyped.
#[test]
#[cfg(unix)]
fn projected_symlinks_are_relative_at_every_depth() {
    let root = TempRoot::new("relative");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    if skip_without_symlinks(&store) {
        return;
    }
    let tree = store.project_snapshot(&id).unwrap();

    for (path, expected_depth) in [("config.json", 2), ("clips/train/v.webm", 4)] {
        let target = fs::read_link(tree.join(path)).unwrap();
        assert!(
            target.is_relative(),
            "{path}: an absolute target would pin the store to a mount path"
        );
        let ups = target
            .components()
            .filter(|component| component.as_os_str() == "..")
            .count();
        assert_eq!(ups, expected_depth, "{path}: wrong link depth");
        assert!(
            target.to_str().unwrap().contains("objects/sha256/"),
            "{path}: a projected blob points into the object tree"
        );
    }

    // Relocating the whole store keeps every link resolving.
    let moved = root.0.parent().unwrap().join(format!(
        "tensorfs-projection-moved-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&moved);
    fs::rename(&root.0, &moved).unwrap();
    let relocated = moved
        .join("snapshots")
        .join(tree.file_name().unwrap())
        .join("clips/train/v.webm");
    assert_eq!(
        fs::read(&relocated).unwrap(),
        fixture.clip,
        "relative links must survive relocating the store"
    );
    let _ = fs::remove_dir_all(&moved);
}

/// Two snapshots sharing a blob share its INODE: the projection deduplicates
/// rather than copying, which is the whole point of the layout.
#[test]
#[cfg(unix)]
fn two_snapshots_sharing_a_blob_project_onto_one_inode() {
    use std::os::unix::fs::MetadataExt as _;

    let root = TempRoot::new("shared");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let first = sealed(&store, &fixture);
    if skip_without_symlinks(&store) {
        return;
    }

    // A second snapshot with the same config.json and a different neighbour.
    store
        .commit_generation("main", &[blob(&store, "extra.txt", b"only in the second")])
        .unwrap();
    let second = store.seal_snapshot("main", None).unwrap();
    assert_ne!(first, second);

    let one = store.project_snapshot(&first).unwrap().join("config.json");
    let two = store.project_snapshot(&second).unwrap().join("config.json");
    assert_eq!(
        fs::metadata(&one).unwrap().ino(),
        fs::metadata(&two).unwrap().ino(),
        "a shared blob must be stored once and pointed at twice"
    );
    assert_eq!(fs::read(&one).unwrap(), fixture.config);
}

/// A filesystem without symlinks keeps correctness and loses dedup: the same
/// tree builder writes copies. Exercised directly so the Windows fallback is
/// covered on the Linux runner too.
#[test]
fn the_no_symlink_fallback_projects_copies_that_read_identically() {
    let root = TempRoot::new("nolinks");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let snapshot = store.load_snapshot(&id).unwrap();

    // A second store root that reports no symlink support, sharing nothing
    // with the first but the manifest.
    let copies = TempRoot::new("nolinks-target");
    let plain = tensorfs_core::store::ObjectStore::open(&copies.0).unwrap();
    for (_, entry) in snapshot.entries() {
        if let Entry::File { body, .. } = entry {
            for record in body.records().iter() {
                if let FileRecord::Data { digest, length } = record {
                    let bytes = read_object(store.store(), digest, *length);
                    plain.put_bytes(&bytes).unwrap();
                }
            }
        }
    }
    let tree = Layout::without_symlinks(&plain).project(&snapshot).unwrap();

    assert_eq!(fs::read(tree.join("config.json")).unwrap(), fixture.config);
    assert_eq!(
        fs::read(tree.join("clips/train/v.webm")).unwrap(),
        fixture.clip
    );
    assert!(
        !fs::symlink_metadata(tree.join("config.json"))
            .unwrap()
            .file_type()
            .is_symlink(),
        "the fallback must project a real file, not a link"
    );
    assert!(
        fs::read(tree.join("model.safetensors"))
            .unwrap()
            .starts_with(STUB_MAGIC),
        "a tensor container is still a stub without symlinks"
    );
}

// ---------------------------------------------------------------------------
// Refs
// ---------------------------------------------------------------------------

/// What the ref swap guarantees, measured under two consumer shapes running
/// against 2,000 `rename(2)`s.
///
/// * **resolvers** call `read_ref` in a tight loop. On POSIX the ref is never
///   absent and never unreadable; on Windows a miss is allowed but must HEAL
///   — see below — and either way they see BOTH ids, so the swap really was
///   concurrent.
/// * **holders** resolve the ref once and then read inside the tree they
///   resolved. A swap never invalidates a resolved tree: trees are immutable
///   and a swap deletes nothing.
///
/// Both run until the writer finishes its swaps — progress toward a goal, not
/// a deadline — and each reports what it observed, so a vacuous run fails
/// rather than passing quietly.
///
/// Red proof: replace `set_ref`'s rename with remove-then-create and the
/// resolvers see the ref absent within a few hundred iterations.
///
/// This test is why refs are files and not symlinks. As symlinks it failed on
/// two platforms for two different real reasons: `readlink` on a name being
/// renamed transiently returns EINVAL on APFS (~2 in 45,000 reads, while the
/// name itself was never once absent), and a consumer re-walking
/// `refs/<name>/<file>` transiently returns ENOENT on Linux, because a
/// multi-component walk THROUGH a renamed name is not one observation. Both
/// of the symlink's advantages fail under exactly the concurrency a ref is
/// for. `open`-then-`read` does not — on POSIX.
///
/// **Windows gets a weaker contract, because its rename cannot give this one**
/// (#103). A superseding rename is not atomic to a concurrent opener: measured
/// on CI over 600 runs of this test, ~2 opens per million transiently returned
/// ERROR_FILE_NOT_FOUND or ERROR_ACCESS_DENIED, and asking for the rename
/// POSIX defines — `FileRenameInfoEx` with `FILE_RENAME_FLAG_POSIX_SEMANTICS`
/// — measured no better. So what Windows promises, and what this asserts, is
/// that a miss is transient: reading again while the swaps run resolves it.
/// The reading again is the CALLER's, here and everywhere: a retry inside
/// `read_ref` can only wait on a peer's progress, and `scrub` proved a peer
/// can be waiting on the reader (#103).
#[test]
fn a_ref_swap_is_atomic_under_concurrent_readers() {
    const SWAPS: usize = 2_000;
    const READERS_PER_CLASS: usize = 2;

    let root = TempRoot::new("refswap");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let first = sealed(&store, &fixture);
    store
        .commit_generation(
            "main",
            &[blob(&store, "second.txt", b"the second snapshot")],
        )
        .unwrap();
    let second = store.seal_snapshot("main", None).unwrap();
    assert_ne!(first, second);

    let first_tree = store.project_snapshot(&first).unwrap();
    store.project_snapshot(&second).unwrap();

    let layout = store.layout();
    layout.set_ref("main", &first).unwrap();
    let held = first_tree.join("config.json");

    // Raised when the swap loop LEAVES, panic included: readers spinning on a
    // flag only a healthy writer sets turn any failure into a hang (#109).
    let stop = harness::StopFlag::new();
    let (resolved, healed, resolve_failures, distinct, holds, hold_failures) =
        std::thread::scope(|scope| {
            let resolvers: Vec<_> = (0..READERS_PER_CLASS)
                .map(|_| {
                    let stop = &stop;
                    let layout = &layout;
                    scope.spawn(move || {
                        let mut good = 0_u64;
                        let mut healed = 0_u64;
                        let mut failures: Vec<String> = Vec::new();
                        let mut seen = std::collections::HashSet::new();
                        while !stop.raised() {
                            let what = match layout.read_ref("main") {
                                Ok(Some(id)) => {
                                    seen.insert(id);
                                    good += 1;
                                    continue;
                                }
                                Ok(None) => "the ref was absent".to_owned(),
                                Err(error) => format!("{error:?}"),
                            };
                            // POSIX renames atomically for an opener, so a
                            // miss there is a defect. Windows cannot, and
                            // that is the contract it gets (#103) — but only
                            // a miss that HEALS: reading again while the
                            // swaps run has to resolve the ref.
                            if cfg!(windows) && resolves_again(layout, stop) {
                                healed += 1;
                            } else {
                                record(&mut failures, what);
                            }
                        }
                        (good, healed, failures, seen)
                    })
                })
                .collect();
            let holders: Vec<_> = (0..READERS_PER_CLASS)
                .map(|_| {
                    let stop = &stop;
                    let held = held.clone();
                    let expected = &fixture.config;
                    scope.spawn(move || {
                        let mut good = 0_u64;
                        let mut failures: Vec<String> = Vec::new();
                        while !stop.raised() {
                            match fs::read(&held) {
                                Ok(bytes) if bytes == *expected => good += 1,
                                Ok(_) => {
                                    record(&mut failures, "a resolved tree changed".to_owned());
                                }
                                Err(error) => record(&mut failures, format!("{error:?}")),
                            }
                        }
                        (good, failures)
                    })
                })
                .collect();

            stop.racing(|| {
                for index in 0..SWAPS {
                    let target = if index % 2 == 0 { second } else { first };
                    layout.set_ref("main", &target).unwrap();
                }
            });

            let mut resolved = 0_u64;
            let mut healed = 0_u64;
            let mut resolve_failures: Vec<String> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for resolver in resolvers {
                let (good, their_healed, failures, their_seen) = resolver.join().unwrap();
                resolved += good;
                healed += their_healed;
                resolve_failures.extend(failures);
                seen.extend(their_seen);
            }
            let mut holds = 0_u64;
            let mut hold_failures: Vec<String> = Vec::new();
            for holder in holders {
                let (good, failures) = holder.join().unwrap();
                holds += good;
                hold_failures.extend(failures);
            }
            (
                resolved,
                healed,
                resolve_failures,
                seen.len(),
                holds,
                hold_failures,
            )
        });

    assert!(
        resolve_failures.is_empty(),
        "the ref was absent or unreadable across {SWAPS} swaps ({resolved} good \
         reads, {healed} Windows misses that resolved on a retry): \
         {resolve_failures:?}"
    );
    assert!(
        hold_failures.is_empty(),
        "a swap invalidated an already-resolved tree ({holds} good reads): \
         {hold_failures:?}"
    );
    assert!(
        resolved > 0 && holds > 0,
        "a reader class never ran ({resolved} resolved, {holds} held); the \
         concurrency claim would be vacuous"
    );
    assert_eq!(
        distinct, 2,
        "the resolvers saw {distinct} distinct ids; the swap was not actually concurrent"
    );
    assert_eq!(layout.read_ref("main").unwrap(), Some(first));
}

/// Reads until the ref resolves or the swapping stops — whether the miss that
/// sent us here was a swap in flight rather than a ref that is gone.
///
/// The wait is on the swap loop's own state and never on a clock: once `done`
/// is set the last `set_ref` has returned, so a read that still misses is
/// missing for real.
fn resolves_again(layout: &Layout<'_>, stop: &harness::StopFlag) -> bool {
    loop {
        if let Ok(Some(_)) = layout.read_ref("main") {
            return true;
        }
        if stop.raised() {
            return false;
        }
    }
}

/// Keeps a bounded, deduplicated sample of what went wrong, so a failure
/// message names the shape without printing a million identical lines.
fn record(failures: &mut Vec<String>, what: String) {
    if failures.len() < 4 && !failures.contains(&what) {
        failures.push(what);
    }
}

/// A ref is a plain text file a naive tool reads with `cat`: one lowercase
/// id and a line feed, naming a tree that is there.
#[test]
fn a_ref_is_plain_text_naming_a_projected_tree() {
    let root = TempRoot::new("reffile");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let tree = store.project_snapshot(&id).unwrap();
    store.layout().set_ref("main", &id).unwrap();

    let reference = root.0.join("refs").join("main");
    let text = fs::read_to_string(&reference).unwrap();
    assert_eq!(text, format!("{id}\n"), "a ref is the id and a line feed");
    assert!(!fs::symlink_metadata(&reference).unwrap().is_symlink());

    // What a consumer does with it: resolve once, then use the tree path.
    let resolved = root.0.join("snapshots").join(text.trim());
    assert_eq!(resolved, tree);
    assert_eq!(
        fs::read(resolved.join("tokenizer/vocab.txt")).unwrap(),
        fixture.vocab
    );
    assert_eq!(store.layout().read_ref("main").unwrap(), Some(id));
    assert_eq!(store.layout().read_ref("absent").unwrap(), None);
}

#[test]
fn ref_names_that_could_escape_the_refs_directory_are_refused() {
    let root = TempRoot::new("refnames");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let layout = store.layout();
    for name in ["", ".", "..", "../escape", "heads/main", ".hidden"] {
        assert!(
            layout.set_ref(name, &id).is_err(),
            "{name:?} must not be usable as a ref name"
        );
    }
}

// ---------------------------------------------------------------------------
// Manifests as blobs at their own id
// ---------------------------------------------------------------------------

/// The manifest bytes live in `objects/` at the snapshot id, the DB row is a
/// roots index carrying no blob, and the pin taken in one transaction
/// protects the immutable object it names against a concurrent delete plus a
/// full GC pass.
///
/// Red proof: drop the manifest digest from `acquire_pending_sync_lease`'s
/// pinned set and the manifest object is collected below.
#[test]
fn a_manifest_is_an_object_at_its_own_id_that_a_pin_protects_from_gc() {
    let root = TempRoot::new("manifest");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);

    // The bytes are resident at the id, and they are the manifest.
    let manifest = ObjectDigest::from_bytes(*id.as_bytes());
    let path = root
        .0
        .join("objects")
        .join("sha256")
        .join(&id.to_string()[..2])
        .join(&id.to_string()[2..4])
        .join(id.to_string());
    assert!(path.is_file(), "the manifest must be an object at its id");
    assert_eq!(
        tensorfs_core::tfm1::decode(&fs::read(&path).unwrap())
            .unwrap()
            .snapshot_id(),
        id
    );
    assert_eq!(
        store.store().verify(&manifest).unwrap(),
        path.metadata().unwrap().len()
    );

    // The row is an index, not a second copy.
    let columns = snapshot_columns(&root.0);
    assert!(
        !columns.contains(&"blob".to_owned()),
        "manifest bytes must have left the database: {columns:?}"
    );

    // Pin in one transaction, then delete the root and collect hard.
    let (lease, pinned) = store.acquire_pending_sync_lease(&id, "transfer").unwrap();
    assert_eq!(pinned.snapshot_id(), id);
    store.delete_snapshot(&id).unwrap();
    assert!(matches!(
        store.acquire_pending_sync_lease(&id, "late"),
        Err(WorkspaceError::UnknownSnapshot(_))
    ));
    for _ in 0..3 {
        store.collect().unwrap();
    }
    assert!(
        store.store().verify(&manifest).is_ok(),
        "GC took the manifest the transfer is still reading"
    );
    let config = pinned
        .entries()
        .iter()
        .find_map(|(path, entry)| match entry {
            Entry::File { body, .. } if path == "config.json" => Some(body.records().to_vec()),
            _ => None,
        })
        .unwrap();
    for record in &config {
        if let FileRecord::Data { digest, length } = record {
            assert_eq!(read_object(store.store(), digest, *length), fixture.config);
        }
    }

    // Released, the pin stops being a root and the bytes go.
    store.release_lease(lease).unwrap();
    for _ in 0..3 {
        store.collect().unwrap();
    }
    assert!(
        store.store().verify(&manifest).is_err(),
        "an unpinned, unrooted manifest must be collectable like any object"
    );
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

/// Deleting a snapshot removes its tree and its refs — and nothing else. A
/// blob shared with a surviving snapshot stays readable through the survivor.
#[test]
fn deleting_a_snapshot_removes_its_tree_and_only_its_tree() {
    let root = TempRoot::new("delete");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let first = sealed(&store, &fixture);
    store
        .commit_generation(
            "main",
            &[blob(&store, "extra.txt", b"only in the survivor")],
        )
        .unwrap();
    let second = store.seal_snapshot("main", None).unwrap();

    let doomed = store.project_snapshot(&first).unwrap();
    let survivor = store.project_snapshot(&second).unwrap();
    let layout = store.layout();
    layout.set_ref("doomed", &first).unwrap();
    layout.set_ref("survivor", &second).unwrap();

    store.delete_snapshot(&first).unwrap();

    assert!(!doomed.exists(), "the deleted snapshot's tree survived");
    assert_eq!(
        layout.read_ref("doomed").unwrap(),
        None,
        "a ref pointing at a deleted snapshot must be dropped"
    );
    assert_eq!(layout.read_ref("survivor").unwrap(), Some(second));
    assert!(survivor.is_dir(), "the peer tree was collateral damage");
    assert_eq!(
        fs::read(survivor.join("config.json")).unwrap(),
        fixture.config,
        "a blob shared with the deleted snapshot must stay readable"
    );
    assert!(matches!(
        store.delete_snapshot(&first),
        Err(WorkspaceError::UnknownSnapshot(_))
    ));
}

/// A removal gives the NAME up first and deletes the bytes second, so a
/// reader holding one file open cannot keep a tree projected or a ref alive,
/// and the bytes it is still holding are taken by the next reap.
///
/// Windows is where this is load-bearing and where it was red: there
/// `remove_dir_all` of a tree with an open file inside refuses with
/// `ERROR_ACCESS_DENIED`, and so does the loser of a removal race — which is
/// how a scrub racing a deletion failed for doing exactly its job, and how
/// that failure became a three-hour hang (#109). POSIX unlinks the name and
/// keeps the bytes for the reader, which is the behaviour asserted here for
/// both.
#[test]
fn a_removal_gives_up_the_name_while_a_reader_still_holds_the_bytes() {
    let root = TempRoot::new("unlink-held");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let layout = store.layout();

    let tree = store.project_snapshot(&id).unwrap();
    layout.set_ref("held", &id).unwrap();
    let ref_path = layout.refs_dir().join("held");
    let held_file = fs::File::open(tree.join("config.json")).unwrap();
    let held_ref = fs::File::open(&ref_path).unwrap();

    // Neither removal may fail: an open handle is a normal state of the
    // store, not an error in the caller.
    let tree_outcome = layout.remove_tree(&id).unwrap();
    let ref_outcome = layout.remove_ref("held").unwrap();
    if cfg!(unix) {
        assert_eq!(
            (tree_outcome, ref_outcome),
            (Removal::Taken, Removal::Taken),
            "POSIX unlinks a held name and keeps the bytes for the reader"
        );
    }

    // Whatever each said has to be true on disk: a name reported taken is
    // gone, and a deferred one is untouched — never a claim that missed.
    for (outcome, path) in [(tree_outcome, &tree), (ref_outcome, &ref_path)] {
        match outcome {
            Removal::Taken | Removal::Absent => {
                assert!(
                    !path.exists(),
                    "{path:?} outlived a removal that claimed it"
                )
            }
            Removal::Deferred => {
                assert!(path.exists(), "a deferred removal took {path:?} anyway")
            }
        }
    }
    if tree_outcome.taken() {
        assert!(!layout.tree_ids().unwrap().contains(&id));
        // The name is free the instant it is given up: re-projecting works
        // while the old bytes are still held open.
        assert_eq!(store.project_snapshot(&id).unwrap(), tree);
    }
    if ref_outcome.taken() {
        assert_eq!(layout.read_ref("held").unwrap(), None);
        assert!(!layout.ref_names().unwrap().contains(&"held".to_owned()));
    }

    // Once the reader lets go, one retry finishes whatever was deferred — no
    // waiting on a clock, and nothing left stranded once the reap runs.
    drop(held_file);
    drop(held_ref);
    assert_ne!(layout.remove_tree(&id).unwrap(), Removal::Deferred);
    assert_ne!(layout.remove_ref("held").unwrap(), Removal::Deferred);
    assert!(!tree.exists());
    assert_eq!(layout.read_ref("held").unwrap(), None);
    layout.reap_scratch().unwrap();
    for directory in [layout.snapshots_dir(), layout.refs_dir()] {
        for entry in fs::read_dir(directory).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".removed-"),
                "a reap left unreachable scratch behind: {name}"
            );
        }
    }
}

/// Projection is idempotent and re-derivable: a deleted tree comes back byte
/// for byte from the manifest alone, which is what makes it disposable.
#[test]
fn a_deleted_tree_re_projects_identically_from_the_manifest() {
    let root = TempRoot::new("reproject");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);

    let tree = store.project_snapshot(&id).unwrap();
    let before = fingerprint(&tree);
    assert!(store.layout().remove_tree(&id).unwrap().taken());
    assert!(!tree.exists());

    let again = store.project_snapshot(&id).unwrap();
    assert_eq!(again, tree);
    assert_eq!(
        before,
        fingerprint(&again),
        "re-projection must be identical"
    );
    assert_eq!(
        store.project_snapshot(&id).unwrap(),
        tree,
        "projecting an existing tree is a no-op"
    );
}

// ---------------------------------------------------------------------------
// Stub shape (the format itself is pinned by #70)
// ---------------------------------------------------------------------------

/// The stub a tree carries is exactly the stub the renderer produces for the
/// manifest entry it projects — the digest and size come from the manifest,
/// never from reading the file.
#[test]
fn a_stub_carries_the_body_digest_and_logical_size_of_its_manifest_entry() {
    let root = TempRoot::new("stub");
    let store = WorkspaceStore::open(&root.0).unwrap();
    let fixture = Fixture::new();
    let id = sealed(&store, &fixture);
    let tree = store.project_snapshot(&id).unwrap();

    let body = store
        .load_snapshot(&id)
        .unwrap()
        .entries()
        .iter()
        .find_map(|(path, entry)| match entry {
            Entry::File { body, .. } if path == "model.safetensors" => Some(body.clone()),
            _ => None,
        })
        .expect("the fixture has a tensor entry");
    assert_eq!(body.logical_size(), fixture.weights.len() as u64);
    let stub = tree.join("model.safetensors");
    assert_eq!(
        fs::read(&stub).unwrap(),
        stub_bytes(&body.body_sha256(), body.logical_size())
    );

    // A stub is a REAL FILE at the real filename — not a symlink, not a
    // directory, not absent — and it is immutable like every other artifact
    // in the tree.
    let metadata = fs::symlink_metadata(&stub).unwrap();
    assert!(metadata.is_file() && !metadata.file_type().is_symlink());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(metadata.permissions().mode() & 0o777, 0o444);
        assert_eq!(
            fs::OpenOptions::new()
                .append(true)
                .open(&stub)
                .expect_err("a stub must refuse a write")
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_object(
    store: &tensorfs_core::store::ObjectStore,
    digest: &ObjectDigest,
    length: u64,
) -> Vec<u8> {
    use std::io::Read as _;
    let mut file = store.open_object(digest).unwrap();
    let mut bytes = Vec::with_capacity(length as usize);
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            found.push(path.clone());
            if fs::symlink_metadata(&path)
                .map(|metadata| metadata.is_dir())
                .unwrap_or(false)
            {
                stack.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Every projected path with what it actually is: link target, or bytes.
fn fingerprint(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for path in walk(root) {
        let relative = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let metadata = fs::symlink_metadata(&path).unwrap();
        let what = if metadata.file_type().is_symlink() {
            format!("link:{}", fs::read_link(&path).unwrap().display())
        } else if metadata.is_dir() {
            "dir".to_owned()
        } else {
            format!(
                "file:{}",
                String::from_utf8_lossy(&fs::read(&path).unwrap())
            )
        };
        out.push((relative, what));
    }
    out
}

fn snapshot_columns(root: &Path) -> Vec<String> {
    let connection = rusqlite::Connection::open(root.join("metadata.sqlite3")).unwrap();
    let mut statement = connection.prepare("PRAGMA table_info(snapshots)").unwrap();
    statement
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}
