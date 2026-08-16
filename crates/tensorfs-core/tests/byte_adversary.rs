//! Family 3 — the bytes on disk are not what we put there.
//!
//! Everything here attacks a store that has already been built honestly:
//! objects are truncated, flipped, swapped, replaced by a symlink or a
//! directory, or made unreadable, and manifests arrive malformed. The claim
//! under test is that every such attack produces a TYPED refusal and never
//! wrong bytes presented as right ones.
//!
//! One boundary is characterised rather than asserted away, because it is a
//! deliberate design choice with a large performance consequence: **the read
//! path does not re-hash.** `open_object` refuses structural attacks
//! (symlink, directory, non-regular file) on every read, but content
//! corruption of an already-admitted object is caught by the explicit
//! `verify` scrub, not by each read. Re-hashing on read would cost the
//! measured 1.00×-of-native cold-read property that motivates the whole
//! design. The tests below pin BOTH halves of that contract so a future
//! change to either is deliberate: structural refusal is guaranteed on read,
//! content verification is guaranteed through `verify`.

#![cfg(any(unix, windows))]

mod harness;

use std::fs;
use std::path::PathBuf;

use harness::{Consistency, Scratch, data_digests, sealed_workspace};
use tensorfs_core::object::ObjectDigest;
use tensorfs_core::store::StoreError;
use tensorfs_core::tfm1::SnapshotId;
use tensorfs_core::workspace::{WorkspaceError, WorkspaceStore};

/// The on-disk home of one object. The two-level fanout is part of the
/// store's published layout, which the daemon and the benchmarks also rely
/// on, so computing it here is reading the contract, not guessing.
fn object_path(root: &std::path::Path, digest: &ObjectDigest) -> PathBuf {
    let hex = digest.to_string();
    let hex = hex.strip_prefix("sha256:").unwrap_or(&hex).to_owned();
    root.join("objects/sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(&hex)
}

/// One way to damage an object's bytes in place.
type Corruption = Box<dyn Fn(&[u8]) -> Vec<u8>>;

/// One way to damage a whole file's bytes in place.
type Shredder = fn(&[u8]) -> Vec<u8>;

fn built(name: &str) -> (Scratch, WorkspaceStore, SnapshotId, ObjectDigest) {
    let scratch = Scratch::new(name);
    let (meta, id) = sealed_workspace(
        scratch.path(),
        "main",
        &[("model.bin", vec![vec![5_u8; 4096], vec![6_u8; 2048]])],
    );
    let digest = data_digests(&meta, &id)[0];
    (scratch, meta, id, digest)
}

/// Root bypasses mode bits, so the permission arm cannot mean anything
/// there. The crate forbids `unsafe`, so the effective uid comes from procfs
/// rather than `geteuid`; an unreadable status file reads as "not root",
/// which only ever makes the test stricter.
#[cfg(unix)]
fn is_root() -> bool {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Uid:"))
                .and_then(|line| line.split_whitespace().nth(2).map(str::to_owned))
        })
        .is_some_and(|effective| effective == "0")
}

/// Content corruption of every shape is caught by `verify`, which reports it
/// and — deliberately — does not delete or repair.
#[test]
fn verify_catches_every_content_corruption_shape() {
    let cases: Vec<(&str, Corruption)> = vec![
        (
            "truncated",
            Box::new(|bytes: &[u8]| bytes[..bytes.len() / 2].to_vec()),
        ),
        (
            "one flipped bit",
            Box::new(|bytes: &[u8]| {
                let mut out = bytes.to_vec();
                out[0] ^= 0x01;
                out
            }),
        ),
        (
            "same length, different content",
            Box::new(|bytes: &[u8]| {
                let mut out = bytes.to_vec();
                let last = out.len() - 1;
                out[last] = out[last].wrapping_add(1);
                out
            }),
        ),
        ("emptied", Box::new(|_: &[u8]| Vec::new())),
        (
            "extended",
            Box::new(|bytes: &[u8]| [bytes, b"trailing"].concat()),
        ),
    ];

    for (name, mutate) in cases {
        let (scratch, meta, _id, digest) = built("content");
        let path = object_path(scratch.path(), &digest);
        let original = fs::read(&path).expect("object reads");
        fs::write(&path, mutate(&original)).expect("corruption writes");

        let error = meta
            .store()
            .verify(&digest)
            .expect_err(&format!("{name} must be reported"));
        assert!(
            matches!(error, StoreError::CorruptObject { expected, .. } if expected == digest),
            "{name} produced {error:?}"
        );

        // Reported, never silently repaired or deleted: the bytes are still
        // there for an operator to inspect.
        assert!(path.exists(), "{name}: verify deleted the evidence");

        // The independent oracle agrees the store is damaged, which is what
        // makes the assertion above meaningful rather than self-referential.
        let scan = Consistency::scan(scratch.path());
        assert_eq!(
            scan.corrupt.len(),
            1,
            "{name}: the on-disk scan disagreed with verify"
        );
    }
}

/// A *different but internally valid* object moved to the wrong digest path.
/// Every byte hashes correctly — just not to the name it is filed under.
#[test]
fn a_valid_object_filed_under_the_wrong_digest_is_refused() {
    let (scratch, meta, _id, digest) = built("swapped");
    let other = meta
        .store()
        .put_bytes(b"a perfectly valid but different object")
        .expect("second object admits");
    let victim = object_path(scratch.path(), &digest);
    let impostor = object_path(scratch.path(), &other.digest());
    fs::copy(&impostor, &victim).expect("impostor overwrites the victim");

    let error = meta
        .store()
        .verify(&digest)
        .expect_err("a swap must refuse");
    assert!(
        matches!(error, StoreError::CorruptObject { expected, actual }
            if expected == digest && actual == other.digest()),
        "swap produced {error:?}"
    );
}

/// Structural attacks are refused on the READ path, not merely by `verify`:
/// a symlink at a digest path (a classic escape primitive) and a directory
/// both refuse at open.
#[test]
fn structural_attacks_are_refused_on_every_read() {
    // Symlink pointing anywhere — the store must never traverse it.
    #[cfg(unix)]
    {
        let (scratch, meta, _id, digest) = built("symlink");
        let path = object_path(scratch.path(), &digest);
        fs::remove_file(&path).expect("victim removes");
        std::os::unix::fs::symlink("/etc/passwd", &path).expect("symlink plants");

        let error = meta
            .store()
            .open_object(&digest)
            .expect_err("a symlink must never open");
        assert!(
            matches!(error, StoreError::NotARegularFile { .. }),
            "symlink produced {error:?}"
        );
        assert!(matches!(
            meta.store().verify(&digest),
            Err(StoreError::NotARegularFile { .. })
        ));
    }

    // A directory where an object belongs.
    //
    // Both platforms answer `NotARegularFile`, but they reach it differently
    // and that difference is worth knowing. On unix the open of a directory
    // SUCCEEDS (`O_RDONLY|O_NOFOLLOW|O_NONBLOCK` on a directory is legal) and
    // the store's own post-open file-type check refuses it. On Windows the
    // open would fail with ERROR_ACCESS_DENIED, so the refusal is made by the
    // type check that already runs BEFORE the open there.
    //
    // That pre-open check is not a tax added to make this test pass: the
    // Windows branch of `open_object` already stats unconditionally to refuse
    // symlinks, because Windows cannot refuse traversal at open the way
    // `O_NOFOLLOW` does. Widening that existing stat from "is it a symlink" to
    // "is it a regular file" costs no additional syscall on Windows and
    // changes nothing at all on the unix hot path — so the uniform typed
    // contract here is free, unlike re-hashing on read, which is not.
    let (scratch, meta, _id, digest) = built("dir");
    let path = object_path(scratch.path(), &digest);
    fs::remove_file(&path).expect("victim removes");
    fs::create_dir(&path).expect("directory plants");

    let error = meta
        .store()
        .open_object(&digest)
        .expect_err("a directory is not an object");
    assert!(
        matches!(error, StoreError::NotARegularFile { .. }),
        "directory produced {error:?}"
    );
}

/// A HARDLINK to a foreign file planted at an object path.
///
/// This is not the divergent-resident case that `store.rs` already covers, and
/// the difference is the whole point: the digest path and the attacker's file
/// are the SAME INODE. Any "repair" that opened the digest path for writing —
/// or unlinked-and-replaced it in the wrong order — would reach straight
/// through the link into a file outside the store. The store's rule is that a
/// divergent resident is reported and never replaced, so this arm asserts the
/// consequence that only a hardlink can expose: the foreign file's bytes, its
/// inode, and its link count all survive a refused re-admission untouched.
#[test]
#[cfg(unix)]
fn a_hardlink_to_a_foreign_file_at_an_object_path_fails_closed() {
    use std::os::unix::fs::MetadataExt;

    let (scratch, meta, _id, digest) = built("hardlink");
    let path = object_path(scratch.path(), &digest);
    let truth = fs::read(&path).expect("object reads");

    let foreign = scratch.path().join("foreign-secret.bin");
    let foreign_bytes = b"a file outside the store that must never be written".to_vec();
    fs::write(&foreign, &foreign_bytes).expect("the foreign file writes");

    fs::remove_file(&path).expect("victim removes");
    fs::hard_link(&foreign, &path).expect("hardlink plants");
    let foreign_ino = fs::symlink_metadata(&foreign).expect("foreign stats").ino();
    assert_eq!(
        fs::symlink_metadata(&path).expect("planted stats").ino(),
        foreign_ino,
        "the fixture must actually share an inode, or nothing hardlink-specific is under test"
    );

    // The impostor IS a regular file, so `open_object` has nothing structural
    // to refuse: the swap is a content attack wearing a structural disguise,
    // and `verify` is the boundary that names it.
    let error = meta
        .store()
        .verify(&digest)
        .expect_err("a hardlink swap must be reported");
    assert!(
        matches!(error, StoreError::CorruptObject { expected, .. } if expected == digest),
        "hardlink swap produced {error:?}"
    );

    // Re-admitting the object's TRUE bytes meets the impostor at install. It
    // must refuse rather than adopt it, and must not write through the link.
    let error = meta
        .store()
        .put_bytes(&truth)
        .expect_err("re-admission over a hardlink swap must refuse");
    assert!(
        matches!(error, StoreError::CorruptObject { expected, .. } if expected == digest),
        "re-admission produced {error:?}"
    );

    assert_eq!(
        fs::read(&foreign).expect("foreign reads"),
        foreign_bytes,
        "a refused admission wrote through the hardlink into a file outside the store"
    );
    let planted = fs::symlink_metadata(&path).expect("planted re-stats");
    assert_eq!(
        planted.ino(),
        foreign_ino,
        "the impostor was replaced instead of reported"
    );
    assert_eq!(planted.nlink(), 2, "the hardlink group was broken");

    let scan = Consistency::scan(scratch.path());
    assert_eq!(
        scan.corrupt.len(),
        1,
        "the on-disk scan disagreed about the hardlink swap"
    );
    // No writer is live here, and the admission under test was refused, so a
    // surviving temp can only be a leak.
    scan.assert_no_temps("after a refused admission over a hardlink swap");
}

/// A FIFO and a unix socket where an object belongs.
///
/// The structural arm above covers a symlink and a directory. These two attack
/// the OPEN ITSELF rather than what follows it, and they reach the refusal by
/// two different routes:
///
///  * A FIFO with no writer makes `open(O_RDONLY)` **block forever**.
///    `open_object` passes `O_NONBLOCK` precisely so that it cannot, and that
///    flag had no test. The open therefore runs on a worker thread against a
///    deadline: a wedged store is a fast, named failure here instead of a hung
///    suite. Past the open, the store's own file-type check refuses it.
///  * `open()` on a socket inode fails outright with ENXIO, so the kernel
///    refuses before the store's own type check ever runs and the refusal
///    surfaces as a bare `Io`, NOT as `NotARegularFile`. Fail-closed, but not
///    the uniform typed contract the arm above describes — and the divergence
///    is real, because the Windows branch of `open_object` stats before
///    opening and would answer `NotARegularFile` for the same inode shape.
///    Measured rather than assumed: narrowing this match to `Io(_)` alone
///    leaves the arm green. Pinned exactly, so unifying it is a deliberate act
///    rather than an accident.
#[test]
#[cfg(unix)]
fn special_files_at_an_object_path_fail_closed_without_blocking() {
    use std::sync::mpsc;

    // --- FIFO ---
    let (scratch, meta, _id, digest) = built("fifo");
    let path = object_path(scratch.path(), &digest);
    fs::remove_file(&path).expect("victim removes");
    let made = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !made {
        eprintln!("skipping the fifo half: mkfifo(1) is unavailable");
    } else {
        // Owned, so a wedged open cannot hold the test hostage through a
        // scope join: the deadline below is the assertion.
        let root = scratch.path().to_path_buf();
        let victim = digest;
        let (sender, answers) = mpsc::channel();
        std::thread::spawn(move || {
            let store = tensorfs_core::store::ObjectStore::open(&root).expect("store reopens");
            let _ = sender.send(format!("{:?}", store.open_object(&victim)));
        });
        let answer = answers
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect(
                "open_object never answered on a FIFO: the store blocked in open(2), which \
                 wedges every caller on a store an attacker can write one inode into",
            );
        assert!(
            answer.contains("NotARegularFile"),
            "a fifo at an object path produced {answer}"
        );
        assert!(
            matches!(
                meta.store().verify(&digest),
                Err(StoreError::NotARegularFile { .. })
            ),
            "verify must refuse a fifo the same way"
        );
    }

    // --- unix socket ---
    //
    // `sun_path` is ~108 bytes, far shorter than a scratch object path, so the
    // socket is bound somewhere short and hardlinked into place. The inode at
    // the digest path is a genuine socket either way.
    let (scratch, meta, _id, digest) = built("socket");
    let path = object_path(scratch.path(), &digest);
    let short = std::env::temp_dir().join(format!("tfs-sock-{}", std::process::id()));
    let _ = fs::remove_file(&short);
    let Ok(listener) = std::os::unix::net::UnixListener::bind(&short) else {
        eprintln!("skipping the socket half: the loopback socket path was refused");
        return;
    };
    fs::remove_file(&path).expect("victim removes");
    let planted = fs::hard_link(&short, &path).is_ok();
    if !planted {
        eprintln!("skipping the socket half: a socket inode could not be linked into the store");
        drop(listener);
        let _ = fs::remove_file(&short);
        return;
    }

    let error = meta
        .store()
        .open_object(&digest)
        .expect_err("a socket is not an object");
    assert!(
        matches!(error, StoreError::Io(_)),
        "a socket at an object path produced {error:?}"
    );
    assert!(
        meta.store().verify(&digest).is_err(),
        "verify must refuse a socket too"
    );
    drop(listener);
    let _ = fs::remove_file(&short);
}

/// A torn or garbage `metadata.sqlite3` fails closed.
///
/// The object store and the metadata database fail independently, and only one
/// of them had adversarial coverage. The claim here is deliberately narrow and
/// therefore checkable: a damaged metadata database may refuse at open, or
/// refuse at read, but it may **never** hand back a tree that differs from the
/// honest one — and a refusal must leave the damaged bytes on disk, exactly as
/// `verify` leaves a corrupt object for an operator rather than "repairing" it.
/// A store that silently recreated an unreadable database would pass every
/// error-shaped assertion while having destroyed the evidence.
///
/// The object store must also be untouched: these two failure domains are
/// separate, and metadata damage is not object damage.
#[test]
fn a_torn_metadata_database_fails_closed_and_never_returns_a_wrong_tree() {
    let shapes: [(&str, Shredder); 4] = [
        ("whole-file garbage", |bytes| {
            bytes.iter().map(|byte| !byte).collect()
        }),
        ("torn in half", |bytes| bytes[..bytes.len() / 2].to_vec()),
        ("emptied", |_| Vec::new()),
        ("header kept, body shredded", |bytes| {
            let mut out = bytes.to_vec();
            for byte in out.iter_mut().skip(16) {
                *byte ^= 0x5A;
            }
            out
        }),
    ];

    let mut refused_at_open = 0_u32;
    let mut refused_at_read = 0_u32;
    for (name, shape) in shapes {
        let scratch = Scratch::new("torn-metadata");
        let (meta, _id) = sealed_workspace(
            scratch.path(),
            "main",
            &[("model.bin", vec![vec![5_u8; 4096], vec![6_u8; 2048]])],
        );
        let honest = tree_paths(&meta);
        assert!(!honest.is_empty(), "{name}: the fixture committed nothing");
        drop(meta);

        // Every file the engine keeps for this database, so the damage cannot
        // be undone from a WAL the shape did not touch.
        let mut damaged = Vec::new();
        for suffix in ["", "-wal", "-shm"] {
            let file = scratch.path().join(format!("metadata.sqlite3{suffix}"));
            let Ok(bytes) = fs::read(&file) else { continue };
            let torn = shape(&bytes);
            fs::write(&file, &torn).expect("the corruption writes");
            damaged.push((file, torn));
        }

        let verdict = match WorkspaceStore::open(scratch.path()) {
            Err(_) => {
                refused_at_open += 1;
                // A refusal reports; it does not silently recreate.
                for (file, torn) in &damaged {
                    assert_eq!(
                        fs::read(file).ok().as_ref(),
                        Some(torn),
                        "{name}: a refused open rewrote {} — the evidence is gone",
                        file.display()
                    );
                }
                "refused at open"
            }
            Ok(store) => match store.head_tree("main") {
                Err(_) => {
                    refused_at_read += 1;
                    "refused at read"
                }
                Ok(_) => {
                    assert_eq!(
                        tree_paths(&store),
                        honest,
                        "{name}: a damaged metadata database returned a DIFFERENT tree"
                    );
                    "tolerated, and byte-identical"
                }
            },
        };
        eprintln!("torn metadata / {name}: {verdict}");

        // Damaging the metadata database is not damaging the object store.
        Consistency::scan(scratch.path()).assert_examined(2, name);
    }

    assert!(
        refused_at_open + refused_at_read > 0,
        "no metadata corruption shape was refused at all — this arm proved nothing"
    );
    assert!(
        refused_at_open > 0,
        "no shape was refused at open, so the evidence-preservation clause never ran"
    );
}

/// The committed paths of a workspace head, sorted.
fn tree_paths(meta: &WorkspaceStore) -> Vec<String> {
    let mut paths: Vec<String> = meta
        .head_tree("main")
        .expect("head tree builds")
        .entries()
        .iter()
        .map(|(path, _)| path.clone())
        .collect();
    paths.sort();
    paths
}

/// An unreadable object surfaces as a typed I/O error, never as empty or
/// partial content.
#[test]
#[cfg(unix)]
fn an_unreadable_object_is_a_typed_error_not_empty_content() {
    if is_root() {
        eprintln!("skipped: running as root, mode bits do not restrict access");
        return;
    }
    use std::os::unix::fs::PermissionsExt;

    let (scratch, meta, _id, digest) = built("perm");
    let path = object_path(scratch.path(), &digest);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("chmod");

    let outcome = meta.store().open_object(&digest);
    // Restore before asserting so the scratch drop can always clean up.
    let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o644));

    let error = outcome.expect_err("an unreadable object must not open");
    assert!(
        matches!(error, StoreError::Io(_)),
        "permission denial produced {error:?}"
    );
}

/// A missing object is `Missing`, distinctly from corrupt or non-regular —
/// the three failure modes must stay distinguishable to callers.
#[test]
fn the_three_object_failure_modes_stay_distinguishable() {
    let (scratch, meta, _id, digest) = built("modes");
    let path = object_path(scratch.path(), &digest);
    fs::remove_file(&path).expect("victim removes");

    assert!(matches!(
        meta.store().open_object(&digest),
        Err(StoreError::Missing { .. })
    ));
    assert!(matches!(
        meta.store().verify(&digest),
        Err(StoreError::Missing { .. })
    ));
    assert!(Consistency::scan(scratch.path()).corrupt.is_empty());
}

/// Malformed manifests are refused by the strict decoder at adoption. A
/// truncated TFM1 and a bit-flipped TFM1 both fail; neither can install a
/// snapshot that names objects we never verified.
#[test]
fn malformed_manifests_are_refused_at_adoption() {
    let (scratch, meta, id, _digest) = built("manifest");
    let honest = meta.load_snapshot(&id).expect("snapshot loads").to_bytes();

    // Truncated at every quarter point: the decoder must refuse each.
    for divisor in [2_usize, 4, 8] {
        let truncated = honest[..honest.len() / divisor].to_vec();
        let error = meta
            .adopt_snapshot(&truncated)
            .expect_err("a truncated manifest must refuse");
        assert!(
            matches!(error, WorkspaceError::Tfm1(_)),
            "truncation by {divisor} produced {error:?}"
        );
    }

    // A flipped bit at several positions across the encoding.
    for position in [0_usize, honest.len() / 3, honest.len() - 1] {
        let mut flipped = honest.clone();
        flipped[position] ^= 0x01;
        if flipped == honest {
            continue;
        }
        match meta.adopt_snapshot(&flipped) {
            Err(WorkspaceError::Tfm1(_) | WorkspaceError::MissingObject { .. }) => {}
            Err(other) => panic!("a flipped bit at {position} produced {other:?}"),
            Ok(adopted) => {
                // If the flip landed somewhere the grammar tolerates, the
                // result must at minimum be a DIFFERENT identity than the
                // honest snapshot — never a silent impersonation of it.
                assert_ne!(
                    adopted, id,
                    "a corrupted manifest adopted under the honest snapshot's identity"
                );
                // ...and its own bytes must still be self-consistent.
                assert_eq!(adopted, SnapshotId::of(&flipped));
            }
        }
    }

    assert!(
        Consistency::scan(scratch.path()).corrupt.is_empty(),
        "manifest attacks must not damage the object store"
    );
}

/// The characterisation half of the read-path contract: content corruption
/// of an already-admitted object is NOT caught by reading it. This is a
/// deliberate performance choice, pinned here so that changing it — in
/// either direction — is a conscious act rather than an accident.
///
/// If this test ever fails because reads began verifying, that is not
/// necessarily a bug: it is a decision to trade cold-read throughput for
/// read-time integrity, and it should be made explicitly.
#[test]
fn reads_do_not_rehash_which_is_why_verify_exists() {
    let (scratch, meta, _id, digest) = built("noverify");
    let path = object_path(scratch.path(), &digest);
    let original = fs::read(&path).expect("object reads");
    let mut corrupted = original.clone();
    corrupted[0] ^= 0xFF;
    fs::write(&path, &corrupted).expect("corruption writes");

    // The read path opens it happily...
    let mut file = meta
        .store()
        .open_object(&digest)
        .expect("open does not rehash");
    let mut seen = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut seen).expect("reads");
    assert_eq!(
        seen, corrupted,
        "reads returned something other than the resident bytes"
    );

    // ...and the scrub is the thing that catches it.
    assert!(
        matches!(
            meta.store().verify(&digest),
            Err(StoreError::CorruptObject { .. })
        ),
        "verify is the integrity boundary and must catch this"
    );
    assert_eq!(Consistency::scan(scratch.path()).corrupt.len(), 1);
}

/// The oracle must refuse to vouch for a root it never actually read.
///
/// `walk` returns silently when `read_dir` fails, so before `namespace_present`
/// a scan of a wrong path — or of a directory that was never opened as a store
/// — reported a perfectly intact store and passed. `kill_matrix` leans on this
/// as its primary independent oracle after every round, so a vacuous pass there
/// would hollow out the whole matrix. `ObjectStore::open` always creates
/// `objects/sha256`, so an absent namespace can only mean the scan looked
/// somewhere no store lives.
#[test]
#[should_panic(expected = "examined nothing")]
fn a_scan_of_a_path_holding_no_store_cannot_report_intact() {
    let scratch = Scratch::new("no-store");
    // A real directory, deliberately never opened as a store.
    Consistency::scan(scratch.path()).assert_intact("a path holding no store");
}

/// `assert_examined` proves the scan rehashed something, so "verified 40
/// objects, all good" can never be confused with "found nothing to verify".
#[test]
#[should_panic(expected = "expected at least 1")]
fn an_empty_but_valid_store_cannot_satisfy_assert_examined() {
    let scratch = Scratch::new("empty-store");
    // Opened for real, so the namespace exists and `assert_intact` is content
    // — which is exactly why the object-count clause has to be separate.
    let meta = WorkspaceStore::open(scratch.path()).expect("store opens");
    drop(meta);
    Consistency::scan(scratch.path()).assert_examined(1, "an empty store");
}
