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
