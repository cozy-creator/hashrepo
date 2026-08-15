#![cfg(any(unix, windows))]

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use tensorfs_core::object::ObjectDigest;
use tensorfs_core::store::{ObjectStore, StoreError, TempCollection};

fn digest_of(bytes: &[u8]) -> ObjectDigest {
    ObjectDigest::from_bytes(Sha256::digest(bytes).into())
}

fn digest_path(root: &Path, digest: &ObjectDigest) -> PathBuf {
    let hex = digest
        .to_string()
        .strip_prefix("sha256:")
        .expect("digests display with an algorithm tag")
        .to_owned();
    root.join("objects")
        .join("sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(hex)
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tensorfs-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        Self(path)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tmp_entries(root: &Path) -> Vec<String> {
    fs::read_dir(root.join("tmp"))
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn admission_converges_on_one_verified_resident_object() {
    let root = TempRoot::new("converge");
    let store = ObjectStore::open(&root.0).unwrap();

    let first = store.put_bytes(b"the object bytes").unwrap();
    assert!(!first.preexisting());
    assert_eq!(first.length(), 16);
    assert_eq!(first.digest(), digest_of(b"the object bytes"));

    #[cfg(unix)]
    let inode_before = {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(digest_path(&root.0, &first.digest()))
            .unwrap()
            .ino()
    };

    let second = store.put_bytes(b"the object bytes").unwrap();
    assert!(second.preexisting());
    assert_eq!(second.digest(), first.digest());

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let inode_after = fs::metadata(digest_path(&root.0, &first.digest()))
            .unwrap()
            .ino();
        assert_eq!(inode_before, inode_after, "convergence must not rewrite");
    }

    let mut resident = store.open_object(&first.digest()).unwrap();
    let mut bytes = Vec::new();
    resident.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"the object bytes");

    assert!(tmp_entries(&root.0).is_empty(), "admission leaves no temp");
}

#[test]
fn finish_expecting_refuses_wrong_length_and_wrong_digest() {
    let root = TempRoot::new("expecting");
    let store = ObjectStore::open(&root.0).unwrap();
    let good = digest_of(b"expected bytes");

    let mut writer = store.writer().unwrap();
    writer.write_all(b"expected bytes").unwrap();
    assert!(matches!(
        writer.finish_expecting(good, 5),
        Err(StoreError::LengthMismatch {
            expected: 5,
            actual: 14
        })
    ));
    assert!(tmp_entries(&root.0).is_empty(), "a refused temp is removed");

    let mut writer = store.writer().unwrap();
    writer.write_all(b"different bytes").unwrap();
    assert!(matches!(
        writer.finish_expecting(good, 15),
        Err(StoreError::DigestMismatch { .. })
    ));
    assert!(tmp_entries(&root.0).is_empty());

    let mut writer = store.writer().unwrap();
    writer.write_all(b"expected bytes").unwrap();
    let admitted = writer.finish_expecting(good, 14).unwrap();
    assert_eq!(admitted.digest(), good);
}

#[test]
fn a_divergent_resident_object_is_reported_and_never_replaced() {
    let root = TempRoot::new("divergent");
    let store = ObjectStore::open(&root.0).unwrap();

    let digest = digest_of(b"the true bytes");
    let path = digest_path(&root.0, &digest);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"an impostor!!!").unwrap();

    assert!(matches!(
        store.put_bytes(b"the true bytes"),
        Err(StoreError::CorruptObject { .. })
    ));
    assert_eq!(
        fs::read(&path).unwrap(),
        b"an impostor!!!",
        "a divergent resident is preserved for repair, not clobbered"
    );
    assert!(tmp_entries(&root.0).is_empty());
}

#[test]
fn open_refuses_missing_and_non_regular_objects() {
    let root = TempRoot::new("open");
    let store = ObjectStore::open(&root.0).unwrap();

    let absent = digest_of(b"never admitted");
    assert!(matches!(
        store.open_object(&absent),
        Err(StoreError::Missing { .. })
    ));

    #[cfg(unix)]
    {
        let target = root.0.join("target");
        fs::write(&target, b"never admitted").unwrap();
        let linked = digest_path(&root.0, &absent);
        fs::create_dir_all(linked.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&target, &linked).unwrap();
        assert!(matches!(
            store.open_object(&absent),
            Err(StoreError::NotARegularFile { .. })
        ));
        assert!(matches!(
            store.verify(&absent),
            Err(StoreError::NotARegularFile { .. })
        ));
    }
}

#[test]
fn verify_reports_corruption_without_deleting_the_object() {
    let root = TempRoot::new("verify");
    let store = ObjectStore::open(&root.0).unwrap();

    let admitted = store.put_bytes(b"clean bytes").unwrap();
    assert_eq!(store.verify(&admitted.digest()).unwrap(), 11);

    let path = digest_path(&root.0, &admitted.digest());
    fs::write(&path, b"tampered!!!").unwrap();
    assert!(matches!(
        store.verify(&admitted.digest()),
        Err(StoreError::CorruptObject { .. })
    ));
    assert!(path.exists(), "verification never deletes");
}

#[test]
fn collection_requires_a_positive_grace() {
    let root = TempRoot::new("grace");
    let store = ObjectStore::open(&root.0).unwrap();
    assert!(matches!(
        store.collect_abandoned_temps(Duration::ZERO),
        Err(StoreError::InvalidGrace)
    ));
}

#[test]
fn young_temps_and_unknown_caller_files_are_retained() {
    let root = TempRoot::new("retain");
    let store = ObjectStore::open(&root.0).unwrap();

    fs::write(root.0.join("tmp").join("obj-9-9-9.tmp"), b"young abandon").unwrap();
    fs::write(root.0.join("tmp").join("caller-owned.dat"), b"not ours").unwrap();

    let report = store
        .collect_abandoned_temps(Duration::from_secs(3600))
        .unwrap();
    assert_eq!(
        report,
        TempCollection {
            examined: 1,
            deleted: 0,
            bytes_deleted: 0
        },
        "a young temp is examined but retained; unknown files are not touched"
    );
    assert_eq!(tmp_entries(&root.0).len(), 2);
}

#[test]
fn an_abandoned_temp_is_reclaimed_only_after_the_grace() {
    let root = TempRoot::new("reclaim");
    let store = ObjectStore::open(&root.0).unwrap();

    let path = root.0.join("tmp").join("obj-1-2-3.tmp");
    fs::write(&path, b"crashed writer residue").unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let report = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        report,
        TempCollection {
            examined: 1,
            deleted: 1,
            bytes_deleted: 22
        }
    );
    assert!(!path.exists());

    let repeat = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        repeat,
        TempCollection::default(),
        "collection is idempotent"
    );
}

#[test]
fn a_live_writer_in_this_process_is_never_collected() {
    let root = TempRoot::new("live");
    let store = ObjectStore::open(&root.0).unwrap();

    let mut writer = store.writer().unwrap();
    writer.write_all(b"still writing").unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let report = store
        .collect_abandoned_temps(Duration::from_millis(100))
        .unwrap();
    assert_eq!(
        report,
        TempCollection {
            examined: 1,
            deleted: 0,
            bytes_deleted: 0
        },
        "the advisory lease outranks the grace"
    );

    let admitted = writer.finish().unwrap();
    assert_eq!(store.verify(&admitted.digest()).unwrap(), 13);
    assert!(tmp_entries(&root.0).is_empty());
}

#[test]
fn dropping_an_unfinished_writer_removes_its_temp() {
    let root = TempRoot::new("drop");
    let store = ObjectStore::open(&root.0).unwrap();

    let mut writer = store.writer().unwrap();
    writer.write_all(b"abandoned in-process").unwrap();
    assert_eq!(tmp_entries(&root.0).len(), 1);
    drop(writer);
    assert!(tmp_entries(&root.0).is_empty());
}

#[test]
fn the_empty_object_admits_and_verifies() {
    let root = TempRoot::new("empty");
    let store = ObjectStore::open(&root.0).unwrap();

    let admitted = store.put_bytes(b"").unwrap();
    assert_eq!(admitted.length(), 0);
    assert_eq!(store.verify(&admitted.digest()).unwrap(), 0);
}
