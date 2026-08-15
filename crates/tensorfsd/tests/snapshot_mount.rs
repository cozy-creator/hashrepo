#![cfg(target_os = "linux")]

use std::fs;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process;

use tensorfs_core::planner::PlannerId;
use tensorfs_core::store::ObjectStore;
use tensorfs_core::tfm1::{FileRecord, SnapshotId};
use tensorfs_core::workspace::{Mutation, WorkspaceStore};
use tensorfsd::mount_snapshot;

fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        eprintln!("skipping: /dev/fuse is not available");
        return false;
    }
    if process::Command::new("fusermount3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: fusermount3 is not available");
        return false;
    }
    true
}

fn unique_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tensorfsd-{label}-{}-{:x}",
        process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .expect("clock is sane")
            .as_nanos()
    ));
    fs::create_dir_all(&path).expect("test directory creates");
    path
}

fn mounted_here(mountpoint: &Path) -> bool {
    let mounts = fs::read_to_string("/proc/self/mounts").expect("mount table reads");
    let needle = mountpoint.to_str().expect("test paths are UTF-8");
    mounts
        .lines()
        .any(|line| line.split_whitespace().nth(1) == Some(needle))
}

fn object_path(root: &Path, digest: &impl std::fmt::Display) -> PathBuf {
    let display = digest.to_string();
    let hex = display
        .strip_prefix("sha256:")
        .unwrap_or(&display)
        .to_owned();
    assert_eq!(hex.len(), 64, "object digests carry 64 hex characters");
    root.join("objects")
        .join("sha256")
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(hex)
}

struct Fixture {
    root: PathBuf,
    snapshot: SnapshotId,
    alpha: Vec<u8>,
    beta: Vec<u8>,
    small: Vec<u8>,
}

/// Builds one committed, sealed snapshot with every entry kind this slice
/// serves: a sparse multi-record file, a hardlink group, a symlink, an
/// executable, and a nested directory.
fn build_fixture(label: &str) -> Fixture {
    let root = unique_dir(&format!("store-{label}"));
    let store = ObjectStore::open(&root).expect("object store opens");
    let workspace = WorkspaceStore::open(&root).expect("workspace store opens");
    workspace
        .create_workspace("build")
        .expect("workspace creates");

    let alpha = vec![0xA5_u8; 8192];
    let beta: Vec<u8> = (0_u32..4096).map(|value| (value % 251) as u8).collect();
    let small = b"tiny but real object bytes".to_vec();
    let alpha_ref = store.put_bytes(&alpha).expect("alpha admits").digest();
    let beta_ref = store.put_bytes(&beta).expect("beta admits").digest();
    let small_ref = store.put_bytes(&small).expect("small admits").digest();

    workspace
        .commit_generation(
            "build",
            &[
                Mutation::Mkdir { path: "sub".into() },
                Mutation::CreateFile {
                    path: "sparse.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![
                        FileRecord::Data {
                            digest: alpha_ref,
                            length: alpha.len() as u64,
                        },
                        FileRecord::Hole { length: 4096 },
                        FileRecord::Data {
                            digest: beta_ref,
                            length: beta.len() as u64,
                        },
                        FileRecord::Hole { length: 8192 },
                    ],
                },
                Mutation::CreateFile {
                    path: "linked-a.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![FileRecord::Data {
                        digest: small_ref,
                        length: small.len() as u64,
                    }],
                },
                Mutation::Hardlink {
                    path: "linked-b.bin".into(),
                    target: "linked-a.bin".into(),
                },
                Mutation::CreateFile {
                    path: "run.sh".into(),
                    executable: true,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![FileRecord::Data {
                        digest: small_ref,
                        length: small.len() as u64,
                    }],
                },
                Mutation::CreateFile {
                    path: "sub/nested.bin".into(),
                    executable: false,
                    planner: PlannerId::RawFixed64mV1,
                    records: vec![FileRecord::Data {
                        digest: beta_ref,
                        length: beta.len() as u64,
                    }],
                },
                Mutation::Symlink {
                    path: "pointer".into(),
                    target: "sparse.bin".into(),
                },
            ],
        )
        .expect("generation commits");
    let snapshot = workspace
        .seal_snapshot("build", None)
        .expect("snapshot seals");

    Fixture {
        root,
        snapshot,
        alpha,
        beta,
        small,
    }
}

#[test]
fn a_sealed_snapshot_serves_ordinary_reads_through_a_real_mount() {
    if !fuse_available() {
        return;
    }
    let fixture = build_fixture("serve");
    let mountpoint = unique_dir("mnt-serve");
    let mount =
        mount_snapshot(&fixture.root, &fixture.snapshot, &mountpoint).expect("snapshot mounts");
    assert!(mounted_here(&mountpoint), "the mount is in the mount table");

    // The tree lists exactly the sealed names.
    let mut names: Vec<String> = fs::read_dir(&mountpoint)
        .expect("root lists")
        .map(|entry| {
            entry
                .expect("entry reads")
                .file_name()
                .into_string()
                .unwrap()
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "linked-a.bin",
            "linked-b.bin",
            "pointer",
            "run.sh",
            "sparse.bin",
            "sub"
        ]
    );

    // Sparse file: byte-exact data runs, zero holes, exact logical size.
    let mut expected = fixture.alpha.clone();
    expected.extend(std::iter::repeat_n(0_u8, 4096));
    expected.extend(&fixture.beta);
    expected.extend(std::iter::repeat_n(0_u8, 8192));
    let sparse = fs::read(mountpoint.join("sparse.bin")).expect("sparse reads");
    assert_eq!(sparse.len() as u64, expected.len() as u64);
    assert_eq!(
        sparse, expected,
        "data runs are byte-exact and holes are zeros"
    );

    // A mid-file ranged read that crosses a hole boundary.
    let mut handle = fs::File::open(mountpoint.join("sparse.bin")).expect("sparse opens");
    use std::io::Seek;
    handle
        .seek(std::io::SeekFrom::Start(8192 - 16))
        .expect("seek lands");
    let mut window = [0_u8; 32];
    handle.read_exact(&mut window).expect("window reads");
    assert_eq!(&window[..16], &fixture.alpha[8192 - 16..]);
    assert_eq!(
        &window[16..],
        &[0_u8; 16],
        "the hole side of the window is zeros"
    );

    // Hardlinks: one inode, nlink 2, same bytes through both names.
    let a = fs::metadata(mountpoint.join("linked-a.bin")).expect("link a stats");
    let b = fs::metadata(mountpoint.join("linked-b.bin")).expect("link b stats");
    assert_eq!(a.ino(), b.ino(), "a hardlink group shares one inode");
    assert_eq!(a.nlink(), 2);
    assert_eq!(b.nlink(), 2);
    assert_eq!(
        fs::read(mountpoint.join("linked-b.bin")).expect("link b reads"),
        fixture.small
    );

    // Executable bit and nested reads.
    let run = fs::metadata(mountpoint.join("run.sh")).expect("run.sh stats");
    assert_eq!(
        run.permissions().mode() & 0o111,
        0o111,
        "executable bit serves"
    );
    assert_eq!(
        fs::read(mountpoint.join("sub/nested.bin")).expect("nested reads"),
        fixture.beta
    );

    // Symlink target reads back verbatim.
    assert_eq!(
        fs::read_link(mountpoint.join("pointer")).expect("symlink reads"),
        PathBuf::from("sparse.bin")
    );

    // Every mutation refuses with EROFS.
    for error in [
        fs::write(mountpoint.join("new.bin"), b"x").unwrap_err(),
        fs::OpenOptions::new()
            .write(true)
            .open(mountpoint.join("sparse.bin"))
            .unwrap_err(),
        fs::remove_file(mountpoint.join("linked-a.bin")).unwrap_err(),
        fs::create_dir(mountpoint.join("newdir")).unwrap_err(),
        fs::rename(mountpoint.join("run.sh"), mountpoint.join("ran.sh")).unwrap_err(),
    ] {
        assert_eq!(error.raw_os_error(), Some(libc::EROFS), "{error}");
    }

    mount.unmount();
    assert!(!mounted_here(&mountpoint), "unmount leaves the table clean");
    assert_eq!(
        fs::read_dir(&mountpoint)
            .expect("bare directory lists")
            .count(),
        0,
        "the bare mountpoint is empty after unmount"
    );

    fs::remove_dir_all(&fixture.root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

#[test]
fn a_missing_object_reads_as_eio_never_as_wrong_bytes() {
    if !fuse_available() {
        return;
    }
    let fixture = build_fixture("eio");
    let mountpoint = unique_dir("mnt-eio");
    let mount =
        mount_snapshot(&fixture.root, &fixture.snapshot, &mountpoint).expect("snapshot mounts");

    // Remove the never-yet-read object behind linked-a/linked-b while the
    // mount is live; sparse.bin's objects stay resident.
    let store = ObjectStore::open(&fixture.root).expect("store reopens");
    let victim = store
        .put_bytes(&fixture.small)
        .expect("same bytes re-admit as the same object")
        .digest();
    drop(store);
    fs::remove_file(object_path(&fixture.root, &victim)).expect("victim object removes");

    let error = fs::read(mountpoint.join("linked-a.bin")).unwrap_err();
    assert_eq!(
        error.raw_os_error(),
        Some(libc::EIO),
        "missing bytes are EIO"
    );

    // The rest of the tree still serves.
    assert_eq!(
        fs::read(mountpoint.join("sub/nested.bin")).expect("healthy file still reads"),
        fixture.beta
    );

    mount.unmount();
    assert!(!mounted_here(&mountpoint));
    fs::remove_dir_all(&fixture.root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}
