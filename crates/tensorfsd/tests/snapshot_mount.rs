#![cfg(target_os = "linux")]

use std::fs;
use std::io::{Read, Seek};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

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

fn mount_options(mountpoint: &Path) -> String {
    let mounts = fs::read_to_string("/proc/self/mounts").expect("mount table reads");
    let needle = mountpoint.to_str().expect("test paths are UTF-8");
    mounts
        .lines()
        .find(|line| line.split_whitespace().nth(1) == Some(needle))
        .and_then(|line| line.split_whitespace().nth(3))
        .expect("the mount is in the table")
        .to_owned()
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

    // Every mutation refuses with EROFS — but mutation testing showed this
    // loop measures the *kernel's* read-only mount, not this module's own
    // EROFS handlers. Swapping a handler's errno, or deleting `open`'s write
    // and O_TRUNC refusal outright, leaves the loop green, because the VFS
    // rejects the call before it can become a FUSE request. So assert the
    // property that is actually load-bearing here, and read the handlers
    // below it as defence-in-depth that no userspace test can reach.
    assert!(
        mount_options(&mountpoint)
            .split(',')
            .any(|option| option == "ro"),
        "the snapshot mount is read-only at the kernel level, which is what refuses the writes \
         below"
    );
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

#[test]
fn a_truncated_object_reads_as_eio_never_as_short_or_zero_padded_bytes() {
    if !fuse_available() {
        return;
    }
    let fixture = build_fixture("trunc");
    let mountpoint = unique_dir("mnt-trunc");
    let mount =
        mount_snapshot(&fixture.root, &fixture.snapshot, &mountpoint).expect("snapshot mounts");

    // Shorten — do not delete — the object behind linked-a/linked-b before it
    // is ever read. A deleted object fails at `open_object`; a *short* one
    // opens fine and is refused only because the read path demands the whole
    // slice. Same-length corruption is served by design (bytes are trusted
    // from admission-time verification and never rehashed here), so a short
    // object is the corruption this layer both can and must turn into EIO.
    let store = ObjectStore::open(&fixture.root).expect("store reopens");
    let victim = store
        .put_bytes(&fixture.small)
        .expect("same bytes re-admit as the same object")
        .digest();
    drop(store);
    let victim_path = object_path(&fixture.root, &victim);
    fs::remove_file(&victim_path).expect("victim object removes");
    fs::write(&victim_path, &fixture.small[..10]).expect("truncated stand-in writes");
    assert_eq!(
        fs::metadata(&victim_path).expect("stand-in stats").len(),
        10,
        "the object on disk is genuinely shorter than the record claims"
    );

    let error = fs::read(mountpoint.join("linked-a.bin")).unwrap_err();
    assert_eq!(
        error.raw_os_error(),
        Some(libc::EIO),
        "a short object is EIO, never a short read and never zero padding"
    );

    // The rest of the tree still serves, so the refusal is object-scoped.
    assert_eq!(
        fs::read(mountpoint.join("sub/nested.bin")).expect("healthy file still reads"),
        fixture.beta
    );

    mount.unmount();
    assert!(!mounted_here(&mountpoint));
    fs::remove_dir_all(&fixture.root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

/// Content whose value depends on its position, so bytes served from the
/// wrong offset inside the right object are detectable. A constant fill (the
/// `alpha` run above) hides exactly that class of bug.
fn positional_bytes(length: usize) -> Vec<u8> {
    (0..length as u64)
        .map(|index| (index.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) as u8)
        .collect()
}

/// One sealed snapshot holding a single file far larger than one FUSE read
/// request, so the kernel is forced to ask for interior windows.
fn build_large_fixture(label: &str) -> (PathBuf, SnapshotId, Vec<u8>) {
    let root = unique_dir(&format!("store-{label}"));
    let store = ObjectStore::open(&root).expect("object store opens");
    let workspace = WorkspaceStore::open(&root).expect("workspace store opens");
    workspace
        .create_workspace("build")
        .expect("workspace creates");

    let big = positional_bytes(4 * 1024 * 1024);
    let big_ref = store.put_bytes(&big).expect("big admits").digest();
    workspace
        .commit_generation(
            "build",
            &[Mutation::CreateFile {
                path: "big.bin".into(),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![FileRecord::Data {
                    digest: big_ref,
                    length: big.len() as u64,
                }],
            }],
        )
        .expect("generation commits");
    let snapshot = workspace
        .seal_snapshot("build", None)
        .expect("snapshot seals");
    (root, snapshot, big)
}

#[test]
fn interior_windows_of_a_multi_request_file_are_byte_exact() {
    if !fuse_available() {
        return;
    }
    let (root, snapshot, big) = build_large_fixture("interior");
    let mountpoint = unique_dir("mnt-interior");
    let mount = mount_snapshot(&root, &snapshot, &mountpoint).expect("snapshot mounts");

    let mut handle = fs::File::open(mountpoint.join("big.bin")).expect("big opens");
    assert_eq!(
        handle.metadata().expect("big stats").len(),
        big.len() as u64
    );

    // Descending, and starting far past any readahead window, so the first
    // request the kernel issues cannot begin at byte zero of the object: the
    // intra-object offset (`from - start`) is genuinely non-zero, and the
    // 1 MiB request that serves a window at 1 MiB genuinely ends *inside* the
    // 4 MiB segment, which is the only way the buffer-end clamp binds.
    let window_len = 64 * 1024;
    for offset in [
        3 * 1024 * 1024_u64,
        2 * 1024 * 1024,
        1024 * 1024 + 4096,
        4096,
        big.len() as u64 - window_len as u64,
    ] {
        handle
            .seek(std::io::SeekFrom::Start(offset))
            .expect("seek lands");
        let mut window = vec![0_u8; window_len];
        handle.read_exact(&mut window).expect("window reads");
        let want = &big[offset as usize..offset as usize + window_len];
        assert!(
            window == want,
            "the window at offset {offset} is served from the wrong place in the object"
        );
    }

    // And the whole file, straight through, matches byte for byte.
    assert!(
        fs::read(mountpoint.join("big.bin")).expect("big reads") == big,
        "a multi-request sequential read reassembles byte-exactly"
    );

    mount.unmount();
    assert!(!mounted_here(&mountpoint));
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}

/// One sealed snapshot with `count` distinct one-object files, so every open
/// is guaranteed to fault in an object the mount has not opened before.
fn build_many_file_fixture(label: &str, count: usize) -> (PathBuf, SnapshotId) {
    let root = unique_dir(&format!("store-{label}"));
    let store = ObjectStore::open(&root).expect("object store opens");
    let workspace = WorkspaceStore::open(&root).expect("workspace store opens");
    workspace
        .create_workspace("build")
        .expect("workspace creates");

    let mutations: Vec<Mutation> = (0..count)
        .map(|index| {
            let bytes = format!("distinct object body number {index}").into_bytes();
            let digest = store.put_bytes(&bytes).expect("body admits").digest();
            Mutation::CreateFile {
                path: format!("file-{index:04}.bin"),
                executable: false,
                planner: PlannerId::RawFixed64mV1,
                records: vec![FileRecord::Data {
                    digest,
                    length: bytes.len() as u64,
                }],
            }
        })
        .collect();
    workspace
        .commit_generation("build", &mutations)
        .expect("generation commits");
    let snapshot = workspace
        .seal_snapshot("build", None)
        .expect("snapshot seals");
    (root, snapshot)
}

/// Counts this process's open descriptors that point inside `root`. Scoping to
/// the fixture root makes the count immune to whatever a sibling test in the
/// same binary is doing.
fn descriptors_into(root: &Path) -> usize {
    fs::read_dir("/proc/self/fd")
        .expect("the fd table lists")
        .filter_map(|entry| fs::read_link(entry.ok()?.path()).ok())
        .filter(|target| target.starts_with(root))
        .count()
}

#[test]
fn closing_a_handle_releases_the_objects_it_opened() {
    if !fuse_available() {
        return;
    }
    const FILES: usize = 64;
    let (root, snapshot) = build_many_file_fixture("release", FILES);
    let mountpoint = unique_dir("mnt-release");
    let mount = mount_snapshot(&root, &snapshot, &mountpoint).expect("snapshot mounts");

    let before = descriptors_into(&root);
    for index in 0..FILES {
        let name = format!("file-{index:04}.bin");
        let body = fs::read(mountpoint.join(&name)).unwrap_or_else(|_| panic!("{name} reads"));
        assert_eq!(
            body,
            format!("distinct object body number {index}").into_bytes()
        );
    }

    // The kernel sends RELEASE asynchronously on the last close, so settle
    // rather than sample once. A leak never settles: this loop is what keeps
    // the assertion from racing the ones that do not leak.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut after = descriptors_into(&root);
    while after > before + 4 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        after = descriptors_into(&root);
    }
    assert!(
        after <= before + 4,
        "reading {FILES} distinct objects left {} store descriptors open (was {before}, now \
         {after}): a closed handle must drop its object cache",
        after.saturating_sub(before)
    );

    mount.unmount();
    assert!(!mounted_here(&mountpoint));
    fs::remove_dir_all(&root).ok();
    fs::remove_dir_all(&mountpoint).ok();
}
