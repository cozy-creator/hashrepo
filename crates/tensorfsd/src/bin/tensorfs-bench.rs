#![forbid(unsafe_code)]

//! `tensorfs-bench run --scale <bytes> --out <dir> [--reps <n>] [--keep]`
//!
//! The pgw#1256 measured-matrix driver. One run emits `matrix-<run>.jsonl`
//! under `--out`: a provenance `meta` row, one `arm` row per repetition, and
//! one `summary` row per arm. Rows are evidence for the quiet-host release
//! gate; nothing here asserts wall-clock floors, and CI never runs this bin.
//!
//! Every mount is served by a SPAWNED `tensorfsd` child process — the
//! production shape — never by an in-process fuser session. A first draft
//! served mounts in-process and deadlocked exactly as FUSE warns: dirty-page
//! writeback from the writing process blocked against the filesystem that the
//! same process had to serve. A progress watchdog additionally bounds every
//! mounted arm: no byte progress for two minutes kills that arm's mount child
//! and records `failed: timeout`, so one wedge costs one row, not the run.

use std::process::ExitCode;

#[cfg(target_os = "linux")]
fn main() -> ExitCode {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() -> ExitCode {
    eprintln!("tensorfs-bench: the measured matrix runs on the Linux FUSE3 adapter only");
    ExitCode::FAILURE
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitCode, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    use sha2::{Digest, Sha256};
    use tensorfs_core::tfm1::{Entry, FileRecord, SnapshotId};
    use tensorfs_core::workspace::WorkspaceStore;
    use tensorfsd::bench::{ArmRow, BENCH_SCHEMA, MetaRow, summarize};

    const USAGE: &str =
        "usage: tensorfs-bench run --scale <bytes> --out <dir> [--reps <n>] [--keep]";
    /// Linux's USER_HZ is fixed at 100 for the userspace ABI; /proc/self/stat
    /// utime/stime are reported in these ticks.
    const USER_HZ: f64 = 100.0;
    const COPY_BUFFER: usize = 4 << 20;
    const LOAD_CAVEAT_THRESHOLD: f64 = 4.0;
    /// Cold sequential floor below which the native baseline is contention,
    /// not hardware. Measured reference on this class of NVMe: `dd bs=1M`
    /// after POSIX_FADV_DONTNEED reaches 1.6 GB/s idle and collapses to
    /// 67 MB/s at 1-min load 31 — the same 5x swing this harness sees, so the
    /// floor exists to disqualify a row, never to assert a speed.
    const NATIVE_READ_SANITY_FLOOR: f64 = 1.0e9;
    /// A mounted arm with no byte progress for this long is wedged, not slow.
    const WATCHDOG_STALL: Duration = Duration::from_secs(120);
    const MOUNT_READY: Duration = Duration::from_secs(15);

    pub fn run() -> ExitCode {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        match parse(&arguments) {
            Some(config) => match execute(&config) {
                Ok(()) => ExitCode::SUCCESS,
                Err(message) => {
                    eprintln!("tensorfs-bench: {message}");
                    ExitCode::FAILURE
                }
            },
            None => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        }
    }

    struct Config {
        scale: u64,
        out: PathBuf,
        reps: u32,
        keep: bool,
    }

    fn parse(arguments: &[String]) -> Option<Config> {
        let mut scale = None;
        let mut out = None;
        let mut reps = 1_u32;
        let mut keep = false;
        let mut cursor = arguments.iter();
        if cursor.next().map(String::as_str) != Some("run") {
            return None;
        }
        while let Some(flag) = cursor.next() {
            match flag.as_str() {
                "--scale" => scale = cursor.next()?.parse().ok(),
                "--out" => out = Some(PathBuf::from(cursor.next()?)),
                "--reps" => reps = cursor.next()?.parse().ok()?,
                "--keep" => keep = true,
                _ => return None,
            }
        }
        Some(Config {
            scale: scale?,
            out: out?,
            reps: reps.max(1),
            keep,
        })
    }

    // ------------------------------------------------------------------
    // Child-process mounts: the production shape
    // ------------------------------------------------------------------

    fn daemon_binary() -> Result<PathBuf, String> {
        let bench = std::env::current_exe().map_err(|error| error.to_string())?;
        let daemon = bench
            .parent()
            .ok_or("bench binary has no parent directory")?
            .join("tensorfsd");
        if !daemon.is_file() {
            return Err(format!(
                "tensorfsd binary not found beside tensorfs-bench at {}",
                daemon.display()
            ));
        }
        Ok(daemon)
    }

    fn is_mounted(mountpoint: &Path) -> bool {
        let target = mountpoint.to_string_lossy();
        fs::read_to_string("/proc/self/mounts")
            .unwrap_or_default()
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(target.as_ref()))
    }

    fn lazy_unmount(mountpoint: &Path) {
        let _ = Command::new("fusermount3")
            .arg("-uz")
            .arg(mountpoint)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    /// One mount served by a spawned `tensorfsd` child. Teardown is bounded on
    /// every path: SIGTERM, a ten-second grace, SIGKILL, then a lazy unmount.
    struct ChildMount {
        child: Child,
        mountpoint: PathBuf,
        done: bool,
    }

    impl ChildMount {
        fn spawn(daemon: &Path, arguments: &[&str], mountpoint: &Path) -> Result<Self, String> {
            fs::create_dir_all(mountpoint).map_err(|error| error.to_string())?;
            let child = Command::new(daemon)
                .args(arguments)
                .arg(mountpoint)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|error| error.to_string())?;
            let mut mount = Self {
                child,
                mountpoint: mountpoint.to_path_buf(),
                done: false,
            };
            let deadline = Instant::now() + MOUNT_READY;
            while Instant::now() < deadline {
                if is_mounted(&mount.mountpoint) {
                    return Ok(mount);
                }
                if let Ok(Some(status)) = mount.child.try_wait() {
                    mount.done = true;
                    return Err(format!("tensorfsd exited before mounting: {status}"));
                }
                thread::sleep(Duration::from_millis(50));
            }
            mount.abort();
            Err(format!(
                "mount at {} never became ready",
                mount.mountpoint.display()
            ))
        }

        fn workspace(
            daemon: &Path,
            store: &Path,
            name: &str,
            mountpoint: &Path,
        ) -> Result<Self, String> {
            let store = store.to_string_lossy();
            Self::spawn(
                daemon,
                &[
                    "mount-workspace",
                    "--store",
                    store.as_ref(),
                    "--workspace",
                    name,
                ],
                mountpoint,
            )
        }

        fn snapshot(
            daemon: &Path,
            store: &Path,
            id: &SnapshotId,
            mountpoint: &Path,
        ) -> Result<Self, String> {
            let store = store.to_string_lossy();
            let hex = id.to_string();
            Self::spawn(
                daemon,
                &[
                    "mount-snapshot",
                    "--store",
                    store.as_ref(),
                    "--snapshot",
                    &hex,
                ],
                mountpoint,
            )
        }

        fn mountpoint(&self) -> &Path {
            &self.mountpoint
        }

        /// The watchdog's hammer: SIGKILL the serving child and lazily detach
        /// the mount, which aborts the FUSE connection and unwedges any
        /// blocked writer with EIO.
        fn abort(&mut self) {
            if self.done {
                return;
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
            if is_mounted(&self.mountpoint) {
                lazy_unmount(&self.mountpoint);
            }
            self.done = true;
        }

        /// Clean teardown: SIGTERM, bounded grace, then escalate.
        fn unmount(mut self) -> Result<(), String> {
            if self.done {
                return Ok(());
            }
            let _ = kill(Pid::from_raw(self.child.id() as i32), Signal::SIGTERM);
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    _ => {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                }
            }
            if is_mounted(&self.mountpoint) {
                lazy_unmount(&self.mountpoint);
                thread::sleep(Duration::from_millis(200));
            }
            self.done = true;
            if is_mounted(&self.mountpoint) {
                return Err(format!("{} is still mounted", self.mountpoint.display()));
            }
            Ok(())
        }
    }

    impl Drop for ChildMount {
        fn drop(&mut self) {
            self.abort();
        }
    }

    /// Total bytes a process has moved, as stall-detector evidence. A big
    /// `fsync` composes for minutes with zero driver-side progress, but the
    /// composing child is reading and admitting objects the whole time — so
    /// child I/O counts as progress too.
    fn child_io_progress(pid: u32) -> u64 {
        let text = fs::read_to_string(format!("/proc/{pid}/io")).unwrap_or_default();
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        field("read_bytes:")
            .wrapping_add(field("write_bytes:"))
            .wrapping_add(field("syscr:"))
            .wrapping_add(field("syscw:"))
    }

    /// Runs one mounted workload on a worker thread under a stall watchdog.
    /// Progress is driver ticks plus the mount child's own I/O counters; only
    /// when BOTH are flat for [`WATCHDOG_STALL`] is the arm wedged, and the
    /// mount child is killed so the worker fails with EIO instead of wedging
    /// the whole run.
    fn with_watchdog<T: Send + 'static>(
        mount: &mut ChildMount,
        progress: Arc<AtomicU64>,
        work: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _ = sender.send(work());
        });
        let child_pid = mount.child.id();
        let observed = |driver: &AtomicU64| {
            driver
                .load(Ordering::Relaxed)
                .wrapping_add(child_io_progress(child_pid))
        };
        let mut last_seen = observed(&progress);
        let mut last_change = Instant::now();
        let result = loop {
            match receiver.recv_timeout(Duration::from_millis(500)) {
                Ok(result) => break result,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let now = observed(&progress);
                    if now != last_seen {
                        last_seen = now;
                        last_change = Instant::now();
                    } else if last_change.elapsed() > WATCHDOG_STALL {
                        mount.abort();
                        let outcome = receiver
                            .recv_timeout(Duration::from_secs(30))
                            .unwrap_or_else(|_| {
                                Err("worker did not return after the mount abort".to_owned())
                            });
                        break match outcome {
                            Ok(_) | Err(_) => Err(format!(
                                "failed: timeout — no driver or child progress for                                  {}s, mount child killed",
                                WATCHDOG_STALL.as_secs()
                            )),
                        };
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err("worker disconnected without a result".to_owned());
                }
            }
        };
        let _ = worker.join();
        result
    }

    // ------------------------------------------------------------------
    // /proc measurement plumbing
    // ------------------------------------------------------------------

    fn proc_io() -> (u64, u64) {
        let text = fs::read_to_string("/proc/self/io").unwrap_or_default();
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.trim().parse().ok())
                .unwrap_or(0)
        };
        (field("read_bytes:"), field("write_bytes:"))
    }

    fn cpu_seconds() -> (f64, f64) {
        let text = fs::read_to_string("/proc/self/stat").unwrap_or_default();
        let after = text.rsplit_once(')').map(|(_, rest)| rest).unwrap_or("");
        let fields: Vec<&str> = after.split_whitespace().collect();
        let tick = |index: usize| {
            fields
                .get(index)
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0)
                / USER_HZ
        };
        (tick(11), tick(12))
    }

    fn peak_rss_bytes() -> u64 {
        fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|rest| {
                rest.trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .map_or(0, |kib| kib * 1024)
    }

    fn load_1m() -> f64 {
        fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .split_whitespace()
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0)
    }

    struct Measured {
        wall_s: f64,
        user_s: f64,
        sys_s: f64,
        io_read_bytes: u64,
        io_write_bytes: u64,
        load_1m: f64,
    }

    fn measure<T>(work: impl FnOnce() -> Result<T, String>) -> Result<(T, Measured), String> {
        let load = load_1m();
        let (read_before, write_before) = proc_io();
        let (user_before, sys_before) = cpu_seconds();
        let started = Instant::now();
        let value = work()?;
        let wall_s = started.elapsed().as_secs_f64();
        let (user_after, sys_after) = cpu_seconds();
        let (read_after, write_after) = proc_io();
        Ok((
            value,
            Measured {
                wall_s,
                user_s: user_after - user_before,
                sys_s: sys_after - sys_before,
                io_read_bytes: read_after.saturating_sub(read_before),
                io_write_bytes: write_after.saturating_sub(write_before),
                load_1m: load,
            },
        ))
    }

    // ------------------------------------------------------------------
    // Deterministic corpus
    // ------------------------------------------------------------------

    struct TensorSpec {
        name: &'static str,
        bytes: u64,
    }

    struct CorpusFile {
        path: PathBuf,
        /// Absolute byte offset inside the first big tensor's interior — the
        /// edit target for reuse arms.
        edit_offset: u64,
        bytes: u64,
    }

    struct Corpus {
        source: PathBuf,
        safetensors: CorpusFile,
        gguf: CorpusFile,
        total_bytes: u64,
    }

    fn fill_deterministic(seed: u64, buffer: &mut [u8]) {
        let mut state = seed | 1;
        for chunk in buffer.chunks_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bytes = state.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    fn write_tensor_bytes(sink: &mut impl Write, seed: u64, length: u64) -> Result<(), String> {
        let mut remaining = length;
        let mut buffer = vec![0_u8; COPY_BUFFER.min(length as usize).max(8)];
        let mut block = 0_u64;
        while remaining > 0 {
            let take = buffer.len().min(remaining as usize);
            fill_deterministic(seed ^ block.wrapping_mul(0x9e37_79b9), &mut buffer[..take]);
            sink.write_all(&buffer[..take])
                .map_err(|error| error.to_string())?;
            remaining -= take as u64;
            block += 1;
        }
        Ok(())
    }

    fn write_safetensors(path: &Path, tensors: &[TensorSpec]) -> Result<CorpusFile, String> {
        let mut header = String::from("{");
        let mut offset = 0_u64;
        for (index, tensor) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let end = offset + tensor.bytes;
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
                tensor.name, tensor.bytes
            ));
            offset = end;
        }
        header.push('}');
        let mut sink = File::create(path).map_err(|error| error.to_string())?;
        sink.write_all(&(header.len() as u64).to_le_bytes())
            .map_err(|error| error.to_string())?;
        sink.write_all(header.as_bytes())
            .map_err(|error| error.to_string())?;
        for (index, tensor) in tensors.iter().enumerate() {
            write_tensor_bytes(&mut sink, 0x5eed_0000 + index as u64, tensor.bytes)?;
        }
        sink.sync_all().map_err(|error| error.to_string())?;
        let header_end = 8 + header.len() as u64;
        Ok(CorpusFile {
            path: path.to_path_buf(),
            edit_offset: header_end + tensors[0].bytes / 2,
            bytes: header_end + offset,
        })
    }

    fn write_gguf(path: &Path, tensors: &[TensorSpec]) -> Result<CorpusFile, String> {
        const ALIGNMENT: u64 = 32;
        let align_up = |value: u64| (value + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        let mut directory = Vec::new();
        directory.extend_from_slice(b"GGUF");
        directory.extend_from_slice(&3_u32.to_le_bytes());
        directory.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        directory.extend_from_slice(&0_u64.to_le_bytes());
        let mut expected_offset = 0_u64;
        for tensor in tensors {
            assert!(tensor.bytes % 4 == 0, "F32 tensors need 4-byte multiples");
            directory.extend_from_slice(&(tensor.name.len() as u64).to_le_bytes());
            directory.extend_from_slice(tensor.name.as_bytes());
            directory.extend_from_slice(&1_u32.to_le_bytes());
            directory.extend_from_slice(&(tensor.bytes / 4).to_le_bytes());
            directory.extend_from_slice(&0_u32.to_le_bytes());
            directory.extend_from_slice(&expected_offset.to_le_bytes());
            expected_offset = align_up(expected_offset + tensor.bytes);
        }
        let data_start = align_up(directory.len() as u64);

        let mut sink = File::create(path).map_err(|error| error.to_string())?;
        sink.write_all(&directory)
            .map_err(|error| error.to_string())?;
        sink.write_all(&vec![0_u8; (data_start - directory.len() as u64) as usize])
            .map_err(|error| error.to_string())?;
        for (index, tensor) in tensors.iter().enumerate() {
            write_tensor_bytes(&mut sink, 0x66c0_0000 + index as u64, tensor.bytes)?;
            let padded = align_up(tensor.bytes) - tensor.bytes;
            sink.write_all(&vec![0_u8; padded as usize])
                .map_err(|error| error.to_string())?;
        }
        sink.sync_all().map_err(|error| error.to_string())?;
        let bytes = data_start + expected_offset;
        Ok(CorpusFile {
            path: path.to_path_buf(),
            edit_offset: data_start + tensors[0].bytes / 2,
            bytes,
        })
    }

    fn generate_corpus(source: &Path, scale: u64) -> Result<Corpus, String> {
        fs::create_dir_all(source).map_err(|error| error.to_string())?;
        // ~70% safetensors across four big tensors plus small ones, ~30% GGUF.
        let big = (scale * 7 / 10) / 4;
        let gguf_big = ((scale * 3 / 10) / 2) & !3;
        let safetensors = write_safetensors(
            &source.join("model-a.safetensors"),
            &[
                TensorSpec {
                    name: "blk.0.attn.weight",
                    bytes: big,
                },
                TensorSpec {
                    name: "blk.0.ffn.weight",
                    bytes: big,
                },
                TensorSpec {
                    name: "blk.1.attn.weight",
                    bytes: big,
                },
                TensorSpec {
                    name: "blk.1.ffn.weight",
                    bytes: big,
                },
                TensorSpec {
                    name: "blk.0.attn.bias",
                    bytes: 4096,
                },
                TensorSpec {
                    name: "blk.0.norm.weight",
                    bytes: 8192,
                },
            ],
        )?;
        let gguf = write_gguf(
            &source.join("model-b.gguf"),
            &[
                TensorSpec {
                    name: "blk.2.attn.weight",
                    bytes: gguf_big,
                },
                TensorSpec {
                    name: "blk.2.ffn.weight",
                    bytes: gguf_big,
                },
                TensorSpec {
                    name: "blk.2.norm.weight",
                    bytes: 8192,
                },
            ],
        )?;
        fs::write(
            source.join("config.json"),
            b"{\"model_type\":\"tensorfs-bench\"}\n",
        )
        .map_err(|error| error.to_string())?;
        let total_bytes = safetensors.bytes + gguf.bytes + 33;
        Ok(Corpus {
            source: source.to_path_buf(),
            safetensors,
            gguf,
            total_bytes,
        })
    }

    // ------------------------------------------------------------------
    // Store accounting
    // ------------------------------------------------------------------

    /// True-cold protocol: evict a file's pages so a "cold" read measures
    /// disk, not yesterday's page cache. POSIX_FADV_DONTNEED needs no
    /// privileges; eviction is advisory but reliable for clean pages.
    fn evict_pages(path: &Path) {
        if let Ok(file) = File::open(path) {
            let _ = file.sync_all();
            let _ = nix::fcntl::posix_fadvise(
                &file,
                0,
                0,
                nix::fcntl::PosixFadviseAdvice::POSIX_FADV_DONTNEED,
            );
        }
    }

    /// Evicts every object file in a store plus the named extra files.
    fn evict_store_and(store: &Path, extra: &[&Path]) {
        let sha = store.join("objects").join("sha256");
        if let Ok(level1) = fs::read_dir(&sha) {
            for l1 in level1.flatten() {
                if let Ok(level2) = fs::read_dir(l1.path()) {
                    for l2 in level2.flatten() {
                        if let Ok(objects) = fs::read_dir(l2.path()) {
                            for object in objects.flatten() {
                                evict_pages(&object.path());
                            }
                        }
                    }
                }
            }
        }
        for path in extra {
            evict_pages(path);
        }
    }

    fn store_census(store: &Path) -> (u64, u64) {
        let mut objects = 0_u64;
        let mut bytes = 0_u64;
        let mut stack = vec![store.join("objects").join("sha256")];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(meta) = entry.metadata() {
                    objects += 1;
                    bytes += meta.size();
                }
            }
        }
        (objects, bytes)
    }

    /// Every (digest, length) data record in one sealed snapshot.
    fn snapshot_records(store: &Path, id: &SnapshotId) -> Result<Vec<(String, u64)>, String> {
        let meta = WorkspaceStore::open(store).map_err(|error| error.to_string())?;
        let snapshot = meta.load_snapshot(id).map_err(|error| error.to_string())?;
        let mut records = Vec::new();
        for (_path, entry) in snapshot.entries() {
            if let Entry::File {
                records: file_records,
                ..
            } = entry
            {
                for record in file_records {
                    if let FileRecord::Data { digest, length } = record {
                        records.push((digest.to_string(), *length));
                    }
                }
            }
        }
        Ok(records)
    }

    fn diff_records(before: &[(String, u64)], after: &[(String, u64)]) -> (u64, u64, u64, u64) {
        use std::collections::HashSet;
        let old: HashSet<&(String, u64)> = before.iter().collect();
        let mut objects_new = 0_u64;
        let mut objects_reused = 0_u64;
        let mut bytes_new = 0_u64;
        let mut bytes_reused = 0_u64;
        for row in after {
            if old.contains(row) {
                objects_reused += 1;
                bytes_reused += row.1;
            } else {
                objects_new += 1;
                bytes_new += row.1;
            }
        }
        (objects_new, objects_reused, bytes_new, bytes_reused)
    }

    // ------------------------------------------------------------------
    // Mounted workloads (run on watchdogged worker threads)
    // ------------------------------------------------------------------

    fn copy_into_mount(
        source: &Path,
        mountpoint: &Path,
        progress: &AtomicU64,
    ) -> Result<u64, String> {
        let mut fsyncs = 0_u64;
        let mut buffer = vec![0_u8; COPY_BUFFER];
        for entry in fs::read_dir(source)
            .map_err(|error| error.to_string())?
            .flatten()
        {
            let from = entry.path();
            if !from.is_file() {
                continue;
            }
            let to = mountpoint.join(from.file_name().expect("corpus files are named"));
            let mut reader = File::open(&from).map_err(|error| error.to_string())?;
            let mut writer = File::create(&to).map_err(|error| error.to_string())?;
            loop {
                let read = reader
                    .read(&mut buffer)
                    .map_err(|error| error.to_string())?;
                if read == 0 {
                    break;
                }
                writer
                    .write_all(&buffer[..read])
                    .map_err(|error| error.to_string())?;
                progress.fetch_add(read as u64, Ordering::Relaxed);
            }
            writer.sync_all().map_err(|error| error.to_string())?;
            progress.fetch_add(1, Ordering::Relaxed);
            fsyncs += 1;
        }
        Ok(fsyncs)
    }

    fn edit_bytes_at(
        path: &Path,
        offset: u64,
        payload: &[u8],
        progress: &AtomicU64,
    ) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| error.to_string())?;
        file.write_all(payload).map_err(|error| error.to_string())?;
        progress.fetch_add(payload.len() as u64, Ordering::Relaxed);
        file.sync_all().map_err(|error| error.to_string())?;
        progress.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn read_fully(path: &Path, progress: &AtomicU64) -> Result<u64, String> {
        let mut file = File::open(path).map_err(|error| error.to_string())?;
        let mut buffer = vec![0_u8; COPY_BUFFER];
        let mut total = 0_u64;
        loop {
            let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
            if read == 0 {
                return Ok(total);
            }
            total += read as u64;
            progress.fetch_add(read as u64, Ordering::Relaxed);
        }
    }

    // ------------------------------------------------------------------
    // Emission
    // ------------------------------------------------------------------

    struct Emitter {
        sink: File,
        run_id: String,
        rows: Vec<ArmRow>,
    }

    impl Emitter {
        fn emit_meta(&mut self, meta: &MetaRow) -> Result<(), String> {
            let line = serde_json::to_string(meta).map_err(|error| error.to_string())?;
            writeln!(self.sink, "{line}").map_err(|error| error.to_string())
        }

        fn emit_arm(&mut self, row: ArmRow) -> Result<(), String> {
            let line = serde_json::to_string(&row).map_err(|error| error.to_string())?;
            writeln!(self.sink, "{line}").map_err(|error| error.to_string())?;
            self.sink.flush().map_err(|error| error.to_string())?;
            eprintln!(
                "  {} rep {}: wall {:.2}s read {} MiB write {} MiB{}",
                row.arm,
                row.rep,
                row.wall_s,
                row.io_read_bytes >> 20,
                row.io_write_bytes >> 20,
                row.note
                    .as_deref()
                    .filter(|note| note.starts_with("failed"))
                    .map(|note| format!(" [{note}]"))
                    .unwrap_or_default(),
            );
            self.rows.push(row);
            Ok(())
        }

        fn emit_summaries(&mut self) -> Result<(), String> {
            let mut seen = Vec::new();
            for row in &self.rows {
                if !seen.contains(&row.arm) {
                    seen.push(row.arm.clone());
                }
            }
            for arm in seen {
                if let Some(summary) = summarize(&self.run_id, &arm, &self.rows) {
                    let line =
                        serde_json::to_string(&summary).map_err(|error| error.to_string())?;
                    writeln!(self.sink, "{line}").map_err(|error| error.to_string())?;
                }
            }
            self.sink.sync_all().map_err(|error| error.to_string())
        }
    }

    fn base_row(run_id: &str, arm: &str, rep: u32, measured: &Measured) -> ArmRow {
        ArmRow {
            schema: BENCH_SCHEMA,
            kind: "arm".to_owned(),
            run_id: run_id.to_owned(),
            arm: arm.to_owned(),
            rep,
            wall_s: measured.wall_s,
            user_s: measured.user_s,
            sys_s: measured.sys_s,
            peak_rss_bytes: peak_rss_bytes(),
            io_read_bytes: measured.io_read_bytes,
            io_write_bytes: measured.io_write_bytes,
            driver_fsyncs: None,
            objects_new: None,
            objects_reused: None,
            bytes_new: None,
            bytes_reused: None,
            bytes_total: None,
            store_objects_after: None,
            store_physical_bytes: None,
            logical_bytes: None,
            load_1m: measured.load_1m,
            load_caveat: measured.load_1m > LOAD_CAVEAT_THRESHOLD,
            note: None,
        }
    }

    fn machine_hash() -> String {
        let hostname = fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(hostname.trim().as_bytes());
        let digest = hasher.finalize();
        digest
            .iter()
            .take(6)
            .fold(String::with_capacity(12), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    }

    fn fs_type_of(path: &Path) -> String {
        let text = fs::read_to_string("/proc/mounts").unwrap_or_default();
        let target = path.to_string_lossy();
        let mut best = ("", "unknown");
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let (Some(_), Some(mountpoint), Some(fstype)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            if target.starts_with(mountpoint) && mountpoint.len() >= best.0.len() {
                best = (mountpoint, fstype);
            }
        }
        best.1.to_owned()
    }

    // ------------------------------------------------------------------
    // The run
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_lines)]
    fn execute(config: &Config) -> Result<(), String> {
        let daemon = daemon_binary()?;
        fs::create_dir_all(&config.out).map_err(|error| error.to_string())?;
        let started_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_secs();
        let run_id = format!("run-{started_unix}-{}", std::process::id());
        let scratch = config.out.join(format!("{run_id}-scratch"));
        fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
        let jsonl = config.out.join(format!("matrix-{run_id}.jsonl"));
        let mut emitter = Emitter {
            sink: File::create(&jsonl).map_err(|error| error.to_string())?,
            run_id: run_id.clone(),
            rows: Vec::new(),
        };
        emitter.emit_meta(&MetaRow {
            schema: BENCH_SCHEMA,
            kind: "meta".to_owned(),
            run_id: run_id.clone(),
            scale_bytes: config.scale,
            reps: config.reps,
            kernel: fs::read_to_string("/proc/sys/kernel/osrelease")
                .unwrap_or_default()
                .trim()
                .to_owned(),
            fs_type: fs_type_of(&config.out),
            machine_hash: machine_hash(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            started_unix,
        })?;

        // -------- fixture (timed separately from every TensorFS arm) --------
        let source = scratch.join("source");
        let (corpus, measured) = measure(|| generate_corpus(&source, config.scale))?;
        let mut row = base_row(&run_id, "fixture", 0, &measured);
        row.bytes_total = Some(corpus.total_bytes);
        row.note = Some("fixture generation is not a TensorFS operation".to_owned());
        emitter.emit_arm(row)?;

        // -------- honesty baselines --------
        let silent = AtomicU64::new(0);
        for rep in 0..config.reps {
            evict_pages(&corpus.safetensors.path);
            evict_pages(&corpus.gguf.path);
            let (bytes, measured) = measure(|| {
                let mut total = read_fully(&corpus.safetensors.path, &silent)?;
                total += read_fully(&corpus.gguf.path, &silent)?;
                Ok(total)
            })?;
            let mut row = base_row(&run_id, "native_read", rep, &measured);
            row.bytes_total = Some(bytes);
            // A baseline anchors every mount-vs-native ratio in this run, so a
            // baseline that is itself wrong poisons the whole table silently.
            // This box's NVMe (Crucial P3 Plus, PCIe 4.0 x4) does >1 GB/s cold
            // sequential when idle; anything far below that is contention, not
            // disk, and the row says so instead of anchoring a ratio.
            let rate = bytes as f64 / measured.wall_s.max(f64::MIN_POSITIVE);
            if rate < NATIVE_READ_SANITY_FLOOR {
                row.note = Some(format!(
                    "BASELINE UNTRUSTWORTHY: {:.0} MB/s cold is below the {:.0} MB/s                      floor for this hardware — the box was contended (1-min load                      {:.1}); ratios against this row are not publishable",
                    rate / 1e6,
                    NATIVE_READ_SANITY_FLOOR / 1e6,
                    measured.load_1m,
                ));
            }
            emitter.emit_arm(row)?;
        }
        for rep in 0..config.reps {
            let (bytes, measured) = measure(|| {
                let mut hasher = Sha256::new();
                let mut total = 0_u64;
                for path in [&corpus.safetensors.path, &corpus.gguf.path] {
                    let mut file = File::open(path).map_err(|error| error.to_string())?;
                    let mut buffer = vec![0_u8; COPY_BUFFER];
                    loop {
                        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
                        if read == 0 {
                            break;
                        }
                        hasher.update(&buffer[..read]);
                        total += read as u64;
                    }
                }
                let _ = hasher.finalize();
                Ok(total)
            })?;
            let mut row = base_row(&run_id, "sha256_floor", rep, &measured);
            row.bytes_total = Some(bytes);
            emitter.emit_arm(row)?;
        }

        // -------- import (each rep a cold, fresh store) --------
        let main_store = scratch.join("store");
        for rep in 0..config.reps {
            let store = if rep == 0 {
                main_store.clone()
            } else {
                scratch.join(format!("import-store-{rep}"))
            };
            let mountpoint = scratch.join(format!("mnt-import-{rep}"));
            let (outcome, measured) = measure(|| {
                let meta = WorkspaceStore::open(&store).map_err(|error| error.to_string())?;
                meta.create_workspace("w")
                    .map_err(|error| error.to_string())?;
                drop(meta);
                let mut mount = ChildMount::workspace(&daemon, &store, "w", &mountpoint)?;
                let progress = Arc::new(AtomicU64::new(0));
                let ticker = Arc::clone(&progress);
                let work_source = corpus.source.clone();
                let work_mount = mountpoint.clone();
                let copied = with_watchdog(&mut mount, progress, move || {
                    copy_into_mount(&work_source, &work_mount, &ticker)
                });
                match copied {
                    Ok(fsyncs) => {
                        mount.unmount()?;
                        Ok(Ok(fsyncs))
                    }
                    Err(failure) => Ok(Err(failure)),
                }
            })?;
            let census = store_census(&store);
            let mut row = base_row(&run_id, "import", rep, &measured);
            match outcome {
                Ok(fsyncs) => {
                    row.driver_fsyncs = Some(fsyncs);
                    row.objects_new = Some(census.0);
                    row.objects_reused = Some(0);
                    row.bytes_new = Some(census.1);
                    row.bytes_reused = Some(0);
                    row.bytes_total = Some(census.1);
                    row.store_objects_after = Some(census.0);
                    row.store_physical_bytes = Some(census.1);
                    row.logical_bytes = Some(corpus.total_bytes);
                }
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
            if rep > 0 {
                fs::remove_dir_all(&store).map_err(|error| error.to_string())?;
            }
        }

        // -------- seal: the planner re-boundary, then its fixpoint --------
        let census_before_seal = store_census(&main_store);
        let (snapshot_a, measured) = measure(|| {
            let meta = WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
            meta.seal_snapshot("w", None)
                .map_err(|error| error.to_string())
        })?;
        let census_after_seal = store_census(&main_store);
        let mut row = base_row(&run_id, "seal_reboundary", 0, &measured);
        row.objects_new = Some(census_after_seal.0 - census_before_seal.0);
        row.bytes_new = Some(census_after_seal.1 - census_before_seal.1);
        row.store_objects_after = Some(census_after_seal.0);
        row.store_physical_bytes = Some(census_after_seal.1);
        row.note = Some("first seal re-slices grid objects to semantic boundaries".to_owned());
        emitter.emit_arm(row)?;

        for rep in 0..config.reps {
            let (_, measured) = measure(|| {
                let meta = WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
                meta.seal_snapshot("w", None)
                    .map_err(|error| error.to_string())
            })?;
            let census = store_census(&main_store);
            let mut row = base_row(&run_id, "seal_fixpoint", rep, &measured);
            row.objects_new = Some(census.0 - census_after_seal.0);
            row.store_objects_after = Some(census.0);
            emitter.emit_arm(row)?;
        }

        let sealed_records = snapshot_records(&main_store, &snapshot_a)?;

        // -------- clone (later arms address clone-0..4 by name) --------
        for rep in 0..config.reps.max(5) {
            let name = format!("clone-{rep}");
            let (_, measured) = measure(|| {
                let meta = WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
                meta.create_workspace_from_snapshot(&name, &snapshot_a)
                    .map_err(|error| error.to_string())
            })?;
            if rep >= config.reps {
                continue;
            }
            let census = store_census(&main_store);
            let mut row = base_row(&run_id, "clone", rep, &measured);
            row.objects_new = Some(census.0 - census_after_seal.0);
            row.note = Some("clones are metadata-only; zero objects expected".to_owned());
            emitter.emit_arm(row)?;
        }

        // Helper: one watchdogged workload against one workspace mount.
        let mounted_arm =
            |workspace: &str,
             mountpoint: &Path,
             work: Box<dyn FnOnce(PathBuf, Arc<AtomicU64>) -> Result<u64, String> + Send>|
             -> Result<Result<u64, String>, String> {
                let mut mount = ChildMount::workspace(&daemon, &main_store, workspace, mountpoint)?;
                let progress = Arc::new(AtomicU64::new(0));
                let ticker = Arc::clone(&progress);
                let root = mount.mountpoint().to_path_buf();
                let outcome = with_watchdog(&mut mount, progress, move || work(root, ticker));
                match outcome {
                    Ok(value) => {
                        mount.unmount()?;
                        Ok(Ok(value))
                    }
                    Err(failure) => Ok(Err(failure)),
                }
            };

        // -------- through-the-mount micro ops --------
        for rep in 0..config.reps {
            let mountpoint = scratch.join(format!("mnt-ops-{rep}"));
            let (outcome, measured) = measure(|| {
                mounted_arm(
                    "clone-0",
                    &mountpoint,
                    Box::new(move |root, ticker| {
                        let path = root.join(format!("scratch-{rep}.bin"));
                        let mut file = File::create(&path).map_err(|error| error.to_string())?;
                        let mut buffer = vec![0_u8; 1 << 20];
                        fill_deterministic(0xabcd + rep as u64, &mut buffer);
                        file.write_all(&buffer).map_err(|error| error.to_string())?;
                        ticker.fetch_add(1, Ordering::Relaxed);
                        file.seek(SeekFrom::Start(64 << 10))
                            .map_err(|error| error.to_string())?;
                        file.write_all(&buffer[..8192])
                            .map_err(|error| error.to_string())?;
                        file.sync_all().map_err(|error| error.to_string())?;
                        ticker.fetch_add(1, Ordering::Relaxed);
                        file.set_len(512 << 10).map_err(|error| error.to_string())?;
                        file.set_len(2 << 20).map_err(|error| error.to_string())?;
                        file.sync_all().map_err(|error| error.to_string())?;
                        ticker.fetch_add(1, Ordering::Relaxed);
                        drop(file);
                        fs::remove_file(&path).map_err(|error| error.to_string())?;
                        Ok(2)
                    }),
                )
            })?;
            let mut row = base_row(&run_id, "write_ops", rep, &measured);
            match outcome {
                Ok(fsyncs) => row.driver_fsyncs = Some(fsyncs),
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
        }

        // -------- the 8 KiB overwrite --------
        for rep in 0..config.reps {
            let mountpoint = scratch.join(format!("mnt-o8k-{rep}"));
            let census_before = store_census(&main_store);
            let edit_offset = corpus.safetensors.edit_offset;
            let (outcome, measured) = measure(|| {
                mounted_arm(
                    "clone-1",
                    &mountpoint,
                    Box::new(move |root, ticker| {
                        let mut payload = [0_u8; 8192];
                        fill_deterministic(0x0eed + rep as u64, &mut payload);
                        edit_bytes_at(
                            &root.join("model-a.safetensors"),
                            edit_offset,
                            &payload,
                            &ticker,
                        )?;
                        Ok(1)
                    }),
                )
            })?;
            let census_after = store_census(&main_store);
            let mut row = base_row(&run_id, "overwrite_8k", rep, &measured);
            match outcome {
                Ok(fsyncs) => {
                    row.driver_fsyncs = Some(fsyncs);
                    row.objects_new = Some(census_after.0 - census_before.0);
                    row.bytes_new = Some(census_after.1 - census_before.1);
                    row.note =
                        Some("expected: one grid object rewritten per distinct payload".to_owned());
                }
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
        }

        // -------- sequential rewrite of the GGUF file --------
        {
            let mountpoint = scratch.join("mnt-rewrite");
            let census_before = store_census(&main_store);
            let gguf_bytes = corpus.gguf.bytes;
            let (outcome, measured) = measure(|| {
                mounted_arm(
                    "clone-2",
                    &mountpoint,
                    Box::new(move |root, ticker| {
                        let mut writer = OpenOptions::new()
                            .write(true)
                            .open(root.join("model-b.gguf"))
                            .map_err(|error| error.to_string())?;
                        let mut remaining = gguf_bytes;
                        let mut buffer = vec![0_u8; COPY_BUFFER];
                        let mut block = 0_u64;
                        while remaining > 0 {
                            let take = buffer.len().min(remaining as usize);
                            fill_deterministic(0x7e77 ^ block, &mut buffer[..take]);
                            writer
                                .write_all(&buffer[..take])
                                .map_err(|error| error.to_string())?;
                            ticker.fetch_add(take as u64, Ordering::Relaxed);
                            remaining -= take as u64;
                            block += 1;
                        }
                        writer.sync_all().map_err(|error| error.to_string())?;
                        Ok(1)
                    }),
                )
            })?;
            let census_after = store_census(&main_store);
            let mut row = base_row(&run_id, "seq_rewrite", 0, &measured);
            match outcome {
                Ok(fsyncs) => {
                    row.driver_fsyncs = Some(fsyncs);
                    row.objects_new = Some(census_after.0 - census_before.0);
                    row.bytes_new = Some(census_after.1 - census_before.1);
                    row.bytes_total = Some(corpus.gguf.bytes);
                }
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
        }

        // -------- semantic reuse: one-tensor edit, reseal --------
        for (arm, workspace, file_name, edit_offset, file_bytes) in [
            (
                "semantic_reuse_safetensors",
                "clone-3",
                "model-a.safetensors",
                corpus.safetensors.edit_offset,
                corpus.safetensors.bytes,
            ),
            (
                "semantic_reuse_gguf",
                "clone-4",
                "model-b.gguf",
                corpus.gguf.edit_offset,
                corpus.gguf.bytes,
            ),
        ] {
            let mountpoint = scratch.join(format!("mnt-{workspace}"));
            let file_owned = file_name.to_owned();
            let (outcome, measured) = measure(|| {
                let edited = mounted_arm(
                    workspace,
                    &mountpoint,
                    Box::new(move |root, ticker| {
                        let mut payload = [0_u8; 8192];
                        fill_deterministic(0xd1ff ^ file_bytes, &mut payload);
                        edit_bytes_at(&root.join(&file_owned), edit_offset, &payload, &ticker)?;
                        Ok(1)
                    }),
                )?;
                match edited {
                    Ok(_) => {
                        let meta =
                            WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
                        let sealed = meta
                            .seal_snapshot(workspace, Some(snapshot_a))
                            .map_err(|error| error.to_string())?;
                        Ok(Ok(sealed))
                    }
                    Err(failure) => Ok(Err(failure)),
                }
            })?;
            let mut row = base_row(&run_id, arm, 0, &measured);
            match outcome {
                Ok(sealed) => {
                    let records = snapshot_records(&main_store, &sealed)?;
                    let (objects_new, objects_reused, bytes_new, bytes_reused) =
                        diff_records(&sealed_records, &records);
                    row.driver_fsyncs = Some(1);
                    row.objects_new = Some(objects_new);
                    row.objects_reused = Some(objects_reused);
                    row.bytes_new = Some(bytes_new);
                    row.bytes_reused = Some(bytes_reused);
                    row.bytes_total = Some(bytes_new + bytes_reused);
                    row.note = Some(format!(
                        "8192 changed bytes in one tensor of {file_name}; reuse ratio {:.6}",
                        bytes_reused as f64 / (bytes_new + bytes_reused) as f64
                    ));
                }
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
        }

        // -------- ten workspaces, ten distinct edits --------
        {
            let census_before = store_census(&main_store);
            let span = (corpus.safetensors.bytes - corpus.safetensors.edit_offset - 8192) / 10;
            let edit_base = corpus.safetensors.edit_offset;
            let (failures, measured) = measure(|| {
                let mut failures = 0_u64;
                for index in 0..10_u64 {
                    let name = format!("fleet-{index}");
                    let meta =
                        WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
                    meta.create_workspace_from_snapshot(&name, &snapshot_a)
                        .map_err(|error| error.to_string())?;
                    drop(meta);
                    let mountpoint = scratch.join(format!("mnt-{name}"));
                    let outcome = mounted_arm(
                        &name,
                        &mountpoint,
                        Box::new(move |root, ticker| {
                            let mut payload = [0_u8; 8192];
                            fill_deterministic(0xf1ee7 + index, &mut payload);
                            edit_bytes_at(
                                &root.join("model-a.safetensors"),
                                edit_base + index * span,
                                &payload,
                                &ticker,
                            )?;
                            Ok(1)
                        }),
                    )?;
                    if outcome.is_err() {
                        failures += 1;
                    }
                }
                Ok(failures)
            })?;
            let census_after = store_census(&main_store);
            let mut row = base_row(&run_id, "ten_workspaces", 0, &measured);
            row.objects_new = Some(census_after.0 - census_before.0);
            row.bytes_new = Some(census_after.1 - census_before.1);
            row.logical_bytes = Some(10 * corpus.total_bytes);
            row.store_physical_bytes = Some(census_after.1);
            row.note = Some(if failures == 0 {
                "ten cloned workspaces with ten distinct 8 KiB edits: logical is \
                 10x the corpus, physical grows by the touched objects only"
                    .to_owned()
            } else {
                format!("failed: {failures} of 10 fleet edits timed out")
            });
            emitter.emit_arm(row)?;
        }

        // -------- remount / cold read (pages evicted: disk, not cache) -----
        for rep in 0..config.reps {
            evict_store_and(&main_store, &[]);
            let mountpoint = scratch.join(format!("mnt-cold-{rep}"));
            let (outcome, measured) = measure(|| {
                let mut mount =
                    ChildMount::snapshot(&daemon, &main_store, &snapshot_a, &mountpoint)?;
                let progress = Arc::new(AtomicU64::new(0));
                let ticker = Arc::clone(&progress);
                let target = mount.mountpoint().join("model-a.safetensors");
                let read =
                    with_watchdog(&mut mount, progress, move || read_fully(&target, &ticker));
                match read {
                    Ok(bytes) => {
                        mount.unmount()?;
                        Ok(Ok(bytes))
                    }
                    Err(failure) => Ok(Err(failure)),
                }
            })?;
            let mut row = base_row(&run_id, "remount_cold_read", rep, &measured);
            match outcome {
                Ok(bytes) => row.bytes_total = Some(bytes),
                Err(failure) => row.note = Some(failure),
            }
            emitter.emit_arm(row)?;
        }

        // -------- direct-path cold read (lane 3: no FUSE) -------------------
        for rep in 0..config.reps {
            evict_store_and(&main_store, &[]);
            let (outcome, measured) = measure(|| {
                let meta = WorkspaceStore::open(&main_store).map_err(|error| error.to_string())?;
                let tree = meta
                    .load_snapshot(&snapshot_a)
                    .map_err(|error| error.to_string())?;
                let mut total = 0_u64;
                let mut buffer = vec![0_u8; COPY_BUFFER];
                for (path, entry) in tree.entries() {
                    if path != "model-a.safetensors" {
                        continue;
                    }
                    let Entry::File { records, .. } = entry else {
                        continue;
                    };
                    for record in records {
                        let FileRecord::Data { digest, .. } = record else {
                            continue;
                        };
                        let mut object = meta
                            .store()
                            .open_object(digest)
                            .map_err(|error| error.to_string())?;
                        loop {
                            let read = object
                                .read(&mut buffer)
                                .map_err(|error| error.to_string())?;
                            if read == 0 {
                                break;
                            }
                            total += read as u64;
                        }
                    }
                }
                Ok(total)
            })?;
            let mut row = base_row(&run_id, "direct_path_read", rep, &measured);
            row.bytes_total = Some(outcome);
            row.note = Some(
                "lane 3: daemon-resolved verified objects read with no FUSE in the path".to_owned(),
            );
            emitter.emit_arm(row)?;
        }

        // -------- direct ingest (no mount: bounded parallel hash+admit) -----
        for rep in 0..config.reps {
            let store = scratch.join(format!("direct-ingest-{rep}"));
            evict_pages(&corpus.safetensors.path);
            evict_pages(&corpus.gguf.path);
            let (bytes, measured) = measure(|| {
                let meta = WorkspaceStore::open(&store).map_err(|error| error.to_string())?;
                let mut total = 0_u64;
                for path in [&corpus.safetensors.path, &corpus.gguf.path] {
                    let file = File::open(path).map_err(|error| error.to_string())?;
                    let size = file.metadata().map_err(|error| error.to_string())?.len();
                    let mut offset = 0_u64;
                    while offset < size {
                        let batch_end =
                            (offset + 4 * tensorfs_core::planner::MAX_OBJECT_SIZE).min(size);
                        let mut slots: Vec<Vec<u8>> = Vec::new();
                        while offset < batch_end {
                            let length = (tensorfs_core::planner::MAX_OBJECT_SIZE)
                                .min(size - offset)
                                as usize;
                            let mut slot = vec![0_u8; length];
                            file.read_exact_at(&mut slot, offset)
                                .map_err(|error| error.to_string())?;
                            offset += length as u64;
                            slots.push(slot);
                        }
                        let store_ref = meta.store();
                        let results: Vec<Result<u64, String>> = thread::scope(|scope| {
                            let workers: Vec<_> = slots
                                .iter()
                                .map(|slot| {
                                    scope.spawn(move || {
                                        store_ref
                                            .put_bytes(slot)
                                            .map(|object| object.length())
                                            .map_err(|error| error.to_string())
                                    })
                                })
                                .collect();
                            workers
                                .into_iter()
                                .map(|w| w.join().expect("ingest worker does not panic"))
                                .collect()
                        });
                        for result in results {
                            total += result?;
                        }
                    }
                }
                Ok(total)
            })?;
            let mut row = base_row(&run_id, "direct_ingest", rep, &measured);
            row.bytes_total = Some(bytes);
            row.note = Some(
                "no mount: raw 64 MiB slots, 4-way parallel hash+admit — the bulk write lane"
                    .to_owned(),
            );
            emitter.emit_arm(row)?;
        }

        emitter.emit_summaries()?;
        eprintln!("rows: {}", emitter.rows.len());
        eprintln!("wrote {}", jsonl.display());

        if !config.keep {
            fs::remove_dir_all(&scratch).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}
