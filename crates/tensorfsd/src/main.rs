#![forbid(unsafe_code)]

//! `tensorfsd mount-snapshot --store <root> --snapshot <hex> <mountpoint>`
//!
//! Foreground read-only snapshot mount. The UDS control plane arrives with a
//! later slice; this binary exists so the first mount is drivable end to end.

use std::process::ExitCode;

#[cfg(target_os = "linux")]
fn main() -> ExitCode {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() -> ExitCode {
    eprintln!("tensorfsd: only the Linux FUSE3 adapter exists in this slice");
    ExitCode::FAILURE
}

#[cfg(target_os = "linux")]
mod linux {
    use std::process::ExitCode;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use tensorfs_core::tfm1::SnapshotId;
    use tensorfsd::mount_snapshot;

    const USAGE: &str =
        "usage: tensorfsd mount-snapshot --store <root> --snapshot <64-hex> <mountpoint>";

    pub fn run() -> ExitCode {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let mut store = None;
        let mut snapshot = None;
        let mut mountpoint = None;
        let mut rest = arguments.iter();
        match rest.next().map(String::as_str) {
            Some("mount-snapshot") => {}
            _ => return usage(),
        }
        while let Some(argument) = rest.next() {
            match argument.as_str() {
                "--store" => store = rest.next(),
                "--snapshot" => snapshot = rest.next(),
                _ if mountpoint.is_none() => mountpoint = Some(argument),
                _ => return usage(),
            }
        }
        let (Some(store), Some(snapshot), Some(mountpoint)) = (store, snapshot, mountpoint) else {
            return usage();
        };
        let Some(id) = SnapshotId::parse_hex(snapshot) else {
            eprintln!("tensorfsd: --snapshot must be 64 lowercase hex characters");
            return ExitCode::FAILURE;
        };

        let mount = match mount_snapshot(store, &id, mountpoint) {
            Ok(mount) => mount,
            Err(error) => {
                eprintln!("tensorfsd: mount failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!("tensorfsd: serving snapshot {snapshot} at {mountpoint} (read-only)");

        let stop = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            if signal_hook::flag::register(signal, Arc::clone(&stop)).is_err() {
                eprintln!("tensorfsd: could not install a signal handler");
                drop(mount);
                return ExitCode::FAILURE;
            }
        }
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
        }
        eprintln!("tensorfsd: unmounting");
        drop(mount);
        ExitCode::SUCCESS
    }

    fn usage() -> ExitCode {
        eprintln!("{USAGE}");
        ExitCode::FAILURE
    }
}
