#![forbid(unsafe_code)]

//! `tensorfsd mount-snapshot --store <root> --snapshot <hex> <mountpoint>`
//! `tensorfsd mount-workspace --store <root> --workspace <name> <mountpoint>`
//! `tensorfsd seal --store <root> --workspace <name> [--parent <hex>]`
//!
//! Foreground mounts and on-demand sealing. The UDS control plane arrives
//! with a later slice; this binary exists so every mount is drivable end to
//! end.

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
    use tensorfs_core::workspace::WorkspaceStore;
    use tensorfsd::{mount_snapshot, mount_workspace};

    const USAGE: &str = "usage:\n  \
        tensorfsd mount-snapshot --store <root> --snapshot <64-hex> <mountpoint>\n  \
        tensorfsd mount-workspace --store <root> --workspace <name> <mountpoint>\n  \
        tensorfsd seal --store <root> --workspace <name> [--parent <64-hex>]";

    struct Parsed {
        store: Option<String>,
        snapshot: Option<String>,
        workspace: Option<String>,
        parent: Option<String>,
        positional: Option<String>,
    }

    fn parse(arguments: &[String]) -> Option<Parsed> {
        let mut parsed = Parsed {
            store: None,
            snapshot: None,
            workspace: None,
            parent: None,
            positional: None,
        };
        let mut rest = arguments.iter();
        while let Some(argument) = rest.next() {
            match argument.as_str() {
                "--store" => parsed.store = rest.next().cloned(),
                "--snapshot" => parsed.snapshot = rest.next().cloned(),
                "--workspace" => parsed.workspace = rest.next().cloned(),
                "--parent" => parsed.parent = rest.next().cloned(),
                _ if parsed.positional.is_none() => parsed.positional = Some(argument.clone()),
                _ => return None,
            }
        }
        Some(parsed)
    }

    pub fn run() -> ExitCode {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let (subcommand, rest) = match arguments.split_first() {
            Some((subcommand, rest)) => (subcommand.as_str(), rest),
            None => return usage(),
        };
        let Some(parsed) = parse(rest) else {
            return usage();
        };
        match subcommand {
            "mount-snapshot" => {
                let (Some(store), Some(snapshot), Some(mountpoint)) =
                    (parsed.store, parsed.snapshot, parsed.positional)
                else {
                    return usage();
                };
                let Some(id) = SnapshotId::parse_hex(&snapshot) else {
                    eprintln!("tensorfsd: --snapshot must be 64 lowercase hex characters");
                    return ExitCode::FAILURE;
                };
                match mount_snapshot(&store, &id, &mountpoint) {
                    Ok(mount) => {
                        eprintln!(
                            "tensorfsd: serving snapshot {snapshot} at {mountpoint} (read-only)"
                        );
                        serve_until_signalled(mount)
                    }
                    Err(error) => {
                        eprintln!("tensorfsd: mount failed: {error}");
                        ExitCode::FAILURE
                    }
                }
            }
            "mount-workspace" => {
                let (Some(store), Some(workspace), Some(mountpoint)) =
                    (parsed.store, parsed.workspace, parsed.positional)
                else {
                    return usage();
                };
                match mount_workspace(&store, &workspace, &mountpoint) {
                    Ok(mount) => {
                        eprintln!(
                            "tensorfsd: serving workspace {workspace} at {mountpoint} (writable)"
                        );
                        serve_until_signalled(mount)
                    }
                    Err(error) => {
                        eprintln!("tensorfsd: mount failed: {error}");
                        ExitCode::FAILURE
                    }
                }
            }
            "seal" => {
                let (Some(store), Some(workspace)) = (parsed.store, parsed.workspace) else {
                    return usage();
                };
                let parent = match parsed.parent {
                    None => None,
                    Some(parent) => match SnapshotId::parse_hex(&parent) {
                        Some(id) => Some(id),
                        None => {
                            eprintln!("tensorfsd: --parent must be 64 lowercase hex characters");
                            return ExitCode::FAILURE;
                        }
                    },
                };
                let meta = match WorkspaceStore::open(&store) {
                    Ok(meta) => meta,
                    Err(error) => {
                        eprintln!("tensorfsd: store open failed: {error}");
                        return ExitCode::FAILURE;
                    }
                };
                match meta.seal_snapshot(&workspace, parent) {
                    Ok(id) => {
                        println!("{id}");
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        eprintln!("tensorfsd: seal failed: {error}");
                        ExitCode::FAILURE
                    }
                }
            }
            _ => usage(),
        }
    }

    fn serve_until_signalled(mount: impl Sized) -> ExitCode {
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
