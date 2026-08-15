//! The daemon control plane: authenticated length-framed JSON over one
//! mode-0600 peer-credential Unix socket.
//!
//! This is deliberately not a byte API. Frames are bounded to 1 MiB and never
//! carry file, block, or manifest bytes; bulk data moves through the mounted
//! filesystem. The protocol is v1 and pre-launch replace-in-place: no version
//! negotiation beyond the `hello` report, no compatibility decoding.
//!
//! The only methods are the eight the spec names: `hello`, `status`,
//! `open_snapshot`, `create_workspace`, `commit_workspace`, `push_snapshot`,
//! `release`, and `delete_workspace`. Mounts are owned by the daemon under
//! one mount root, named by their lease, and torn down on `release`, on the
//! owning connection closing, and on daemon shutdown. `push_snapshot` exists
//! with an honest typed refusal until the pgw#1258/th#1960 sync engine lands.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nix::sys::socket::getsockopt;
use nix::sys::socket::sockopt::PeerCredentials;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tensorfs_core::tfm1::SnapshotId;
use tensorfs_core::workspace::{WorkspaceError, WorkspaceStore};

use crate::{MountError, SnapshotMount, WorkspaceMount, mount_snapshot, mount_workspace};

/// One frame's byte bound. A frame is control metadata only, so anything
/// larger is a protocol violation, refused before allocation.
pub const MAX_FRAME_BYTES: u32 = 1024 * 1024;
/// The wire protocol generation reported by `hello`.
pub const PROTOCOL: u64 = 1;

const ACCEPT_POLL: Duration = Duration::from_millis(200);

/// One stable refusal. The kebab-case code is the cross-language contract;
/// the message is for humans and never load-bearing.
#[derive(Debug)]
pub struct RpcError {
    code: &'static str,
    message: String,
}

impl RpcError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
}

fn map_workspace_error(error: &WorkspaceError) -> &'static str {
    match error {
        WorkspaceError::UnknownWorkspace(_) => "unknown-workspace",
        WorkspaceError::WorkspaceExists(_) => "workspace-exists",
        WorkspaceError::UnknownSnapshot(_) => "unknown-snapshot",
        WorkspaceError::SnapshotCorrupt => "snapshot-corrupt",
        _ => "internal",
    }
}

fn workspace_error(error: WorkspaceError) -> RpcError {
    RpcError::new(map_workspace_error(&error), error.to_string())
}

fn mount_error(error: MountError) -> RpcError {
    match error {
        MountError::Workspace(inner) => workspace_error(inner),
        other => RpcError::new("mount-failed", other.to_string()),
    }
}

/// Held only for its `Drop`: dropping the guard unmounts. Nothing reads the
/// mount back out, which is exactly the ownership story.
enum Guard {
    Snapshot { _mount: SnapshotMount },
    Workspace { _mount: WorkspaceMount },
}

struct MountEntry {
    kind: &'static str,
    target: String,
    mountpoint: PathBuf,
    guard: Option<Guard>,
}

struct DaemonState {
    store: PathBuf,
    mounts_root: PathBuf,
    next_lease: u64,
    mounts: HashMap<u64, MountEntry>,
}

impl DaemonState {
    /// Detaches one lease's entry. The caller drops it OUTSIDE the state
    /// mutex: unmounting can be slow, and a wedged unmount must never hold
    /// every other connection hostage.
    fn detach(&mut self, lease: u64) -> Option<MountEntry> {
        self.mounts.remove(&lease)
    }
}

impl Drop for MountEntry {
    fn drop(&mut self) {
        // Dropping the guard unmounts; only then can the mountpoint go.
        self.guard.take();
        let _ = std::fs::remove_dir(&self.mountpoint);
    }
}

#[derive(Deserialize)]
struct Request {
    id: u64,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Deserialize)]
struct SnapshotParams {
    snapshot: String,
}

#[derive(Deserialize)]
struct CreateWorkspaceParams {
    workspace: String,
    #[serde(default)]
    from_snapshot: Option<String>,
}

#[derive(Deserialize)]
struct CommitWorkspaceParams {
    workspace: String,
    #[serde(default)]
    parent: Option<String>,
}

#[derive(Deserialize)]
struct ReleaseParams {
    lease: u64,
}

#[derive(Deserialize)]
struct DeleteWorkspaceParams {
    workspace: String,
}

#[derive(Serialize)]
struct MountRow {
    lease: u64,
    kind: &'static str,
    target: String,
    mountpoint: String,
}

/// Serves the control socket until `stop` is set. Blocks the calling thread;
/// every connection runs on its own thread and every mount it opened is torn
/// down when it disconnects.
pub fn serve(
    store: impl AsRef<Path>,
    socket: impl AsRef<Path>,
    mounts_root: impl AsRef<Path>,
    stop: &Arc<AtomicBool>,
) -> io::Result<()> {
    let socket = socket.as_ref();
    let mounts_root = mounts_root.as_ref().to_path_buf();
    std::fs::create_dir_all(&mounts_root)?;
    if socket.exists() {
        std::fs::remove_file(socket)?;
    }
    let listener = UnixListener::bind(socket)?;
    // The bind mode depends on the caller's umask; the socket is authenticated
    // by mode AND peer credentials, so clamp the mode immediately. The window
    // before the clamp is harmless: the credential check still refuses.
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;

    let state = Arc::new(Mutex::new(DaemonState {
        store: store.as_ref().to_path_buf(),
        mounts_root,
        next_lease: 0,
        mounts: HashMap::new(),
    }));
    let mut workers = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let state = Arc::clone(&state);
                let stop = Arc::clone(stop);
                workers.push(std::thread::spawn(move || {
                    serve_connection(stream, &state, &stop);
                }));
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(error) => return Err(error),
        }
        workers.retain(|worker| !worker.is_finished());
    }
    drop(listener);
    let _ = std::fs::remove_file(socket);
    for worker in workers {
        let _ = worker.join();
    }
    // Dropping the state unmounts anything a still-open connection held.
    drop(state);
    Ok(())
}

/// One readdir completes the kernel's INIT handshake, so the session a
/// client receives is live and a guard dropped moments later never joins an
/// uninitialized session.
fn warm_mount(mountpoint: &Path) {
    let _ = std::fs::read_dir(mountpoint).map(|entries| entries.count());
}

fn peer_is_authorized(stream: &UnixStream) -> bool {
    match getsockopt(stream, PeerCredentials) {
        Ok(credentials) => credentials.uid() == nix::unistd::geteuid().as_raw(),
        Err(_) => false,
    }
}

fn serve_connection(mut stream: UnixStream, state: &Mutex<DaemonState>, stop: &AtomicBool) {
    let mut owned: Vec<u64> = Vec::new();
    // Shutdown must never wait for a client to hang up: the read ticks so the
    // stop flag is observed even mid-idle-connection.
    let _ = stream.set_read_timeout(Some(ACCEPT_POLL));
    if !peer_is_authorized(&stream) {
        let _ = write_frame(
            &mut stream,
            &json!({"id": 0, "error": {"code": "not-authorized",
                "message": "the peer uid does not match the daemon uid"}}),
        );
        return;
    }
    while !stop.load(Ordering::Relaxed) {
        let request = match read_frame(&mut stream) {
            Ok(Some(bytes)) => bytes,
            // Clean EOF: the client hung up.
            Ok(None) => break,
            // Idle tick: nothing arrived within the poll window.
            Err(refusal) if refusal.code == "idle" => continue,
            Err(refusal) => {
                let _ = write_frame(
                    &mut stream,
                    &json!({"id": 0, "error": {"code": refusal.code, "message": refusal.message}}),
                );
                break;
            }
        };
        let Ok(request) = serde_json::from_slice::<Request>(&request) else {
            let _ = write_frame(
                &mut stream,
                &json!({"id": 0, "error": {"code": "malformed-frame",
                    "message": "a request is one JSON object with id and method"}}),
            );
            break;
        };
        let id = request.id;
        let response = match dispatch(&request, state, &mut owned) {
            Ok(ok) => json!({"id": id, "ok": ok}),
            Err(error) => {
                json!({"id": id, "error": {"code": error.code, "message": error.message}})
            }
        };
        if write_frame(&mut stream, &response).is_err() {
            break;
        }
    }
    let mut detached = Vec::new();
    {
        let mut state = state.lock().expect("daemon state mutex is healthy");
        for lease in owned {
            detached.extend(state.detach(lease));
        }
    }
    drop(detached);
}

/// Fills `buffer` from `filled` onward, riding out poll-window timeouts once
/// any byte of the frame has arrived. `read_exact` cannot be retried after
/// `WouldBlock` — it leaves the buffer unspecified and the stream position
/// advanced — so this tracks the fill offset explicitly.
fn fill_frame_bytes(
    stream: &mut UnixStream,
    buffer: &mut [u8],
    frame_started: bool,
) -> Result<Option<usize>, RpcError> {
    let mut filled = 0_usize;
    while filled < buffer.len() {
        match stream.read(&mut buffer[filled..]) {
            Ok(0) => return Ok(None),
            Ok(read) => filled += read,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.kind() == io::ErrorKind::TimedOut =>
            {
                if !frame_started && filled == 0 {
                    // Nothing of this frame has arrived; let the caller
                    // re-check the stop flag.
                    return Err(RpcError::new("idle", "no frame within the poll window"));
                }
                // A frame is in flight; keep riding out idle ticks.
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(filled))
}

fn read_frame(stream: &mut UnixStream) -> Result<Option<Vec<u8>>, RpcError> {
    let mut length = [0_u8; 4];
    if fill_frame_bytes(stream, &mut length, false)?.is_none() {
        return Ok(None);
    }
    let length = u32::from_le_bytes(length);
    if length == 0 || length > MAX_FRAME_BYTES {
        return Err(RpcError::new(
            "frame-too-large",
            format!("frames are 1..={MAX_FRAME_BYTES} bytes, got {length}"),
        ));
    }
    let mut bytes = vec![0_u8; length as usize];
    // The body follows its length immediately; a body read never treats the
    // poll window as idle because the frame is already in flight.
    match fill_frame_bytes(stream, &mut bytes, true)? {
        Some(_) => Ok(Some(bytes)),
        None => Ok(None),
    }
}

fn write_frame(stream: &mut UnixStream, value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let length = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    if length > MAX_FRAME_BYTES {
        return Err(io::Error::other("response frame exceeds the bound"));
    }
    stream.write_all(&length.to_le_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()
}

fn parse_params<'de, T: Deserialize<'de>>(request: &'de Request) -> Result<T, RpcError> {
    T::deserialize(&request.params)
        .map_err(|error| RpcError::new("invalid-params", error.to_string()))
}

fn parse_snapshot_id(text: &str) -> Result<SnapshotId, RpcError> {
    SnapshotId::parse_hex(text).ok_or_else(|| {
        RpcError::new(
            "invalid-params",
            "a snapshot id is 64 lowercase hex characters",
        )
    })
}

fn open_store(state: &DaemonState) -> Result<WorkspaceStore, RpcError> {
    WorkspaceStore::open(&state.store).map_err(workspace_error)
}

fn claim_mountpoint(state: &mut DaemonState) -> (u64, PathBuf) {
    state.next_lease += 1;
    let lease = state.next_lease;
    let mountpoint = state.mounts_root.join(format!("lease-{lease}"));
    (lease, mountpoint)
}

fn dispatch(
    request: &Request,
    state: &Mutex<DaemonState>,
    owned: &mut Vec<u64>,
) -> Result<Value, RpcError> {
    match request.method.as_str() {
        "hello" => Ok(json!({
            "protocol": PROTOCOL,
            "server": "tensorfsd",
            "version": env!("CARGO_PKG_VERSION"),
        })),
        "status" => {
            let state = state.lock().expect("daemon state mutex is healthy");
            let mut rows: Vec<MountRow> = state
                .mounts
                .iter()
                .map(|(lease, entry)| MountRow {
                    lease: *lease,
                    kind: entry.kind,
                    target: entry.target.clone(),
                    mountpoint: entry.mountpoint.display().to_string(),
                })
                .collect();
            rows.sort_by_key(|row| row.lease);
            Ok(json!({
                "store": state.store.display().to_string(),
                "mounts": rows,
            }))
        }
        "open_snapshot" => {
            let params: SnapshotParams = parse_params(request)?;
            let id = parse_snapshot_id(&params.snapshot)?;
            let mut state = state.lock().expect("daemon state mutex is healthy");
            let (lease, mountpoint) = claim_mountpoint(&mut state);
            std::fs::create_dir_all(&mountpoint)
                .map_err(|error| RpcError::new("mount-failed", error.to_string()))?;
            match mount_snapshot(&state.store, &id, &mountpoint) {
                Ok(mount) => {
                    warm_mount(&mountpoint);
                    state.mounts.insert(
                        lease,
                        MountEntry {
                            kind: "snapshot",
                            target: params.snapshot.clone(),
                            mountpoint: mountpoint.clone(),
                            guard: Some(Guard::Snapshot { _mount: mount }),
                        },
                    );
                    owned.push(lease);
                    Ok(json!({"lease": lease, "mountpoint": mountpoint.display().to_string()}))
                }
                Err(error) => {
                    let _ = std::fs::remove_dir(&mountpoint);
                    Err(mount_error(error))
                }
            }
        }
        "create_workspace" => {
            let params: CreateWorkspaceParams = parse_params(request)?;
            let mut state = state.lock().expect("daemon state mutex is healthy");
            let meta = open_store(&state)?;
            match &params.from_snapshot {
                None => meta
                    .create_workspace(&params.workspace)
                    .map_err(workspace_error)?,
                Some(snapshot) => {
                    let id = parse_snapshot_id(snapshot)?;
                    meta.create_workspace_from_snapshot(&params.workspace, &id)
                        .map_err(workspace_error)?;
                }
            }
            drop(meta);
            let (lease, mountpoint) = claim_mountpoint(&mut state);
            std::fs::create_dir_all(&mountpoint)
                .map_err(|error| RpcError::new("mount-failed", error.to_string()))?;
            match mount_workspace(&state.store, &params.workspace, &mountpoint) {
                Ok(mount) => {
                    warm_mount(&mountpoint);
                    state.mounts.insert(
                        lease,
                        MountEntry {
                            kind: "workspace",
                            target: params.workspace.clone(),
                            mountpoint: mountpoint.clone(),
                            guard: Some(Guard::Workspace { _mount: mount }),
                        },
                    );
                    owned.push(lease);
                    Ok(json!({"lease": lease, "mountpoint": mountpoint.display().to_string()}))
                }
                Err(error) => {
                    let _ = std::fs::remove_dir(&mountpoint);
                    Err(mount_error(error))
                }
            }
        }
        "commit_workspace" => {
            let params: CommitWorkspaceParams = parse_params(request)?;
            let parent = match &params.parent {
                None => None,
                Some(parent) => Some(parse_snapshot_id(parent)?),
            };
            let state = state.lock().expect("daemon state mutex is healthy");
            let meta = open_store(&state)?;
            let id = meta
                .seal_snapshot(&params.workspace, parent)
                .map_err(workspace_error)?;
            Ok(json!({"snapshot": id.to_string()}))
        }
        "push_snapshot" => {
            let _params: SnapshotParams = parse_params(request)?;
            Err(RpcError::new(
                "unimplemented",
                "push_snapshot is the pgw#1258/th#1960 sync engine, not this slice",
            ))
        }
        "release" => {
            let params: ReleaseParams = parse_params(request)?;
            if !owned.contains(&params.lease) {
                return Err(RpcError::new(
                    "unknown-lease",
                    "leases are connection-bound; this connection holds no such lease",
                ));
            }
            let detached = {
                let mut state = state.lock().expect("daemon state mutex is healthy");
                state.detach(params.lease)
            };
            let Some(detached) = detached else {
                return Err(RpcError::new("unknown-lease", "the lease is already gone"));
            };
            drop(detached);
            owned.retain(|lease| *lease != params.lease);
            Ok(json!({}))
        }
        "delete_workspace" => {
            let params: DeleteWorkspaceParams = parse_params(request)?;
            let state = state.lock().expect("daemon state mutex is healthy");
            let mounted = state
                .mounts
                .values()
                .any(|entry| entry.kind == "workspace" && entry.target == params.workspace);
            if mounted {
                return Err(RpcError::new(
                    "workspace-mounted",
                    "release the workspace mount before deleting it",
                ));
            }
            let meta = open_store(&state)?;
            meta.delete_workspace(&params.workspace)
                .map_err(workspace_error)?;
            Ok(json!({}))
        }
        _ => Err(RpcError::new(
            "unknown-method",
            format!("no method {:?} in protocol v1", request.method),
        )),
    }
}
