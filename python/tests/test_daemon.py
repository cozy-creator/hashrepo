"""Client round-trip against a real tensorfsd, when one is available.

The python CI job has no cargo and builds no daemon, so this module skips
cleanly there; point ``TENSORFS_DAEMON_BIN`` at a built ``tensorfsd`` binary
(and run on a host with FUSE) to execute it for real.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tensorfs import DaemonClient, DaemonError

BINARY = os.environ.get("TENSORFS_DAEMON_BIN", "")

pytestmark = pytest.mark.skipif(
    not (BINARY and Path(BINARY).is_file() and Path("/dev/fuse").exists()),
    reason="set TENSORFS_DAEMON_BIN to a built tensorfsd on a FUSE-capable host",
)


@pytest.fixture
def daemon(tmp_path: Path) -> Iterator[str]:
    socket_path = tmp_path / "control.sock"
    child = subprocess.Popen(
        [
            BINARY,
            "serve",
            "--store",
            str(tmp_path / "store"),
            "--socket",
            str(socket_path),
            "--mounts",
            str(tmp_path / "mnt"),
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not socket_path.exists():
            assert time.monotonic() < deadline, "the control socket never appeared"
            time.sleep(0.05)
        yield str(socket_path)
    finally:
        child.terminate()
        child.wait(timeout=10)
        mounts = Path("/proc/mounts").read_text()
        assert str(tmp_path) not in mounts, "no mount may outlive the daemon"


def test_the_typed_client_drives_the_full_lifecycle(daemon: str) -> None:
    with DaemonClient(daemon) as client:
        hello = client.hello()
        assert hello.protocol == 1
        assert hello.server == "tensorfsd"

        assert client.status().mounts == ()

        workspace = client.create_workspace("ws")
        payload = b"bytes through the typed client's mounted path"
        target = Path(workspace.mountpoint) / "model.bin"
        target.write_bytes(payload)
        descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        snapshot = client.commit_workspace("ws")
        assert len(snapshot) == 64

        view = client.open_snapshot(snapshot)
        assert (Path(view.mountpoint) / "model.bin").read_bytes() == payload
        assert len(client.status().mounts) == 2

        with pytest.raises(DaemonError) as refusal:
            client.delete_workspace("ws")
        assert refusal.value.code == "workspace-mounted"

        with pytest.raises(DaemonError) as refusal:
            client.push_snapshot(snapshot)
        assert refusal.value.code == "unimplemented"

        client.release(view.lease)
        client.release(workspace.lease)
        client.delete_workspace("ws")
        assert client.status().mounts == ()


def test_connection_close_is_the_lease_boundary(daemon: str) -> None:
    with DaemonClient(daemon) as client:
        opened = client.create_workspace("bound")
        mountpoint = Path(opened.mountpoint)
        assert mountpoint.exists()

    with DaemonClient(daemon) as client:
        deadline = time.monotonic() + 10
        while client.status().mounts:
            assert time.monotonic() < deadline, "the daemon must reap on hangup"
            time.sleep(0.1)
        assert not mountpoint.exists()
