"""The pure-Python typed client for the tensorfsd control socket.

This module is a control plane only: length-framed JSON over the daemon's
mode-0600 Unix socket, eight methods, frames bounded to 1 MiB, and never a
file, block, or manifest byte. Bulk data moves through the mounted paths the
daemon returns. There is no parser, hashing, native extension, or fallback
engine here, and importing it starts nothing.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from types import TracebackType
from typing import Any

MAX_FRAME_BYTES = 1024 * 1024
PROTOCOL = 1

__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL",
    "DaemonClient",
    "DaemonError",
    "DaemonHello",
    "DaemonMount",
    "DaemonStatus",
    "MountedPath",
]


class DaemonError(RuntimeError):
    """One typed daemon refusal carrying the server's stable kebab code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class DaemonHello:
    protocol: int
    server: str
    version: str


@dataclass(frozen=True, slots=True)
class DaemonMount:
    lease: int
    kind: str
    target: str
    mountpoint: str


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    store: str
    mounts: tuple[DaemonMount, ...]


@dataclass(frozen=True, slots=True)
class MountedPath:
    lease: int
    mountpoint: str


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise DaemonError("malformed-frame", f"{field} must be a string")
    return value


def _require_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise DaemonError("malformed-frame", f"{field} must be an integer")
    return value


class DaemonClient:
    """One connection to a tensorfsd control socket.

    Leases are connection-bound: every mount opened through this client is
    torn down by the daemon when the connection closes, so long-lived mounts
    need a long-lived client.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._socket: socket.socket | None = None
        self._next_id = 0

    def __enter__(self) -> DaemonClient:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(self._path)
        self._socket = connection
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _live(self) -> socket.socket:
        if self._socket is None:
            raise DaemonError("not-connected", "use the client as a context manager")
        return self._socket

    def _read_exact(self, count: int) -> bytes:
        connection = self._live()
        chunks = bytearray()
        while len(chunks) < count:
            block = connection.recv(count - len(chunks))
            if not block:
                raise DaemonError("connection-closed", "the daemon hung up mid-frame")
            chunks.extend(block)
        return bytes(chunks)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        connection = self._live()
        self._next_id += 1
        request = json.dumps(
            {"id": self._next_id, "method": method, "params": params}
        ).encode()
        if len(request) > MAX_FRAME_BYTES:
            raise DaemonError("frame-too-large", "request exceeds the 1 MiB frame bound")
        connection.sendall(struct.pack("<I", len(request)) + request)

        (length,) = struct.unpack("<I", self._read_exact(4))
        if length == 0 or length > MAX_FRAME_BYTES:
            raise DaemonError("frame-too-large", f"daemon frame of {length} bytes refused")
        raw = json.loads(self._read_exact(length))
        if not isinstance(raw, dict):
            raise DaemonError("malformed-frame", "a response is one JSON object")
        error = raw.get("error")
        if isinstance(error, dict):
            raise DaemonError(
                _require_str(error.get("code"), "error code"),
                _require_str(error.get("message"), "error message"),
            )
        if raw.get("id") != self._next_id:
            raise DaemonError("malformed-frame", "response id does not match the request")
        ok = raw.get("ok")
        if not isinstance(ok, dict):
            raise DaemonError("malformed-frame", "a success response carries an ok object")
        return ok

    def hello(self) -> DaemonHello:
        ok = self._call("hello", {})
        return DaemonHello(
            protocol=_require_int(ok.get("protocol"), "protocol"),
            server=_require_str(ok.get("server"), "server"),
            version=_require_str(ok.get("version"), "version"),
        )

    def status(self) -> DaemonStatus:
        ok = self._call("status", {})
        rows = ok.get("mounts")
        if not isinstance(rows, list):
            raise DaemonError("malformed-frame", "status carries a mounts list")
        mounts = tuple(
            DaemonMount(
                lease=_require_int(row.get("lease"), "lease"),
                kind=_require_str(row.get("kind"), "kind"),
                target=_require_str(row.get("target"), "target"),
                mountpoint=_require_str(row.get("mountpoint"), "mountpoint"),
            )
            for row in rows
            if isinstance(row, dict)
        )
        return DaemonStatus(store=_require_str(ok.get("store"), "store"), mounts=mounts)

    def open_snapshot(self, snapshot: str) -> MountedPath:
        ok = self._call("open_snapshot", {"snapshot": snapshot})
        return MountedPath(
            lease=_require_int(ok.get("lease"), "lease"),
            mountpoint=_require_str(ok.get("mountpoint"), "mountpoint"),
        )

    def create_workspace(
        self, workspace: str, *, from_snapshot: str | None = None
    ) -> MountedPath:
        params: dict[str, Any] = {"workspace": workspace}
        if from_snapshot is not None:
            params["from_snapshot"] = from_snapshot
        ok = self._call("create_workspace", params)
        return MountedPath(
            lease=_require_int(ok.get("lease"), "lease"),
            mountpoint=_require_str(ok.get("mountpoint"), "mountpoint"),
        )

    def commit_workspace(self, workspace: str, *, parent: str | None = None) -> str:
        params: dict[str, Any] = {"workspace": workspace}
        if parent is not None:
            params["parent"] = parent
        ok = self._call("commit_workspace", params)
        return _require_str(ok.get("snapshot"), "snapshot")

    def push_snapshot(self, snapshot: str) -> None:
        self._call("push_snapshot", {"snapshot": snapshot})

    def release(self, lease: int) -> None:
        self._call("release", {"lease": lease})

    def delete_workspace(self, workspace: str) -> None:
        self._call("delete_workspace", {"workspace": workspace})
