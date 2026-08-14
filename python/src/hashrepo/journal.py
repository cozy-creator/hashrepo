from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .refs import CASRef


@dataclass(frozen=True, slots=True)
class TransferSession:
    """The remote session that owns one manifest upload in progress."""

    name: str
    session_id: str
    manifest: CASRef

    def __post_init__(self) -> None:
        if not self.name or any(ord(char) < 32 or ord(char) == 127 for char in self.name):
            raise ValueError("journal name must be non-empty and contain no controls")
        if not self.session_id or any(
            ord(char) < 32 or ord(char) == 127 for char in self.session_id
        ):
            raise ValueError("session id must be non-empty and contain no controls")
        object.__setattr__(self, "manifest", CASRef.parse(self.manifest))


class TransferJournal:
    """A process-safe durable map from caller operation names to sessions.

    The caller owns the meaning of ``name`` and obtains grants. HashRepo stores
    only the remote session and manifest identity needed to ask that service for
    a resumed plan after a restart.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _read_unlocked(self) -> dict[str, TransferSession]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_bytes())
        if not isinstance(raw, dict) or raw.get("format") != 1:
            raise ValueError(f"transfer journal {self.path} is malformed")
        sessions = raw.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError(f"transfer journal {self.path} is malformed")
        result: dict[str, TransferSession] = {}
        for item in sessions:
            if not isinstance(item, dict) or set(item) != {"name", "session_id", "manifest"}:
                raise ValueError(f"transfer journal {self.path} is malformed")
            if not all(isinstance(item[field], str) for field in item):
                raise ValueError(f"transfer journal {self.path} is malformed")
            session = TransferSession(
                name=item["name"],
                session_id=item["session_id"],
                manifest=CASRef.parse(item["manifest"]),
            )
            if session.name in result:
                raise ValueError(f"transfer journal {self.path} contains a duplicate name")
            result[session.name] = session
        return result

    def _write_unlocked(self, sessions: dict[str, TransferSession]) -> None:
        payload = json.dumps(
            {
                "format": 1,
                "sessions": [
                    {
                        "name": session.name,
                        "session_id": session.session_id,
                        "manifest": str(session.manifest),
                    }
                    for session in sorted(sessions.values(), key=lambda item: item.name)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(raw_path)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(
                self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _lock(self) -> BinaryIO:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        return self.lock_path.open("a+b")

    def find(self, name: str, manifest: str | CASRef) -> TransferSession | None:
        expected = CASRef.parse(manifest)
        with self._lock() as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            session = self._read_unlocked().get(name)
        return session if session is not None and session.manifest == expected else None

    def record(self, session: TransferSession) -> None:
        with self._lock() as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            sessions = self._read_unlocked()
            sessions[session.name] = session
            self._write_unlocked(sessions)

    def clear(self, name: str, *, session_id: str) -> bool:
        with self._lock() as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            sessions = self._read_unlocked()
            current = sessions.get(name)
            if current is None or current.session_id != session_id:
                return False
            del sessions[name]
            self._write_unlocked(sessions)
            return True
