from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hashrepo import CASRef, TransferJournal, TransferSession


def test_transfer_session_survives_process_exit_and_matches_manifest(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    manifest = CASRef.digest_bytes(b"manifest")
    script = """
from hashrepo import CASRef, TransferJournal, TransferSession
from pathlib import Path
import sys
journal = TransferJournal(Path(sys.argv[1]))
journal.record(TransferSession("publish", "session-1", CASRef.parse(sys.argv[2])))
"""
    subprocess.run([sys.executable, "-c", script, str(path), str(manifest)], check=True)

    journal = TransferJournal(path)
    assert journal.find("publish", manifest) == TransferSession("publish", "session-1", manifest)
    assert journal.find("publish", CASRef.digest_bytes(b"different")) is None
    assert journal.clear("publish", session_id="stale") is False
    assert journal.clear("publish", session_id="session-1") is True
    assert journal.find("publish", manifest) is None


def test_stale_clear_cannot_remove_a_newer_remote_session(tmp_path: Path) -> None:
    journal = TransferJournal(tmp_path / "journal.json")
    manifest = CASRef.digest_bytes(b"manifest")
    journal.record(TransferSession("publish", "session-1", manifest))
    journal.record(TransferSession("publish", "session-2", manifest))
    assert journal.clear("publish", session_id="session-1") is False
    assert journal.find("publish", manifest) == TransferSession("publish", "session-2", manifest)
