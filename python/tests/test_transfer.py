from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from chunked_cas import (
    CASRef,
    LocalCAS,
    TransferGrant,
    TransientTransferError,
    download,
    upload,
)


class MemoryTransport:
    def __init__(self, objects: dict[CASRef, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.uploads: list[CASRef] = []
        self.downloads: list[CASRef] = []
        self.transient: dict[CASRef, int] = {}
        self.lock = threading.Lock()

    def _maybe_fail(self, digest: CASRef) -> None:
        with self.lock:
            remaining = self.transient.get(digest, 0)
            if remaining:
                self.transient[digest] = remaining - 1
                raise TransientTransferError("retry me")

    def upload(self, grant: TransferGrant, source: Path) -> None:
        self._maybe_fail(grant.digest)
        with self.lock:
            self.objects[grant.digest] = source.read_bytes()
            self.uploads.append(grant.digest)

    def download(self, grant: TransferGrant, destination: Path) -> None:
        self._maybe_fail(grant.digest)
        destination.write_bytes(self.objects[grant.digest])
        with self.lock:
            self.downloads.append(grant.digest)


def grant(ref: CASRef, size: int, *, expires_at: str | None = None) -> TransferGrant:
    return TransferGrant(ref, size, f"https://objects.invalid/{ref.digest}", expires_at=expires_at)


def test_upload_retries_transient_failures_and_preserves_denominators(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "source")
    first = cas.put_bytes(b"first")
    second = cas.put_bytes(b"second")
    transport = MemoryTransport()
    transport.transient[first] = 2
    beats: list[tuple[CASRef, int]] = []
    report = upload(
        [grant(first, 5), grant(second, 6)],
        cas,
        transport=transport,
        parallel=2,
        sleep=lambda _: None,
        progress=lambda ref, size: beats.append((ref, size)),
    )
    assert report.ok
    assert (report.examined, report.succeeded, report.bytes_transferred) == (2, 2, 11)
    assert transport.objects == {first: b"first", second: b"second"}
    assert set(beats) == {(first, 5), (second, 6)}


def test_download_uses_verified_objects_as_restart_journal(tmp_path: Path) -> None:
    first = CASRef.digest_bytes(b"first")
    second = CASRef.digest_bytes(b"second")
    transport = MemoryTransport({first: b"first", second: b"second"})
    cas = LocalCAS(tmp_path / "destination")
    cas.put_bytes(b"first", expected=first)
    report = download([grant(first, 5), grant(second, 6)], cas, transport=transport, parallel=2)
    assert report.ok
    assert (report.skipped_resident, report.succeeded, report.bytes_transferred) == (1, 1, 6)
    assert transport.downloads == [second]
    assert cas.verify_object(second).read_bytes() == b"second"


def test_download_repairs_corrupt_resident_object_after_verified_refetch(tmp_path: Path) -> None:
    ref = CASRef.digest_bytes(b"correct")
    cas = LocalCAS(tmp_path / "destination")
    cas.put_bytes(b"correct")
    cas.object_path(ref).write_bytes(b"corrupt")
    transport = MemoryTransport({ref: b"correct"})
    report = download([grant(ref, 7)], cas, transport=transport)
    assert report.ok
    assert cas.verify_object(ref).read_bytes() == b"correct"


def test_expired_grant_is_a_replan_not_a_byte_failure(tmp_path: Path) -> None:
    ref = CASRef.digest_bytes(b"data")
    cas = LocalCAS(tmp_path / "source")
    cas.put_bytes(b"data")
    expiry = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    report = upload([grant(ref, 4, expires_at=expiry)], cas, transport=MemoryTransport())
    assert report.needs_replan
    assert report.expired == [ref]
    assert not report.failures


def test_wrong_download_bytes_never_enter_the_cas(tmp_path: Path) -> None:
    wanted = CASRef.digest_bytes(b"wanted")
    transport = MemoryTransport({wanted: b"wrong!"})
    cas = LocalCAS(tmp_path / "destination")
    report = download([grant(wanted, 6)], cas, transport=transport)
    assert not report.ok
    assert len(report.failures) == 1
    assert not cas.object_path(wanted).exists()


@pytest.mark.parametrize(
    "expiry",
    ["tomorrow", "2026-08-13T12:00:00", "2026-08-13 12:00:00Z", "2026-08-13T12:00:00+00:00"],
)
def test_non_rfc3339_utc_expiry_is_refused(expiry: str) -> None:
    ref = CASRef.digest_bytes(b"data")
    with pytest.raises(ValueError, match="RFC 3339 UTC"):
        grant(ref, 4, expires_at=expiry)


def test_progress_runs_once_on_the_caller_thread_after_success(tmp_path: Path) -> None:
    ref = CASRef.digest_bytes(b"data")
    cas = LocalCAS(tmp_path / "source")
    cas.put_bytes(b"data")
    caller = threading.get_ident()
    callbacks: list[int] = []
    report = upload(
        [grant(ref, 4)],
        cas,
        transport=MemoryTransport(),
        progress=lambda _ref, _size: callbacks.append(threading.get_ident()),
    )
    assert report.ok
    assert callbacks == [caller]


def test_python_consumes_and_reproduces_shared_go_grant_vector() -> None:
    path = Path("spec/v1/vectors/upload_grant.json")
    raw = path.read_bytes().strip()
    parsed = json.loads(raw)
    transfer_grant = TransferGrant.from_wire(parsed)
    encoded = json.dumps(transfer_grant.to_wire(), separators=(",", ":")).encode()
    assert encoded == raw
    assert transfer_grant.staging_key.startswith("staging/session-1/")
