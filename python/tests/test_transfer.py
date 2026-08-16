from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tensorfs import (
    CASRef,
    LocalCAS,
    TransferGrant,
)
from tensorfs.transfer import (
    _download,
    _TransientTransferError,
    _upload,
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
                raise _TransientTransferError("retry me")

    def upload(
        self,
        grant: TransferGrant,
        source: Path,
        on_bytes: Callable[[int], None] = lambda _moved: None,
    ) -> None:
        self._maybe_fail(grant.digest)
        with self.lock:
            self.objects[grant.digest] = source.read_bytes()
            self.uploads.append(grant.digest)

    def download(
        self,
        grant: TransferGrant,
        destination: Path,
        on_bytes: Callable[[int], None] = lambda _moved: None,
    ) -> None:
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
    report = _upload(
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
    report = _download(
        [grant(first, 5), grant(second, 6)], cas, transport=transport, parallel=2
    )
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
    report = _download([grant(ref, 7)], cas, transport=transport)
    assert report.ok
    assert cas.verify_object(ref).read_bytes() == b"correct"


def test_expired_grant_is_a_replan_not_a_byte_failure(tmp_path: Path) -> None:
    ref = CASRef.digest_bytes(b"data")
    cas = LocalCAS(tmp_path / "source")
    cas.put_bytes(b"data")
    expiry = (datetime.now(UTC) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    report = _upload(
        [grant(ref, 4, expires_at=expiry)], cas, transport=MemoryTransport()
    )
    assert report.needs_replan
    assert report.expired == [ref]
    assert not report.failures


def test_wrong_download_bytes_never_enter_the_cas(tmp_path: Path) -> None:
    wanted = CASRef.digest_bytes(b"wanted")
    transport = MemoryTransport({wanted: b"wrong!"})
    cas = LocalCAS(tmp_path / "destination")
    report = _download([grant(wanted, 6)], cas, transport=transport)
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
    report = _upload(
        [grant(ref, 4)],
        cas,
        transport=MemoryTransport(),
        progress=lambda _ref, _size: callbacks.append(threading.get_ident()),
    )
    assert report.ok
    assert callbacks == [caller]


def test_progress_reports_a_landed_object_before_the_batch_finishes(tmp_path: Path) -> None:
    """A live transfer must be visible WHILE it is live, not only once it ends.

    pgw#1287: `progress` used to be called from the report-assembly loop, which
    runs after every future has drained. A 31.6 GB snapshot therefore reported
    `0 / total` for its entire duration, and the hub — whose transfer freshness
    window is six minutes — condemned the pod while the download was proceeding
    correctly. A test that only checks the final callback set cannot see this:
    the totals were always right, only their timing was wrong.

    So this asserts the timing directly. The second object refuses to finish
    until the first object's callback has been observed, which is impossible if
    callbacks wait for the batch. If the emission regresses, the second worker
    times out and the report carries its failure instead of hanging the suite.
    """

    first = CASRef.digest_bytes(b"first")
    second = CASRef.digest_bytes(b"second")
    first_reported = threading.Event()

    class GatedTransport(MemoryTransport):
        def download(
            self,
            grant: TransferGrant,
            destination: Path,
            on_bytes: Callable[[int], None] = lambda _moved: None,
        ) -> None:
            if grant.digest == second and not first_reported.wait(timeout=10):
                raise AssertionError(
                    "no progress was reported for the first object while the "
                    "second was still in flight — the whole transfer is "
                    "invisible until it ends"
                )
            super().download(grant, destination, on_bytes)

    transport = GatedTransport({first: b"first", second: b"second"})
    beats: list[CASRef] = []

    def progress(ref: CASRef, _size: int) -> None:
        beats.append(ref)
        if ref == first:
            first_reported.set()

    report = _download(
        [grant(first, 5), grant(second, 6)],
        LocalCAS(tmp_path / "destination"),
        transport=transport,
        parallel=2,
        progress=progress,
    )

    assert report.failures == []
    assert report.ok
    assert set(beats) == {first, second}


def test_a_resident_object_advances_progress_before_the_batch_finishes(tmp_path: Path) -> None:
    """A resumed transfer must look alive, not dead.

    pgw#1287: an object already on disk was `skipped` and reported nothing, so
    a resume whose previous attempt landed most objects was indistinguishable
    from one making no headway at all — and a caller that brakes on two
    consecutive zero-progress attempts would condemn the healthiest possible
    pod. Residency is not free either: `contains` REHASHES every byte, so a
    mostly-resident snapshot can spend minutes verifying while reporting
    nothing.

    `progress` answers "is this advancing toward readiness", and a verified
    resident object is completed work toward readiness. Same arrival-style
    gate as the fetched case, so the emission must be incremental too.
    """

    resident = CASRef.digest_bytes(b"already here")
    fetched = CASRef.digest_bytes(b"still remote")
    resident_reported = threading.Event()

    class GatedTransport(MemoryTransport):
        def download(
            self,
            grant: TransferGrant,
            destination: Path,
            on_bytes: Callable[[int], None] = lambda _moved: None,
        ) -> None:
            if not resident_reported.wait(timeout=10):
                raise AssertionError(
                    "the resident object reported no progress while the "
                    "fetched one was still in flight — a resume that is "
                    "mostly complete reads as a transfer making no headway"
                )
            super().download(grant, destination, on_bytes)

    cas = LocalCAS(tmp_path / "destination")
    cas.put_bytes(b"already here", expected=resident)
    transport = GatedTransport({fetched: b"still remote"})
    beats: list[CASRef] = []

    def progress(ref: CASRef, _size: int) -> None:
        beats.append(ref)
        if ref == resident:
            resident_reported.set()

    report = _download(
        [grant(resident, 12), grant(fetched, 12)],
        cas,
        transport=transport,
        parallel=2,
        progress=progress,
    )

    assert report.failures == []
    assert report.skipped_resident == 1
    assert report.succeeded == 1
    # Only the fetched object moved bytes, but BOTH advanced readiness.
    assert report.bytes_transferred == 12
    assert set(beats) == {resident, fetched}


class _DrippingServer(http.server.ThreadingHTTPServer):
    """Serves one object in pieces, gated on the client reporting the first."""

    body: bytes
    piece: int
    reported: threading.Event
    stalled: bool
    uploaded: bytes


class _DrippingHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _DrippingServer

    def do_GET(self) -> None:
        body = self.server.body
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[: self.server.piece])
        self.wfile.flush()
        # Nothing more moves until the client has REPORTED what already
        # arrived. The bounded wait turns a regression into a failing test
        # instead of a hung suite; it is a test bailout, not a health rule.
        if not self.server.reported.wait(timeout=10):
            self.server.stalled = True
            return
        self.wfile.write(body[self.server.piece :])

    def do_PUT(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.uploaded = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def _dripping_server(body: bytes, piece: int) -> _DrippingServer:
    server = _DrippingServer(("127.0.0.1", 0), _DrippingHandler)
    server.body = body
    server.piece = piece
    server.reported = threading.Event()
    server.stalled = False
    server.uploaded = b""
    return server


def test_bytes_progress_reports_a_partial_object_while_it_is_still_arriving(
    tmp_path: Path,
) -> None:
    """One object can be a whole transfer, and it must be visible while it runs.

    `progress` fires per completed object, and objects are capped at 64 MiB —
    so on a slow link a single object is minutes of silence, and a consumer
    watching for liveness has nothing to watch. `bytes_progress` is the finer
    signal: the server here refuses to send the rest of the object until the
    client has reported the piece already delivered, which a transport that
    only reports whole objects can never do.
    """

    body = b"\x5a" * (3 << 20)
    server = _dripping_server(body, 1 << 20)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    beats: list[int] = []
    lock = threading.Lock()

    def bytes_progress(_ref: CASRef, moved: int) -> None:
        with lock:
            beats.append(moved)
        server.reported.set()

    try:
        port = int(server.server_address[1])
        ref = CASRef.digest_bytes(body)
        report = download(
            [TransferGrant(ref, len(body), f"http://127.0.0.1:{port}/objects/{ref.digest}")],
            LocalCAS(tmp_path / "destination"),
            bytes_progress=bytes_progress,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert not server.stalled, (
        "the download reported nothing while the object was still arriving — "
        "one large object is invisible for its whole duration"
    )
    assert report.ok
    assert len(beats) >= 2, f"a {len(body)}-byte object reported {len(beats)} time(s)"
    assert sum(beats) == len(body)


def test_bytes_progress_reports_an_upload_as_the_blocks_leave(tmp_path: Path) -> None:
    body = b"\xc3" * (3 << 20)
    server = _dripping_server(b"", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    cas = LocalCAS(tmp_path / "source")
    ref = cas.put_bytes(body)
    beats: list[int] = []
    lock = threading.Lock()

    def bytes_progress(_ref: CASRef, moved: int) -> None:
        with lock:
            beats.append(moved)

    try:
        port = int(server.server_address[1])
        report = upload(
            [TransferGrant(ref, len(body), f"http://127.0.0.1:{port}/objects/{ref.digest}")],
            cas,
            bytes_progress=bytes_progress,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert report.ok
    assert server.uploaded == body, "the exact bytes reach the store"
    assert len(beats) >= 2, f"a {len(body)}-byte upload reported {len(beats)} time(s)"
    assert sum(beats) == len(body)


def test_python_consumes_and_reproduces_shared_go_grant_vector() -> None:
    path = Path("spec/v1/vectors/upload_grant.json")
    raw = path.read_bytes().strip()
    parsed = json.loads(raw)
    transfer_grant = TransferGrant.from_wire(parsed)
    encoded = json.dumps(transfer_grant.to_wire(), separators=(",", ":")).encode()
    assert encoded == raw
    assert transfer_grant.staging_key.startswith("staging/sha256/session-1/")


def test_python_refuses_every_shared_invalid_grant() -> None:
    path = Path("spec/v1/vectors/invalid_upload_grants.json")
    for vector in json.loads(path.read_text()):
        with pytest.raises((TypeError, ValueError), match=".+"):
            TransferGrant.from_wire(vector["grant"])
