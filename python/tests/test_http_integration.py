from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from tensorfs import (
    MAX_CHUNK_SIZE,
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    TransferGrant,
    download,
    read_entry,
    upload,
)


class GrantServer(http.server.ThreadingHTTPServer):
    objects: dict[str, bytes]
    delay_digest: str
    completed: list[str]


class GrantHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: GrantServer
    requests: ClassVar[list[tuple[str, str]]] = []

    def do_PUT(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        digest = self.path.removeprefix("/objects/")
        if hashlib.sha256(body).hexdigest() != digest:
            self.send_error(400)
            return
        self.server.objects[digest] = body
        self.requests.append(("PUT", digest))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        digest = self.path.removeprefix("/objects/")
        body = self.server.objects[digest]
        self.requests.append(("GET", digest))
        if digest == self.server.delay_digest:
            time.sleep(0.05)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.completed.append(digest)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def test_real_http_partial_resume_and_repository_materialization(tmp_path: Path) -> None:
    server = GrantServer(("127.0.0.1", 0), GrantHandler)
    server.objects = {}
    server.delay_digest = ""
    server.completed = []
    GrantHandler.requests = []
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        source_root = tmp_path / "source"
        source_root.mkdir()
        (source_root / "first").write_bytes(b"first")
        (source_root / "second").write_bytes(b"second")
        source = LocalCAS(tmp_path / "source-cas")
        manifest = source.ingest_repository(source_root)
        expiry = (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace(
            "+00:00", "Z"
        )

        def grant(ref: CASRef, size: int) -> TransferGrant:
            return TransferGrant(
                ref,
                size,
                f"http://127.0.0.1:{server.server_port}/objects/{ref.digest}",
                headers={},
                expires_at=expiry,
            )

        grants = [grant(ref, size) for entry in manifest.files for ref, size in entry.objects()]
        uploaded = upload(grants, source, parallel=2)
        assert uploaded.ok

        destination = LocalCAS(tmp_path / "destination-cas")
        first = grants[0]
        first_object = source.verify_object(first.digest, size=first.size_bytes)
        destination.put_file(first_object, expected=first.digest, size=first.size_bytes)

        fetched = download(grants, destination, parallel=2)
        assert fetched.ok
        assert fetched.skipped_resident == 1
        restored = {entry.path: read_entry(destination, entry) for entry in manifest.files}
        assert restored == {"first": b"first", "second": b"second"}
        assert GrantHandler.requests.count(("GET", first.digest.digest)) == 0
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_python_grant_shape_matches_shared_vector_over_json() -> None:
    raw = Path("spec/v1/vectors/upload_grant.json").read_bytes().strip()
    grant = TransferGrant.from_wire(json.loads(raw))
    assert json.dumps(grant.to_wire(), separators=(",", ":")).encode() == raw


def test_real_http_out_of_order_chunks_reassemble_and_check_whole_digest(
    tmp_path: Path,
) -> None:
    server = GrantServer(("127.0.0.1", 0), GrantHandler)
    first_bytes = b"x" * MAX_CHUNK_SIZE
    last_bytes = b"end"
    first = CASRef.digest_bytes(first_bytes)
    last = CASRef.digest_bytes(last_bytes)
    whole = hashlib.sha256()
    whole.update(first_bytes)
    whole.update(last_bytes)
    entry = FileEntry(
        "large.bin",
        MAX_CHUNK_SIZE + len(last_bytes),
        CASRef(whole.hexdigest()),
        (Chunk(first, MAX_CHUNK_SIZE), Chunk(last, len(last_bytes))),
    )
    server.objects = {first.digest: first_bytes, last.digest: last_bytes}
    server.delay_digest = first.digest
    server.completed = []
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        expiry = (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace(
            "+00:00", "Z"
        )
        grants = [
            TransferGrant(
                ref,
                size,
                f"http://127.0.0.1:{server.server_port}/objects/{ref.digest}",
                expires_at=expiry,
            )
            for ref, size in entry.objects()
        ]
        cas = LocalCAS(tmp_path / "cas")
        report = download(grants, cas, parallel=2)
        assert report.ok
        assert server.completed == [last.digest, first.digest]

        restored = read_entry(cas, entry)
        assert len(restored) == MAX_CHUNK_SIZE + len(last_bytes)
        assert restored[:1] == b"x"
        assert restored[-3:] == b"end"

        wrong = FileEntry(entry.path, entry.size_bytes, CASRef.digest_bytes(b"wrong"), entry.chunks)
        try:
            read_entry(cas, wrong)
        except ValueError as exc:
            assert "whole" in str(exc) or "reconstructed" in str(exc)
        else:
            raise AssertionError("wrong whole-file digest was accepted")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
