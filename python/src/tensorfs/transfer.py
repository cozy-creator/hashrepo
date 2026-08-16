from __future__ import annotations

import http.client
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .local import DigestMismatch, LocalCAS
from .refs import CASRef

_DEFAULT_PARALLEL = 8
_PROCESS_TRANSFER_BUDGET = threading.BoundedSemaphore(16)
_EXPIRY_MARGIN_SECONDS = 120.0
_RFC3339_UTC = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z"
)


class _GrantExpired(RuntimeError):
    """The caller should obtain a fresh remote plan, not retry this URL."""


class _TransferRefused(RuntimeError):
    """The remote endpoint rejected the bytes or authorization terminally."""


class _TransientTransferError(RuntimeError):
    """A transport failure that may succeed on the same grant after backoff."""


class _HTTPStatusError(RuntimeError):
    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"HTTP {status}: {detail}".rstrip())
        self.status = status
        self.detail = detail


@dataclass(frozen=True, slots=True)
class TransferGrant:
    """One opaque, expiring authorization for one exact CAS object."""

    digest: CASRef
    size_bytes: int
    url: str
    staging_key: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    expires_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", CASRef.parse(self.digest))
        object.__setattr__(self, "headers", dict(self.headers))
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("grant size must be a non-negative integer")
        if not self.url.strip():
            raise ValueError("grant URL must not be empty")
        self.expiry_unix()

    def expiry_unix(self) -> float | None:
        if self.expires_at is None:
            return None
        text = self.expires_at.strip()
        if not text:
            raise ValueError("grant expiry must be an RFC 3339 timestamp or None")
        if _RFC3339_UTC.fullmatch(text) is None:
            raise ValueError("grant expiry must be an RFC 3339 UTC timestamp ending in Z")
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError(f"invalid grant expiry {self.expires_at!r}") from exc
        return parsed.timestamp()

    def expired(self, *, now: float | None = None, margin: float = _EXPIRY_MARGIN_SECONDS) -> bool:
        expiry = self.expiry_unix()
        return expiry is not None and (time.time() if now is None else now) + margin >= expiry

    @classmethod
    def from_wire(cls, value: Mapping[str, object]) -> TransferGrant:
        """Decode the exact v1 upload-grant object emitted by the Go planner."""

        required = {"digest", "size_bytes", "staging_key", "url", "headers", "expires_at"}
        if set(value) != required:
            raise ValueError(f"upload grant fields must be exactly {sorted(required)}")
        headers = value["headers"]
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in headers.items()
        ):
            raise ValueError("upload grant headers must be a string-to-string object")
        digest = value["digest"]
        size_bytes = value["size_bytes"]
        staging_key = value["staging_key"]
        url = value["url"]
        expires_at = value["expires_at"]
        if not isinstance(digest, str):
            raise ValueError("upload grant digest must be a string")
        if type(size_bytes) is not int:
            raise ValueError("upload grant size_bytes must be an integer")
        if not isinstance(staging_key, str) or not staging_key:
            raise ValueError("upload grant staging_key must be a non-empty string")
        if not isinstance(url, str):
            raise ValueError("upload grant URL must be a string")
        if not isinstance(expires_at, str):
            raise ValueError("upload grant expires_at must be a string")
        return cls(
            CASRef.parse(digest),
            size_bytes,
            url,
            staging_key=staging_key,
            headers=headers,
            expires_at=expires_at,
        )

    def to_wire(self) -> dict[str, object]:
        if not self.staging_key:
            raise ValueError("wire upload grants require a staging key")
        if self.expires_at is None:
            raise ValueError("wire upload grants require an expiry")
        return {
            "digest": str(self.digest),
            "size_bytes": self.size_bytes,
            "staging_key": self.staging_key,
            "url": self.url,
            "headers": dict(self.headers),
            "expires_at": self.expires_at,
        }


def _discard_bytes(_moved: int) -> None:
    """The byte reporter of a transfer nobody is watching."""
    return None


def _byte_reporter(
    sink: Callable[[CASRef, int], None] | None, digest: CASRef
) -> Callable[[int], None]:
    if sink is None:
        return _discard_bytes

    def moved(count: int) -> None:
        sink(digest, count)

    return moved


class _GrantTransport(Protocol):
    def upload(
        self, grant: TransferGrant, source: Path, on_bytes: Callable[[int], None]
    ) -> None: ...

    def download(
        self, grant: TransferGrant, destination: Path, on_bytes: Callable[[int], None]
    ) -> None: ...


class _HTTPTransport:
    """Small stdlib transport for verbatim grant URLs and headers."""

    def __init__(self, *, timeout: float = 120.0, block_bytes: int = 1 << 20) -> None:
        self.timeout = timeout
        self.block_bytes = block_bytes

    def _open(self, request: urllib.request.Request) -> object:
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(1024).decode("utf-8", "replace")
            finally:
                exc.close()
            raise _HTTPStatusError(exc.code, detail) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise _TransientTransferError(str(exc)) from exc

    def upload(
        self,
        grant: TransferGrant,
        source: Path,
        on_bytes: Callable[[int], None] = _discard_bytes,
    ) -> None:
        if source.stat().st_size != grant.size_bytes:
            raise DigestMismatch(
                f"{grant.digest}: upload source is {source.stat().st_size} bytes, "
                f"expected {grant.size_bytes}"
            )

        def body() -> Iterator[bytes]:
            with source.open("rb") as handle:
                while block := handle.read(self.block_bytes):
                    on_bytes(len(block))
                    yield block

        headers = dict(grant.headers)
        content_lengths = [
            value for key, value in headers.items() if key.lower() == "content-length"
        ]
        if content_lengths and content_lengths != [str(grant.size_bytes)]:
            raise _TransferRefused(f"{grant.digest}: grant has an incorrect Content-Length")
        if not content_lengths:
            headers["Content-Length"] = str(grant.size_bytes)
        request = urllib.request.Request(
            grant.url, data=body(), headers=headers, method="PUT"
        )
        response = self._open(request)
        with response:  # type: ignore[attr-defined]
            status = int(getattr(response, "status", 0))
            if not 200 <= status < 300:
                raise _HTTPStatusError(status)

    def download(
        self,
        grant: TransferGrant,
        destination: Path,
        on_bytes: Callable[[int], None] = _discard_bytes,
    ) -> None:
        request = urllib.request.Request(grant.url, headers=dict(grant.headers), method="GET")
        response = self._open(request)
        copied = 0
        try:
            with response:  # type: ignore[attr-defined]
                status = int(getattr(response, "status", 0))
                if not 200 <= status < 300:
                    raise _HTTPStatusError(status)
                with destination.open("wb") as output:
                    while block := response.read(self.block_bytes):  # type: ignore[attr-defined]
                        copied += len(block)
                        if copied > grant.size_bytes:
                            raise DigestMismatch(
                                f"{grant.digest}: response exceeds {grant.size_bytes} bytes"
                            )
                        output.write(block)
                        # Reported as the block lands, not once the object is
                        # whole: a 64 MiB object on a slow link is minutes of
                        # silence otherwise.
                        on_bytes(len(block))
                    output.flush()
                    os.fsync(output.fileno())
        except (http.client.IncompleteRead, OSError, urllib.error.URLError) as exc:
            raise _TransientTransferError(str(exc)) from exc
        if copied != grant.size_bytes:
            raise DigestMismatch(
                f"{grant.digest}: response is {copied} bytes, expected {grant.size_bytes}"
            )


@dataclass(slots=True)
class TransferReport:
    examined: int = 0
    succeeded: int = 0
    skipped_resident: int = 0
    bytes_transferred: int = 0
    expired: list[CASRef] = field(default_factory=list)
    failures: list[tuple[CASRef, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.expired
            and not self.failures
            and self.succeeded + self.skipped_resident == self.examined
        )

    @property
    def needs_replan(self) -> bool:
        return bool(self.expired) and not self.failures


def _retry(
    grant: TransferGrant,
    action: Callable[[], None],
    *,
    max_attempts: int,
    sleep: Callable[[float], None],
) -> None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    last: BaseException | None = None
    for attempt in range(max_attempts):
        if grant.expired():
            raise _GrantExpired(f"grant for {grant.digest} is expired or inside its safety margin")
        try:
            action()
            return
        except _HTTPStatusError as exc:
            if exc.status == 403 and grant.expired(margin=0):
                raise _GrantExpired(f"grant for {grant.digest} expired during transfer") from exc
            if exc.status not in (408, 425, 429) and exc.status < 500:
                raise _TransferRefused(f"{grant.digest}: {exc}") from exc
            last = exc
        except _TransientTransferError as exc:
            last = exc
        if attempt + 1 < max_attempts:
            ceiling = min(30.0, 0.25 * (2**attempt))
            sleep(random.uniform(ceiling / 2, ceiling))
    raise _TransientTransferError(
        f"{grant.digest}: transfer failed after {max_attempts} attempts: {last}"
    )


def _unique(grants: Sequence[TransferGrant]) -> tuple[TransferGrant, ...]:
    ordered = tuple(grants)
    seen: set[CASRef] = set()
    for grant in ordered:
        if grant.digest in seen:
            raise ValueError(f"duplicate grant for {grant.digest}")
        seen.add(grant.digest)
    return ordered


def _run_parallel(
    grants: Sequence[TransferGrant],
    worker: Callable[[TransferGrant], bool],
    *,
    parallel: int,
    progress: Callable[[CASRef, int], None] | None,
) -> TransferReport:
    ordered = _unique(grants)
    if parallel < 1:
        raise ValueError("parallel must be positive")
    results: list[tuple[str, str] | None] = [None] * len(ordered)

    def run(index: int, grant: TransferGrant) -> None:
        with _PROCESS_TRANSFER_BUDGET:
            try:
                skipped = worker(grant)
                results[index] = ("skipped" if skipped else "succeeded", "")
            except _GrantExpired as exc:
                results[index] = ("expired", str(exc))
            except Exception as exc:
                results[index] = ("failed", f"{type(exc).__name__}: {exc}")

    # `progress` answers "is this transfer advancing toward readiness", so a
    # RESIDENT object advances it exactly as a fetched one does: verifying an
    # object already on disk is completed work toward readiness, and callers
    # size their denominator over every planned object. Counting only fetched
    # bytes made a resumed transfer — where the previous attempt already landed
    # most objects — look identical to a dead one, which is how a healthy pod
    # gets condemned. `bytes_transferred` still counts only what actually moved.
    #
    # `progress` fires as each object lands, NOT from the report loop below.
    # Emitting after the drain means a multi-gigabyte transfer reports nothing
    # until its last object is on disk, and a caller that watches the counter
    # for liveness — tensorhub's transfer freshness window, and the worker's
    # own stall detector — condemns a download that is proceeding correctly.
    # Objects are capped at 64 MiB, so a live transfer now advances at least
    # once per 64 MiB.
    #
    # It still runs on the CALLER's thread, from the completion loop rather
    # than from inside `run`. Callers hand this callback shared state that is
    # not theirs to lock, so moving it onto the pool's threads would fix the
    # timing by breaking the contract.
    #
    # `bytes_progress` is the finer signal underneath it, and it cannot share
    # that thread: bytes move inside the worker, and queueing them for the
    # completion loop would only report them once the object landed — which is
    # the coarse signal again. So it fires on the worker moving the bytes and
    # says so in its docstring; a consumer that wants both watches the object
    # counter for readiness and the byte counter for liveness. A retried block
    # is reported twice, because it moved twice.
    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="tensorfs") as pool:
        futures = {
            pool.submit(run, index, grant): index
            for index, grant in enumerate(ordered)
        }
        for future in as_completed(futures):
            future.result()
            index = futures[future]
            result = results[index]
            assert result is not None
            if result[0] in ("succeeded", "skipped") and progress is not None:
                progress(ordered[index].digest, ordered[index].size_bytes)

    report = TransferReport(examined=len(ordered))
    for grant, result in zip(ordered, results, strict=True):
        assert result is not None
        status, detail = result
        if status == "succeeded":
            report.succeeded += 1
            report.bytes_transferred += grant.size_bytes
        elif status == "skipped":
            report.skipped_resident += 1
        elif status == "expired":
            report.expired.append(grant.digest)
        else:
            report.failures.append((grant.digest, detail))
    return report


def _upload(
    grants: Sequence[TransferGrant],
    source: LocalCAS,
    *,
    transport: _GrantTransport,
    parallel: int = _DEFAULT_PARALLEL,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[CASRef, int], None] | None = None,
    bytes_progress: Callable[[CASRef, int], None] | None = None,
) -> TransferReport:
    """Upload a remote plan's missing objects from an authoritative local CAS."""

    def one(grant: TransferGrant) -> bool:
        path = source.verify_object(grant.digest, size=grant.size_bytes)
        moved = _byte_reporter(bytes_progress, grant.digest)
        _retry(
            grant,
            lambda: transport.upload(grant, path, moved),
            max_attempts=max_attempts,
            sleep=sleep,
        )
        return False

    return _run_parallel(grants, one, parallel=parallel, progress=progress)


def upload(
    grants: Sequence[TransferGrant],
    source: LocalCAS,
    *,
    parallel: int = _DEFAULT_PARALLEL,
    max_attempts: int = 5,
    progress: Callable[[CASRef, int], None] | None = None,
    bytes_progress: Callable[[CASRef, int], None] | None = None,
) -> TransferReport:
    """Upload a remote plan's missing objects through their HTTP grants.

    `progress(digest, size)` fires on the caller's thread as each object
    lands. `bytes_progress(digest, moved)` fires on the worker thread that is
    moving those bytes, once per block, so a single large object is visible
    while it is still in flight; it must be safe to call from any thread.
    """

    return _upload(
        grants,
        source,
        transport=_HTTPTransport(),
        parallel=parallel,
        max_attempts=max_attempts,
        progress=progress,
        bytes_progress=bytes_progress,
    )


def _download(
    grants: Sequence[TransferGrant],
    destination: LocalCAS,
    *,
    transport: _GrantTransport,
    parallel: int = _DEFAULT_PARALLEL,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[CASRef, int], None] | None = None,
    bytes_progress: Callable[[CASRef, int], None] | None = None,
) -> TransferReport:
    """Fetch missing objects; resident verified objects are the resume journal."""

    def one(grant: TransferGrant) -> bool:
        try:
            if destination.contains(grant.digest, size=grant.size_bytes):
                return True
        except DigestMismatch:
            # Re-fetch. LocalCAS.adopt_file atomically replaces only the corrupt
            # object at this same digest after checking the new bytes.
            pass
        descriptor, name = tempfile.mkstemp(prefix="download-", dir=destination.tmp)
        os.close(descriptor)
        temporary = Path(name)
        moved = _byte_reporter(bytes_progress, grant.digest)
        try:
            _retry(
                grant,
                lambda: transport.download(grant, temporary, moved),
                max_attempts=max_attempts,
                sleep=sleep,
            )
            destination.adopt_file(temporary, expected=grant.digest, size=grant.size_bytes)
        finally:
            temporary.unlink(missing_ok=True)
        return False

    return _run_parallel(grants, one, parallel=parallel, progress=progress)


def download(
    grants: Sequence[TransferGrant],
    destination: LocalCAS,
    *,
    parallel: int = _DEFAULT_PARALLEL,
    max_attempts: int = 5,
    progress: Callable[[CASRef, int], None] | None = None,
    bytes_progress: Callable[[CASRef, int], None] | None = None,
) -> TransferReport:
    """Download missing objects through HTTP grants into an authoritative CAS.

    `progress(digest, size)` fires on the caller's thread as each object lands
    or is found resident. `bytes_progress(digest, moved)` fires on the worker
    thread that is moving those bytes, once per block, so a 64 MiB object on a
    slow link reports while it arrives; it must be safe to call from any
    thread.
    """

    return _download(
        grants,
        destination,
        transport=_HTTPTransport(),
        parallel=parallel,
        max_attempts=max_attempts,
        progress=progress,
        bytes_progress=bytes_progress,
    )
