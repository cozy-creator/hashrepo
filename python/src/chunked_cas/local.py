from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from .manifest import CHUNK_SIZE, Chunk, FileEntry, RepositoryManifest
from .refs import CASRef

_COPY_BUFFER = 1 << 20


class DigestMismatch(ValueError):
    """Bytes did not hash to the content reference they were stored under."""


class RefConflict(RuntimeError):
    """A logical ref changed from the value the caller observed."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_and_hash(
    source: BinaryIO, destination: BinaryIO, limit: int | None = None
) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied = 0
    while limit is None or copied < limit:
        wanted = _COPY_BUFFER if limit is None else min(_COPY_BUFFER, limit - copied)
        data = source.read(wanted)
        if not data:
            break
        destination.write(data)
        digest.update(data)
        copied += len(data)
    return digest.hexdigest(), copied


class LocalCAS:
    """Authoritative immutable local storage.

    Objects are installed without overwrite. Logical refs are tiny atomic
    records updated under a process-shared file lock. A new ``LocalCAS``
    instance can resolve everything written by a previous process.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.refs = self.root / "refs"
        self.locks = self.root / "locks"
        self.tmp = self.root / "tmp"
        for directory in (self.objects, self.refs, self.locks, self.tmp):
            directory.mkdir(parents=True, exist_ok=True)

    def object_path(self, ref: str | CASRef) -> Path:
        parsed = CASRef.parse(ref)
        return self.root / parsed.object_key()

    def verify_object(self, ref: str | CASRef, *, size: int | None = None) -> Path:
        parsed = CASRef.parse(ref)
        path = self.object_path(parsed)
        stat = path.stat()
        if size is not None and stat.st_size != size:
            raise DigestMismatch(f"{parsed}: object is {stat.st_size} bytes, expected {size}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while data := handle.read(_COPY_BUFFER):
                digest.update(data)
        if digest.hexdigest() != parsed.digest:
            raise DigestMismatch(f"{parsed}: local object bytes do not match their digest")
        return path

    def contains(self, ref: str | CASRef, *, size: int | None = None, verify: bool = True) -> bool:
        path = self.object_path(ref)
        if not path.is_file():
            return False
        if not verify:
            return size is None or path.stat().st_size == size
        self.verify_object(ref, size=size)
        return True

    def _commit_temp(self, temporary: Path, ref: CASRef, size: int) -> Path:
        destination = self.object_path(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(temporary, destination)
            _fsync_dir(destination.parent)
        except FileExistsError:
            self.verify_object(ref, size=size)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def put_bytes(self, data: bytes, *, expected: str | CASRef | None = None) -> CASRef:
        ref = CASRef.digest_bytes(data)
        expected_ref = CASRef.parse(expected) if expected is not None else None
        if expected_ref is not None and ref != expected_ref:
            raise DigestMismatch(f"bytes hash to {ref}, expected {expected_ref}")
        fd, raw_path = tempfile.mkstemp(prefix="put-", dir=self.tmp)
        temporary = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._commit_temp(temporary, ref, len(data))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return ref

    def _put_small_file(self, source: Path, size: int) -> CASRef:
        fd, raw_path = tempfile.mkstemp(prefix="put-", dir=self.tmp)
        temporary = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as writer:
                with source.open("rb") as reader:
                    before = os.fstat(reader.fileno())
                    digest, copied = _copy_and_hash(reader, writer)
                    after = os.fstat(reader.fileno())
                writer.flush()
                os.fsync(writer.fileno())
            if (
                copied != size
                or before.st_size != size
                or after.st_size != size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise OSError(f"{source} changed while it was being ingested")
            ref = CASRef(digest)
            self._commit_temp(temporary, ref, copied)
            return ref
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def ingest_file(self, source: str | Path, *, manifest_path: str | None = None) -> FileEntry:
        path = Path(source)
        initial = path.stat()
        if initial.st_size <= CHUNK_SIZE:
            digest = self._put_small_file(path, initial.st_size)
            return FileEntry(manifest_path or path.name, initial.st_size, digest)

        whole = hashlib.sha256()
        chunks: list[Chunk] = []
        copied = 0
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while data := handle.read(CHUNK_SIZE):
                copied += len(data)
                whole.update(data)
                digest = self.put_bytes(data)
                chunks.append(Chunk(digest, len(data)))
            after = os.fstat(handle.fileno())
        if (
            copied != initial.st_size
            or before.st_size != initial.st_size
            or after.st_size != initial.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise OSError(f"{path} changed while it was being ingested")
        return FileEntry(
            manifest_path or path.name,
            copied,
            CASRef(whole.hexdigest()),
            tuple(chunks),
        )

    def materialize(self, entry: FileEntry, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(raw_path)
        whole = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(fd, "wb") as writer:
                for ref, size in entry.objects():
                    source = self.verify_object(ref, size=size)
                    with source.open("rb") as reader:
                        object_hash = hashlib.sha256()
                        remaining = size
                        while remaining:
                            data = reader.read(min(_COPY_BUFFER, remaining))
                            if not data:
                                raise DigestMismatch(f"{ref}: object ended before {size} bytes")
                            writer.write(data)
                            object_hash.update(data)
                            whole.update(data)
                            total += len(data)
                            remaining -= len(data)
                        if reader.read(1):
                            raise DigestMismatch(f"{ref}: object exceeds {size} bytes")
                    if object_hash.hexdigest() != ref.digest:
                        raise DigestMismatch(f"{ref}: object changed while materializing")
                writer.flush()
                os.fsync(writer.fileno())
            if total != entry.size_bytes or whole.hexdigest() != entry.digest.digest:
                raise DigestMismatch(
                    f"{entry.path}: reconstructed bytes do not match the file manifest"
                )
            os.replace(temporary, target)
            _fsync_dir(target.parent)
            return target
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def store_manifest(self, manifest: RepositoryManifest) -> CASRef:
        return self.put_bytes(manifest.canonical_bytes())

    def load_manifest(self, ref: str | CASRef) -> RepositoryManifest:
        path = self.verify_object(ref)
        return RepositoryManifest.from_bytes(path.read_bytes())

    @staticmethod
    def _ref_id(name: str) -> str:
        if not name or any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError("logical ref name must be non-empty and contain no controls")
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def _read_ref_unlocked(self, name: str) -> CASRef | None:
        path = self.refs / self._ref_id(name)
        if not path.exists():
            return None
        raw = json.loads(path.read_bytes())
        if not isinstance(raw, dict) or raw.get("format") != 1 or raw.get("name") != name:
            raise ValueError(f"logical ref {name!r} is malformed")
        return CASRef.parse(str(raw.get("target", "")))

    def read_ref(self, name: str) -> CASRef | None:
        return self._read_ref_unlocked(name)

    @contextmanager
    def _ref_lock(self, name: str) -> Iterator[None]:
        path = self.locks / self._ref_id(name)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def compare_and_swap_ref(
        self,
        name: str,
        target: str | CASRef,
        *,
        expected: str | CASRef | None,
    ) -> CASRef:
        desired = CASRef.parse(target)
        expected_ref = CASRef.parse(expected) if expected is not None else None
        if not self.contains(desired):
            raise FileNotFoundError(f"cannot point {name!r} at absent object {desired}")
        with self._ref_lock(name):
            current = self._read_ref_unlocked(name)
            if current == desired:
                return desired
            if current != expected_ref:
                raise RefConflict(f"logical ref {name!r} is {current}, expected {expected_ref}")
            record = json.dumps(
                {"format": 1, "name": name, "target": str(desired)},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            fd, raw_path = tempfile.mkstemp(prefix="ref-", dir=self.refs)
            temporary = Path(raw_path)
            try:
                with os.fdopen(fd, "wb") as writer:
                    writer.write(record)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(temporary, self.refs / self._ref_id(name))
                _fsync_dir(self.refs)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        return desired
