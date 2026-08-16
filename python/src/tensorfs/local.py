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

from .manifest import RepositoryManifest
from .refs import CASRef

_COPY_BUFFER = 1 << 20


class DigestMismatch(ValueError):
    """Bytes did not hash to the content reference they were stored under."""


class RefConflict(RuntimeError):
    """A logical ref changed from the value the caller observed."""


class _ObjectWriter:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        written = self._handle.write(data)
        self._digest.update(data[:written])
        self.size += written
        return written

    def flush(self) -> None:
        self._handle.flush()

    def fileno(self) -> int:
        return self._handle.fileno()

    @property
    def ref(self) -> CASRef:
        return CASRef(self._digest.hexdigest())


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

    @contextmanager
    def _store_lock(self, *, exclusive: bool = False) -> Iterator[None]:
        """Coordinate collection with object and logical-ref operations."""

        with (self.locks / "store").open("a+b") as handle:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def object_path(self, ref: str | CASRef) -> Path:
        parsed = CASRef.parse(ref)
        return self.root / parsed.object_key()

    def _verify_object_unlocked(self, parsed: CASRef, size: int | None) -> Path:
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

    def verify_object(self, ref: str | CASRef, *, size: int | None = None) -> Path:
        parsed = CASRef.parse(ref)
        with self._store_lock():
            with self._object_lock(parsed):
                return self._verify_object_unlocked(parsed, size)

    def contains(self, ref: str | CASRef, *, size: int | None = None) -> bool:
        try:
            self.verify_object(ref, size=size)
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _require_identity(path: Path, expected: tuple[int, int]) -> None:
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise OSError(f"{path} changed after verification") from exc
        if (current.st_dev, current.st_ino) != expected:
            raise OSError(f"{path} changed after verification")

    def _commit_temp(
        self,
        temporary: Path,
        ref: CASRef,
        size: int,
        *,
        verified_identity: tuple[int, int] | None = None,
    ) -> Path:
        destination = self.object_path(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._store_lock():
            with self._object_lock(ref):
                try:
                    if verified_identity is not None:
                        self._require_identity(temporary, verified_identity)
                    os.link(temporary, destination, follow_symlinks=False)
                    if verified_identity is not None:
                        try:
                            self._require_identity(destination, verified_identity)
                        except OSError:
                            destination.unlink(missing_ok=True)
                            _fsync_dir(destination.parent)
                            raise
                    _fsync_dir(destination.parent)
                except FileExistsError:
                    try:
                        self._verify_object_unlocked(ref, size)
                    except DigestMismatch:
                        # The named object is already unusable. Replace it atomically
                        # with bytes that were verified before reaching this method.
                        if verified_identity is not None:
                            self._require_identity(temporary, verified_identity)
                        os.replace(temporary, destination)
                        if verified_identity is not None:
                            try:
                                self._require_identity(destination, verified_identity)
                            except OSError:
                                destination.unlink(missing_ok=True)
                                _fsync_dir(destination.parent)
                                raise
                        _fsync_dir(destination.parent)
                finally:
                    temporary.unlink(missing_ok=True)
        return destination

    def put_bytes(self, data: bytes, *, expected: str | CASRef | None = None) -> CASRef:
        ref = CASRef.digest_bytes(data)
        expected_ref = CASRef.parse(expected) if expected is not None else None
        if expected_ref is not None and ref != expected_ref:
            raise DigestMismatch(f"bytes hash to {ref}, expected {expected_ref}")
        with self._store_lock():
            with self._object_lock(ref):
                try:
                    self._verify_object_unlocked(ref, len(data))
                except (FileNotFoundError, DigestMismatch):
                    pass
                else:
                    return ref
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

    @contextmanager
    def open_writer(
        self,
        expected: str | CASRef,
        *,
        size: int,
    ) -> Iterator[_ObjectWriter]:
        """Hash a byte stream and atomically install its CAS object.

        The object becomes visible only after the context exits successfully,
        its digest and size match the declaration, and its bytes are durable.
        """

        expected_ref = CASRef.parse(expected)
        if type(size) is not int or size < 0:
            raise ValueError("object size must be a non-negative integer")
        fd, raw_path = tempfile.mkstemp(prefix="stream-", dir=self.tmp)
        temporary = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as handle:
                writer = _ObjectWriter(handle)
                yield writer
                writer.flush()
                os.fsync(writer.fileno())
                written = os.fstat(writer.fileno()).st_size
                if writer.size != size or written != size:
                    raise DigestMismatch(
                        f"{expected_ref}: stream is {written} bytes, expected {size}"
                    )
                if writer.ref != expected_ref:
                    raise DigestMismatch(f"stream hashes to {writer.ref}, expected {expected_ref}")
            self._commit_temp(temporary, expected_ref, size)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def adopt_file(
        self,
        temporary: str | Path,
        *,
        expected: str | CASRef,
        size: int,
    ) -> CASRef:
        """Verify and consume a file created in this CAS's temporary directory.

        The verified file itself is linked into the object namespace; its bytes
        are never copied into another temporary file. The input path is removed
        after either a successful install or a verification failure.
        """

        source = Path(temporary)
        expected_ref = CASRef.parse(expected)
        if type(size) is not int or size < 0:
            raise ValueError("object size must be a non-negative integer")
        if source.parent.resolve() != self.tmp.resolve():
            raise ValueError(f"adopted files must be direct children of {self.tmp}")

        try:
            digest = hashlib.sha256()
            with source.open("rb+") as handle:
                before = os.fstat(handle.fileno())
                copied = 0
                while data := handle.read(_COPY_BUFFER):
                    digest.update(data)
                    copied += len(data)
                os.fsync(handle.fileno())
                after = os.fstat(handle.fileno())
            if (
                copied != size
                or before.st_size != size
                or after.st_size != size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise DigestMismatch(f"{source}: file is not the declared {size} bytes")
            actual_ref = CASRef(digest.hexdigest())
            if actual_ref != expected_ref:
                raise DigestMismatch(
                    f"{source}: bytes hash to {actual_ref}, expected {expected_ref}"
                )
            self._commit_temp(
                source,
                expected_ref,
                size,
                verified_identity=(after.st_dev, after.st_ino),
            )
            return expected_ref
        except BaseException:
            source.unlink(missing_ok=True)
            raise

    def put_file(
        self,
        source: str | Path,
        *,
        expected: str | CASRef | None = None,
        size: int | None = None,
    ) -> CASRef:
        """Install one object from a file after hashing it exactly once."""

        source_path = Path(source)
        initial = source_path.stat()
        expected_size = initial.st_size if size is None else size
        if initial.st_size != expected_size:
            raise DigestMismatch(
                f"{source_path}: source is {initial.st_size} bytes, expected {expected_size}"
            )
        fd, raw_path = tempfile.mkstemp(prefix="put-", dir=self.tmp)
        temporary = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as writer:
                with source_path.open("rb") as reader:
                    before = os.fstat(reader.fileno())
                    digest, copied = _copy_and_hash(reader, writer)
                    after = os.fstat(reader.fileno())
                writer.flush()
                os.fsync(writer.fileno())
            if (
                copied != expected_size
                or before.st_size != expected_size
                or after.st_size != expected_size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise OSError(f"{source_path} changed while it was being ingested")
            ref = CASRef(digest)
            expected_ref = CASRef.parse(expected) if expected is not None else None
            if expected_ref is not None and ref != expected_ref:
                raise DigestMismatch(f"{source_path}: bytes hash to {ref}, expected {expected_ref}")
            self._commit_temp(temporary, ref, copied)
            return ref
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def store_manifest(self, manifest: RepositoryManifest) -> CASRef:
        return self.put_bytes(manifest.canonical_bytes())

    def load_manifest(self, ref: str | CASRef) -> RepositoryManifest:
        parsed = CASRef.parse(ref)
        with self._store_lock():
            with self._object_lock(parsed):
                path = self._verify_object_unlocked(parsed, None)
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
        if (
            not isinstance(raw, dict)
            or set(raw) != {"format", "name", "target"}
            or raw.get("format") != 1
            or raw.get("name") != name
        ):
            raise ValueError(f"logical ref {name!r} is malformed")
        return CASRef.parse(str(raw.get("target", "")))

    def read_ref(self, name: str) -> CASRef | None:
        with self._store_lock():
            return self._read_ref_unlocked(name)

    @contextmanager
    def _object_lock(self, ref: CASRef) -> Iterator[None]:
        path = self.locks / f"object-{ref.digest}"
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
        target: str | CASRef | None,
        *,
        expected: str | CASRef | None,
    ) -> CASRef | None:
        desired = CASRef.parse(target) if target is not None else None
        expected_ref = CASRef.parse(expected) if expected is not None else None
        with self._store_lock():
            if desired is not None:
                try:
                    with self._object_lock(desired):
                        self._verify_object_unlocked(desired, None)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"cannot point {name!r} at absent object {desired}"
                    ) from None
            with self._ref_lock(name):
                current = self._read_ref_unlocked(name)
                if current == desired:
                    return desired
                if current != expected_ref:
                    raise RefConflict(f"logical ref {name!r} is {current}, expected {expected_ref}")
                destination = self.refs / self._ref_id(name)
                if desired is None:
                    destination.unlink(missing_ok=True)
                    _fsync_dir(self.refs)
                    return None
                record = json.dumps(
                    {"format": 1, "name": name, "target": str(desired)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                fd, raw_path = tempfile.mkstemp(prefix="ref-", dir=self.tmp)
                temporary = Path(raw_path)
                try:
                    with os.fdopen(fd, "wb") as writer:
                        writer.write(record)
                        writer.flush()
                        os.fsync(writer.fileno())
                    os.replace(temporary, destination)
                    _fsync_dir(self.refs)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
        return desired
