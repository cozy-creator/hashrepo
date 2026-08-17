from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import BinaryIO

from .manifest import RepositoryManifest
from .refs import CASRef

try:
    import fcntl
except ModuleNotFoundError as exc:  # pragma: no cover - not reachable on POSIX
    # There is no Windows wheel and `import tensorfs` cannot succeed there:
    # the store's advisory locking is `fcntl.flock`, and `msvcrt.locking` is
    # byte-range with no shared mode, so it is not a drop-in. Say that here
    # rather than leaving a bare `No module named 'fcntl'`, which names the
    # symptom and not the support boundary. Restoring Windows means writing a
    # real POSIX-lock replacement (tensorfs#57), not editing a wheel matrix.
    raise ImportError(
        "tensorfs supports Linux and macOS only. This interpreter has no "
        "`fcntl` module, so the content store cannot take its advisory locks."
    ) from exc

_COPY_BUFFER = 1 << 20

# The namespace every temporary this library creates lives in, and nothing
# else does. A file under `tmp/` outside it was made by a caller, carries no
# lease, and is never collected.
_TEMP_PREFIX = "tfs-"

# Half-built projection trees. The rest of the name is the lease token, so a
# reaper correlates the two without a registry.
_BUILDING_PREFIX = ".building-"


class DigestMismatch(ValueError):
    """Bytes did not hash to the content reference they were stored under."""


class RefConflict(RuntimeError):
    """A logical ref changed from the value the caller observed."""


@dataclass(frozen=True)
class Reclaimed:
    """One abandoned artifact, described while it still exists."""

    name: str
    creator: int | None
    bytes_deleted: int


@dataclass(frozen=True)
class TempCollection:
    """What one sweep of the temporary directory examined and reclaimed.

    ``reclaimed`` is the evidence, captured before each unlink: a sweep that
    reports only a count leaves nobody able to say afterwards whose resources
    it freed, which is how the orphaned-FUSE-connection incident (#96) stayed
    invisible. It travels out in the report so a caller can log it wherever
    survives the process that swept.
    """

    examined: int = 0
    deleted: int = 0
    bytes_deleted: int = 0
    reclaimed: tuple[Reclaimed, ...] = ()


def _tree_bytes(root: Path) -> int:
    """What a scratch tree occupies, counted without following links."""

    total = 0
    for directory, _subdirectories, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(Path(directory) / name).st_size
            except OSError:
                continue
    return total


def _creator_of(name: str) -> int | None:
    """The pid a library artifact's name records, for evidence only."""

    parts = name.split("-")
    return int(parts[2]) if len(parts) > 3 and parts[2].isdigit() else None


def _identity(status: os.stat_result) -> tuple[int, int, int, int]:
    """The inode a name resolved to, so a recycled name is never mistaken."""

    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


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

    @contextmanager
    def _leased_temp(
        self, kind: str, *, store_locked: bool = False
    ) -> Iterator[tuple[BinaryIO, Path]]:
        """A temporary file that exists only while its creator does.

        The lease is an exclusive ``flock`` on the temp's own descriptor, held
        for the whole write, verification and installation lifetime, and taken
        while the store lock is held. Collection runs under the exclusive store
        lock, so it can never observe a temp that is not yet leased -- which is
        why reclaiming needs no clock at all.
        """

        guard = nullcontext() if store_locked else self._store_lock()
        with guard:
            # The pid is in the NAME as evidence for whoever reads a reap
            # report, never as the ownership test -- that is the lease.
            prefix = f"{_TEMP_PREFIX}{kind}-{os.getpid()}-"
            descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=self.tmp)
            path = Path(raw_path)
            try:
                # A freshly created unique name cannot already be leased.
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle: BinaryIO = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                path.unlink(missing_ok=True)
                raise
        try:
            yield handle, path
        finally:
            handle.close()
            path.unlink(missing_ok=True)

    @contextmanager
    def open_temp(self) -> Iterator[Path]:
        """A leased temporary file for a caller to fill and then ``adopt_file``.

        Bytes staged through this path are reclaimable the instant this process
        dies and are never reclaimed while it lives. A temporary the caller
        creates itself holds no lease, so the collector leaves it alone.
        """

        with self._leased_temp("adopt") as (_handle, path):
            yield path

    @contextmanager
    def scratch_lease(self) -> Iterator[str]:
        """A token that names projection scratch, leased while it exists.

        Held for the whole build and rename, so ``reap_projection_scratch``
        can tell a projection that crashed from one that is merely slow. The
        lease is a file in this store's ``tmp/`` named by the token itself,
        because a directory cannot portably carry an advisory lock.
        """

        with self._leased_temp("scratch") as (_handle, path):
            yield path.name

    def reap_projection_scratch(self, parent: str | Path) -> TempCollection:
        """Removes projection scratch under ``parent`` whose creator is gone.

        Scratch is built beside the tree it will become, which can be any
        directory, so the caller names it. Liveness decides and nothing else:
        a half-built tree goes only when the lease naming it is free or has
        already been reclaimed, so a projection still running -- for a second
        or for an hour -- is never touched.
        """

        examined = 0
        evidence: list[Reclaimed] = []
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.name.startswith(_BUILDING_PREFIX) or not entry.is_dir(
                    follow_symlinks=False
                ):
                    continue
                token = entry.name[len(_BUILDING_PREFIX) :]
                # Only tokens this library issued: anything else is a foreign
                # `.building-…` and is left alone, the way a foreign temp is.
                if not token.startswith(_TEMP_PREFIX):
                    continue
                examined += 1
                if not self._lease_is_free(self.tmp / token):
                    continue
                # Measured and recorded BEFORE the removal: afterwards
                # nothing on disk can say who left it or what it cost.
                freed = _tree_bytes(Path(entry.path))
                creator = _creator_of(token)
                try:
                    shutil.rmtree(entry.path)
                except OSError:
                    continue
                evidence.append(Reclaimed(entry.name, creator, freed))
        return TempCollection(
            examined,
            len(evidence),
            sum(item.bytes_deleted for item in evidence),
            tuple(evidence),
        )

    @staticmethod
    def _lease_is_free(path: Path) -> bool:
        """Whether no live process holds this lease.

        One sample answers it. A drain check across two samples -- the shape
        #96 needs for a wedged FUSE connection, which has no owner left to
        ask -- could only repeat this answer, because here the owner IS the
        lock.
        """

        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        finally:
            os.close(descriptor)
        return True

    def collect_abandoned_temps(self) -> TempCollection:
        """Reclaims the temporaries of creators that are gone.

        No age and no clock: a library temp is leased before it is visible to
        this sweep and stays leased until it is installed or removed, so a
        candidate whose lease this call can take has no live holder. That is
        strictly better than a wall-clock grace, which must either delete a
        slow writer's temp or keep a crashed writer's forever.

        Cadence stays with the caller; there is no background thread and no
        startup scan.
        """

        examined = 0
        evidence: list[Reclaimed] = []
        with self._store_lock(exclusive=True):
            with os.scandir(self.tmp) as entries:
                for entry in entries:
                    if not entry.name.startswith(_TEMP_PREFIX):
                        continue
                    try:
                        listed = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if not S_ISREG(listed.st_mode):
                        continue
                    examined += 1
                    reclaimed = self._reclaim_temp(Path(entry.path), listed)
                    if reclaimed is not None:
                        evidence.append(
                            Reclaimed(entry.name, _creator_of(entry.name), reclaimed)
                        )
            if evidence:
                _fsync_dir(self.tmp)
        return TempCollection(
            examined,
            len(evidence),
            sum(item.bytes_deleted for item in evidence),
            tuple(evidence),
        )

    @staticmethod
    def _reclaim_temp(path: Path, listed: os.stat_result) -> int | None:
        """Bytes reclaimed, or ``None`` when either guard retains the file.

        The guards are the lease, which decides liveness, and the inode this
        name resolved to at listing, so a name recycled between the listing
        and the open is never the file that gets unlinked.
        """

        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            return None
        try:
            held = os.fstat(descriptor)
            if _identity(held) != _identity(listed):
                return None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return None
            os.unlink(path)
        except OSError:
            return None
        finally:
            os.close(descriptor)
        return held.st_size

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
        """Whether the object is resident. Presence, not integrity.

        One ``lstat``. Bytes are verified once, by the writer that admits
        them, and this store never rehashes to answer "do I have it?" -- the
        question a resume journal, a plan or a fetch decision actually asks.
        A declared ``size`` is compared against the stat already taken, so a
        truncated object still answers False for free; bytes corrupted in
        place answer True, by design. ``verify_object`` is the scrub for
        suspicion, and it is the only thing that rehashes.
        """

        try:
            status = self.object_path(ref).stat(follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            return False
        if not S_ISREG(status.st_mode):
            return False
        return size is None or status.st_size == size

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
        with self._leased_temp("put") as (handle, temporary):
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            self._commit_temp(temporary, ref, len(data))
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
        with self._leased_temp("stream") as (handle, temporary):
            writer = _ObjectWriter(handle)
            yield writer
            writer.flush()
            os.fsync(writer.fileno())
            written = os.fstat(writer.fileno()).st_size
            if writer.size != size or written != size:
                raise DigestMismatch(f"{expected_ref}: stream is {written} bytes, expected {size}")
            if writer.ref != expected_ref:
                raise DigestMismatch(f"stream hashes to {writer.ref}, expected {expected_ref}")
            self._commit_temp(temporary, expected_ref, size)

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
        with self._leased_temp("put") as (writer, temporary):
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
            # Presence, not integrity: pointing a ref at an object is not an
            # admission, and rehashing a 64 MiB object to answer "is it there?"
            # is the cost issue #42 is about.
            if desired is not None and not self.contains(desired):
                raise FileNotFoundError(f"cannot point {name!r} at absent object {desired}")
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
                with self._leased_temp("ref", store_locked=True) as (writer, temporary):
                    writer.write(record)
                    writer.flush()
                    os.fsync(writer.fileno())
                    os.replace(temporary, destination)
                    _fsync_dir(self.refs)
        return desired
