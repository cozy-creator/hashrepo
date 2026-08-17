"""Project a manifest into a snapshot tree, in pure Python.

`Layout::project` (`crates/tensorfs-core/src/layout.rs`) is the Rust half of
this, and it is the half that cannot travel: python-gen-worker VENDORS
`python/src/tensorfs` as source (pgw#1310 -- both PyPI projects were deleted,
and a git pin is workspace-only metadata that a published wheel strips), so a
PyO3 extension is not reachable from pgw's wheel. Everything a consumer needs
to STOP materializing therefore has to exist here, without `_tensorfs`:

* :func:`stub_bytes` / :func:`parse_stub` -- TFSSTUB1, byte for byte;
* :func:`project_snapshot` -- the tree itself.

A projected tree derives entirely from the manifest, pins nothing, and can be
thrown away and rebuilt byte for byte. It costs one inode per file and no
tensor bytes at all: a blob is a relative symlink into ``objects/``, a tensor
container is a ~128 B pointer stub, and the real bytes are read through
:mod:`tensorfs.tensors`.

The one place this module writes real bytes is a CHUNKED non-tensor file --
a file whose contents exist only as several objects, so there is no inode to
point at and no tensor reader to serve it. That is bounded by the CAS scope
ruling (repos only; large non-tensor files do not belong in the chunked
store), and it is stated here rather than hidden because a projection that
silently copied would defeat the point of the layout.

## `body_sha256` per manifest dialect

`spec/v1/TFSSTUB1.md` defines the field as the TFM1 file-*body* hash and
argues the negative -- *"not the SHA-256 of the file's content"* -- from a
TFM1 fact: TFM1 gives a tensor container no whole-file hash, so demanding one
would force the projector to read every tensor byte. The v1 JSON manifest
(:mod:`tensorfs.manifest`, the dialect tensorhub speaks) does not share that
fact: every :class:`~tensorfs.manifest.FileEntry` already carries a whole-file
digest, derivable from the manifest alone at zero read cost. So in this
dialect the entry digest IS the body identity, and the spec says so.
"""

from __future__ import annotations

import errno
import itertools
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .manifest import FileEntry, RepositoryManifest
from .refs import CASRef

if TYPE_CHECKING:
    from .local import LocalCAS

__all__ = [
    "STUB_MAGIC",
    "TENSOR_SUFFIXES",
    "ProjectionError",
    "Stub",
    "is_tensor_container",
    "parse_stub",
    "project_snapshot",
    "read_stub",
    "stub_bytes",
    "tree_bytes",
]

#: The first eight bytes of every pointer stub, so a tool classifies one
#: without parsing JSON. Neither a safetensors `u64` header length nor the
#: GGUF magic -- a naive `open()` fails loudly at the parse site.
STUB_MAGIC = b"TFSSTUB1"

#: A file is a tensor container when its name says so. This is the same
#: suffix dispatch `TensorReader._parse_container` uses to decide what it can
#: serve, and the two must agree: a stub is a promise that the reader will
#: answer for that path.
TENSOR_SUFFIXES = (".safetensors", ".gguf")

_LOWER_HEX = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_MODE = 0o444
_SCRATCH_SEQUENCE = itertools.count()
_COPY_BLOCK = 1 << 20


class ProjectionError(RuntimeError):
    """A tree could not be projected from the manifest it was given."""


@dataclass(frozen=True, slots=True)
class Stub:
    """What one pointer stub says: a body identity and a logical size."""

    body_sha256: str
    size: int


def stub_bytes(body_sha256: str | CASRef, size: int) -> bytes:
    """One pointer stub's bytes: the magic, one space, one line of JSON.

    Emitted by hand rather than by a serializer, because the bytes are the
    contract (`spec/v1/TFSSTUB1.md`, `spec/v1/tfsstub1-vectors/`): key order,
    no whitespace, one trailing line feed.
    """

    digest = CASRef.parse(body_sha256) if ":" in str(body_sha256) else CASRef(str(body_sha256))
    if type(size) is not int or not 0 <= size < (1 << 64):
        raise ValueError(f"stub size must be a u64, got {size!r}")
    body = f'{{"body_sha256":"{digest.digest}","size":{size},"read":"tensorfs"}}'
    return STUB_MAGIC + b" " + body.encode("ascii") + b"\n"


def parse_stub(data: bytes) -> Stub | None:
    """Read one pointer stub, or ``None`` when the bytes are not one."""

    if not data.startswith(STUB_MAGIC + b" ") or not data.endswith(b"\n"):
        return None
    try:
        raw = json.loads(data[len(STUB_MAGIC) + 1 : -1])
    except ValueError:
        return None
    if not isinstance(raw, dict) or set(raw) != {"body_sha256", "size", "read"}:
        return None
    if raw["read"] != "tensorfs" or type(raw["size"]) is not int:
        return None
    # `CASRef` normalises case for a caller's convenience; a stub's bytes are
    # a contract, so the reader refuses what the writer would never emit.
    if not _LOWER_HEX.fullmatch(str(raw["body_sha256"])):
        return None
    if not 0 <= raw["size"] < (1 << 64):
        return None
    return Stub(str(raw["body_sha256"]), raw["size"])


def read_stub(path: str | Path) -> Stub | None:
    """The stub at ``path``, or ``None`` when that file is not one.

    Reads a bounded prefix, so pointing this at a real multi-gigabyte file
    costs one block rather than the file.
    """

    try:
        with open(path, "rb") as handle:
            head = handle.read(512)
    except OSError:
        return None
    return parse_stub(head)


def is_tensor_container(path: str) -> bool:
    """Whether a manifest path names a file the tensor reader can serve."""

    return path.endswith(TENSOR_SUFFIXES)


def _single_object(entry: FileEntry) -> CASRef | None:
    """The one CAS object holding this entry's whole contents, if there is one.

    A chunkless entry is one whole blob. A one-chunk entry is the same blob
    under a grid the manifest happens to spell out -- `FileEntry` already
    refuses a one-chunk entry whose digest disagrees, so the two are the same
    object and a symlink is exact.
    """

    if not entry.chunks:
        return entry.digest
    if len(entry.chunks) == 1:
        return entry.digest
    return None


def _relative_object_target(link: Path, object_path: Path) -> Path:
    """The symlink body: ``objects/…`` reached from the link's own directory.

    Computed from the two real paths rather than from the manifest path's
    depth, so a tree projected anywhere -- not only at
    ``<root>/snapshots/<id>`` -- gets a correct relative link.
    """

    return Path(os.path.relpath(object_path, link.parent))


def _install_bytes(target: Path, data: bytes) -> None:
    """Install a projection artifact -- a stub or an empty blob -- immutable."""

    with open(target, "wb") as handle:
        handle.write(data)
    os.chmod(target, _ARTIFACT_MODE)


def _copy_entry(cas: LocalCAS, entry: FileEntry, target: Path) -> None:
    """Reassemble a chunked non-tensor file: the one real copy this makes."""

    from .tensors import TensorReader

    reader = TensorReader(cas, RepositoryManifest((entry,)))
    try:
        with open(target, "wb") as handle:
            offset = 0
            while offset < entry.size_bytes:
                span = min(_COPY_BLOCK, entry.size_bytes - offset)
                handle.write(reader.read_range(entry.path, offset, span))
                offset += span
    finally:
        reader.close()
    os.chmod(target, _ARTIFACT_MODE)


def _project_entry(cas: LocalCAS, entry: FileEntry, target: Path, *, symlinks: bool) -> None:
    """One entry's projection.

    The dispatch is on what the ENTRY says and nothing else -- never on a
    source file that may not exist, which is what makes a projection cost no
    tensor bytes.
    """

    if is_tensor_container(entry.path):
        _install_bytes(target, stub_bytes(entry.digest, entry.size_bytes))
        return
    if entry.size_bytes == 0:
        _install_bytes(target, b"")
        return
    ref = _single_object(entry)
    if ref is None:
        _copy_entry(cas, entry, target)
        return
    source = cas.object_path(ref)
    if symlinks:
        os.symlink(_relative_object_target(target, source), target)
    else:
        # No symlinks on this filesystem: the tree carries a copy. Local
        # dedup is lost, correctness is kept.
        shutil.copyfile(source, target)
        os.chmod(target, _ARTIFACT_MODE)


def _fill(cas: LocalCAS, manifest: RepositoryManifest, root: Path, *, symlinks: bool) -> None:
    for entry in manifest.files:
        target = root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        _project_entry(cas, entry, target, symlinks=symlinks)


def _supports_symlinks(directory: Path) -> bool:
    probe = directory / f".symlink-probe-{os.getpid()}-{next(_SCRATCH_SEQUENCE)}"
    try:
        os.symlink("probe-target", probe)
    except (OSError, NotImplementedError):
        return False
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    return True


def project_snapshot(
    cas: LocalCAS,
    manifest: RepositoryManifest | CASRef | str,
    target: str | Path,
    *,
    symlinks: bool | None = None,
) -> Path:
    """Project one manifest as a tree at ``target`` and return that path.

    The tree is built under a private scratch name beside ``target`` and
    renamed into place, so no reader ever sees a half-built projection.
    Projection is idempotent and racy-safe in the same breath: an existing
    tree wins, and the loser's scratch is removed. Two projectors of the same
    manifest produce the same tree by construction, so a lost race costs
    nothing but the scratch.

    ``symlinks`` defaults to probing the target's filesystem once.
    """

    if not isinstance(manifest, RepositoryManifest):
        manifest = cas.load_manifest(manifest)
    final = Path(target)
    if final.exists():
        return final
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    if symlinks is None:
        symlinks = _supports_symlinks(parent)

    scratch = parent / f".building-{final.name}-{os.getpid()}-{next(_SCRATCH_SEQUENCE)}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    try:
        _fill(cas, manifest, scratch, symlinks=symlinks)
        os.rename(scratch, final)
    except OSError as error:
        shutil.rmtree(scratch, ignore_errors=True)
        # Another projector of the same manifest won the race. Its tree has
        # the same content by construction.
        if error.errno in (errno.EEXIST, errno.ENOTEMPTY) and final.exists():
            return final
        raise
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return final


def tree_bytes(root: str | Path) -> int:
    """Bytes the tree ITSELF occupies -- stubs and symlinks, never targets.

    The measurement the layout exists to make small, and the one a caller
    proving single-residency wants: `du` over a projected tree must be O(the
    number of files), not O(the model).
    """

    total = 0
    for directory, _subdirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            if not stat.S_ISLNK(info.st_mode):
                total += info.st_size
    return total
