"""The extension exports a real ``Py_buffer``, and the read path uses it.

This is #57's abi3 row, forced rather than argued. The question that row asks
is whether the C buffer protocol is expressible under the Limited API at this
distribution's floor. Documentation says it is (``Misc/stable_abi.toml``,
bpo-45459); PyO3 agrees by gating its own buffer-protocol tests on exactly
``any(not(Py_LIMITED_API), Py_3_11)``. Neither is evidence. A built abi3
extension handing out a mapped, borrowed, read-only buffer is.

The load-bearing assertion is :func:`test_a_view_tracks_the_file_it_maps`: it
edits the mapped file underneath a live view and demands the view see it. No
implementation that copies -- at the Rust boundary, in ``bytes``, into a
``bytearray``, anywhere -- can pass it.
"""

from __future__ import annotations

import inspect
import io
import random
import struct
import sys
from pathlib import Path

import pytest
from tensorfs import LocalCAS, RepositoryManifest, TensorWriter, open_tensors
from tensorfs.native import MappedObject


def _object(tmp_path: Path, body: bytes) -> Path:
    path = tmp_path / "object"
    path.write_bytes(body)
    return path


# ---------------------------------------------------------------------------
# The export itself
# ---------------------------------------------------------------------------


def test_the_extension_is_the_abi3_build_that_exports_the_buffer() -> None:
    """Tie the export to the artifact whose ABI floor is in question."""

    extension = Path(sys.modules["tensorfs._tensorfs"].__file__ or "")
    assert "abi3" in extension.name, extension
    assert MappedObject.__module__ == "tensorfs._tensorfs"


def test_a_mapped_object_exports_a_read_only_c_buffer(tmp_path: Path) -> None:
    body = random.Random("buffer").randbytes(4096)
    mapped = MappedObject(_object(tmp_path, body))

    view = memoryview(mapped)
    assert view.obj is mapped, "the exporter is the mapping, so nothing was copied"
    assert view.readonly
    assert view.format == "B"
    assert view.ndim == 1
    assert view.shape == (len(body),)
    assert view.nbytes == len(body) == len(mapped) == mapped.length
    assert bytes(view) == body


def test_a_writable_buffer_request_is_refused(tmp_path: Path) -> None:
    """A CAS object is named by its own bytes; a writable view could rename it."""

    mapped = MappedObject(_object(tmp_path, b"immutable"))

    # `readinto` is the portable way to demand PyBUF_WRITABLE from Python.
    # CPython reports the refused export as a TypeError naming the type.
    with pytest.raises(TypeError, match="read-write"):
        io.BytesIO(b"Y" * 9).readinto(mapped)

    if sys.version_info >= (3, 12):
        # PEP 688 exposes the flags directly, so the refusal can be read in
        # the extension's own words.
        with pytest.raises(BufferError, match="immutable"):
            mapped.__buffer__(inspect.BufferFlags.WRITABLE)

    # The read-only export is unaffected.
    assert bytes(memoryview(mapped)) == b"immutable"


def test_an_empty_object_exports_an_empty_buffer(tmp_path: Path) -> None:
    """mmap(2) refuses a zero-length mapping; the export must not."""

    mapped = MappedObject(_object(tmp_path, b""))
    assert len(mapped) == 0
    assert bytes(memoryview(mapped)) == b""


def test_the_buffer_keeps_the_mapping_alive(tmp_path: Path) -> None:
    """Ownership is the buffer protocol's, not the caller's."""

    body = random.Random("outlive").randbytes(1024)
    view = memoryview(MappedObject(_object(tmp_path, body)))
    # The only strong reference to the MappedObject is the one Py_buffer took.
    assert bytes(view) == body


def test_a_view_tracks_the_file_it_maps(tmp_path: Path) -> None:
    """The discriminator: a copy cannot see an edit made after it was taken.

    Every byte-equality assertion elsewhere in this suite is satisfied by an
    implementation that reads the object into a fresh buffer. This one is not.
    """

    body = b"\x00" * 64
    path = _object(tmp_path, body)
    view = memoryview(MappedObject(path))
    assert bytes(view) == body

    with path.open("r+b") as handle:
        handle.seek(8)
        handle.write(b"\xff" * 8)
        handle.flush()

    assert bytes(view[8:16]) == b"\xff" * 8, "the view is a copy, not the mapping"


# ---------------------------------------------------------------------------
# The read path rides on it
# ---------------------------------------------------------------------------


def _snapshot(cas: LocalCAS) -> tuple[RepositoryManifest, bytes]:
    body = random.Random("tensor").randbytes(4096)
    writer = TensorWriter(cas, "model.safetensors")
    writer.add("block.0.weight", "F32", (1024,), body)
    return RepositoryManifest((writer.finish(),)), body


def test_a_tensor_piece_is_exported_by_the_extension(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    manifest, body = _snapshot(cas)

    with open_tensors(cas, manifest) as tensors:
        piece = next(tensors["block.0.weight"].pieces())
        assert isinstance(piece.obj, MappedObject), type(piece.obj)
        assert piece.readonly
        assert bytes(piece) == body


def test_a_tensor_read_never_copies_the_object(tmp_path: Path) -> None:
    """The end-to-end form of :func:`test_a_view_tracks_the_file_it_maps`.

    ``verify=False`` because the edit below deliberately breaks the object's
    digest -- the point is where the bytes live, not whether they are still
    the ones the manifest names.
    """

    cas = LocalCAS(tmp_path / "cas")
    manifest, body = _snapshot(cas)
    entry = manifest.files[0]
    # The tensor's own object: the header is object zero, the tensor object
    # one, because the writer emits the seal planner's per-tensor grid.
    tensor_object = cas.object_path(entry.chunks[1].digest)
    assert tensor_object.stat().st_size == len(body)

    with open_tensors(cas, manifest, verify=False) as tensors:
        piece = next(tensors["block.0.weight"].pieces())
        assert bytes(piece) == body

        with tensor_object.open("r+b") as handle:
            handle.seek(0)
            handle.write(struct.pack("<I", 0xDEADBEEF))
            handle.flush()

        assert bytes(piece[:4]) == struct.pack("<I", 0xDEADBEEF), (
            "the read path copied the object instead of mapping it"
        )
