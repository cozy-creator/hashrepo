"""Whole-blob entries: one chunkless object of ANY size (tensorfs#68).

The 64 MiB bound is the tensor chunk grid constant, not a blob cap. A
non-tensor file above 64 MiB is exactly one chunkless ``FileEntry`` whose
digest is its whole-file SHA-256, and the reader serves it — ``read_range``,
``read_file``-style access, and ``extract()`` — without any grid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tensorfs import (
    MAX_CHUNK_SIZE,
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    open_tensors,
)


def _blob_bytes() -> bytes:
    # One 64 KiB positional block repeated past the 64 MiB line, plus an
    # off-grid tail: nothing large is held per-block and the content is not
    # constant, so a shifted read cannot compare equal.
    block = bytes((index * 31 + 7) % 251 for index in range(64 * 1024))
    repeats = (MAX_CHUNK_SIZE // len(block)) + 3
    return block * repeats + b"tail-past-the-grid"


def test_a_chunkless_entry_above_64_mib_is_a_valid_blob() -> None:
    digest = CASRef.parse("sha256:" + hashlib.sha256(_blob_bytes()).hexdigest())
    entry = FileEntry("video.webm", len(_blob_bytes()), digest)
    assert entry.chunks == ()
    assert entry.objects() == ((entry.digest, entry.size_bytes),)

    manifest = RepositoryManifest((entry,))
    assert RepositoryManifest.from_bytes(manifest.canonical_bytes()) == manifest


def test_a_chunk_above_the_tensor_grid_still_refuses() -> None:
    digest = CASRef.parse("sha256:" + "ab" * 32)
    with pytest.raises(ValueError, match="chunk length"):
        FileEntry(
            "model.safetensors",
            MAX_CHUNK_SIZE + 1,
            digest,
            chunks=(Chunk(digest, MAX_CHUNK_SIZE + 1),),
        )


def test_the_reader_serves_and_extracts_a_blob_above_64_mib(tmp_path: Path) -> None:
    payload = _blob_bytes()
    assert len(payload) > MAX_CHUNK_SIZE

    cas = LocalCAS(tmp_path / "cas")
    ref = cas.put_bytes(payload)
    entry = FileEntry("clips/video.webm", len(payload), ref)
    manifest = RepositoryManifest((entry,))

    with open_tensors(cas, manifest) as reader:
        # Ranged reads touch only the one object and are byte-exact across
        # the old grid line, which no longer exists.
        edge = MAX_CHUNK_SIZE - 8
        assert reader.read_range("clips/video.webm", edge, 16) == payload[edge : edge + 16]
        assert reader.read_range("clips/video.webm", 0, 64) == payload[:64]

        # extract() streams the whole blob out and verifies it on the way.
        target = reader.extract("clips/video.webm", tmp_path / "out" / "video.webm")
        assert target.read_bytes() == payload


def test_extract_still_verifies_blob_bytes_against_the_manifest(tmp_path: Path) -> None:
    payload = b"small blob body"
    cas = LocalCAS(tmp_path / "cas")
    ref = cas.put_bytes(payload)
    entry = FileEntry("config.json", len(payload), ref)

    # Corrupt the resident object without changing its length; with per-object
    # verification off, extract()'s own whole-file check is the last fence.
    resident = cas.object_path(ref)
    resident.chmod(0o644)
    corrupted = b"X" + payload[1:]
    resident.write_bytes(corrupted)

    with open_tensors(cas, RepositoryManifest((entry,)), verify=False) as reader:
        with pytest.raises(ValueError, match="do not match"):
            reader.extract("config.json", tmp_path / "out.json")
