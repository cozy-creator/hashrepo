from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from chunked_cas import CHUNK_SIZE, FileEntry, LocalCAS, RefConflict, RepositoryManifest


def test_local_cas_survives_restart_and_materializes_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"local-only bytes")
    root = tmp_path / "cas"

    first = LocalCAS(root)
    entry = first.ingest_file(source, manifest_path="artifacts/graph.bin")
    manifest = RepositoryManifest((entry,))
    manifest_ref = first.store_manifest(manifest)
    first.compare_and_swap_ref("compiled/cg-key-v1-example", manifest_ref, expected=None)

    second = LocalCAS(root)
    assert second.read_ref("compiled/cg-key-v1-example") == manifest_ref
    loaded = second.load_manifest(manifest_ref)
    destination = tmp_path / "materialized" / "graph.bin"
    second.materialize(loaded.files[0], destination)
    assert destination.read_bytes() == source.read_bytes()


def test_same_object_is_idempotent_across_concurrent_writers(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"same immutable object" * 1024
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(cas.put_bytes, [payload] * 32))
    assert len(set(refs)) == 1
    assert cas.verify_object(refs[0]).read_bytes() == payload


def test_logical_ref_is_compare_and_swap_not_silent_overwrite(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    first = cas.put_bytes(b"first")
    second = cas.put_bytes(b"second")
    cas.compare_and_swap_ref("graph", first, expected=None)
    with pytest.raises(RefConflict):
        cas.compare_and_swap_ref("graph", second, expected=None)
    assert cas.read_ref("graph") == first
    cas.compare_and_swap_ref("graph", second, expected=first)
    assert cas.read_ref("graph") == second


def test_materialize_refuses_a_corrupt_resident_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    ref = cas.put_bytes(b"good")
    cas.object_path(ref).write_bytes(b"evil")
    entry = FileEntry("file", 4, ref)
    with pytest.raises(ValueError, match="object bytes do not match"):
        cas.materialize(entry, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_file_above_chunk_boundary_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    block = b"chunked-cas" * 95325
    with source.open("wb") as handle:
        remaining = CHUNK_SIZE + 3
        while remaining:
            data = block[: min(len(block), remaining)]
            handle.write(data)
            remaining -= len(data)

    cas = LocalCAS(tmp_path / "cas")
    entry = cas.ingest_file(source)
    assert entry.size_bytes == CHUNK_SIZE + 3
    assert [chunk.length for chunk in entry.chunks] == [CHUNK_SIZE, 3]

    destination = tmp_path / "rebuilt.bin"
    cas.materialize(entry, destination)
    assert destination.read_bytes() == source.read_bytes()
