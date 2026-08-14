from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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


def test_fresh_process_reuses_local_cas_without_loading_network_stack(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    create = """
from chunked_cas import LocalCAS
from pathlib import Path
cas = LocalCAS(Path(__import__('sys').argv[1]))
ref = cas.put_bytes(b'fresh-process')
cas.compare_and_swap_ref('graph', ref, expected=None)
"""
    reuse = """
import sys
from chunked_cas import LocalCAS
assert 'urllib.request' not in sys.modules
cas = LocalCAS(__import__('pathlib').Path(sys.argv[1]))
ref = cas.read_ref('graph')
assert ref is not None
assert cas.verify_object(ref).read_bytes() == b'fresh-process'
assert 'urllib.request' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", create, str(root)], check=True)
    subprocess.run([sys.executable, "-c", reuse, str(root)], check=True)


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


def test_put_file_atomically_repairs_only_a_corrupt_digest_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    source = tmp_path / "source"
    source.write_bytes(b"good")
    ref = cas.put_file(source)
    cas.object_path(ref).write_bytes(b"evil")
    cas.put_file(source, expected=ref, size=4)
    assert cas.verify_object(ref, size=4).read_bytes() == b"good"


def test_stream_writer_hashes_and_installs_without_exposing_partial_bytes(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"streamed object"
    expected = cas.put_bytes(payload)
    cas.object_path(expected).unlink()

    with cas.open_writer(expected, size=len(payload)) as writer:
        writer.write(payload[:7])
        assert not cas.object_path(expected).exists()
        writer.write(payload[7:])

    assert cas.verify_object(expected, size=len(payload)).read_bytes() == payload


@pytest.mark.parametrize("payload", [b"short", b"wrongly bytes!"])
def test_stream_writer_refuses_size_or_digest_mismatch(tmp_path: Path, payload: bytes) -> None:
    cas = LocalCAS(tmp_path / "cas")
    expected = cas.put_bytes(b"expected bytes")
    cas.object_path(expected).unlink()

    with pytest.raises(ValueError):
        with cas.open_writer(expected, size=len(b"expected bytes")) as writer:
            writer.write(payload)

    assert not cas.object_path(expected).exists()
    assert list(cas.tmp.iterdir()) == []


def test_adopt_file_installs_the_same_inode_and_repairs_corruption(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"adopt this exact file"
    expected = cas.put_bytes(payload)
    cas.object_path(expected).write_bytes(b"corrupt resident")
    descriptor, raw_path = tempfile.mkstemp(prefix="download-", dir=cas.tmp)
    temporary = Path(raw_path)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
    inode = temporary.stat().st_ino

    assert cas.adopt_file(temporary, expected=expected, size=len(payload)) == expected
    assert not temporary.exists()
    resident = cas.verify_object(expected, size=len(payload))
    assert resident.stat().st_ino == inode
    assert resident.read_bytes() == payload


def test_adopt_file_consumes_mismatch_without_touching_resident_bytes(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    expected = cas.put_bytes(b"resident")
    descriptor, raw_path = tempfile.mkstemp(prefix="download-", dir=cas.tmp)
    temporary = Path(raw_path)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(b"imposter")

    with pytest.raises(ValueError, match="bytes hash"):
        cas.adopt_file(temporary, expected=expected, size=8)

    assert not temporary.exists()
    assert cas.verify_object(expected, size=8).read_bytes() == b"resident"


def test_concurrent_adoptions_are_idempotent_and_consume_every_temp(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"one object from many downloads" * 1024
    expected = cas.put_bytes(payload)
    cas.object_path(expected).unlink()
    temporaries: list[Path] = []
    for _ in range(8):
        descriptor, raw_path = tempfile.mkstemp(prefix="download-", dir=cas.tmp)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        temporaries.append(Path(raw_path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(
            pool.map(
                lambda path: cas.adopt_file(path, expected=expected, size=len(payload)),
                temporaries,
            )
        )

    assert refs == [expected] * len(temporaries)
    assert not any(path.exists() for path in temporaries)
    assert cas.verify_object(expected, size=len(payload)).read_bytes() == payload


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
