from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from repo_ingest import ingest_file, ingest_repository
from tensorfs import (
    MAX_CHUNK_SIZE,
    FileEntry,
    LocalCAS,
    RefConflict,
    RepositoryManifest,
    read_entry,
)


def test_local_cas_survives_restart_and_rereads_committed_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"local-only bytes")
    root = tmp_path / "cas"

    first = LocalCAS(root)
    entry = ingest_file(first, source, manifest_path="artifacts/graph.bin")
    manifest = RepositoryManifest((entry,))
    manifest_ref = first.store_manifest(manifest)
    first.compare_and_swap_ref("compiled/cg-key-v1-example", manifest_ref, expected=None)

    second = LocalCAS(root)
    assert second.read_ref("compiled/cg-key-v1-example") == manifest_ref
    loaded = second.load_manifest(manifest_ref)
    assert read_entry(second, loaded.files[0]) == source.read_bytes()


def test_fresh_process_reuses_local_cas_without_loading_network_stack(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    create = """
from tensorfs import LocalCAS
from pathlib import Path
cas = LocalCAS(Path(__import__('sys').argv[1]))
ref = cas.put_bytes(b'fresh-process')
cas.compare_and_swap_ref('graph', ref, expected=None)
"""
    reuse = """
import sys
from tensorfs import LocalCAS
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


def test_resident_put_bytes_never_creates_a_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"already durable"
    expected = cas.put_bytes(payload)

    def reject_temporary(*args: object, **kwargs: object) -> tuple[int, str]:
        pytest.fail("verified resident object must return before creating a temporary file")

    monkeypatch.setattr(tempfile, "mkstemp", reject_temporary)
    assert cas.put_bytes(payload) == expected
    assert cas.verify_object(expected, size=len(payload)).read_bytes() == payload


def test_same_object_is_idempotent_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    script = """
from tensorfs import LocalCAS
from pathlib import Path
import sys
cas = LocalCAS(Path(sys.argv[1]))
print(cas.put_bytes(b'multi-process' * 4096))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(root)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    refs = [process.communicate(timeout=30)[0].strip() for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert len(set(refs)) == 1
    assert LocalCAS(root).verify_object(refs[0]).read_bytes() == b"multi-process" * 4096


def test_killed_writer_cannot_damage_a_committed_ref(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    cas = LocalCAS(root)
    admitted = cas.put_bytes(b"admitted")
    cas.compare_and_swap_ref("graph", admitted, expected=None)
    replacement = LocalCAS(root).put_bytes(b"replacement")
    LocalCAS(root).object_path(replacement).unlink()
    script = """
from tensorfs import CASRef, LocalCAS
from pathlib import Path
import os, sys
cas = LocalCAS(Path(sys.argv[1]))
with cas.open_writer(CASRef.parse(sys.argv[2]), size=len(b'replacement')) as writer:
    writer.write(b'replace')
    writer.flush()
    os.fsync(writer.fileno())
    os._exit(91)
"""
    killed = subprocess.run([sys.executable, "-c", script, str(root), str(replacement)])
    assert killed.returncode == 91

    restarted = LocalCAS(root)
    assert restarted.read_ref("graph") == admitted
    assert restarted.verify_object(admitted).read_bytes() == b"admitted"
    assert not restarted.object_path(replacement).exists()


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


def test_reading_refuses_a_corrupt_resident_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    ref = cas.put_bytes(b"good")
    cas.object_path(ref).write_bytes(b"evil")
    entry = FileEntry("file", 4, ref)
    with pytest.raises(ValueError, match="object bytes do not match"):
        read_entry(cas, entry)


def test_put_file_atomically_repairs_only_a_corrupt_digest_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    source = tmp_path / "source"
    source.write_bytes(b"good")
    ref = cas.put_file(source)
    cas.object_path(ref).write_bytes(b"evil")
    cas.put_file(source, expected=ref, size=4)
    assert cas.verify_object(ref, size=4).read_bytes() == b"good"


def test_put_bytes_atomically_repairs_only_a_corrupt_digest_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    ref = cas.put_bytes(b"good")
    cas.object_path(ref).write_bytes(b"evil")
    assert cas.put_bytes(b"good", expected=ref) == ref
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


def test_adopt_file_refuses_a_pathname_swap_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    payload = b"verified bytes"
    expected = cas.put_bytes(payload)
    cas.object_path(expected).unlink()
    descriptor, raw_path = tempfile.mkstemp(prefix="download-", dir=cas.tmp)
    temporary = Path(raw_path)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)

    real_link = os.link

    def swap_then_link(
        source: Path, destination: Path, *, follow_symlinks: bool = True
    ) -> None:
        replacement = source.with_name(f"{source.name}.replacement")
        replacement.write_bytes(b"attacker bytes")
        os.replace(replacement, source)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", swap_then_link)
    with pytest.raises(OSError, match="changed after verification"):
        cas.adopt_file(temporary, expected=expected, size=len(payload))

    assert not cas.object_path(expected).exists()
    assert not temporary.exists()


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


def test_a_non_tensor_file_above_the_grid_is_one_whole_blob(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    block = b"tensorfs" * 95325
    with source.open("wb") as handle:
        remaining = MAX_CHUNK_SIZE + 3
        while remaining:
            data = block[: min(len(block), remaining)]
            handle.write(data)
            remaining -= len(data)

    cas = LocalCAS(tmp_path / "cas")
    entry = ingest_file(cas, source)
    assert entry.size_bytes == MAX_CHUNK_SIZE + 3
    assert entry.chunks == (), "the raw grid is dead: one blob, no chunks"
    assert entry.objects() == ((entry.digest, entry.size_bytes),)

    assert read_entry(cas, entry) == source.read_bytes()


def test_repository_tree_round_trips_with_portable_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "config.json").write_bytes(b"{}")
    (source / "nested" / "weights.bin").write_bytes(b"weights")
    cas = LocalCAS(tmp_path / "cas")

    manifest = ingest_repository(cas, source)
    assert [entry.path for entry in manifest.files] == ["config.json", "nested/weights.bin"]
    manifest_ref = cas.store_manifest(manifest)
    cas.compare_and_swap_ref("snapshot", manifest_ref, expected=None)

    restarted = LocalCAS(tmp_path / "cas")
    reloaded = restarted.load_manifest(manifest_ref)
    contents = {entry.path: read_entry(restarted, entry) for entry in reloaded.files}
    assert contents == {"config.json": b"{}", "nested/weights.bin": b"weights"}




_NO_FCNTL_PROBE = """
import sys

class RefuseFcntl:
    def find_spec(self, name, path=None, target=None):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
        return None

sys.modules.pop("fcntl", None)
sys.meta_path.insert(0, RefuseFcntl())
try:
    import tensorfs
except ImportError as exc:
    print(f"{type(exc).__name__}: {exc}")
else:
    print("IMPORTED")
"""


def test_an_interpreter_without_fcntl_is_refused_by_name() -> None:
    """The platform boundary announces itself (tensorfs#57).

    There is no Windows wheel because `LocalCAS` locks with `fcntl`. A user
    who installs the sdist there gets an import failure either way; what this
    fixes is *which* failure. A bare `No module named 'fcntl'` names a missing
    stdlib module and invites the reader to go install one. The support
    boundary is the actual answer, so it is what gets raised.

    Simulated rather than skipped: refusing `fcntl` through `sys.meta_path`
    reproduces the Windows import exactly, and runs on the platforms CI has.
    """

    result = subprocess.run(
        [sys.executable, "-c", _NO_FCNTL_PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=tempfile.gettempdir(),
    )
    message = result.stdout.strip()
    assert message.startswith("ImportError: "), message
    assert "Linux and macOS only" in message, message
    assert "fcntl" in message, message
