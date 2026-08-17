"""Projecting a snapshot tree from pure Python.

The claims worth testing are the ones a consumer's cutover rests on, and they
are tested against the readers that actually meet the tree:

* **it works with no compiled extension** -- pgw vendors this package as
  source into a pure-Python wheel, so a projection that needs `_tensorfs` is
  a projection pgw cannot use;
* **the stub bytes are the Rust bytes** -- against the committed corpus AND
  against `native.stub_bytes` / `native.parse_stub` side by side;
* **discovery works and a naive open fails loudly** -- `rglob` finds the real
  filename, and the real `safetensors` / `gguf` readers refuse a stub at the
  *parse* site rather than as the `FileNotFoundError` absence would give;
* **no bytes are copied** -- a config read through the tree is a plain-file
  open of the shared blob, and the tree itself is O(files), not O(model).
"""

from __future__ import annotations

import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import gguf
import pytest
import safetensors
from repo_ingest import ingest_file, ingest_with_grid
from tensorfs import (
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    is_tensor_container,
    open_tensors,
    parse_stub,
    project_snapshot,
    read_stub,
    stub_bytes,
    tree_bytes,
)

CORPUS = Path(__file__).resolve().parents[2] / "spec" / "v1" / "tfsstub1-vectors"

_TENSORS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("block.0.weight", "F32", (512, 512)),
    ("block.1.weight", "BF16", (256, 256)),
)
_DTYPE_BYTES = {"F32": 4, "BF16": 2}
_CONFIG = b'{"model_type":"llama","hidden_size":4096}\n'
_TOKENIZER = b'{"version":"1.0","model":{"type":"BPE"}}\n'


def _build_safetensors(path: Path) -> dict[str, bytes]:
    bodies: dict[str, bytes] = {}
    header: dict[str, object] = {}
    cursor = 0
    for name, dtype, shape in _TENSORS:
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * _DTYPE_BYTES[dtype]
        bodies[name] = random.Random(name).randbytes(size)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        for name, _dtype, _shape in _TENSORS:
            handle.write(bodies[name])
    return bodies


@pytest.fixture
def repo(tmp_path: Path) -> tuple[LocalCAS, RepositoryManifest, dict[str, bytes]]:
    """A multi-component model: two configs, one safetensors, one GGUF."""

    source = tmp_path / "source"
    (source / "text_encoder").mkdir(parents=True)
    (source / "transformer").mkdir()
    (source / "model_index.json").write_bytes(_CONFIG)
    (source / "text_encoder" / "tokenizer.json").write_bytes(_TOKENIZER)
    bodies = _build_safetensors(source / "transformer" / "diffusion.safetensors")
    # A valid, empty GGUF v3: discovery and stub projection read no tensor
    # bytes, so the smallest structurally real file is the honest fixture.
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    (source / "transformer" / "qwen3-Q4_K_M.gguf").write_bytes(
        header + b"\0" * (-len(header) % 32)
    )

    cas = LocalCAS(tmp_path / "cas")
    entries = [
        ingest_file(cas, path, manifest_path=path.relative_to(source).as_posix())
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    return cas, RepositoryManifest(tuple(entries)), bodies


# -- TFSSTUB1, byte for byte ---------------------------------------------


def test_golden_corpus_renders_and_parses_in_pure_python() -> None:
    vectors = json.loads((CORPUS / "tfsstub1-vectors.json").read_text())
    assert vectors["magic"] == "TFSSTUB1"
    assert vectors["golden"], "the corpus must not be empty"
    for case in vectors["golden"]:
        expected = (CORPUS / case["fixture"]).read_bytes()
        rendered = stub_bytes(case["body_sha256"], case["size"])
        assert rendered == expected, case["name"]
        parsed = parse_stub(expected)
        assert parsed is not None, case["name"]
        assert parsed.body_sha256 == case["body_sha256"]
        assert parsed.size == case["size"]


def test_rust_and_python_stubs_are_the_same_bytes() -> None:
    """The two halves of one distribution must not drift apart."""

    native = pytest.importorskip("tensorfs.native")
    digest = "b" * 64
    for size in (0, 1, 4096, (1 << 40) + 7):
        assert stub_bytes(digest, size) == bytes(native.stub_bytes(digest, size))
    # Each side parses the other's bytes, so a drift in either direction fails.
    theirs = native.parse_stub(stub_bytes(digest, 4096))
    mine = parse_stub(bytes(native.stub_bytes(digest, 4096)))
    assert theirs is not None and mine is not None
    assert (mine.body_sha256, mine.size) == tuple(theirs) == (digest, 4096)


@pytest.mark.parametrize(
    "corrupt",
    [
        b"TFSSTUB2 " + b'{"body_sha256":"' + b"a" * 64 + b'","size":4,"read":"tensorfs"}\n',
        b"TFSSTUB1" + b'{"body_sha256":"' + b"a" * 64 + b'","size":4,"read":"tensorfs"}\n',
        b"TFSSTUB1 " + b'{"body_sha256":"' + b"a" * 64 + b'","size":4,"read":"tensorfs"}',
        b"TFSSTUB1 " + b'{"body_sha256":"' + b"A" * 64 + b'","size":4,"read":"tensorfs"}\n',
        b"TFSSTUB1 " + b'{"body_sha256":"' + b"a" * 64 + b'","size":4,"read":"fuse"}\n',
        b"TFSSTUB1 " + b'{"body_sha256":"' + b"a" * 64 + b'","size":"4","read":"tensorfs"}\n',
        b"TFSSTUB1 " + b'{"body_sha256":"' + b"a" * 64 + b'","size":4}\n',
        b'{"body_sha256":"' + b"a" * 64 + b'"}\n',
        b"",
    ],
)
def test_a_near_miss_is_not_a_stub(corrupt: bytes) -> None:
    assert parse_stub(corrupt) is None


def test_projection_imports_without_the_compiled_extension() -> None:
    """pgw's wheel has no `_tensorfs`. Prove the module never reaches for one."""

    program = """
import sys

class Refuse:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.startswith("tensorfs._tensorfs") or name == "tensorfs.native":
            raise AssertionError("the pure-Python projection reached for the extension: " + name)
        return None

sys.meta_path.insert(0, Refuse())
from tensorfs import project_snapshot, parse_stub, stub_bytes  # noqa: F401
from tensorfs.project import project_snapshot as _p  # noqa: F401
assert parse_stub(stub_bytes("c" * 64, 7)).size == 7
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# -- the tree ------------------------------------------------------------


def test_projected_tree_is_symlinks_and_stubs(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")

    assert tree.is_dir()
    projected = {
        path.relative_to(tree).as_posix() for path in tree.rglob("*") if not path.is_dir()
    }
    assert projected == {entry.path for entry in manifest.files}

    for entry in manifest.files:
        member = tree / entry.path
        if is_tensor_container(entry.path):
            assert not member.is_symlink(), f"{entry.path} must be a stub, not a link"
            stub = read_stub(member)
            assert stub is not None, f"{entry.path} is not a TFSSTUB1"
            assert stub.size == entry.size_bytes
            assert stub.body_sha256 == entry.digest.digest
            assert member.stat().st_mode & 0o777 == 0o444
        else:
            assert member.is_symlink(), f"{entry.path} must be a symlink"
            target = os.readlink(member)
            assert not os.path.isabs(target), "a tree must be relocatable"
            assert member.resolve() == cas.object_path(entry.digest).resolve()


def test_a_config_read_through_the_tree_is_the_original_bytes(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """The non-weight half of `from_pretrained(dir)`: a plain-file open."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")
    assert (tree / "model_index.json").read_bytes() == _CONFIG
    assert (tree / "text_encoder" / "tokenizer.json").read_bytes() == _TOKENIZER
    assert json.loads((tree / "model_index.json").read_text())["model_type"] == "llama"


def test_the_tree_costs_inodes_not_bytes(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """A resident model occupies disk ONCE: in the CAS. The tree is metadata."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")

    model_bytes = sum(entry.size_bytes for entry in manifest.files)
    tensor_bytes = sum(
        entry.size_bytes for entry in manifest.files if is_tensor_container(entry.path)
    )
    assert tensor_bytes > 0
    # Stubs only. Every real byte the tree "contains" lives in the CAS.
    assert tree_bytes(tree) < 1024
    assert tree_bytes(tree) < model_bytes / 100

    du = subprocess.run(
        ["du", "-sb", "--", str(tree)], capture_output=True, text=True, check=True
    )
    assert int(du.stdout.split()[0]) < model_bytes / 10


def test_discovery_finds_real_filenames_over_stubs(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """pgw's GGUF discovery is `is_dir()` + `rglob("*.gguf")`, unchanged."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")
    assert tree.is_dir()
    found = sorted(path.name for path in tree.rglob("*.gguf"))
    assert found == ["qwen3-Q4_K_M.gguf"]
    assert sorted(path.name for path in tree.rglob("*.safetensors")) == ["diffusion.safetensors"]


def test_a_naive_open_fails_at_the_parse_site(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """Not FileNotFoundError -- absence reads as a corrupt snapshot."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")

    with pytest.raises(Exception) as safetensors_error:
        safetensors.safe_open(str(tree / "transformer" / "diffusion.safetensors"), framework="pt")
    assert not isinstance(safetensors_error.value, FileNotFoundError)

    with pytest.raises(Exception) as gguf_error:
        gguf.GGUFReader(str(tree / "transformer" / "qwen3-Q4_K_M.gguf"))
    assert not isinstance(gguf_error.value, FileNotFoundError)


def test_weights_load_natively_from_the_same_manifest(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """The other half of the cutover: the stub's path is served by the reader."""

    cas, manifest, bodies = repo
    project_snapshot(cas, manifest, tmp_path / "snapshots" / "abc")
    with open_tensors(cas, manifest) as reader:
        for name, expected in bodies.items():
            assert reader[name].tobytes() == expected


# -- edges ---------------------------------------------------------------


def test_projection_is_idempotent_and_leaves_no_scratch(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    cas, manifest, _bodies = repo
    root = tmp_path / "snapshots"
    first = project_snapshot(cas, manifest, root / "abc")
    second = project_snapshot(cas, manifest, root / "abc")
    assert first == second
    assert [entry.name for entry in root.iterdir()] == ["abc"]


def test_a_failed_projection_publishes_nothing(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """A missing object must not leave a half-built tree under the final name."""

    cas, manifest, _bodies = repo
    victim = next(entry for entry in manifest.files if not is_tensor_container(entry.path))
    cas.object_path(victim.digest).unlink()
    root = tmp_path / "snapshots"
    # The symlink itself does not need the target, so force the copy path,
    # which does: that is the arm where a missing object is an error.
    with pytest.raises(OSError):
        project_snapshot(cas, manifest, root / "abc", symlinks=False)
    assert not (root / "abc").exists()
    assert list(root.iterdir()) == []


def test_a_chunked_non_tensor_file_is_reassembled_exactly(tmp_path: Path) -> None:
    """The one arm that writes real bytes, and it must be byte-exact."""

    cas = LocalCAS(tmp_path / "cas")
    payload = random.Random("chunked").randbytes(3000)
    source = tmp_path / "vocab.txt"
    source.write_bytes(payload)
    entry = ingest_with_grid(cas, source, [1000, 1000, 1000], manifest_path="vocab.txt")
    assert len(entry.chunks) == 3

    tree = project_snapshot(cas, RepositoryManifest((entry,)), tmp_path / "tree")
    member = tree / "vocab.txt"
    assert not member.is_symlink()
    assert member.read_bytes() == payload


def test_an_empty_file_projects_as_an_empty_file(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    empty = cas.put_bytes(b"")
    manifest = RepositoryManifest((FileEntry("__init__.py", 0, empty),))
    tree = project_snapshot(cas, manifest, tmp_path / "tree")
    assert (tree / "__init__.py").read_bytes() == b""
    assert not (tree / "__init__.py").is_symlink()


def test_a_one_chunk_entry_still_symlinks(tmp_path: Path) -> None:
    """One chunk IS the whole blob -- `FileEntry` refuses any other reading."""

    cas = LocalCAS(tmp_path / "cas")
    payload = b"{}\n"
    ref = cas.put_bytes(payload)
    manifest = RepositoryManifest(
        (FileEntry("config.json", len(payload), ref, (Chunk(ref, len(payload)),)),)
    )
    tree = project_snapshot(cas, manifest, tmp_path / "tree")
    assert (tree / "config.json").is_symlink()
    assert (tree / "config.json").read_bytes() == payload


def test_without_symlinks_the_tree_is_a_copy(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """Correctness is kept where symlinks are not available; dedup is lost."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "copies", symlinks=False)
    assert not (tree / "model_index.json").is_symlink()
    assert (tree / "model_index.json").read_bytes() == _CONFIG
    # Tensor containers are STILL stubs: a copy fallback is about inodes, not
    # about giving up on direct reads.
    assert read_stub(tree / "transformer" / "diffusion.safetensors") is not None


def test_a_deep_path_links_relatively_and_survives_a_move(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """A relative link computed from the wrong depth breaks only on a move."""

    cas, manifest, _bodies = repo
    original = tmp_path / "cas" / "snapshots" / "abc"
    tree = project_snapshot(cas, manifest, original)
    deep = tree / "text_encoder" / "tokenizer.json"
    assert deep.read_bytes() == _TOKENIZER

    moved = tmp_path / "cas" / "snapshots" / "moved"
    os.rename(tree, moved)
    assert (moved / "text_encoder" / "tokenizer.json").read_bytes() == _TOKENIZER


def test_a_projected_stub_is_not_mistaken_for_the_file(
    repo: tuple[LocalCAS, RepositoryManifest, dict[str, bytes]], tmp_path: Path
) -> None:
    """`stat` sees ~128 B; the stub carries the truth `stat` cannot tell."""

    cas, manifest, _bodies = repo
    tree = project_snapshot(cas, manifest, tmp_path / "tree")
    entry = next(entry for entry in manifest.files if entry.path.endswith(".safetensors"))
    member = tree / entry.path
    assert member.stat().st_size < 200
    stub = read_stub(member)
    assert stub is not None and stub.size == entry.size_bytes > 200
