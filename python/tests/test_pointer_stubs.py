"""Pointer stubs from the consumer's side.

A stub is the one projection artifact a foreign tool parses, so the claims
worth testing are the ones the design argued (``docs/mixed-cas-layout.md``
§4), tested against the readers that will actually meet it:

* **discovery works** -- ``Path.rglob`` finds ``*.safetensors`` / ``*.gguf``
  by real filename, which is exactly pgw's GGUF discovery;
* **a naive open fails loudly at the parse site** -- the real ``safetensors``
  library and the real ``gguf`` reader both refuse a stub as a *parse*
  failure, not as the ``FileNotFoundError`` that absence would give and that
  a caller misreads as a corrupt snapshot;
* **the bytes are a contract** -- the committed corpus in
  ``spec/v1/tfsstub1-vectors/`` parses here, in Python, byte for byte.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import gguf
import pytest
import safetensors
from tensorfs.native import STUB_MAGIC, ObjectStore, Snapshot, decode_snapshot, parse_stub
from tensorfs.native import stub_bytes as render_stub
from tfm1_encode import Blob, Directory, Entry, Tensor, encode

CORPUS = Path(__file__).resolve().parents[2] / "spec" / "v1" / "tfsstub1-vectors"

# Four files whose bytes all differ, so no assertion can pass by crossing two
# of them.
_CONFIG = b'{"model_type":"llama","hidden_size":4096}'
_SAFETENSORS_HEADER = b'\x28\x00\x00\x00\x00\x00\x00\x00{"w":{"dtype":"U8","shape":[4]}}'
_SAFETENSORS_TENSOR = bytes(range(64)) * 4
_GGUF_HEADER = b"GGUF\x03\x00\x00\x00" + bytes(range(32, 96))


def _fixture(tmp_path: Path) -> tuple[ObjectStore, Snapshot, Path]:
    """A projected tree holding one blob, one safetensors and one GGUF file."""
    store = ObjectStore(tmp_path / "store")
    config = store.put_bytes(_CONFIG)
    header = store.put_bytes(_SAFETENSORS_HEADER)
    tensor = store.put_bytes(_SAFETENSORS_TENSOR)
    gguf_header = store.put_bytes(_GGUF_HEADER)

    entries: list[Entry] = [
        Directory("weights"),
        Blob("config.json", config.digest, config.length),
        Tensor(
            "weights/model.safetensors",
            "safetensors-v1",
            ((header.digest, header.length), (tensor.digest, tensor.length)),
        ),
        Tensor(
            "weights/qwen3-Q4_K_M.gguf",
            "gguf-v1",
            ((gguf_header.digest, gguf_header.length),),
        ),
    ]
    # The hand-written encoder is validated by the real decoder before any
    # assertion rests on it.
    snapshot = decode_snapshot(encode(entries))
    return store, snapshot, store.project_snapshot(snapshot)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_rglob_discovery_finds_tensor_files_by_their_real_names(tmp_path: Path) -> None:
    """pgw's discovery is ``is_dir()`` + ``rglob``; stubs keep it working.

    The red arm is in the same test on purpose: with the stubs removed --
    which is exactly what "project tensor files as absence" would produce --
    the weights directory looks empty and every layout probe fails far from
    the cause.
    """
    _, _, tree = _fixture(tmp_path)

    assert (tree / "weights").is_dir()
    assert sorted(path.name for path in tree.rglob("*.safetensors")) == [
        "weights/model.safetensors".rsplit("/", 1)[1]
    ]
    assert sorted(path.name for path in tree.rglob("*.gguf")) == ["qwen3-Q4_K_M.gguf"]

    for path in list(tree.rglob("*.safetensors")) + list(tree.rglob("*.gguf")):
        path.chmod(0o644)
        path.unlink()

    assert list(tree.rglob("*.safetensors")) == []
    assert list(tree.rglob("*.gguf")) == []
    assert list((tree / "weights").iterdir()) == [], (
        "without stubs a weights directory is indistinguishable from an empty one"
    )


# ---------------------------------------------------------------------------
# Loud failure at the parse site
# ---------------------------------------------------------------------------


def test_the_real_safetensors_library_refuses_a_stub_at_its_parse_site(
    tmp_path: Path,
) -> None:
    """The actual library, not our parser.

    `TFSSTUB1` read as safetensors' leading little-endian u64 header length is
    an absurd number, so the refusal happens inside header deserialization.
    That is the whole argument for a stub over absence: a parse error the
    caller can act on, against a `FileNotFoundError` it will misread.
    """
    _, _, tree = _fixture(tmp_path)
    stub = tree / "weights" / "model.safetensors"

    with pytest.raises(safetensors.SafetensorError) as parsed:
        with safetensors.safe_open(stub, framework="numpy"):
            pass
    assert "header" in str(parsed.value).lower()

    # The contrast that makes it meaningful: absence is an OSError, and a
    # parse error is not.
    assert not isinstance(parsed.value, OSError)
    with pytest.raises(FileNotFoundError):
        with safetensors.safe_open(tree / "weights" / "absent.safetensors", framework="numpy"):
            pass


def test_the_real_gguf_reader_refuses_a_stub_at_its_parse_site(tmp_path: Path) -> None:
    _, _, tree = _fixture(tmp_path)
    stub = tree / "weights" / "qwen3-Q4_K_M.gguf"

    with pytest.raises(ValueError) as parsed:
        gguf.GGUFReader(stub)
    assert "magic" in str(parsed.value).lower()
    assert not isinstance(parsed.value, OSError)

    with pytest.raises(FileNotFoundError):
        gguf.GGUFReader(tree / "weights" / "absent.gguf")


def test_eight_bytes_classify_a_stub_without_parsing_it(tmp_path: Path) -> None:
    _, _, tree = _fixture(tmp_path)
    assert len(STUB_MAGIC) == 8
    for name in ("weights/model.safetensors", "weights/qwen3-Q4_K_M.gguf"):
        with (tree / name).open("rb") as handle:
            assert handle.read(8) == STUB_MAGIC
    # A real tensor file never collides with the magic.
    assert not _SAFETENSORS_HEADER.startswith(STUB_MAGIC)
    assert not _GGUF_HEADER.startswith(STUB_MAGIC)
    assert parse_stub(_GGUF_HEADER) is None


# ---------------------------------------------------------------------------
# The stub as a projection artifact
# ---------------------------------------------------------------------------


def test_a_projected_stub_is_immutable_and_named_exactly_like_the_file(
    tmp_path: Path,
) -> None:
    _, _, tree = _fixture(tmp_path)
    stub = tree / "weights" / "model.safetensors"
    assert stub.is_file() and not stub.is_symlink()
    assert stat.S_IMODE(stub.stat().st_mode) == 0o444
    with pytest.raises(PermissionError):
        stub.open("ab")


def test_a_stub_round_trips_against_the_manifest_entry_it_projects(
    tmp_path: Path,
) -> None:
    """Both fields come from the manifest, and both are load-bearing.

    Corrupting either one is visible, so the equality is not a tautology over
    a value the stub invented.
    """
    _, snapshot, tree = _fixture(tmp_path)
    entries = {entry.path: entry for entry in snapshot.entries}

    for path in ("weights/model.safetensors", "weights/qwen3-Q4_K_M.gguf"):
        entry = entries[path]
        assert entry.body_sha256 is not None and entry.logical_size is not None
        parsed = parse_stub((tree / path).read_bytes())
        assert parsed == (entry.body_sha256, entry.logical_size)

        # Red arms: change one field, lose the match.
        wrong_digest = "0" * 63 + "1"
        assert parse_stub(render_stub(wrong_digest, entry.logical_size)) != parsed
        assert parse_stub(render_stub(entry.body_sha256, entry.logical_size + 1)) != parsed

    # The two tensor entries have different bodies, so different stubs. A
    # projection that rendered one digest for both would pass every assertion
    # above.
    first, second = (
        (tree / "weights/model.safetensors").read_bytes(),
        (tree / "weights/qwen3-Q4_K_M.gguf").read_bytes(),
    )
    assert first != second


def test_the_stub_carries_the_logical_size_that_stat_cannot(tmp_path: Path) -> None:
    _, snapshot, tree = _fixture(tmp_path)
    entry = next(
        item for item in snapshot.entries if item.path == "weights/model.safetensors"
    )
    stub = tree / "weights" / "model.safetensors"
    parsed = parse_stub(stub.read_bytes())
    assert parsed is not None
    assert parsed[1] == entry.logical_size
    assert stub.stat().st_size != entry.logical_size, (
        "the honest cost: stat sees the stub, the size field sees the file"
    )


# ---------------------------------------------------------------------------
# The committed corpus, read by a Python consumer
# ---------------------------------------------------------------------------


def test_the_committed_stub_corpus_parses_byte_for_byte_in_python() -> None:
    index = json.loads((CORPUS / "tfsstub1-vectors.json").read_text(encoding="utf-8"))
    assert index["magic"].encode("ascii") == STUB_MAGIC
    assert len(index["golden"]) >= 4

    seen: list[bytes] = []
    for case in index["golden"]:
        committed = (CORPUS / case["fixture"]).read_bytes()
        assert committed == render_stub(case["body_sha256"], case["size"]), case["name"]
        assert parse_stub(committed) == (case["body_sha256"], case["size"]), case["name"]

        # A consumer parses it as plain JSON after the magic, with no
        # tensorfs code involved at all.
        magic, _, line = committed.partition(b" ")
        assert magic == STUB_MAGIC
        document = json.loads(line)
        assert document == {
            "body_sha256": case["body_sha256"],
            "size": case["size"],
            "read": "tensorfs",
        }
        assert isinstance(document["size"], int), "size must survive as an integer"

        assert committed not in seen, f"{case['name']} duplicates another fixture"
        seen.append(committed)
