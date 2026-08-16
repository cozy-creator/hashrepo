"""GGUF tensors read direct from CAS objects, byte-exact and without a file.

Same discipline as the safetensors arm: pseudorandom per-tensor payloads so a
shifted read cannot compare equal, and one tensor above 64 MiB so the
multi-object path is exercised rather than merely available.

Note what this fixture does NOT prove. The Python chunker has no GGUF planner,
so these files commit under the bounded fixed grid and their tensors are
arbitrary slices of 64 MiB objects. That is the harder case for the reader and
it is the one asserted here. Tensor-aligned GGUF objects come from the Rust
seal planner (``crates/tensorfs-core/src/planner/gguf.rs``), which is not
exercised by this test.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest
from tensorfs import LocalCAS, TensorError, open_tensors
from tensorfs.gguf import GGML_LAYOUT, GGUFError
from tensorfs.manifest import MAX_CHUNK_SIZE

# Deliberately NOT the GGUF default of 32. A fixture aligned to the default
# cannot tell whether the reader honoured general.alignment or just guessed it.
ALIGNMENT = 64

# (name, ggml type id, shape). Q4_K is 256 elements per 144-byte block.
_TENSORS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("blk.0.attn.weight", 0, (27_262_976,)),  # F32, 104 MiB: spans objects
    ("blk.0.ffn.weight", 12, (512, 4)),  # Q4_K, quantized
    ("blk.0.norm.weight", 1, (1024,)),  # F16
    ("output.weight", 30, (2048, 2)),  # BF16
)


def _nbytes(ggml_type: int, shape: tuple[int, ...]) -> int:
    elements, block_bytes = GGML_LAYOUT[ggml_type]
    total = (shape[0] // elements) * block_bytes
    for dimension in shape[1:]:
        total *= dimension
    return total


def _string(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _build_gguf(path: Path) -> dict[str, bytes]:
    """Write a structurally valid GGUF v3 file and return each tensor's bytes."""

    metadata = b"".join(
        (
            _string(b"general.alignment") + struct.pack("<II", 4, ALIGNMENT),
            _string(b"general.architecture") + struct.pack("<I", 8) + _string(b"fixture"),
            _string(b"general.block_count") + struct.pack("<II", 4, 1),
            # An array of fixed-width elements, which the parser must skip
            # wholesale rather than walk.
            _string(b"fixture.list")
            + struct.pack("<I", 9)
            + struct.pack("<IQ", 4, 3)
            + struct.pack("<III", 7, 8, 9),
        )
    )

    bodies: dict[str, bytes] = {}
    infos = b""
    cursor = 0
    for name, ggml_type, shape in _TENSORS:
        size = _nbytes(ggml_type, shape)
        bodies[name] = random.Random(name).randbytes(size)
        infos += (
            _string(name.encode())
            + struct.pack("<I", len(shape))
            + b"".join(struct.pack("<Q", dimension) for dimension in shape)
            + struct.pack("<IQ", ggml_type, cursor)
        )
        cursor += size
        cursor += -cursor % ALIGNMENT  # every tensor starts aligned

    # Pad the directory until rounding it up to 32 and to 64 give DIFFERENT
    # answers, so a reader that ignores general.alignment lands on the wrong
    # data offset and every tensor comes back shifted.
    filler = b""
    while True:
        tail = _string(b"fixture.pad") + struct.pack("<I", 8) + _string(filler)
        header = b"GGUF" + struct.pack("<IQQ", 3, len(_TENSORS), 5) + metadata + tail + infos
        if 0 < len(header) % 64 < 32:
            break
        filler += b"."
    assert -len(header) % 32 != -len(header) % 64, "the fixture stopped discriminating"
    padding = -len(header) % ALIGNMENT

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(b"\0" * padding)
        written = 0
        for name, _type, _shape in _TENSORS:
            payload = bodies[name]
            handle.write(payload)
            written += len(payload)
            gap = -written % ALIGNMENT
            handle.write(b"\0" * gap)
            written += gap
    return bodies


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("gguf-direct-reads")
    source = root / "source"
    source.mkdir()
    bodies = _build_gguf(source / "model.gguf")
    cas = LocalCAS(root / "cas")
    manifest = cas.ingest_repository(source)
    return {
        "cas_root": root / "cas",
        "manifest": manifest,
        "ref": cas.store_manifest(manifest),
        "bodies": bodies,
    }


def test_the_gguf_fixture_spans_more_than_one_object(fixture: dict[str, object]) -> None:
    """Guard the guard: without a multi-object tensor the next test is weaker."""

    manifest = fixture["manifest"]
    entry = next(item for item in manifest.files if item.path == "model.gguf")  # type: ignore[attr-defined]
    lengths = [chunk.length for chunk in entry.chunks]
    assert len(lengths) > 1, lengths
    assert lengths[0] == MAX_CHUNK_SIZE, lengths


def test_every_gguf_tensor_is_byte_exact(fixture: dict[str, object]) -> None:
    cas_root = Path(str(fixture["cas_root"]))
    cas = LocalCAS(cas_root)
    bodies: dict[str, bytes] = fixture["bodies"]  # type: ignore[assignment]

    before = {path: path.stat().st_size for path in sorted(cas_root.rglob("*"))}
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        assert set(tensors) == set(bodies)
        for name, expected in bodies.items():
            view = tensors[name]
            assert view.nbytes == len(expected), name
            assert view.tobytes() == expected, name
    assert before == {path: path.stat().st_size for path in sorted(cas_root.rglob("*"))}


def test_a_gguf_tensor_above_the_object_ceiling_reassembles(
    fixture: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(fixture["cas_root"])))
    expected: bytes = fixture["bodies"]["blk.0.attn.weight"]  # type: ignore[index]
    assert len(expected) > MAX_CHUNK_SIZE

    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        view = tensors["blk.0.attn.weight"]
        assert len(list(view.pieces())) > 1
        assert view.tobytes() == expected


def test_quantization_block_geometry_reaches_the_caller(
    fixture: dict[str, object],
) -> None:
    """The one thing safetensors has no equivalent for, and GGUF consumers need."""

    cas = LocalCAS(Path(str(fixture["cas_root"])))
    with open_tensors(cas, fixture["ref"]) as tensors:  # type: ignore[arg-type]
        quantized = tensors["blk.0.ffn.weight"]
        assert quantized.format == "gguf-v1"
        assert quantized.dtype == "Q4_K"
        assert quantized.shape == (512, 4)
        assert (quantized.block.elements, quantized.block.nbytes) == (256, 144)
        assert quantized.block.quantized

        plain = tensors["blk.0.norm.weight"]
        assert plain.dtype == "F16"
        assert (plain.block.elements, plain.block.nbytes) == (1, 2)
        assert not plain.block.quantized

        assert tensors["output.weight"].dtype == "BF16"


def test_a_gguf_tensor_using_a_removed_ggml_type_is_refused(tmp_path: Path) -> None:
    """ggml ids 4 and 5 were removed upstream; guessing their size would corrupt."""

    source = tmp_path / "source"
    source.mkdir()
    infos = _string(b"bad.weight") + struct.pack("<I", 1) + struct.pack("<QIQ", 32, 4, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + infos
    (source / "model.gguf").write_bytes(header + b"\0" * (64 - len(header) % 64))

    cas = LocalCAS(tmp_path / "cas")
    ref = cas.store_manifest(cas.ingest_repository(source))
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(TensorError, match="unsupported ggml type 4"):
            list(tensors)


def test_a_tensor_declared_past_the_end_of_the_file_is_refused(tmp_path: Path) -> None:
    """A bad offset must be refused, not turned into a short or wild read."""

    source = tmp_path / "source"
    source.mkdir()
    # One F32 tensor of 32 elements (128 bytes) declared at offset 4096, in a
    # file that stops well before that.
    infos = _string(b"far.weight") + struct.pack("<I", 1) + struct.pack("<QIQ", 32, 0, 4096)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + infos
    (source / "model.gguf").write_bytes(header + b"\0" * (-len(header) % 32) + b"\0" * 128)

    cas = LocalCAS(tmp_path / "cas")
    ref = cas.store_manifest(cas.ingest_repository(source))
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(TensorError, match="past the end"):
            list(tensors)


def test_a_truncated_gguf_directory_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    # Claims one tensor but supplies no tensor-info record at all.
    (source / "model.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + b"\0" * 4)

    cas = LocalCAS(tmp_path / "cas")
    ref = cas.store_manifest(cas.ingest_repository(source))
    with open_tensors(cas, ref) as tensors:
        with pytest.raises(TensorError):
            list(tensors)


def test_the_ggml_table_matches_the_rust_planner() -> None:
    """Both implementations must accept and reject exactly the same files."""

    rust = Path("crates/tensorfs-core/src/planner/gguf.rs").read_text()
    section = rust.split("fn ggml_layout")[1].split("#[cfg(test)]")[0]
    pinned: dict[int, tuple[int, int]] = {}
    for line in section.splitlines():
        stripped = line.strip()
        if "=> (" not in stripped:
            continue
        key, _, rest = stripped.partition(" => (")
        if not key.strip().isdigit():
            continue
        elements, block_bytes = rest.split(")")[0].split(",")
        pinned[int(key)] = (int(elements), int(block_bytes))

    assert pinned, "failed to parse the Rust table; this test would prove nothing"
    assert pinned == GGML_LAYOUT


def test_gguf_error_is_reported_as_a_tensor_error(tmp_path: Path) -> None:
    assert issubclass(GGUFError, ValueError)
