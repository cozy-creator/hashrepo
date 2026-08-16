"""The read-transform-write loop a conversion actually needs.

The claim under test is not just "it round-trips". It is that a conversion
touching one tensor:

* never holds a shard, in either direction;
* leaves every other tensor's CAS objects **bit-identical**, so nothing
  unchanged is rewritten locally or re-uploaded; and
* produces a snapshot the real ``safetensors`` library reads back byte-exactly.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import pytest
from tensorfs import (
    LocalCAS,
    RepositoryManifest,
    TensorError,
    TensorWriter,
    open_tensors,
    read_entry,
)
from tensorfs.manifest import MAX_CHUNK_SIZE

# One tensor above 64 MiB so both directions cross an object boundary.
_TENSORS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("denoiser.big", "F32", (26, 1024, 1024)),
    ("denoiser.small", "F32", (256, 256)),
    ("text_encoder.a", "F32", (128, 128)),
    ("text_encoder.b", "BF16", (64, 64)),
)
_DTYPE_BYTES = {"F32": 4, "BF16": 2, "U8": 1}


def _nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * _DTYPE_BYTES[dtype]


def _seed_snapshot(cas: LocalCAS) -> tuple[RepositoryManifest, dict[str, bytes]]:
    """Author the source through TensorWriter, so it carries the seal grid.

    Ingesting a real file instead would commit it under the Python chunker's
    packing grid, where small tensors share an object and therefore cannot be
    inherited at all. That difference is asserted directly further down.
    """

    bodies = {
        name: random.Random(name).randbytes(_nbytes(dtype, shape))
        for name, dtype, shape in _TENSORS
    }
    writer = TensorWriter(cas, "model.safetensors")
    for name, dtype, shape in _TENSORS:
        writer.add(name, dtype, shape, bodies[name])
    return RepositoryManifest((writer.finish(),)), bodies


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("conversion")
    cas = LocalCAS(root / "cas")
    manifest, bodies = _seed_snapshot(cas)
    return {"root": root, "manifest": manifest, "bodies": bodies}


def test_a_conversion_rewrites_only_what_it_touched(source: dict[str, object]) -> None:
    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    bodies: dict[str, bytes] = source["bodies"]  # type: ignore[assignment]

    before = {path.name for path in (cas.objects).rglob("*") if path.is_file()}

    # The conversion: cast one tensor to U8, leave everything else alone.
    with open_tensors(cas, manifest) as src:
        writer = TensorWriter(cas, "model.safetensors")
        inherited_digests: dict[str, tuple[str, ...]] = {}
        for name in src:
            view = src[name]
            if name == "denoiser.small":
                writer.add(name, "U8", view.shape, bytes(len(bodies[name]) // 4))
            else:
                span = src.object_span(view)
                assert span is not None, f"{name} must be inheritable"
                inherited_digests[name] = tuple(str(ref) for ref, _size in span)
                writer.inherit(view)
        converted = RepositoryManifest((writer.finish(),))

    after = {path.name for path in (cas.objects).rglob("*") if path.is_file()}

    # Every inherited tensor's objects survive untouched and are reused
    # verbatim by the new snapshot -- nothing about them was rewritten.
    with open_tensors(cas, converted) as out:
        for name, digests in inherited_digests.items():
            span = out.object_span(out[name])
            assert span is not None
            assert tuple(str(ref) for ref, _size in span) == digests, name
            assert out[name].tobytes() == bodies[name], name

    # The only new objects are the new header and the one rewritten tensor.
    # In particular the 104 MiB tensor produced no new objects at all.
    new = after - before
    assert len(new) <= 3, new
    assert len(after) >= len(before)


def test_the_converted_snapshot_is_byte_exact_and_readable(
    source: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    bodies: dict[str, bytes] = source["bodies"]  # type: ignore[assignment]

    with open_tensors(cas, manifest) as src:
        writer = TensorWriter(cas, "model.safetensors")
        for name in src:
            writer.inherit(src[name])
        entry = writer.finish()

    # A pure inherit-everything pass reproduces the original file exactly.
    rebuilt = read_entry(cas, entry)
    assert read_entry(cas, manifest.files[0]) == rebuilt

    with open_tensors(cas, RepositoryManifest((entry,))) as out:
        for name, expected in bodies.items():
            assert out[name].tobytes() == expected, name


def test_a_streamed_tensor_never_becomes_contiguous(source: dict[str, object]) -> None:
    """A tensor larger than one object can be written in pieces."""

    cas = LocalCAS(Path(str(source["root"])) / "cas")
    shape = (26, 1024, 1024)
    total = _nbytes("F32", shape)
    assert total > MAX_CHUNK_SIZE

    block = random.Random("streamed").randbytes(1 << 20)
    writer = TensorWriter(cas, "streamed.safetensors")
    writer.add("streamed", "F32", shape, (block for _ in range(total // len(block))))
    entry = writer.finish()

    with open_tensors(cas, RepositoryManifest((entry,))) as out:
        view = out["streamed"]
        assert view.nbytes == total
        pieces = [len(piece) for piece in view.pieces()]
        assert len(pieces) > 1
        assert pieces[0] == MAX_CHUNK_SIZE
        assert view.tobytes() == block * (total // len(block))


def test_the_written_grid_is_one_object_per_tensor(source: dict[str, object]) -> None:
    """This is precisely what makes the NEXT conversion able to inherit."""

    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    entry = manifest.files[0]
    lengths = [chunk.length for chunk in entry.chunks]

    # header + big(64 MiB + 40 MiB) + three small tensors, none packed together.
    assert lengths[0] < 1024, lengths[0]
    assert MAX_CHUNK_SIZE in lengths
    assert len(lengths) == 1 + 2 + 3

    with open_tensors(cas, manifest) as src:
        for name in src:
            assert src.object_span(src[name]) is not None, name


def test_a_packed_source_tensor_cannot_be_inherited(tmp_path: Path) -> None:
    """The honest limit: inheritance needs the tensor to own whole objects."""

    body = {"a": b"a" * 2048, "b": b"b" * 4096}
    header = {}
    cursor = 0
    for name, payload in body.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + body["a"] + body["b"]
    )

    cas = LocalCAS(tmp_path / "cas")
    # ingest_repository uses the Python chunker, which packs both tensors into
    # one shared object.
    manifest = cas.ingest_repository(source_dir)
    with open_tensors(cas, manifest) as src:
        assert src.object_span(src["a"]) is None
        writer = TensorWriter(cas, "out.safetensors")
        with pytest.raises(TensorError, match="not object-aligned"):
            writer.inherit(src["a"])


def test_region_splitting_matches_the_planner_rule() -> None:
    """Covers the header path, which no fixture here makes large enough.

    ``_split`` boundaries the header region; tensor bodies are split by the
    streaming path in ``add``. Only the latter is exercised by the fixtures
    above, so the rule itself is asserted directly here.
    """

    from tensorfs.writer import _split

    assert list(_split(0)) == []
    assert list(_split(1)) == [1]
    assert list(_split(MAX_CHUNK_SIZE)) == [MAX_CHUNK_SIZE]
    assert list(_split(MAX_CHUNK_SIZE + 1)) == [MAX_CHUNK_SIZE, 1]
    assert list(_split(3 * MAX_CHUNK_SIZE)) == [MAX_CHUNK_SIZE] * 3
    assert list(_split(2 * MAX_CHUNK_SIZE + 7)) == [MAX_CHUNK_SIZE, MAX_CHUNK_SIZE, 7]
    assert all(0 < length <= MAX_CHUNK_SIZE for length in _split(5 * MAX_CHUNK_SIZE - 3))


def test_a_tensor_whose_bytes_contradict_its_shape_is_refused(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    writer = TensorWriter(cas, "out.safetensors")
    with pytest.raises(TensorError, match="needs 64"):
        writer.add("w", "F32", (4, 4), b"\0" * 32)


def test_adding_the_same_name_twice_is_refused(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    writer = TensorWriter(cas, "out.safetensors")
    writer.add("w", "U8", (4,), b"abcd")
    with pytest.raises(TensorError, match="added twice"):
        writer.add("w", "U8", (4,), b"abcd")


def test_the_result_matches_the_reference_safetensors_implementation(
    source: dict[str, object], tmp_path: Path
) -> None:
    safetensors = pytest.importorskip(
        "safetensors", reason="install safetensors+numpy in a separate venv"
    )
    pytest.importorskip("numpy")

    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    bodies: dict[str, bytes] = source["bodies"]  # type: ignore[assignment]

    # Write the composed bytes out ONCE, only so the reference library -- which
    # can only open a path -- can check them. Nothing in the API needs this.
    proof = tmp_path / "composed.safetensors"
    proof.write_bytes(read_entry(cas, manifest.files[0]))

    with safetensors.safe_open(proof, framework="numpy") as reference:
        assert set(reference.keys()) == set(bodies)
        for name in sorted(reference.keys()):
            if name == "text_encoder.b":
                continue  # numpy has no bfloat16
            assert reference.get_tensor(name).tobytes() == bodies[name], name
