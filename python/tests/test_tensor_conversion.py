"""The read-transform-write loop a conversion actually needs.

The claim under test is not just "it round-trips". It is that a conversion
touching one tensor:

* never holds a shard, in either direction;
* leaves every other tensor's CAS objects **bit-identical**, so nothing
  unchanged is rewritten locally or re-uploaded; and
* produces a snapshot the real ``safetensors`` library reads back byte-exactly.
"""

from __future__ import annotations

import inspect
import json
import random
import struct
import textwrap
from pathlib import Path

import pytest
from repo_ingest import ingest_file, ingest_with_grid
from tensorfs import (
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    TensorError,
    TensorWriter,
    open_tensors,
    read_entry,
)
from tensorfs.manifest import MAX_CHUNK_SIZE
from tensorfs.native import plan_and_hash_bytes

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

    A packed grid — where small tensors share an object and therefore cannot
    be inherited at all — is the wire-legal alternative, and that difference
    is asserted directly further down.
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
    # One whole-file object — the wire allows it, and old packing-grid
    # snapshots look exactly like this — so both tensors share an object.
    model = source_dir / "model.safetensors"
    entry = ingest_with_grid(
        cas, model, [model.stat().st_size], manifest_path=model.name
    )
    manifest = RepositoryManifest((entry,))
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


# ---------------------------------------------------------------------------
# The double-hash fence (#61)
#
# `plan_seal_job` re-plans every committed file and reuses an object only when
# an admitted region matches the planner's `(offset, length)` exactly. If the
# write API picks boundaries the planner would not, NOTHING matches and seal
# re-reads and re-hashes the whole file -- the API becomes slower than
# `save_file` while still looking correct. So the grid is not asserted against
# a hand-written list of lengths; it is asserted against the canonical planner
# itself, digest for digest.
# ---------------------------------------------------------------------------


def _grid(entry: FileEntry) -> list[tuple[int, str]]:
    """A committed file's object grid, spelled the way the planner spells it."""

    return [(chunk.length, str(chunk.digest).removeprefix("sha256:")) for chunk in entry.chunks]


def test_the_written_grid_is_exactly_the_grid_the_planner_would_choose(
    source: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    entry = manifest.files[0]

    composed = read_entry(cas, entry)
    plan = plan_and_hash_bytes(composed)

    assert plan.planner == "safetensors-v1"
    assert plan.file_size == entry.size_bytes
    # Every object the writer admitted is one the planner would have chosen,
    # in the same order and at the same length. A re-plan at seal therefore
    # admits zero additional objects.
    assert [(o.length, o.digest) for o in plan.objects] == _grid(entry)


def test_a_boundary_the_planner_would_not_choose_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The red arm for the fence above: perturb one boundary, stay byte-exact.

    The composed *bytes* are unchanged -- only where the writer cut them is.
    That is exactly the regression the fence exists to catch, and it is
    invisible to every other assertion in this file.
    """

    body = random.Random("fence").randbytes(4 << 20)
    shape = (len(body) // 4,)

    canonical = TensorWriter(LocalCAS(tmp_path / "canonical"), "model.safetensors")
    canonical.add("w", "F32", shape, body)
    good = canonical.finish()

    # Half-sized objects: still wire-legal, still the same file, wrong grid.
    monkeypatch.setattr("tensorfs.writer.MAX_CHUNK_SIZE", 1 << 20)
    perturbed = TensorWriter(LocalCAS(tmp_path / "perturbed"), "model.safetensors")
    perturbed.add("w", "F32", shape, body)
    bad = perturbed.finish()

    assert read_entry(LocalCAS(tmp_path / "canonical"), good) == read_entry(
        LocalCAS(tmp_path / "perturbed"), bad
    )
    assert good.digest == bad.digest, "the perturbation must not change the file"

    plan = plan_and_hash_bytes(read_entry(LocalCAS(tmp_path / "canonical"), good))
    reference = [(o.length, o.digest) for o in plan.objects]
    assert reference == _grid(good)
    assert reference != _grid(bad)


def test_the_composed_file_is_byte_identical_to_save_file(tmp_path: Path) -> None:
    """The oracle #61 asks for: `safetensors.save_file` then ingest, compared.

    The comparison runs all the way to the snapshot id, not just the bytes:
    same file, same planner grid, same manifest digest. It holds because
    ``TensorWriter`` lays tensors out in the order they were added and a
    conversion adds them in the order it read them -- which for any file the
    reference library wrote is the order that library chose.
    """

    numpy = pytest.importorskip("numpy")
    safetensors_numpy = pytest.importorskip("safetensors.numpy")

    # Deliberately NOT the order the reference library lays these out in, so
    # the test proves the layout came from the file rather than from luck.
    weights = {
        "block.0.scale": numpy.arange(16, dtype=numpy.uint8).reshape(4, 4),
        "block.1.weight": (numpy.arange(256, dtype=numpy.float32) * 7).reshape(16, 16),
        "block.0.weight": numpy.arange(4096, dtype=numpy.float32).reshape(64, 64),
    }
    reference = tmp_path / "reference.safetensors"
    safetensors_numpy.save_file(weights, reference)
    raw = reference.read_bytes()

    # The reference library orders the header itself; read that order back out
    # rather than guessing at it.
    header = json.loads(raw[8 : 8 + struct.unpack("<Q", raw[:8])[0]])
    order = [name for name in header if name != "__metadata__"]
    assert order != list(weights), "the fixture must exercise a reordering"

    cas = LocalCAS(tmp_path / "cas")
    writer = TensorWriter(cas, "reference.safetensors")
    for name in order:
        array = weights[name]
        dtype = {"float32": "F32", "uint8": "U8"}[str(array.dtype)]
        writer.add(name, dtype, array.shape, array.tobytes())
    composed = writer.finish()

    assert read_entry(cas, composed) == raw
    ingested = ingest_file(cas, reference, manifest_path="reference.safetensors")
    assert composed == ingested, "same digest, same size, same object grid"
    assert cas.store_manifest(RepositoryManifest((composed,))) == cas.store_manifest(
        RepositoryManifest((ingested,))
    ), "the same snapshot id"


# ---------------------------------------------------------------------------
# The complexity gate (#61)
#
# Paul's constraint was "without needing to be complex". The loop it replaces
# is `python-gen-worker/src/gen_worker/models/w8a8.py:993-1022`, 30 lines. So
# the loop is written here as a real function, run end to end, and measured.
# ---------------------------------------------------------------------------


def _conversion_loop(cas: LocalCAS, manifest: RepositoryManifest, target: str) -> FileEntry:
    """Read one tensor at a time, transform one of them, write them back."""

    with open_tensors(cas, manifest) as source:
        writer = TensorWriter(cas, manifest.files[0].path)
        for name in source:
            view = source[name]
            if name == target:
                writer.add(name, "U8", view.shape, bytes(view.nbytes // 4))
            else:
                writer.inherit(view)
        return writer.finish()


def test_the_conversion_loop_is_no_more_complex_than_save_file(
    source: dict[str, object],
) -> None:
    cas = LocalCAS(Path(str(source["root"])) / "cas")
    manifest: RepositoryManifest = source["manifest"]  # type: ignore[assignment]
    bodies: dict[str, bytes] = source["bodies"]  # type: ignore[assignment]

    entry = _conversion_loop(cas, manifest, "denoiser.small")

    with open_tensors(cas, RepositoryManifest((entry,))) as out:
        assert out["denoiser.small"].dtype == "U8"
        assert out["text_encoder.a"].tobytes() == bodies["text_encoder.a"]

    body = [
        line
        for line in textwrap.dedent(inspect.getsource(_conversion_loop)).splitlines()
        if line.strip() and not line.strip().startswith(('"""', "#"))
    ]
    # 30 lines is `w8a8.py:993-1022`, the loop this replaces. Longer than that
    # and the API failed its own acceptance criterion.
    assert len(body) <= 30, len(body)


# ---------------------------------------------------------------------------
# Emission order (#61's recorded design point)
#
# `TensorWriter` emits in insertion order; `safetensors.save_file` sorts. The
# question left open on the issue was whether a caller that invents a fresh
# order therefore "will not dedup against a hub copy". Measured rather than
# assumed: it does dedup. Order decides the FILE's identity -- its digest, and
# so the snapshot id -- and nothing else. The object grid is per tensor and
# objects are named by their own bytes, so every tensor object is shared with
# the canonically ordered copy and only the header object differs.
#
# That is why the writer keeps insertion order rather than adopting a canonical
# one: the order a conversion needs is its SOURCE's order, which is the only
# order that makes the untouched half of a checkpoint byte-identical.
# ---------------------------------------------------------------------------


def test_reordering_costs_the_header_object_and_nothing_else(tmp_path: Path) -> None:
    bodies = {
        "block.0.weight": random.Random("w0").randbytes(4096),
        "block.1.weight": random.Random("w1").randbytes(2048),
        "block.2.weight": random.Random("w2").randbytes(1024),
    }
    names = list(bodies)

    def compose(root: str, order: list[str]) -> FileEntry:
        cas = LocalCAS(tmp_path / root)
        writer = TensorWriter(cas, "model.safetensors")
        for name in order:
            writer.add(name, "U8", (len(bodies[name]),), bodies[name])
        return writer.finish()

    forward = compose("forward", names)
    backward = compose("backward", list(reversed(names)))

    # A different file, therefore a different snapshot id. That is the whole
    # cost, and it is real.
    assert forward.digest != backward.digest
    assert forward.size_bytes == backward.size_bytes

    forward_objects = [str(chunk.digest) for chunk in forward.chunks]
    backward_objects = [str(chunk.digest) for chunk in backward.chunks]

    # Every tensor object survives the reordering. Only the header differs, and
    # exactly one object is new on each side.
    assert set(forward_objects[1:]) == set(backward_objects[1:])
    assert set(backward_objects) - set(forward_objects) == {backward_objects[0]}
    assert len(set(forward_objects) ^ set(backward_objects)) == 2

    # And the reordered file is still a correct file.
    cas = LocalCAS(tmp_path / "backward")
    with open_tensors(cas, RepositoryManifest((backward,))) as out:
        assert list(out) == list(reversed(names))
        for name, body in bodies.items():
            assert out[name].tobytes() == body, name


def test_a_corrupted_object_is_caught_while_the_file_digest_is_taken(
    tmp_path: Path,
) -> None:
    """`finish()` verifies every object it folds in, in the same single pass.

    It used to call `verify_object` and then read the object again for the
    whole-file hash -- two SHA-256 passes over the same bytes, which is the
    exact double-hash `ObjectStore::admit_regions` warns about. Collapsing them
    must not have collapsed the check, so the check is asserted directly.
    """

    cas = LocalCAS(tmp_path / "cas")
    seed = TensorWriter(cas, "seed.safetensors")
    seed.add("w", "U8", (4096,), random.Random("corrupt").randbytes(4096))
    entry = seed.finish()

    with open_tensors(cas, RepositoryManifest((entry,))) as src:
        view = src["w"]
        writer = TensorWriter(cas, "out.safetensors")
        writer.inherit(view)

    # Rewrite the object under its own name, keeping the length.
    victim = cas.object_path(entry.chunks[1].digest)
    victim.chmod(0o644)
    victim.write_bytes(bytes(4096))

    with pytest.raises(TensorError, match="do not match their digest"):
        writer.finish()
