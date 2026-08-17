"""The GGUF half of the write path (#61 row 2).

The safetensors arm is proved in ``test_tensor_conversion.py``. This is the
same claim for GGUF, and the oracle is the same shape: the real ``gguf``
library writes a file, we compose the same file out of CAS objects, and the two
must agree byte for byte, grid for grid, id for id -- plus the real reader must
open ours and hand back the quantized block type it went in with.

GGUF is the harder container and the difference is padding. Its planner emits
the metadata block, the tensor directory and the pre-data padding as three
separate header regions, and every tensor's trailing alignment padding as a
region of its own. A writer that folded padding into the tensor before it would
still produce a valid, readable, byte-identical file -- and every object in it
would miss at seal. That is what
``test_a_gguf_boundary_the_planner_would_not_choose_is_caught`` exists for.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import gguf as reference
import numpy
import pytest
from repo_ingest import ingest_file
from tensorfs import (
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    TensorError,
    TensorWriter,
    open_tensors,
    read_entry,
)
from tensorfs.gguf import GGUFHeader, align_up
from tensorfs.manifest import MAX_CHUNK_SIZE
from tensorfs.native import plan_and_hash_bytes

# Q4_K is 256 elements per 144-byte block; the block geometry is the thing
# safetensors cannot express and the thing this row is really about.
_Q4_K = reference.GGMLQuantizationType.Q4_K
_BLOCK_BYTES = 144

_Array = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.generic]]
_Tensor = tuple[str, _Array, "reference.GGMLQuantizationType | None"]


def _quant_blocks(seed: str, rows: int) -> _Array:
    """`rows` Q4_K blocks of pseudorandom bytes, in the library's byte shape."""

    raw = random.Random(seed).randbytes(rows * _BLOCK_BYTES)
    return numpy.frombuffer(raw, dtype=numpy.uint8).reshape(rows, _BLOCK_BYTES)


def _float_rows(seed: str, rows: int, columns: int) -> _Array:
    values = numpy.frombuffer(
        random.Random(seed).randbytes(rows * columns * 4), dtype=numpy.uint8
    )
    return values.view(numpy.float32).reshape(rows, columns)


def _write_reference(path: Path, tensors: list[_Tensor]) -> Path:
    """Author a GGUF with the real library, in the order given."""

    writer = reference.GGUFWriter(path, "llama")
    for name, body, raw_dtype in tensors:
        if raw_dtype is None:
            writer.add_tensor(name, body)
        else:
            writer.add_tensor(name, body, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _source(tmp_path: Path) -> list[_Tensor]:
    """Deliberately mixed: quantized, F32, and one tensor above 64 MiB.

    The big tensor is what forces the multi-object split inside a single
    tensor, and the quantized one is what forces the block geometry through.
    Nothing is a constant fill, so a shifted read cannot compare equal.
    """

    return [
        ("blk.0.ffn.weight", _quant_blocks("ffn", 4), _Q4_K),
        # 68 MiB of F32: two objects, the second a remainder.
        ("blk.0.attn.weight", _float_rows("attn", 17, 1024 * 1024), None),
        ("blk.0.norm.weight", _float_rows("norm", 3, 7), None),
        # 5 Q4_K blocks is 720 bytes, which is NOT a multiple of the 32-byte
        # alignment -- so the LAST tensor is padded too. A fixture whose
        # final tensor happens to land on the alignment cannot see a writer
        # that skips the last padding run, and the byte-identity assertion
        # goes green while the file is short. (Measured: it did.)
        ("output.weight", _quant_blocks("out", 5), _Q4_K),
    ]


@pytest.fixture(scope="module")
def committed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("gguf-conversion")
    original = _write_reference(root / "model.gguf", _source(root))
    cas = LocalCAS(root / "cas")
    entry = ingest_file(cas, original, manifest_path="model.gguf")
    return {"root": root, "cas_root": root / "cas", "manifest": RepositoryManifest((entry,))}


def _cas(committed: dict[str, object]) -> LocalCAS:
    return LocalCAS(Path(str(committed["cas_root"])))


def _grid(entry: FileEntry) -> list[tuple[int, str]]:
    return [(chunk.length, str(chunk.digest).removeprefix("sha256:")) for chunk in entry.chunks]


# ---------------------------------------------------------------------------
# The fixture has to be the hard case, or none of the below discriminates
# ---------------------------------------------------------------------------


def test_the_fixture_forces_padding_and_a_split_tensor(committed: dict[str, object]) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]

    with open_tensors(cas, manifest) as src:
        header = src.gguf_header("model.gguf")
        assert header.alignment == 32
        # At least one tensor is not a whole number of alignment units, so a
        # writer that forgot trailing padding would produce a shorter file...
        assert any(tensor.nbytes % header.alignment for tensor in header.tensors)
        # ...and the LAST one is such a tensor, which is the case a writer that
        # only skips the final padding run would otherwise slip past.
        assert header.tensors[-1].nbytes % header.alignment
        # And one crosses the 64 MiB object boundary.
        assert any(tensor.nbytes > MAX_CHUNK_SIZE for tensor in header.tensors)
        # And one is quantized.
        assert any(src[tensor.name].block.quantized for tensor in header.tensors)


# ---------------------------------------------------------------------------
# Row 2: byte-identity with the reference writer, and the block type survives
# ---------------------------------------------------------------------------


def _convert(cas: LocalCAS, manifest: RepositoryManifest, target: str) -> FileEntry:
    """The conversion loop, in GGUF: rewrite one tensor, inherit the rest."""

    with open_tensors(cas, manifest) as src:
        writer = TensorWriter(cas, "model.gguf", gguf_header=src.gguf_header("model.gguf"))
        for name in src:
            view = src[name]
            if name == target:
                # A real requantization would produce different bytes; this
                # one just replaces them, which exercises the same path.
                writer.add(name, view.dtype, view.shape, _replacement(view.nbytes))
            else:
                writer.inherit(view)
        return writer.finish()


def _replacement(nbytes: int) -> bytes:
    return random.Random("requantized").randbytes(nbytes)


def test_a_gguf_conversion_is_byte_identical_to_the_reference_writer(
    committed: dict[str, object], tmp_path: Path
) -> None:
    """The oracle this row asks for, in the direction that proves something.

    The reference library writes the POST-conversion file; we compose it out of
    inherited objects plus one new tensor. Same bytes, same object grid, same
    manifest digest, same snapshot id.
    """

    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]
    target = "output.weight"

    composed = _convert(cas, manifest, target)

    # What the reference library would have written for the same model.
    swapped: list[_Tensor] = []
    for name, body, raw_dtype in _source(tmp_path):
        if name == target:
            replaced = numpy.frombuffer(
                _replacement(int(body.nbytes)), dtype=numpy.uint8
            ).reshape(body.shape)
            swapped.append((name, replaced, raw_dtype))
        else:
            swapped.append((name, body, raw_dtype))
    expected = _write_reference(tmp_path / "expected.gguf", swapped)

    assert read_entry(cas, composed) == expected.read_bytes()

    ingested = ingest_file(cas, expected, manifest_path="model.gguf")
    assert composed == ingested, "same digest, same size, same object grid"
    assert cas.store_manifest(RepositoryManifest((composed,))) == cas.store_manifest(
        RepositoryManifest((ingested,))
    ), "the same snapshot id"


def test_the_real_gguf_reader_opens_the_converted_file(
    committed: dict[str, object], tmp_path: Path
) -> None:
    """Including the quantized block type, which is what GGUF adds over safetensors."""

    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]
    composed = _convert(cas, manifest, "output.weight")

    proof = tmp_path / "converted.gguf"
    proof.write_bytes(read_entry(cas, composed))
    served = {tensor.name: tensor for tensor in reference.GGUFReader(proof).tensors}

    original = {name: (body, raw) for name, body, raw in _source(tmp_path)}
    assert set(served) == set(original)

    for name, (body, raw_dtype) in original.items():
        tensor = served[name]
        # The block type is the thing safetensors has no equivalent for, so it
        # is the thing that has to survive: a quantized tensor must come back
        # as Q4_K, not as the raw bytes of one.
        expected_type = reference.GGMLQuantizationType.F32 if raw_dtype is None else _Q4_K
        assert tensor.tensor_type == expected_type, name
        assert tensor.n_bytes == body.nbytes, name
        # Bytes rather than values: the fixture is random, so a float compare
        # would trip over NaN and prove less, not more.
        expected_bytes = (
            _replacement(int(body.nbytes)) if name == "output.weight" else body.tobytes()
        )
        assert numpy.asarray(tensor.data).tobytes() == expected_bytes, name


def test_a_pure_inherit_pass_reproduces_the_source_gguf(
    committed: dict[str, object],
) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]

    with open_tensors(cas, manifest) as src:
        writer = TensorWriter(cas, "model.gguf", gguf_header=src.gguf_header("model.gguf"))
        for name in src:
            writer.inherit(src[name])
        rebuilt = writer.finish()

    assert rebuilt == manifest.files[0], "inheriting everything is the identity"


# ---------------------------------------------------------------------------
# Row 3 for GGUF: the reuse property, through the new API
# ---------------------------------------------------------------------------


def test_a_gguf_conversion_rewrites_only_what_it_touched(
    committed: dict[str, object],
) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]

    before = {path.name for path in cas.objects.rglob("*") if path.is_file()}
    inherited: dict[str, tuple[str, ...]] = {}
    with open_tensors(cas, manifest) as src:
        for name in src:
            if name == "output.weight":
                continue
            span = src.object_span(src[name])
            assert span is not None, name
            inherited[name] = tuple(str(ref) for ref, _size in span)

    composed = _convert(cas, manifest, "output.weight")
    after = {path.name for path in cas.objects.rglob("*") if path.is_file()}

    with open_tensors(cas, RepositoryManifest((composed,))) as out:
        for name, digests in inherited.items():
            span = out.object_span(out[name])
            assert span is not None
            assert tuple(str(ref) for ref, _size in span) == digests, name

    # New objects: the directory (the metadata block and the prefix are one
    # region and unchanged), and the one rewritten tensor. The 68 MiB tensor
    # contributes nothing.
    assert len(after - before) <= 2, sorted(after - before)


# ---------------------------------------------------------------------------
# Row 4 for GGUF: the double-hash fence
# ---------------------------------------------------------------------------


def test_the_written_gguf_grid_is_exactly_the_grid_the_planner_would_choose(
    committed: dict[str, object],
) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]
    composed = _convert(cas, manifest, "output.weight")

    plan = plan_and_hash_bytes(read_entry(cas, composed))
    assert plan.planner == "gguf-v1"
    assert plan.file_size == composed.size_bytes
    assert [(o.length, o.digest) for o in plan.objects] == _grid(composed)


def test_a_gguf_boundary_the_planner_would_not_choose_is_caught(tmp_path: Path) -> None:
    """The red arm: fold each tensor's trailing padding into the tensor.

    The bytes are unchanged and the file is still a valid GGUF the reference
    reader opens. Only where the writer cut them moves -- and every tensor
    object in the file misses at seal because of it.
    """

    original = _write_reference(
        tmp_path / "model.gguf",
        [
            ("blk.0.ffn.weight", _quant_blocks("fence-ffn", 3), _Q4_K),
            ("blk.0.norm.weight", _float_rows("fence-norm", 3, 5), None),
        ],
    )
    cas = LocalCAS(tmp_path / "cas")
    entry = ingest_file(cas, original, manifest_path="model.gguf")
    manifest = RepositoryManifest((entry,))

    with open_tensors(cas, manifest) as src:
        header = src.gguf_header("model.gguf")
        assert any(tensor.nbytes % header.alignment for tensor in header.tensors), (
            "the fixture must have padding to fold"
        )
        bodies = {name: src[name].tobytes() for name in src}
        shapes = {name: (src[name].dtype, src[name].shape) for name in src}

    good = TensorWriter(cas, "model.gguf", gguf_header=header)
    for name, body in bodies.items():
        dtype, shape = shapes[name]
        good.add(name, dtype, shape, body)
    canonical = good.finish()
    assert canonical == entry, "the writer must reproduce the reference file first"

    composed_bytes = read_entry(cas, canonical)
    merged = _fold_padding(
        LocalCAS(tmp_path / "perturbed"), composed_bytes, header, canonical
    )

    assert read_entry(LocalCAS(tmp_path / "perturbed"), merged) == composed_bytes, (
        "the perturbation must not change the file"
    )
    assert merged.digest == canonical.digest

    plan = plan_and_hash_bytes(composed_bytes)
    canonical_grid = [(o.length, o.digest) for o in plan.objects]
    assert canonical_grid == _grid(canonical)
    assert canonical_grid != _grid(merged), "the fence did not catch the folded padding"


def _fold_padding(
    cas: LocalCAS, composed: bytes, header: GGUFHeader, canonical: FileEntry
) -> FileEntry:
    """Re-cut `composed` with each tensor's trailing padding inside the tensor.

    Built from the header rather than by guessing at the object lengths, so the
    perturbation is exactly the mistake it claims to be.
    """

    lengths = [
        header.directory_start,
        header.directory_end - header.directory_start,
        header.data_start - header.directory_end,
    ]
    for tensor in header.tensors:
        padded = align_up(tensor.nbytes, header.alignment)
        remaining = padded
        while remaining > MAX_CHUNK_SIZE:
            lengths.append(MAX_CHUNK_SIZE)
            remaining -= MAX_CHUNK_SIZE
        lengths.append(remaining)
    lengths = [length for length in lengths if length]
    assert lengths != [chunk.length for chunk in canonical.chunks], (
        "nothing was folded, so this proves nothing"
    )
    assert sum(lengths) == len(composed)

    chunks = []
    at = 0
    whole = hashlib.sha256()
    for length in lengths:
        piece = composed[at : at + length]
        chunks.append(Chunk(cas.put_bytes(piece), length))
        whole.update(piece)
        at += length
    return FileEntry(canonical.path, len(composed), CASRef(whole.hexdigest()), tuple(chunks))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_gguf_writer_without_a_header_is_refused(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    with pytest.raises(TensorError, match="needs its metadata block"):
        TensorWriter(cas, "model.gguf")


def test_a_safetensors_tensor_cannot_be_inherited_into_a_gguf(
    committed: dict[str, object], tmp_path: Path
) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]

    other = LocalCAS(tmp_path / "other")
    safetensors_writer = TensorWriter(other, "model.safetensors")
    safetensors_writer.add("w", "F32", (4,), b"\1\2\3\4" * 4)
    plain = RepositoryManifest((safetensors_writer.finish(),))

    with open_tensors(cas, manifest) as src, open_tensors(other, plain) as flat:
        writer = TensorWriter(cas, "model.gguf", gguf_header=src.gguf_header("model.gguf"))
        with pytest.raises(TensorError, match="cannot inherit a safetensors-v1"):
            writer.inherit(flat["w"])

        gguf_writer = TensorWriter(other, "out.safetensors")
        with pytest.raises(TensorError, match="cannot inherit a gguf-v1"):
            gguf_writer.inherit(src["output.weight"])


def test_an_unknown_ggml_type_is_refused(committed: dict[str, object]) -> None:
    cas = _cas(committed)
    manifest: RepositoryManifest = committed["manifest"]  # type: ignore[assignment]
    with open_tensors(cas, manifest) as src:
        writer = TensorWriter(cas, "model.gguf", gguf_header=src.gguf_header("model.gguf"))
        with pytest.raises(TensorError, match="unknown ggml type"):
            writer.add("w", "NOT_A_TYPE", (32,), b"\0" * 32)


def test_the_gguf_conversion_loop_is_no_more_complex_than_save_file() -> None:
    """Row 7's gate, for the container it did not cover.

    The GGUF loop is the safetensors loop plus one argument -- the source's
    metadata block. If adding a container had cost a second API shape, that is
    where it would show.
    """

    import inspect
    import textwrap

    body = [
        line
        for line in textwrap.dedent(inspect.getsource(_convert)).splitlines()
        if line.strip() and not line.strip().startswith(('"""', "#"))
    ]
    assert len(body) <= 30, len(body)
