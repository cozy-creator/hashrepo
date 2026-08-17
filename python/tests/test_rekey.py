"""#80: re-keying a checkpoint costs one header object and shares every chunk.

The claims worth testing are the ones a consumer meets, so they are tested
against the REAL readers:

* the re-keyed snapshot opens in the real ``safetensors`` library under the new
  names, byte-for-byte -- with the red arm in the same test: corrupt the
  recomputed ``data_offsets`` and that same library refuses it;
* the GGUF twin opens in the real ``gguf`` reader the same way;
* the store grows by exactly the new header object plus the manifest, and every
  tensor object is the source's own; and
* the read-time mapping layer serves the other spelling with no storage at all.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import NamedTuple

import gguf
import numpy
import pytest
import safetensors.numpy
from numpy.typing import NDArray
from safetensors import safe_open
from tensorfs import LocalCAS, RepositoryManifest, TensorError, TensorWriter, open_tensors
from tensorfs.native import FileRecord, ObjectStore, RecordsReader, rekey
from tfm1_encode import Tensor, encode

Array = NDArray[numpy.float32]

# ComfyUI-style names in, diffusers-style names out. The mapping is the whole
# operation.
_RENAME = {
    "model.diffusion_model.blocks.0.attn.qkv.weight": "transformer.blocks.0.attn.qkv.weight",
    "model.diffusion_model.blocks.0.mlp.fc1.weight": "transformer.blocks.0.mlp.fc1.weight",
    "first_stage_model.decoder.conv_out.bias": "vae.decoder.conv_out.bias",
}


def _bodies() -> dict[str, Array]:
    generator = numpy.random.default_rng(80)
    return {
        "model.diffusion_model.blocks.0.attn.qkv.weight": generator.random(
            (64, 192), dtype=numpy.float32
        ),
        "model.diffusion_model.blocks.0.mlp.fc1.weight": generator.random(
            (32, 128), dtype=numpy.float32
        ),
        "first_stage_model.decoder.conv_out.bias": generator.random((3,), dtype=numpy.float32),
    }


def _objects(store: ObjectStore) -> set[str]:
    return {path.name for path in (store.root / "objects").rglob("*") if path.is_file()}


def _ingest(store: ObjectStore, path: Path) -> tuple[str, list[FileRecord]]:
    """Commit one file exactly as the seal planner shapes it."""

    plan, admitted = store.admit_file(path)
    return plan.planner, [FileRecord.data(item.digest, item.length) for item in admitted]


def _materialize(store: ObjectStore, records: list[FileRecord], path: Path) -> Path:
    """Write composed bytes out ONCE, only because a real library opens paths.

    Nothing in the API needs a file: the same records are read in place by
    ``RecordsReader``. This exists so the reference readers can be pointed at
    something.
    """

    reader = RecordsReader(store, records)
    path.write_bytes(reader.read_at(0, reader.length))
    return path


def _digests(records: list[FileRecord]) -> list[str]:
    return [record.digest for record in records if record.digest is not None]


def _refusal(path: Path) -> BaseException | None:
    """What the real safetensors library does with these bytes."""

    try:
        with safe_open(path, framework="numpy") as reference:
            for name in reference.keys():
                reference.get_tensor(name)
    except Exception as error:
        return error
    return None


def _snapshot(path: str, planner: str, records: list[FileRecord]) -> bytes:
    return encode(
        [
            Tensor(
                path,
                planner,
                tuple(
                    (record.digest, record.length)
                    for record in records
                    if record.digest is not None
                ),
            )
        ]
    )


class Source(NamedTuple):
    store: ObjectStore
    planner: str
    records: list[FileRecord]
    bodies: dict[str, Array]


@pytest.fixture
def source(tmp_path: Path) -> Source:
    bodies = _bodies()
    original = tmp_path / "model.safetensors"
    safetensors.numpy.save_file(bodies, original)
    store = ObjectStore(tmp_path / "store")
    planner, records = _ingest(store, original)
    assert planner == "safetensors-v1"
    return Source(store, planner, records, bodies)


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


def test_the_real_safetensors_library_opens_the_rekeyed_snapshot(
    source: Source, tmp_path: Path
) -> None:
    composed = rekey(source.store, source.planner, source.records, _RENAME)

    proof = _materialize(source.store, composed, tmp_path / "rekeyed.safetensors")
    assert _refusal(proof) is None
    with safe_open(proof, framework="numpy") as reference:
        assert set(reference.keys()) == set(_RENAME.values())
        for old, new in _RENAME.items():
            assert numpy.array_equal(reference.get_tensor(new), source.bodies[old]), new

    # The red arm, in the same test: `data_offsets` are the one thing a
    # composition derives rather than inherits, so corrupt them and the same
    # library must refuse the file.
    raw = proof.read_bytes()
    header_length: int = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    victim = header[next(iter(_RENAME.values()))]
    victim["data_offsets"] = [victim["data_offsets"][0] + 8, victim["data_offsets"][1] + 8]
    corrupted = json.dumps(header, separators=(",", ":")).encode()
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(struct.pack("<Q", len(corrupted)) + corrupted + raw[8 + header_length :])
    assert _refusal(broken) is not None


def test_the_store_grows_by_exactly_the_header_and_the_manifest(source: Source) -> None:
    before = _objects(source.store)

    composed = rekey(source.store, source.planner, source.records, _RENAME)
    source.store.put_bytes(_snapshot("model.safetensors", source.planner, composed))

    new = _objects(source.store) - before
    assert len(new) == 2, new
    assert _digests(composed)[1:] == _digests(source.records)[1:], (
        "every tensor object is inherited verbatim, in order"
    )
    assert _digests(composed)[0] != _digests(source.records)[0], "the header is the new object"
    assert _digests(composed)[0] in new


def test_the_composed_records_are_what_a_re_ingest_would_produce(
    source: Source, tmp_path: Path
) -> None:
    """So a foreign tool that writes the renamed file itself dedups fully."""

    composed = rekey(source.store, source.planner, source.records, _RENAME)
    written = _materialize(source.store, composed, tmp_path / "rekeyed.safetensors")

    fresh = ObjectStore(tmp_path / "fresh")
    _, reingested = _ingest(fresh, written)
    assert _digests(reingested) == _digests(composed)


def test_a_renaming_that_is_not_a_bijection_is_refused(source: Source) -> None:
    before = _objects(source.store)
    everything = dict(_RENAME)

    with pytest.raises(Exception, match="does not name source tensor"):
        rekey(source.store, source.planner, source.records, dict(list(everything.items())[:2]))
    with pytest.raises(Exception, match="would both be named"):
        rekey(source.store, source.planner, source.records, dict.fromkeys(everything, "one"))
    with pytest.raises(Exception, match="holds no tensor named"):
        rekey(source.store, source.planner, source.records, {**everything, "absent": "x"})

    assert _objects(source.store) == before, "a refused composition admits nothing"


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


def test_the_real_gguf_reader_opens_the_rekeyed_twin(tmp_path: Path) -> None:
    bodies = {
        "blk.0.attn_q.weight": numpy.arange(256, dtype=numpy.float32).reshape(16, 16),
        "blk.0.attn_k.weight": numpy.arange(64, dtype=numpy.float32).reshape(8, 8),
        "output.weight": numpy.arange(96, dtype=numpy.float32).reshape(8, 12),
    }
    original = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(original, "llama")
    for name, body in bodies.items():
        writer.add_tensor(name, body)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    store = ObjectStore(tmp_path / "store")
    planner, records = _ingest(store, original)
    assert planner == "gguf-v1"

    mapping = {
        "blk.0.attn_q.weight": "layers.0.self_attn.q_proj.weight",
        "blk.0.attn_k.weight": "layers.0.self_attn.k_proj.weight",
        "output.weight": "lm_head.weight",
    }
    before = _objects(store)
    composed = rekey(store, planner, records, mapping)

    # Only the tensor directory and the padding that follows it can be new; the
    # metadata block, every tensor and every alignment run are inherited.
    assert len(_objects(store) - before) <= 2
    shared = set(_digests(records)) & set(_digests(composed))
    assert len(set(_digests(composed)) - shared) <= 2

    proof = _materialize(store, composed, tmp_path / "rekeyed.gguf")
    served = {tensor.name: tensor for tensor in gguf.GGUFReader(proof).tensors}
    assert set(served) == set(mapping.values())
    for old, new in mapping.items():
        assert numpy.array_equal(
            numpy.asarray(served[new].data).reshape(bodies[old].shape), bodies[old]
        ), new


# ---------------------------------------------------------------------------
# The read-time mapping layer: the same rename with no storage at all
# ---------------------------------------------------------------------------


def test_the_read_time_mapping_layer_admits_nothing(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    bodies = {name: body.tobytes() for name, body in _bodies().items()}
    writer = TensorWriter(cas, "model.safetensors")
    for name, body in bodies.items():
        writer.add(name, "U8", (len(body),), body)
    manifest = RepositoryManifest((writer.finish(),))

    before = {path.name for path in cas.objects.rglob("*") if path.is_file()}
    with open_tensors(cas, manifest) as reader:
        served = reader.rekeyed(_RENAME)
        assert set(served) == set(_RENAME.values())
        for old, new in _RENAME.items():
            assert served[new].name == new
            assert served[new].tobytes() == bodies[old], new

        # An unmapped tensor keeps its own name, and two may not collide.
        first = next(iter(_RENAME))
        partial = reader.rekeyed({first: "renamed"})
        assert set(partial) == (set(_RENAME) - {first}) | {"renamed"}
        with pytest.raises(TensorError, match="both be served"):
            reader.rekeyed(dict.fromkeys(_RENAME, "collision"))
        with pytest.raises(TensorError, match="no tensor named"):
            reader.rekeyed({"absent": "whatever"})

    assert {path.name for path in cas.objects.rglob("*") if path.is_file()} == before
