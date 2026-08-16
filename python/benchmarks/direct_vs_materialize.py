"""Direct tensor reads versus whole-file materialization, on one fixture.

Each arm runs in its own subprocess so peak RSS is a real high-water mark for
that arm alone. Bytes come from ``/proc/self/io`` rather than from an estimate:
``rchar``/``wchar`` are the bytes the process asked the kernel to move,
``read_bytes``/``write_bytes`` are the bytes that actually reached the device.

Read the page-cache caveat in the output before quoting the wall times.

    python3 python/benchmarks/direct_vs_materialize.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfs import (  # noqa: E402
    CASRef,
    Chunk,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    native,
    open_tensors,
)
from tensorfs.manifest import MAX_CHUNK_SIZE  # noqa: E402


def ingest_repository(cas: LocalCAS, root: Path) -> RepositoryManifest:
    """Commit every file under the Rust planner's grid — the only chunker."""

    entries: list[FileEntry] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        whole = hashlib.sha256()
        chunks: list[Chunk] = []
        with path.open("rb") as handle:
            for region in native.plan_file(path).regions:
                data = handle.read(region.length)
                whole.update(data)
                chunks.append(Chunk(cas.put_bytes(data), region.length))
        entries.append(
            FileEntry(
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                CASRef(whole.hexdigest()),
                tuple(chunks),
            )
        )
    return RepositoryManifest(tuple(entries))

MIB = 1 << 20
BIG = "dense.weight"

# One tensor above 64 MiB so it spans several objects, plus enough smaller
# tensors that the file is meaningfully larger than the tensor under test.
_TENSORS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    (BIG, "F32", (26, 1024, 1024)),
    *((f"block.{i}.weight", "F32", (8, 1024, 1024)) for i in range(12)),
    *((f"block.{i}.bias", "F32", (1024,)) for i in range(12)),
)
_DTYPE_BYTES = {"F32": 4}


def build_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    model = source / "model.safetensors"
    if not model.exists():
        header: dict[str, object] = {}
        cursor = 0
        bodies = []
        for name, dtype, shape in _TENSORS:
            count = 1
            for dimension in shape:
                count *= dimension
            size = count * _DTYPE_BYTES[dtype]
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + size],
            }
            bodies.append((name, size))
            cursor += size
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with model.open("wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for name, size in bodies:
                handle.write(random.Random(name).randbytes(size))
    return source, model


def counters() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, _, value = line.partition(": ")
        out[key] = int(value)
    return out


def faults() -> tuple[int, int]:
    """Minor/major page faults: where an mmap's byte movement actually shows.

    Direct reads move bytes by faulting mapped pages, not by ``read()``, so
    their ``rchar`` is legitimately zero. Without this counter that zero looks
    like the arm did no work.
    """

    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[7]), int(fields[9])


def rss() -> dict[str, int]:
    """Peak total RSS, plus the anonymous/file split that peak hides.

    VmHWM counts mmapped file pages, but those are clean and the kernel can
    drop them under pressure. The anonymous component is the part that cannot
    be reclaimed and is what actually exhausts a box, so both are reported.
    """

    out = {"peak_rss": 0, "rss_anon": 0, "rss_file": 0}
    keys = {"VmHWM:": "peak_rss", "RssAnon:": "rss_anon", "RssFile:": "rss_file"}
    for line in Path("/proc/self/status").read_text().splitlines():
        field = line.split()
        if field and field[0] in keys:
            out[keys[field[0]]] = int(field[1]) * 1024
    return out


# -- the file-shaped reader the incumbent route needs -----------------------


def read_tensor_from_file(path: Path, wanted: str | None) -> int:
    """Read one tensor (or all) out of a real safetensors file via mmap.

    This stands in for ``safe_open``. It is deliberately the cheapest possible
    implementation — no library, no object construction — so the comparison is
    generous to the route being replaced.
    """

    import mmap

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ) as region:
            size = int.from_bytes(region[:8], "little")
            header = json.loads(region[8 : 8 + size])
            base = 8 + size
            total = 0
            for name, descriptor in header.items():
                if name == "__metadata__" or (wanted is not None and name != wanted):
                    continue
                start, stop = descriptor["data_offsets"]
                total += len(bytes(region[base + start : base + stop]))
            return total


# -- arms -------------------------------------------------------------------


def arm_direct_one(cas: LocalCAS, ref: str, **_: object) -> int:
    with open_tensors(cas, ref) as tensors:
        return len(tensors[BIG].tobytes())


def arm_direct_one_noverify(cas: LocalCAS, ref: str, **_: object) -> int:
    with open_tensors(cas, ref, verify=False) as tensors:
        return len(tensors[BIG].tobytes())


def _drain(pieces: object, staging: bytearray) -> int:
    """Copy each piece into a reused bounded buffer, touching every byte.

    This is what a GPU loader does: the destination is device memory, so the
    host never holds more than one piece. Counting ``len(piece)`` instead
    would fault in no pages at all and measure nothing.
    """

    moved = 0
    for piece in pieces:  # type: ignore[attr-defined]
        size = len(piece)
        staging[:size] = piece
        moved += size
    return moved


def arm_direct_one_pieces(cas: LocalCAS, ref: str, **_: object) -> int:
    """Never holds the whole tensor: the shape a GPU loader actually uses."""

    staging = bytearray(MAX_CHUNK_SIZE)
    with open_tensors(cas, ref, verify=False) as tensors:
        return _drain(tensors[BIG].pieces(), staging)


def arm_direct_all(cas: LocalCAS, ref: str, **_: object) -> int:
    with open_tensors(cas, ref) as tensors:
        return sum(len(tensors[name].tobytes()) for name in tensors)


def arm_direct_all_pieces(cas: LocalCAS, ref: str, **_: object) -> int:
    staging = bytearray(MAX_CHUNK_SIZE)
    with open_tensors(cas, ref, verify=False) as tensors:
        return sum(_drain(tensors[name].pieces(), staging) for name in tensors)


def legacy_materialize(cas: LocalCAS, entry: FileEntry, target: Path) -> Path:
    """The deleted ``LocalCAS.materialize``, reproduced verbatim in behaviour.

    This is the route being replaced. It is kept here, and only here, so the
    comparison stays runnable after the hard cut removed it from the library.
    Note what it costs: every object is read once to verify its digest, read
    again while being copied, hashed a second time per object and a third time
    into a whole-file digest, and then written out and fsynced.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    whole = hashlib.sha256()
    total = 0
    with target.open("wb") as writer:
        for ref, size in entry.objects():
            source = cas.verify_object(ref, size=size)
            with source.open("rb") as reader:
                per_object = hashlib.sha256()
                remaining = size
                while remaining:
                    data = reader.read(min(1 << 20, remaining))
                    if not data:
                        raise RuntimeError(f"{ref}: object ended before {size} bytes")
                    writer.write(data)
                    per_object.update(data)
                    whole.update(data)
                    total += len(data)
                    remaining -= len(data)
            if per_object.hexdigest() != ref.digest:
                raise RuntimeError(f"{ref}: object changed while materializing")
        writer.flush()
        os.fsync(writer.fileno())
    if total != entry.size_bytes or whole.hexdigest() != entry.digest.digest:
        raise RuntimeError(f"{entry.path}: reconstruction does not match the manifest")
    return target


def _materialized(cas: LocalCAS, ref: str, scratch: Path) -> Path:
    manifest = cas.load_manifest(ref)
    entry = next(item for item in manifest.files if item.path == "model.safetensors")
    return legacy_materialize(cas, entry, scratch / "model.safetensors")


def arm_materialize_one(cas: LocalCAS, ref: str, scratch: Path, **_: object) -> int:
    return read_tensor_from_file(_materialized(cas, ref, scratch), BIG)


def arm_materialize_all(cas: LocalCAS, ref: str, scratch: Path, **_: object) -> int:
    return read_tensor_from_file(_materialized(cas, ref, scratch), None)


ARMS = {
    "direct-one": arm_direct_one,
    "direct-one-noverify": arm_direct_one_noverify,
    "direct-one-pieces": arm_direct_one_pieces,
    "materialize-one": arm_materialize_one,
    "direct-all": arm_direct_all,
    "direct-all-pieces": arm_direct_all_pieces,
    "materialize-all": arm_materialize_all,
}


def run_arm(name: str, root: Path) -> dict[str, object]:
    cas = LocalCAS(root / "cas")
    ref = (root / "ref").read_text().strip()
    scratch = root / "scratch" / name
    scratch.mkdir(parents=True, exist_ok=True)
    before = counters()
    minor_before, major_before = faults()
    started = time.perf_counter()
    produced = ARMS[name](cas=cas, ref=ref, scratch=scratch)
    elapsed = time.perf_counter() - started
    after = counters()
    minor_after, major_after = faults()
    return {
        "arm": name,
        "seconds": round(elapsed, 4),
        "tensor_bytes": produced,
        "rchar": after["rchar"] - before["rchar"],
        "wchar": after["wchar"] - before["wchar"],
        "read_bytes": after["read_bytes"] - before["read_bytes"],
        "write_bytes": after["write_bytes"] - before["write_bytes"],
        "faulted": (minor_after - minor_before) * os.sysconf("SC_PAGE_SIZE"),
        "major_faults": major_after - major_before,
        **rss(),
    }


def human(value: int) -> str:
    return f"{value / MIB:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm")
    parser.add_argument("--root", default="/tmp/tensorfs-direct-bench")
    args = parser.parse_args()
    root = Path(args.root)

    if args.arm:
        print(json.dumps(run_arm(args.arm, root)))
        return 0

    source, model = build_fixture(root)
    cas = LocalCAS(root / "cas")
    manifest = ingest_repository(cas, source)
    ref = cas.store_manifest(manifest)
    (root / "ref").write_text(str(ref))
    entry = next(item for item in manifest.files if item.path == "model.safetensors")
    lengths = [chunk.length for chunk in entry.chunks]
    big_objects = sum(1 for length in lengths if length == MAX_CHUNK_SIZE)

    print(f"fixture      {model.stat().st_size / MIB:.1f} MiB, {len(lengths)} CAS objects")
    print(f"             {BIG} spans >1 object: {big_objects >= 1}")
    print("             page cache is warm for every arm; see the caveat below\n")

    rows = []
    for name in ARMS:
        proc = subprocess.run(
            [sys.executable, __file__, "--arm", name, "--root", str(root)],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(json.loads(proc.stdout))

    head = (
        f"{'arm':<22}{'tensor':>9}{'read()':>9}{'wrote':>8}"
        f"{'moved':>9}{'peakRSS':>9}{'anonEnd':>9}{'sec':>7}"
    )
    print(head)
    print("-" * len(head))
    for row in rows:
        moved = row["rchar"] + row["wchar"] + row["faulted"]
        print(
            f"{row['arm']:<22}"
            f"{human(row['tensor_bytes']):>9}"
            f"{human(row['rchar']):>9}"
            f"{human(row['wchar']):>8}"
            f"{human(moved):>9}"
            f"{human(row['peak_rss']):>9}"
            f"{human(row['rss_anon']):>9}"
            f"{row['seconds']:>7.2f}"
        )
    print(
        "\nAll figures MiB except sec.\n"
        "  read()/wrote  rchar/wchar from /proc/self/io.\n"
        "  moved         read() + wrote + faulted pages. A direct arm's read()\n"
        "                is legitimately zero because an mmap moves bytes by\n"
        "                faulting, not by read(); this column compares fairly.\n"
        "  peakRSS       VmHWM. It counts mmapped file pages, which are clean\n"
        "                and reclaimable, so it OVERSTATES an mmap arm's cost.\n"
        "                It does not separate the arms and should not be used\n"
        "                to rank them.\n"
        "  anonEnd       RssAnon sampled at arm END, not a peak. It is here to\n"
        "                show the resident anonymous tail, NOT to support any\n"
        "                claim about peak allocation. Peak anonymous memory was\n"
        "                not measured; do not read a bound into this column.\n"
        "The page cache is warm for every arm, so wall times understate the\n"
        "cost of arms that write, and device reads are ~zero throughout.\n"
        "Bytes moved is the durable number; seconds and RSS are not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
