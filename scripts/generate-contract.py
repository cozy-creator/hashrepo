#!/usr/bin/env python3
"""Generate a CANDIDATE lane contract from real checkpoint headers.

The control-plane half of the generate -> human-ratify -> publish path
(tensorfs#130). It fetches nothing but HEADERS: a safetensors header is the
first bytes of the file and a GGUF directory sits right behind its 24-byte
prefix, so every source below is read with HTTP range requests measured in
kilobytes. **No weight byte crosses the machine that runs this** - that is the
weights-locality rule, and it is the reason the derivation can run on the
control plane at all.

    scripts/generate-contract.py \
        --name ernie.diffusers-bf16 --version 1 \
        --source base=hf:baidu/ERNIE-Image:transformer \
        --source turbo=hf:baidu/ERNIE-Image-Turbo:transformer \
        --out spec/v1/contracts/ernie.diffusers-bf16.v1.json

A source is ``<label>=<ref>[,<ref>...]`` where each ref is

    hf:<repo>[@<rev>]:<path>   a file, or a DIRECTORY (its *.safetensors /
                               *.gguf members, non-recursive)
    hf:<repo>[@<rev>]          the repo root, same directory rule
    https://...                one file
    /a/local/file.safetensors  one file

Several sources = several checkpoints of ONE family: a declaration is
``required`` only when every source carries it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from tensorfs import gguf  # noqa: E402
from tensorfs.generate import Observed, candidate, render  # noqa: E402

_HF = "https://huggingface.co"
_UA = {"User-Agent": "tensorfs-generate-contract/1 (header ranges only)"}
_MAX_HEADER_BYTES = 100_000_000


class SourceError(RuntimeError):
    """A source could not be read. Never downgraded to a warning: a document
    generated from a header set that silently lost a shard describes a
    checkpoint that does not exist."""


# ── fetching: headers only ───────────────────────────────────────────────────


def _auth() -> dict[str, str]:
    """``HF_TOKEN`` if the environment carries one. Gated repos (krea) refuse
    an anonymous range read with 401, and a header range is the smallest
    possible authenticated read of a weights repo."""

    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers={**_UA, **_auth(), **headers})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:  # pragma: no cover - network
        raise SourceError(f"{url}: HTTP {error.code}") from None
    except OSError as error:  # pragma: no cover - network
        raise SourceError(f"{url}: {error}") from None


def _ranged(url: str) -> Callable[[int, int], bytes]:
    def read(offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = offset + length - 1
        data = _get(url, {"Range": f"bytes={offset}-{end}"})
        if len(data) < length:
            raise SourceError(f"{url}: short range read ({len(data)} < {length})")
        return data[:length]

    return read


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={**_UA, **_auth()})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise SourceError(f"{url}: no Content-Length")
            return int(length)
    except urllib.error.HTTPError as error:  # pragma: no cover - network
        raise SourceError(f"{url}: HTTP {error.code}") from None


def _local(path: Path) -> tuple[Callable[[int, int], bytes], int]:
    handle = path.open("rb")

    def read(offset: int, length: int) -> bytes:
        handle.seek(offset)
        return handle.read(length)

    return read, path.stat().st_size


# ── header parsing ───────────────────────────────────────────────────────────


def _safetensors(read: Callable[[int, int], bytes], label: str) -> Iterator[Observed]:
    size = int.from_bytes(read(0, 8), "little")
    if not 2 <= size <= _MAX_HEADER_BYTES:
        raise SourceError(f"{label}: implausible safetensors header length {size}")
    header = json.loads(read(8, size))
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        yield Observed(name=name, dtype=descriptor["dtype"], rank=len(descriptor["shape"]))


def _gguf(read: Callable[[int, int], bytes], size: int) -> Iterator[Observed]:
    for tensor in gguf.read_header(read, size).tensors:
        yield Observed(name=tensor.name, dtype=tensor.dtype, rank=len(tensor.shape))


def _inventory(url_or_path: str) -> list[Observed]:
    if url_or_path.startswith("json:"):
        # A PRE-COMPUTED inventory: [{"name", "dtype", "rank"}, ...]. The escape
        # hatch for an artifact this platform PRODUCES rather than mirrors
        # (tensorhub/rife-4.25 is written pod-side by `cozy_rife.save_artifact`
        # from an upstream pickle), where the producing module's own state dict
        # IS the header - and deriving from it is what stops the document and
        # the producer from being able to disagree.
        raw = json.loads(Path(url_or_path[5:]).read_text(encoding="utf-8"))
        return [Observed(name=e["name"], dtype=e["dtype"], rank=int(e["rank"])) for e in raw]
    if url_or_path.startswith(("http://", "https://")):
        read, size = _ranged(url_or_path), None
    else:
        read, size = _local(Path(url_or_path))
    if url_or_path.endswith(".gguf"):
        if size is None:
            size = _size(url_or_path)
        return list(_gguf(read, size))
    return list(_safetensors(read, url_or_path))


# ── ref resolution ───────────────────────────────────────────────────────────


def _hf_tree(repo: str, rev: str, path: str) -> list[dict[str, object]]:
    url = f"{_HF}/api/models/{repo}/tree/{rev}/{path}".rstrip("/")
    return json.loads(_get(url, {}))


def _resolve(ref: str) -> list[str]:
    """A ref to the concrete file URLs it names."""

    if ref.startswith(("http://", "https://", "/", "./", "json:")):
        return [ref]
    if not ref.startswith("hf:"):
        raise SourceError(f"unrecognised ref {ref!r}")
    body = ref[3:]
    repo_rev, _, path = body.partition(":")
    repo, _, rev = repo_rev.partition("@")
    rev = rev or "main"
    if path and Path(path).suffix in {".safetensors", ".gguf", ".bin", ".ckpt"}:
        return [f"{_HF}/{repo}/resolve/{rev}/{path}"]
    entries = _hf_tree(repo, rev, path)
    files = sorted(
        str(entry["path"])
        for entry in entries
        if entry.get("type") == "file" and str(entry["path"]).endswith((".safetensors", ".gguf"))
    )
    if not files:
        listing = ", ".join(str(entry.get("path")) for entry in entries[:20]) or "(empty)"
        raise SourceError(
            f"{ref}: no .safetensors/.gguf under {path or '<root>'}; it holds: {listing}"
        )
    return [f"{_HF}/{repo}/resolve/{rev}/{name}" for name in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="library handle; omit for an anonymous custom")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--dtype",
        help="the LANE'S QUANTIZATION, torch-precise (float8_e4m3fn, float4_e2m1fn, "
        "bfloat16). Required whenever the header does not settle it by itself - a "
        "quantized lane's dtype is authored knowledge, not a vote over tensor counts.",
    )
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="this is a COMPONENT fragment (sdxl.clip-g-*, dit.blocks-fused-qkv), not a "
        "serve lane, so it may legitimately carry no top-level dtype. Anything else "
        "without one is undeclarable by pgw#1597's always-required lanes= and refuses.",
    )
    parser.add_argument(
        "--dtype-unknown-to-gen-worker",
        action="store_true",
        help="acknowledge a --dtype spelling DTYPE_MIN_SM does not know. It derives a "
        "floor of 0 SILENTLY, so only pass this after teaching that table the spelling "
        "and its true floor.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=REF[,REF...]",
        help="one checkpoint of the family",
    )
    parser.add_argument(
        "--literal",
        action="append",
        default=[],
        metavar="PATTERN",
        help="pin a collapsed pattern back to its literal names (a DEGENERATE hole)",
    )
    parser.add_argument(
        "--pin-trivial",
        action="store_true",
        help="auto-pin every DEGENERATE hole whose one value is 0 or 1 (the nn.Sequential "
        "position shape: to_out.0, lastconv.0, adaLN_modulation.1, T5 block.{i}.layer.0/1). "
        "A single value ABOVE 1 is left as a hole: it is a real index that this checkpoint "
        "happens to use once (the GGUF MTP block at 64), not a fixed position.",
    )
    parser.add_argument("--summary", default="", help="one paragraph of human context")
    parser.add_argument("--ratify", action="append", default=[], help="extra ratification item")
    parser.add_argument(
        "--ratified",
        action="append",
        default=[],
        metavar="EVIDENCE",
        help="an ANSWER to one of the four standing owed items (roles, fusions, sets, "
        "scope). Supplying any suppresses those four; with nothing left in --ratify the "
        "GENERATED CANDIDATE marker comes off and the document reads as ratified. State "
        "the evidence, not the conclusion.",
    )
    parser.add_argument("--out", help="write here; default stdout")
    arguments = parser.parse_args()

    sources: dict[str, list[Observed]] = {}
    sharded = False
    for raw in arguments.source:
        label, _, refs = raw.partition("=")
        if not label or not refs:
            parser.error(f"--source wants LABEL=REF, got {raw!r}")
        urls = [url for ref in refs.split(",") for url in _resolve(ref)]
        print(f"[{label}] {len(urls)} file(s)", file=sys.stderr)
        found: list[Observed] = []
        for url in urls:
            entries = _inventory(url)
            print(f"    {url.rsplit('/', 1)[-1]}: {len(entries)} tensors", file=sys.stderr)
            found.extend(entries)
        sources[label] = found
        # A source that resolved to SEVERAL members makes every declaration
        # optional — the matcher runs per member file, so all-required over a
        # sharded checkpoint refuses the tree this document was derived from.
        # Only this loop knows the member count; `candidate` sees one flat list.
        if len(urls) > 1:
            sharded = True

    pins = list(arguments.literal)
    if arguments.pin_trivial:
        # `fragment=True` only to keep this throwaway pass from tripping the
        # lane-dtype refusal; the real pass below applies it for real.
        _, first = candidate(
            sources, name=arguments.name, version=arguments.version, fragment=True,
            sharded=sharded,
        )
        trivial: dict[str, list[int | None]] = {}
        for pattern, position, value in first.degenerate_holes:
            if value in (0, 1):
                slots = trivial.setdefault(pattern, [None] * pattern.count("{i}"))
                slots[position] = value
        for pattern, slots in trivial.items():
            pieces = re.split(r"(\{i\})", pattern)
            hole = 0
            rebuilt = []
            for piece in pieces:
                if piece == "{i}":
                    rebuilt.append("{i}" if slots[hole] is None else str(slots[hole]))
                    hole += 1
                else:
                    rebuilt.append(piece)
            pins.append("".join(rebuilt))
        print(f"  --pin-trivial pinned {len(trivial)} pattern(s)", file=sys.stderr)

    document, report = candidate(
        sources,
        name=arguments.name,
        version=arguments.version,
        dtype=arguments.dtype,
        fragment=arguments.fragment,
        allow_unknown_dtype=arguments.dtype_unknown_to_gen_worker,
        literals=pins,
        summary=arguments.summary,
        ratification=arguments.ratify,
        ratified=arguments.ratified,
        sharded=sharded,
    )
    for line in report.lines():
        print(f"  {line}", file=sys.stderr)

    text = render(document)
    from tensorfs.contract import Contract  # validated by THE Rust validator

    Contract.from_document(text)
    print("  validated by the tensorfs core", file=sys.stderr)

    if arguments.out:
        Path(arguments.out).write_text(text, encoding="utf-8")
        print(f"  wrote {arguments.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
