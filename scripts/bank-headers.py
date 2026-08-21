#!/usr/bin/env python3
"""Bank REAL checkpoint headers as inventories — the input every v2 proof runs on.

A safetensors header is the first bytes of the file and a GGUF directory sits
behind a 24-byte prefix, so every source here is read with HTTP range requests
measured in kilobytes. **No weight byte crosses the machine that runs this.**

v1's generator kept name/dtype/rank; this keeps the SHAPE and the byte extent,
because a v2 topology IS a map of shapes. The banked files are what the v2
engine is measured against, so they are committed, not fetched at test time: a
proof that needs the network is a proof that is red on an airplane.

    scripts/bank-headers.py                 # every source in SOURCES.tsv
    scripts/bank-headers.py sdxl-base sd15  # just these
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from tensorfs import gguf  # noqa: E402

_HF = "https://huggingface.co"
_UA = {"User-Agent": "tensorfs-bank-headers/1 (header ranges only)"}
_MAX_HEADER_BYTES = 100_000_000
_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "spec" / "v2" / "headers"


class SourceError(RuntimeError):
    """A source could not be read. Never downgraded to a warning."""


def _auth(url: str) -> dict[str, str]:
    """The HF bearer, and ONLY to HF. A hub source hands out a PRESIGNED R2 URL
    whose signature is in the query string; sending a bearer alongside it is a
    second auth mechanism and S3 rejects the request outright."""

    token = os.environ.get("HF_TOKEN", "")
    if not token or not url.startswith(_HF + "/"):
        return {}
    return {"Authorization": f"Bearer {token}"}


def _get(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers={**_UA, **_auth(url), **headers})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise SourceError(f"{url}: HTTP {error.code}") from None
    except OSError as error:
        raise SourceError(f"{url}: {error}") from None


def _ranged(url: str) -> Callable[[int, int], bytes]:
    def read(offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        data = _get(url, {"Range": f"bytes={offset}-{offset + length - 1}"})
        if len(data) < length:
            raise SourceError(f"{url}: short range read ({len(data)} < {length})")
        return data[:length]

    return read


def _chunked(urls: list[str], chunk_bytes: int) -> Callable[[int, int], bytes]:
    """A hub blob above the chunk threshold is stored as N objects, so the
    header can straddle a boundary. Reading only chunk 0 would work for every
    checkpoint measured so far and would be a silent time bomb on the first one
    whose header does not fit; this walks the boundaries instead."""

    def read(offset: int, length: int) -> bytes:
        out = b""
        while length > 0:
            index, within = divmod(offset, chunk_bytes)
            if index >= len(urls):
                raise SourceError(f"chunk {index} is past the end of a {len(urls)}-chunk blob")
            take = min(length, chunk_bytes - within)
            data = _get(urls[index], {"Range": f"bytes={within}-{within + take - 1}"})
            if len(data) < take:
                raise SourceError(f"{urls[index]}: short range read ({len(data)} < {take})")
            out += data[:take]
            offset += take
            length -= take
        return out

    return read


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={**_UA, **_auth(url)})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise SourceError(f"{url}: no Content-Length")
        return int(length)


def _local(path: Path) -> tuple[Callable[[int, int], bytes], int]:
    handle = path.open("rb")

    def read(offset: int, length: int) -> bytes:
        handle.seek(offset)
        return handle.read(length)

    return read, path.stat().st_size


def _safetensors(read: Callable[[int, int], bytes], label: str) -> Iterator[dict]:
    size = int.from_bytes(read(0, 8), "little")
    if not 2 <= size <= _MAX_HEADER_BYTES:
        raise SourceError(f"{label}: implausible safetensors header length {size}")
    header = json.loads(read(8, size))
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        begin, end = descriptor["data_offsets"]
        yield {
            "name": name,
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "length": int(end) - int(begin),
        }


def _gguf_tensors(read: Callable[[int, int], bytes], size: int) -> Iterator[dict]:
    for tensor in gguf.read_header(read, size).tensors:
        # `shape` is logical row-major (GGUF `ne` reversed) — the same convention
        # the Rust inventory and the contract matcher carry.
        yield {
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "length": 0,
        }


def _inventory(source: object, label: str) -> list[dict]:
    """`source` is a URL, a local path, or a hub blob's chunk list."""

    if isinstance(source, tuple):
        urls, chunk_bytes, total = source
        read, size = _chunked(urls, chunk_bytes), total
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        read, size = _ranged(source), None
    elif isinstance(source, str):
        read, size = _local(Path(source))
    else:
        raise SourceError(f"{label}: unreadable source {source!r}")
    if label.endswith(".gguf"):
        if size is None:
            size = _size(str(source))
        return list(_gguf_tensors(read, size))
    return list(_safetensors(read, label))


def _hf_tree(repo: str, rev: str, path: str) -> list[dict]:
    url = f"{_HF}/api/models/{repo}/tree/{rev}/{path}".rstrip("/")
    return json.loads(_get(url, {}))


def _recast(identifier: str, dtype: str) -> list[dict]:
    """A banked set, re-typed. THE ONE SYNTHETIC SOURCE, and it exists for one
    reason: the SDXL-inpainting hole is only visible against a BF16 packaging.
    Upstream ships the inpainting UNet in fp32, so v1's refusal there is a DTYPE
    refusal that hides the real one; re-typed to bf16 it is the case v1 legally
    ADMITS while the tensor set says [320, 9, 3, 3] where SDXL says [320, 4, 3,
    3] — the packaging tensorhub would publish, and the one v2 has to refuse.
    Names, shapes and member layout are the real header's, unmodified."""

    source = json.loads(_read_banked(_OUT / f"{identifier}.json.gz"))
    for member in source["files"]:
        for tensor in member["tensors"]:
            tensor["dtype"] = dtype
    return source["files"]


def _read_banked(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def _hub_resolve(ref: str) -> list[tuple[str, object]]:
    """`hub:<org>/<repo>@<release>` — a checkpoint TENSORHUB PRODUCED.

    Not every reference checkpoint has an upstream. `tensorhub/rife-4.25` is
    written pod-side by `cozy_rife.save_artifact` out of Practical-RIFE's
    `flownet.pkl`; the pickle is upstream, the safetensors TREE is ours and
    exists nowhere else. Deriving its topology from the producing module's
    state_dict would make the corpus's one hand-typed record — the exact thing
    v2 exists to stop — so the bytes the hub serves are the source, read the
    same way every other source is: a signed URL and a ranged header read.

    `COZY_HUB` (default the standing stack) and `COZY_HUB_TOKEN`; mint one with
    scripts/live-hub.sh's login call.
    """

    body, _, selector = ref[len("hub:"):].partition("@")
    org, _, name = body.partition("/")
    if not org or not name or not selector:
        raise SourceError(f"{ref}: expected hub:<org>/<repo>@<release-or-sha256:digest>")
    hub = os.environ.get("COZY_HUB", "http://127.0.0.1:31550").rstrip("/")
    token = os.environ.get("COZY_HUB_TOKEN", "")
    if not token:
        raise SourceError(f"{ref}: COZY_HUB_TOKEN is unset — a hub source is not public")
    # A release NAME moves; a checkpoint DIGEST does not. Pin the digest when a
    # specific packaging is the point — the reference for a topology must not
    # change under it the next time somebody cuts a release.
    key = "digest" if selector.startswith("sha256:") else "release"
    url = f"{hub}/api/v1/repos/{org}/{name}/resolve?{key}={urllib.parse.quote(selector)}"
    request = urllib.request.Request(
        url, headers={**_UA, "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            resolved = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SourceError(f"{url}: HTTP {error.code}") from None
    except OSError as error:
        raise SourceError(f"{url}: {error}") from None
    members: list[tuple[str, object]] = []
    for entry in resolved.get("files", []):
        path = str(entry.get("path", ""))
        if not path.endswith((".safetensors", ".gguf")):
            continue
        if "url" in entry:
            members.append((path, str(entry["url"])))
            continue
        if "chunk_urls" not in entry:
            raise SourceError(f"{ref}: {path} resolves to neither a url nor chunks")
        members.append((path, ([str(one) for one in entry["chunk_urls"]],
                               int(entry["chunk_size_bytes"]), int(entry["size_bytes"]))))
    if not members:
        raise SourceError(f"{ref}: no .safetensors/.gguf in the resolved checkpoint")
    return sorted(members, key=lambda member: member[0])


def _check_complete(ref: str, note: str) -> None:
    """REFUSE an hf source that covers a strict subset of the repo's tensor-
    bearing SUBFOLDERS, unless the row says `SUB-NETWORK` on purpose.

    This is the tensorfs#153 guard, and it runs HERE because completeness is a
    fact about the source of truth, which no offline test can see: seven
    topologies were extracted from `hf:…:unet`-shaped refs, every offline gate
    re-derived them perfectly from their own (equally partial) banks, and
    production stamped 12 of 501. Root-level tensor files are alternate
    single-file packagings by convention (FLUX's `flux1-dev.safetensors`,
    krea's `raw.safetensors`) and do not count against a subfolder tree; a
    subfolder full of tensors that the ref does not name DOES.
    """

    declared: set[str] = set()
    repos: set[tuple[str, str]] = set()
    for one in ref.split(","):
        if not one.startswith("hf:"):
            return  # hub: resolves the whole checkpoint; local: is unverifiable
        repo_rev, _, path = one[3:].partition(":")
        repo, _, rev = repo_rev.partition("@")
        repos.add((repo, rev or "main"))
        top = path.split("/", 1)[0] if path else ""
        declared.add(top)
    if "SUB-NETWORK" in note:
        return
    for repo, rev in sorted(repos):
        url = f"{_HF}/api/models/{repo}/tree/{rev}?recursive=true"
        entries = json.loads(_get(url, {}))
        tensorDirs = {
            str(entry["path"]).split("/", 1)[0]
            for entry in entries
            if entry.get("type") == "file"
            and str(entry["path"]).endswith((".safetensors", ".gguf"))
            and "/" in str(entry["path"])
        }
        undeclared = sorted(
            directory for directory in tensorDirs - declared
            # A named exclusion is a decision, not an omission: the note may
            # say `EXCLUDES vae_1_0 (…)` for an alternate component packaging
            # that shares the repo — the check then holds for everything the
            # note does NOT own up to.
            if not ("EXCLUDES" in note and directory in note)
        )
        if undeclared:
            raise SourceError(
                f"{ref}: INCOMPLETE — {repo} carries tensor members under "
                f"{undeclared} that this ref does not name. A topology is the "
                "WHOLE published tree (tensorfs#153); either add the missing "
                "subfolders or mark the row SUB-NETWORK with the reason it is "
                "deliberately partial")


def _resolve(ref: str) -> list[tuple[str, object]]:
    """A ref to [(member path, readable url-or-path)]."""

    if ref.startswith("local:"):
        root = Path(ref[6:])
        if root.is_file():
            return [(root.name, str(root))]
        members = sorted(root.rglob("*.safetensors")) + sorted(root.rglob("*.gguf"))
        if not members:
            raise SourceError(f"{ref}: no tensor members under {root}")
        return [(str(member.relative_to(root)), str(member)) for member in members]
    if ref.startswith(("http://", "https://")):
        return [(ref.rsplit("/", 1)[-1], ref)]
    if ref.startswith("hub:"):
        return _hub_resolve(ref)
    if not ref.startswith("hf:"):
        raise SourceError(f"unrecognised ref {ref!r}")
    repo_rev, _, path = ref[3:].partition(":")
    repo, _, rev = repo_rev.partition("@")
    rev = rev or "main"
    if path and Path(path).suffix in {".safetensors", ".gguf"}:
        return [(path, f"{_HF}/{repo}/resolve/{rev}/{path}")]
    entries = _hf_tree(repo, rev, path)
    files = sorted(
        str(entry["path"])
        for entry in entries
        if entry.get("type") == "file" and str(entry["path"]).endswith((".safetensors", ".gguf"))
    )
    if not files:
        raise SourceError(f"{ref}: no .safetensors/.gguf under {path or '<root>'}")
    return [(name, f"{_HF}/{repo}/resolve/{rev}/{name}") for name in files]


def _sources() -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    for line in (_OUT / "SOURCES.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        identifier, ref, note, member_filter = (line.split("\t") + ["", "", ""])[:4]
        out[identifier] = (ref, note, member_filter)
    return out


def _keep(path: str, member_filter: str) -> bool:
    """A repo ships several PACKAGINGS side by side (`.fp16.`, `.non_ema.`); a
    checkpoint is one of them. The filter says which, because banking all of
    them as one tree describes a checkpoint nobody serves."""

    if not member_filter:
        return True
    if member_filter.startswith("!"):
        return not any(token and token in path for token in member_filter[1:].split("|"))
    return any(token and token in path for token in member_filter.split("|"))


def main() -> int:
    wanted = set(sys.argv[1:])
    _OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for identifier, (ref, note, member_filter) in _sources().items():
        if wanted and identifier not in wanted:
            continue
        try:
            if ref.startswith("recast:"):
                banked, dtype = ref[len("recast:"):].split(":", 1)
                files = _recast(banked, dtype)
            else:
                _check_complete(ref, note)
                members = [member for one in ref.split(",") for member in _resolve(one)]
                members = [member for member in members if _keep(member[0], member_filter)]
                if not members:
                    raise SourceError(f"filter {member_filter!r} kept no member")
                files = []
                for path, source in members:
                    files.append({"path": path, "tensors": _inventory(source, path)})
            total = sum(len(entry["tensors"]) for entry in files)
        except SourceError as error:
            print(f"[{identifier}] UNREACHABLE: {error}", file=sys.stderr)
            failures.append(identifier)
            continue
        document = {"id": identifier, "source": ref, "note": note, "files": files}
        target = _OUT / f"{identifier}.json.gz"
        payload = (json.dumps(document, separators=(",", ":")) + "\n").encode("utf-8")
        # mtime=0 so re-running the bank produces byte-identical output and a
        # no-op diff rather than a noisy one.
        with target.open("wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as handle:
            handle.write(payload)
        print(
            f"[{identifier}] {len(files)} member(s), {total} tensors "
            f"-> {target.relative_to(_ROOT)} ({target.stat().st_size // 1024} KiB)",
            file=sys.stderr,
        )
    if failures:
        print(f"UNREACHABLE: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
