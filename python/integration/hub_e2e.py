"""Live end-to-end proof of TensorFS against a real Tensorhub.

Opt-in, never a CI gate. Requires a running hub and credentials:

    TENSORFS_HUB_URL      hub base URL (default http://127.0.0.1:31550)
    TENSORFS_HUB_LOGIN    password-login user
    TENSORFS_HUB_PASSWORD password-login password
    TENSORFS_E2E_DIR      working directory (default: a fresh temp dir)

Phases, each independently asserted:

  A  first publish: ingest -> declare -> granted uploads -> complete -> promoted
  B  dedup republish: 8 changed bytes re-upload only the intersected chunks
  C  fetch: fresh empty CAS -> resolve -> download -> byte-exact reconstruction;
     a second download transfers zero bytes (resident objects short-circuit)
  D  local read/write: verified reads, a local edit, and a third publish
     reusing every unchanged chunk
  E  robustness: an uploader SIGKILLed mid-transfer resumes from `staged`;
     a downloader SIGKILLed mid-transfer resumes from verified residency

The script prints one JSON evidence block and exits non-zero on any failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfs import LocalCAS, RepositoryManifest, TransferGrant, download, read_entry, upload

HUB = os.environ.get("TENSORFS_HUB_URL", "http://127.0.0.1:31550").rstrip("/")
LOGIN = os.environ.get("TENSORFS_HUB_LOGIN", "")
PASSWORD = os.environ.get("TENSORFS_HUB_PASSWORD", "")

EVIDENCE: dict[str, Any] = {"hub": HUB, "phases": {}}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    print(json.dumps(EVIDENCE, indent=2, default=str))
    raise SystemExit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)
    print(f"  ok: {message}")


class Hub:
    """Minimal typed client for the hub's v2 publish and resolve routes."""

    def __init__(self) -> None:
        self._token = ""
        self._token_at = 0.0

    def _headers(self) -> dict[str, str]:
        # Password-login tokens live ~15 minutes; refresh with a plain
        # re-login (never the shared refresh token).
        if not self._token or time.time() - self._token_at > 600:
            body = json.dumps({"login": LOGIN, "password": PASSWORD}).encode()
            raw = self._raw("POST", "/api/v1/password/login", body, auth=False)
            self._token = str(raw["access_token"])
            self._token_at = time.time()
        return {"Authorization": f"Bearer {self._token}"}

    def _raw(
        self, method: str, path: str, body: bytes | None, *, auth: bool = True
    ) -> dict[str, Any]:
        code, payload = self._attempt(method, path, body, auth=auth)
        if code >= 300:
            fail(f"{method} {path} -> HTTP {code}: {json.dumps(payload)[:400]}")
        return payload

    def _attempt(
        self, method: str, path: str, body: bytes | None, *, auth: bool = True
    ) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(HUB + path, data=body, method=method)
        request.add_header("content-type", "application/json")
        if auth:
            for key, value in self._headers().items():
                request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload, code = response.read(), response.status
        except urllib.error.HTTPError as error:
            payload, code = error.read(), error.code
        try:
            decoded = json.loads(payload) if payload else {}
        except ValueError:
            decoded = {"raw": payload.decode(errors="replace")[:400]}
        return code, decoded

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._raw("POST", path, json.dumps(body).encode())

    def post_attempt(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return self._attempt("POST", path, json.dumps(body).encode())

    def get(self, path: str) -> dict[str, Any]:
        return self._raw("GET", path, None)


def deterministic_tensor(seed: int, length: int) -> bytes:
    """Pseudo-random bytes that a shared stack cannot already hold."""

    block = hashlib.sha256(f"tensorfs-e2e-{os.getpid()}-{seed}".encode()).digest()
    repeats = length // len(block) + 1
    return (block * repeats)[:length]


def write_safetensors(path: Path, tensors: list[tuple[str, bytes]]) -> None:
    header: dict[str, Any] = {}
    offset = 0
    for name, data in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as sink:
        sink.write(struct.pack("<Q", len(raw)))
        sink.write(raw)
        for _, data in tensors:
            sink.write(data)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def tree_digests(root: Path) -> dict[str, str]:
    return {
        str(item.relative_to(root)): sha256_file(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def manifest_digests(cas: LocalCAS, manifest: RepositoryManifest) -> dict[str, str]:
    """Every file's digest reconstructed from CAS objects, with no tree on disk."""

    return {
        entry.path: hashlib.sha256(read_entry(cas, entry)).hexdigest()
        for entry in manifest.files
    }


def write_tree(cas: LocalCAS, manifest: RepositoryManifest, root: Path) -> Path:
    """Write a working tree this script intends to edit in place.

    TensorFS deliberately owns no materializer any more: a consumer that wants
    bytes on disk asks for the bytes and writes them. Phase D needs a real
    editable directory, so it does exactly that -- and nothing in the read path
    depends on it.
    """

    if root.exists():
        shutil.rmtree(root)
    for entry in manifest.files:
        target = root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_entry(cas, entry))
    return root


def upload_grants(need: list[dict[str, Any]]) -> list[TransferGrant]:
    # The hub keeps its established `put_url` spelling; TensorFS grants are
    # URL-generic, so the one consumer mapping is put_url -> url.
    return [
        TransferGrant(
            row["digest"],
            row["size_bytes"],
            row["put_url"],
            staging_key=row["staging_key"],
            headers=row.get("headers") or {},
            expires_at=row.get("expires_at"),
        )
        for row in need
    ]


def tagged(digest: str) -> str:
    # Resolve emits bare hex for chunk digests; TensorFS refs are strictly
    # algorithm-tagged, so the client seam restores the tag.
    return digest if ":" in digest else f"sha256:{digest}"


def resolve_grants(resolved: dict[str, Any]) -> list[TransferGrant]:
    grants: dict[str, TransferGrant] = {}
    for file in resolved["files"]:
        chunks = file.get("chunks") or []
        urls = file.get("chunk_urls") or []
        if chunks:
            check(len(chunks) == len(urls), f"{file['path']}: chunk_urls align with chunks")
            for chunk, url in zip(chunks, urls, strict=True):
                grants.setdefault(
                    tagged(chunk["digest"]),
                    TransferGrant(tagged(chunk["digest"]), chunk["len"], url),
                )
        else:
            grants.setdefault(
                tagged(file["digest"]),
                TransferGrant(tagged(file["digest"]), file["size_bytes"], file["url"]),
            )
    return list(grants.values())


def manifest_from_resolve(resolved: dict[str, Any]) -> RepositoryManifest:
    files = []
    for file in resolved["files"]:
        row: dict[str, Any] = {
            "path": file["path"],
            "size_bytes": file["size_bytes"],
            "digest": tagged(file["digest"]),
        }
        if file.get("chunks"):
            row["chunks"] = [
                {"digest": tagged(chunk["digest"]), "len": chunk["len"]}
                for chunk in file["chunks"]
            ]
        files.append(row)
    return RepositoryManifest.from_dict({"format": 1, "files": files})


def declare(hub: Hub, org: str, repo: str, manifest: RepositoryManifest) -> dict[str, Any]:
    # th#1400: a v2 publish mints a fresh checkpoint identity and inherits no
    # tags; every re-publish restates `latest` so resolve tracks the newest.
    files = [entry.to_dict() for entry in manifest.files]
    return hub.post(
        f"/api/v1/repos/{org}/{repo}/publishes",
        {"files": files, "tags": [{"tag": "latest"}]},
    )


def complete(hub: Hub, org: str, repo: str, publish_id: str) -> dict[str, Any]:
    """Drive `complete` to a terminal stage.

    Promotion is incremental: the hub answers 409 `promote_incomplete`
    (retryable by contract) until every staged object is promoted, so the
    client re-drives the same call. Only a terminal failure code stops it.
    """

    path = f"/api/v1/repos/{org}/{repo}/publishes/{publish_id}/complete"
    deadline = time.time() + 600
    attempts = 0
    while True:
        attempts += 1
        code, result = hub.post_attempt(path, {})
        status = result.get("status") if isinstance(result.get("status"), dict) else result
        stage = str(status.get("stage", ""))
        failure = status.get("failure") or {}
        if stage == "promoted":
            print(f"  promoted after {attempts} complete call(s)")
            return status
        if stage in {"repudiated", "expired"}:
            fail(f"publish {publish_id} terminal stage {stage}: {json.dumps(failure)[:300]}")
        if failure and not failure.get("retryable", False):
            fail(f"publish {publish_id} terminal failure: {json.dumps(failure)[:300]}")
        if time.time() > deadline:
            fail(f"publish {publish_id} not promoted in time (stage={stage}, HTTP {code})")
        time.sleep(2)


def killed_child(function: str, arguments: list[str], after_s: float) -> int:
    """Run a transfer in a child that a timer SIGKILLs mid-byte-stream.

    A progress-callback kill only fires between whole objects; the timer lands
    inside one, so the resume genuinely starts from a torn transfer.
    """

    code = (
        "import os, sys, threading, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "import json\n"
        "from tensorfs import LocalCAS, TransferGrant, upload, download\n"
        "grants = [TransferGrant(**row) for row in json.loads(Path(sys.argv[3]).read_text())]\n"
        "cas = LocalCAS(sys.argv[2])\n"
        f"delay = {after_s}\n"
        "threading.Thread(\n"
        "    target=lambda: (time.sleep(delay), os.kill(os.getpid(), 9)), daemon=True\n"
        ").start()\n"
        f"{function}(grants, cas, parallel=1)\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", code, *arguments],
        capture_output=True,
        timeout=600,
        check=False,
    )
    return child.returncode


def main() -> None:
    if not LOGIN or not PASSWORD:
        print("skipping: set TENSORFS_HUB_LOGIN and TENSORFS_HUB_PASSWORD")
        return

    base = Path(os.environ.get("TENSORFS_E2E_DIR", "") or tempfile.mkdtemp(prefix="tfs-e2e-"))
    base.mkdir(parents=True, exist_ok=True)
    source = base / "source"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)

    hub = Hub()
    repo = f"tensorfs-e2e-{int(time.time())}"
    created = hub.post("/api/v1/repos", {"name": repo})
    org = str(created.get("org") or created.get("org_slug") or LOGIN)
    EVIDENCE["repo"] = f"{org}/{repo}"
    print(f"repo: {org}/{repo}")

    mib = 1024 * 1024
    # The run nonce rides the tensor NAMES so even the header chunks are
    # unique per run; a shared long-lived hub keeps every prior run's objects.
    nonce = f"r{os.getpid()}"
    write_safetensors(
        source / "model-00001.safetensors",
        [
            (f"blk.{nonce}.0.attn.weight", deterministic_tensor(1, 100 * mib)),
            (f"blk.{nonce}.0.attn.bias", deterministic_tensor(2, 4096)),
            (f"blk.{nonce}.0.ffn.weight", deterministic_tensor(3, 40 * mib)),
        ],
    )
    write_safetensors(
        source / "model-00002.safetensors",
        [
            (f"blk.{nonce}.1.attn.weight", deterministic_tensor(4, 70 * mib)),
            (f"blk.{nonce}.1.norm.weight", deterministic_tensor(5, 8192)),
        ],
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "tensorfs-e2e", "run": os.getpid()}) + "\n"
    )
    source_digests = tree_digests(source)

    # ---- Phase A: first publish -------------------------------------------
    print("phase A: first publish")
    cas_a = LocalCAS(base / "cas-publisher")
    started = time.time()
    manifest = cas_a.ingest_repository(source)
    declared = declare(hub, org, repo, manifest)
    check(declared["resident_objects"] == 0, "fresh content: nothing resident before upload")
    need = declared["need"]
    check(len(need) == declared["distinct_objects"], "every distinct object is granted")
    report = upload(upload_grants(need), cas_a)
    check(report.ok, f"upload clean: {report}")
    complete(hub, org, repo, declared["publish_id"])
    EVIDENCE["phases"]["A"] = {
        "declared_bytes": declared["declared_bytes"],
        "distinct_objects": declared["distinct_objects"],
        "granted": len(need),
        "uploaded_objects": report.succeeded,
        "uploaded_bytes": report.bytes_transferred,
        "wall_s": round(time.time() - started, 2),
    }

    # ---- Phase B: dedup republish -----------------------------------------
    print("phase B: 8-byte edit republished")
    target = source / "model-00001.safetensors"
    with target.open("r+b") as handle:
        handle.seek(8 + 200 + 80 * mib)  # inside blk.0.attn.weight, chunk 2
        handle.write(b"EDITED!!")
    objects_before = {str(ref) for entry in manifest.files for ref, _ in entry.objects()}
    manifest_b = cas_a.ingest_repository(source)
    objects_after = {str(ref) for entry in manifest_b.files for ref, _ in entry.objects()}
    new_objects = objects_after - objects_before
    declared_b = declare(hub, org, repo, manifest_b)
    check(
        len(declared_b["need"]) == len(new_objects),
        f"hub grants exactly the {len(new_objects)} locally new objects "
        f"(need={len(declared_b['need'])})",
    )
    check(
        declared_b["resident_objects"] == declared_b["distinct_objects"] - len(new_objects),
        "every unchanged chunk is already resident on the hub",
    )
    report_b = upload(upload_grants(declared_b["need"]), cas_a)
    check(report_b.ok, f"delta upload clean: {report_b}")
    complete(hub, org, repo, declared_b["publish_id"])
    EVIDENCE["phases"]["B"] = {
        "declared_bytes": declared_b["declared_bytes"],
        "distinct_objects": declared_b["distinct_objects"],
        "resident_objects": declared_b["resident_objects"],
        "delta_objects": len(declared_b["need"]),
        "delta_bytes": report_b.bytes_transferred,
        "dedup_ratio": round(
            1 - report_b.bytes_transferred / declared_b["declared_bytes"], 6
        ),
    }
    source_digests = tree_digests(source)

    # ---- Phase C: fetch into a fresh CAS ----------------------------------
    print("phase C: fetch into a fresh empty CAS")
    resolved = hub.get(f"/api/v1/repos/{org}/{repo}/resolve?tag=latest")
    grants_c = resolve_grants(resolved)
    cas_c = LocalCAS(base / "cas-fetcher")
    report_c = download(grants_c, cas_c)
    check(report_c.ok, f"download clean: {report_c}")
    check(report_c.succeeded == len(grants_c), "empty CAS fetched every object")
    remote_manifest = manifest_from_resolve(resolved)
    check(
        manifest_digests(cas_c, remote_manifest) == source_digests,
        "every fetched file reconstructs byte-exactly from CAS objects",
    )
    report_c2 = download(grants_c, cas_c)
    check(
        report_c2.succeeded == 0 and report_c2.bytes_transferred == 0,
        "second fetch transfers zero bytes (verified residency short-circuits)",
    )
    EVIDENCE["phases"]["C"] = {
        "objects": len(grants_c),
        "first_fetch_bytes": report_c.bytes_transferred,
        "second_fetch_bytes": report_c2.bytes_transferred,
    }

    # ---- Phase D: local read/write and third publish ----------------------
    print("phase D: local reads, local edit, third publish")
    sample = remote_manifest.files[0]
    for ref, size in sample.objects()[:3]:
        path = cas_c.verify_object(ref, size=size)
        check(path.is_file(), f"verified local read of {str(ref)[:23]}...")
    fetched = write_tree(cas_c, remote_manifest, base / "fetched")
    edited = fetched / "model-00002.safetensors"
    with edited.open("r+b") as handle:
        handle.seek(-16, os.SEEK_END)  # inside blk.1.norm.weight (small tensor)
        handle.write(b"LOCAL-WRITE-EDIT")
    manifest_d = cas_c.ingest_repository(fetched)
    declared_d = declare(hub, org, repo, manifest_d)
    check(
        0 < len(declared_d["need"]) <= 2,
        f"local edit re-uploads at most its packed chunk (need={len(declared_d['need'])})",
    )
    report_d = upload(upload_grants(declared_d["need"]), cas_c)
    check(report_d.ok, f"third publish delta clean: {report_d}")
    complete(hub, org, repo, declared_d["publish_id"])
    EVIDENCE["phases"]["D"] = {
        "delta_objects": len(declared_d["need"]),
        "delta_bytes": report_d.bytes_transferred,
    }

    # ---- Phase E: kill/resume robustness ----------------------------------
    print("phase E: SIGKILL mid-transfer, then resume")
    (source / "extra-00003.safetensors").write_bytes(b"")  # placeholder removed below
    (source / "extra-00003.safetensors").unlink()
    write_safetensors(
        source / "extra-00003.safetensors",
        [
            (f"resume.{nonce}.a", deterministic_tensor(6, 66 * mib)),
            (f"resume.{nonce}.b", deterministic_tensor(7, 66 * mib)),
        ],
    )
    manifest_e = cas_a.ingest_repository(source)
    declared_e = declare(hub, org, repo, manifest_e)
    need_e = declared_e["need"]
    check(len(need_e) >= 3, f"resume fixture adds several new objects ({len(need_e)})")
    grant_rows = json.dumps(
        [
            {
                "digest": row["digest"],
                "size_bytes": row["size_bytes"],
                "url": row["put_url"],
                "staging_key": row["staging_key"],
                "headers": row.get("headers") or {},
                "expires_at": row.get("expires_at"),
            }
            for row in need_e
        ]
    )
    grants_path = base / "grants-e.json"
    grants_path.write_text(grant_rows)
    code = killed_child(
        "upload",
        [str(Path(__file__).resolve().parents[1] / "src"), str(cas_a.root), str(grants_path)],
        after_s=1.2,
    )
    check(code == -9, f"uploader died by SIGKILL mid-transfer (rc={code})")
    # Resume rides the SAME publish session: staging keys are per-publish,
    # so only the grants route can see what the killed run already staged.
    redeclared = hub.post(
        f"/api/v1/repos/{org}/{repo}/publishes/{declared_e['publish_id']}/grants", {}
    )
    staged_after_kill = len(redeclared["staged"])
    check(
        staged_after_kill + len(redeclared["need"]) == len(need_e),
        f"every object is exactly staged ({staged_after_kill}) or still needed "
        f"({len(redeclared['need'])}) — none lost, none duplicated",
    )
    check(
        len(redeclared["need"]) >= 1,
        "the mid-stream kill left at least one object genuinely untransferred",
    )
    report_e = upload(upload_grants(redeclared["need"]), cas_a)
    check(report_e.ok, f"resumed upload clean: {report_e}")
    complete(hub, org, repo, declared_e["publish_id"])

    resolved_e = hub.get(f"/api/v1/repos/{org}/{repo}/resolve?tag=latest")
    grants_e2 = resolve_grants(resolved_e)
    cas_e = LocalCAS(base / "cas-resume-fetch")
    kill_rows = json.dumps(
        [
            {"digest": str(grant.digest), "size_bytes": grant.size_bytes, "url": grant.url}
            for grant in grants_e2
        ]
    )
    grants_path.write_text(kill_rows)
    code = killed_child(
        "download",
        [str(Path(__file__).resolve().parents[1] / "src"), str(cas_e.root), str(grants_path)],
        after_s=1.2,
    )
    check(code == -9, f"downloader died by SIGKILL mid-transfer (rc={code})")
    partial = sum(
        1 for grant in grants_e2 if cas_e.contains(grant.digest, size=grant.size_bytes)
    )
    check(
        partial < len(grants_e2),
        f"the kill left {len(grants_e2) - partial} objects genuinely unfetched",
    )
    report_e2 = download(grants_e2, cas_e)
    check(report_e2.ok, "resumed download clean")
    check(
        report_e2.skipped_resident == partial
        and report_e2.succeeded == len(grants_e2) - partial,
        f"resume skipped exactly the {partial} resident objects and fetched the rest",
    )
    remote_manifest_e = manifest_from_resolve(resolved_e)
    check(
        manifest_digests(cas_e, remote_manifest_e) == tree_digests(source),
        "post-resume reconstruction is byte-exact",
    )
    EVIDENCE["phases"]["E"] = {
        "upload_staged_after_kill": staged_after_kill,
        "upload_resumed_objects": len(redeclared["need"]),
        "upload_resumed_bytes": report_e.bytes_transferred,
        "download_resident_after_kill": partial,
        "download_resumed_fetched": report_e2.succeeded,
        "download_resumed_skipped": report_e2.skipped_resident,
    }

    print(json.dumps(EVIDENCE, indent=2, default=str))
    print("ALL PHASES PASSED")


if __name__ == "__main__":
    main()
