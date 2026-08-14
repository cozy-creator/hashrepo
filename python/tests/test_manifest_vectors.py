from __future__ import annotations

import json
from pathlib import Path

import pytest
from hashrepo import CASRef, RepositoryManifest

ROOT = Path(__file__).parents[2]
VECTORS = ROOT / "spec" / "v1" / "vectors"


def test_python_reproduces_the_shared_canonical_manifest() -> None:
    data = (VECTORS / "manifest.json").read_bytes()
    manifest = RepositoryManifest.from_bytes(data)
    assert manifest.canonical_bytes() == data.strip()
    assert [entry.path for entry in manifest.files] == [
        "empty.bin",
        "large.bin",
        "text/hello.txt",
        "unicode/<&☃\u2028.txt",
    ]


def test_every_shared_invalid_manifest_is_refused() -> None:
    vectors = json.loads((VECTORS / "invalid.json").read_text())
    for vector in vectors:
        with pytest.raises((TypeError, ValueError), match=".+"):
            RepositoryManifest.from_dict(vector["manifest"])


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0" * 64,
        "blake3:" + "0" * 64,
        "sha256:short",
        "sha256:" + "g" * 64,
    ],
)
def test_v1_refuses_every_non_sha256_or_untagged_ref(value: str) -> None:
    with pytest.raises(ValueError):
        CASRef.parse(value)


def test_ref_is_canonical_and_has_a_portable_object_key() -> None:
    parsed = CASRef.parse(" SHA256:" + "AB" * 32 + " ")
    assert str(parsed) == "sha256:" + "ab" * 32
    assert parsed.object_key() == "objects/sha256/ab/ab/" + "ab" * 32
