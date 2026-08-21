"""Layout contracts as objects: the actual layout, imported and passed around.

A lane declaration is not a name or an enum pointing at a layout — it IS the
layout: ``lanes=(tensorfs.contracts.SDXL_DIFFUSERS_BF16, ...)``. The
predefined constants (:mod:`tensorfs.contracts`) are typed helpers over the
curated library; an author who needs a layout the library does not ship
constructs one inline::

    my_layout = Contract(
        dtype="bfloat16",
        tensors=[
            TensorDecl(
                role="blocks.{i}.attn.qkv",
                pattern="blocks.{i}.attn.qkv_proj.weight",
                rank=2,
                fusion=Fusion(parts=[("q", 1), ("k", 1), ("v", 1)]),
            ),
        ],
    )

Custom contracts are ANONYMOUS: there is deliberately no ``name=`` — identity
is the content digest (``sha256:<hex>``), because a free-text name on an
inline object validates nothing and can lie or collide. The author's variable
name is their label; platform surfaces spell the digest. ``name@version``
survives only for the curated library, where CI digest-pinning makes the name
a real promise.

Construction serializes to a v1 document and runs THE Rust validator, so a
malformed contract refuses at author-module import time with the Rust error,
never at deploy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .native import contract_info, contract_recipe, contract_verdict

__all__ = [
    "Contract",
    "Fusion",
    "MissingDtype",
    "Permute",
    "TensorDecl",
]


class MissingDtype(ValueError):
    """A lane read asked for a load dtype the contract does not declare."""


@dataclass(frozen=True, slots=True)
class Fusion:
    """A fusion along the outermost axis: ``groups`` repetitions of the
    ordered ``parts`` cycle, each part ``(role_suffix, share)``."""

    parts: Sequence[tuple[str, int]]
    groups: int = 1

    def _raw(self) -> dict[str, Any]:
        return {
            "axis": 0,
            **({"groups": self.groups} if self.groups != 1 else {}),
            "parts": [{"role": role, "share": share} for role, share in self.parts],
        }


@dataclass(frozen=True, slots=True)
class Permute:
    """A generalized permute: reshape to ``view``, permute those axes,
    reshape back. ``view`` entries are literals, ``"shape[k]"`` /
    ``"shape[k]/n"``, or ``"auto"``."""

    view: Sequence[int | str]
    axes: Sequence[int]

    def _raw(self) -> dict[str, Any]:
        return {"view": list(self.view), "axes": list(self.axes)}


@dataclass(frozen=True, slots=True)
class TensorDecl:
    """One declared tensor family: its spelling-independent role, its spelling
    in the file, and the constraints that make the contract falsifiable."""

    role: str
    pattern: str
    dtypes: Sequence[str] = ()
    rank: int | None = None
    required: bool = True
    fusion: Fusion | None = None
    permute: Permute | None = None

    def _raw(self) -> dict[str, Any]:
        raw: dict[str, Any] = {"role": self.role, "pattern": self.pattern}
        if self.dtypes:
            raw["dtypes"] = list(self.dtypes)
        if self.rank is not None:
            raw["rank"] = self.rank
        if not self.required:
            raw["required"] = False
        if self.fusion is not None:
            raw["fusion"] = self.fusion._raw()
        if self.permute is not None:
            raw["permute"] = self.permute._raw()
        return raw


def _decl_of(raw: Mapping[str, Any]) -> TensorDecl:
    fusion = None
    if "fusion" in raw:
        fusion = Fusion(
            parts=tuple((part["role"], part["share"]) for part in raw["fusion"]["parts"]),
            groups=raw["fusion"].get("groups", 1),
        )
    permute = None
    if "permute" in raw:
        permute = Permute(
            view=tuple(raw["permute"]["view"]),
            axes=tuple(raw["permute"]["axes"]),
        )
    return TensorDecl(
        role=raw["role"],
        pattern=raw["pattern"],
        dtypes=tuple(raw.get("dtypes", ())),
        rank=raw.get("rank"),
        required=raw.get("required", True),
        fusion=fusion,
        permute=permute,
    )


class Contract:
    """One frozen, Rust-validated layout contract.

    The predefined :mod:`tensorfs.contracts` constants are instances of this
    class; ``Contract(...)`` constructs an ANONYMOUS custom (no ``name=`` — a
    custom's identity is its digest). Equality and hashing are by digest.
    """

    _digest: str
    _document: str
    _dtype: str | None
    _name: str | None
    _raw: dict[str, Any]
    _stamp: str
    _version: int | None

    __slots__ = ("_digest", "_document", "_dtype", "_name", "_raw", "_stamp", "_version")

    def __init__(
        self,
        *,
        tensors: Sequence[TensorDecl],
        dtype: str | None = None,
        sets: Mapping[str, Sequence[str]] | None = None,
        description: str = "",
    ) -> None:
        raw: dict[str, Any] = {"format": "tensorfs-contract-v1"}
        if description:
            raw["description"] = description
        if dtype is not None:
            raw["dtype"] = dtype
        raw["tensors"] = [decl._raw() for decl in tensors]
        if sets:
            raw["sets"] = {name: list(patterns) for name, patterns in sets.items()}
        self._seal(raw)

    def _seal(self, raw: dict[str, Any]) -> None:
        """Canonicalize, validate through the Rust parser, and freeze."""

        document = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        info = contract_info(document)  # the Rust refusal propagates verbatim
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_name", info.name)
        object.__setattr__(self, "_version", info.version)
        object.__setattr__(self, "_dtype", info.dtype)
        object.__setattr__(self, "_digest", info.digest)
        object.__setattr__(self, "_stamp", info.stamp)

    def __setattr__(self, attribute: str, value: object) -> None:
        raise AttributeError("a Contract is frozen")

    @classmethod
    def from_document(cls, document: str) -> Contract:
        """A contract from a raw v1 JSON document (library or custom)."""

        contract = cls.__new__(cls)
        contract._seal(json.loads(document))
        return contract

    @classmethod
    def from_file(cls, path: str) -> Contract:
        with open(path, encoding="utf-8") as handle:
            return cls.from_document(handle.read())

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str | None:
        """The library name; ``None`` on customs, which have none."""

        return self._name

    @property
    def version(self) -> int | None:
        return self._version

    @property
    def digest(self) -> str:
        """Bare hex SHA-256 of the canonical rendering — identical to the
        Rust ``Contract::digest``, and a custom's entire identity."""

        return self._digest

    @property
    def stamp(self) -> str:
        """``name@version`` for library documents, ``sha256:<hex>`` for
        customs: what a snapshot records."""

        return self._stamp

    @property
    def label(self) -> str:
        """The human spelling: the name for library entries, a digest prefix
        like ``sha256:ab12cd34…`` for customs."""

        if self._name is not None:
            return self._name
        return f"sha256:{self._digest[:8]}…"

    # -- the bind verdict -------------------------------------------------

    @property
    def recipe(self) -> str:
        """The conversion this lane's bytes are MADE by — ``dtype-cast`` or
        ``fp8-rowwise`` — derived from the declarations and stored nowhere
        (tcg#53). There is no ``recipe`` field and there must never be one."""

        return contract_recipe(self._document)

    def verdict(
        self, files: Mapping[str, Iterable[Sequence[Any]]]
    ) -> Any:  # tensorfs._tensorfs.Verdict
        """Does this checkpoint bind to this lane, and if not, what would fix
        it: ``satisfies`` | ``derivable`` | ``incompatible``.

        ``files`` maps a member path to its header entries, each
        ``(name, dtype, shape)`` or ``(name, dtype, shape, byte_length)``.
        ``dtype`` is the CONTAINER's spelling (``"BF16"``, a ggml type name),
        which is what the declarations list. An omitted byte length is
        "not supplied" and skips the byte half of the fusion rule rather than
        manufacturing a refusal out of an absent number.

        THE SAME RULE THE GO BIND GATE APPLIES, not a Python approximation of
        it: both call one implementation (``crates/tensorfs-core/src/verdict.rs``),
        and ``scripts/prove-verdict-parity.sh`` runs both over one corpus and
        diffs the renderings. A second implementation of an admit decision is a
        second chance to admit a bind that should have been refused, which
        stays invisible until a pod 500s.
        """

        members = [
            (
                path,
                [
                    (
                        str(entry[0]),
                        str(entry[1]),
                        [int(dimension) for dimension in entry[2]],
                        int(entry[3]) if len(entry) > 3 else 0,
                    )
                    for entry in tensors
                ],
            )
            for path, tensors in files.items()
        ]
        return contract_verdict(self._document, members)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contract):
            return NotImplemented
        return self._digest == other._digest

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        return f"Contract({self.label!r})"

    # -- the document -----------------------------------------------------

    @property
    def document(self) -> str:
        """The canonical JSON serialization: what travels in a release
        derive document, and what every ``_tensorfs`` entrypoint accepts."""

        return self._document

    @property
    def description(self) -> str:
        return str(self._raw.get("description", ""))

    @property
    def tensors(self) -> tuple[TensorDecl, ...]:
        return tuple(_decl_of(raw) for raw in self._raw["tensors"])

    @property
    def sets(self) -> dict[str, tuple[str, ...]]:
        return {
            name: tuple(patterns) for name, patterns in self._raw.get("sets", {}).items()
        }

    # -- the serve-side read ----------------------------------------------

    @property
    def dtype(self) -> str:
        """The declared load dtype (``ctx.lane.dtype`` reads this). Reading
        it on a contract that declares none is an author error, refused
        loudly rather than answered with a guess."""

        if self._dtype is None:
            raise MissingDtype(
                f"contract {self.label} declares no top-level dtype; "
                "declare one on the lane contract to read it"
            )
        return self._dtype

    @property
    def torch_dtype(self) -> Any:
        """``self.dtype`` resolved against torch, imported lazily."""

        import importlib

        torch = importlib.import_module("torch")
        name = self.dtype
        resolved = getattr(torch, name, None)
        if resolved is None:
            raise MissingDtype(f"torch has no dtype named {name!r}")
        return resolved
