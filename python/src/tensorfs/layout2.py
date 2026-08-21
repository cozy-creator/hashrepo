"""Read the v2 expected header Go computed. NOT a matcher, NOT an evaluator.

A v2 LAYOUT is ``quant(topology)``: a finite map ``{key -> dtypes, shape}`` per
component, computed by the Go engine (``engine.go``) and never stored. The hub
links that package and owns the verdict; everything else CONSUMES the layout it
was handed. So this module parses and it accesses, and that is the whole of it.

There is deliberately no ``matches()``, no ``verdict()``, no pattern grammar and
no eligibility predicate here. v1 shipped three copies of one matching rule
(Rust, Go, and a port inside ``convert.py``), and the copies were free to
disagree about an admit decision — a divergence that stays invisible until a pod
serves the wrong bytes. v2 has one evaluator. Adding a second one here, in any
form, undoes the cut.

To obtain a layout::

    go run ./scripts/compute_layout 'sdxl.diffusers@1+cozy.fp8-rowwise@1'

and hand what it prints to :meth:`ExpectedHeader.from_document`.

:func:`rules` reads the vendored quant-rule corpus for IDENTITY FACTS ONLY —
handle, declared dtype, capability floor, conventions. It is what replaces
gen-worker's ``KNOWN_CONTRACTS`` table plus its ``DTYPE_MIN_SM`` lookup
(pgw#1621): the sm floor is a property of the RULE, not of a dtype spelling, and
a table keyed on the spelling silently loses the floor when a lane is spelled
differently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Arrangement",
    "ExpectedHeader",
    "LayoutTensor",
    "Quant",
    "SubAxis",
    "identity_arrangement",
    "layouts",
    "rules",
    "topologies",
]


@dataclass(frozen=True, slots=True)
class LayoutTensor:
    """One key of the computed header: what it may be, and what shape it is."""

    dtypes: tuple[str, ...]
    """Container spellings, most-canonical first. Several entries mean the rule
    is REFERENCE-TOLERANT: the reference packaging itself shipped this key in a
    wider type (an F32 ``logit_scale`` on a bf16 tree), and refusing it would
    refuse the checkpoint the topology was extracted from."""
    shape: tuple[int, ...]
    optional: bool = False

    def accepts(self, dtype: str) -> bool:
        return dtype in self.dtypes


@dataclass(frozen=True, slots=True)
class Quant:
    """A quant rule's identity, as the layout carries it."""

    handle: str
    """``cozy.fp8-rowwise@1``. The version is part of it: two versions of a rule
    are two rules."""
    declared_dtype: str
    """The torch spelling (``float8_e4m3fn``), which is what a loader wants."""
    capability_floor_sm: int
    conventions: Mapping[str, str]
    """How to READ the bytes — scale granularity, layout, divisor. Prose the
    rule author wrote, not a schema, because a consumer that gets this wrong
    produces plausible numbers rather than an error."""
    lossy: bool
    inverse: str = ""
    """The dequant identity, spelled out. Empty for a rule that transforms
    nothing."""
    transformed: int = 0
    """How many keys the rule actually transformed in THIS layout. Zero for a
    plain rule, and zero from :func:`rules`, which reads the rule alone and has
    no topology to count against."""
    digest: str = ""
    """Go's canonical rule digest, carried in the document. Never recomputed
    here: the canonicalizer lives in Go, and a second one would let two
    spellings of the same rule disagree about its identity."""

    @property
    def family(self) -> str:
        """The handle without its version — what a producer switches a kernel
        on, since a rule's kernel is a property of the family."""

        return self.handle.split("@", 1)[0]


@dataclass(frozen=True, slots=True)
class SubAxis:
    """One factor a layout splits a logical axis into, outermost first.

    ``extent`` is the shape FORMULA as the record writes it (``d0``, ``4``,
    ``ceil(d1/16)``), deliberately not evaluated here. Evaluating it is the
    planner's job and there are already two planners — Go's ``v2doc.go`` and
    Rust's ``Plan`` — held together by a banked vector corpus. A third
    evaluator in Python is the thing this module's own docstring forbids.
    """

    axis: int
    extent: str


@dataclass(frozen=True, slots=True)
class Arrangement:
    """One ratified layout morphism's IDENTITY, read from its record.

    A layout morphism is a named, versioned byte-layout map over a tensor's
    storage: a factorization of the logical axes into ``sub_axes`` and a
    ``permutation`` of those factors giving storage order, outermost first.
    That is the whole of it as data. **The transform is not here** — it lives
    once, in ``tensorfs-core``, with a CPU backend and a device backend behind
    one ``FillSink`` trait. This class exists so a Python consumer can name a
    layout, check that it is ratified, and compare it against whatever the
    consumer itself believes about the arrangement — never so it can apply one.
    """

    handle: str
    """``torch.channels_last-2d@1``. The version is part of it."""
    cls: str
    """``inductor`` or ``endpoint-declared``. The first class is what a
    compiler's own layout pass asks for; the second is quant-rule identity —
    the kernel-library layouts where reading one packaging as another produces
    plausible numbers rather than an error."""
    rank: int | None
    """The tensor rank this arrangement is defined for. ``None`` means every
    rank, which is true only of the identity: row-major is row-major at rank 1
    and at rank 5, so pinning it to a rank would refuse tensors it arranges
    correctly."""
    permutation: tuple[int, ...]
    """Storage order over ``sub_axes``, outermost first. Empty for the identity,
    which permutes nothing."""
    sub_axes: tuple[SubAxis, ...]
    digest: str = ""
    """The record's canonical digest, carried in the document and never
    recomputed here — same rule as :class:`Quant`."""

    @property
    def family(self) -> str:
        return self.handle.split("@", 1)[0]

    @property
    def is_identity(self) -> bool:
        return not self.permutation

    def applies_to(self, rank: int) -> bool:
        """Whether this arrangement is defined at one tensor rank.

        An arrangement that does not apply leaves the tensor row-major — a 1-D
        bias under a ``channels_last`` declaration is contiguous, not an error,
        because the permutation is the identity there.
        """

        return self.rank is None or self.rank == rank


@dataclass(frozen=True, slots=True)
class ExpectedHeader:
    """``quant(topology)``: the header a conforming checkpoint must present."""

    stamp: str
    topology: str
    topology_digest: str
    """Pins the record the layout was computed from. Two hubs that computed
    from different topology records must not be able to call the result the
    same thing."""
    quant: Quant
    components: Mapping[str, Mapping[str, LayoutTensor]]
    roles: Mapping[str, str]
    """component -> ``denoiser`` | ``vae`` | ``text_encoder`` | ... . A rule
    scopes itself by role, so the role is not decoration."""

    @classmethod
    def from_document(cls, document: str | bytes | Mapping[str, Any]) -> ExpectedHeader:
        raw: Mapping[str, Any]
        raw = document if isinstance(document, Mapping) else json.loads(document)
        quant = raw["quant"]
        components: dict[str, Mapping[str, LayoutTensor]] = {}
        roles: dict[str, str] = {}
        for component in raw["components"]:
            name = str(component["name"])
            roles[name] = str(component.get("role", ""))
            components[name] = {
                str(key): LayoutTensor(
                    dtypes=tuple(str(spelling) for spelling in entry["dtypes"]),
                    shape=tuple(int(dimension) for dimension in entry["shape"]),
                    optional=bool(entry.get("optional", False)),
                )
                for key, entry in component["tensors"].items()
            }
        return cls(
            stamp=str(raw["stamp"]),
            topology=str(raw["topology"]),
            topology_digest=str(raw["topology_digest"]),
            quant=Quant(
                handle=str(quant["handle"]),
                declared_dtype=str(quant["declared_dtype"]),
                capability_floor_sm=int(quant["capability_floor_sm"]),
                conventions=dict(quant.get("conventions") or {}),
                lossy=bool(quant["lossy"]),
                inverse=str(quant.get("inverse", "")),
                transformed=int(quant.get("transformed", 0)),
                digest=str(quant.get("digest", "")),
            ),
            components=components,
            roles=roles,
        )

    def component(self, name: str) -> Mapping[str, LayoutTensor]:
        try:
            return self.components[name]
        except KeyError:
            known = ", ".join(sorted(self.components))
            raise KeyError(
                f"{self.stamp} has no component {name!r}; it has: {known}"
            ) from None

    def tensors(self, component: str | None = None) -> Mapping[str, LayoutTensor]:
        """One component's finite map.

        ``component=None`` is only answerable when the layout has exactly one.
        A multi-component checkpoint that guessed here would be v1's
        text_encoder/text_encoder_2 collision again, so it refuses instead.
        """

        if component is not None:
            return self.component(component)
        if len(self.components) == 1:
            return next(iter(self.components.values()))
        known = ", ".join(sorted(self.components))
        raise KeyError(
            f"{self.stamp} has {len(self.components)} components; name one of: {known}"
        )

    def __str__(self) -> str:
        return self.stamp


# -- the vendored corpus ----------------------------------------------------


def _root(root: Path | None) -> Path:
    if root is not None:
        # An unreadable corpus must refuse rather than answer "no rules": an
        # empty result reads as a valid, tiny corpus at every call site.
        if not Path(root).is_dir():
            raise FileNotFoundError(f"{root}: not a spec/v2 corpus directory")
        return Path(root)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "spec" / "v2"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "no spec/v2 corpus above tensorfs/layout2.py; pass root=<spec/v2>. "
        "The wheel does not carry the corpus, so a wheel consumer must supply it"
    )


def _handle(raw: Mapping[str, Any]) -> str:
    return f"{raw['name']}@{raw['version']}"


def rules(root: Path | None = None) -> dict[str, Quant]:
    """handle -> :class:`Quant`, read from ``spec/v2/rules/*.json``.

    Identity facts only. The eligibility predicate and the emissions each rule
    document also carries are the EVALUATOR's, deliberately not read here.
    """

    corpus: dict[str, Quant] = {}
    for path in sorted((_root(root) / "rules").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        corpus[_handle(raw)] = Quant(
            handle=_handle(raw),
            declared_dtype=str(raw["declared_dtype"]),
            capability_floor_sm=int(raw["capability_floor_sm"]),
            conventions=dict(raw.get("conventions") or {}),
            lossy=bool(raw["lossy"]),
            inverse=str(raw.get("inverse", "")),
            digest=str(raw.get("digest", "")),
        )
    return corpus


def layouts(root: Path | None = None) -> dict[str, Arrangement]:
    """handle -> :class:`Arrangement`, read from ``spec/v2/layouts/*.json``.

    The THIRD reader of one document set: Rust vendors these records at build
    time (``build.rs`` -> ``include_str!``) and Go embeds them (``go:embed``).
    Both of those APPLY the arrangement; this one only names it. There is one
    producer of the vocabulary — the records — and adding a fourth author of a
    handle, a rank or a permutation anywhere is what this function exists to
    make unnecessary.

    Identity facts only, in the shape of its two siblings. No plan, no
    evaluator, no ``apply``.
    """

    directory = _root(root) / "layouts"
    if not directory.is_dir():
        # An absent class must refuse rather than answer "nothing is ratified":
        # an empty result reads as a valid, tiny catalog at every call site, and
        # the call site's next move is to fall back to row-major -- silently
        # undoing a delivery instead of naming a missing corpus.
        raise FileNotFoundError(
            f"{directory}: no layout-morphism records. A corpus without "
            f"spec/v2/layouts is not a corpus with no ratified layouts"
        )
    corpus: dict[str, Arrangement] = {}
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        rank = raw.get("rank")
        corpus[_handle(raw)] = Arrangement(
            handle=_handle(raw),
            cls=str(raw["class"]),
            rank=None if rank is None else int(rank),
            permutation=tuple(int(v) for v in raw.get("permutation") or ()),
            sub_axes=tuple(
                SubAxis(axis=int(entry["axis"]), extent=str(entry["extent"]))
                for entry in raw.get("sub_axes") or ()
            ),
            digest=str(raw.get("digest", "")),
        )
    return corpus


def identity_arrangement(root: Path | None = None) -> Arrangement:
    """The row-major record — the default every stamp written before layouts
    joined the stamp means, and the permanent fallback a compiler declares when
    its wish is outside the catalog.

    DERIVED, never spelled: it is the unique rankless record. A consumer that
    hard-codes ``"torch.contiguous@1"`` becomes a second author of the one
    handle every existing artifact carries, which is the whole failure this
    reader exists to remove. Two rankless records, or none, is a refusal — an
    ambiguous default is worse than an absent one, because it picks silently.
    """

    rankless = sorted(
        (a for a in layouts(root).values() if a.rank is None),
        key=lambda a: a.handle,
    )
    if len(rankless) != 1:
        raise ValueError(
            f"the layout corpus must hold exactly ONE rankless record (the "
            f"identity); it holds {len(rankless)}: "
            f"{[a.handle for a in rankless]!r}"
        )
    return rankless[0]


def topologies(root: Path | None = None) -> dict[str, dict[str, dict[str, tuple[int, ...]]]]:
    """handle -> component -> key -> logical shape, from
    ``spec/v2/topologies/*.json``. Shapes only: a topology carries no dtype,
    because the dtype is the quant rule's half of ``quant(topology)``."""

    corpus: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    for path in sorted((_root(root) / "topologies").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        corpus[_handle(raw)] = {
            str(component["name"]): {
                str(key): tuple(int(dimension) for dimension in shape)
                for key, shape in component["tensors"].items()
            }
            for component in raw["components"]
        }
    return corpus
