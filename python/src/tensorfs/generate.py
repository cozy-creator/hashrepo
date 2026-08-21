"""Generate a CANDIDATE lane contract from safetensors/GGUF headers.

A contract describes a header, so the header can write most of it. This module
is that derivation: (one or more header inventories) -> a v1 document plus a
REPORT of everything the machine could not decide. It reads no tensor byte and
has no network — the caller supplies inventories, which on the control plane
come from ranged header reads (``scripts/generate-contract.py``).

The output is a CANDIDATE, never a publication. Its ``description`` opens with
``GENERATED CANDIDATE — NOT RATIFIED`` and carries the derivation's own
evidence, so a document that reaches ``spec/v1/contracts`` still carrying that
marker is visibly unratified. Ratification is a human act (rename mechanical
roles to spelling-independent ones, declare fusions and sets, decide the
component scope) and it is the only thing that removes the marker.

Three things are DERIVED and three are deliberately NOT.

Derived: the pattern set (whole-segment integers become ``{i}`` holes, and
nothing else does), the per-declaration ``rank``/``dtypes`` constraints, and
``required`` — a declaration is required only when EVERY source carries it, so
handing the generator two checkpoints of one family produces the
one-document-two-checkpoints shape as a measurement rather than a claim.

Not derived: ``role`` (mechanically the pattern itself — a spelling-independent
identity is a human judgement about what the bytes ARE), ``fusion`` (a fused
axis is invisible in a header: ``[3*d, d]`` and ``[3d, d]`` are the same
numbers) and ``sets``. Each is named in the ratification checklist instead of
being guessed, because a wrong fusion is a ~90%-error split that never crashes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "GENERATED_MARKER",
    "RATIFIED_MARKER",
    "UndeclarableLane",
    "Observed",
    "Report",
    "candidate",
    "collapse",
    "expand",
]

#: Every generated description opens with this. Grep for it to find every
#: unratified document in the library.
GENERATED_MARKER = "GENERATED CANDIDATE - NOT RATIFIED."

#: What replaces it once every standing owed item has an answer. A document
#: carries exactly one of the two, so "is this ratified?" is a grep and never a
#: judgement about how complete the prose looks.
RATIFIED_MARKER = "GENERATED, THEN RATIFIED."

#: torch spelling for a safetensors element type, for the top-level `dtype`.
#: GGUF block-quant containers are absent on purpose: they have no torch
#: spelling, so a GGUF candidate declares no top-level dtype and says so.
_TORCH_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "BOOL": "bool",
}

_SEPARATORS = re.compile(r"([./])")


#: The dtype spellings gen-worker's ``DTYPE_MIN_SM`` recognizes
#: (``python-gen-worker/src/gen_worker/models/tensor_layout_contract.py``), which
#: is the ONLY producer of a lane's sm floor.
#:
#: A MIRROR, and labelled as one because a second copy of someone else's table is
#: a drift hazard by construction. What it mirrors is deliberately narrow: the SET
#: OF SPELLINGS, never the floors. tensorfs does not decide placement and must not
#: look like it does.
#:
#: It earns the duplication by closing a SILENT failure. ``capability_floor_for_dtype``
#: answers 0 for a spelling it does not know — correct for fp32, and catastrophic for a
#: quantized lane, which would then be offered to every card in the fleet with no floor
#: at all. A wrong floor is loud (the pod refuses); an ABSENT floor is silent, and buys
#: hardware that cannot run the arithmetic.
#:
#: Staleness is bounded and cheap: if gen-worker learns a spelling this set has not,
#: the author passes ``allow_unknown_dtype`` once. It can never make a document wrong,
#: only make an author say the quiet part out loud.
_KNOWN_TO_GEN_WORKER = frozenset({
    "float32", "float", "fp32", "tf32",
    "float16", "half", "fp16",
    "bfloat16", "bf16",
    "int8", "uint8",
    "int4", "uint4",
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
    "fp8_e4m3", "fp8_e5m2", "fp8",
    "float4_e2m1fn", "nvfp4", "fp4", "mxfp4",
})


class UndeclarableLane(ValueError):
    """A LANE document has no top-level dtype and the header cannot supply one.

    Raised instead of emitting the document, because a serve lane without a
    top-level dtype is not a document with a small gap — it is a document no
    endpoint can declare. `lanes={contract: floor}` is required on every Model
    subclass (pgw#1597) and the declaration reads this field, so omitting it
    turns into a refusal with no remedy at the far end of a vendor bump, which
    is the worst place to discover it.

    The remedy is always the same and the caller always knows it: pass the
    lane's quantization explicitly. THE LANE'S QUANTIZATION IS AUTHORED
    KNOWLEDGE, exactly like roles and fusions — it is not a majority vote over
    the header. `sdxl.diffusers-fp8-rowwise@1` is float8_e4m3fn with 221 of its
    257 declarations BF16; `sdxl.diffusers-nvfp4-flat@1` is float4_e2m1fn with
    NO declaration spelling fp4 at all, because the resident weights are packed
    U8 pairs. Both of those a header vote gets wrong.
    """


@dataclass(frozen=True, slots=True)
class Observed:
    """One tensor as its header states it: name, element type, rank."""

    name: str
    dtype: str
    rank: int


def collapse(name: str) -> str:
    """The tensor's name with every WHOLE-SEGMENT integer replaced by ``{i}``.

    Whole-segment, and this is the load-bearing restriction. ``layers.10.mlp``
    is an index and collapses; ``conv1``/``conv2`` and ``attn1``/``attn2`` are
    distinct MODULES whose names merely end in a digit, and collapsing those
    would fuse two declarations that a header can tell apart — under-
    constraining the contract in exactly the direction that stops it being
    falsifiable.
    """

    if "{" in name or "}" in name:
        raise ValueError(f"tensor name {name!r} contains a brace; it cannot be a pattern")
    parts = _SEPARATORS.split(name)
    return "".join("{i}" if _is_index(part) else part for part in parts)


def _is_index(part: str) -> bool:
    return part.isdigit() and (part == "0" or not part.startswith("0"))


def expand(pattern: str) -> re.Pattern[str]:
    """The matcher for ``pattern``: ``{i}`` accepts one non-negative integer
    without leading zeros, everything else is literal."""

    body = "".join(
        r"(0|[1-9][0-9]*)" if piece == "{i}" else re.escape(piece)
        for piece in re.split(r"(\{i\})", pattern)
        if piece
    )
    return re.compile(f"^{body}$")


@dataclass
class Report:
    """What the derivation MEASURED, and what it refuses to decide."""

    sources: dict[str, int] = field(default_factory=dict)
    declarations: int = 0
    covered: int = 0
    dtype_histogram: Counter[str] = field(default_factory=Counter)
    optional: list[str] = field(default_factory=list)
    rank_conflicts: list[str] = field(default_factory=list)
    dtype_split: list[str] = field(default_factory=list)
    index_gaps: list[str] = field(default_factory=list)
    degenerate_holes: list[tuple[str, int, int]] = field(default_factory=list)
    singletons: int = 0

    def lines(self) -> list[str]:
        out = [f"{label}: {count} tensors" for label, count in sorted(self.sources.items())]
        out.append(
            f"{self.declarations} declarations cover {self.covered} tensors "
            f"({self.singletons} singleton, {self.declarations - self.singletons} indexed)"
        )
        out.append(
            "dtypes: "
            + ", ".join(f"{name} x{count}" for name, count in sorted(self.dtype_histogram.items()))
        )
        for label, entries in (
            ("OPTIONAL (absent from some source)", self.optional),
            ("RANK CONFLICT (collapse refused)", self.rank_conflicts),
            ("MIXED DTYPE within one pattern", self.dtype_split),
            ("NON-CONTIGUOUS index range", self.index_gaps),
        ):
            for entry in entries:
                out.append(f"{label}: {entry}")
        for pattern, position, value in self.degenerate_holes:
            out.append(
                f"DEGENERATE hole (one value - probably not an index): "
                f"{pattern} hole {position} == {value}"
            )
        return out


def candidate(
    sources: Mapping[str, Iterable[Observed]],
    *,
    name: str | None = None,
    version: int = 1,
    dtype: str | None = None,
    fragment: bool = False,
    allow_unknown_dtype: bool = False,
    literals: Sequence[str] = (),
    summary: str = "",
    ratification: Sequence[str] = (),
    ratified: Sequence[str] = (),
) -> tuple[dict[str, object], Report]:
    """A v1 candidate document over ``sources`` (label -> inventory).

    Several sources = several CHECKPOINTS of one family. A declaration every
    source carries is ``required``; one only some carry is optional, which is
    how the format states "these two checkpoints are one layout, and here is
    the delta".

    ``fragment`` marks a COMPONENT fragment (``sdxl.clip-g-*``,
    ``dit.blocks-fused-qkv``) rather than a serve lane. Only a fragment may
    carry no top-level dtype; a lane that cannot derive one raises
    :class:`UndeclarableLane` rather than shipping undeclarable.

    ``ratified`` carries the human step's ANSWERS to the four standing owed
    items (roles, fusions, sets, scope). Supplying it suppresses those four —
    the author is asserting they were addressed, and each line says how. With
    nothing left in ``ratification``, the ``GENERATED CANDIDATE`` marker comes
    OFF: the document is ratified. Partial is expressible and useful — answer
    what the evidence settles, leave the rest owed, keep the marker — which is
    what shrinks a human's checklist to the items that genuinely need eyes.

    :raises UndeclarableLane: a non-fragment whose dtype is neither given nor
        uniform in the header.
    """

    if not sources:
        raise ValueError("a candidate needs at least one header inventory")

    report = Report()
    #: pattern -> label -> [Observed]
    grouped: dict[str, dict[str, list[Observed]]] = {}
    for label, entries in sources.items():
        seen = list(entries)
        report.sources[label] = len(seen)
        for item in seen:
            pattern = collapse(item.name)
            grouped.setdefault(pattern, {}).setdefault(label, []).append(item)
            report.dtype_histogram[item.dtype] += 1

    # A collapse that fuses different ranks is WRONG, not merely coarse: rank is
    # a matcher constraint and one declaration carries one. Refuse it and fall
    # back to the literal names, which always match.
    # A PIN is the pattern the ratifier wants, spelled out: it collapses to one
    # generated pattern and says which of that pattern's holes were never
    # indices. `layers.{i}.self_attention.to_out.0.weight` keeps the layer stack
    # and pins the Sequential position.
    masks: dict[str, list[bool]] = {}
    for pin in literals:
        base = collapse(pin.replace("{i}", "\x00")).replace("\x00", "{i}")
        if base not in grouped:
            raise ValueError(f"pin {pin!r} collapses to {base!r}, which nothing generated")
        holes = re.findall(r"\{i\}|(?:^|(?<=[./]))[0-9]+(?=$|[./])", pin)
        if len(holes) != base.count("{i}"):  # pragma: no cover - defensive
            raise ValueError(f"pin {pin!r} does not line up with {base!r}")
        masks[base] = [piece == "{i}" for piece in holes]

    # A collapse that fuses different ranks is WRONG, not merely coarse: rank is
    # a matcher constraint and one declaration carries one. Regrind it to fully
    # literal names, which always match.
    for pattern, by_label in list(grouped.items()):
        ranks = {item.rank for items in by_label.values() for item in items}
        if len(ranks) > 1:
            count = len({item.name for items in by_label.values() for item in items})
            report.rank_conflicts.append(f"{pattern} ({sorted(ranks)}) -> {count} literals")
            masks[pattern] = [False] * pattern.count("{i}")

    for pattern, mask in masks.items():
        by_label = grouped.pop(pattern)
        matcher = expand(pattern)
        for label, items in by_label.items():
            for item in items:
                found = matcher.match(item.name)
                assert found is not None
                pieces = re.split(r"(\{i\})", pattern)
                hole = 0
                rebuilt = []
                for piece in pieces:
                    if piece == "{i}":
                        rebuilt.append("{i}" if mask[hole] else found.group(hole + 1))
                        hole += 1
                    else:
                        rebuilt.append(piece)
                grouped.setdefault("".join(rebuilt), {}).setdefault(label, []).append(item)

    tensors: list[dict[str, object]] = []
    for pattern in sorted(grouped):
        by_label = grouped[pattern]
        items = [item for entries in by_label.values() for item in entries]
        dtypes = sorted({item.dtype for item in items})
        rank = items[0].rank
        required = len(by_label) == len(sources)
        if not required:
            report.optional.append(f"{pattern} (only in {sorted(by_label)})")
        if len(dtypes) > 1:
            report.dtype_split.append(f"{pattern}: {dtypes}")
        indices = _indices(pattern, [item.name for item in items])
        if indices is None:
            report.singletons += 1
        else:
            for position, values in enumerate(indices):
                # ONE value behind a hole is the `nn.Sequential` shape, not a
                # layer stack: `adaLN_modulation.1` and `to_out.0` are fixed
                # positions. Collapsing them widens the contract to spellings
                # the checkpoint never contains, which is the one direction a
                # generated pattern can be wrong in without ever failing to
                # match. Flag it; the ratifier pins it back to the literal.
                if len(values) == 1:
                    report.degenerate_holes.append((pattern, position, values[0]))
                if values and sorted(values) != list(range(max(values) + 1)):
                    report.index_gaps.append(
                        f"{pattern} hole {position}: {sorted(values)[:8]}... "
                        f"({len(values)} of 0..{max(values)})"
                    )
        declaration: dict[str, object] = {
            "role": pattern,
            "pattern": pattern,
            "dtypes": dtypes,
            "rank": rank,
        }
        if not required:
            declaration["required"] = False
        tensors.append(declaration)

    report.declarations = len(tensors)
    report.covered = _verify_coverage(sources, tensors)

    authored_dtype = dtype is not None
    if dtype is None:
        dtype = _uniform_dtype(report.dtype_histogram)
    if dtype is None and not fragment:
        observed = ", ".join(sorted(report.dtype_histogram))
        raise UndeclarableLane(
            f"{name or 'this candidate'} is a LANE document with no derivable top-level "
            f"dtype (the header carries {observed}). A lane's dtype names its "
            "QUANTIZATION, which is authored, not voted on: pass --dtype <spelling>. "
            "If this is a component FRAGMENT rather than a serve lane, say so with "
            "--fragment and it may legitimately carry none."
        )
    if dtype is not None and dtype not in _KNOWN_TO_GEN_WORKER and not allow_unknown_dtype:
        raise UndeclarableLane(
            f"top-level dtype {dtype!r} is not a spelling gen-worker's DTYPE_MIN_SM "
            "knows, and an unknown spelling derives a floor of 0 SILENTLY - a "
            "quantized lane offered to every card in the fleet. Use the torch-precise "
            "spelling (float8_e4m3fn, float4_e2m1fn, bfloat16, ...), or - if this "
            "layout genuinely has no torch scalar type, as a GGUF k-quant container "
            "does not - TEACH DTYPE_MIN_SM the spelling and its true floor FIRST, then "
            "pass allow_unknown_dtype to record that you did it deliberately."
        )

    document: dict[str, object] = {"format": "tensorfs-contract-v1"}
    if name is not None:
        document["name"] = name
        document["version"] = version
    if dtype is not None:
        document["dtype"] = dtype
    document["description"] = _description(
        report, summary, ratification, dtype, authored_dtype, ratified
    )
    document["tensors"] = tensors
    return document, report


def _indices(pattern: str, names: Sequence[str]) -> list[list[int]] | None:
    """Per-hole index values seen for ``pattern``, or None if it has no hole."""

    holes = pattern.count("{i}")
    if holes == 0:
        return None
    matcher = expand(pattern)
    values: list[list[int]] = [[] for _ in range(holes)]
    for name in names:
        found = matcher.match(name)
        if found is None:  # pragma: no cover -- a pattern always matches its source
            raise AssertionError(f"{pattern!r} does not match its own source {name!r}")
        for position, raw in enumerate(found.groups()):
            values[position].append(int(raw))
    return [sorted(set(column)) for column in values]


def _verify_coverage(
    sources: Mapping[str, Iterable[Observed]], tensors: Sequence[Mapping[str, object]]
) -> int:
    """Expand every declaration back over every source: each tensor must match
    exactly one. Asserting coverage instead of measuring it is how a document
    ships describing a file it does not describe."""

    matchers = [(str(entry["pattern"]), expand(str(entry["pattern"]))) for entry in tensors]
    covered = 0
    for label, entries in sources.items():
        for item in entries:
            hits = [pattern for pattern, matcher in matchers if matcher.match(item.name)]
            if len(hits) != 1:
                raise AssertionError(
                    f"{label}: {item.name!r} matched {len(hits)} declarations ({hits[:4]}); "
                    "the generated pattern set is not a partition"
                )
            covered += 1
    return covered


def _uniform_dtype(histogram: Counter[str]) -> str | None:
    """The lane dtype ONLY when the header settles it beyond argument: one
    element type, or one plus an F32 island.

    Deliberately not a majority vote. A vote reads
    `sdxl.diffusers-fp8-rowwise@1` as bfloat16 (221 of 257 declarations are)
    and `qwen3.6-35b-a3b.vllm-fp8@1` as bfloat16 too (BF16 outnumbers F8_E4M3
    by tensor count), and both are fp8 lanes. Where a header is mixed the
    answer is authored, and this returns None so the caller must say so.

    The F32 island is the one carve-out, and it is a fact about diffusion
    trees rather than a fudge: adaptive-norm and modulation tables ship F32
    inside otherwise-uniform bf16 DiTs (ltx-2, z-image, krea-2), and the LANE
    is what the model loads AS, not what its smallest table is.
    """

    named = {name for name in histogram if name in _TORCH_DTYPE}
    if named != set(histogram):
        return None  # a container this table cannot spell: authored, not voted
    body = named - {"F32"}
    if len(body) == 1:
        return _TORCH_DTYPE[body.pop()]
    if not body and named == {"F32"}:
        return _TORCH_DTYPE["F32"]
    return None


_DEFAULT_RATIFICATION = (
    "ROLES are mechanical (role == pattern). Rename any tensor whose bytes are "
    "the same bytes another packaging spells differently - that is the only "
    "thing that makes a cross-packaging derive possible instead of a refusal.",
    "FUSIONS are undeclared. A header cannot see a fused axis, so a fused "
    "qkv/gate-up must be declared by a human reading the model's config "
    "(head count, hidden size); a wrong one is a ~90%-error split that never "
    "crashes.",
    "SETS are undeclared (no subset-snapshot groups).",
    "COMPONENT SCOPE is the caller's: declaring a shared encoder ties this "
    "document to every family that ships the same one (tensorfs#121).",
)


def _description(
    report: Report,
    summary: str,
    ratification: Sequence[str],
    dtype: str | None,
    authored_dtype: bool = False,
    ratified: Sequence[str] = (),
) -> str:
    owed = tuple(ratification) + ((), _DEFAULT_RATIFICATION)[not ratified]
    parts = [GENERATED_MARKER] if owed else [RATIFIED_MARKER]
    if summary:
        parts.append(summary)
    parts.append("Derived mechanically by tensorfs.generate from headers only - "
                 "no tensor byte was read. " + " ".join(report.lines()) + ".")
    if dtype is None:
        parts.append(
            "NO top-level dtype, which only a component FRAGMENT may carry: this "
            "document describes part of a tree, not a serve lane, so there is no "
            "lane load dtype for it to name."
        )
    if authored_dtype and owed:
        owed = (
            f"THE TOP-LEVEL DTYPE {dtype!r} WAS AUTHORED, not read: the header did not "
            "settle it, so a human named the lane's quantization. Check it against what "
            "the lane's bytes ARE - it is the field gen-worker derives the sm floor from, "
            "so a wrong one is a placement claim with nothing behind it.",
            *owed,
        )
    if ratified:
        parts.append(
            "RATIFIED: " + " ".join(f"({n + 1}) {item}" for n, item in enumerate(ratified))
        )
    if owed:
        parts.append(
            "RATIFICATION OWED: "
            + " ".join(f"({n + 1}) {item}" for n, item in enumerate(owed))
        )
    return " ".join(parts)


def render(document: Mapping[str, object]) -> str:
    """The document as it lands on disk: 2-space JSON, trailing newline."""

    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"
