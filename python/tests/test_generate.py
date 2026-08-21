"""The candidate generator, and the one field it must never omit in silence.

A lane document with no top-level ``dtype`` is not a document with a small gap:
`lanes={contract: floor}` is required on every gen-worker Model subclass
(pgw#1597) and the declaration reads that field, so omitting it produces a
refusal with no remedy — in another repo, at the far end of a vendor bump. The
generator emitted exactly that twice before these tests existed.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from tensorfs.contract import Contract
from tensorfs.generate import (
    GENERATED_MARKER,
    RATIFIED_MARKER,
    Observed,
    UndeclarableLane,
    candidate,
    collapse,
)


def _lane(*entries: tuple[str, str, int]) -> dict[str, list[Observed]]:
    return {"only": [Observed(name=n, dtype=d, rank=r) for n, d, r in entries]}


def _decls(document: dict[str, object]) -> list[dict[str, Any]]:
    """A document is JSON, so its values are `object`; the tests know better."""

    return cast("list[dict[str, Any]]", document["tensors"])


# ── the pattern half ─────────────────────────────────────────────────────────


def test_only_whole_segment_integers_become_holes() -> None:
    # `layers.10` is an index; the `1`/`2` in conv1/conv2 and attn1/attn2 are
    # part of the MODULE NAME, and collapsing them would fuse declarations a
    # header can tell apart — under-constraint, the direction a wrong pattern
    # can be wrong in without ever failing to match.
    assert collapse("layers.10.mlp.weight") == "layers.{i}.mlp.weight"
    assert collapse("res_blocks.3.conv1.weight") == "res_blocks.{i}.conv1.weight"
    assert collapse("blocks.0.attn2.to_q.weight") == "blocks.{i}.attn2.to_q.weight"
    # A leading-zero segment is not an index spelling and stays literal.
    assert collapse("layers.01.weight") == "layers.01.weight"


def test_coverage_is_verified_not_asserted() -> None:
    document, report = candidate(
        _lane(("blocks.0.w", "BF16", 2), ("blocks.1.w", "BF16", 2), ("head.w", "BF16", 2)),
        name="t.demo",
        dtype="bfloat16",
    )
    assert report.covered == 3
    assert {entry["pattern"] for entry in _decls(document)} == {"blocks.{i}.w", "head.w"}


def test_a_rank_conflict_refuses_the_collapse_rather_than_averaging_it() -> None:
    # One declaration carries one rank, so a collapse spanning two is WRONG,
    # not merely coarse. It reverts to literals, which always match.
    document, report = candidate(
        _lane(("b.0.w", "BF16", 2), ("b.1.w", "BF16", 4)), name="t.demo", dtype="bfloat16"
    )
    assert report.rank_conflicts
    assert {entry["pattern"] for entry in _decls(document)} == {"b.0.w", "b.1.w"}


def test_required_is_the_intersection_across_checkpoints() -> None:
    document, _ = candidate(
        {
            "base": [Observed("a.w", "BF16", 2), Observed("b.w", "BF16", 2)],
            "turbo": [Observed("a.w", "BF16", 2)],
        },
        name="t.demo",
        dtype="bfloat16",
    )
    by_pattern = {entry["pattern"]: entry for entry in _decls(document)}
    assert "required" not in by_pattern["a.w"]  # every source carries it
    assert by_pattern["b.w"]["required"] is False


# ── the dtype half: derive, or refuse; never omit ────────────────────────────


def test_a_uniform_header_derives_the_lane_dtype() -> None:
    document, _ = candidate(_lane(("a.w", "BF16", 2), ("b.w", "BF16", 1)), name="t.demo")
    assert document["dtype"] == "bfloat16"
    assert "WAS AUTHORED" not in str(document["description"])


def test_an_f32_island_does_not_stop_the_derivation() -> None:
    # Adaptive-norm and modulation tables ship F32 inside otherwise-uniform
    # bf16 DiTs (ltx-2, z-image, krea-2). The LANE is what the model loads AS.
    document, _ = candidate(
        _lane(("a.w", "BF16", 2), ("norm.scale_table", "F32", 2)), name="t.demo"
    )
    assert document["dtype"] == "bfloat16"


def test_a_mixed_quant_header_REFUSES_instead_of_voting() -> None:
    # BF16 outnumbers F8_E4M3 here, exactly as it does in the real
    # qwen3.6-35b-a3b-fp8 tree — and that lane is fp8. A vote gets it wrong,
    # so the generator declines to guess.
    with pytest.raises(UndeclarableLane, match="authored, not voted on"):
        candidate(
            _lane(("a.w", "BF16", 2), ("b.w", "BF16", 2), ("c.w", "F8_E4M3", 2)),
            name="t.demo",
        )


def test_an_authored_dtype_is_recorded_as_owed() -> None:
    document, _ = candidate(
        _lane(("a.w", "BF16", 2), ("c.w", "F8_E4M3", 2)),
        name="t.demo",
        dtype="float8_e4m3fn",
    )
    assert document["dtype"] == "float8_e4m3fn"
    description = str(document["description"])
    assert description.startswith(GENERATED_MARKER)
    assert "WAS AUTHORED" in description  # a ratifier must check it


def test_a_spelling_gen_worker_cannot_price_refuses() -> None:
    # An unknown spelling derives a floor of 0 SILENTLY, which offers a
    # quantized lane to every card in the fleet. Silent is the whole problem.
    with pytest.raises(UndeclarableLane, match="SILENTLY"):
        candidate(_lane(("a.w", "Q4_K", 2)), name="t.demo", dtype="q4_k")


def test_an_unpriceable_spelling_may_be_acknowledged_deliberately() -> None:
    # A GGUF k-quant container genuinely has no torch scalar type. It ships
    # only when the author says so, and the escape hatch is the record.
    document, _ = candidate(
        _lane(("a.w", "Q4_K", 2)), name="t.demo", dtype="q4_k", allow_unknown_dtype=True
    )
    assert document["dtype"] == "q4_k"


def test_only_a_fragment_may_carry_no_dtype() -> None:
    entries = _lane(("a.w", "Q4_K", 2))
    with pytest.raises(UndeclarableLane):
        candidate(entries, name="t.demo")
    document, _ = candidate(entries, name="t.demo", fragment=True)
    assert "dtype" not in document
    assert "only a component FRAGMENT" in str(document["description"])


def test_the_candidate_is_a_valid_document_and_is_marked_unratified() -> None:
    document, _ = candidate(_lane(("a.w", "BF16", 2)), name="t.demo", version=1)
    contract = Contract.from_document(json.dumps(document))
    assert contract.stamp == "t.demo@1"
    assert contract.description.startswith(GENERATED_MARKER)


# ── ratification: the human step, expressible so scripts and files agree ─────


def test_answering_the_standing_items_takes_the_marker_off() -> None:
    document, _ = candidate(
        _lane(("a.w", "BF16", 2)),
        name="t.demo",
        ratified=["FUSIONS: none, and none is expressible - no concatenated projection."],
    )
    description = str(document["description"])
    assert description.startswith(RATIFIED_MARKER)
    assert "RATIFICATION OWED" not in description
    assert "FUSIONS: none" in description


def test_partial_ratification_keeps_the_marker_and_shrinks_the_list() -> None:
    # The useful middle state: answer what the evidence settles, leave the rest
    # owed. It is what shrinks a human's checklist to the items needing eyes.
    document, _ = candidate(
        _lane(("a.w", "BF16", 2)),
        name="t.demo",
        ratified=["SETS: none."],
        ratification=["ROLES stay mechanical; a second packaging may yet appear."],
    )
    description = str(document["description"])
    assert description.startswith(GENERATED_MARKER)
    assert "RATIFIED: (1) SETS: none." in description
    assert "RATIFICATION OWED: (1) ROLES stay mechanical" in description
    # Answering ANY standing item retires the four defaults; what is left owed
    # is exactly what the author still names.
    assert "FUSIONS are undeclared" not in description


def test_an_unratified_document_still_owes_the_standing_four() -> None:
    document, _ = candidate(_lane(("a.w", "BF16", 2)), name="t.demo")
    description = str(document["description"])
    assert description.startswith(GENERATED_MARKER)
    for standing in ("ROLES are mechanical", "FUSIONS are undeclared", "SETS are undeclared",
                     "COMPONENT SCOPE"):
        assert standing in description


def test_the_two_markers_are_mutually_exclusive() -> None:
    ratified, _ = candidate(_lane(("a.w", "BF16", 2)), name="t.demo", ratified=["SETS: none."])
    candidate_doc, _ = candidate(_lane(("a.w", "BF16", 2)), name="t.demo")
    assert not str(ratified["description"]).startswith(GENERATED_MARKER)
    assert not str(candidate_doc["description"]).startswith(RATIFIED_MARKER)
