"""Tests for core-coverage gold geometry scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from compare_review_csvs import (  # noqa: E402
    Box,
    compare_review_csvs,
    extract_core_box_indices,
    match_boxes,
    match_boxes_core_coverage,
    union_gold_coverage,
)

_PROFESSOR_GOLD = _EVAL / "gold" / "example_emails_to_professor_agentic"
_PROFESSOR_RUN = (
    _EVAL / "runs" / "gradio_pnofp535_20261104_142940" / "agent_review_file.csv"
)


def test_extract_core_drops_nested_dr():
    gold = [
        Box(1, 0.74, 0.30, 0.76, 0.32, text="Dr", label="TITLES"),
        Box(1, 0.74, 0.30, 0.84, 0.32, text="Dr David R.", label="TITLES - PERSON"),
        Box(1, 0.77, 0.30, 0.84, 0.32, text="David R.", label="PERSON"),
    ]
    core, removed = extract_core_box_indices(gold, subume_threshold=0.9)
    assert len(core) == 1
    assert gold[core[0]].text == "Dr David R."
    assert set(removed) == {0, 2}


def test_core_coverage_nested_dr_perfect_recall():
    gold = [
        Box(1, 0.74, 0.30, 0.76, 0.32, text="Dr", label="TITLES"),
        Box(1, 0.74, 0.30, 0.84, 0.32, text="Dr David R.", label="TITLES - PERSON"),
    ]
    agent = [
        Box(1, 0.74, 0.30, 0.76, 0.32, text="Dr", label="TITLES"),
        Box(1, 0.74, 0.30, 0.84, 0.32, text="Dr David R.", label="TITLES - PERSON"),
    ]
    report = match_boxes_core_coverage(gold, agent, threshold=0.85)
    assert report.scoring == "core_coverage"
    assert report.core_gold_count == 1
    assert report.recall == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)


def test_union_two_agent_boxes_cover_hyde_lab():
    gold = [
        Box(1, 0.0, 0.0, 1.0, 0.2, text="Hyde lab", label="CUSTOM"),
    ]
    agent = [
        Box(1, 0.0, 0.0, 0.55, 0.2, text="Hyde", label="CUSTOM"),
        Box(1, 0.55, 0.0, 1.0, 0.2, text="lab", label="CUSTOM"),
    ]
    cov = union_gold_coverage(gold[0], agent)
    assert cov == pytest.approx(1.0)
    report = match_boxes_core_coverage(gold, agent, threshold=0.85)
    assert report.recall == pytest.approx(1.0)


def test_legacy_vs_core_on_professor_run():
    if not (_PROFESSOR_GOLD / "ideal_review_file.csv").is_file():
        pytest.skip("local professor gold case not present")
    if not _PROFESSOR_RUN.is_file():
        pytest.skip("mirrored professor run not present")
    ideal = _PROFESSOR_GOLD / "ideal_review_file.csv"
    legacy = compare_review_csvs(
        ideal,
        _PROFESSOR_RUN,
        threshold=0.85,
        mode="gold_contained",
        many_to_one=True,
        scoring="legacy_box_match",
    )
    core = compare_review_csvs(
        ideal,
        _PROFESSOR_RUN,
        threshold=0.85,
        scoring="core_coverage",
    )
    assert core.core_gold_count < legacy.gold_count
    assert core.scoring == "core_coverage"
    hyde_unmatched = [
        p
        for p in core.per_core_gold
        if "hyde" in (p.get("text") or "").casefold() and not p.get("matched")
    ]
    assert hyde_unmatched, "core scoring should still flag partial Hyde/lab spans"


def test_example_smoke_core_scoring_still_passes():
    smoke = _EVAL / "gold" / "example_smoke"
    ideal = smoke / "ideal_review_file.csv"
    agent = smoke / "agent_review_file.csv"
    if not ideal.is_file():
        pytest.skip("example_smoke gold missing")
    report = compare_review_csvs(
        ideal,
        agent,
        threshold=0.65,
        scoring="core_coverage",
    )
    assert report.recall == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)
