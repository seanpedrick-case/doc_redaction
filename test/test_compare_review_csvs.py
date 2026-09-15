"""Unit tests for agent-redact/eval/compare_review_csvs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from compare_review_csvs import (  # noqa: E402
    Box,
    box_iou,
    compare_review_csvs,
    load_review_boxes,
    match_boxes,
)
from compare_review_csvs import (
    main as compare_main,
)

_FIXTURE = _EVAL / "fixtures" / "example_smoke"


def test_box_iou_identical():
    a = Box(1, 0.1, 0.2, 0.3, 0.4)
    assert box_iou(a, a) == pytest.approx(1.0)


def test_box_iou_disjoint_pages():
    a = Box(1, 0.1, 0.2, 0.3, 0.4)
    b = Box(2, 0.1, 0.2, 0.3, 0.4)
    assert box_iou(a, b) == 0.0


def test_match_perfect_overlap():
    gold = [Box(1, 0.1, 0.2, 0.3, 0.4, label="PERSON")]
    agent = [Box(1, 0.1, 0.2, 0.3, 0.4, label="PERSON")]
    report = match_boxes(gold, agent, threshold=0.5)
    assert report.strict_pass
    assert report.f1 == pytest.approx(1.0)
    assert report.label_agreement == pytest.approx(1.0)


def test_match_extra_agent_box_fails_strict():
    gold = [Box(1, 0.1, 0.2, 0.3, 0.4)]
    agent = [
        Box(1, 0.1, 0.2, 0.3, 0.4),
        Box(1, 0.5, 0.5, 0.6, 0.6),
    ]
    report = match_boxes(gold, agent, threshold=0.5)
    assert not report.strict_pass
    assert report.recall == pytest.approx(1.0)
    assert report.precision == pytest.approx(0.5)


def test_match_missing_gold_lowers_recall():
    gold = [
        Box(1, 0.1, 0.2, 0.3, 0.4),
        Box(1, 0.5, 0.5, 0.7, 0.7),
    ]
    agent = [Box(1, 0.1, 0.2, 0.3, 0.4)]
    report = match_boxes(gold, agent, threshold=0.5)
    assert report.recall == pytest.approx(0.5)
    assert report.precision == pytest.approx(1.0)


def test_gold_contained_mode_accepts_larger_agent_box():
    gold = [Box(1, 0.20, 0.20, 0.30, 0.30)]
    agent = [Box(1, 0.10, 0.10, 0.40, 0.40)]
    iou_report = match_boxes(gold, agent, threshold=0.5, mode="iou")
    # IoU of nested box is area_gold/area_agent = 0.01/0.09 ≈ 0.11
    assert iou_report.f1 < 0.5
    contained = match_boxes(gold, agent, threshold=0.5, mode="gold_contained")
    assert contained.strict_pass
    assert contained.f1 == pytest.approx(1.0)


def test_many_to_one_covers_gold_with_split_agent_boxes():
    gold = [Box(1, 0.10, 0.20, 0.40, 0.30)]
    agent = [
        Box(1, 0.10, 0.20, 0.22, 0.30),
        Box(1, 0.23, 0.20, 0.40, 0.30),
    ]
    one_to_one = match_boxes(gold, agent, threshold=0.3, mode="iou")
    # Single gold can only bind one agent in 1:1 mode → precision may be 0.5
    assert one_to_one.matched_gold == 1
    assert one_to_one.matched_agent == 1
    many = match_boxes(gold, agent, threshold=0.3, mode="iou", many_to_one=True)
    assert many.matched_gold == 1
    assert many.matched_agent == 2
    assert many.precision == pytest.approx(1.0)


def test_load_and_compare_example_smoke_fixture():
    gold = load_review_boxes(_FIXTURE / "ideal_review_file.csv")
    agent = load_review_boxes(_FIXTURE / "agent_review_file.csv")
    assert len(gold) == 1
    assert len(agent) == 1
    report = compare_review_csvs(
        _FIXTURE / "ideal_review_file.csv",
        _FIXTURE / "agent_review_file.csv",
        threshold=0.5,
    )
    assert report.f1 >= 0.9
    assert report.strict_pass


def test_cli_min_f1_and_json(tmp_path: Path):
    out = tmp_path / "report.json"
    code = compare_main(
        [
            str(_FIXTURE / "ideal_review_file.csv"),
            str(_FIXTURE / "agent_review_file.csv"),
            "--min-f1",
            "0.9",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["f1"] >= 0.9
    assert payload["strict_pass"] is True


def test_cli_fails_on_missing_boxes(tmp_path: Path):
    gold_csv = tmp_path / "gold.csv"
    agent_csv = tmp_path / "agent.csv"
    gold_csv.write_text(
        "page,label,xmin,ymin,xmax,ymax,text,id\n" "1,PERSON,0.1,0.2,0.3,0.4,Alice,1\n",
        encoding="utf-8-sig",
    )
    agent_csv.write_text(
        "page,label,xmin,ymin,xmax,ymax,text,id\n",
        encoding="utf-8-sig",
    )
    code = compare_main(
        [str(gold_csv), str(agent_csv), "--min-f1", "0.5", "--require-strict"]
    )
    assert code == 1
