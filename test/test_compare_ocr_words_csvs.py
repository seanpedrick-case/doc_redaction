"""Unit tests for agent-redact/eval/compare_ocr_words_csvs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from compare_ocr_words_csvs import (  # noqa: E402
    compare_ocr_gold_for_run,
    compare_ocr_words_csvs,
    ocr_method_mismatch,
)


def _write_ocr_csv(path: Path, rows: list[tuple[int, str, float, float, float, float]]) -> None:
    lines = ["page,line,word_text,word_x0,word_y0,word_x1,word_y1"]
    for page, word, x0, y0, x1, y1 in rows:
        lines.append(f"{page},1,{word},{x0},{y0},{x1},{y1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def test_compare_ocr_words_perfect_match(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    obs = tmp_path / "obs.csv"
    rows = [(1, "Hello", 0.1, 0.2, 0.2, 0.3), (1, "World", 0.3, 0.2, 0.4, 0.3)]
    _write_ocr_csv(gold, rows)
    _write_ocr_csv(obs, rows)
    report = compare_ocr_words_csvs(gold, obs)
    assert report.f1 == 1.0
    assert report.matched == 2
    assert report.gold_word_count == 2


def test_compare_ocr_words_partial_recall(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    obs = tmp_path / "obs.csv"
    _write_ocr_csv(gold, [(1, "Alpha", 0, 0, 0.1, 0.1), (1, "Beta", 0, 0, 0.1, 0.1)])
    _write_ocr_csv(obs, [(1, "Alpha", 0, 0, 0.1, 0.1)])
    report = compare_ocr_words_csvs(gold, obs)
    assert report.recall == 0.5
    assert report.precision == 1.0
    assert abs(report.f1 - 2 / 3) < 1e-6


def test_ocr_method_mismatch_skips(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    run = tmp_path / "run"
    fixture.mkdir()
    run.mkdir()
    (fixture / "manifest.json").write_text(
        json.dumps({"ocr_method": "pymupdf"}), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps({"ocr_method": "paddle"}), encoding="utf-8"
    )
    skip, reason = ocr_method_mismatch(gold_case_dir=fixture, run_dir=run)
    assert skip is True
    assert "mismatch" in reason


def test_compare_ocr_gold_for_run_writes_skipped(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    run = tmp_path / "run"
    fixture.mkdir()
    run.mkdir()
    report = compare_ocr_gold_for_run(gold_case_dir=fixture, run_dir=run)
    assert report.skipped is True
