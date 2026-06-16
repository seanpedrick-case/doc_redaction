"""Tests for optional post-redaction Pass 1 QA hook."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "doc_redaction" / "example_data" / "example_outputs"


def _partnership_paths() -> tuple[Path, Path] | None:
    review = EXAMPLE / "Partnership-Agreement-Toolkit_0_0.pdf_review_file.csv"
    words = (
        EXAMPLE
        / "Partnership-Agreement-Toolkit_0_0_ocr_results_with_words_local_ocr.csv"
    )
    if review.is_file() and words.is_file():
        return review, words
    return None


def test_run_post_redaction_pass1_qa_disabled() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.post_redaction_pass1_qa import run_post_redaction_pass1_qa

    out = run_post_redaction_pass1_qa(
        review_csv_path=review,
        ocr_words_csv_path=words,
        enabled=False,
    )
    assert out["enabled"] is False
    assert out["paths_created"] == []
    assert out["report"] is None


def test_run_post_redaction_pass1_qa_writes_report(tmp_path: Path) -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.post_redaction_pass1_qa import run_post_redaction_pass1_qa

    out = run_post_redaction_pass1_qa(
        review_csv_path=review,
        ocr_words_csv_path=words,
        output_folder=str(tmp_path),
        enabled=True,
        total_pages=8,
        must_redact=[],
        must_not_redact=[],
        include_in_outputs=True,
    )
    assert out["enabled"] is True
    assert out["report"] is not None
    assert "pass_strict" in out["report"]
    assert len(out["paths_created"]) >= 1
    report_path = Path(out["coverage_report_path"])
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "summary" in payload


def test_auto_prune_writes_sibling_and_keeps_original(tmp_path: Path) -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review_src, words = paths

    review_copy = tmp_path / review_src.name
    review_copy.write_bytes(review_src.read_bytes())
    orig_rows = sum(1 for _ in csv.DictReader(review_copy.open(encoding="utf-8-sig")))

    from tools.post_redaction_pass1_qa import run_post_redaction_pass1_qa

    out = run_post_redaction_pass1_qa(
        review_csv_path=review_copy,
        ocr_words_csv_path=words,
        output_folder=str(tmp_path),
        enabled=True,
        auto_prune=True,
        total_pages=8,
        must_redact=[r"Partnership"],
        must_not_redact=[],
        include_in_outputs=True,
    )
    pruned = out["prune_log"]
    assert pruned is not None
    pruned_path = Path(pruned["output_csv"])
    assert pruned_path.is_file()
    assert pruned_path.name.endswith("_pruned.csv")
    assert pruned["kept_count"] <= orig_rows
    assert (
        sum(1 for _ in csv.DictReader(review_copy.open(encoding="utf-8-sig")))
        == orig_rows
    )


def test_merge_policy_patterns_from_deny_allow() -> None:
    from tools.post_redaction_pass1_qa import merge_policy_patterns

    must, must_not = merge_policy_patterns(
        ["cora|fuller"],
        [r"dr\."],
        use_deny_allow_lists=True,
    )
    assert "cora|fuller" in must
    assert r"dr\." in must_not


def test_load_regex_patterns_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "patterns.csv"
    csv_path.write_text("alpha|beta\n gamma \n", encoding="utf-8")

    from tools.post_redaction_pass1_qa import load_regex_patterns_from_csv

    patterns = load_regex_patterns_from_csv(csv_path)
    assert patterns == ["alpha|beta", "gamma"]
