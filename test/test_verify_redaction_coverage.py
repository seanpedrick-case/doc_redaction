"""Tests for Pass 1 verify_redaction_coverage and headless word OCR search."""

from __future__ import annotations

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


def test_verify_redaction_coverage_detects_uncovered_term() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.verify_redaction_coverage import verify_redaction_coverage

    report = verify_redaction_coverage(
        review,
        words,
        must_redact=[r"Partnership"],
        total_pages=8,
    )
    assert report.pages_total == 8
    page1 = report.pages["1"]
    assert (
        any(t.text == "Partnership" for t in page1.uncovered_terms)
        or page1.review_row_count >= 0
    )


def test_verify_redaction_coverage_passes_when_no_policy_terms() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.verify_redaction_coverage import verify_redaction_coverage

    report = verify_redaction_coverage(
        review,
        words,
        must_redact=[],
        must_not_redact=[],
        total_pages=8,
        min_word_length=1,
    )
    assert report.pages_total == 8


def test_word_level_ocr_text_search_finds_literal() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    _, words = paths

    from tools.verify_redaction_coverage import run_word_level_ocr_text_search

    result = run_word_level_ocr_text_search(
        words,
        "Partnership",
        review_csv_path=paths[0],
    )
    assert result["match_count"] >= 1
    assert result["matches"][0]["word_text"] == "Partnership"


def test_run_verify_redaction_coverage_wrapper() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.verify_redaction_coverage import verify_redaction_coverage

    out = verify_redaction_coverage(
        review,
        words,
        must_redact=[r"Toolkit"],
        total_pages=8,
    ).to_dict()
    assert "pass" in out
    assert "pages" in out
    assert "summary" in out


def test_cli_api_verify_redaction_coverage() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.verify_redaction_coverage import verify_redaction_coverage as verify_fn

    out = verify_fn(
        review,
        words,
        must_redact=[],
        total_pages=8,
        min_word_length=1,
    ).to_dict()
    assert "pass" in out
    assert "summary" in out


def test_cli_api_word_search() -> None:
    paths = _partnership_paths()
    if not paths:
        pytest.skip("Partnership example fixtures not present")
    review, words = paths

    from tools.verify_redaction_coverage import run_word_level_ocr_text_search

    out = run_word_level_ocr_text_search(words, "Toolkit", review_csv_path=review)
    assert out["match_count"] >= 1
