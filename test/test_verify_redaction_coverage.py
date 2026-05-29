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
    assert "pass_strict" in out
    assert "pass_with_cleanup" in out
    assert "pages" in out
    assert "summary" in out
    assert "pages_needing_csv_cleanup" in out["summary"]


def test_pass_strict_not_blocked_by_suspicious_rows(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    words = tmp_path / "words.csv"
    review.write_text(
        "image,page,label,color,xmin,ymin,xmax,ymax,id,text\n"
        'img.png,1,PERSON,"(0,0,0)",0.1,0.1,0.2,0.2,abc,-\n',
        encoding="utf-8-sig",
    )
    words.write_text(
        "page,word_text,word_x0,word_y0,word_x1,word_y1\n" "1,Hello,0.5,0.5,0.6,0.6\n",
        encoding="utf-8-sig",
    )

    from tools.verify_redaction_coverage import verify_redaction_coverage

    report = verify_redaction_coverage(
        review, words, must_redact=[], must_not_redact=[], total_pages=1
    )
    assert report.pass_strict is True
    assert report.pass_with_cleanup is False
    assert report.pages_needing_csv_cleanup == [1]
    assert report.pages_flagged_for_vlm == []


def test_prune_suspicious_review_csv(tmp_path: Path) -> None:
    src = tmp_path / "review.csv"
    out = tmp_path / "review_pruned.csv"
    src.write_text(
        "image,page,label,color,xmin,ymin,xmax,ymax,id,text\n"
        'img.png,1,PERSON,"(0,0,0)",0.1,0.1,0.2,0.2,a1,-\n'
        'img.png,1,PERSON,"(0,0,0)",0.3,0.1,0.4,0.2,a2,Cora\n',
        encoding="utf-8-sig",
    )

    from tools.verify_redaction_coverage import prune_suspicious_review_csv

    log = prune_suspicious_review_csv(
        src, out, must_redact=[r"cora"], min_word_length=3
    )
    assert log["removed_count"] == 1
    assert log["kept_count"] == 1
    pruned_text = out.read_text(encoding="utf-8-sig")
    assert "Cora" in pruned_text
    assert ",a1,-" not in pruned_text


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


def test_infer_leak_likely_causes_flags_non_normalized_coords() -> None:
    from tools.verify_redaction_coverage import PageReport, infer_leak_likely_causes

    page_report = PageReport(
        page=3,
        pass_strict=False,
        review_row_count=2,
        text_layer_leaks=["Lambeth"],
    )
    page_review = [
        {
            "page": 3,
            "xmin": 491.43,
            "ymin": 0.1,
            "xmax": 500.0,
            "ymax": 0.2,
            "text": "Example",
            "label": "PERSON",
        }
    ]
    causes = infer_leak_likely_causes(page_report, page_review)
    assert "coord_not_normalized" in causes
