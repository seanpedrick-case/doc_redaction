"""Tests for wiring harness outputs into Gradio tabs."""

from __future__ import annotations

from pathlib import Path

from tools.harness_gradio_review_wire import classify_agent_output_paths


def test_classify_agent_output_paths(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    review = tmp_path / "doc.pdf_review_file.csv"
    lines = tmp_path / "doc.pdf_ocr_output_local_ocr.csv"
    words = tmp_path / "doc.pdf_ocr_results_with_words_local_ocr.csv"
    overlay = tmp_path / "doc_redactions_for_review.pdf"
    for p in (source, review, lines, words, overlay):
        p.write_text("x", encoding="utf-8")

    artifacts = classify_agent_output_paths(
        [str(review), str(lines), str(words), str(overlay)],
        source_hint=str(source),
    )
    assert artifacts.source_pdf == str(source)
    assert artifacts.review_csv == str(review)
    assert artifacts.ocr_lines_csv == str(lines)
    assert artifacts.ocr_words_csv == str(words)
    assert artifacts.review_overlay_pdf == str(overlay)
