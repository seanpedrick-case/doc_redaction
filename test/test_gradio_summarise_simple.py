"""Tests for tools.simplified_api summarise helper (no LLM calls in default tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_summarise_simple_rejects_missing_pdf() -> None:
    from tools.simplified_api import summarise_document_from_upload_for_gradio_api

    with pytest.raises(ValueError, match="not found|missing"):
        summarise_document_from_upload_for_gradio_api(
            str(REPO_ROOT / "nonexistent_document_12345.pdf")
        )


def test_summarise_simple_rejects_non_pdf() -> None:
    from tools.simplified_api import summarise_document_from_upload_for_gradio_api

    csv_path = REPO_ROOT / "example_data" / "combined_case_notes.csv"
    if not csv_path.is_file():
        pytest.skip("fixture missing")
    with pytest.raises(ValueError, match="expects a PDF"):
        summarise_document_from_upload_for_gradio_api(str(csv_path))
