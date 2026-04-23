"""Tests for headless /agent apply_review_redactions orchestration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_pdf() -> Path:
    return (
        REPO_ROOT
        / "example_data"
        / "example_of_emails_sent_to_a_professor_before_applying.pdf"
    )


def _example_review_csv() -> Path:
    return (
        REPO_ROOT
        / "example_data"
        / "example_outputs"
        / "example_of_emails_sent_to_a_professor_before_applying_review_file.csv"
    )


@pytest.fixture
def tmp_out_dir() -> str:
    # Security: `run_apply_review_redactions()` now enforces output_dir is under OUTPUT_FOLDER.
    from tools.config import OUTPUT_FOLDER

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    return tempfile.mkdtemp(prefix="test_agent_apply_", dir=OUTPUT_FOLDER)


def test_run_apply_review_redactions_writes_redacted_pdf(tmp_out_dir: str) -> None:
    pdf = _example_pdf()
    csv = _example_review_csv()
    if not pdf.is_file() or not csv.is_file():
        pytest.skip("example_data fixtures not present")

    from tools.simplified_api import run_apply_review_redactions

    result = run_apply_review_redactions(
        pdf_path=str(pdf),
        review_csv_path=str(csv),
        output_dir=tmp_out_dir,
    )
    paths = result.get("output_paths") or []
    out_dir = str(result.get("output_dir") or tmp_out_dir)
    safe_out_root = os.path.realpath(out_dir)
    redacted = [p for p in paths if p.endswith("_redacted.pdf")]
    assert redacted, f"expected *_redacted.pdf in {paths!r}"
    assert all(
        (
            (resolved_p := os.path.realpath(p))
            and os.path.commonpath([safe_out_root, resolved_p]) == safe_out_root
            and Path(resolved_p).is_file()
        )
        for p in redacted
    )


def test_gradio_api_wrapper_accepts_string_paths(tmp_out_dir: str) -> None:
    pdf = _example_pdf()
    csv = _example_review_csv()
    if not pdf.is_file() or not csv.is_file():
        pytest.skip("example_data fixtures not present")

    from tools.simplified_api import (
        apply_review_redactions_from_uploads_for_gradio_api,
    )

    paths, msg = apply_review_redactions_from_uploads_for_gradio_api(
        str(pdf), str(csv), tmp_out_dir
    )
    assert isinstance(paths, list)
    assert any(p.endswith("_redacted.pdf") for p in paths)
    assert msg


def test_review_csv_basename_must_contain_review_file(tmp_out_dir: str) -> None:
    pdf = _example_pdf()
    wrong_csv = REPO_ROOT / "example_data" / "combined_case_notes.csv"
    if not pdf.is_file() or not wrong_csv.is_file():
        pytest.skip("example_data fixtures not present")

    from tools.simplified_api import run_apply_review_redactions

    with pytest.raises(ValueError, match="_review_file"):
        run_apply_review_redactions(
            pdf_path=str(pdf),
            review_csv_path=str(wrong_csv),
            output_dir=tmp_out_dir,
        )
