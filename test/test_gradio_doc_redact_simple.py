"""Tests for tools.simplified_api doc_redact helper."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_pdf() -> Path:
    return (
        REPO_ROOT
        / "example_data"
        / "example_of_emails_sent_to_a_professor_before_applying.pdf"
    )


def test_doc_redact_simple_rejects_missing_file() -> None:
    from tools.simplified_api import redact_document_from_upload_for_gradio_api

    with pytest.raises(ValueError, match="not found|missing"):
        redact_document_from_upload_for_gradio_api(
            str(REPO_ROOT / "does_not_exist_12345.pdf")
        )


def test_doc_redact_simple_calls_cli_and_returns_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.config import OUTPUT_FOLDER
    from tools.simplified_api import redact_document_from_upload_for_gradio_api

    pdf = _example_pdf()
    if not pdf.is_file():
        pytest.skip("fixture missing")

    safe_out = Path(OUTPUT_FOLDER).resolve()
    safe_out.mkdir(parents=True, exist_ok=True)
    expected = safe_out / "mock_redacted.pdf"
    expected.write_bytes(b"%PDF-1.4\n%mock\n")

    def _mock_redact_document(*args, **kwargs):
        return [str(expected)]

    monkeypatch.setattr("doc_redaction.cli_api.redact_document", _mock_redact_document)

    paths, msg = redact_document_from_upload_for_gradio_api(
        str(pdf),
        redact_entities=["PERSON"],
    )
    assert paths == [str(expected)]
    assert msg
