"""Tests for Textract vs CUSTOM_VLM_SIGNATURE priority."""

from __future__ import annotations

import pytest

pytest.importorskip("pikepdf")

from tools.aws_textract import textract_prioritizes_signature_extraction
from tools.config import TEXTRACT_TEXT_EXTRACT_OPTION


def test_textract_prioritizes_signature_when_extract_signatures_selected() -> None:
    assert textract_prioritizes_signature_extraction(
        TEXTRACT_TEXT_EXTRACT_OPTION,
        ["Extract handwriting", "Extract signatures"],
    )


def test_textract_does_not_prioritize_without_extract_signatures() -> None:
    assert not textract_prioritizes_signature_extraction(
        TEXTRACT_TEXT_EXTRACT_OPTION,
        ["Extract handwriting"],
    )
    assert not textract_prioritizes_signature_extraction(
        TEXTRACT_TEXT_EXTRACT_OPTION,
        [],
    )


def test_non_textract_never_prioritizes_textract_signatures() -> None:
    assert not textract_prioritizes_signature_extraction(
        "Local OCR",
        ["Extract signatures"],
    )
