"""Tests for derive_app_eval_instructions."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import (  # noqa: E402
    AppEvalRedactionSettings,
    derive_app_eval_instructions,
    settings_from_legacy_redactor_kwargs,
    synthetic_doc_redact_tool_args,
)


def test_entities_and_deny_sorted_and_quoted():
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=["PHONE_NUMBER", "PERSON", "ADDRESS"],
        deny_list=["Mr Smith", "O'Brien"],
    )
    text = derive_app_eval_instructions(settings)
    assert "- Redact any instance of ADDRESS, PERSON, PHONE_NUMBER" in text
    assert "- Redact any instances of the term 'Mr Smith'" in text
    assert '"O\'Brien"' in text


def test_extract_text_only():
    settings = AppEvalRedactionSettings(text_extraction_only=True)
    text = derive_app_eval_instructions(settings)
    assert "Extract text only" in text
    assert "Redact any instance" not in text


def test_allow_list_bullet():
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=["PERSON"],
        allow_list=["Public Body"],
    )
    text = derive_app_eval_instructions(settings)
    assert "Do not redact" in text
    assert "Public Body" in text


def test_legacy_kwargs_mapping():
    flat = {
        "chosen_redact_entities": ["PERSON"],
        "in_deny_list": ["x"],
        "text_extraction_method": "PyMuPDF",
        "pii_identification_method": "Local",
        "text_extraction_only": False,
    }
    settings = settings_from_legacy_redactor_kwargs(flat)
    assert settings.merged_entities() == ["PERSON"]
    assert settings.deny_list == ["x"]
    tool = synthetic_doc_redact_tool_args(settings)
    assert tool["tool_calls"][0]["args"]["deny_list"] == ["x"]
