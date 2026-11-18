"""Tests for partial gold policy scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"
_REPO = _EVAL.parents[1]
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import AppEvalRedactionSettings  # noqa: E402
from gold_policy_scaffold import (  # noqa: E402
    build_partial_must_redact_lines,
    write_partial_policy_files,
)


def test_deny_list_in_must_redact(tmp_path):
    settings = AppEvalRedactionSettings(
        deny_list=["Sister City"],
        chosen_redact_entities=["CUSTOM"],
    )
    lines = build_partial_must_redact_lines(
        settings,
        repo_root=_REPO,
        input_pdf_basename="Partnership-Agreement-Toolkit_0_0.pdf",
    )
    body = "\n".join(lines)
    assert "Sister\\ City" in body or "Sister City" in body
    assert "deny list" in body


def test_llm_literals_and_write_files(tmp_path):
    settings = AppEvalRedactionSettings(
        custom_llm_instructions="Redact Lauren's name and university names.",
        chosen_redact_entities=["CUSTOM"],
    )
    write_partial_policy_files(
        tmp_path,
        settings,
        repo_root=_REPO,
        input_pdf_basename="example_of_emails_sent_to_a_professor_before_applying.pdf",
    )
    must = (tmp_path / "must_redact.txt").read_text(encoding="utf-8")
    assert "Lauren" in must
    assert (tmp_path / "must_redact.txt").is_file()
