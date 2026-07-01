"""Tests for LangGraph redaction tools (coverage, workspace I/O)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_AGENT_REDACT = Path(__file__).resolve().parents[1] / "agent-redact"
if str(_AGENT_REDACT) not in sys.path:
    sys.path.insert(0, str(_AGENT_REDACT))

from pi_test_support import ensure_gradio_importable

ensure_gradio_importable()

from redaction_langgraph.tools import (  # noqa: E402
    _discover_ocr_words_csv,
    read_workspace_text,
    write_workspace_text,
)
from redaction_langgraph.verify_coverage_lib import (  # noqa: E402
    compile_patterns,
    normalize_regex_patterns,
)


def test_discover_ocr_words_csv(tmp_path):
    review = tmp_path / "doc_review_file.csv"
    review.write_text("id,page\n", encoding="utf-8-sig")
    words = tmp_path / "doc_word_level_ocr.csv"
    words.write_text("word_text,page\n", encoding="utf-8-sig")
    assert _discover_ocr_words_csv(review) == words


def test_read_workspace_text_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    out = read_workspace_text("missing.csv", session_hash="sess")
    payload = json.loads(out)
    assert "error" in payload


def test_workspace_text_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    out = write_workspace_text("a.txt", "hello", session_hash="sess")
    assert json.loads(out)["written"] == "a.txt"
    assert read_workspace_text("a.txt", session_hash="sess") == "hello"


def test_normalize_regex_patterns_pipe_string():
    raw = r"Hyde|Lauren\s+Lilley|Lauren|Lilley|University\s+of\s+Notre\s+Dame|Notre\s+Dame"
    assert normalize_regex_patterns(raw) == [
        "Hyde",
        r"Lauren\s+Lilley",
        "Lauren",
        "Lilley",
        r"University\s+of\s+Notre\s+Dame",
        r"Notre\s+Dame",
    ]


def test_compile_patterns_accepts_pipe_string_not_characters():
    patterns = compile_patterns(r"Kornbluth|Poss\b")
    assert len(patterns) == 2
    assert patterns[0].search("Kornbluth")
    assert patterns[1].search("Poss")


def test_compile_patterns_user_pipe_string():
    raw = r"Hyde|Lauren|Lilley|University of Notre Dame|David R\."
    patterns = compile_patterns(raw)
    assert len(patterns) == 5
    assert patterns[0].search("Hyde")
    assert patterns[-1].search("David R.")


def test_compile_patterns_fallback_literal_on_invalid_regex():
    patterns = compile_patterns(r"bad[")
    assert len(patterns) == 1
    assert patterns[0].search("bad[")
