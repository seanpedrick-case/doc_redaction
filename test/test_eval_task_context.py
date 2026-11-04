"""Tests for multi-turn Gradio redaction task eval context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SHARED = _REPO / "agent-redact" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from agent_runtime import AgentStreamEvent  # noqa: E402
from eval_task_context import (  # noqa: E402
    append_turn_tool_calls,
    begin_redaction_task,
    get_active_task_for_eval,
    note_follow_up_or_steer_message,
)
from session_workspace import active_redaction_task_path  # noqa: E402


def test_begin_task_and_follow_up_steer_count(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    session = "abc123session"

    begin_redaction_task(session, "doc.pdf", "redact names")
    task = get_active_task_for_eval(session)
    assert task is not None
    assert task.follow_up_steer_count == 0
    assert task.document_name == "doc.pdf"

    assert note_follow_up_or_steer_message(session) is True
    assert note_follow_up_or_steer_message(session) is True
    task = get_active_task_for_eval(session)
    assert task is not None
    assert task.follow_up_steer_count == 2


def test_begin_task_resets_prior_steer_count(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    session = "resettest01"

    begin_redaction_task(session, "a.pdf", "task one")
    note_follow_up_or_steer_message(session)
    begin_redaction_task(session, "b.pdf", "task two")
    task = get_active_task_for_eval(session)
    assert task is not None
    assert task.follow_up_steer_count == 0
    assert task.document_name == "b.pdf"


def test_append_turn_tool_calls_accumulates(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    session = "tooltrace01"
    begin_redaction_task(session, "doc.pdf", "task")

    append_turn_tool_calls(
        session,
        [
            AgentStreamEvent(
                kind="tool_start",
                tool_name="doc_redact",
                tool_args={"x": 1},
            ),
        ],
    )
    append_turn_tool_calls(
        session,
        [
            AgentStreamEvent(
                kind="tool_start",
                tool_name="verify_coverage",
                tool_args={},
            ),
        ],
    )
    task = get_active_task_for_eval(session)
    assert task is not None
    assert [c["tool"] for c in task.cumulative_tool_calls] == [
        "doc_redact",
        "verify_coverage",
    ]


def test_active_redaction_task_path_stays_under_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "true")
    path = active_redaction_task_path("../../etc/passwd")
    assert path is not None
    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert path.name == "active_redaction_task.json"


def test_active_task_persisted_as_json(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    session = "persist01"
    begin_redaction_task(session, "doc.pdf", "instructions")
    note_follow_up_or_steer_message(session)

    path = tmp_path / session / "eval" / "active_redaction_task.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["follow_up_steer_count"] == 1
