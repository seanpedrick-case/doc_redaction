"""Tests for LangGraph workflow-incomplete / auto-continue helpers."""

from __future__ import annotations

import json

import pytest
from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

from redaction_langgraph.workflow_continue import (  # noqa: E402
    build_identical_error_breaker_prompt,
    build_tool_call_json_retry_prompt,
    build_workflow_continue_prompt,
    consecutive_python_writes_without_run,
    identical_tool_error_streak,
    redaction_workflow_incomplete,
    tool_output_error_signature,
)


def _write_out(written: str) -> str:
    return json.dumps({"written": written, "bytes": 10})


def test_incomplete_false_for_explore_only():
    tools = {"list_workspace_files", "read_workspace_text"}
    outputs = [
        ("list_workspace_files", "[]"),
        ("read_workspace_text", "page text"),
    ]
    assert redaction_workflow_incomplete(tools, outputs) is False


def test_incomplete_pass1_doc_redact_without_apply():
    assert redaction_workflow_incomplete({"doc_redact"}, []) is True
    assert redaction_workflow_incomplete({"doc_redact", "review_apply"}, []) is False


def test_incomplete_followup_pending_python_script():
    tools = {"list_workspace_files", "read_workspace_text", "write_workspace_text"}
    outputs = [
        ("list_workspace_files", "[]"),
        ("read_workspace_text", "ocr"),
        ("write_workspace_text", _write_out("redact/doc/output_redact/extract.py")),
    ]
    assert redaction_workflow_incomplete(tools, outputs) is True
    prompt = build_workflow_continue_prompt(tools, outputs)
    assert "extract.py" in prompt
    assert "run_workspace_python_script" in prompt


def test_incomplete_false_after_script_run_and_apply():
    tools = {
        "write_workspace_text",
        "run_workspace_python_script",
        "review_apply",
    }
    outputs = [
        ("write_workspace_text", _write_out("fix_policy.py")),
        ("run_workspace_python_script", '{"ok": true}'),
        ("review_apply", '{"ok": true}'),
    ]
    assert redaction_workflow_incomplete(tools, outputs) is False


def test_incomplete_after_script_run_without_apply():
    tools = {"write_workspace_text", "run_workspace_python_script"}
    outputs = [
        ("write_workspace_text", _write_out("fix_policy.py")),
        ("run_workspace_python_script", '{"ok": true}'),
    ]
    assert redaction_workflow_incomplete(tools, outputs) is True


def test_incomplete_after_review_csv_write_without_apply():
    tools = {"write_workspace_text"}
    outputs = [
        (
            "write_workspace_text",
            _write_out("redact/doc/output_redact/doc_review_file.csv"),
        )
    ]
    assert redaction_workflow_incomplete(tools, outputs) is True


def test_langgraph_trace_config_includes_session_metadata():
    from eval.arize_monitoring import arize_session_id, langgraph_trace_config

    assert arize_session_id(None) is None
    assert arize_session_id("  ") is None
    assert arize_session_id("abc123") == "abc123"

    bare = langgraph_trace_config(None, recursion_limit=50)
    assert bare == {"recursion_limit": 50}

    cfg = langgraph_trace_config("sess-1", recursion_limit=25)
    assert cfg["recursion_limit"] == 25
    assert cfg["metadata"]["session_id"] == "sess-1"
    assert cfg["metadata"]["thread_id"] == "sess-1"
    assert cfg["configurable"]["thread_id"] == "sess-1"


def test_incomplete_false_for_plain_text_write():
    tools = {"write_workspace_text"}
    outputs = [("write_workspace_text", _write_out("notes.txt"))]
    assert redaction_workflow_incomplete(tools, outputs) is False


def test_continue_prompt_generic_when_no_pending_script():
    tools = {"run_workspace_python_script"}
    outputs = [("run_workspace_python_script", '{"ok": true}')]
    prompt = build_workflow_continue_prompt(tools, outputs)
    assert "review_apply" in prompt
    assert "NOT complete" in prompt
    assert "compact" in prompt.lower() or "hard-coded" in prompt.lower()


def test_tool_call_json_retry_prompt_asks_for_compact_script():
    prompt = build_tool_call_json_retry_prompt()
    assert "JSON" in prompt
    assert "write_workspace_text" in prompt
    assert "run_workspace_python_script" in prompt
    assert "compact" in prompt.lower() or "SHORT" in prompt


def test_identical_tool_error_streak_detects_repeat():
    err = json.dumps({"error": "pdf_relative_path must be a plain string"})
    outputs = [
        ("doc_redact", err),
        ("doc_redact", err),
    ]
    streak = identical_tool_error_streak(outputs, min_streak=2)
    assert streak == ("doc_redact", tool_output_error_signature(err))


def test_identical_tool_error_streak_ignores_success():
    outputs = [
        ("doc_redact", json.dumps({"error": "bad"})),
        ("doc_redact", json.dumps({"ok": True})),
    ]
    assert identical_tool_error_streak(outputs, min_streak=2) is None


def test_continue_prompt_uses_error_breaker_on_streak():
    err = json.dumps({"error": "Tool relative_path is empty."})
    outputs = [
        ("write_workspace_text", err),
        ("write_workspace_text", err),
    ]
    prompt = build_workflow_continue_prompt({"write_workspace_text"}, outputs)
    assert "STOP" in prompt
    assert "flat" in prompt.lower() or "nested" in prompt.lower()
    assert "write_workspace_text" in prompt


def test_build_identical_error_breaker_prompt_includes_error():
    err = json.dumps({"error": "nested object not allowed"})
    prompt = build_identical_error_breaker_prompt(
        "doc_redact",
        [("doc_redact", err)],
    )
    assert "doc_redact" in prompt
    assert "nested object not allowed" in prompt


def test_consecutive_python_writes_without_run():
    outputs = [
        ("write_workspace_text", _write_out("a.py")),
        ("write_workspace_text", _write_out("a.py")),
        ("write_workspace_text", _write_out("a.py")),
    ]
    path, count = consecutive_python_writes_without_run(outputs)
    assert path == "a.py"
    assert count == 3


def test_continue_prompt_write_storm_nudge():
    outputs = [
        ("write_workspace_text", _write_out("fix_policy.py")),
        ("write_workspace_text", _write_out("fix_policy.py")),
    ]
    prompt = build_workflow_continue_prompt({"write_workspace_text"}, outputs)
    assert "run_workspace_python_script" in prompt
    assert "fix_policy.py" in prompt


@pytest.mark.parametrize(
    "text",
    [
        "Error code: 500 - {'error': {'code': 500, 'message': 'Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 6673: syntax error while parsing value - invalid string: missing closing quote', 'type': 'server_error'}}",
        "failed to parse tool call arguments as json",
        "json.exception.parse_error.101",
        "invalid string: missing closing quote",
    ],
)
def test_is_tool_call_json_parse_error(text):
    from redaction_langgraph.llm_errors import is_tool_call_json_parse_error

    assert is_tool_call_json_parse_error(RuntimeError(text)) is True


def test_is_tool_call_json_parse_error_negative():
    from redaction_langgraph.llm_errors import is_tool_call_json_parse_error

    assert is_tool_call_json_parse_error(RuntimeError("connection refused")) is False
    assert (
        is_tool_call_json_parse_error(RuntimeError("context_length_exceeded")) is False
    )
