"""Tests for LangGraph workflow-incomplete / auto-continue helpers."""

from __future__ import annotations

import json

from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

from redaction_langgraph.workflow_continue import (  # noqa: E402
    build_workflow_continue_prompt,
    redaction_workflow_incomplete,
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
