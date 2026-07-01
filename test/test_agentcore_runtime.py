"""Tests for AgentCore runtime URL parsing and response mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from agentcore_runtime import (  # noqa: E402
    AgentCoreAgentRuntime,
    agentcore_runtime_url,
    parse_agentcore_runtime_url,
)


def test_parse_agentcore_runtime_url_from_base():
    url = (
        "https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/"
        "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-2%3A404053085091%3Aruntime%2FRedactionAgent"
    )
    region, arn = parse_agentcore_runtime_url(url)
    assert region == "eu-west-2"
    assert (
        arn == "arn:aws:bedrock-agentcore:eu-west-2:404053085091:runtime/RedactionAgent"
    )


def test_parse_agentcore_runtime_url_strips_invocations_suffix():
    url = (
        "https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/"
        "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-2%3A404053085091%3Aruntime%2FRedactionAgent"
        "/invocations"
    )
    region, arn = parse_agentcore_runtime_url(url)
    assert region == "eu-west-2"
    assert arn.endswith("runtime/RedactionAgent")


def test_agentcore_runtime_url_strips_invocations(monkeypatch):
    monkeypatch.setenv(
        "AGENTCORE_RUNTIME_URL",
        "https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/arn%3Ax/invocations",
    )
    assert agentcore_runtime_url().endswith("arn%3Ax")
    assert not agentcore_runtime_url().endswith("/invocations")


def test_iter_json_response_result_field():
    runtime = AgentCoreAgentRuntime()
    events = list(runtime._iter_json_response(json.dumps({"result": "hello"}).encode()))
    assert len(events) == 1
    assert events[0].kind == "text_snapshot"
    assert events[0].text == "hello"


def test_map_message_update_tool_calls():
    runtime = AgentCoreAgentRuntime(session_hash="sess")
    event = {
        "type": "message_update",
        "role": "assistant",
        "content": "Running doc_redact.",
        "tool_calls": [{"name": "doc_redact", "args": {"pdf_relative_path": "a.pdf"}}],
    }
    kinds = [e.kind for e in runtime._map_agentcore_event(event)]
    assert kinds == ["tool_start", "text_snapshot"]


def test_map_message_update_tool_result():
    runtime = AgentCoreAgentRuntime(session_hash="sess")
    event = {
        "type": "message_update",
        "role": "tool",
        "tool_name": "doc_redact",
        "content": '{"message": "done"}',
    }
    events = list(runtime._map_agentcore_event(event))
    assert len(events) == 1
    assert events[0].kind == "tool_end"
    assert events[0].tool_name == "doc_redact"
