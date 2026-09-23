"""Tests for AgentCore runtime URL parsing and response mapping."""

from __future__ import annotations

import json

from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

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


def test_agentcore_runtime_session_id_is_long_and_stable():
    from agentcore_runtime import agentcore_runtime_session_id

    first = agentcore_runtime_session_id("abc")
    second = agentcore_runtime_session_id("abc")
    other = agentcore_runtime_session_id("xyz")
    assert first == second
    assert first != other
    assert len(first) >= 33


def test_iter_sse_response_accepts_bare_ndjson():
    runtime = AgentCoreAgentRuntime(session_hash="sess")
    lines = [
        json.dumps({"type": "status", "message": "Working…"}),
        "data: " + json.dumps({"type": "agent_end", "message": "done"}),
    ]
    kinds = [event.kind for event in runtime._iter_sse_response(lines)]
    assert kinds == ["status", "status"]


def test_bedrock_agentcore_client_sets_long_read_timeout(monkeypatch):

    created: dict = {}

    class _FakeSession:
        def __init__(self, region_name=None):
            self.region_name = region_name

        def client(self, service_name, region_name=None, config=None):
            if service_name == "sts":
                return type("STS", (), {"get_caller_identity": lambda self: {}})()
            created["service"] = service_name
            created["region"] = region_name
            created["config"] = config
            return object()

    monkeypatch.setattr("boto3.Session", _FakeSession)
    monkeypatch.setattr("pi_agent_config.configure_aws_credentials", lambda: None)
    monkeypatch.setenv("AGENTCORE_BOTO_READ_TIMEOUT_S", "1234")
    monkeypatch.setenv("AGENTCORE_BOTO_CONNECT_TIMEOUT_S", "12")

    # Re-import timeouts after env change — module constants are set at import.
    import importlib

    import agentcore_boto as boto_mod

    importlib.reload(boto_mod)
    boto_mod.bedrock_agentcore_client("eu-west-2")
    assert created["service"] == "bedrock-agentcore"
    assert created["region"] == "eu-west-2"
    assert created["config"].read_timeout == 1234.0
    assert created["config"].connect_timeout == 12.0
