"""Tests for the pluggable agent runtime layer."""

from __future__ import annotations

from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

from agent_runtime import (  # noqa: E402
    AgentStreamEvent,
    PiAgentRuntime,
    coerce_agent_runtime,
    normalize_orchestrator,
    orchestrator_label,
)
from pi_rpc_client import PiStreamEvent  # noqa: E402


def test_normalize_orchestrator_defaults_to_pi(monkeypatch):
    monkeypatch.delenv("AGENT_ORCHESTRATOR", raising=False)
    assert normalize_orchestrator(None) == "pi"
    assert normalize_orchestrator("langgraph") == "langgraph"
    assert normalize_orchestrator("unknown") == "pi"


def test_orchestrator_label():
    assert orchestrator_label("langgraph") == "LangGraph"
    assert orchestrator_label("agentcore") == "Bedrock AgentCore Runtime"
    assert orchestrator_label("agentcore-harness") == "Bedrock AgentCore Harness"


def test_pi_event_mapping():
    from agent_runtime import _pi_event_to_agent_event

    mapped = _pi_event_to_agent_event(
        PiStreamEvent(kind="text_delta", text="hello", tool_name="bash")
    )
    assert mapped.kind == "text_delta"
    assert mapped.text == "hello"
    assert mapped.tool_name == "bash"


def test_coerce_agent_runtime_accepts_pi_adapter():
    class _FakePi:
        running = True

        def drain_pending_ui_history(self):
            return []

    wrapped = PiAgentRuntime(_FakePi())
    assert coerce_agent_runtime(wrapped) is wrapped


def test_agent_stream_event_fields():
    event = AgentStreamEvent(kind="done", text="Agent finished.")
    assert event.kind == "done"
    assert event.is_error is False
