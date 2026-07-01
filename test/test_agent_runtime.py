"""Tests for the pluggable agent runtime layer."""

from __future__ import annotations

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from agent_runtime import (  # noqa: E402
    AgentStreamEvent,
    PiAgentRuntime,
    coerce_agent_runtime,
    normalize_orchestrator,
    orchestrator_label,
)
from pi_rpc_client import PiStreamEvent  # noqa: E402


def test_normalize_orchestrator_defaults_to_pi():
    assert normalize_orchestrator(None) == "pi"
    assert normalize_orchestrator("langgraph") == "langgraph"
    assert normalize_orchestrator("unknown") == "pi"


def test_orchestrator_label():
    assert orchestrator_label("langgraph") == "LangGraph"
    assert orchestrator_label("agentcore") == "Bedrock AgentCore"


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
