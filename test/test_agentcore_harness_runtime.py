"""Tests for AgentCore Harness runtime helpers and stream mapping."""

from __future__ import annotations

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from agent_runtime import normalize_orchestrator  # noqa: E402
from agentcore_harness_runtime import (  # noqa: E402
    harness_runtime_session_id,
    map_harness_stream_event,
    parse_agentcore_harness_arn,
)


def test_harness_runtime_session_id_minimum_length():
    session_id = harness_runtime_session_id("abc")
    assert len(session_id) >= 33
    assert session_id == harness_runtime_session_id("abc")


def test_parse_agentcore_harness_arn():
    arn = "arn:aws:bedrock-agentcore:eu-west-2:404053085091:harness/MyHarness-abc123"
    region, parsed = parse_agentcore_harness_arn(arn)
    assert region == "eu-west-2"
    assert parsed == arn


def test_map_content_block_delta_text():
    events = list(
        map_harness_stream_event({"contentBlockDelta": {"delta": {"text": "Hello"}}})
    )
    assert len(events) == 1
    assert events[0].kind == "text_delta"
    assert events[0].text == "Hello"


def test_map_tool_use_start():
    events = list(
        map_harness_stream_event(
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "bash",
                            "input": {"command": "ls"},
                        }
                    }
                }
            }
        )
    )
    assert events[0].kind == "tool_start"
    assert events[0].tool_name == "bash"


def test_map_runtime_client_error():
    events = list(
        map_harness_stream_event({"runtimeClientError": {"message": "Harness failed"}})
    )
    assert events[0].kind == "error"
    assert events[0].is_error is True


def test_harness_s3_input_uri_explicit_prefix(monkeypatch):
    from harness_input_bridge import harness_s3_input_uri

    monkeypatch.setenv(
        "AGENTCORE_HARNESS_S3_INPUT_PREFIX",
        "s3://my-bucket/prefix",
    )
    bucket, key, uri = harness_s3_input_uri("sess-1", "doc.pdf")
    assert bucket == "my-bucket"
    assert key == "prefix/doc.pdf"
    assert uri == "s3://my-bucket/prefix/doc.pdf"


def test_normalize_orchestrator_harness_alias():
    import os

    prev = os.environ.get("AGENT_ORCHESTRATOR")
    try:
        os.environ["AGENT_ORCHESTRATOR"] = "harness"
        assert normalize_orchestrator() == "agentcore-harness"
    finally:
        if prev is None:
            os.environ.pop("AGENT_ORCHESTRATOR", None)
        else:
            os.environ["AGENT_ORCHESTRATOR"] = prev
