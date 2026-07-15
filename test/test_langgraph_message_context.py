"""Tests for LangGraph message context trimming / compaction helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

from redaction_langgraph.message_context import (  # noqa: E402
    build_pre_model_hook,
    get_trim_stats,
    is_context_overflow_error,
    langgraph_compaction_enabled,
    langgraph_llm_input_max_tokens,
    reset_trim_stats,
    set_aggressive_trim,
    trim_messages_for_llm,
)


def _long_tool_history(n_rounds: int = 40) -> list:
    messages: list = [SystemMessage(content="You are a redaction assistant.")]
    messages.append(HumanMessage(content="Redact the uploaded PDF end to end."))
    blob = "x" * 4000
    for i in range(n_rounds):
        messages.append(
            AIMessage(
                content=f"Calling tool round {i}",
                tool_calls=[
                    {
                        "name": "read_workspace_text",
                        "args": {"relative_path": f"file_{i}.csv"},
                        "id": f"call_{i}",
                        "type": "tool_call",
                    }
                ],
            )
        )
        messages.append(
            ToolMessage(
                content=f"preview {i}: {blob}",
                tool_call_id=f"call_{i}",
                name="read_workspace_text",
            )
        )
    messages.append(HumanMessage(content="Continue the workflow."))
    return messages


def test_trim_preserves_system_and_shrinks(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "114688")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "28672")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "8192")
    messages = _long_tool_history(50)
    budget = 8_000
    before = count_tokens_approximately(messages)
    assert before > budget
    trimmed = trim_messages_for_llm(messages, max_tokens=budget)
    after = count_tokens_approximately(trimmed)
    assert len(trimmed) < len(messages)
    assert after <= budget + 500  # allow tiny approx overhead
    assert isinstance(trimmed[0], SystemMessage)
    assert "redaction assistant" in str(trimmed[0].content)


def test_aggressive_budget_smaller_than_normal(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "114688")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "28672")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "8192")
    normal = langgraph_llm_input_max_tokens(aggressive=False)
    aggressive = langgraph_llm_input_max_tokens(aggressive=True)
    floor = max(2_048, 114688 // 8)
    assert aggressive < normal
    assert aggressive >= floor
    assert aggressive == max(floor, normal // 2)


def test_langgraph_llm_input_max_tokens_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "65536")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "10000")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "4096")
    # 65536 - 10000 - 4096 = 51440
    assert langgraph_llm_input_max_tokens() == 51440


def test_pre_model_hook_returns_llm_input_messages(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "32000")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "8000")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "2048")
    reset_trim_stats()
    hook = build_pre_model_hook()
    messages = _long_tool_history(30)
    result = hook({"messages": messages})
    assert "llm_input_messages" in result
    assert "messages" not in result
    trimmed = result["llm_input_messages"]
    assert len(trimmed) < len(messages)
    stats = get_trim_stats()
    assert stats is not None
    assert stats.trimmed
    assert stats.messages_before == len(messages)
    assert stats.messages_after == len(trimmed)


def test_aggressive_hook_trims_more(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "64000")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "8000")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "2048")
    messages = _long_tool_history(40)
    normal = build_pre_model_hook(aggressive=False)({"messages": messages})[
        "llm_input_messages"
    ]
    aggressive = build_pre_model_hook(aggressive=True)({"messages": messages})[
        "llm_input_messages"
    ]
    assert count_tokens_approximately(aggressive) <= count_tokens_approximately(normal)
    assert len(aggressive) <= len(normal)


def test_thread_local_aggressive_override(monkeypatch):
    monkeypatch.setenv("AGENT_LLAMA_CONTEXT_WINDOW", "64000")
    monkeypatch.setenv("LANGGRAPH_COMPACTION_RESERVE_TOKENS", "8000")
    monkeypatch.setenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "2048")
    messages = _long_tool_history(40)
    hook = build_pre_model_hook(aggressive=False)
    set_aggressive_trim(False)
    normal_len = len(hook({"messages": messages})["llm_input_messages"])
    set_aggressive_trim(True)
    try:
        aggressive_len = len(hook({"messages": messages})["llm_input_messages"])
    finally:
        set_aggressive_trim(False)
    assert aggressive_len <= normal_len


def test_compaction_enabled_default_and_override(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_COMPACTION_ENABLED", raising=False)
    assert langgraph_compaction_enabled() is True
    monkeypatch.setenv("LANGGRAPH_COMPACTION_ENABLED", "false")
    assert langgraph_compaction_enabled() is False
    monkeypatch.setenv("LANGGRAPH_COMPACTION_ENABLED", "1")
    assert langgraph_compaction_enabled() is True


@pytest.mark.parametrize(
    "text",
    [
        "Error code: 400 - {'error': {'message': 'request (114870 tokens) exceeds the available context size (114688 tokens)', 'type': 'exceed_context_size_error'}}",
        "context_length_exceeded",
        "This model's maximum context length is 8192 tokens",
    ],
)
def test_is_context_overflow_error(text):
    assert is_context_overflow_error(RuntimeError(text)) is True


def test_is_context_overflow_error_negative():
    assert is_context_overflow_error(RuntimeError("connection refused")) is False
