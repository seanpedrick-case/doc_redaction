"""Classify inference-server errors for LangGraph retry paths (no LangChain import)."""

from __future__ import annotations


def is_context_overflow_error(exc: BaseException) -> bool:
    """True when llama.cpp / OpenAI-compatible APIs reject an oversized prompt."""
    text = str(exc).lower()
    markers = (
        "exceed_context_size_error",
        "exceeds the available context size",
        "context length",
        "maximum context length",
        "prompt is too long",
        "n_prompt_tokens",
        "context_length_exceeded",
    )
    return any(marker in text for marker in markers)


def is_tool_call_json_parse_error(exc: BaseException) -> bool:
    """True when the inference server rejects malformed tool-call argument JSON.

    Local models often fail when ``write_workspace_text`` content is huge or
    contains nested quotes that break the tool-call JSON envelope (truncated or
    unescaped strings). The failure happens *before* LangGraph tools run.
    """
    text = str(exc).lower()
    markers = (
        "failed to parse tool call arguments as json",
        "parse tool call arguments",
        "tool call arguments as json",
        "invalid string: missing closing quote",
        "json.exception.parse_error",
    )
    return any(marker in text for marker in markers)
