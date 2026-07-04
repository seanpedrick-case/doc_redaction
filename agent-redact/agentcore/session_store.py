"""In-process conversation history for AgentCore / LangGraph orchestration."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_lock = threading.Lock()
_sessions: dict[str, list[BaseMessage]] = {}


def _session_key(session_hash: str | None) -> str:
    key = (session_hash or "").strip()
    return key or "default"


def get_messages(session_hash: str | None) -> list[BaseMessage]:
    """Return a copy of stored messages for *session_hash*."""
    key = _session_key(session_hash)
    with _lock:
        return list(_sessions.get(key, []))


def clear_session(session_hash: str | None) -> None:
    """Drop conversation history for *session_hash*."""
    key = _session_key(session_hash)
    with _lock:
        _sessions.pop(key, None)


def append_turn(
    session_hash: str | None,
    *,
    user_text: str,
    assistant_text: str = "",
) -> None:
    """Append one user turn and optional assistant reply."""
    key = _session_key(session_hash)
    with _lock:
        history = _sessions.setdefault(key, [])
        history.append(HumanMessage(content=user_text))
        if assistant_text.strip():
            history.append(AIMessage(content=assistant_text.strip()))


def stringify_message_content(content: Any) -> str:
    """Normalize LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")
