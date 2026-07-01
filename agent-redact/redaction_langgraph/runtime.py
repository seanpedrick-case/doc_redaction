"""LangGraph :class:`AgentRuntime` implementation for the Gradio UI."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_AGENT_REDACT_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_REDACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_REDACT_ROOT))

_PI_DIR = _AGENT_REDACT_ROOT / "pi"
if str(_PI_DIR) not in sys.path:
    sys.path.insert(0, str(_PI_DIR))

from agent_runtime import (
    AgentRuntime,
    AgentRuntimeError,
    AgentStreamEvent,
)  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from redaction_langgraph.graph import build_redaction_agent  # noqa: E402


class LangGraphAgentRuntime(AgentRuntime):
    """Session-scoped LangGraph ReAct agent (curated tools, no shell)."""

    def __init__(self, *, session_hash: str | None = None) -> None:
        self._session_hash = session_hash
        self._graph: Any = None
        self._system_message: Any = None
        self._messages: list[Any] = []
        self._running = False
        self._prompt_stream_depth = 0
        self._abort_requested = False
        self._lock = threading.Lock()
        self._pending_ui_notices: list[dict[str, Any]] = []
        self._pending_ui_history: list[dict[str, Any]] = []

    @property
    def orchestrator(self) -> str:
        return "langgraph"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def prompt_stream_active(self) -> bool:
        return self._prompt_stream_depth > 0

    def start(self) -> None:
        if self._graph is None:
            self._graph, self._system_message = build_redaction_agent(
                self._session_hash
            )
        self._running = True

    def close(self) -> None:
        self._running = False
        self._graph = None
        self._messages = []

    def abort(self) -> None:
        self._abort_requested = True

    def new_session(self) -> None:
        self._messages = []
        self._abort_requested = False

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        os.environ["PI_DEFAULT_PROVIDER"] = provider
        os.environ["PI_DEFAULT_MODEL"] = model_id
        if provider == "llama-cpp":
            os.environ["PI_LLAMA_MODEL_ID"] = model_id
        self._graph = None
        self.start()
        return {"provider": provider, "model": model_id}

    def apply_backend(self, provider: str, model_id: str) -> None:
        self.set_model(provider, model_id)
        self.new_session()

    def get_state(self) -> dict[str, Any]:
        return {
            "isStreaming": self.prompt_stream_active,
            "isCompacting": False,
            "provider": os.environ.get("PI_DEFAULT_PROVIDER"),
            "model": {
                "provider": os.environ.get("PI_DEFAULT_PROVIDER"),
                "id": os.environ.get("PI_DEFAULT_MODEL")
                or os.environ.get("PI_LLAMA_MODEL_ID"),
            },
        }

    def get_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in self._messages:
            if isinstance(message, HumanMessage):
                out.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                out.append({"role": "assistant", "content": str(message.content or "")})
        return out

    def stage_ui_chat_notice(self, label: str, message: str) -> None:
        text = message.strip()
        if not text:
            return
        self._pending_ui_history.append(
            {"role": "user", "content": f"_**{label}:**_ {text}"}
        )
        self._pending_ui_history.append({"role": "assistant", "content": ""})

    def drain_pending_ui_history(self) -> list[dict[str, Any]]:
        pending = self._pending_ui_history[:]
        self._pending_ui_history.clear()
        return pending

    def prompt_events(self, message: str) -> Iterator[AgentStreamEvent]:
        self._prompt_stream_depth += 1
        self._abort_requested = False
        try:
            if not self._running:
                self.start()
            if self._graph is None:
                raise AgentRuntimeError("LangGraph agent is not initialized.")

            from redaction_langgraph.graph import graph_recursion_limit

            yield AgentStreamEvent(kind="status", text="LangGraph agent started…")
            inputs = {
                "messages": [
                    self._system_message,
                    *self._messages,
                    HumanMessage(content=message),
                ]
            }
            self._messages.append(HumanMessage(content=message))

            assistant_chunks: list[str] = []
            stream_config = {"recursion_limit": graph_recursion_limit()}
            for event in self._graph.stream(
                inputs, stream_mode="updates", config=stream_config
            ):
                if self._abort_requested:
                    yield AgentStreamEvent(kind="done", text="Agent aborted.")
                    return
                for _node, update in event.items():
                    messages = update.get("messages") or []
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            text = self._stringify_content(msg.content)
                            if text:
                                assistant_chunks.append(text)
                                yield AgentStreamEvent(kind="text_snapshot", text=text)
                            for call in msg.tool_calls or []:
                                name = str(call.get("name") or "tool")
                                args = (
                                    call.get("args")
                                    if isinstance(call.get("args"), dict)
                                    else {}
                                )
                                yield AgentStreamEvent(
                                    kind="tool_start",
                                    tool_name=name,
                                    tool_args=args,
                                    text=name,
                                )
                        elif isinstance(msg, ToolMessage):
                            yield AgentStreamEvent(
                                kind="tool_end",
                                tool_name=str(msg.name or "tool"),
                                tool_output=str(msg.content or ""),
                                is_error=False,
                            )
            if assistant_chunks:
                self._messages.append(AIMessage(content="\n".join(assistant_chunks)))
            yield AgentStreamEvent(kind="done", text="Agent finished.")
        finally:
            self._prompt_stream_depth = max(0, self._prompt_stream_depth - 1)

    @staticmethod
    def _stringify_content(content: Any) -> str:
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
