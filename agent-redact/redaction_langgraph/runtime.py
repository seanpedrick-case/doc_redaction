"""LangGraph :class:`AgentRuntime` implementation for the Gradio UI."""

from __future__ import annotations

import json
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

_WORKFLOW_CONTINUE_PROMPT = """Pass 1 redaction is NOT complete yet. Continue now:
1. Edit the *_review_file.csv for the user requirements (write_workspace_text or run_workspace_python_script)
2. Run verify_coverage until pass_strict is true
3. Run review_apply once on the source PDF and edited review CSV
Call the next required tool — do not stop after read_workspace_text or write_workspace_text."""


def _last_written_python_script(tool_outputs: list[tuple[str, str]]) -> str | None:
    for name, output in reversed(tool_outputs):
        if name != "write_workspace_text":
            continue
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            continue
        written = str(data.get("written") or "")
        if written.lower().endswith(".py"):
            return written
    return None


def _build_workflow_continue_prompt(
    tool_names_seen: set[str],
    tool_outputs: list[tuple[str, str]],
) -> str:
    if (
        "write_workspace_text" in tool_names_seen
        and "run_workspace_python_script" not in tool_names_seen
    ):
        script_path = _last_written_python_script(tool_outputs)
        if script_path:
            return (
                "Pass 1 is NOT complete. The Python script is already saved at "
                f"`{script_path}` — do NOT call write_workspace_text again. "
                f"Call run_workspace_python_script with relative_path={script_path!r} "
                "now, then verify_coverage and review_apply."
            )
    return _WORKFLOW_CONTINUE_PROMPT


def _langgraph_auto_continue_enabled() -> bool:
    return os.environ.get(
        "LANGGRAPH_AUTO_CONTINUE_WORKFLOW", "true"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _langgraph_max_continuations() -> int:
    raw = os.environ.get("LANGGRAPH_WORKFLOW_CONTINUATIONS", "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _redaction_workflow_incomplete(tool_names: set[str]) -> bool:
    return "doc_redact" in tool_names and "review_apply" not in tool_names


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

    def _yield_message_updates(
        self,
        msg: Any,
        *,
        assistant_chunks: list[str],
        tool_names_seen: set[str],
        tool_outputs: list[tuple[str, str]],
    ) -> Iterator[AgentStreamEvent]:
        if isinstance(msg, AIMessage):
            text = self._stringify_content(msg.content)
            if text:
                assistant_chunks.append(text)
                yield AgentStreamEvent(kind="text_snapshot", text=text)
            for call in msg.tool_calls or []:
                name = str(call.get("name") or "tool")
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                yield AgentStreamEvent(
                    kind="tool_start",
                    tool_name=name,
                    tool_args=args,
                    text=name,
                )
        elif isinstance(msg, ToolMessage):
            name = str(msg.name or "tool")
            tool_names_seen.add(name)
            output = str(msg.content or "")
            tool_outputs.append((name, output))
            yield AgentStreamEvent(
                kind="tool_end",
                tool_name=name,
                tool_output=output,
                is_error=False,
            )

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
            graph_messages: list[Any] = [
                self._system_message,
                *self._messages,
                HumanMessage(content=message),
            ]
            self._messages.append(HumanMessage(content=message))

            assistant_chunks: list[str] = []
            tool_names_seen: set[str] = set()
            tool_outputs: list[tuple[str, str]] = []
            stream_config = {"recursion_limit": graph_recursion_limit()}
            max_rounds = 1 + (
                _langgraph_max_continuations()
                if _langgraph_auto_continue_enabled()
                else 0
            )

            for round_idx in range(max_rounds):
                if round_idx > 0:
                    yield AgentStreamEvent(
                        kind="status",
                        text="Pass 1 incomplete — nudging agent to continue workflow…",
                    )
                for event in self._graph.stream(
                    {"messages": graph_messages},
                    stream_mode="updates",
                    config=stream_config,
                ):
                    if self._abort_requested:
                        yield AgentStreamEvent(kind="done", text="Agent aborted.")
                        return
                    for _node, update in event.items():
                        for msg in update.get("messages") or []:
                            graph_messages.append(msg)
                            yield from self._yield_message_updates(
                                msg,
                                assistant_chunks=assistant_chunks,
                                tool_names_seen=tool_names_seen,
                                tool_outputs=tool_outputs,
                            )
                if not _redaction_workflow_incomplete(tool_names_seen):
                    break
                if round_idx >= max_rounds - 1:
                    break
                graph_messages.append(
                    HumanMessage(
                        content=_build_workflow_continue_prompt(
                            tool_names_seen, tool_outputs
                        )
                    )
                )

            if assistant_chunks:
                self._messages.append(AIMessage(content="\n".join(assistant_chunks)))
            workflow_incomplete = _redaction_workflow_incomplete(tool_names_seen)
            done_text = "Agent finished."
            if workflow_incomplete:
                done_text = (
                    "Agent finished (Pass 1 incomplete — review_apply not run; "
                    "use **Send** to continue or restart the task)."
                )
            yield AgentStreamEvent(
                kind="done",
                text=done_text,
                meta={"workflow_incomplete": workflow_incomplete},
            )
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
