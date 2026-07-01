"""Pluggable agent orchestration runtimes for the Gradio agentic UI."""

from __future__ import annotations

import os
import queue
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_REDACT_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_REDACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_REDACT_ROOT))


class AgentRuntimeError(RuntimeError):
    """Base error for agent runtime failures."""


@dataclass
class AgentStreamEvent:
    """Normalized streaming event for Gradio chat/activity panels."""

    kind: str
    text: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def normalize_orchestrator(raw: str | None = None) -> str:
    """Return a supported orchestrator id: pi | langgraph | agentcore | agentcore-harness."""
    value = (raw or os.environ.get("AGENT_ORCHESTRATOR") or "pi").strip().lower()
    if value == "harness":
        value = "agentcore-harness"
    if value in {"pi", "langgraph", "agentcore", "agentcore-harness"}:
        return value
    return "pi"


def orchestrator_label(orchestrator: str | None = None) -> str:
    labels = {
        "pi": "Pi coding agent",
        "langgraph": "LangGraph",
        "agentcore": "Bedrock AgentCore Runtime",
        "agentcore-harness": "Bedrock AgentCore Harness",
    }
    return labels.get(normalize_orchestrator(orchestrator), "Agent")


class AgentRuntime(ABC):
    """Common interface consumed by ``gradio_app.py``."""

    @property
    @abstractmethod
    def orchestrator(self) -> str:
        """Runtime id: pi | langgraph | agentcore | agentcore-harness."""

    @property
    @abstractmethod
    def running(self) -> bool:
        """True when the runtime is ready to accept prompts."""

    @property
    def prompt_stream_active(self) -> bool:
        """True while :meth:`prompt_events` is consuming a prompt stream."""
        return False

    @abstractmethod
    def start(self) -> None:
        """Start or warm the runtime."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the runtime."""

    @abstractmethod
    def abort(self) -> None:
        """Request cancellation of the active turn."""

    @abstractmethod
    def prompt_events(self, message: str) -> Iterator[AgentStreamEvent]:
        """Stream normalized events for one user prompt."""

    def get_state(self) -> dict[str, Any]:
        return {}

    def get_messages(self) -> list[dict[str, Any]]:
        return []

    def get_session_stats(self) -> dict[str, Any]:
        return {}

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        return {}

    def new_session(self) -> None:
        return None

    def steer(self, message: str) -> None:
        return None

    def follow_up(self, message: str) -> None:
        return None

    def stage_ui_chat_notice(self, label: str, message: str) -> None:
        return None

    def take_pending_ui_chat_notices(self) -> list[dict[str, Any]]:
        return []

    def drain_pending_ui_history(self) -> list[dict[str, Any]]:
        return []

    def apply_backend(self, provider: str, model_id: str) -> None:
        """Reconfigure the orchestration model after UI **Apply backend**."""
        self.set_model(provider, model_id)
        self.new_session()


class PiAgentRuntime(AgentRuntime):
    """Adapter around :class:`pi_rpc_client.PiRpcClient`."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def orchestrator(self) -> str:
        return "pi"

    @property
    def client(self) -> Any:
        return self._client

    @property
    def running(self) -> bool:
        return bool(self._client.running)

    @property
    def prompt_stream_active(self) -> bool:
        return bool(self._client.prompt_stream_active)

    def start(self) -> None:
        self._client.start()

    def close(self) -> None:
        self._client.close()

    def abort(self) -> None:
        self._client.abort()

    def prompt_events(self, message: str) -> Iterator[AgentStreamEvent]:
        from pi_rpc_client import PiStreamEvent

        for event in self._client.prompt_events(message):
            if isinstance(event, PiStreamEvent):
                yield _pi_event_to_agent_event(event)
            elif isinstance(event, AgentStreamEvent):
                yield event
            else:
                yield AgentStreamEvent(kind="status", text=str(event))

    def get_state(self) -> dict[str, Any]:
        return dict(self._client.get_state())

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self._client.get_messages())

    def get_session_stats(self) -> dict[str, Any]:
        return dict(self._client.get_session_stats())

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        return dict(self._client.set_model(provider, model_id))

    def new_session(self) -> None:
        self._client.new_session()

    def steer(self, message: str) -> None:
        self._client.steer(message)

    def follow_up(self, message: str) -> None:
        self._client.follow_up(message)

    def stage_ui_chat_notice(self, label: str, message: str) -> None:
        self._client.stage_ui_chat_notice(label, message)

    def take_pending_ui_chat_notices(self) -> list[dict[str, Any]]:
        return []

    def drain_pending_ui_history(self) -> list[dict[str, Any]]:
        return list(self._client.drain_pending_ui_history())


def _pi_event_to_agent_event(event: Any) -> AgentStreamEvent:
    return AgentStreamEvent(
        kind=str(event.kind),
        text=str(event.text or ""),
        tool_name=event.tool_name,
        tool_call_id=event.tool_call_id,
        tool_args=event.tool_args,
        tool_output=event.tool_output,
        is_error=bool(event.is_error),
        meta=dict(event.meta or {}),
    )


def create_agent_runtime(session_hash: str | None = None) -> AgentRuntime:
    """Factory for the configured orchestration backend."""
    orchestrator = normalize_orchestrator()
    if orchestrator == "langgraph":
        from langgraph_runtime import LangGraphAgentRuntime

        return LangGraphAgentRuntime(session_hash=session_hash)
    if orchestrator == "agentcore":
        from agentcore_runtime import AgentCoreAgentRuntime

        return AgentCoreAgentRuntime(session_hash=session_hash)
    if orchestrator == "agentcore-harness":
        from agentcore_harness_runtime import AgentCoreHarnessRuntime

        return AgentCoreHarnessRuntime(session_hash=session_hash)
    from pi_rpc_client import default_client

    return PiAgentRuntime(default_client(session_hash))


def start_agent_prompt_event_worker(
    runtime: AgentRuntime,
    event_queue: queue.Queue[AgentStreamEvent | None],
    prompt: str,
) -> None:
    """Run ``runtime.prompt_events`` on a background thread, feeding *event_queue*."""

    def _worker() -> None:
        try:
            for event in runtime.prompt_events(prompt):
                event_queue.put(event)
        except Exception as exc:
            event_queue.put(
                AgentStreamEvent(kind="error", text=str(exc), is_error=True)
            )
        finally:
            event_queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()


def coerce_agent_runtime(client: Any) -> AgentRuntime | None:
    if client is None:
        return None
    if isinstance(client, AgentRuntime):
        return client
    if isinstance(client, PiAgentRuntime):
        return client
    # Legacy Gradio state may still hold a bare PiRpcClient.
    from pi_rpc_client import PiRpcClient

    if isinstance(client, PiRpcClient):
        return PiAgentRuntime(client)
    return None
