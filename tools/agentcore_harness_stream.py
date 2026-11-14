"""
Dependency-free helpers for Bedrock AgentCore Harness ``InvokeHarness`` streams.

Shared by the main app driver (:mod:`tools.agentcore_harness`) and the agent-redact
Gradio client (``agent-redact/agentcore/agentcore_harness_runtime.py``). Keep this module
free of app/Gradio imports so both images can load it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TOOLS_SPEC_PATH = Path(__file__).with_name("agentcore_harness_tools.json")

# Built-ins on the harness VM (not doc_redaction inline tools).
HARNESS_BUILTIN_ALLOWED_TOOLS = (
    "shell",
    "read",
    "write",
    "edit",
    "todo_write",
    "subagent",
    "retrieve_context",
    "skills",
)

# Harness ``AllowedTools`` is builtins-only (see CDK). Invoke-time ``inline_function`` tools
# are listed in the ``tools`` array but are not exposed to the model unless ``allowedTools``
# is overridden on each ``InvokeHarness``. Wildcard is required today: listing builtin names
# plus inline tool names does not enable client-side inline handoff (AgentCore 2026-03).
HARNESS_INVOKE_ALLOWED_TOOLS = ("*",)


def harness_invoke_allowed_tools() -> list[str]:
    """``allowedTools`` for each ``InvokeHarness`` when passing doc_redaction inline tools."""
    return list(HARNESS_INVOKE_ALLOWED_TOOLS)


# Stop reasons after which the harness is not waiting for anything from the caller.
TERMINAL_STOP_REASONS = frozenset(
    {
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "content_filtered",
        "malformed_model_output",
        "malformed_tool_use",
        "interrupted",
        "model_context_window_exceeded",
        "max_iterations_exceeded",
        "max_output_tokens_exceeded",
        "timeout_exceeded",
        "hook_stopped",
    }
)


def harness_runtime_session_id(seed: str | None) -> str:
    """Stable AgentCore Harness session id (the API requires at least 33 characters)."""
    base = (seed or "default").strip() or "default"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"harness-{digest}"


def region_from_harness_arn(arn: str) -> str:
    """Return the AWS region from ``arn:aws:bedrock-agentcore:<region>:<acct>:harness/<id>``."""
    normalized = (arn or "").strip()
    parts = normalized.split(":")
    if (
        len(parts) < 6
        or parts[0] != "arn"
        or parts[2] != "bedrock-agentcore"
        or ":harness/" not in normalized
    ):
        raise ValueError(f"Invalid AgentCore harness ARN: {arn!r}")
    region = parts[3].strip()
    if not region:
        raise ValueError(f"Could not parse region from harness ARN: {arn!r}")
    return region


def load_inline_tool_specs(path: Path | None = None) -> list[dict[str, Any]]:
    """Load inline function tool definitions in ``InvokeHarness`` (camelCase) form."""
    spec_path = path or TOOLS_SPEC_PATH
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise ValueError(f"{spec_path} must contain a JSON list of tools")
    return specs


def inline_tool_names(specs: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(s.get("name"))
        for s in specs
        if s.get("type") == "inline_function" and s.get("name")
    }


@dataclass
class HarnessTokenUsage:
    """Bedrock usage aggregated from ``InvokeHarness`` stream ``metadata`` events."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    metadata_events: int = 0

    def absorb_stream_event(self, event: dict[str, Any]) -> None:
        meta = event.get("metadata")
        if not isinstance(meta, dict):
            return
        usage = meta.get("usage")
        if not isinstance(usage, dict):
            return
        self.metadata_events += 1
        self.input_tokens += int(usage.get("inputTokens") or 0)
        self.output_tokens += int(usage.get("outputTokens") or 0)
        self.total_tokens += int(usage.get("totalTokens") or 0)
        self.cache_read_input_tokens += int(usage.get("cacheReadInputTokens") or 0)
        self.cache_write_input_tokens += int(usage.get("cacheWriteInputTokens") or 0)


@dataclass
class HarnessStreamEvent:
    """Normalised view of one harness stream event, for activity logs and UIs."""

    kind: str  # text_delta | thinking_delta | tool_start | tool_result | status | error
    text: str = ""
    tool_name: str | None = None
    tool_use_id: str | None = None
    is_error: bool = False


def map_harness_stream_event(event: dict[str, Any]) -> Iterator[HarnessStreamEvent]:
    """Map one raw ``InvokeHarness`` stream event to zero or more normalised events."""
    for err_key in (
        "runtimeClientError",
        "internalServerException",
        "validationException",
    ):
        if err_key in event:
            err = event.get(err_key) or {}
            message = str(err.get("message") or err.get("errorMessage") or err_key)
            yield HarnessStreamEvent(kind="error", text=message, is_error=True)
            return

    if "contentBlockDelta" in event:
        delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
        text = delta.get("text")
        if text:
            yield HarnessStreamEvent(kind="text_delta", text=str(text))
        reasoning = delta.get("reasoningContent") or {}
        if isinstance(reasoning, dict) and reasoning.get("text"):
            yield HarnessStreamEvent(kind="thinking_delta", text=str(reasoning["text"]))
        return

    if "contentBlockStart" in event:
        start = (event.get("contentBlockStart") or {}).get("start") or {}
        tool_use = start.get("toolUse") or {}
        if isinstance(tool_use, dict) and tool_use.get("name"):
            yield HarnessStreamEvent(
                kind="tool_start",
                text=str(tool_use.get("name")),
                tool_name=str(tool_use.get("name")),
                tool_use_id=str(tool_use.get("toolUseId") or "") or None,
            )
            return
        tool_result = start.get("toolResult") or {}
        if isinstance(tool_result, dict) and tool_result.get("toolUseId"):
            status = str(tool_result.get("status") or "")
            yield HarnessStreamEvent(
                kind="tool_result",
                text=status,
                tool_use_id=str(tool_result.get("toolUseId")),
                is_error=status.lower() == "error",
            )
        return

    if "messageStop" in event:
        reason = str((event.get("messageStop") or {}).get("stopReason") or "")
        if reason:
            yield HarnessStreamEvent(kind="status", text=f"stop: {reason}")
        return


@dataclass
class PendingToolUse:
    tool_use_id: str
    name: str
    input_json: str = ""
    initial_input: Any = None

    def arguments(self) -> dict[str, Any]:
        if self.input_json.strip():
            try:
                parsed = json.loads(self.input_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool input is not valid JSON: {exc}") from exc
        else:
            parsed = self.initial_input
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("Tool input must be a JSON object")
        return parsed


@dataclass
class HarnessTurn:
    """Accumulated state of one ``InvokeHarness`` response stream."""

    text: str = ""
    stop_reason: str = ""
    errors: list[str] = field(default_factory=list)
    tool_uses: list[PendingToolUse] = field(default_factory=list)
    resolved_tool_use_ids: set[str] = field(default_factory=set)
    _by_block_index: dict[int, PendingToolUse] = field(default_factory=dict)

    def pending_inline_calls(self, inline_names: set[str]) -> list[PendingToolUse]:
        """Unresolved inline tool uses seen in the stream."""
        return [
            t
            for t in self.tool_uses
            if t.name in inline_names
            and t.tool_use_id not in self.resolved_tool_use_ids
        ]

    def handoff_inline_calls(self, inline_names: set[str]) -> list[PendingToolUse]:
        """Inline calls the harness handed off (client may return ``toolResult``)."""
        if self.stop_reason != "tool_use":
            return []
        return self.pending_inline_calls(inline_names)


class HarnessTurnCollector:
    """Consume raw stream events and build a :class:`HarnessTurn`."""

    def __init__(self) -> None:
        self.turn = HarnessTurn()
        self._text_parts: list[str] = []

    def feed(
        self,
        event: dict[str, Any],
        *,
        inline_names: set[str] | None = None,
    ) -> list[HarnessStreamEvent]:
        turn = self.turn
        if "contentBlockStart" in event:
            block = event.get("contentBlockStart") or {}
            index = int(block.get("contentBlockIndex") or 0)
            start = block.get("start") or {}
            tool_use = start.get("toolUse") or {}
            if isinstance(tool_use, dict) and tool_use.get("toolUseId"):
                pending = PendingToolUse(
                    tool_use_id=str(tool_use["toolUseId"]),
                    name=str(tool_use.get("name") or ""),
                    initial_input=tool_use.get("input"),
                )
                turn.tool_uses.append(pending)
                turn._by_block_index[index] = pending
            tool_result = start.get("toolResult") or {}
            if isinstance(tool_result, dict) and tool_result.get("toolUseId"):
                tid = str(tool_result["toolUseId"])
                matched = next(
                    (t for t in turn.tool_uses if t.tool_use_id == tid), None
                )
                # Inline tools run in the app; ignore spurious harness-side toolResult
                # (e.g. "Unknown tool") so the client can still dispatch handlers.
                if matched is None or (inline_names and matched.name in inline_names):
                    pass
                else:
                    turn.resolved_tool_use_ids.add(tid)
        elif "contentBlockDelta" in event:
            block = event.get("contentBlockDelta") or {}
            index = int(block.get("contentBlockIndex") or 0)
            delta = block.get("delta") or {}
            tool_delta = delta.get("toolUse") or {}
            if isinstance(tool_delta, dict) and "input" in tool_delta:
                pending = turn._by_block_index.get(index)
                if pending is not None:
                    raw = tool_delta.get("input")
                    pending.input_json += (
                        raw if isinstance(raw, str) else json.dumps(raw)
                    )
            if delta.get("text"):
                self._text_parts.append(str(delta["text"]))
        elif "contentBlockStop" in event:
            block = event.get("contentBlockStop") or {}
            stop = block.get("stop") or {}
            tool_result = stop.get("toolResult") or {}
            if isinstance(tool_result, dict) and tool_result.get("toolUseId"):
                tid = str(tool_result["toolUseId"])
                matched = next(
                    (t for t in turn.tool_uses if t.tool_use_id == tid), None
                )
                if matched is None or (inline_names and matched.name in inline_names):
                    pass
                else:
                    turn.resolved_tool_use_ids.add(tid)
        elif "messageStop" in event:
            turn.stop_reason = str(
                (event.get("messageStop") or {}).get("stopReason") or ""
            )
        mapped = list(map_harness_stream_event(event))
        for m in mapped:
            if m.kind == "error":
                turn.errors.append(m.text)
        turn.text = "".join(self._text_parts)
        return mapped


def tool_result_message(
    results: list[tuple[str, str, bool]],
) -> dict[str, Any]:
    """Build the ``user`` message returning inline results ``(toolUseId, text, ok)``."""
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": text}],
                    "status": "success" if ok else "error",
                }
            }
            for tool_use_id, text, ok in results
        ],
    }
