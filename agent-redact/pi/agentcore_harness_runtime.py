"""Bedrock AgentCore Harness client :class:`AgentRuntime` for the Gradio UI."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from typing import Any

from agent_runtime import AgentRuntime, AgentRuntimeError, AgentStreamEvent
from agentcore_boto import bedrock_agentcore_client, region_from_agentcore_arn


def agentcore_harness_arn() -> str:
    raw = (os.environ.get("AGENTCORE_HARNESS_ARN") or "").strip()
    if not raw:
        raise AgentRuntimeError(
            "AGENTCORE_HARNESS_ARN is not set. Create a Harness in the AgentCore console "
            "or use AGENT_ORCHESTRATOR=pi|langgraph|agentcore for other backends."
        )
    return raw


def agentcore_harness_endpoint() -> str:
    return (
        os.environ.get("AGENTCORE_HARNESS_ENDPOINT") or "DEFAULT"
    ).strip() or "DEFAULT"


def harness_runtime_session_id(session_hash: str | None) -> str:
    """Stable AgentCore Harness session id (must be at least 33 characters)."""
    base = (session_hash or "default").strip() or "default"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"harness-{digest}"


def parse_agentcore_harness_arn(arn: str) -> tuple[str, str]:
    """Return ``(region, harness_arn)``."""
    normalized = (arn or "").strip()
    return region_from_agentcore_arn(normalized, resource_label="harness"), normalized


def map_harness_stream_event(event: dict[str, Any]) -> Iterator[AgentStreamEvent]:
    """Map one ``InvokeHarness`` stream event to normalized Gradio events."""
    if "runtimeClientError" in event:
        err = event.get("runtimeClientError") or {}
        message = str(err.get("message") or err.get("errorMessage") or "Harness error")
        yield AgentStreamEvent(kind="error", text=message, is_error=True)
        return

    if "contentBlockDelta" in event:
        delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
        text = delta.get("text")
        if text:
            yield AgentStreamEvent(kind="text_delta", text=str(text))
        reasoning = (delta.get("reasoningContent") or {}).get("text")
        if reasoning:
            yield AgentStreamEvent(kind="thinking_delta", text=str(reasoning))
        tool_input = delta.get("toolUse") or {}
        if isinstance(tool_input, dict) and tool_input.get("input"):
            yield AgentStreamEvent(
                kind="status",
                text=f"Tool input: {json.dumps(tool_input.get('input'), default=str)[:500]}",
            )
        return

    if "contentBlockStart" in event:
        start = (event.get("contentBlockStart") or {}).get("start") or {}
        tool_use = start.get("toolUse") or {}
        if isinstance(tool_use, dict) and tool_use.get("name"):
            name = str(tool_use.get("name") or "tool")
            tool_id = str(tool_use.get("toolUseId") or "")
            args = (
                tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {}
            )
            yield AgentStreamEvent(
                kind="tool_start",
                tool_name=name,
                tool_call_id=tool_id or None,
                tool_args=args,
                text=name,
            )
        return

    if "contentBlockStop" in event:
        stop = event.get("contentBlockStop") or {}
        tool_result = stop.get("toolResult") or {}
        if isinstance(tool_result, dict) and tool_result:
            name = str(tool_result.get("toolName") or tool_result.get("name") or "tool")
            content = (
                tool_result.get("content") or tool_result.get("output") or tool_result
            )
            output = (
                content
                if isinstance(content, str)
                else json.dumps(content, default=str)
            )
            yield AgentStreamEvent(
                kind="tool_end",
                tool_name=name,
                tool_output=output,
                is_error="error" in output.lower(),
            )
        return

    if "messageStop" in event:
        stop = event.get("messageStop") or {}
        reason = str(stop.get("stopReason") or "")
        if reason:
            yield AgentStreamEvent(kind="status", text=f"Harness stopped: {reason}")
        return

    if "messageStart" in event:
        yield AgentStreamEvent(kind="status", text="Harness message started…")
        return


class AgentCoreHarnessRuntime(AgentRuntime):
    """Proxy that streams events from a remote Bedrock AgentCore Harness."""

    def __init__(self, *, session_hash: str | None = None) -> None:
        self._session_hash = session_hash
        self._running = False
        self._prompt_stream_depth = 0
        self._abort_requested = False
        self._pending_ui_history: list[dict[str, Any]] = []
        self._pending_prompt_prefix = ""

    @property
    def orchestrator(self) -> str:
        return "agentcore-harness"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def prompt_stream_active(self) -> bool:
        return self._prompt_stream_depth > 0

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    def start(self) -> None:
        agentcore_harness_arn()
        self._running = True

    def close(self) -> None:
        self._running = False

    def abort(self) -> None:
        self._abort_requested = True
        try:
            region, _arn = parse_agentcore_harness_arn(agentcore_harness_arn())
            client = bedrock_agentcore_client(region)
            stop = getattr(client, "stop_runtime_session", None)
            if callable(stop):
                stop(runtimeSessionId=harness_runtime_session_id(self._session_hash))
        except Exception:
            pass

    def new_session(self) -> None:
        self._abort_requested = False

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        os.environ["PI_DEFAULT_PROVIDER"] = provider
        os.environ["PI_DEFAULT_MODEL"] = model_id
        return {"provider": provider, "model": model_id}

    def get_state(self) -> dict[str, Any]:
        return {
            "isStreaming": self.prompt_stream_active,
            "provider": "agentcore-harness",
            "model": {
                "provider": "agentcore-harness",
                "id": agentcore_harness_arn(),
                "endpoint": agentcore_harness_endpoint(),
            },
        }

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

    def stage_prompt_prefix(self, prefix: str) -> None:
        text = (prefix or "").strip()
        if text:
            self._pending_prompt_prefix = f"{text.rstrip()}\n\n"

    def prompt_events(self, message: str) -> Iterator[AgentStreamEvent]:
        self._prompt_stream_depth += 1
        self._abort_requested = False
        try:
            if not self._running:
                self.start()
            harness_arn = agentcore_harness_arn()
            region, _parsed = parse_agentcore_harness_arn(harness_arn)
            client = bedrock_agentcore_client(region)
            invoke = getattr(client, "invoke_harness", None)
            if not callable(invoke):
                raise AgentRuntimeError(
                    "Your boto3 bedrock-agentcore client does not support invoke_harness. "
                    "Upgrade boto3/botocore in the pi-agent environment."
                )

            prompt = f"{self._pending_prompt_prefix}{message}"
            self._pending_prompt_prefix = ""
            session_id = harness_runtime_session_id(self._session_hash)
            request: dict[str, Any] = {
                "harnessArn": harness_arn,
                "runtimeSessionId": session_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            }
            endpoint = agentcore_harness_endpoint()
            if endpoint and endpoint.upper() != "DEFAULT":
                request["endpointName"] = endpoint

            yield AgentStreamEvent(kind="status", text="AgentCore Harness started…")
            try:
                response = invoke(**request)
            except Exception as exc:
                from botocore.exceptions import ClientError

                if isinstance(exc, ClientError):
                    code = exc.response.get("Error", {}).get("Code", "")
                    msg = exc.response.get("Error", {}).get("Message", str(exc))
                    hint = (
                        " Ensure your IAM identity has bedrock-agentcore:InvokeHarness on "
                        f"{harness_arn}."
                    )
                    raise AgentRuntimeError(f"{code}: {msg}.{hint}") from exc
                raise AgentRuntimeError(str(exc)) from exc

            stream = response.get("stream") or []
            assistant_text: list[str] = []
            for event in stream:
                if self._abort_requested:
                    yield AgentStreamEvent(kind="done", text="Agent aborted.")
                    return
                if not isinstance(event, dict):
                    continue
                for mapped in map_harness_stream_event(event):
                    if mapped.kind == "text_delta" and mapped.text:
                        assistant_text.append(mapped.text)
                    yield mapped

            if assistant_text:
                yield AgentStreamEvent(
                    kind="text_snapshot", text="".join(assistant_text)
                )
            yield AgentStreamEvent(kind="done", text="Agent finished.")
        finally:
            self._prompt_stream_depth = max(0, self._prompt_stream_depth - 1)
