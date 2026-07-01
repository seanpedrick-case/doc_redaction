"""AgentCore HTTP/SSE client :class:`AgentRuntime` for the Gradio UI."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from agent_runtime import AgentRuntime, AgentRuntimeError, AgentStreamEvent
from agentcore_boto import bedrock_agentcore_client, region_from_agentcore_arn


def agentcore_runtime_url() -> str:
    raw = (os.environ.get("AGENTCORE_RUNTIME_URL") or "").strip().rstrip("/")
    if raw.endswith("/invocations"):
        raw = raw[: -len("/invocations")].rstrip("/")
    if not raw:
        raise AgentRuntimeError(
            "AGENTCORE_RUNTIME_URL is not set. Deploy AgentCore runtime or use "
            "AGENT_ORCHESTRATOR=pi|langgraph for local orchestration."
        )
    return raw


def parse_agentcore_runtime_url(url: str) -> tuple[str, str]:
    """Return ``(region, agent_runtime_arn)`` from an AgentCore runtime base URL."""
    normalized = (url or "").strip().rstrip("/")
    if normalized.endswith("/invocations"):
        normalized = normalized[: -len("/invocations")].rstrip("/")
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").strip()
    if not host.startswith("bedrock-agentcore."):
        raise AgentRuntimeError(
            f"AGENTCORE_RUNTIME_URL must be a bedrock-agentcore HTTPS URL, got: {url!r}"
        )
    region = host.removeprefix("bedrock-agentcore.").split(".", 1)[0].strip()
    if not region:
        raise AgentRuntimeError(
            f"Could not parse AWS region from AgentCore URL: {url!r}"
        )

    prefix = "/runtimes/"
    path = parsed.path or ""
    if not path.startswith(prefix):
        raise AgentRuntimeError(
            f"AGENTCORE_RUNTIME_URL must include /runtimes/<arn>, got path: {path!r}"
        )
    arn = unquote(path[len(prefix) :].strip("/"))
    if not arn.startswith("arn:"):
        raise AgentRuntimeError(
            f"Could not parse runtime ARN from AgentCore URL: {url!r}"
        )
    return region_from_agentcore_arn(arn, resource_label="runtime"), arn


def _agentcore_api_key() -> str:
    return (os.environ.get("AGENTCORE_API_KEY") or "").strip()


def _bedrock_agentcore_client(region: str):
    return bedrock_agentcore_client(region)


class AgentCoreAgentRuntime(AgentRuntime):
    """Proxy that streams events from a remote Bedrock AgentCore runtime."""

    def __init__(self, *, session_hash: str | None = None) -> None:
        self._session_hash = session_hash
        self._running = False
        self._prompt_stream_depth = 0
        self._abort_requested = False
        self._pending_ui_notices: list[dict[str, Any]] = []
        self._pending_ui_history: list[dict[str, Any]] = []
        self._pending_workspace_files: list[dict[str, str]] = []
        self._sync_workspace_files = True

    @property
    def orchestrator(self) -> str:
        return "agentcore"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def prompt_stream_active(self) -> bool:
        return self._prompt_stream_depth > 0

    def start(self) -> None:
        agentcore_runtime_url()
        self._running = True

    def close(self) -> None:
        self._running = False

    def abort(self) -> None:
        self._abort_requested = True

    def new_session(self) -> None:
        self._abort_requested = False

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        os.environ["PI_DEFAULT_PROVIDER"] = provider
        os.environ["PI_DEFAULT_MODEL"] = model_id
        return {"provider": provider, "model": model_id}

    def get_state(self) -> dict[str, Any]:
        return {
            "isStreaming": self.prompt_stream_active,
            "provider": "agentcore",
            "model": {"provider": "agentcore", "id": agentcore_runtime_url()},
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

    def stage_workspace_files(self, files: list[dict[str, str]]) -> None:
        """Queue files to upload into the remote AgentCore session workspace on next invoke."""
        for item in files:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("relative_path") or item.get("name") or "").strip()
            encoded = str(item.get("content_base64") or "").strip()
            if relative and encoded:
                self._pending_workspace_files.append(
                    {"relative_path": relative, "content_base64": encoded}
                )

    def set_sync_workspace_files(self, enabled: bool) -> None:
        self._sync_workspace_files = bool(enabled)

    def _write_local_workspace_file(
        self, relative_path: str, content_base64: str
    ) -> None:
        if not self._session_hash:
            return
        from session_workspace import session_workspace_dir

        root = session_workspace_dir(self._session_hash).resolve()
        dest = (root / relative_path).resolve()
        try:
            dest.relative_to(root)
        except ValueError:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(content_base64, validate=True))

    def prompt_events(self, message: str) -> Iterator[AgentStreamEvent]:
        self._prompt_stream_depth += 1
        self._abort_requested = False
        try:
            if not self._running:
                self.start()
            payload: dict[str, Any] = {
                "prompt": message,
                "session_hash": self._session_hash or "",
            }
            if self._pending_workspace_files:
                payload["workspace_files"] = self._pending_workspace_files[:]
                self._pending_workspace_files.clear()
            if self._sync_workspace_files:
                payload["sync_workspace_files"] = True
            yield AgentStreamEvent(kind="status", text="AgentCore runtime started…")
            if _agentcore_api_key():
                yield from self._prompt_events_httpx(payload)
            else:
                yield from self._prompt_events_boto3(payload)
            yield AgentStreamEvent(kind="done", text="Agent finished.")
        except httpx.HTTPError as exc:
            raise AgentRuntimeError(str(exc)) from exc
        finally:
            self._prompt_stream_depth = max(0, self._prompt_stream_depth - 1)

    def _prompt_events_httpx(
        self, payload: dict[str, Any]
    ) -> Iterator[AgentStreamEvent]:
        url = f"{agentcore_runtime_url()}/invocations"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {_agentcore_api_key()}",
        }
        timeout = httpx.Timeout(connect=30.0, read=1800.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                yield from self._iter_sse_response(response.iter_lines())

    def _prompt_events_boto3(
        self, payload: dict[str, Any]
    ) -> Iterator[AgentStreamEvent]:
        from botocore.exceptions import ClientError

        region, runtime_arn = parse_agentcore_runtime_url(agentcore_runtime_url())
        client = _bedrock_agentcore_client(region)
        body = json.dumps(payload).encode("utf-8")
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                payload=body,
                contentType="application/json",
                accept="text/event-stream",
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            message = exc.response.get("Error", {}).get("Message", str(exc))
            hint = (
                " Ensure your IAM identity has bedrock-agentcore:InvokeAgentRuntime on "
                f"{runtime_arn}."
            )
            if "initialization" in message.lower() or code == "RuntimeClientError":
                hint += (
                    " The runtime container failed to start (import error or slow init). "
                    "Check CloudWatch log group "
                    f"/aws/bedrock-agentcore/runtimes/{runtime_arn.rsplit('/', 1)[-1]}/"
                    " then re-run package_runtime.py and agentcore deploy."
                )
            raise AgentRuntimeError(f"{code}: {message}.{hint}") from exc

        status_code = int(response.get("statusCode") or 200)
        if status_code >= 400:
            raise AgentRuntimeError(
                f"AgentCore invoke failed with HTTP {status_code} for runtime {runtime_arn}."
            )

        stream = response.get("response")
        if stream is None:
            return
        if hasattr(stream, "iter_lines"):
            yield from self._iter_sse_response(
                (
                    line.decode("utf-8", errors="replace")
                    if isinstance(line, (bytes, bytearray))
                    else str(line)
                )
                for line in stream.iter_lines()
            )
            return
        raw = stream.read() if hasattr(stream, "read") else stream
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = bytes(raw or b"")
        content_type = str(response.get("contentType") or "").lower()
        if "event-stream" in content_type or raw_bytes.strip().startswith(b"data:"):
            yield from self._iter_sse_lines(
                line.decode("utf-8", errors="replace")
                for line in raw_bytes.splitlines()
            )
        else:
            yield from self._iter_json_response(raw_bytes)

    def _iter_sse_response(self, lines: Iterator[str]) -> Iterator[AgentStreamEvent]:
        for line in lines:
            if self._abort_requested:
                yield AgentStreamEvent(kind="done", text="Agent aborted.")
                return
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                yield AgentStreamEvent(kind="text_delta", text=data)
                continue
            yield from self._map_agentcore_event(event)

    def _iter_sse_lines(self, lines: Iterator[str]) -> Iterator[AgentStreamEvent]:
        yield from self._iter_sse_response(lines)

    def _iter_json_response(self, raw: bytes) -> Iterator[AgentStreamEvent]:
        if not raw.strip():
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            yield AgentStreamEvent(
                kind="text_snapshot", text=raw.decode("utf-8", errors="replace")
            )
            return
        if isinstance(payload, dict):
            if payload.get("type") == "error":
                yield AgentStreamEvent(
                    kind="error",
                    text=str(payload.get("message") or "AgentCore error"),
                    is_error=True,
                )
                return
            if "result" in payload:
                yield AgentStreamEvent(
                    kind="text_snapshot", text=str(payload["result"])
                )
                return
            yield AgentStreamEvent(
                kind="text_snapshot",
                text=json.dumps(payload, default=str),
            )
        else:
            yield AgentStreamEvent(kind="text_snapshot", text=str(payload))

    def _map_agentcore_event(self, event: dict[str, Any]) -> Iterator[AgentStreamEvent]:
        event_type = str(event.get("type") or "")
        if event_type == "agent_start":
            yield AgentStreamEvent(kind="status", text="Agent started…")
        elif event_type == "agent_end":
            yield AgentStreamEvent(
                kind="status", text=str(event.get("message") or "Agent finished.")
            )
        elif event_type == "status":
            yield AgentStreamEvent(kind="status", text=str(event.get("message") or ""))
        elif event_type == "error":
            yield AgentStreamEvent(
                kind="error",
                text=str(event.get("message") or "AgentCore error"),
                is_error=True,
            )
        elif event_type == "workspace_file":
            relative = str(event.get("relative_path") or "").strip()
            encoded = str(event.get("content_base64") or "").strip()
            if relative and encoded:
                try:
                    self._write_local_workspace_file(relative, encoded)
                    yield AgentStreamEvent(
                        kind="status",
                        text=f"Downloaded `{relative}` from AgentCore workspace.",
                    )
                    if relative.lower().endswith("_redacted.pdf"):
                        yield AgentStreamEvent(kind="workspace_sync")
                except (OSError, ValueError) as exc:
                    yield AgentStreamEvent(
                        kind="status",
                        text=f"Could not save `{relative}` locally: {exc}",
                    )
        elif event_type == "message_update":
            role = str(event.get("role") or "")
            if role == "tool":
                tool_name = str(event.get("tool_name") or "tool")
                content = event.get("content")
                output = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, default=str)
                )
                is_error = "error" in output.lower() or "not found" in output.lower()
                yield AgentStreamEvent(
                    kind="tool_end",
                    tool_name=tool_name,
                    tool_output=output,
                    is_error=is_error,
                )
            else:
                content = event.get("content")
                text = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, default=str)
                )
                tool_calls = event.get("tool_calls") or []
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
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
                if text.strip():
                    yield AgentStreamEvent(kind="text_snapshot", text=text)
        else:
            yield AgentStreamEvent(
                kind="status", text=json.dumps(event, default=str)[:500]
            )
