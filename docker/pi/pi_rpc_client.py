"""Python client for Pi RPC mode (JSONL over stdin/stdout)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


class PiRpcError(RuntimeError):
    pass


@dataclass
class PiStreamEvent:
    """Structured event from Pi RPC for UI layers."""

    kind: str
    text: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def extract_tool_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    content = payload.get("content")
    if content is None and isinstance(payload.get("partialResult"), dict):
        content = payload["partialResult"].get("content")
    if content is None and isinstance(payload.get("result"), dict):
        content = payload["result"].get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(parts).strip()


def extract_assistant_display(message: dict[str, Any] | None) -> tuple[str, str]:
    """Extract visible text and thinking from a partial assistant message."""
    if not message or message.get("role") != "assistant":
        return "", ""
    content = message.get("content")
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""

    texts: list[str] = []
    thinkings: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                texts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in (None, "text", "output_text"):
            text = block.get("text") or block.get("content") or ""
            if text:
                texts.append(str(text))
        elif block_type in ("thinking", "reasoning", "thought"):
            thought = (
                block.get("thinking")
                or block.get("text")
                or block.get("reasoning")
                or block.get("content")
                or ""
            )
            if thought:
                thinkings.append(str(thought))
    return "".join(texts), "".join(thinkings)


def assistant_chat_text(visible: str, thinking: str) -> str:
    """Text to show in the main chat — visible answer, or thinking when Gemini sends only that."""
    if visible.strip():
        return visible
    return thinking


def partial_message_from_update(event: dict[str, Any]) -> dict[str, Any] | None:
    delta = event.get("assistantMessageEvent") or {}
    partial = delta.get("partial")
    if isinstance(partial, dict):
        return partial
    message = event.get("message")
    if isinstance(message, dict):
        return message
    return None


def format_tool_args(tool_name: str | None, args: dict[str, Any] | None) -> str:
    if not args:
        return ""
    name = (tool_name or "").lower()
    if name == "bash" and args.get("command"):
        cmd = str(args["command"]).replace("\n", " ↵ ")
        return f"`{cmd[:240]}{'…' if len(cmd) > 240 else ''}`"
    if name in {"read", "write", "edit"} and args.get("path"):
        return f"`{args['path']}`"
    compact = json.dumps(args, ensure_ascii=False)
    if len(compact) > 280:
        compact = compact[:277] + "…"
    return compact


class PiRpcClient:
    """Drive a long-lived ``pi --mode rpc`` subprocess."""

    def __init__(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pi_args: list[str] | None = None,
    ) -> None:
        self._cwd = cwd
        self._env = env
        self._pi_args = pi_args or []
        self._proc: subprocess.Popen[str] | None = None
        self._io_lock = threading.Lock()
        self._abort_requested = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        command = ["pi", "--mode", "rpc", *self._pi_args]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self._cwd,
            env=self._env,
        )

    def close(self) -> None:
        if not self._proc:
            return
        if self.running:
            try:
                self.abort()
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _ensure_running(self) -> subprocess.Popen[str]:
        if not self.running:
            self.start()
        assert self._proc is not None
        return self._proc

    def _read_line(self) -> dict[str, Any]:
        proc = self._ensure_running()
        assert proc.stdout is not None
        with self._io_lock:
            line = proc.stdout.readline()
        if not line:
            code = proc.poll()
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read() or ""
            raise PiRpcError(
                f"Pi RPC process exited (code={code})."
                + (f" stderr: {err[:500]}" if err else "")
            )
        line = line.rstrip("\r\n")
        if not line:
            return self._read_line()
        return json.loads(line)

    def _write_command(self, command: dict[str, Any]) -> None:
        proc = self._ensure_running()
        assert proc.stdin is not None
        with self._io_lock:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()

    def _send_command(
        self,
        command: dict[str, Any],
        *,
        wait_response: bool = True,
    ) -> dict[str, Any] | None:
        req_id = command.setdefault("id", str(uuid.uuid4()))
        self._write_command(command)
        if not wait_response:
            return None
        while True:
            event = self._read_line()
            if event.get("type") == "response" and event.get("id") == req_id:
                if not event.get("success", False):
                    error = (
                        event.get("error") or event.get("message") or "command failed"
                    )
                    raise PiRpcError(str(error))
                return event

    def abort(self) -> None:
        """Request abort without reading stdout (the active stream consumer drains events)."""
        if not self.running:
            return
        self._abort_requested = True
        try:
            self._send_command({"type": "abort"}, wait_response=False)
        except OSError:
            pass

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    def clear_abort(self) -> None:
        self._abort_requested = False

    def new_session(self) -> None:
        self._send_command({"type": "new_session"})

    def get_state(self) -> dict[str, Any]:
        response = self._send_command({"type": "get_state"})
        data = response.get("data") if response else {}
        return data if isinstance(data, dict) else {}

    def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        response = self._send_command(
            {
                "type": "set_model",
                "provider": provider,
                "modelId": model_id,
            }
        )
        data = response.get("data") if response else {}
        return data if isinstance(data, dict) else {}

    def get_available_models(self) -> list[dict[str, Any]]:
        response = self._send_command({"type": "get_available_models"})
        data = response.get("data") if response else {}
        models = data.get("models") if isinstance(data, dict) else []
        return models if isinstance(models, list) else []

    def restart(self) -> None:
        self.close()
        self.start()

    def prompt_events(self, message: str) -> Iterator[PiStreamEvent]:
        """Send a user message and yield structured events until ``agent_end``."""
        self.clear_abort()
        req_id = str(uuid.uuid4())
        self._send_command(
            {"id": req_id, "type": "prompt", "message": message},
            wait_response=False,
        )

        while True:
            event = self._read_line()
            if event.get("type") == "response" and event.get("id") == req_id:
                if not event.get("success", False):
                    error = (
                        event.get("error") or event.get("message") or "prompt rejected"
                    )
                    yield PiStreamEvent(kind="error", text=str(error), is_error=True)
                    return
                break

        yield from self._iter_agent_events()

    def _iter_agent_events(self) -> Iterator[PiStreamEvent]:
        while True:
            event = self._read_line()
            event_type = event.get("type")

            if event_type == "agent_start":
                yield PiStreamEvent(kind="status", text="Agent started…")

            elif event_type == "turn_start":
                yield PiStreamEvent(kind="status", text="Turn started.")

            elif event_type == "turn_end":
                yield PiStreamEvent(kind="status", text="Turn finished.")

            elif event_type == "message_update":
                yield from self._parse_message_update(event)

            elif event_type == "tool_execution_start":
                tool_name = event.get("toolName")
                tool_args = (
                    event.get("args") if isinstance(event.get("args"), dict) else {}
                )
                yield PiStreamEvent(
                    kind="tool_start",
                    tool_name=str(tool_name) if tool_name else "tool",
                    tool_call_id=event.get("toolCallId"),
                    tool_args=tool_args,
                    text=format_tool_args(
                        str(tool_name) if tool_name else None,
                        tool_args,
                    ),
                )

            elif event_type == "tool_execution_update":
                output = extract_tool_text(event)
                yield PiStreamEvent(
                    kind="tool_update",
                    tool_name=event.get("toolName"),
                    tool_call_id=event.get("toolCallId"),
                    tool_output=output,
                )

            elif event_type == "tool_execution_end":
                result = (
                    event.get("result") if isinstance(event.get("result"), dict) else {}
                )
                output = extract_tool_text(result)
                yield PiStreamEvent(
                    kind="tool_end",
                    tool_name=event.get("toolName"),
                    tool_call_id=event.get("toolCallId"),
                    tool_output=output,
                    is_error=bool(event.get("isError")),
                )

            elif event_type == "queue_update":
                steering = event.get("steering") or []
                follow_up = event.get("followUp") or []
                if steering or follow_up:
                    yield PiStreamEvent(
                        kind="status",
                        text="Queue updated.",
                        meta={"steering": steering, "follow_up": follow_up},
                    )

            elif event_type == "compaction_start":
                reason = event.get("reason") or "unknown"
                yield PiStreamEvent(
                    kind="status",
                    text=f"Compaction started ({reason})…",
                    meta={"reason": reason},
                )

            elif event_type == "compaction_end":
                if event.get("aborted"):
                    text = "Compaction aborted."
                elif event.get("errorMessage"):
                    text = f"Compaction failed: {event['errorMessage']}"
                    yield PiStreamEvent(kind="error", text=text, is_error=True)
                    continue
                elif event.get("willRetry"):
                    text = "Compaction complete — retrying prompt…"
                else:
                    tokens = (event.get("result") or {}).get("tokensBefore")
                    text = (
                        f"Compaction complete ({tokens:,} tokens before)."
                        if isinstance(tokens, int)
                        else "Compaction complete."
                    )
                yield PiStreamEvent(kind="status", text=text, meta=event)

            elif event_type == "auto_retry_start":
                attempt = event.get("attempt")
                max_attempts = event.get("maxAttempts")
                delay_ms = event.get("delayMs")
                msg = event.get("errorMessage") or "transient error"
                yield PiStreamEvent(
                    kind="status",
                    text=(
                        f"Auto-retry {attempt}/{max_attempts} in {delay_ms}ms "
                        f"({str(msg)[:120]})"
                    ),
                    meta=event,
                )

            elif event_type == "auto_retry_end":
                if event.get("success"):
                    yield PiStreamEvent(
                        kind="status",
                        text=f"Auto-retry succeeded on attempt {event.get('attempt')}.",
                    )
                else:
                    yield PiStreamEvent(
                        kind="error",
                        text=f"Auto-retry failed: {event.get('finalError', 'unknown error')}",
                        is_error=True,
                    )

            elif event_type == "extension_error":
                yield PiStreamEvent(
                    kind="error",
                    text=str(event.get("error") or "extension error"),
                    is_error=True,
                )

            elif event_type == "agent_end":
                aborted = self._abort_requested
                self.clear_abort()
                yield PiStreamEvent(
                    kind="done",
                    text="Agent aborted." if aborted else "Agent finished.",
                )
                return

    def _parse_message_update(self, event: dict[str, Any]) -> Iterator[PiStreamEvent]:
        delta = event.get("assistantMessageEvent") or {}
        delta_type = delta.get("type")
        partial = partial_message_from_update(event)
        if partial is not None:
            visible, thinking = extract_assistant_display(partial)
            chat_text = assistant_chat_text(visible, thinking)
            if chat_text:
                yield PiStreamEvent(kind="text_snapshot", text=chat_text)
            if thinking.strip() and visible.strip():
                yield PiStreamEvent(kind="thinking_snapshot", text=thinking)

        if delta_type == "text_delta":
            chunk = delta.get("delta") or ""
            if chunk and partial is None:
                yield PiStreamEvent(kind="text_delta", text=chunk)

        elif delta_type == "thinking_delta":
            chunk = delta.get("delta") or ""
            if chunk:
                if partial is None:
                    # No snapshot yet — stream thinking into chat (Gemini reasoning path).
                    yield PiStreamEvent(kind="text_delta", text=chunk)
                yield PiStreamEvent(kind="thinking_delta", text=chunk)

        elif delta_type == "toolcall_start":
            tool_call = delta.get("toolCall") or {}
            tool_name = tool_call.get("name") or delta.get("toolName") or "tool"
            tool_args = tool_call.get("arguments")
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"raw": tool_args}
            if not isinstance(tool_args, dict):
                tool_args = {}
            yield PiStreamEvent(
                kind="status",
                text=f"Calling `{tool_name}`…",
                tool_name=str(tool_name),
                tool_args=tool_args,
            )

        elif delta_type == "error":
            yield PiStreamEvent(
                kind="error",
                text=str(
                    delta.get("message") or delta.get("error") or "generation error"
                ),
                is_error=True,
            )

    def prompt_stream(
        self, message: str, *, show_tool_status: bool = True
    ) -> Iterator[str]:
        """Backward-compatible text stream (assistant visible text + optional tool status)."""
        for event in self.prompt_events(message):
            if event.kind == "text_delta":
                yield event.text
            elif show_tool_status and event.kind == "tool_start":
                yield f"\n\n_[Running {event.tool_name}…]_\n"
            elif event.kind == "error":
                yield f"\n\n**Error:** {event.text}\n"


def default_client() -> PiRpcClient:
    repo_root = os.environ.get("PI_WORKDIR", "/workspace/doc_redaction")
    env = os.environ.copy()
    env.setdefault("HOME", os.path.expanduser("~"))
    if not env.get("GEMINI_API_KEY") and env.get("GOOGLE_API_KEY"):
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    if not env.get("HF_TOKEN") and env.get("DOC_REDACTION_HF_TOKEN"):
        env["HF_TOKEN"] = env["DOC_REDACTION_HF_TOKEN"]
    return PiRpcClient(cwd=repo_root, env=env)
