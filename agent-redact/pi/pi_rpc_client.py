"""Python client for Pi RPC mode (JSONL over stdin/stdout)."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


class PiRpcError(RuntimeError):
    pass


# Sentinel pushed to every pending response slot and the events queue when the
# Pi RPC subprocess exits, so blocked waiters unblock with a clear error instead
# of hanging forever.
_PI_PROCESS_EXIT = object()


# Pi RPC is JSONL over pipes; always UTF-8 (Windows default locale is cp1252).
_PI_SUBPROCESS_ENCODING = "utf-8"
_PI_SUBPROCESS_ENCODING_ERRORS = "replace"

_PI_INSTALL_HINT = (
    "Install the Pi coding agent CLI, then restart the Gradio app:  \n"
    "`npm install -g @earendil-works/pi-coding-agent`  \n"
    "On Windows, ensure Node.js/npm are on PATH (or set `PI_EXECUTABLE` to the "
    "full path to `pi.cmd`, e.g. `%APPDATA%\\npm\\pi.cmd`).  \n"
    "Docker users: run the Pi UI via `docker compose` (`pi-agent` service) instead "
    "of `python gradio_app.py` on the host."
)


def resolve_pi_executable() -> str:
    """Return a path to the ``pi`` RPC executable (raises ``PiRpcError`` if missing)."""
    override = os.environ.get("PI_EXECUTABLE", "").strip()
    if override:
        if os.path.isfile(override) or shutil.which(override):
            return override
        raise PiRpcError(
            f"PI_EXECUTABLE is set but not found: `{override}`  \n\n{_PI_INSTALL_HINT}"
        )
    for name in ("pi", "pi.cmd"):
        found = shutil.which(name)
        if found:
            return found
    raise PiRpcError(f"Pi CLI (`pi`) not found on PATH.  \n\n{_PI_INSTALL_HINT}")


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


def _tool_lines_from_content(content: list[Any]) -> list[str]:
    tool_lines: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type not in {"toolCall", "tool_use", "functionCall"}:
            continue
        name = str(block.get("name") or block.get("toolName") or "tool")
        args = block.get("arguments") or block.get("input") or block.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}
        tool_lines.append(f"**{name}:** {format_tool_args(name, args)}")
    return tool_lines


def format_assistant_message_for_chat(message: dict[str, Any]) -> str:
    """Render one assistant message for the chat UI (visible text or tool calls; no thinking)."""
    visible, _thinking = extract_assistant_display(message)
    if visible.strip():
        return visible

    content = message.get("content")
    if not isinstance(content, list):
        return ""

    return "\n".join(_tool_lines_from_content(content))


def chat_text_from_assistant_message(message: dict[str, Any] | None) -> str:
    """Non-thinking chat text from a Pi/Gemini assistant message snapshot."""
    if not message or message.get("role") != "assistant":
        return ""
    return format_assistant_message_for_chat(message)


_RATE_LIMIT_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate-limit",
    "resource_exhausted",
    "too many requests",
)


def is_rate_limit_error(text: str | None) -> bool:
    """True when *text* looks like a provider quota or rate-limit failure."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def last_assistant_turn_error(messages: list[dict[str, Any]]) -> str | None:
    """Return the latest assistant error in the current user turn, if any."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index

    turn_messages = messages[last_user + 1 :] if last_user >= 0 else messages
    for message in reversed(turn_messages):
        if message.get("role") != "assistant":
            continue
        error = message.get("errorMessage")
        if error:
            return str(error)
        if message.get("stopReason") == "error":
            visible, _ = extract_assistant_display(message)
            if visible.strip():
                return visible
            return "assistant turn failed"
    return None


def assistant_text_since_last_user(messages: list[dict[str, Any]]) -> str:
    """Combine assistant messages from the latest user turn."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index

    turn_messages = messages[last_user + 1 :] if last_user >= 0 else messages
    parts: list[str] = []
    for message in turn_messages:
        if message.get("role") != "assistant":
            continue
        part = format_assistant_message_for_chat(message)
        if part.strip():
            parts.append(part)
    return "\n\n".join(parts)


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

    # Extension UI dialog methods block Pi until the client replies; auto-cancel
    # them so a missing UI layer can never wedge the RPC process.
    _EXTENSION_UI_DIALOG_METHODS = frozenset({"select", "confirm", "input", "editor"})

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
        self._write_lock = threading.Lock()
        self._abort_requested = False
        self._prompt_stream_depth = 0
        self._pending_follow_ups = 0
        self._pending_ui_history: list[dict[str, Any]] = []
        # Single stdout reader thread demultiplexes the JSONL stream: command
        # responses go to per-id slots, agent events go to ``_events``. This lets
        # any thread (e.g. post-task logging) call the client safely while a
        # prompt stream is active.
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._events: queue.Queue[Any] = queue.Queue()
        self._pending_lock = threading.Lock()
        self._pending_responses: dict[str, queue.Queue[Any]] = {}
        self._stderr_buffer: deque[str] = deque(maxlen=200)
        self._closing = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def prompt_stream_active(self) -> bool:
        """True while :meth:`prompt_events` is consuming the RPC event stream."""
        return self._prompt_stream_depth > 0

    def start(self) -> None:
        if self.running:
            return
        command = [resolve_pi_executable(), "--mode", "rpc", *self._pi_args]
        self._closing = False
        self._abort_requested = False
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding=_PI_SUBPROCESS_ENCODING,
            errors=_PI_SUBPROCESS_ENCODING_ERRORS,
            bufsize=1,
            cwd=self._cwd,
            env=self._env,
        )
        self._proc = proc
        # Fresh demux state for this process.
        self._events = queue.Queue()
        with self._pending_lock:
            self._pending_responses = {}
        self._stderr_buffer = deque(maxlen=200)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(proc,),
            name="pi-rpc-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        if proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop,
                args=(proc,),
                name="pi-rpc-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

    def close(self) -> None:
        if not self._proc:
            return
        self._closing = True
        proc = self._proc
        if proc.poll() is None:
            try:
                self.abort()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Process exit makes ``readline`` return EOF; the reader thread then
        # notifies waiters. Nudge waiters here too in case the threads are slow.
        self._notify_process_exit()
        for thread in (self._reader_thread, self._stderr_thread):
            if (
                thread is not None
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=2)
        self._reader_thread = None
        self._stderr_thread = None
        self._proc = None

    def _ensure_running(self) -> subprocess.Popen[str]:
        if not self.running:
            self.start()
        assert self._proc is not None
        return self._proc

    def _recent_stderr(self) -> str:
        return "\n".join(self._stderr_buffer)

    def _process_exit_error(self) -> PiRpcError:
        code = self._proc.poll() if self._proc else None
        err = self._recent_stderr()
        return PiRpcError(
            f"Pi RPC process exited (code={code})."
            + (f" stderr: {err[:500]}" if err else "")
        )

    def _notify_process_exit(self) -> None:
        """Unblock every pending response slot and the events queue on exit."""
        with self._pending_lock:
            pending = list(self._pending_responses.values())
            self._pending_responses.clear()
        for slot in pending:
            try:
                slot.put_nowait(_PI_PROCESS_EXIT)
            except queue.Full:
                pass
        try:
            self._events.put_nowait(_PI_PROCESS_EXIT)
        except queue.Full:
            pass

    def _stderr_loop(self, proc: subprocess.Popen[str]) -> None:
        """Continuously drain stderr into a bounded buffer (prevents pipe deadlock)."""
        stream = proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr_buffer.append(line.rstrip("\r\n"))
        except (ValueError, OSError):
            pass

    def _reader_loop(self, proc: subprocess.Popen[str]) -> None:
        """Read every stdout line and route responses vs. agent events."""
        stream = proc.stdout
        if stream is None:
            self._notify_process_exit()
            return
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._dispatch_message(message)
        except (ValueError, OSError):
            pass
        finally:
            self._notify_process_exit()

    def _dispatch_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        msg_type = message.get("type")
        if msg_type == "response":
            req_id = message.get("id")
            slot: queue.Queue[Any] | None = None
            if req_id is not None:
                with self._pending_lock:
                    slot = self._pending_responses.pop(str(req_id), None)
            if slot is not None:
                try:
                    slot.put_nowait(message)
                except queue.Full:
                    pass
            return
        if msg_type == "extension_ui_request":
            self._auto_reply_extension_ui(message)
            return
        # Agent event — consumed by the active ``prompt_events`` stream.
        self._events.put(message)

    def _auto_reply_extension_ui(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        req_id = message.get("id")
        if req_id is None or method not in self._EXTENSION_UI_DIALOG_METHODS:
            return
        try:
            self._write_command(
                {"type": "extension_ui_response", "id": req_id, "cancelled": True}
            )
        except (OSError, PiRpcError):
            pass

    def _write_command(self, command: dict[str, Any]) -> None:
        proc = self._ensure_running()
        assert proc.stdin is not None
        with self._write_lock:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()

    def _send_command(
        self,
        command: dict[str, Any],
        *,
        wait_response: bool = True,
    ) -> dict[str, Any] | None:
        req_id = str(command.setdefault("id", str(uuid.uuid4())))
        if not wait_response:
            self._write_command(command)
            return None
        slot: queue.Queue[Any] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending_responses[req_id] = slot
        try:
            self._write_command(command)
        except Exception:
            with self._pending_lock:
                self._pending_responses.pop(req_id, None)
            raise
        result = slot.get()
        if result is _PI_PROCESS_EXIT:
            raise self._process_exit_error()
        if not result.get("success", False):
            error = result.get("error") or result.get("message") or "command failed"
            raise PiRpcError(str(error))
        return result

    def abort(self) -> None:
        """Request abort without reading stdout (the active stream consumer drains events)."""
        if not self.running:
            return
        self._abort_requested = True
        try:
            self._send_command({"type": "abort"}, wait_response=False)
        except OSError:
            pass

    def stage_ui_chat_notice(self, label: str, message: str) -> None:
        """Stage user/assistant chat rows for the active prompt stream to merge on yield."""
        text = message.strip()
        if not text:
            return
        self._pending_ui_history.append(
            {"role": "user", "content": f"_**{label}:**_ {text}"}
        )
        self._pending_ui_history.append({"role": "assistant", "content": ""})

    def drain_pending_ui_history(self) -> list[dict[str, Any]]:
        """Return and clear UI chat rows staged by :meth:`stage_ui_chat_notice`."""
        pending = self._pending_ui_history[:]
        self._pending_ui_history.clear()
        return pending

    def steer(self, message: str) -> None:
        """Queue a steering message (delivered after the current tool step completes)."""
        if not message.strip():
            return
        self._send_command(
            {"type": "steer", "message": message},
            wait_response=False,
        )

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message for when the agent stops."""
        if not message.strip():
            return
        self._pending_follow_ups += 1
        self._send_command(
            {"type": "follow_up", "message": message},
            wait_response=False,
        )

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

    def get_messages(self) -> list[dict[str, Any]]:
        response = self._send_command({"type": "get_messages"})
        data = response.get("data") if response else {}
        messages = data.get("messages") if isinstance(data, dict) else []
        return messages if isinstance(messages, list) else []

    def get_session_stats(self) -> dict[str, Any]:
        """Token usage and cost totals for the active session (Pi RPC ``get_session_stats``)."""
        response = self._send_command({"type": "get_session_stats"})
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
        self._prompt_stream_depth += 1
        try:
            yield from self._prompt_events_impl(message)
        finally:
            self._prompt_stream_depth = max(0, self._prompt_stream_depth - 1)

    def _drain_events(self) -> None:
        """Discard stale events left over from a prior stream (single active prompt)."""
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                return
            if item is _PI_PROCESS_EXIT:
                # Preserve the exit signal for the consumer to observe.
                try:
                    self._events.put_nowait(_PI_PROCESS_EXIT)
                except queue.Full:
                    pass
                return

    def _prompt_events_impl(self, message: str) -> Iterator[PiStreamEvent]:
        self.clear_abort()
        self._drain_events()
        try:
            self._send_command({"type": "prompt", "message": message})
        except PiRpcError as exc:
            yield PiStreamEvent(kind="error", text=str(exc), is_error=True)
            return

        yield from self._iter_agent_events()

    def _iter_agent_events(self) -> Iterator[PiStreamEvent]:
        while True:
            event = self._events.get()
            if event is _PI_PROCESS_EXIT:
                raise self._process_exit_error()
            event_type = event.get("type")

            if event_type == "agent_start":
                yield PiStreamEvent(kind="status", text="Agent started…")

            elif event_type == "turn_start":
                yield PiStreamEvent(kind="status", text="Turn started.")

            elif event_type == "turn_end":
                yield PiStreamEvent(kind="turn_end", text="Turn finished.")

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
                        kind="queue_update",
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
                # Pi delivers queued ``follow_up`` messages after ``agent_end`` and
                # continues streaming; do not stop the stdout consumer until they run.
                if self._pending_follow_ups > 0:
                    self._pending_follow_ups -= 1
                    yield PiStreamEvent(
                        kind="status",
                        text="Follow-up queued — continuing…",
                    )
                    continue
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
            if visible.strip():
                yield PiStreamEvent(kind="text_snapshot", text=visible)
            elif chat_text := chat_text_from_assistant_message(partial):
                yield PiStreamEvent(kind="text_snapshot", text=chat_text)
            if thinking.strip():
                yield PiStreamEvent(kind="thinking_snapshot", text=thinking)

        if delta_type == "text_delta":
            chunk = delta.get("delta") or ""
            if chunk:
                yield PiStreamEvent(kind="text_delta", text=chunk)

        elif delta_type == "thinking_delta":
            chunk = delta.get("delta") or ""
            if chunk:
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
            detail = format_tool_args(str(tool_name), tool_args)
            chat_line = f"**{tool_name}:** {detail}" if detail else f"**{tool_name}**"
            yield PiStreamEvent(kind="text_snapshot", text=chat_line)

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


def default_client(session_hash: str | None = None) -> PiRpcClient:
    from pi_agent_config import configure_aws_credentials
    from pi_workspace_skills import ensure_workspace_skills, pi_rpc_args, pi_rpc_cwd

    configure_aws_credentials()
    ensure_workspace_skills()
    env = os.environ.copy()
    env.setdefault("HOME", os.path.expanduser("~"))
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    from session_workspace import workspace_base_dir

    env.setdefault("PI_WORKSPACE_DIR", str(workspace_base_dir()))
    if not env.get("GEMINI_API_KEY") and env.get("GOOGLE_API_KEY"):
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    if not env.get("HF_TOKEN") and env.get("DOC_REDACTION_HF_TOKEN"):
        env["HF_TOKEN"] = env["DOC_REDACTION_HF_TOKEN"]
    return PiRpcClient(
        cwd=pi_rpc_cwd(session_hash),
        env=env,
        pi_args=pi_rpc_args(),
    )
