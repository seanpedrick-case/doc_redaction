#!/usr/bin/env python3
"""
Gradio chat UI for agentic redaction.

Streams events into a chatbot, activity log, tool output panel, and
optional thinking trace. Includes a redaction task panel driven by the
partnership prompt template.
"""

from __future__ import annotations

import base64
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Enable debug-level logging and stderr output from this module when PI_RPC_DEBUG is set
if os.environ.get("PI_RPC_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.INFO)
_logger = logging.getLogger(__name__)

if os.environ.get("PI_RPC_DEBUG") and not _logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    _logger.addHandler(handler)

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_pi_config import ensure_pi_config_env

ensure_pi_config_env(_REPO_ROOT)

import gradio as gr
from agent_runtime import (
    AgentRuntime,
    AgentRuntimeError,
    AgentStreamEvent,
    coerce_agent_runtime,
    create_agent_runtime,
    normalize_orchestrator,
    orchestrator_label,
    start_agent_prompt_event_worker,
)
from output_files import (
    collect_final_output_files,
    gradio_allowed_paths,
    latest_redacted_pdf_path,
    preview_pdf_path_for_gradio,
    refresh_workspace_output_files_stub,
    refresh_workspace_panel,
    workspace_files_download_fn,
)
from pi_agent_config import (
    LLAMA_BASE_URL,
    PROVIDER_LLAMA,
    apply_session_credentials,
    configure_aws_credentials,
    credential_status_markdown,
    default_model_for_provider,
    gemini_api_key_configured,
    get_default_provider,
    is_hf_space_profile,
    llama_model_id,
    mirror_hf_token_from_env,
    models_for_provider,
    normalize_backend_model,
    normalize_provider,
    provider_choices,
    provider_label,
    resolved_default_model,
    write_runtime_config,
)
from pi_examples import example_rows, examples_status_markdown
from pi_rpc_client import (
    PiRpcError,
    assistant_text_since_last_user,
    format_tool_chat_line,
    is_rate_limit_error,
    last_assistant_turn_error,
)
from redaction_prompt import (
    DEFAULT_OCR_METHOD,
    DEFAULT_PII_METHOD,
    OCR_METHOD_CHOICES,
    PII_METHOD_CHOICES,
    RedactionTaskSettings,
    pages_to_process_count,
    pdf_page_count,
    prepare_redaction_task,
)
from session_logs import collect_session_log_download, persist_session_log

# Before any ``tools.config`` import (e.g. session_workspace): compose may inject
# empty AWS_REGION= which would freeze a blank region in tools.config.AWS_REGION.
mirror_hf_token_from_env()
configure_aws_credentials()

from pi_session_usage import resolve_session_token_usage, usage_for_completed_turn
from session_workspace import (
    init_session_workspace,
    prepare_session_workspace,
    session_workspace_dir,
    workspace_base_dir,
    workspace_context_prefix,
)

from tools.aws_functions import export_outputs_to_s3, s3_outputs_upload_ready
from tools.config import (
    ACTIVITY_MAX_LINES,
    EMPTY_SEND_WITH_FILE_HINT,
    FASTAPI_ROOT_PATH,
    HOST_NAME,
    PI_INTRO_TEXT,
    PI_ROOT_PATH,
    PI_UI_HOST,
    PI_UI_PORT,
    PI_UI_TITLE,
    QUOTA_CONTINUE_PROMPT,
    QUOTA_RETRY_ATTEMPTS,
    QUOTA_RETRY_DELAY_S,
    RUN_FASTAPI,
    SAVE_OUTPUTS_TO_S3,
    SHOW_THINKING,
    SHOW_TOOL_OUTPUT,
    THINKING_DISPLAY_MAX,
    THINKING_PANEL_CSS,
    TOOL_OUTPUT_MAX,
)

# After ``tools.config`` import: it may set ``PI_DEFAULT_PROVIDER`` / ``PI_DEFAULT_MODEL``
# when unset (must match ``pi_agent_config.get_default_provider``, not always Gemini).
write_runtime_config()

from tools.gradio_platform import (
    create_fastapi_app,
    log_agent_usage_event,
    log_platform_access,
    mount_or_launch,
)

IS_HF_SPACE = is_hf_space_profile()

AGENT_FINISH_SIGNAL_NONE = ""
AGENT_FINISH_SIGNAL_FINISHED = "finished"
AGENT_FINISH_SIGNAL_ABORTED = "aborted"
AGENT_FINISH_SIGNAL_ERROR = "error"

# Must match ``chat_outputs`` in :func:`build_ui` (Gradio validates return count).
_CHAT_OUTPUT_COMPONENT_COUNT = 15
# Index of ``agent_finish_signal`` in ``chat_outputs`` (for finish-notification JS).
_CHAT_OUTPUT_AGENT_FINISH_SIGNAL_IDX = 13
# File/PDF slots in ``chat_outputs`` — must not be ``.then()`` *inputs* (Gradio 6 stores
# prior ``gr.skip()`` as ``{'__type__': 'update'}``, which fails FileData validation).
_CHAT_FILE_OUTPUT_INDICES = frozenset({10, 11, 12})

PI_AGENT_FINISH_HEAD_HTML = """
<script>
(function () {
  function requestNotificationPermissionOnce() {
    if (typeof Notification === "undefined") return;
    if (Notification.permission !== "default") return;
    try { Notification.requestPermission(); } catch (e) {}
  }
  document.addEventListener("click", requestNotificationPermissionOnce, { once: true });
  document.addEventListener("keydown", requestNotificationPermissionOnce, { once: true });
})();
</script>
"""

PI_AGENT_FINISH_NOTIFY_JS = (
    """
async (...outputs) => {
  const finishSignal = outputs["""
    + str(_CHAT_OUTPUT_AGENT_FINISH_SIGNAL_IDX)
    + """];
  if (!finishSignal) {
    return outputs;
  }
  const isAborted = finishSignal === "aborted";
  const isError = finishSignal === "error";
  const title = isAborted ? "Agent stopped" : (isError ? "Agent error" : "Agent finished");
  const body = isAborted
    ? "The agent run was aborted."
    : (isError
      ? "The agent run ended with an error."
      : "The agent has finished its task. Review the chat for results.");
  const originalTitle = document.title;
  let flashOn = true;
  const flashInterval = setInterval(() => {
    document.title = flashOn ? ("✓ " + title) : originalTitle;
    flashOn = !flashOn;
  }, 1000);
  setTimeout(() => {
    clearInterval(flashInterval);
    document.title = originalTitle;
  }, 15000);
  if (typeof Notification !== "undefined") {
    try {
      if (Notification.permission === "granted") {
        new Notification(title, { body: body, tag: "pi-agent-finish" });
      } else if (Notification.permission === "default") {
        const perm = await Notification.requestPermission();
        if (perm === "granted") {
          new Notification(title, { body: body, tag: "pi-agent-finish" });
        }
      }
    } catch (e) {}
  }
  outputs["""
    + str(_CHAT_OUTPUT_AGENT_FINISH_SIGNAL_IDX)
    + """] = "";
  return outputs;
}
"""
)

app = None


def _agent_finish_chat_notice(*, aborted: bool = False, error: bool = False) -> str:
    if aborted:
        return (
            "---\n\n"
            "**Agent stopped** — the run was aborted. You can send a follow-up message "
            "or start a new task."
        )
    if error:
        return (
            "---\n\n"
            "**Agent stopped** — the run ended with an error. Review the activity log "
            "and send a follow-up if needed."
        )
    return (
        "---\n\n"
        "**Agent finished** — the task is complete. Review the outputs below or send "
        "a follow-up message if you need changes."
    )


def _show_agent_finish_toast(*, aborted: bool = False, error: bool = False) -> None:
    try:
        if aborted:
            gr.Info("Agent stopped (aborted).", duration=8)
        elif error:
            gr.Info("Agent stopped with an error.", duration=8)
        else:
            gr.Info("Agent finished — task complete.", duration=8)
    except Exception:
        pass


def _agent_finish_signal_value(*, aborted: bool = False, error: bool = False) -> str:
    if error:
        return AGENT_FINISH_SIGNAL_ERROR
    if aborted:
        return AGENT_FINISH_SIGNAL_ABORTED
    return AGENT_FINISH_SIGNAL_FINISHED


def _notify_agent_finished(*, aborted: bool = False, error: bool = False) -> str:
    """Show Gradio toast and return browser-notify signal for the finish handler."""
    _show_agent_finish_toast(aborted=aborted, error=error)
    return _agent_finish_signal_value(aborted=aborted, error=error)


def _append_agent_finish_notice(
    history: list[dict[str, Any]],
    completed_segments: list[str],
    streaming_text: str,
    *,
    aborted: bool = False,
    error: bool = False,
) -> tuple[list[dict[str, Any]], list[str], str]:
    note = _agent_finish_chat_notice(aborted=aborted, error=error)
    completed_segments, streaming_text = _append_chat_segment(
        completed_segments, streaming_text, note
    )
    if history and _last_assistant_index(history) >= 0:
        _set_last_assistant_content(
            history,
            _assistant_display_text(completed_segments, streaming_text),
        )
    return history, completed_segments, streaming_text


def _chat_outputs_notify_inputs(chat_outputs: list[Any]) -> list[Any]:
    """``chat_outputs`` minus File/PDF components (safe as ``.then()`` inputs)."""
    return [
        component
        for index, component in enumerate(chat_outputs)
        if index not in _CHAT_FILE_OUTPUT_INDICES
    ]


def _passthrough_chat_outputs(*outputs: Any) -> tuple[Any, ...]:
    """
    Passthrough for ``.then(js=...)`` — Gradio forces ``queue=False`` when ``fn is None``.

    When the parent generator is cancelled or input resolution fails, Gradio may invoke
    this with fewer than ``_CHAT_OUTPUT_COMPONENT_COUNT`` values. Pad with ``gr.skip()``
    so validation always receives the expected number of outputs.
    """
    n = _CHAT_OUTPUT_COMPONENT_COUNT
    if len(outputs) >= n:
        return tuple(outputs[:n])
    padded = list(outputs) + [gr.skip()] * (n - len(outputs))
    return tuple(padded)


def _passthrough_chat_outputs_for_notify(*non_file_outputs: Any) -> tuple[Any, ...]:
    """
    Echo non-file ``chat_outputs`` for finish-notification ``.then(js=...)`` chains.

    File/PDF components are omitted from ``.then()`` *inputs* because Gradio 6 rejects
    their stored ``gr.skip()`` placeholder (``{'__type__': 'update'}``) during
    ``FileData`` validation — e.g. when **Abort** cancels an in-flight chat run.
    Outputs use ``gr.skip()`` for those slots so existing downloads/previews are kept.
    """
    expected_non_file = _CHAT_OUTPUT_COMPONENT_COUNT - len(_CHAT_FILE_OUTPUT_INDICES)
    values = list(non_file_outputs)
    if len(values) < expected_non_file:
        values.extend([gr.skip()] * (expected_non_file - len(values)))
    values = values[:expected_non_file]
    non_file_iter = iter(values)
    full: list[Any] = []
    for index in range(_CHAT_OUTPUT_COMPONENT_COUNT):
        if index in _CHAT_FILE_OUTPUT_INDICES:
            full.append(gr.skip())
        else:
            full.append(next(non_file_iter))
    return tuple(full)


def _client_provider_model(client: AgentRuntime | None) -> tuple[str, str]:
    if client is None:
        return "", ""
    try:
        state = client.get_state()
    except PiRpcError:
        return "", ""
    model = state.get("model") or {}
    provider = str(model.get("provider") or state.get("provider") or "")
    model_label = str(model.get("id") or model.get("name") or "")
    return provider, model_label


def _llm_model_label(client: AgentRuntime | None) -> str:
    provider, model = _client_provider_model(client)
    if provider and model:
        return f"{provider}/{model}"
    return model or provider


def _schedule_post_pi_task(
    *,
    session_hash: str,
    client: AgentRuntime | None,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    document_name: str = "",
    started_at: float | None = None,
    base_file: str | None = None,
    ocr_method: str = "",
    pii_method: str = "",
    total_page_count: int = 0,
    vlm_model_name: str | None = None,
    usage_baseline: Any = None,
) -> None:
    """
    Run usage logging, session-log export, and optional S3 upload off the hot path.

    Keeps the Gradio generator alive only long enough to release follow-up Send
    handlers after the agent finishes.
    """
    usage = usage_for_completed_turn(client, usage_baseline)
    # Resolve everything that needs an RPC read *now* (on the generator thread),
    # so the background worker never touches the Pi stdout/stdin pipes — avoiding
    # contention with the next follow-up prompt stream.
    from session_logs import pi_session_file_from_client

    llm_model_label = _llm_model_label(client)
    session_log_source = pi_session_file_from_client(client)

    def _work() -> None:
        try:
            _after_pi_task(
                session_hash=session_hash,
                client=client,
                s3_output_folder=s3_output_folder,
                save_outputs_to_s3=save_outputs_to_s3,
                document_name=document_name,
                started_at=started_at,
                base_file=base_file,
                ocr_method=ocr_method,
                pii_method=pii_method,
                total_page_count=total_page_count,
                vlm_model_name=vlm_model_name,
                llm_input_tokens=usage.llm_input_tokens,
                llm_output_tokens=usage.llm_output_tokens,
                llm_model_label=llm_model_label,
                session_log_source=session_log_source,
                read_client=False,
            )
        except Exception:
            pass

    threading.Thread(target=_work, daemon=True).start()


def _after_pi_task(
    *,
    session_hash: str,
    client: AgentRuntime | None,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    document_name: str = "",
    started_at: float | None = None,
    base_file: str | None = None,
    ocr_method: str = "",
    pii_method: str = "",
    total_page_count: int = 0,
    vlm_model_name: str | None = None,
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    llm_model_label: str | None = None,
    session_log_source: Any = None,
    read_client: bool = True,
) -> str | None:
    """
    Run post-task logging/upload. Returns a user-visible warning when S3 upload fails.

    When called off the hot path (background thread), the caller passes
    *llm_model_label* / *session_log_source* (resolved synchronously) and
    ``read_client=False`` so no RPC reads happen on the background thread.
    """
    duration = round(time.time() - started_at, 2) if started_at else ""
    if llm_model_label is None and read_client:
        llm_model_label = _llm_model_label(client)
    log_agent_usage_event(
        session_hash=session_hash,
        duration_seconds=duration,
        document_name=document_name,
        total_page_count=total_page_count,
        ocr_method=ocr_method,
        pii_method=pii_method,
        llm_model_name=llm_model_label or "",
        vlm_model_name=vlm_model_name or os.environ.get("PI_VLM_MODEL", ""),
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        task="agent",
    )
    persist_session_log(
        client if read_client else None,
        session_hash=session_hash,
        source=session_log_source,
    )
    file_paths = collect_final_output_files(session_hash)
    if (
        file_paths
        and s3_output_folder
        and s3_outputs_upload_ready(save_outputs_to_s3=save_outputs_to_s3)
    ):
        return export_outputs_to_s3(
            file_paths,
            s3_output_folder,
            save_outputs_to_s3,
            base_file,
        )
    return None


def _export_workspace_outputs(
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    base_file: str | None = None,
) -> None:
    file_paths = collect_final_output_files(session_hash)
    if (
        file_paths
        and s3_output_folder
        and s3_outputs_upload_ready(save_outputs_to_s3=save_outputs_to_s3)
    ):
        export_outputs_to_s3(
            file_paths,
            s3_output_folder,
            save_outputs_to_s3,
            base_file,
        )


def _clone_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": item["role"], "content": item["content"]} for item in history]


def _last_assistant_index(history: list[dict[str, Any]]) -> int:
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "assistant":
            return index
    return -1


def _set_last_assistant_content(history: list[dict[str, Any]], content: str) -> None:
    """Update the latest assistant bubble (never a queued steer user line)."""
    index = _last_assistant_index(history)
    if index >= 0:
        history[index]["content"] = content
    else:
        history.append({"role": "assistant", "content": content})


def _append_rate_limit_wait_notice(
    history: list[dict[str, Any]],
    completed_segments: list[str],
    streaming_text: str,
    message: str,
) -> tuple[list[dict[str, Any]], list[str], str]:
    completed_segments, streaming_text = _append_chat_segment(
        completed_segments,
        streaming_text,
        message,
    )
    _set_last_assistant_content(
        history, _assistant_display_text(completed_segments, streaming_text)
    )
    return history, completed_segments, streaming_text


def _user_notice_content(label: str, message: str) -> str:
    return f"_**{label}:**_ {message.strip()}"


def _history_has_user_notice(
    history: list[dict[str, Any]],
    *,
    label: str,
    message: str,
) -> bool:
    expected = _user_notice_content(label, message)
    return any(
        item.get("role") == "user" and item.get("content") == expected
        for item in history
    )


def _append_user_steer_notice(
    history: list[dict[str, Any]],
    *,
    label: str,
    message: str,
) -> list[dict[str, Any]]:
    """Append a steer/follow-up user line and an empty assistant slot for the reply."""
    text = message.strip()
    if not text:
        return history
    history = _clone_history(history)
    history.append({"role": "user", "content": _user_notice_content(label, text)})
    history.append({"role": "assistant", "content": ""})
    return history


def _integrate_pending_chat_notices(
    history: list[dict[str, Any]],
    client: AgentRuntime,
    completed_segments: list[str],
    streaming_text: str,
) -> tuple[list[dict[str, Any]], list[str], str]:
    """
    Merge steer/follow-up rows staged on the RPC client into the active stream history.

    The Send handler stages notices immediately; the long-running prompt stream must
    absorb them so its next yield does not overwrite the chatbot with stale history.
    """
    pending = client.drain_pending_ui_history()
    if not pending:
        return history, completed_segments, streaming_text

    history = _clone_history(history)
    added = False
    index = 0
    while index < len(pending):
        item = pending[index]
        if item.get("role") == "user":
            content = str(item.get("content") or "")
            if any(
                existing.get("role") == "user" and existing.get("content") == content
                for existing in history
            ):
                index += 2 if index + 1 < len(pending) else 1
                continue
            history.append(item)
            added = True
            if (
                index + 1 < len(pending)
                and pending[index + 1].get("role") == "assistant"
            ):
                history.append(pending[index + 1])
                index += 2
            else:
                history.append({"role": "assistant", "content": ""})
                index += 1
            continue
        history.append(item)
        added = True
        index += 1

    if added:
        return history, [], ""
    return history, completed_segments, streaming_text


def _truncate_thinking(text: str, limit: int = THINKING_DISPLAY_MAX) -> str:
    if len(text) <= limit:
        return text
    hidden = len(text) - limit
    return f"… [{hidden:,} earlier chars hidden]\n\n{text[-limit:]}"


def _assistant_display_text(completed_segments: list[str], current: str) -> str:
    parts = [segment.strip() for segment in completed_segments if segment.strip()]
    if current.strip():
        parts.append(current.strip())
    return "\n\n".join(parts)


def _assistant_turn_has_response(messages: list[dict[str, Any]]) -> bool:
    """True when the latest user turn includes any assistant message."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    turn_messages = messages[last_user + 1 :] if last_user >= 0 else messages
    return any(message.get("role") == "assistant" for message in turn_messages)


def _is_agent_finish_notice_only(content: str) -> bool:
    """True when chat shows only the generic end-of-run banner (no real answer)."""
    stripped = content.strip()
    if not stripped:
        return False
    markers = (
        "**Agent finished**",
        "**Agent stopped**",
    )
    return any(marker in stripped for marker in markers) and len(stripped) < 400


def _silent_llama_failure_message() -> str:
    return (
        f"**LLM:** no response from the orchestration model.  \n\n"
        f"Configured endpoint: `{LLAMA_BASE_URL}` · model `{llama_model_id()}`.  \n\n"
        f"Set **`PI_LLAMA_BASE_URL`** in `config/pi_agent.env` (OpenAI-compatible URL "
        f"including `/v1`, e.g. `http://192.168.0.220:8080/v1`), confirm the model id "
        f"matches your server (`GET …/v1/models`), then click **Apply backend** or "
        f"restart this UI.  \n\n"
        f"On llama-swap, the first request can take 30–60s while the model loads."
    )


def _llama_terminated_message() -> str:
    return (
        f"**LLM:** llama.cpp closed the connection (`terminated`).  \n\n"
        f"This usually means the **llama server process was killed** during prompt "
        f"prefill (common causes: GPU out-of-memory, or the model is still loading/unloading). "
        f"The server often exits **without an error line** when the OS kills it.  \n\n"
        f"Endpoint: `{LLAMA_BASE_URL}` · model `{llama_model_id()}`.  \n\n"
        f"Wait until `GET {LLAMA_BASE_URL.rstrip('/')}/models` responds, ensure no other "
        f"client is hitting the same endpoint (doc_redaction local PII shares it), then "
        f"retry **Start redaction task**. If using Pi, after changing skills sync, set "
        f"`PI_SKILLS_RESYNC=true` once and restart the Gradio UI."
    )


def _format_llama_turn_error(turn_error: str) -> str:
    if (turn_error or "").strip().lower() == "terminated":
        return _llama_terminated_message()
    return turn_error


def _wait_for_llama_inference_ready(*, timeout_s: float = 120.0) -> str | None:
    """Return an error message when llama.cpp is not reachable before orchestration."""
    import urllib.error
    import urllib.request

    url = f"{LLAMA_BASE_URL.rstrip('/')}/models"
    deadline = time.time() + max(timeout_s, 1.0)
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return None
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    return (
        f"llama.cpp not ready at `{url}` after {int(timeout_s)}s "
        f"(last error: {last_error or 'timeout'})"
    )


def _prepare_llama_before_orchestration_prompt() -> str | None:
    """Pause and verify llama.cpp responds before a large orchestration prefill."""
    if not _uses_local_llama_orchestrator():
        return None
    delay = float(os.environ.get("PI_LLAMA_POST_RESTART_DELAY_S", "2"))
    if delay > 0:
        time.sleep(delay)
    timeout = float(os.environ.get("PI_LLAMA_READY_TIMEOUT_S", "120"))
    return _wait_for_llama_inference_ready(timeout_s=timeout)


def _uses_local_llama_orchestrator() -> bool:
    """True when Pi/LangGraph orchestration calls a local llama.cpp OpenAI endpoint."""
    if normalize_orchestrator() in {"agentcore", "agentcore-harness"}:
        return False
    return normalize_provider(get_default_provider()) == PROVIDER_LLAMA


def _finalize_assistant_chat(
    client: AgentRuntime,
    history: list[dict[str, Any]],
    *,
    completed_segments: list[str],
    streaming_text: str,
    activity: list[str],
) -> None:
    """Fill an empty assistant bubble after tool-only Gemini turns."""
    assistant_index = _last_assistant_index(history)
    if assistant_index < 0:
        return
    if _assistant_display_text(completed_segments, streaming_text).strip():
        _set_last_assistant_content(
            history,
            _assistant_display_text(completed_segments, streaming_text),
        )
        return
    existing = history[assistant_index].get("content", "").strip()
    if existing and not _is_agent_finish_notice_only(existing):
        return

    try:
        messages = client.get_messages()
        fallback = assistant_text_since_last_user(messages)
        turn_error = last_assistant_turn_error(messages)
    except PiRpcError:
        messages = []
        fallback = ""
        turn_error = None

    if turn_error:
        content = _format_llama_turn_error(turn_error)
        if not content.startswith("**LLM:"):
            content = f"**LLM error:** {turn_error}"
        _set_last_assistant_content(history, content)
        return

    if fallback.strip():
        _set_last_assistant_content(history, fallback)
        return

    if (
        normalize_provider(get_default_provider()) == PROVIDER_LLAMA
        and _uses_local_llama_orchestrator()
    ):
        _set_last_assistant_content(history, _silent_llama_failure_message())
        return

    if activity:
        _set_last_assistant_content(
            history,
            (
                "_This run completed using tools only (no assistant prose was streamed). "
                "See **Thinking log** for step-by-step activity._"
            ),
        )


def _gemini_key_error() -> str | None:
    if IS_HF_SPACE and not gemini_api_key_configured():
        return (
            "**Gemini API key required.** Paste your key in **Agent backend** and click "
            "**Apply backend** before chatting or starting a redaction task."
        )
    return None


def _ensure_client(
    client: AgentRuntime | None,
    session_hash: str = "",
) -> AgentRuntime:
    key_error = _gemini_key_error()
    if key_error:
        raise PiRpcError(key_error)
    if isinstance(client, AgentRuntime) and client.running:
        return client
    client = create_agent_runtime(session_hash or None)
    client.start()
    provider = normalize_provider(get_default_provider())
    model = resolved_default_model(provider)
    try:
        client.set_model(provider, model)
    except PiRpcError:
        pass
    return client


def _coerce_client(client: Any) -> AgentRuntime | None:
    return coerce_agent_runtime(client)


def _truncate(text: str, limit: int = TOOL_OUTPUT_MAX) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + f"\n\n… [{len(text) - limit + 40} chars truncated]"


def _format_activity(lines: list[str]) -> str:
    if not lines:
        return "_No activity yet._"
    return "\n".join(f"- {line}" for line in lines[-ACTIVITY_MAX_LINES:])


def _append_activity(lines: list[str], text: str) -> list[str]:
    text = text.strip()
    if text:
        lines.append(text)
    return lines


def _chat_segment_tool_label(segment: str) -> str | None:
    """Tool name from a chat line like ``**bash:** ...`` or bare ``**tool**``."""
    line = segment.strip().split("\n", 1)[0]
    if not line.startswith("**"):
        return None
    if ":**" in line:
        return line[2:].split(":**", 1)[0].strip().lower() or None
    if line.endswith("**") and line.count("**") == 2:
        return line[2:-2].strip().lower() or None
    return None


def _is_empty_tool_chat_segment(segment: str) -> bool:
    """Skip ephemeral snapshots before tool arguments are populated."""
    label = _chat_segment_tool_label(segment)
    if label is None:
        return False
    if ":**" not in segment.split("\n", 1)[0]:
        return True
    detail = segment.split(":**", 1)[1].strip()
    if not detail:
        return True
    return '{"command": ""}' in detail or detail in ("`{}`", "{}")


def _should_replace_tool_chat_segment(previous: str, segment: str) -> bool:
    """True when *segment* updates the same in-flight tool line as *previous*."""
    prev_label = _chat_segment_tool_label(previous)
    new_label = _chat_segment_tool_label(segment)
    if prev_label and new_label and prev_label == new_label:
        return True
    if prev_label == "tool" and new_label:
        return True
    if previous.strip() == "**tool**" and new_label:
        return True
    return False


def _queue_user_notice(
    history: list[dict[str, Any]],
    *,
    label: str,
    message: str,
) -> list[dict[str, Any]]:
    """Append a user chat line and assistant slot for a queued steer or follow-up."""
    return _append_user_steer_notice(history, label=label, message=message)


def _format_queue_update_activity(
    steering: list[Any],
    follow_up: list[Any],
) -> list[str]:
    lines: list[str] = []
    for message in steering:
        text = str(message).strip()
        if text:
            preview = text if len(text) <= 200 else text[:197] + "…"
            lines.append(f"**Steer queued:** {preview}")
    for message in follow_up:
        text = str(message).strip()
        if text:
            preview = text if len(text) <= 200 else text[:197] + "…"
            lines.append(f"**Follow-up queued:** {preview}")
    return lines


def _append_chat_segment(
    completed_segments: list[str],
    streaming_text: str,
    segment: str,
) -> tuple[list[str], str]:
    """Append a new visible chat segment (tool line or prose), preserving prior segments."""
    segment = segment.strip()
    if not segment:
        return completed_segments, streaming_text
    if _is_empty_tool_chat_segment(segment):
        if completed_segments and _should_replace_tool_chat_segment(
            completed_segments[-1], segment
        ):
            return completed_segments, streaming_text
        if not completed_segments:
            return completed_segments, streaming_text
    if streaming_text.strip():
        completed_segments = completed_segments + [streaming_text.strip()]
        streaming_text = ""
    if completed_segments and _should_replace_tool_chat_segment(
        completed_segments[-1], segment
    ):
        completed_segments = completed_segments[:-1] + [segment]
    elif not completed_segments or completed_segments[-1] != segment:
        completed_segments = completed_segments + [segment]
    return completed_segments, streaming_text


def _apply_event(
    event: AgentStreamEvent,
    *,
    history: list[dict[str, Any]],
    activity: list[str],
    thinking: str,
    tool_output: str,
    tool_heading: str,
    completed_segments: list[str],
    streaming_text: str,
    append_finish_notice: bool = True,
) -> tuple[list[dict[str, Any]], list[str], str, str, str, list[str], str]:
    if event.kind == "text_snapshot":
        if event.text.strip().startswith("**") and ":" in event.text.split("\n", 1)[0]:
            completed_segments, streaming_text = _append_chat_segment(
                completed_segments, streaming_text, event.text
            )
        else:
            streaming_text = event.text
        _set_last_assistant_content(
            history,
            _assistant_display_text(completed_segments, streaming_text),
        )

    elif event.kind == "text_delta":
        streaming_text += event.text
        _set_last_assistant_content(
            history,
            _assistant_display_text(completed_segments, streaming_text),
        )

    elif event.kind == "thinking_snapshot":
        if SHOW_THINKING:
            thinking = event.text

    elif event.kind == "thinking_delta":
        if SHOW_THINKING:
            thinking += event.text

    elif event.kind == "status":
        activity = _append_activity(activity, event.text)

    elif event.kind == "turn_end":
        activity = _append_activity(activity, event.text)

    elif event.kind == "tool_start":
        if streaming_text.strip():
            completed_segments.append(streaming_text.strip())
            streaming_text = ""
        label = event.tool_name or "tool"
        tool_line = format_tool_chat_line(label, event.tool_args)
        detail = event.text or tool_line or label
        completed_segments, streaming_text = _append_chat_segment(
            completed_segments, streaming_text, tool_line
        )
        _set_last_assistant_content(
            history,
            _assistant_display_text(completed_segments, streaming_text),
        )
        activity = _append_activity(activity, f"**Tool start:** `{label}` — {detail}")
        tool_heading = f"### {label}\n{detail}\n\n```\n"
        tool_output = ""

    elif event.kind in {"tool_update", "tool_end"} and SHOW_TOOL_OUTPUT:
        if event.tool_output is not None:
            tool_output = _truncate(event.tool_output)
        if event.kind == "tool_end":
            status = "failed" if event.is_error else "completed"
            activity = _append_activity(
                activity,
                f"**Tool {status}:** `{event.tool_name or 'tool'}`",
            )

    elif event.kind == "error":
        activity = _append_activity(activity, f"**Error:** {event.text}")
        error_text = _assistant_display_text(completed_segments, streaming_text)
        if error_text.strip():
            error_text += f"\n\n**Error:** {event.text}"
        else:
            error_text = f"**Error:** {event.text}"
        _set_last_assistant_content(history, error_text)

    elif event.kind == "queue_update":
        steering = event.meta.get("steering") or []
        follow_up = event.meta.get("follow_up") or []
        added_notice = False
        for message in steering:
            text = str(message)
            if _history_has_user_notice(history, label="Steer", message=text):
                continue
            history = _queue_user_notice(history, label="Steer", message=text)
            added_notice = True
        for message in follow_up:
            text = str(message)
            if _history_has_user_notice(history, label="Follow-up", message=text):
                continue
            history = _queue_user_notice(history, label="Follow-up", message=text)
            added_notice = True
        if added_notice:
            completed_segments = []
            streaming_text = ""
        for line in _format_queue_update_activity(steering, follow_up):
            activity = _append_activity(activity, line)

    elif event.kind == "done":
        if streaming_text.strip():
            completed_segments.append(streaming_text)
            streaming_text = ""
        aborted = event.text.strip().lower().startswith("agent aborted")
        activity = _append_activity(activity, event.text)
        if append_finish_notice:
            history, completed_segments, streaming_text = _append_agent_finish_notice(
                history,
                completed_segments,
                streaming_text,
                aborted=aborted,
            )

    return (
        history,
        activity,
        thinking,
        tool_output,
        tool_heading,
        completed_segments,
        streaming_text,
    )


def _format_tool_panel(heading: str, body: str) -> str:
    if not heading and not body:
        return ""
    if heading.endswith("```\n") and body:
        return f"{heading}{body}\n```"
    if heading and not body:
        return heading.rstrip("`") + "…`\n```" if heading.endswith("```\n") else heading
    return heading + body


def _active_pi_provider(client: AgentRuntime | None) -> str:
    """Resolved Pi provider from the running RPC client, else configured default."""
    if client is not None and client.running:
        try:
            state = client.get_state()
            model = state.get("model") or {}
            provider = str(model.get("provider") or state.get("provider") or "")
            if provider:
                return normalize_provider(provider)
        except PiRpcError:
            pass
    return normalize_provider(get_default_provider())


def _pi_agent_model_label(client: AgentRuntime | None) -> str:
    """Active Pi orchestration model, or configured defaults before Apply backend."""
    if client is not None and client.running:
        try:
            state = client.get_state()
            model = state.get("model") or {}
            provider = str(model.get("provider") or state.get("provider") or "")
            model_label = str(model.get("id") or model.get("name") or "")
            if provider and model_label:
                return f"{provider_label(provider)} / {model_label}"
            return model_label or provider or "—"
        except PiRpcError:
            pass
    provider = normalize_provider(get_default_provider())
    model = resolved_default_model(provider)
    return f"{provider_label(provider)} / {model} (default until backend applied)"


def _agent_status_markdown(client: AgentRuntime | None = None) -> str:
    """Redaction backend URL, agent model, and credentials — shown at top of the UI."""
    from redaction_prompt import doc_redaction_gradio_url

    orchestrator = normalize_orchestrator()
    agent_label = orchestrator_label(orchestrator)
    lines = [
        f"**Redaction backend:** `{doc_redaction_gradio_url()}`",
        f"**Orchestrator:** `{agent_label}`",
    ]
    if orchestrator == "agentcore-harness":
        harness_arn = (
            os.environ.get("AGENTCORE_HARNESS_ARN") or ""
        ).strip() or "(not set)"
        lines.append(f"**Harness ARN:** `{harness_arn}`")
    elif orchestrator == "agentcore":
        runtime_url = (
            os.environ.get("AGENTCORE_RUNTIME_URL") or ""
        ).strip() or "(not set)"
        lines.append(f"**AgentCore runtime:** `{runtime_url}`")
    else:
        lines.append(f"**Agent model:** `{_pi_agent_model_label(client)}`")
    if (
        orchestrator not in {"agentcore", "agentcore-harness"}
        and not is_hf_space_profile()
        and normalize_provider(get_default_provider()) == PROVIDER_LLAMA
    ):
        lines.append(f"**LLM endpoint:** `{LLAMA_BASE_URL}`")
    if client is None or not client.running:
        lines.insert(0, "**Status:** Ready")
        lines.append("")
        lines.append(
            "_Set `DOC_REDACTION_GRADIO_URL` in `config/pi_agent.env` if the doc_redaction "
            "app is not at the URL above. Apply **Agent backend** to start the agent._"
        )
    else:
        lines.insert(0, f"**Status:** {agent_label} connected")
    lines.append("")
    lines.append(credential_status_markdown(provider=_active_pi_provider(client)))
    return "  \n".join(lines)


def _pi_agent_is_streaming(client: AgentRuntime | None) -> bool:
    """True while Pi RPC reports an active agent turn (authoritative vs Gradio state)."""
    rpc = _coerce_client(client)
    if rpc is None or not rpc.running:
        return False
    try:
        state = rpc.get_state()
    except PiRpcError:
        return False
    return bool(state.get("isStreaming"))


_PI_IDLE_POLL_INTERVAL_S = 0.25
_PI_IDLE_MAX_WAIT_S = float(os.environ.get("PI_IDLE_MAX_WAIT_S", "5"))


def _pi_wait_until_idle(
    client: AgentRuntime | None,
    *,
    max_wait_s: float | None = None,
) -> bool:
    """
    Wait briefly for Pi ``isStreaming`` to clear after a run ends.

    Skips while :attr:`AgentRuntime.prompt_stream_active` — the active
    ``prompt_events`` consumer owns stdout and ``get_state`` would steal lines.
    """
    rpc = _coerce_client(client)
    if rpc is None or not rpc.running:
        return True
    if rpc.prompt_stream_active:
        return False
    deadline = time.time() + (
        max_wait_s if max_wait_s is not None else _PI_IDLE_MAX_WAIT_S
    )
    while time.time() < deadline:
        try:
            if not _pi_agent_is_streaming(rpc):
                return True
        except PiRpcError:
            return True
        time.sleep(_PI_IDLE_POLL_INTERVAL_S)
    try:
        return not _pi_agent_is_streaming(rpc)
    except PiRpcError:
        return True


def _refresh_pi_client_model(client: AgentRuntime) -> None:
    """Re-apply provider/model on the live RPC client (no session reset)."""
    provider = normalize_provider(get_default_provider())
    model = resolved_default_model(provider)
    try:
        client.set_model(provider, model)
    except PiRpcError:
        pass


def _build_pi_prompt_message(
    session_hash: str,
    message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    is_followup: bool = False,
) -> str:
    """Prefix a user message the same way as **Start redaction task**."""
    from pi_workspace_skills import workspace_boundary_prefix

    prefix = workspace_boundary_prefix(session_hash) + workspace_context_prefix(
        session_hash
    )
    if is_followup and normalize_orchestrator() == "agentcore":
        from agentcore_workspace_bridge import build_agentcore_followup_context

        prefix += build_agentcore_followup_context(session_hash, history)
    return prefix + message.strip()


def _should_queue_agent_message(
    client: AgentRuntime | None,
    *,
    message: str,
) -> bool:
    """
    Route Send to steer only while this UI owns an active prompt stream.

    Uses :attr:`AgentRuntime.prompt_stream_active` (authoritative) plus Pi
    ``isStreaming``. After the agent is finished, Send goes through the normal
    chat path instead of queuing a follow-up event. Gradio ``agent_running`` and
    stale ``isStreaming`` alone are not reliable after llama.cpp runs.
    """
    if not (message or "").strip():
        return False
    rpc = _coerce_client(client)
    if rpc is None or not rpc.prompt_stream_active:
        return False
    return _pi_agent_is_streaming(client)


def _session_summary(client: AgentRuntime) -> str:
    try:
        state = client.get_state()
    except PiRpcError as exc:
        return f"{_agent_status_markdown(client)}  \n\n_Could not read session state: {exc}_"
    session_file = state.get("sessionFile") or "—"
    streaming = state.get("isStreaming")
    compacting = state.get("isCompacting")
    return (
        f"{_agent_status_markdown(client)}  \n\n"
        f"**Streaming:** `{streaming}` · **Compacting:** `{compacting}`  \n"
        f"**Session log:** `{session_file}`"
    )


def _backend_model_choices_update(provider: str):
    normalized = normalize_provider(provider)
    models = models_for_provider(normalized)
    return gr.update(choices=models, value=default_model_for_provider(normalized))


def apply_backend(
    provider: str,
    model_id: str,
    gemini_api_key: str,
    hf_token: str,
    aws_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
    client: AgentRuntime | None,
    session_hash: str,
):
    normalized = normalize_provider(provider)
    model = normalize_backend_model(normalized, model_id)

    apply_session_credentials(
        gemini_api_key=gemini_api_key or None,
        hf_token=hf_token or None,
        aws_region=aws_region or None,
        aws_access_key_id=aws_access_key_id or None,
        aws_secret_access_key=aws_secret_access_key or None,
        aws_session_token=aws_session_token or None,
    )
    if hf_token and hf_token.strip():
        os.environ["_HF_TOKEN_FROM_UI"] = "1"
    write_runtime_config(default_provider=normalized, default_model=model)

    existing = _coerce_client(client)
    if existing is not None:
        existing.close()

    key_error = _gemini_key_error()
    if key_error:
        return (
            None,
            key_error,
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )

    rpc = create_agent_runtime(session_hash or None)
    try:
        rpc.start()
        rpc.set_model(normalized, model)
        rpc.new_session()
        summary = (
            f"**Backend applied:** `{provider_label(normalized)}` / `{model}`  \n\n"
            f"{_session_summary(rpc)}"
        )
    except (PiRpcError, AgentRuntimeError, FileNotFoundError, OSError) as exc:
        rpc.close()
        rpc = None
        summary = (
            f"**Backend error:** {exc}  \n\n"
            f"{credential_status_markdown(provider=normalized)}"
        )

    return (
        rpc,
        summary,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )


def _reset_pi_rpc_client(
    client: AgentRuntime | None,
    session_hash: str,
) -> AgentRuntime | None:
    """
    Stop and recreate the Pi RPC subprocess for a clean orchestration context.

    Used on page reload and before each **Start redaction task**. Workspace files on
    disk are unchanged; only Pi in-memory session history is cleared.
    """
    rpc = _coerce_client(client)
    if rpc is None:
        return None
    _pi_wait_until_idle(rpc, max_wait_s=2.0)
    try:
        if rpc.running:
            if _pi_agent_is_streaming(rpc):
                rpc.abort()
                _pi_wait_until_idle(rpc, max_wait_s=2.0)
            return _restart_pi_rpc_client(session_hash, prior=rpc)
    except (PiRpcError, OSError, ValueError):
        pass
    try:
        rpc.close()
    except (PiRpcError, OSError, ValueError):
        pass
    return None


def _reset_pi_on_page_load(
    client: AgentRuntime | None,
    session_hash: str,
) -> AgentRuntime | None:
    """Alias for :func:`_reset_pi_rpc_client` (page-load handler)."""
    return _reset_pi_rpc_client(client, session_hash)


def _fresh_task_chat_outputs(
    client: AgentRuntime | None,
    session_hash: str,
    *,
    activity_line: str,
    session_note: str,
) -> tuple[Any, ...]:
    """Clear chat UI and return ``chat_outputs`` values after a session reset."""
    rpc = _coerce_client(client)
    if rpc is not None and rpc.running:
        session_md = f"{_session_summary(rpc)}  \n\n{session_note}"
        return _chat_yield(
            [],
            rpc,
            [activity_line],
            "",
            "",
            "",
            msg="",
            session_info=session_md,
            session_hash=session_hash,
            refresh_final_files=True,
            refresh_pdf_preview=True,
        )
    return (
        [],
        None,
        "",
        "_No activity yet._",
        "",
        "",
        _startup_session_info(),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
        collect_final_output_files(session_hash),
        gr.skip(),
        latest_redacted_pdf_path(session_hash) or gr.skip(),
        AGENT_FINISH_SIGNAL_NONE,
        False,
    )


def _initial_chat_outputs_on_page_load(
    client: AgentRuntime | None,
    session_hash: str,
) -> tuple[Any, ...]:
    """Clear chat UI and return ``chat_outputs`` values after a page load."""
    return _fresh_task_chat_outputs(
        client,
        session_hash,
        activity_line="Page loaded — agent session reset.",
        session_note=(
            "_Page reload — agent process and chat history reset "
            "(workspace files kept)._"
        ),
    )


def _init_session_ui(
    client: AgentRuntime | None,
    request: gr.Request,
) -> tuple[Any, ...]:
    session_hash, explorer, status, s3_prefix = init_session_workspace(request)
    log_platform_access(session_hash, HOST_NAME)
    fresh_client = _reset_pi_on_page_load(client, session_hash)
    return (
        session_hash,
        explorer,
        status,
        s3_prefix,
        *_initial_chat_outputs_on_page_load(fresh_client, session_hash),
    )


def _chat_yield(
    history: list[dict[str, Any]],
    client: AgentRuntime,
    activity: list[str],
    thinking: str,
    tool_heading: str,
    tool_output: str,
    *,
    msg: str | None = None,
    send_enabled: bool = True,
    abort_enabled: bool = False,
    redact_enabled: bool = True,
    agent_running: bool = False,
    session_info: str | None = None,
    session_hash: str = "",
    refresh_final_files: bool = False,
    refresh_pdf_preview: bool = False,
    agent_finish_signal: str = AGENT_FINISH_SIGNAL_NONE,
):
    final_files: list[str] | None | dict[str, Any]
    session_log: str | None | dict[str, Any]
    if refresh_final_files:
        final_files = collect_final_output_files(session_hash)
        session_log = collect_session_log_download(client)
    else:
        # Gradio 6 File components reject ``gr.update()`` as a stored value on replay.
        final_files = gr.skip()
        session_log = gr.skip()

    if refresh_pdf_preview or refresh_final_files:
        path = preview_pdf_path_for_gradio(session_hash)
        # Avoid pushing ``None`` into the PDF component (clears/breaks the viewer).
        pdf_preview = path if path else gr.skip()
    else:
        pdf_preview = gr.skip()

    msg_out: str | dict[str, Any] = gr.update() if msg is None else msg

    return (
        _clone_history(history),
        client,
        msg_out,
        _format_activity(activity),
        _format_tool_panel(tool_heading, tool_output),
        _truncate_thinking(thinking),
        session_info if session_info is not None else _session_summary(client),
        gr.update(interactive=send_enabled),
        gr.update(interactive=abort_enabled),
        gr.update(interactive=redact_enabled),
        final_files,
        session_log,
        pdf_preview,
        agent_finish_signal,
        agent_running,
    )


def _steer_yield(
    history: list[dict[str, Any]] | None = None,
    *,
    msg: str | None = "",
):
    """Single-yield passthrough while a steer/follow-up RPC is sent during a live run."""
    history_out: list[dict[str, Any]] | dict[str, Any] = (
        _clone_history(history) if history is not None else gr.update()
    )
    msg_out: str | dict[str, Any] = gr.update() if msg is None else msg
    return (
        history_out,
        gr.update(),
        msg_out,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(),
        gr.update(),
    )


def _steer_agent_message_sync(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    *,
    session_hash: str,
) -> tuple[Any, ...]:
    """Steer agentduring an active run without starting a new prompt stream."""
    if not message or not message.strip():
        return _steer_yield()

    rpc = _coerce_client(client)
    if rpc is None or not rpc.running:
        return _steer_yield()

    try:
        rpc.steer(message.strip())
        if os.environ.get("PI_RPC_DEBUG"):
            preview = (message or "").strip()[:200]
            _logger.debug(
                "_steer_agent_message_sync: queued steer for session_hash=%s preview=%s",
                session_hash,
                preview,
            )
            sys.stderr.write(
                f"DEBUG-FOLLOWUP queued_steer: session_hash={session_hash} preview={preview!r}\n"
            )
    except (PiRpcError, OSError, ValueError) as exc:
        gr.Warning(f"Could not queue message for Pi: {exc}")
        if os.environ.get("PI_RPC_DEBUG"):
            _logger.exception(
                "_steer_agent_message_sync: could not queue steer for session_hash=%s",
                session_hash,
            )
            sys.stderr.write(
                f"DEBUG-FOLLOWUP queued_steer_failed: session_hash={session_hash} error={exc!r}\n"
            )
        return _steer_yield()
    rpc.stage_ui_chat_notice("Steer", message.strip())
    history, _, _ = _integrate_pending_chat_notices(
        list(history or []),
        rpc,
        [],
        "",
    )
    return _steer_yield(history, msg="")


def _queue_agent_message(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    *,
    session_hash: str,
):
    """Steer agent during an active run without starting a new prompt stream."""
    yield _steer_agent_message_sync(
        message,
        history,
        client,
        session_hash=session_hash,
    )


def _followup_queued_skip_yield() -> tuple[Any, ...]:
    """Passthrough while the queued follow-up step has nothing to run."""
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(),
        gr.update(),
    )


def _log_followup_diagnostics(
    message: str,
    client: AgentRuntime | None,
    *,
    session_hash: str,
) -> None:
    if os.environ.get("PI_RPC_DEBUG"):
        sys.stderr.write(
            f"DEBUG-FOLLOWUP ENTER: session_hash={session_hash} message_len={len(message or '')}\n"
        )
    try:
        rpc = _coerce_client(client)
        client_running = bool(rpc and getattr(rpc, "running", False))
        prompt_stream_active = bool(rpc and getattr(rpc, "prompt_stream_active", False))
        try:
            pi_is_streaming = _pi_agent_is_streaming(rpc)
        except Exception:
            pi_is_streaming = False
        msg_present = bool(message and message.strip())
        if os.environ.get("PI_RPC_DEBUG"):
            preview = (message or "").strip()[:200]
            _logger.debug(
                "submit_followup_message diagnostics: client_running=%s prompt_stream_active=%s pi_is_streaming=%s message_present=%s",
                client_running,
                prompt_stream_active,
                pi_is_streaming,
                msg_present,
            )
            sys.stderr.write(
                f"DEBUG-FOLLOWUP start: client_running={client_running} prompt_stream_active={prompt_stream_active} pi_is_streaming={pi_is_streaming} message_present={msg_present} preview={preview!r}\n"
            )
    except Exception:
        _logger.exception("submit_followup_message diagnostic failure")


def _log_followup_branch(
    should_queue: bool,
    message: str,
    *,
    session_hash: str,
) -> None:
    if not os.environ.get("PI_RPC_DEBUG"):
        return
    try:
        preview = (message or "").strip()[:200]
    except Exception:
        preview = ""
    _logger.debug(
        "submit_followup_message branch: should_queue=%s session_hash=%s message_preview=%s",
        should_queue,
        session_hash,
        preview,
    )
    sys.stderr.write(
        f"DEBUG-FOLLOWUP branch: should_queue={should_queue} session_hash={session_hash} preview={preview!r}\n"
    )


def route_followup_message(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
):
    """
    Fast ``queue=False`` router for Send / msg.submit.

    Steers during a live prompt stream; otherwise clears the textbox and stashes
    the message for :func:`submit_followup_chat_queued` (queued generator).
    """
    del s3_output_folder, save_outputs_to_s3
    _log_followup_diagnostics(message, client, session_hash=session_hash)
    try:
        should_queue = _should_queue_agent_message(client, message=message)
    except Exception:
        should_queue = False
    _log_followup_branch(should_queue, message, session_hash=session_hash)

    if should_queue:
        steer_out = _steer_agent_message_sync(
            message,
            history,
            client,
            session_hash=session_hash,
        )
        return (*steer_out, "")

    if not message or not message.strip():
        return (*_steer_yield(), "")

    return (*_steer_yield(msg=""), message.strip())


def submit_followup_chat_queued(
    pending_message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
):
    """Queued generator for idle follow-ups (multi-yield ``_run_pi_chat``)."""
    if not (pending_message or "").strip():
        yield _followup_queued_skip_yield()
        return
    yield from _run_pi_chat(
        pending_message,
        history,
        client,
        session_hash=session_hash,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
    )


def _run_pi_chat(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    *,
    chat_user_message: str | None = None,
    session_hash: str = "",
    initial_session_info: str | None = None,
    s3_output_folder: str = "",
    save_outputs_to_s3: bool = False,
    document_name: str = "",
    base_file: str | None = None,
    ocr_method: str = "",
    pii_method: str = "",
    total_page_count: int = 0,
    vlm_model_name: str | None = None,
    redact_file: str | None = None,
):
    if not message or not message.strip():
        client = client if client and client.running else None
        hint_activity = [EMPTY_SEND_WITH_FILE_HINT] if redact_file else []
        if client:
            yield _chat_yield(
                history or [],
                client,
                hint_activity,
                "",
                "",
                "",
                session_hash=session_hash,
            )
        else:
            activity_text = (
                _format_activity(hint_activity)
                if hint_activity
                else "_No activity yet._"
            )
            yield (
                history or [],
                None,
                "",
                activity_text,
                "",
                "",
                "_Ready._",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                AGENT_FINISH_SIGNAL_NONE,
                False,
            )
        return

    history = list(history or [])
    client = _ensure_client(client, session_hash)
    _refresh_pi_client_model(client)
    llama_ready_err = _prepare_llama_before_orchestration_prompt()
    if llama_ready_err:
        history.append(
            {"role": "user", "content": chat_user_message or message.strip()}
        )
        history.append(
            {
                "role": "assistant",
                "content": f"**LLM:** {llama_ready_err}",
            }
        )
        yield _chat_yield(
            history,
            client,
            [f"**LLM:** {llama_ready_err}"],
            "",
            "",
            "",
            msg="" if chat_user_message is None else None,
            send_enabled=True,
            abort_enabled=False,
            redact_enabled=True,
            session_info=_agent_status_markdown(client),
            session_hash=session_hash,
            refresh_final_files=True,
        )
        return

    activity: list[str] = []
    thinking = ""
    tool_output = ""
    tool_heading = ""
    completed_segments: list[str] = []
    streaming_text = ""
    task_started_at = time.time()
    usage_baseline = resolve_session_token_usage(client)

    def _complete_pi_task() -> str | None:
        usage = usage_for_completed_turn(client, usage_baseline)
        return _after_pi_task(
            session_hash=session_hash,
            client=client,
            s3_output_folder=s3_output_folder,
            save_outputs_to_s3=save_outputs_to_s3,
            document_name=document_name,
            started_at=task_started_at,
            base_file=base_file,
            ocr_method=ocr_method,
            pii_method=pii_method,
            total_page_count=total_page_count,
            vlm_model_name=vlm_model_name,
            llm_input_tokens=usage.llm_input_tokens,
            llm_output_tokens=usage.llm_output_tokens,
        )

    def _activity_with_s3_warning(act: list[str]) -> list[str]:
        warning = _complete_pi_task()
        if warning:
            return _append_activity(act, f"**S3 upload:** {warning}")
        return act

    history.append({"role": "user", "content": chat_user_message or message.strip()})
    history.append({"role": "assistant", "content": ""})
    prompt_activity = (
        "Prompt sent." if chat_user_message is not None else "Follow-up prompt sent."
    )
    if _uses_local_llama_orchestrator():
        prompt_activity += " Waiting for orchestration model."
    activity = _append_activity(activity, prompt_activity)
    if initial_session_info:
        activity = _append_activity(
            activity,
            f"Using workspace `{session_workspace_dir(session_hash).as_posix()}/`.",
        )
    session_info = _session_summary(client)
    if initial_session_info:
        session_info = f"{initial_session_info}\n\n{session_info}"

    yield _chat_yield(
        history,
        client,
        activity,
        thinking,
        tool_heading,
        tool_output,
        msg="" if chat_user_message is None else None,
        send_enabled=True,
        abort_enabled=True,
        redact_enabled=False,
        agent_running=True,
        session_info=session_info,
        session_hash=session_hash,
        refresh_pdf_preview=True,
    )

    if os.environ.get("PI_RPC_DEBUG"):
        sys.stderr.write(
            f"DEBUG-FOLLOWUP resume: session_hash={session_hash} message_len={len(message or '')}\n"
        )
    prompt_to_send = _build_pi_prompt_message(
        session_hash,
        message,
        history=history,
        is_followup=not bool(document_name.strip()),
    )
    if os.environ.get("PI_RPC_DEBUG"):
        try:
            preview = (message or "").strip()[:400]
        except Exception:
            preview = ""
        _logger.debug(
            "_run_pi_chat: sending prompt for session_hash=%s preview=%s",
            session_hash,
            preview,
        )
        sys.stderr.write(
            f"DEBUG-FOLLOWUP send_prompt: session_hash={session_hash} preview={preview!r}\n"
        )

    _stage_agentcore_workspace_upload(client, session_hash, document_name)
    _stage_harness_input(client, session_hash, document_name)

    event_queue: queue.Queue[AgentStreamEvent | None] = queue.Queue()
    start_agent_prompt_event_worker(client, event_queue, prompt_to_send)

    quota_failures = 0
    finish_aborted = False
    done_event_received = False

    try:
        while True:
            turn_error: str | None = None
            try:
                while True:
                    event = event_queue.get()
                    if event is None:
                        break
                    history, completed_segments, streaming_text = (
                        _integrate_pending_chat_notices(
                            history,
                            client,
                            completed_segments,
                            streaming_text,
                        )
                    )
                    is_done_event = event.kind == "done"
                    if is_done_event:
                        finish_aborted = (
                            event.text.strip().lower().startswith("agent aborted")
                        )
                        done_event_received = True
                    (
                        history,
                        activity,
                        thinking,
                        tool_output,
                        tool_heading,
                        completed_segments,
                        streaming_text,
                    ) = _apply_event(
                        event,
                        history=history,
                        activity=activity,
                        thinking=thinking,
                        tool_output=tool_output,
                        tool_heading=tool_heading,
                        completed_segments=completed_segments,
                        streaming_text=streaming_text,
                        append_finish_notice=not is_done_event,
                    )
                    yield _chat_yield(
                        history,
                        client,
                        activity,
                        thinking,
                        tool_heading,
                        tool_output,
                        send_enabled=True,
                        abort_enabled=not is_done_event,
                        redact_enabled=False,
                        agent_running=True,
                        session_info=session_info,
                        session_hash=session_hash,
                        refresh_pdf_preview=event.kind
                        in ("turn_end", "workspace_sync"),
                        refresh_final_files=event.kind == "done",
                    )
                turn_error = last_assistant_turn_error(client.get_messages())
            except PiRpcError as exc:
                if not is_rate_limit_error(str(exc)):
                    raise
                turn_error = str(exc)

            if turn_error and not is_rate_limit_error(turn_error):
                err_text = _format_llama_turn_error(turn_error)
                if not err_text.startswith("**LLM:"):
                    err_text = f"**LLM error:** {turn_error}"
                _set_last_assistant_content(history, err_text)
                activity = _append_activity(activity, err_text)
                history, completed_segments, streaming_text = (
                    _append_agent_finish_notice(
                        history,
                        completed_segments,
                        streaming_text,
                        error=True,
                    )
                )
                activity = _activity_with_s3_warning(activity)
                finish_signal = _notify_agent_finished(error=True)
                yield _chat_yield(
                    history,
                    client,
                    activity,
                    thinking,
                    tool_heading,
                    tool_output,
                    send_enabled=True,
                    abort_enabled=False,
                    redact_enabled=True,
                    session_info=_session_summary(client),
                    session_hash=session_hash,
                    refresh_final_files=True,
                    agent_finish_signal=finish_signal,
                )
                return

            if turn_error and is_rate_limit_error(turn_error):
                quota_failures += 1
                if quota_failures >= QUOTA_RETRY_ATTEMPTS:
                    err_summary = turn_error[:500].replace("\n", " ")
                    _set_last_assistant_content(
                        history,
                        (
                            f"**API rate limit hit:** stopped after "
                            f"{QUOTA_RETRY_ATTEMPTS} consecutive attempts.\n\n"
                            f"{err_summary}"
                        ),
                    )
                    activity = _append_activity(
                        activity,
                        f"**Quota retries exhausted** ({QUOTA_RETRY_ATTEMPTS} attempts).",
                    )
                    history, completed_segments, streaming_text = (
                        _append_agent_finish_notice(
                            history,
                            completed_segments,
                            streaming_text,
                            error=True,
                        )
                    )
                    activity = _activity_with_s3_warning(activity)
                    finish_signal = _notify_agent_finished(error=True)
                    yield _chat_yield(
                        history,
                        client,
                        activity,
                        thinking,
                        tool_heading,
                        tool_output,
                        send_enabled=True,
                        abort_enabled=False,
                        redact_enabled=True,
                        session_info=_session_summary(client),
                        session_hash=session_hash,
                        refresh_final_files=True,
                        agent_finish_signal=finish_signal,
                    )
                    return

                wait_message = (
                    f"API rate limit hit — waiting {QUOTA_RETRY_DELAY_S}s before "
                    f"retry {quota_failures}/{QUOTA_RETRY_ATTEMPTS}…"
                )
                activity = _append_activity(activity, wait_message)
                history, completed_segments, streaming_text = (
                    _append_rate_limit_wait_notice(
                        history,
                        completed_segments,
                        streaming_text,
                        wait_message,
                    )
                )
                yield _chat_yield(
                    history,
                    client,
                    activity,
                    thinking,
                    tool_heading,
                    tool_output,
                    send_enabled=True,
                    abort_enabled=True,
                    redact_enabled=False,
                    agent_running=True,
                    session_info=session_info,
                    session_hash=session_hash,
                )
                time.sleep(QUOTA_RETRY_DELAY_S)
                prompt_to_send = QUOTA_CONTINUE_PROMPT
                history.append({"role": "assistant", "content": ""})
                completed_segments = []
                streaming_text = ""
                done_event_received = False
                finish_aborted = False
                event_queue = queue.Queue()
                start_agent_prompt_event_worker(client, event_queue, prompt_to_send)
                continue

            break
    except PiRpcError as exc:
        _set_last_assistant_content(history, f"**Error:** {exc}")
        activity = _append_activity(activity, f"**Error:** {exc}")
        history, completed_segments, streaming_text = _append_agent_finish_notice(
            history,
            completed_segments,
            streaming_text,
            error=True,
        )
        activity = _activity_with_s3_warning(activity)
        finish_signal = _notify_agent_finished(error=True)
        yield _chat_yield(
            history,
            client,
            activity,
            thinking,
            tool_heading,
            tool_output,
            send_enabled=True,
            abort_enabled=False,
            redact_enabled=True,
            session_info=_session_summary(client),
            session_hash=session_hash,
            refresh_final_files=True,
            agent_finish_signal=finish_signal,
        )
        return
    except Exception:
        if getattr(client, "abort_requested", False):
            activity = _append_activity(activity, "**Aborted.**")
            history, completed_segments, streaming_text = _append_agent_finish_notice(
                history,
                completed_segments,
                streaming_text,
                aborted=True,
            )
            activity = _activity_with_s3_warning(activity)
            finish_signal = _notify_agent_finished(aborted=True)
            yield _chat_yield(
                history,
                client,
                activity,
                thinking,
                tool_heading,
                tool_output,
                send_enabled=True,
                abort_enabled=False,
                redact_enabled=True,
                session_info=_session_summary(client),
                session_hash=session_hash,
                refresh_final_files=True,
                agent_finish_signal=finish_signal,
            )
            return
        raise

    if done_event_received:
        history, completed_segments, streaming_text = _append_agent_finish_notice(
            history,
            completed_segments,
            streaming_text,
            aborted=finish_aborted,
        )

    _finalize_assistant_chat(
        client,
        history,
        completed_segments=completed_segments,
        streaming_text=streaming_text,
        activity=activity,
    )

    finish_signal = _notify_agent_finished(aborted=finish_aborted)
    yield _chat_yield(
        history,
        client,
        activity,
        thinking,
        tool_heading,
        tool_output,
        send_enabled=True,
        abort_enabled=False,
        redact_enabled=True,
        agent_running=False,
        session_info=_session_summary(client),
        session_hash=session_hash,
        refresh_final_files=True,
        refresh_pdf_preview=True,
        agent_finish_signal=finish_signal,
    )
    _schedule_post_pi_task(
        session_hash=session_hash,
        client=client,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
        document_name=document_name,
        started_at=task_started_at,
        base_file=base_file,
        ocr_method=ocr_method,
        pii_method=pii_method,
        total_page_count=total_page_count,
        vlm_model_name=vlm_model_name,
        usage_baseline=usage_baseline,
    )


def submit_followup_message(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
):
    """
    Send a user message to agent (programmatic / legacy single-handler API).

    The Gradio UI uses :func:`route_followup_message` (``queue=False``) plus
    :func:`submit_followup_chat_queued` (queued generator) instead.
    """
    _log_followup_diagnostics(message, client, session_hash=session_hash)
    try:
        should_queue = _should_queue_agent_message(client, message=message)
    except Exception:
        should_queue = False
    _log_followup_branch(should_queue, message, session_hash=session_hash)
    if should_queue:
        yield from _queue_agent_message(
            message,
            history,
            client,
            session_hash=session_hash,
        )
        return
    yield from _run_pi_chat(
        message,
        history,
        client,
        session_hash=session_hash,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
    )


def chat_respond(
    message: str,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    agent_running: bool,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    redact_file: str | None,
):
    del agent_running, redact_file
    yield from submit_followup_message(
        message,
        history,
        client,
        session_hash,
        s3_output_folder,
        save_outputs_to_s3,
    )


def _redaction_page_count(upload_file: str | None, page_range: str) -> int:
    if not upload_file or not str(upload_file).lower().endswith(".pdf"):
        return 0
    try:
        total = pdf_page_count(upload_file)
        return pages_to_process_count(page_range or "all", total)
    except (ValueError, OSError):
        return 0


def _restart_pi_rpc_client(
    session_hash: str,
    *,
    prior: AgentRuntime | None = None,
) -> AgentRuntime:
    """Stop and recreate the Pi RPC subprocess with the configured provider/model."""
    if prior is not None:
        try:
            prior.close()
        except (PiRpcError, OSError, ValueError):
            pass
    provider = normalize_provider(get_default_provider())
    model = resolved_default_model(provider)
    rpc = create_agent_runtime(session_hash or None)
    rpc.start()
    rpc.set_model(provider, model)
    rpc.new_session()
    return rpc


def _stage_agentcore_workspace_upload(
    client: AgentRuntime,
    session_hash: str,
    document_name: str,
) -> None:
    """Upload session workspace files into the remote AgentCore workspace."""
    if normalize_orchestrator() != "agentcore":
        return
    from agentcore_runtime import AgentCoreAgentRuntime
    from agentcore_workspace_bridge import (
        collect_session_files_for_agentcore_upload,
        discover_session_document_name,
    )

    if not isinstance(client, AgentCoreAgentRuntime):
        return

    doc = (document_name or "").strip() or (
        discover_session_document_name(session_hash) or ""
    )
    staged = collect_session_files_for_agentcore_upload(
        session_hash,
        document_name=doc or None,
    )
    max_bytes = int(os.environ.get("AGENTCORE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

    repo_root = Path(__file__).resolve().parents[2]
    skill_names = (
        "doc-redaction-app",
        "doc-redaction-modifications",
    )
    for skill_name in skill_names:
        skill_path = repo_root / "skills" / skill_name / "SKILL.md"
        if not skill_path.is_file():
            continue
        payload = skill_path.read_bytes()
        if len(payload) > max_bytes:
            continue
        staged.append(
            {
                "relative_path": f".pi/skills/{skill_name}/SKILL.md",
                "content_base64": base64.b64encode(payload).decode("ascii"),
            }
        )

    if not staged and doc:
        root = session_workspace_dir(session_hash)
        src = root / doc
        if src.is_file():
            size = src.stat().st_size
            if size > max_bytes:
                client.stage_ui_chat_notice(
                    "AgentCore",
                    f"Could not sync `{doc}` ({size:,} bytes exceeds {max_bytes:,} byte limit). "
                    "Set AGENTCORE_MAX_UPLOAD_BYTES or use a smaller file.",
                )
            else:
                staged.append(
                    {
                        "relative_path": doc,
                        "content_base64": base64.b64encode(src.read_bytes()).decode(
                            "ascii"
                        ),
                    }
                )

    if staged:
        client.stage_workspace_files(staged)
        if not document_name and doc:
            client.stage_ui_chat_notice(
                "AgentCore",
                f"Synced {len(staged)} workspace file(s) for follow-up (document `{doc}`).",
            )
    client.set_sync_workspace_files(True)


def _stage_harness_input(
    client: AgentRuntime,
    session_hash: str,
    document_name: str,
) -> None:
    """Upload task PDF to S3 and prepend fetch instructions for AgentCore Harness."""
    orch = normalize_orchestrator()
    if orch not in {"agentcore-harness", "harness"} or not document_name:
        return
    from agentcore_harness_runtime import AgentCoreHarnessRuntime
    from harness_input_bridge import build_harness_document_prompt_prefix

    if not isinstance(client, AgentCoreHarnessRuntime):
        return
    prefix = build_harness_document_prompt_prefix(session_hash, document_name)
    if prefix:
        client.stage_prompt_prefix(prefix)


def _ensure_pi_client_for_redaction(
    client: AgentRuntime | None,
    session_hash: str,
) -> AgentRuntime:
    """
    Apply the same Pi reset as page load, then ensure a running client for the task.
    """
    client = _reset_pi_rpc_client(client, session_hash)
    if isinstance(client, AgentRuntime) and client.running:
        return client
    return _ensure_client(client, session_hash)


def prepare_redaction_session_ui(
    session_hash: str,
    request: gr.Request,
) -> tuple[str, str]:
    """Create session workspace folder before redaction runs (updates UI immediately)."""
    effective, _workspace, status = prepare_session_workspace(session_hash, request)
    return effective, status


def submit_redaction_task(
    upload_file: str | None,
    user_instructions: str,
    page_range: str,
    ocr_method: str,
    pii_method: str,
    encourage_vlm_faces: bool,
    encourage_vlm_signatures: bool,
    history: list[dict[str, Any]] | None,
    client: AgentRuntime | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    request: gr.Request,
):
    session_hash, _workspace_path, workspace_status = prepare_session_workspace(
        session_hash, request
    )
    settings = (
        RedactionTaskSettings.hf_space_defaults()
        if IS_HF_SPACE
        else RedactionTaskSettings.from_ui(
            ocr_method,
            pii_method,
            encourage_vlm_faces,
            encourage_vlm_signatures,
        )
    )
    try:
        _file_name, prompt, renamed_from = prepare_redaction_task(
            upload_file,
            user_instructions,
            page_range=page_range or "all",
            settings=settings,
            workspace_dir=_workspace_path,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        history = list(history or [])
        history.append(
            {"role": "user", "content": f"_Redaction task not started: {exc}_"}
        )
        client = (
            _ensure_client(client, session_hash)
            if client and client.running
            else client
        )
        yield (
            _clone_history(history),
            client,
            "",
            _format_activity([f"**Redaction task error:** {exc}"]),
            "",
            "",
            (
                _session_summary(client)
                if client and client.running
                else _agent_status_markdown(client)
            ),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            AGENT_FINISH_SIGNAL_NONE,
            False,
        )
        return

    page_count = _redaction_page_count(upload_file, page_range or "all")
    chat_summary = (
        f"**Redaction task:** `{_file_name}`  \n"
        f"**Page range:** `{page_range or 'all'}`  \n"
        f"**OCR / text extraction:** `{settings.ocr_method}`  \n"
        f"**PII model:** `{settings.pii_method}`  \n"
        f"**VLM faces guidance:** {'on' if settings.encourage_vlm_faces else 'off'}  \n"
        f"**VLM signature guidance:** {'on' if settings.encourage_vlm_signatures else 'off'}\n\n"
        f"{user_instructions.strip()}"
    )
    if renamed_from:
        chat_summary = (
            f"_Your uploaded file `{renamed_from}` was saved as `{_file_name}` for this "
            f"task because the original name contained characters that are unsafe for "
            f"file paths._\n\n{chat_summary}"
        )
    try:
        client = _ensure_pi_client_for_redaction(client, session_hash)
    except PiRpcError as exc:
        yield (
            [],
            client,
            "",
            _format_activity([f"**Redaction task error:** {exc}"]),
            "",
            "",
            _agent_status_markdown(client),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            AGENT_FINISH_SIGNAL_NONE,
            False,
        )
        return

    reset_outputs = _fresh_task_chat_outputs(
        client,
        session_hash,
        activity_line="Starting redaction task — agent session reset.",
        session_note=(
            f"{workspace_status}\n\n"
            "_agent process and chat history reset before this redaction task "
            "(workspace files kept)._"
        ),
    )
    yield reset_outputs

    session_info = (
        f"{workspace_status}\n\n"
        "_agent process and chat history reset before this redaction task "
        "(workspace files kept)._"
    )
    yield from _run_pi_chat(
        prompt,
        [],
        client,
        chat_user_message=chat_summary,
        session_hash=session_hash,
        initial_session_info=session_info,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
        document_name=_file_name,
        base_file=upload_file,
        ocr_method=settings.ocr_method,
        pii_method=settings.pii_method,
        total_page_count=page_count,
        vlm_model_name=os.environ.get("PI_VLM_MODEL"),
    )


def abort_agent(client: AgentRuntime | None):
    rpc = _coerce_client(client)
    if rpc is not None and rpc.running:
        try:
            rpc.abort()
        except (PiRpcError, OSError, ValueError):
            pass
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def new_chat(
    _history,
    client: AgentRuntime | None,
    session_hash: str,
):
    if client is not None:
        try:
            client.new_session()
        except PiRpcError:
            client.close()
            client = create_agent_runtime(session_hash or None)
            client.start()
    else:
        client = create_agent_runtime(session_hash or None)
        client.start()
    return _chat_yield(
        [],
        client,
        ["New session."],
        "",
        "",
        "",
        msg="",
        session_hash=session_hash,
        refresh_final_files=True,
        refresh_pdf_preview=True,
    )


def _startup_session_info() -> str:
    if IS_HF_SPACE:
        return (
            "**Hugging Face Space profile** — Gemini orchestration with remote Document Redaction App "
            "backend.  \n\n"
            "1. Paste your **Gemini API key** (and optional **HF token** for a private "
            "redaction Space).  \n"
            "2. Click **Apply backend**.  \n\n"
            f"{_agent_status_markdown(None)}"
        )
    return _agent_status_markdown(None)


def build_ui():
    from gradio_pdf_redaction import PDF

    hf_redaction_blurb = (
        "Upload a document and add bullet-point requirements. Redaction runs on a **remote** "
        "Redaction App Hugging Face Space.  \n"
        "When ready, use **Start redaction task** under the chat panel to the right."
        if IS_HF_SPACE
        else (
            "Upload a PDF (or other supported document). Add bullet-point instructions for redaction below. \n"
            "When ready, use **Start redaction task** under the chat panel to the right."
        )
    )
    backend_blurb = (
        "Gemini powers the agent on this Space. Paste your **Gemini API key** "
        "(session-only, not stored on disk). Optionally override the **HF token** used "
        "to reach the private redaction backend."
        if IS_HF_SPACE
        else (
            "Choose which LLM powers the agent (chat and redaction orchestration). "
            "Credentials from the UI apply **for this container session only**; "
            "defaults can be set via `config/pi_agent.env` or compose environment."
        )
    )
    hf_locked_settings_md = (
        f"**Locked defaults (HF Space):**  \n"
        f"- Text extraction: `{DEFAULT_OCR_METHOD}`  \n"
        f"- PII model: `{DEFAULT_PII_METHOD}`  \n"
        f"- Face/signature VLM: unavailable"
        if IS_HF_SPACE
        else ""
    )

    with gr.Blocks(
        title=PI_UI_TITLE,
        fill_height=True,
    ) as demo:
        gr.Markdown(PI_INTRO_TEXT)
        client_state = gr.State(None)
        session_hash_state = gr.State("")
        s3_output_folder_state = gr.State("")
        save_outputs_to_s3_state = gr.State(SAVE_OUTPUTS_TO_S3)

        with gr.Accordion("View session info", open=False):
            session_info = gr.Markdown(_startup_session_info())

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):

                with gr.Accordion("Redaction task", open=True):
                    gr.Markdown(hf_redaction_blurb)

                    pi_example_rows, pi_example_labels = example_rows()

                    redact_file = gr.File(
                        label="Document to redact",
                        file_types=[
                            ".pdf",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".docx",
                            ".csv",
                            ".xlsx",
                        ],
                        type="filepath",
                        render=False,
                    )
                    redact_instructions = gr.Textbox(
                        label="Redaction requirements",
                        placeholder=(
                            "- Redact all personal names\n"
                            "- Remove organisation addresses\n"
                            "- Keep publication titles visible"
                        ),
                        lines=8,
                        render=False,
                    )
                    page_range = gr.Textbox(
                        label="Page range",
                        value="all",
                        placeholder="all or e.g. 1-56",
                        render=False,
                    )
                    if IS_HF_SPACE:
                        ocr_method = gr.State(DEFAULT_OCR_METHOD)
                        pii_method = gr.State(DEFAULT_PII_METHOD)
                        encourage_vlm_faces = gr.State(False)
                        encourage_vlm_signatures = gr.State(False)
                        settings_accordion = None
                    else:
                        settings_accordion = gr.Accordion(
                            "Redaction settings (prompt defaults)",
                            open=False,
                            render=False,
                        )
                        with settings_accordion:
                            gr.Markdown(
                                "These values are injected into the task prompt under "
                                "**Technical constraints** — they suggest defaults to the agent for "
                                "`/doc_redact`, not hard-coded app settings."
                            )
                            ocr_method = gr.Dropdown(
                                label="Default text extraction method",
                                choices=list(OCR_METHOD_CHOICES),
                                value=DEFAULT_OCR_METHOD,
                                allow_custom_value=True,
                            )
                            pii_method = gr.Dropdown(
                                label="Default PII identification model",
                                choices=list(PII_METHOD_CHOICES),
                                value=DEFAULT_PII_METHOD,
                                allow_custom_value=True,
                            )
                            encourage_vlm_faces = gr.Checkbox(
                                label="Encourage CUSTOM_VLM_FACES when user asks to redact faces",
                                value=True,
                            )
                            encourage_vlm_signatures = gr.Checkbox(
                                label=(
                                    "Encourage CUSTOM_VLM_SIGNATURE when user asks "
                                    "to redact signatures"
                                ),
                                value=True,
                            )

                    if pi_example_rows:
                        gr.Markdown(
                            "### Try an example\n"
                            "Click a row to load the sample PDF and redaction instructions, "
                            "then **Start redaction task** under the chat panel to the right."
                        )
                        gr.Examples(
                            examples=pi_example_rows,
                            inputs=[
                                redact_file,
                                redact_instructions,
                                page_range,
                                ocr_method,
                                pii_method,
                                encourage_vlm_faces,
                                encourage_vlm_signatures,
                            ],
                            example_labels=pi_example_labels,
                            examples_per_page=2,
                            cache_examples=False,
                        )
                    else:
                        gr.Markdown(examples_status_markdown())

                    redact_file.render()
                    redact_instructions.render()
                    page_range.render()
                    if IS_HF_SPACE:
                        gr.Markdown(hf_locked_settings_md)
                    elif settings_accordion is not None:
                        settings_accordion.render()

                    with gr.Accordion("Agent backend/API keys", open=IS_HF_SPACE):
                        gr.Markdown(backend_blurb)
                        backend_provider = gr.Radio(
                            label="Provider",
                            choices=[
                                (provider_label(key), key) for key in provider_choices()
                            ],
                            value=get_default_provider(),
                        )
                        backend_model = gr.Dropdown(
                            label="Model",
                            choices=models_for_provider(get_default_provider()),
                            value=default_model_for_provider(get_default_provider()),
                            allow_custom_value=True,
                        )
                        gemini_api_key = gr.Textbox(
                            label=(
                                "Gemini API key (required on HF Space)"
                                if IS_HF_SPACE
                                else "Gemini API key (session override)"
                            ),
                            type="password",
                            placeholder=(
                                "Required — get a key from Google AI Studio"
                                if IS_HF_SPACE
                                else "Uses GEMINI_API_KEY / GOOGLE_API_KEY from env if empty"
                            ),
                        )
                        hf_token = gr.Textbox(
                            label="HF token for redaction space (optional, neededfor private spaces)",
                            type="password",
                            placeholder="Uses HF_TOKEN Space secret if empty",
                            visible=IS_HF_SPACE,
                        )
                        with gr.Accordion(
                            "AWS credentials (optional)",
                            open=False,
                            visible=not IS_HF_SPACE,
                        ):
                            aws_region = gr.Textbox(
                                label="AWS region (session override)",
                                placeholder="e.g. eu-west-2",
                                visible=not IS_HF_SPACE,
                            )
                            aws_access_key_id = gr.Textbox(
                                label="AWS access key ID (session override)",
                                type="password",
                                visible=not IS_HF_SPACE,
                            )
                            aws_secret_access_key = gr.Textbox(
                                label="AWS secret access key (session override)",
                                type="password",
                                visible=not IS_HF_SPACE,
                            )
                            aws_session_token = gr.Textbox(
                                label="AWS session token (optional)",
                                type="password",
                                visible=False,  # not IS_HF_SPACE,
                            )
                        apply_backend_btn = gr.Button(
                            "Apply backend",
                            variant="primary",
                        )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Task progress", height=480, resizable=True)
                with gr.Row():
                    start_redact_btn = gr.Button(
                        "Start redaction task",
                        variant="primary",
                    )
                    abort_btn = gr.Button("Abort", variant="stop", interactive=False)
                clear = gr.Button("New session")
                with gr.Accordion("Follow-up / steer chat (optional)", open=False):
                    gr.Markdown(
                        "While the agent is **running**, **Send** steers it after the "
                        "current step. After **Agent finished**, **Send** starts a new "
                        "normal chat turn in the same session."
                    )
                    msg = gr.Textbox(
                        label="Message",
                        placeholder=(
                            "e.g. Stop — only redact names on page 3, not addresses"
                        ),
                    )
                    send = gr.Button("Send", variant="secondary")

                with gr.Accordion("Preview latest redacted PDF", open=False):
                    pdf_preview = PDF(
                        label="Preview latest redacted PDF",
                        interactive=True,
                        height=480,
                        visible=True,
                    )

                with gr.Accordion("Activity log", open=False):
                    activity_log = gr.Markdown(
                        value="_No activity yet._", max_height=480, height=480
                    )
                    tool_panel = gr.Markdown(value="", max_height=480, height=480)
                    thinking_panel = gr.Textbox(
                        label="Thinking (stream)",
                        lines=12,
                        max_lines=50,
                        interactive=False,
                        visible=SHOW_THINKING,
                        elem_classes=["thinking-panel"],
                        autoscroll=True,
                    )

        with gr.Accordion("Workspace output files", open=True):
            workspace_session_info = gr.Markdown(
                "_Loading your session workspace…_",
            )
            gr.Markdown(
                "**Final outputs** will appear below. "
                "Downloads below are available in your session's `output_final_download/` folder."
                "Use the file explorer below to browse or download other workspace files."
            )
            workspace_output_download = gr.File(
                label="Final deliverables (download)",
                file_count="multiple",
                file_types=[
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".csv",
                    ".xlsx",
                    ".xls",
                    ".txt",
                    ".doc",
                    ".docx",
                    ".json",
                    ".jsonl",
                    ".zip",
                ],
                interactive=False,
                height=200,
            )
            refresh_outputs_btn = gr.Button(
                "Refresh workspace files",
                variant="secondary",
            )
            with gr.Accordion("Download other files from workspace", open=False):
                workspace_output_explorer = gr.FileExplorer(
                    root_dir=str(workspace_base_dir()),
                    label="Browse session workspace",
                    file_count="multiple",
                    interactive=True,
                    max_height=400,
                )
                workspace_output_explorer_download = gr.File(
                    label="Download selected files",
                    file_count="multiple",
                    file_types=[
                        ".pdf",
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".csv",
                        ".xlsx",
                        ".xls",
                        ".txt",
                        ".doc",
                        ".docx",
                        ".json",
                        ".jsonl",
                        ".zip",
                    ],
                    interactive=False,
                    height=200,
                )

        with gr.Accordion("Session log outputs", open=False):
            gr.Markdown(
                "If using Pi as the agent, it writes a **JSONL** transcript for the active agent session under "
                "its `sessions/` directory. The file refreshes after each chat message "
                "or redaction task completes."
            )
            session_log_download = gr.File(
                label="Session log (JSONL)",
                file_count="single",
                file_types=[".jsonl"],
                interactive=False,
            )
            agent_finish_signal = gr.State(AGENT_FINISH_SIGNAL_NONE)
            agent_running_state = gr.State(False)
            pending_followup_message = gr.State("")

        chat_outputs = [
            chatbot,
            client_state,
            msg,
            activity_log,
            tool_panel,
            thinking_panel,
            session_info,
            send,
            abort_btn,
            start_redact_btn,
            workspace_output_download,
            session_log_download,
            pdf_preview,
            agent_finish_signal,
            agent_running_state,
        ]

        _followup_route_inputs = [
            msg,
            chatbot,
            client_state,
            session_hash_state,
            s3_output_folder_state,
            save_outputs_to_s3_state,
        ]
        _followup_queued_inputs = [
            pending_followup_message,
            chatbot,
            client_state,
            session_hash_state,
            s3_output_folder_state,
            save_outputs_to_s3_state,
        ]
        run_chat_send = send.click(
            route_followup_message,
            inputs=_followup_route_inputs,
            outputs=[*chat_outputs, pending_followup_message],
            queue=False,
            api_name="send_followup_message",
        )
        run_chat_queued_send = run_chat_send.then(
            submit_followup_chat_queued,
            inputs=_followup_queued_inputs,
            outputs=chat_outputs,
            api_visibility="undocumented",
        )
        notify_after_chat_send = run_chat_queued_send.then(
            _passthrough_chat_outputs_for_notify,
            inputs=_chat_outputs_notify_inputs(chat_outputs),
            outputs=chat_outputs,
            js=PI_AGENT_FINISH_NOTIFY_JS,
            api_visibility="undocumented",
        )
        run_chat_msg = msg.submit(
            route_followup_message,
            inputs=_followup_route_inputs,
            outputs=[*chat_outputs, pending_followup_message],
            queue=False,
            api_visibility="undocumented",
        )
        run_chat_queued_msg = run_chat_msg.then(
            submit_followup_chat_queued,
            inputs=_followup_queued_inputs,
            outputs=chat_outputs,
            api_visibility="undocumented",
        )
        notify_after_chat_msg = run_chat_queued_msg.then(
            _passthrough_chat_outputs_for_notify,
            inputs=_chat_outputs_notify_inputs(chat_outputs),
            outputs=chat_outputs,
            js=PI_AGENT_FINISH_NOTIFY_JS,
            api_visibility="undocumented",
        )
        run_redact_prepare = start_redact_btn.click(
            prepare_redaction_session_ui,
            inputs=[session_hash_state],
            outputs=[session_hash_state, workspace_session_info],
            api_visibility="undocumented",
        )
        run_redact_task = run_redact_prepare.then(
            submit_redaction_task,
            inputs=[
                redact_file,
                redact_instructions,
                page_range,
                ocr_method,
                pii_method,
                encourage_vlm_faces,
                encourage_vlm_signatures,
                chatbot,
                client_state,
                session_hash_state,
                s3_output_folder_state,
                save_outputs_to_s3_state,
            ],
            outputs=chat_outputs,
            api_name="run_agentic_redaction_task",
        )
        notify_after_redact_task = run_redact_task.then(
            _passthrough_chat_outputs_for_notify,
            inputs=_chat_outputs_notify_inputs(chat_outputs),
            outputs=chat_outputs,
            js=PI_AGENT_FINISH_NOTIFY_JS,
            api_visibility="undocumented",
        )
        abort_btn.click(
            abort_agent,
            inputs=[client_state],
            outputs=[send, abort_btn, start_redact_btn],
            # Steer uses ``queue=False`` on route_followup_message; idle follow-ups
            # stream via the queued ``submit_followup_chat_queued`` step (Gradio
            # needs the queue for multi-yield generators).
            cancels=[
                run_redact_prepare,
                run_redact_task,
                run_chat_queued_send,
                run_chat_queued_msg,
                notify_after_chat_send,
                notify_after_chat_msg,
                notify_after_redact_task,
            ],
            queue=False,
            api_visibility="undocumented",
        )
        clear.click(
            new_chat,
            inputs=[chatbot, client_state, session_hash_state],
            outputs=chat_outputs,
            api_visibility="undocumented",
        )

        if not IS_HF_SPACE:
            backend_provider.change(
                _backend_model_choices_update,
                inputs=[backend_provider],
                outputs=[backend_model],
                api_visibility="undocumented",
            )
        apply_backend_btn.click(
            apply_backend,
            inputs=[
                backend_provider,
                backend_model,
                gemini_api_key,
                hf_token,
                aws_region,
                aws_access_key_id,
                aws_secret_access_key,
                aws_session_token,
                client_state,
                session_hash_state,
            ],
            outputs=[
                client_state,
                session_info,
                gemini_api_key,
                hf_token,
                aws_secret_access_key,
                aws_session_token,
            ],
            api_name="apply_model_backend",
        )

        refresh_outputs_btn.click(
            fn=refresh_workspace_output_files_stub,
            inputs=None,
            outputs=workspace_output_explorer,
            api_visibility="undocumented",
        ).success(
            fn=refresh_workspace_panel,
            inputs=[session_hash_state],
            outputs=[workspace_output_explorer, workspace_output_explorer_download],
            api_visibility="undocumented",
        ).success(
            fn=preview_pdf_path_for_gradio,
            inputs=[session_hash_state],
            outputs=pdf_preview,
            api_visibility="undocumented",
        ).success(
            fn=_export_workspace_outputs,
            inputs=[
                session_hash_state,
                s3_output_folder_state,
                save_outputs_to_s3_state,
            ],
            outputs=None,
            api_visibility="undocumented",
        )

        workspace_output_explorer.input(
            fn=workspace_files_download_fn,
            inputs=[workspace_output_explorer, session_hash_state],
            outputs=workspace_output_explorer_download,
            api_visibility="undocumented",
        )

        demo.load(
            fn=_init_session_ui,
            inputs=[client_state],
            outputs=[
                session_hash_state,
                workspace_output_explorer,
                workspace_session_info,
                s3_output_folder_state,
                *chat_outputs,
            ],
            api_visibility="undocumented",
        )

    return demo


def launch_pi_ui() -> FastAPI | None:
    """Build UI and mount on FastAPI or launch Gradio directly."""
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    pi_root = (PI_ROOT_PATH or "").strip()
    fastapi_root = pi_root or FASTAPI_ROOT_PATH
    return mount_or_launch(
        demo,
        fastapi_app=create_fastapi_app(root_path=fastapi_root) if RUN_FASTAPI else None,
        allowed_paths=gradio_allowed_paths(),
        css=THINKING_PANEL_CSS,
        head_extra=PI_AGENT_FINISH_HEAD_HTML,
        server_name=PI_UI_HOST,
        server_port=PI_UI_PORT,
        root_path=pi_root,
        fastapi_root_path=fastapi_root,
    )


if RUN_FASTAPI:
    app = launch_pi_ui()
else:
    app = None


if __name__ == "__main__":
    if RUN_FASTAPI:
        import uvicorn

        uvicorn.run(
            "gradio_app:app",
            host=PI_UI_HOST,
            port=PI_UI_PORT,
            factory=False,
        )
    else:
        launch_pi_ui()
