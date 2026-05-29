#!/usr/bin/env python3
"""
Gradio chat UI for Pi (RPC mode).

Streams Pi RPC events into a chatbot, activity log, tool output panel, and
optional thinking trace. Includes a redaction task panel driven by the
partnership prompt template.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_pi_config import ensure_pi_config_env

ensure_pi_config_env(_REPO_ROOT)

import gradio as gr
from output_files import (
    collect_final_output_files,
    gradio_allowed_paths,
    refresh_workspace_output_files_stub,
    refresh_workspace_panel,
    workspace_files_download_fn,
)
from pi_agent_config import (
    apply_session_credentials,
    configure_aws_credentials,
    credential_status_markdown,
    default_model_for_provider,
    gemini_api_key_configured,
    get_default_provider,
    is_hf_space_profile,
    mirror_hf_token_from_env,
    models_for_provider,
    normalize_provider,
    provider_choices,
    provider_label,
    resolved_default_model,
    write_runtime_config,
)
from pi_examples import example_rows, examples_status_markdown
from pi_rpc_client import (
    PiRpcClient,
    PiRpcError,
    PiStreamEvent,
    assistant_text_since_last_user,
    default_client,
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

from session_workspace import (
    init_session_workspace,
    session_workspace_dir,
    workspace_base_dir,
    workspace_context_prefix,
)

from tools.aws_functions import export_outputs_to_s3
from tools.config import (
    ACTIVITY_MAX_LINES,
    EMPTY_SEND_WITH_FILE_HINT,
    HOST_NAME,
    PI_GRADIO_PORT,
    PI_INTRO_TEXT,
    PI_UI_HOST,
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
from tools.gradio_platform import (
    create_fastapi_app,
    log_agent_usage_event,
    log_platform_access,
    mount_or_launch,
)

IS_HF_SPACE = is_hf_space_profile()
# Use PI_GRADIO_PORT only — GRADIO_SERVER_PORT is the main app's default (7860) and is
# written into os.environ during tools.config import, which would override 7862 here.
PI_UI_PORT = PI_GRADIO_PORT

app = None


def _client_provider_model(client: PiRpcClient | None) -> tuple[str, str]:
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


def _llm_model_label(client: PiRpcClient | None) -> str:
    provider, model = _client_provider_model(client)
    if provider and model:
        return f"{provider}/{model}"
    return model or provider


def _after_pi_task(
    *,
    session_hash: str,
    client: PiRpcClient | None,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    document_name: str = "",
    started_at: float | None = None,
    base_file: str | None = None,
    ocr_method: str = "",
    pii_method: str = "",
    total_page_count: int = 0,
    vlm_model_name: str | None = None,
) -> None:
    duration = round(time.time() - started_at, 2) if started_at else ""
    log_agent_usage_event(
        session_hash=session_hash,
        duration_seconds=duration,
        document_name=document_name,
        total_page_count=total_page_count,
        ocr_method=ocr_method,
        pii_method=pii_method,
        llm_model_name=_llm_model_label(client),
        vlm_model_name=vlm_model_name or os.environ.get("PI_VLM_MODEL", ""),
        task="agent",
    )
    persist_session_log(client, session_hash=session_hash)
    file_paths = collect_final_output_files(session_hash)
    if file_paths and save_outputs_to_s3 and s3_output_folder:
        export_outputs_to_s3(
            file_paths,
            s3_output_folder,
            save_outputs_to_s3,
            base_file,
        )


def _export_workspace_outputs(
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    base_file: str | None = None,
) -> None:
    file_paths = collect_final_output_files(session_hash)
    if file_paths and save_outputs_to_s3 and s3_output_folder:
        export_outputs_to_s3(
            file_paths,
            s3_output_folder,
            save_outputs_to_s3,
            base_file,
        )


def _clone_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": item["role"], "content": item["content"]} for item in history]


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


def _finalize_assistant_chat(
    client: PiRpcClient,
    history: list[dict[str, Any]],
    *,
    completed_segments: list[str],
    streaming_text: str,
    activity: list[str],
) -> None:
    """Fill an empty assistant bubble after tool-only Gemini turns."""
    if not history or history[-1].get("role") != "assistant":
        return
    if _assistant_display_text(completed_segments, streaming_text).strip():
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
        )
        return
    if history[-1].get("content", "").strip():
        return

    try:
        fallback = assistant_text_since_last_user(client.get_messages())
    except PiRpcError:
        fallback = ""

    if fallback.strip():
        history[-1]["content"] = fallback
        return

    if activity:
        history[-1]["content"] = (
            "_This run completed using tools only (no assistant prose was streamed). "
            "See **Thinking log** for step-by-step activity._"
        )


def _gemini_key_error() -> str | None:
    if IS_HF_SPACE and not gemini_api_key_configured():
        return (
            "**Gemini API key required.** Paste your key in **Agent backend** and click "
            "**Apply backend** before chatting or starting a redaction task."
        )
    return None


def _ensure_client(
    client: PiRpcClient | None,
    session_hash: str = "",
) -> PiRpcClient:
    key_error = _gemini_key_error()
    if key_error:
        raise PiRpcError(key_error)
    if isinstance(client, PiRpcClient) and client.running:
        return client
    client = default_client(session_hash or None)
    client.start()
    provider = normalize_provider(get_default_provider())
    model = resolved_default_model(provider)
    try:
        client.set_model(provider, model)
    except PiRpcError:
        pass
    return client


def _coerce_client(client: Any) -> PiRpcClient | None:
    return client if isinstance(client, PiRpcClient) else None


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


def _append_chat_segment(
    completed_segments: list[str],
    streaming_text: str,
    segment: str,
) -> tuple[list[str], str]:
    """Append a new visible chat segment (tool line or prose), preserving prior segments."""
    segment = segment.strip()
    if not segment:
        return completed_segments, streaming_text
    if streaming_text.strip():
        completed_segments = completed_segments + [streaming_text.strip()]
        streaming_text = ""
    if not completed_segments or completed_segments[-1] != segment:
        completed_segments = completed_segments + [segment]
    return completed_segments, streaming_text


def _apply_event(
    event: PiStreamEvent,
    *,
    history: list[dict[str, Any]],
    activity: list[str],
    thinking: str,
    tool_output: str,
    tool_heading: str,
    completed_segments: list[str],
    streaming_text: str,
) -> tuple[list[dict[str, Any]], list[str], str, str, str, list[str], str]:
    if event.kind == "text_snapshot":
        if event.text.strip().startswith("**") and ":" in event.text.split("\n", 1)[0]:
            completed_segments, streaming_text = _append_chat_segment(
                completed_segments, streaming_text, event.text
            )
        else:
            streaming_text = event.text
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
        )

    elif event.kind == "text_delta":
        streaming_text += event.text
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
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
        detail = event.text or label
        tool_line = f"**{label}:** {detail}" if detail != label else f"**{label}**"
        completed_segments, streaming_text = _append_chat_segment(
            completed_segments, streaming_text, tool_line
        )
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
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
        history[-1]["content"] = _assistant_display_text(
            completed_segments,
            streaming_text,
        )
        history[-1]["content"] += f"\n\n**Error:** {event.text}"

    elif event.kind == "done":
        if streaming_text.strip():
            completed_segments.append(streaming_text)
            streaming_text = ""
            history[-1]["content"] = _assistant_display_text(
                completed_segments, streaming_text
            )
        activity = _append_activity(activity, event.text)

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


def _pi_agent_model_label(client: PiRpcClient | None) -> str:
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


def _agent_status_markdown(client: PiRpcClient | None = None) -> str:
    """Redaction backend URL, Pi model, and credentials — shown at top of the UI."""
    from redaction_prompt import doc_redaction_gradio_url

    lines = [
        f"**Redaction backend:** `{doc_redaction_gradio_url()}`",
        f"**Pi agent model:** `{_pi_agent_model_label(client)}`",
    ]
    if client is None or not client.running:
        lines.insert(0, "**Status:** Ready")
        lines.append("")
        lines.append(
            "_Set `DOC_REDACTION_GRADIO_URL` in `config/pi_agent.env` if the doc_redaction "
            "app is not at the URL above. Apply **Agent backend** to start Pi._"
        )
    else:
        lines.insert(0, "**Status:** Pi agent connected")
    lines.append("")
    lines.append(credential_status_markdown())
    return "  \n".join(lines)


def _session_summary(client: PiRpcClient) -> str:
    try:
        state = client.get_state()
    except PiRpcError as exc:
        return f"{_agent_status_markdown(client)}  \n\n_Could not read Pi state: {exc}_"
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
    client: PiRpcClient | None,
    session_hash: str,
):
    normalized = normalize_provider(provider)
    model = (model_id or default_model_for_provider(normalized)).strip()
    if model not in models_for_provider(normalized):
        model = default_model_for_provider(normalized)

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

    rpc = default_client(session_hash or None)
    try:
        rpc.start()
        rpc.set_model(normalized, model)
        rpc.new_session()
        summary = (
            f"**Backend applied:** `{provider_label(normalized)}` / `{model}`  \n\n"
            f"{_session_summary(rpc)}"
        )
    except (PiRpcError, FileNotFoundError, OSError) as exc:
        rpc.close()
        rpc = None
        summary = f"**Backend error:** {exc}  \n\n{credential_status_markdown()}"

    return (
        rpc,
        summary,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )


def _init_session_ui(
    request: gr.Request,
) -> tuple[str, Any, str, list[str] | None, str]:
    session_hash, explorer, status, s3_prefix = init_session_workspace(request)
    log_platform_access(session_hash, HOST_NAME)
    return (
        session_hash,
        explorer,
        status,
        collect_final_output_files(session_hash),
        s3_prefix,
    )


def _chat_yield(
    history: list[dict[str, Any]],
    client: PiRpcClient,
    activity: list[str],
    thinking: str,
    tool_heading: str,
    tool_output: str,
    *,
    msg: str = "",
    send_enabled: bool = True,
    abort_enabled: bool = False,
    redact_enabled: bool = True,
    session_info: str | None = None,
    session_hash: str = "",
    refresh_final_files: bool = False,
):
    final_files: list[str] | None | dict[str, Any]
    session_log: str | None | dict[str, Any]
    if refresh_final_files:
        final_files = collect_final_output_files(session_hash)
        session_log = collect_session_log_download(client)
    else:
        final_files = gr.update()
        session_log = gr.update()

    return (
        _clone_history(history),
        client,
        msg,
        _format_activity(activity),
        _format_tool_panel(tool_heading, tool_output),
        _truncate_thinking(thinking),
        session_info if session_info is not None else _session_summary(client),
        gr.update(interactive=send_enabled),
        gr.update(interactive=abort_enabled),
        gr.update(interactive=redact_enabled),
        final_files,
        session_log,
    )


def _run_pi_chat(
    message: str,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
    *,
    chat_user_message: str | None = None,
    session_hash: str = "",
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
                gr.update(),
                gr.update(),
            )
        return

    history = list(history or [])
    client = _ensure_client(client, session_hash)
    activity: list[str] = []
    thinking = ""
    tool_output = ""
    tool_heading = ""
    completed_segments: list[str] = []
    streaming_text = ""
    task_started_at = time.time()

    def _complete_pi_task() -> None:
        _after_pi_task(
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
        )

    history.append({"role": "user", "content": chat_user_message or message.strip()})
    history.append({"role": "assistant", "content": ""})
    activity = _append_activity(activity, "Prompt sent.")
    session_info = _session_summary(client)

    yield _chat_yield(
        history,
        client,
        activity,
        thinking,
        tool_heading,
        tool_output,
        send_enabled=False,
        abort_enabled=True,
        redact_enabled=False,
        session_info=session_info,
        session_hash=session_hash,
    )

    from pi_workspace_skills import workspace_boundary_prefix

    pi_message = (
        workspace_boundary_prefix(session_hash)
        + workspace_context_prefix(session_hash)
        + message.strip()
    )
    prompt_to_send = pi_message
    quota_failures = 0

    try:
        while True:
            turn_error: str | None = None
            try:
                for event in client.prompt_events(prompt_to_send):
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
                    )
                    yield _chat_yield(
                        history,
                        client,
                        activity,
                        thinking,
                        tool_heading,
                        tool_output,
                        send_enabled=False,
                        abort_enabled=True,
                        redact_enabled=False,
                        session_info=session_info,
                        session_hash=session_hash,
                    )
                turn_error = last_assistant_turn_error(client.get_messages())
            except PiRpcError as exc:
                if not is_rate_limit_error(str(exc)):
                    raise
                turn_error = str(exc)

            if turn_error and is_rate_limit_error(turn_error):
                quota_failures += 1
                if quota_failures >= QUOTA_RETRY_ATTEMPTS:
                    err_summary = turn_error[:500].replace("\n", " ")
                    history[-1]["content"] = (
                        f"**Gemini rate limit / quota:** stopped after "
                        f"{QUOTA_RETRY_ATTEMPTS} consecutive attempts.\n\n"
                        f"{err_summary}"
                    )
                    activity = _append_activity(
                        activity,
                        f"**Quota retries exhausted** ({QUOTA_RETRY_ATTEMPTS} attempts).",
                    )
                    _complete_pi_task()
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
                    )
                    return

                activity = _append_activity(
                    activity,
                    (
                        f"Gemini rate limit — waiting {QUOTA_RETRY_DELAY_S}s before "
                        f"retry {quota_failures}/{QUOTA_RETRY_ATTEMPTS}…"
                    ),
                )
                yield _chat_yield(
                    history,
                    client,
                    activity,
                    thinking,
                    tool_heading,
                    tool_output,
                    send_enabled=False,
                    abort_enabled=True,
                    redact_enabled=False,
                    session_info=session_info,
                    session_hash=session_hash,
                )
                time.sleep(QUOTA_RETRY_DELAY_S)
                prompt_to_send = QUOTA_CONTINUE_PROMPT
                history.append({"role": "assistant", "content": ""})
                completed_segments = []
                streaming_text = ""
                continue

            break
    except PiRpcError as exc:
        history[-1]["content"] = f"**Pi error:** {exc}"
        activity = _append_activity(activity, f"**Pi error:** {exc}")
        _complete_pi_task()
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
        )
        return
    except Exception:
        if client.abort_requested:
            activity = _append_activity(activity, "**Aborted.**")
            _complete_pi_task()
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
            )
            return
        raise

    _finalize_assistant_chat(
        client,
        history,
        completed_segments=completed_segments,
        streaming_text=streaming_text,
        activity=activity,
    )

    _complete_pi_task()
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
    )


def chat_respond(
    message: str,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    redact_file: str | None,
):
    yield from _run_pi_chat(
        message,
        history,
        client,
        session_hash=session_hash,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
        redact_file=redact_file,
    )


def _redaction_page_count(upload_file: str | None, page_range: str) -> int:
    if not upload_file or not str(upload_file).lower().endswith(".pdf"):
        return 0
    try:
        total = pdf_page_count(upload_file)
        return pages_to_process_count(page_range or "all", total)
    except (ValueError, OSError):
        return 0


def submit_redaction_task(
    upload_file: str | None,
    user_instructions: str,
    page_range: str,
    ocr_method: str,
    pii_method: str,
    encourage_vlm_faces: bool,
    encourage_vlm_signatures: bool,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
    session_hash: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
):
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
    workspace_path = (
        session_workspace_dir(session_hash) if session_hash.strip() else None
    )
    try:
        _file_name, prompt = prepare_redaction_task(
            upload_file,
            user_instructions,
            page_range=page_range or "all",
            settings=settings,
            workspace_dir=workspace_path,
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
            gr.update(),
            gr.update(),
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
    yield from _run_pi_chat(
        prompt,
        history,
        client,
        chat_user_message=chat_summary,
        session_hash=session_hash,
        s3_output_folder=s3_output_folder,
        save_outputs_to_s3=save_outputs_to_s3,
        document_name=_file_name,
        base_file=upload_file,
        ocr_method=settings.ocr_method,
        pii_method=settings.pii_method,
        total_page_count=page_count,
        vlm_model_name=os.environ.get("PI_VLM_MODEL"),
    )


def abort_agent(client: PiRpcClient | None):
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
    client: PiRpcClient | None,
    session_hash: str,
):
    if client is not None:
        try:
            client.new_session()
        except PiRpcError:
            client.close()
            client = default_client(session_hash or None)
            client.start()
    else:
        client = default_client(session_hash or None)
        client.start()
    return _chat_yield(
        [],
        client,
        ["New session."],
        "",
        "",
        "",
        session_hash=session_hash,
        refresh_final_files=True,
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
    hf_redaction_blurb = (
        "Upload a document and add bullet-point requirements. Redaction runs on a **remote** "
        "Redaction App Hugging Face Space.  \n"
        "When ready, use **Start redaction task** under the chat panel."
        if IS_HF_SPACE
        else (
            "Upload a PDF (or other supported document). Add bullet-point instructions for redaction below. \n"
            "When ready, use **Start redaction task** under the chat panel."
        )
    )
    backend_blurb = (
        "Gemini powers the Pi agent on this Space. Paste your **Gemini API key** "
        "(session-only, not stored on disk). Optionally override the **HF token** used "
        "to reach the private redaction backend."
        if IS_HF_SPACE
        else (
            "Choose which LLM powers the Pi agent (chat and redaction orchestration). "
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
                                "**Technical constraints** — they suggest defaults to Pi for "
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
                            "then **Start redaction task** under the chat panel."
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
                        label="HF token for redaction Space (session override)",
                        type="password",
                        placeholder="Uses HF_TOKEN Space secret if empty",
                        visible=IS_HF_SPACE,
                    )
                    with gr.Accordion("AWS credentials (optional)", open=False):
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
                chatbot = gr.Chatbot(label="Task progress", height=480)
                with gr.Row():
                    start_redact_btn = gr.Button(
                        "Start redaction task",
                        variant="primary",
                    )
                    abort_btn = gr.Button("Abort", variant="stop", interactive=False)
                clear = gr.Button("New session")
                with gr.Accordion("Follow-up chat (optional)", open=False):
                    msg = gr.Textbox(
                        label="Message",
                        placeholder=(
                            "Optional message after a redaction task (e.g. fix page 3)"
                        ),
                        lines=3,
                    )
                    send = gr.Button("Send follow-up", variant="secondary")

                with gr.Accordion("Thinking log", open=False):
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
                "**Final deliverables** appear automatically when the agent saves to "
                "`review/output_review_final/` (or `review/output_final/`). "
                "Downloads below are copied to your session's `output_final_download/` "
                "prefixes removed, and duplicate filenames collapsed to the newest file. "
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
                    ".zip",
                ],
                interactive=False,
                height=200,
            )
            refresh_outputs_btn = gr.Button(
                "Refresh workspace files",
                variant="secondary",
            )
            workspace_output_explorer = gr.FileExplorer(
                root_dir=str(workspace_base_dir()),
                label="Browse session workspace",
                file_count="multiple",
                interactive=True,
                max_height=400,
            )

        with gr.Accordion("Session log outputs", open=False):
            gr.Markdown(
                "Pi writes a **JSONL** transcript for the active agent session under "
                "its `sessions/` directory. The file refreshes after each chat message "
                "or redaction task completes."
            )
            session_log_download = gr.File(
                label="Pi session log (JSONL)",
                file_count="single",
                file_types=[".jsonl"],
                interactive=False,
            )

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
        ]

        run_event = send.click(
            chat_respond,
            inputs=[
                msg,
                chatbot,
                client_state,
                session_hash_state,
                s3_output_folder_state,
                save_outputs_to_s3_state,
                redact_file,
            ],
            outputs=chat_outputs,
        )
        msg.submit(
            chat_respond,
            inputs=[
                msg,
                chatbot,
                client_state,
                session_hash_state,
                s3_output_folder_state,
                save_outputs_to_s3_state,
                redact_file,
            ],
            outputs=chat_outputs,
        )
        run_redact_event = start_redact_btn.click(
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
        )
        abort_btn.click(
            abort_agent,
            inputs=[client_state],
            outputs=[send, abort_btn, start_redact_btn],
            cancels=[run_event, run_redact_event],
            queue=False,
        )
        clear.click(
            new_chat,
            inputs=[chatbot, client_state, session_hash_state],
            outputs=chat_outputs,
        )

        if not IS_HF_SPACE:
            backend_provider.change(
                _backend_model_choices_update,
                inputs=[backend_provider],
                outputs=[backend_model],
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
        )

        refresh_outputs_btn.click(
            fn=refresh_workspace_output_files_stub,
            inputs=None,
            outputs=workspace_output_explorer,
        ).success(
            fn=refresh_workspace_panel,
            inputs=[session_hash_state],
            outputs=[workspace_output_explorer, workspace_output_download],
        ).success(
            fn=_export_workspace_outputs,
            inputs=[
                session_hash_state,
                s3_output_folder_state,
                save_outputs_to_s3_state,
            ],
            outputs=None,
        )

        workspace_output_explorer.input(
            fn=workspace_files_download_fn,
            inputs=[workspace_output_explorer, session_hash_state],
            outputs=workspace_output_download,
        )

        demo.load(
            fn=_init_session_ui,
            inputs=None,
            outputs=[
                session_hash_state,
                workspace_output_explorer,
                workspace_session_info,
                workspace_output_download,
                s3_output_folder_state,
            ],
        )

    return demo


def launch_pi_ui() -> FastAPI | None:
    """Build UI and mount on FastAPI or launch Gradio directly."""
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    return mount_or_launch(
        demo,
        fastapi_app=create_fastapi_app() if RUN_FASTAPI else None,
        allowed_paths=gradio_allowed_paths(),
        css=THINKING_PANEL_CSS,
        server_name=PI_UI_HOST,
        server_port=PI_UI_PORT,
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
