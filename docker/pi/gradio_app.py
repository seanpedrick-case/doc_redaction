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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from output_files import (
    gradio_allowed_paths,
    load_workspace_output_files,
    refresh_workspace_output_files_stub,
    workspace_files_download_fn,
)
from pi_rpc_client import PiRpcClient, PiRpcError, PiStreamEvent, default_client
from redaction_prompt import prepare_redaction_task

PI_UI_TITLE = os.environ.get("PI_GRADIO_TITLE", "Agentic Document Redaction")
PI_UI_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))
PI_UI_HOST = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
SHOW_THINKING = os.environ.get("PI_GRADIO_SHOW_THINKING", "false").lower() in {
    "1",
    "true",
    "yes",
}
SHOW_TOOL_OUTPUT = os.environ.get("PI_GRADIO_SHOW_TOOL_OUTPUT", "true").lower() in {
    "1",
    "true",
    "yes",
}
TOOL_OUTPUT_MAX = int(os.environ.get("PI_GRADIO_TOOL_OUTPUT_MAX", "12000"))
ACTIVITY_MAX_LINES = int(os.environ.get("PI_GRADIO_ACTIVITY_MAX_LINES", "50"))
THINKING_DISPLAY_MAX = int(os.environ.get("PI_GRADIO_THINKING_MAX_CHARS", "16000"))
THINKING_PANEL_CSS = """
.thinking-panel textarea {
    max-height: 280px !important;
    overflow-y: auto !important;
}
"""


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


def _ensure_client(client: PiRpcClient | None) -> PiRpcClient:
    if isinstance(client, PiRpcClient) and client.running:
        return client
    client = default_client()
    client.start()
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
        streaming_text = event.text
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
        )

    elif event.kind == "text_delta":
        streaming_text += event.text
        history[-1]["content"] = _assistant_display_text(
            completed_segments, streaming_text
        )

    elif event.kind == "thinking_snapshot" and SHOW_THINKING:
        thinking = event.text

    elif event.kind == "thinking_delta" and SHOW_THINKING:
        thinking += event.text

    elif event.kind == "status":
        activity = _append_activity(activity, event.text)

    elif event.kind == "tool_start":
        if streaming_text.strip():
            completed_segments.append(streaming_text)
            streaming_text = ""
            history[-1]["content"] = _assistant_display_text(
                completed_segments, streaming_text
            )
        label = event.tool_name or "tool"
        detail = event.text or label
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


def _session_summary(client: PiRpcClient) -> str:
    try:
        state = client.get_state()
    except PiRpcError as exc:
        return f"_Could not read Pi state: {exc}_"
    model = state.get("model") or {}
    model_label = model.get("id") or model.get("name") or "unknown"
    session_file = state.get("sessionFile") or "—"
    streaming = state.get("isStreaming")
    compacting = state.get("isCompacting")
    return (
        f"**Model:** `{model_label}`  \n"
        f"**Streaming:** `{streaming}` · **Compacting:** `{compacting}`  \n"
        f"**Session:** `{session_file}`"
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
):
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
    )


def _run_pi_chat(
    message: str,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
    *,
    chat_user_message: str | None = None,
):
    if not message or not message.strip():
        client = client if client and client.running else None
        if client:
            yield _chat_yield(history or [], client, [], "", "", "")
        else:
            yield (
                history or [],
                None,
                "",
                "_No activity yet._",
                "",
                "",
                "_Ready._",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
        return

    history = list(history or [])
    client = _ensure_client(client)
    activity: list[str] = []
    thinking = ""
    tool_output = ""
    tool_heading = ""
    completed_segments: list[str] = []
    streaming_text = ""

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
    )

    try:
        for event in client.prompt_events(message.strip()):
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
            )
    except PiRpcError as exc:
        history[-1]["content"] = f"**Pi error:** {exc}"
        activity = _append_activity(activity, f"**Pi error:** {exc}")
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
        )
        return
    except Exception:
        if client.abort_requested:
            activity = _append_activity(activity, "**Aborted.**")
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
            )
            return
        raise

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
    )


def chat_respond(
    message: str,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
):
    yield from _run_pi_chat(message, history, client)


def submit_redaction_task(
    upload_file: str | None,
    user_instructions: str,
    page_range: str,
    history: list[dict[str, Any]] | None,
    client: PiRpcClient | None,
):
    try:
        _file_name, prompt = prepare_redaction_task(
            upload_file,
            user_instructions,
            page_range=page_range or "all",
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        history = list(history or [])
        history.append(
            {"role": "user", "content": f"_Redaction task not started: {exc}_"}
        )
        client = _ensure_client(client) if client and client.running else client
        yield (
            _clone_history(history),
            client,
            "",
            _format_activity([f"**Redaction task error:** {exc}"]),
            "",
            "",
            _session_summary(client) if client and client.running else "_Ready._",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )
        return

    chat_summary = (
        f"**Redaction task:** `{_file_name}`  \n"
        f"**Page range:** `{page_range or 'all'}`\n\n"
        f"{user_instructions.strip()}"
    )
    yield from _run_pi_chat(
        prompt,
        history,
        client,
        chat_user_message=chat_summary,
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


def new_chat(_history, client: PiRpcClient | None):
    if client is not None:
        try:
            client.new_session()
        except PiRpcError:
            client.close()
            client = default_client()
            client.start()
    else:
        client = default_client()
        client.start()
    return _chat_yield([], client, ["New session."], "", "", "")


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=PI_UI_TITLE,
        fill_height=True,
    ) as demo:
        gr.Markdown(
            f"# {PI_UI_TITLE}\n"
            "Upload a document, add redaction requirements, and start a task — or chat with Pi directly."
        )
        client_state = gr.State(None)

        session_info = gr.Markdown("_Ready._")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Accordion("Redaction task", open=True):
                    gr.Markdown(
                        "Upload a PDF (or other supported document). Add bullet-point instructions, "
                        "then **Start redaction task**. Pi receives the full task prompt from "
                        "`skills/Example prompt partnership.txt` with your file copied to the shared "
                        "workspace (`/home/user/app/workspace/` in Docker)."
                    )
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
                    )
                    redact_instructions = gr.Textbox(
                        label="Redaction requirements",
                        placeholder=(
                            "- Redact all personal names\n"
                            "- Remove organisation addresses\n"
                            "- Keep publication titles visible"
                        ),
                        lines=8,
                    )
                    page_range = gr.Textbox(
                        label="Page range",
                        value="all",
                        placeholder="all or e.g. 1-56",
                    )
                    start_redact_btn = gr.Button(
                        "Start redaction task",
                        variant="primary",
                    )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=480)
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Optional follow-up message to Pi",
                    lines=3,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="secondary")
                    abort_btn = gr.Button("Abort", variant="stop", interactive=False)
                    clear = gr.Button("New session")

            with gr.Column(scale=2):
                with gr.Accordion("Thinking log", open=True):
                    activity_log = gr.Markdown(value="_No activity yet._")
                    tool_panel = gr.Markdown(value="", max_height=800)
                    thinking_panel = gr.Textbox(
                        label="Thinking (stream)",
                        lines=12,
                        max_lines=50,
                        interactive=False,
                        visible=SHOW_THINKING,
                        elem_classes=["thinking-panel"],
                        autoscroll=True,
                    )

        with gr.Accordion("Workspace output files", open=False):
            gr.Markdown(
                "Browse files written under the shared workspace (e.g. "
                "`redact/<document>/output_redact/`). Select files in the explorer, "
                "then download them below."
            )
            refresh_outputs_btn = gr.Button(
                "Refresh workspace files",
                variant="secondary",
            )
            workspace_output_explorer = gr.FileExplorer(
                root_dir=str(
                    os.environ.get("PI_WORKSPACE_DIR", "/home/user/app/workspace")
                ),
                label="Workspace files",
                file_count="multiple",
                interactive=True,
                max_height=400,
            )
            workspace_output_download = gr.File(
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
                    ".zip",
                ],
                interactive=False,
                height=200,
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
        ]

        run_event = send.click(
            chat_respond,
            inputs=[msg, chatbot, client_state],
            outputs=chat_outputs,
        )
        msg.submit(
            chat_respond,
            inputs=[msg, chatbot, client_state],
            outputs=chat_outputs,
        )
        run_redact_event = start_redact_btn.click(
            submit_redaction_task,
            inputs=[
                redact_file,
                redact_instructions,
                page_range,
                chatbot,
                client_state,
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
        clear.click(new_chat, inputs=[chatbot, client_state], outputs=chat_outputs)

        refresh_outputs_btn.click(
            fn=refresh_workspace_output_files_stub,
            inputs=None,
            outputs=workspace_output_explorer,
        ).success(
            fn=load_workspace_output_files,
            inputs=None,
            outputs=workspace_output_explorer,
        )

        workspace_output_explorer.input(
            fn=workspace_files_download_fn,
            inputs=workspace_output_explorer,
            outputs=workspace_output_download,
        )

        demo.load(
            fn=load_workspace_output_files,
            inputs=None,
            outputs=workspace_output_explorer,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Default(primary_hue="blue"),
        css=THINKING_PANEL_CSS,
        server_name=PI_UI_HOST,
        server_port=PI_UI_PORT,
        show_error=True,
        allowed_paths=gradio_allowed_paths(),
    )
