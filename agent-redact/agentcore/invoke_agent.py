"""Shared AgentCore invoke logic (monorepo entrypoint + packaged runtime)."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from session_store import (
    append_turn,
    clear_session,
    get_messages,
    stringify_message_content,
)


def configure_import_paths(app_root: Path | None = None) -> tuple[Path, Path]:
    """
    Ensure imports resolve in the monorepo or a packaged AgentCore app folder.

    Returns ``(repo_root, agent_redact_root)`` for bootstrap_pi_config.
    """
    root = (app_root or Path(__file__).resolve().parent).resolve()
    agent_redact = root
    pi_dir = root / "pi"
    for path in (root, agent_redact, pi_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    repo_root = root
    if (root / "agent-redact").is_dir():
        repo_root = root
        agent_redact = root / "agent-redact"
    elif (
        root.name == "RedactionAgent" and (root.parent.parent / "agent-redact").is_dir()
    ):
        repo_root = root.parent.parent.parent
        agent_redact = repo_root / "agent-redact"
    return repo_root, agent_redact


def bootstrap_runtime_env(app_root: Path) -> None:
    """
    Lightweight env setup for AgentCore (no Pi skills sync or monorepo ``tools/``).

    Full :func:`bootstrap_pi_config.ensure_pi_config_env` pulls in ``pi_workspace_skills``
    and repo ``skills/`` — not vendored in the CodeZip bundle and will fail or stall
    runtime init on AWS.
    """
    from dotenv import load_dotenv

    root = app_root.resolve()
    for env_name in ("agentcore.env", ".env"):
        env_file = root / env_name
        if env_file.is_file():
            load_dotenv(env_file, override=False)

    os.environ.setdefault("AGENT_WORKSPACE_DIR", "/tmp/agentcore-workspace")
    os.environ.setdefault("AGENT_REDACTION_SPLIT_BACKEND", "true")
    os.environ.setdefault("AGENT_DEFAULT_PROVIDER", "amazon-bedrock")
    os.environ.setdefault("AGENT_DEFAULT_MODEL", "anthropic.claude-sonnet-4-6")
    os.environ.setdefault(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")
    )
    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])
    Path(os.environ["AGENT_WORKSPACE_DIR"]).mkdir(parents=True, exist_ok=True)


INVOKE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "DOC_REDACTION_GRADIO_URL",
        "DOC_REDACTION_GRADIO_AUTH_USER",
        "DOC_REDACTION_GRADIO_AUTH_PASSWORD",
        # CloudFront magic-link cookie: the AgentCore runtime runs outside the VPC
        # and reaches doc_redaction through CloudFront, so the token forwarded by
        # build_agentcore_invoke_runtime_config must be applied here or every
        # backend request hits the login wall ("credentials were not provided").
        "DOC_REDACTION_AUTH_TOKEN",
        "DOC_REDACTION_AUTH_COOKIE_NAME",
        "AGENT_DEFAULT_OCR_METHOD",
        "AGENT_DEFAULT_PII_METHOD",
        "HF_TOKEN",
        "DOC_REDACTION_HF_TOKEN",
    }
)


def apply_invoke_runtime_config(request: dict) -> None:
    """
      Apply per-invoke backend settings from the Gradio UI (overrides agentcore.env).

      The AgentCore runtime on AWS has its own ``agentcore.env``; without this, a
    deployed HF Space URL can win over the operator's local ``agent.env``.
    """
    raw = request.get("runtime_config") or request.get("runtime_env") or {}
    if not isinstance(raw, dict):
        return
    for key in INVOKE_RUNTIME_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            os.environ[key] = text


async def invoke_redaction_agent(request: dict) -> AsyncIterator[dict]:
    """Stream LangGraph agent events for one user prompt (multi-turn per session_hash)."""
    # Must run before LangChain imports (OpenInference patch order).
    try:
        from eval.arize_monitoring import setup_arize_ax_tracing

        setup_arize_ax_tracing()
    except ImportError:
        pass

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from workspace_sync import (
        apply_workspace_files,
        collect_workspace_files_for_sync,
    )

    apply_invoke_runtime_config(request)

    prompt = str(request.get("prompt") or request.get("message") or "").strip()
    session_hash = str(request.get("session_hash") or "").strip() or None
    if request.get("new_session"):
        clear_session(session_hash)

    if not prompt:
        yield {"type": "error", "message": "prompt is required"}
        return

    incoming_files = request.get("workspace_files") or []
    if isinstance(incoming_files, list) and incoming_files:
        written = apply_workspace_files(session_hash, incoming_files)
        if written:
            yield {
                "type": "status",
                "message": f"Synced {len(written)} file(s) into AgentCore workspace.",
            }

    backend_url = (os.environ.get("DOC_REDACTION_GRADIO_URL") or "").strip().rstrip("/")
    if backend_url:
        yield {
            "type": "status",
            "message": f"Redaction backend for this turn: {backend_url}",
        }

    from eval.arize_monitoring import arize_session_context, langgraph_trace_config
    from redaction_langgraph.graph import build_redaction_agent, graph_recursion_limit
    from redaction_langgraph.message_context import (
        get_trim_stats,
        is_context_overflow_error,
        reset_trim_stats,
        set_aggressive_trim,
    )

    graph, system_message = build_redaction_agent(session_hash)
    prior = get_messages(session_hash)
    inputs = {"messages": [system_message, *prior, HumanMessage(content=prompt)]}
    yield {"type": "agent_start"}

    assistant_chunks: list[str] = []
    stream_config = langgraph_trace_config(
        session_hash, recursion_limit=graph_recursion_limit()
    )

    def _emit_stream(active_graph) -> list[dict]:
        """Collect stream events; callers yield them and handle errors."""
        out: list[dict] = []
        reset_trim_stats()
        compaction_noted = False
        for event in active_graph.stream(
            inputs, stream_mode="updates", config=stream_config
        ):
            stats = get_trim_stats()
            if stats is not None and stats.trimmed and not compaction_noted:
                compaction_noted = True
                out.append(
                    {
                        "type": "status",
                        "message": (
                            f"Context compaction ({stats.tokens_before:,} → "
                            f"{stats.tokens_after:,} tokens)."
                        ),
                    }
                )
            for node, update in event.items():
                messages = update.get("messages") or []
                for message in messages:
                    if isinstance(message, AIMessage):
                        text = stringify_message_content(message.content)
                        if text:
                            assistant_chunks.append(text)
                        out.append(
                            {
                                "type": "message_update",
                                "node": node,
                                "role": "assistant",
                                "content": text,
                                "tool_calls": message.tool_calls or [],
                            }
                        )
                    elif isinstance(message, ToolMessage):
                        out.append(
                            {
                                "type": "message_update",
                                "node": node,
                                "role": "tool",
                                "content": stringify_message_content(message.content),
                                "tool_name": str(message.name or "tool"),
                            }
                        )
                    else:
                        content = getattr(message, "content", "")
                        out.append(
                            {
                                "type": "message_update",
                                "node": node,
                                "role": getattr(message, "type", "unknown"),
                                "content": content,
                            }
                        )
        return out

    try:
        with arize_session_context(session_hash):
            try:
                for item in _emit_stream(graph):
                    yield item
            except Exception as exc:
                if not is_context_overflow_error(exc):
                    raise
                assistant_chunks.clear()
                yield {
                    "type": "status",
                    "message": (
                        "Prompt exceeded model context — retrying once with "
                        "aggressive compaction…"
                    ),
                }
                set_aggressive_trim(True)
                try:
                    graph, _ = build_redaction_agent(
                        session_hash, aggressive_compaction=True
                    )
                    for item in _emit_stream(graph):
                        yield item
                finally:
                    set_aggressive_trim(False)
    except Exception as exc:
        yield {"type": "error", "message": f"LangGraph agent failed: {exc}"}
        return

    append_turn(
        session_hash,
        user_text=prompt,
        assistant_text="\n".join(assistant_chunks),
    )
    if request.get("sync_workspace_files"):
        for item in collect_workspace_files_for_sync(session_hash):
            yield {"type": "workspace_file", **item}
    yield {"type": "agent_end", "message": "Agent finished."}
