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

    os.environ.setdefault("PI_WORKSPACE_DIR", "/tmp/agentcore-workspace")
    os.environ.setdefault("PI_REDACTION_SPLIT_BACKEND", "true")
    os.environ.setdefault("PI_DEFAULT_PROVIDER", "amazon-bedrock")
    os.environ.setdefault("PI_DEFAULT_MODEL", "anthropic.claude-sonnet-4-6")
    os.environ.setdefault(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")
    )
    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])
    Path(os.environ["PI_WORKSPACE_DIR"]).mkdir(parents=True, exist_ok=True)


async def invoke_redaction_agent(request: dict) -> AsyncIterator[dict]:
    """Stream LangGraph agent events for one user prompt (multi-turn per session_hash)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from workspace_sync import (
        apply_workspace_files,
        collect_workspace_files_for_sync,
    )

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

    from redaction_langgraph.graph import build_redaction_agent, graph_recursion_limit

    graph, system_message = build_redaction_agent(session_hash)
    prior = get_messages(session_hash)
    inputs = {"messages": [system_message, *prior, HumanMessage(content=prompt)]}
    yield {"type": "agent_start"}

    assistant_chunks: list[str] = []
    stream_config = {"recursion_limit": graph_recursion_limit()}
    try:
        for event in graph.stream(inputs, stream_mode="updates", config=stream_config):
            for node, update in event.items():
                messages = update.get("messages") or []
                for message in messages:
                    if isinstance(message, AIMessage):
                        text = stringify_message_content(message.content)
                        if text:
                            assistant_chunks.append(text)
                        yield {
                            "type": "message_update",
                            "node": node,
                            "role": "assistant",
                            "content": text,
                            "tool_calls": message.tool_calls or [],
                        }
                    elif isinstance(message, ToolMessage):
                        yield {
                            "type": "message_update",
                            "node": node,
                            "role": "tool",
                            "content": stringify_message_content(message.content),
                            "tool_name": str(message.name or "tool"),
                        }
                    else:
                        content = getattr(message, "content", "")
                        yield {
                            "type": "message_update",
                            "node": node,
                            "role": getattr(message, "type", "unknown"),
                            "content": content,
                        }
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
