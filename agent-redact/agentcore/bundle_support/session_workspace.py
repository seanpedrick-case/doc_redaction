"""Gradio-free session workspace helpers for AgentCore runtime bundles."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9_@.+-]+")


def workspace_base_dir() -> Path:
    raw = (os.environ.get("PI_WORKSPACE_DIR") or "").strip()
    if raw:
        path = Path(raw)
    else:
        path = Path("/tmp/agentcore-workspace")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def session_workspace_enabled() -> bool:
    raw = os.environ.get("PI_SESSION_WORKSPACE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def sanitize_session_id(raw: str) -> str:
    cleaned = _SESSION_ID_RE.sub("_", (raw or "").strip())[:128].strip("_")
    return cleaned or "default"


def session_workspace_dir(session_hash: str) -> Path:
    base = workspace_base_dir().resolve()
    if not session_workspace_enabled():
        return base
    safe_id = sanitize_session_id(session_hash)
    candidate = (base / safe_id).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return (base / "default").resolve()
    return candidate


def ensure_session_workspace(session_hash: str) -> Path:
    workspace = session_workspace_dir(session_hash)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
