"""Per-session workspace paths for the Pi Gradio UI (mirrors main app session folders)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gradio as gr
from pi_agent_config import is_hf_space_profile

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.config import SESSION_OUTPUT_FOLDER  # noqa: E402

_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9_@.+-]+")


def workspace_base_dir() -> Path:
    return Path(os.environ.get("PI_WORKSPACE_DIR", "/home/user/app/workspace"))


# Back-compat alias used by output_files / redaction_prompt.
WORKSPACE_BASE_DIR = workspace_base_dir()


def session_workspace_enabled() -> bool:
    """When true, each Gradio session gets `{PI_WORKSPACE_DIR}/{session_hash}/`."""
    raw = os.environ.get("PI_SESSION_WORKSPACE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if SESSION_OUTPUT_FOLDER:
        return True
    return is_hf_space_profile()


def sanitize_session_id(raw: str) -> str:
    cleaned = _SESSION_ID_RE.sub("_", (raw or "").strip())[:128].strip("_")
    return cleaned or "default"


def resolve_session_hash(request: gr.Request | None) -> str:
    """Resolve session id using the same rules as the main app."""
    if request is None:
        return "default"
    from tools.gradio_platform import resolve_session_identity

    identity = resolve_session_identity(request)
    return sanitize_session_id(str(identity))


def session_s3_outputs_prefix(session_hash: str) -> str:
    """Session-scoped S3 output prefix (shared env vars with main app)."""
    from tools.gradio_platform import build_s3_outputs_prefix

    return build_s3_outputs_prefix(
        session_hash,
        session_scoped=session_workspace_enabled(),
    )


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


def init_session_workspace(
    request: gr.Request,
) -> tuple[str, gr.FileExplorer, str, str]:
    """
    App-load handler: create the session subfolder and scope the file explorer.

    Returns ``(session_hash, file_explorer_update, status_markdown, s3_output_prefix)``.
    """
    session_hash = resolve_session_hash(request)
    workspace = ensure_session_workspace(session_hash)
    workspace_posix = workspace.as_posix()
    s3_prefix = session_s3_outputs_prefix(session_hash)

    if session_workspace_enabled():
        status = (
            f"**Session id:** `{session_hash}`  \n"
            f"**Your workspace:** `{workspace_posix}/`  \n"
            "_Save all redaction outputs under this folder only._"
        )
    else:
        status = f"**Workspace:** `{workspace_posix}/`"

    return (
        session_hash,
        gr.FileExplorer(root_dir=workspace_posix),
        status,
        s3_prefix,
    )


def workspace_context_prefix(session_hash: str) -> str:
    """Prefix Pi prompts so the agent uses the session workspace."""
    if not session_workspace_enabled() or not session_hash.strip():
        return ""
    root = session_workspace_dir(session_hash).as_posix().rstrip("/")
    return (
        f"**Session workspace (mandatory):** all uploads, downloads, and redaction "
        f"artifacts for this user must live under `{root}/`. "
        f"Use `{root}/redact/<document>/` for per-document output trees. "
        f"Do not read or write other session folders under `{workspace_base_dir().as_posix()}/`.\n\n"
    )
