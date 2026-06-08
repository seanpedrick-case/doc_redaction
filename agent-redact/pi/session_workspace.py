"""Per-session workspace paths for the Pi Gradio UI (mirrors main app session folders)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gradio as gr

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9_@.+-]+")


def workspace_base_dir() -> Path:
    """Shared Pi workspace root (see ``bootstrap_pi_config.ensure_pi_workspace_dir``)."""
    raw = (os.environ.get("PI_WORKSPACE_DIR") or "").strip()
    if raw:
        path = Path(raw)
    else:
        from bootstrap_pi_config import ensure_pi_workspace_dir

        return Path(ensure_pi_workspace_dir(_REPO_ROOT))
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _session_output_folder_enabled() -> bool:
    """Read at call time so ``pi_agent.env`` / dotenv apply before first use."""
    raw = (os.environ.get("SESSION_OUTPUT_FOLDER") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def session_workspace_enabled() -> bool:
    """
    When true, each Gradio session uses ``{PI_WORKSPACE_DIR}/{session_hash}/``.

    Controlled by ``PI_SESSION_WORKSPACE`` in ``config/pi_agent.env`` (default on when unset).
    Set ``PI_SESSION_WORKSPACE=false`` for a single shared workspace root.
    """
    raw = os.environ.get("PI_SESSION_WORKSPACE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if _session_output_folder_enabled():
        return True
    return True


def workspace_base_dir_resolved() -> Path:
    """Current workspace root (never cached at import)."""
    return workspace_base_dir()


def sanitize_session_id(raw: str) -> str:
    cleaned = _SESSION_ID_RE.sub("_", (raw or "").strip())[:128].strip("_")
    return cleaned or "default"


def resolve_session_hash(request: gr.Request | None) -> str:
    """
    Resolve Gradio session id for per-user workspace folders.

    Prefers ``request.session_hash`` (local Pi UI). Falls back to the main app's
    Cognito/OIDC resolver when a deployment header is configured.
    """
    if request is None:
        return "default"
    gradio_hash = getattr(request, "session_hash", None)
    if gradio_hash is not None and str(gradio_hash).strip():
        return sanitize_session_id(str(gradio_hash))
    from tools.gradio_platform import resolve_session_identity

    try:
        identity = resolve_session_identity(request)
    except ValueError:
        return "default"
    return sanitize_session_id(str(identity))


def effective_session_hash(
    session_hash: str,
    request: gr.Request | None = None,
) -> str:
    """
    Use ``session_hash_state`` when set; otherwise resolve from the active request.

    Gradio ``demo.load`` may run before ``request.session_hash`` exists, so handlers
    should pass ``request`` and call this on each event.
    """
    stored = (session_hash or "").strip()
    if stored and stored != "default":
        return sanitize_session_id(stored)
    if request is not None:
        resolved = resolve_session_hash(request)
        if resolved and resolved != "default":
            return resolved
    if stored:
        return sanitize_session_id(stored)
    return "default"


def session_workspace_status_markdown(session_hash: str) -> str:
    """Markdown for the workspace panel."""
    workspace = ensure_session_workspace(session_hash)
    path = workspace.as_posix()
    if session_workspace_enabled():
        return (
            f"**Session id:** `{session_hash}`  \n" f"**Your workspace:** `{path}/`  \n"
        )
    return f"**Workspace:** `{path}/`"


def prepare_session_workspace(
    session_hash: str,
    request: gr.Request | None = None,
) -> tuple[str, Path, str]:
    """
    Resolve session id, create ``{PI_WORKSPACE_DIR}/{hash}/``, return status text.

    Call at the start of redaction (and on page load) so the folder always exists.
    """
    effective = effective_session_hash(session_hash, request)
    workspace = ensure_session_workspace(effective)
    return effective, workspace, session_workspace_status_markdown(effective)


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
    session_hash, workspace, status = prepare_session_workspace("", request)
    s3_prefix = session_s3_outputs_prefix(session_hash)

    return (
        session_hash,
        gr.FileExplorer(root_dir=workspace.as_posix()),
        status,
        s3_prefix,
    )


def workspace_context_prefix(session_hash: str) -> str:
    """Prefix Pi prompts so the agent uses the session workspace."""
    if not session_workspace_enabled() or not session_hash.strip():
        return ""
    root = session_workspace_dir(session_hash).as_posix().rstrip("/")
    lines = [
        f"**Session workspace (mandatory):** all uploads, downloads, and redaction "
        f"artifacts for this user must live under `{root}/`. "
        f"Use `{root}/redact/<document>/output_redact/` for Pass 1 downloads and "
        f"`{root}/redact/<document>/review/output_review_final/` after `/review_apply`. "
        f"Do not write to `{root}/output_final_download/` (UI-managed download copies only). "
        f"Do not read or write other session folders under `{workspace_base_dir().as_posix()}/`.",
    ]
    try:
        from pi_agent_config import uses_split_redaction_backend
        from redaction_prompt import doc_redaction_gradio_url

        if uses_split_redaction_backend():
            helpers = (
                f"{workspace_base_dir().as_posix()}/.pi/helpers/remote_redaction.py"
            )
            lines.append(
                f"**Redaction outputs (split backend):** doc_redaction at "
                f"`{doc_redaction_gradio_url()}` writes to its own container — download "
                f"artifacts into `{root}/redact/<document>/output_redact/` via "
                f"`{helpers}` (`fetch_redaction_files`). Do not `find` or `ls` "
                f"`/workspace/doc_redaction/output` from this agent."
            )
    except ImportError:
        pass
    return "\n".join(lines) + "\n\n"
