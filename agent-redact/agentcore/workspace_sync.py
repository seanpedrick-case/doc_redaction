"""Sync session workspace files to/from AgentCore invoke payloads."""

from __future__ import annotations

import base64
import os
from pathlib import Path

_DEFAULT_MAX_BYTES = 8 * 1024 * 1024


def max_workspace_sync_bytes() -> int:
    raw = (os.environ.get("AGENTCORE_MAX_UPLOAD_BYTES") or "").strip()
    if raw.isdigit():
        return int(raw)
    return _DEFAULT_MAX_BYTES


def _ensure_session_workspace(session_hash: str | None) -> Path:
    try:
        from bundle_support.session_workspace import ensure_session_workspace
    except ImportError:
        from pi.session_workspace import (
            ensure_session_workspace,  # type: ignore[no-redef]
        )

    return ensure_session_workspace(session_hash or "")


def apply_workspace_files(session_hash: str | None, files: list[dict]) -> list[str]:
    """Write base64-encoded files into the session workspace. Returns relative paths written."""
    if not files:
        return []
    root = _ensure_session_workspace(session_hash).resolve()
    written: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("relative_path") or item.get("name") or "").strip()
        encoded = str(item.get("content_base64") or "").strip()
        if not relative or not encoded:
            continue
        dest = (root / relative).resolve()
        try:
            dest.relative_to(root)
        except ValueError:
            continue
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) > max_workspace_sync_bytes():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        written.append(str(dest.relative_to(root)).replace("\\", "/"))
    return written


def collect_workspace_files_for_sync(
    session_hash: str | None,
    *,
    prefix: str = "redact/",
) -> list[dict[str, str]]:
    """Collect workspace files under *prefix* for download to the Gradio client."""
    root = _ensure_session_workspace(session_hash).resolve()
    if not root.is_dir():
        return []
    limit = max_workspace_sync_bytes()
    out: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if prefix and not rel.startswith(prefix.lstrip("/")):
            continue
        size = path.stat().st_size
        if size > limit:
            continue
        out.append(
            {
                "relative_path": rel,
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return out
