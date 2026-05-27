"""Browse and download files from the Pi agent shared workspace."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

WORKSPACE_DIR = Path(os.environ.get("PI_WORKSPACE_DIR", "/home/user/app/workspace"))
# Harmless alternate root used to force FileExplorer refresh (Gradio workaround).
REFRESH_STUB_DIR = Path(os.environ.get("PI_FILEEXPLORER_STUB_DIR", "/tmp"))


def _is_file_path(path: str) -> bool:
    if not path or not path.strip():
        return False
    name = Path(path.rstrip("/\\")).name
    if not name or "." not in name:
        return False
    ext = name.rsplit(".", 1)[-1]
    return bool(ext and len(ext) <= 10 and ext.isalnum())


def _resolve_under_workspace(path: str) -> Path | None:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(WORKSPACE_DIR.resolve())
    except (ValueError, OSError):
        return None
    return resolved if resolved.is_file() else None


def load_workspace_output_files() -> gr.FileExplorer:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return gr.FileExplorer(root_dir=str(WORKSPACE_DIR.resolve()))


def refresh_workspace_output_files_stub() -> gr.FileExplorer:
    return gr.FileExplorer(root_dir=str(REFRESH_STUB_DIR.resolve()))


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio may serve via gr.File (must include the shared workspace)."""
    paths: list[str] = []
    for raw in (
        WORKSPACE_DIR,
        os.environ.get("PI_WORKDIR", "/workspace/doc_redaction"),
        REFRESH_STUB_DIR,
        "/tmp",
    ):
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            continue
        if resolved not in paths:
            paths.append(resolved)
    return paths


def workspace_files_download_fn(selected: list[str] | None) -> list[str] | None:
    """Return only file paths under the workspace (for gr.File download)."""
    if not selected:
        return None
    downloads: list[str] = []
    for raw in selected:
        if not _is_file_path(raw):
            continue
        resolved = _resolve_under_workspace(raw)
        if resolved is not None:
            downloads.append(str(resolved))
    return downloads or None
