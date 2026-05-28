"""Browse and download files from the Pi agent shared workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
from pi_examples import gradio_example_allowed_paths
from session_workspace import WORKSPACE_BASE_DIR, workspace_base_dir

# Back-compat alias for modules that import WORKSPACE_DIR.
WORKSPACE_DIR = WORKSPACE_BASE_DIR
REFRESH_STUB_DIR = Path(os.environ.get("PI_FILEEXPLORER_STUB_DIR", "/tmp"))

# Folder names under ``.../review/`` where Pass 1 deliverables are saved (see partnership prompt).
_DEFAULT_FINAL_OUTPUT_FOLDER_NAMES = ("output_review_final", "output_final")


def final_output_folder_names() -> frozenset[str]:
    raw = os.environ.get("PI_FINAL_OUTPUT_FOLDER_NAMES", "").strip()
    if raw:
        names = {part.strip() for part in raw.split(",") if part.strip()}
        if names:
            return frozenset(names)
    return frozenset(_DEFAULT_FINAL_OUTPUT_FOLDER_NAMES)


def _is_under_final_output_dir(relative_path: Path) -> bool:
    parts = relative_path.parts
    names = final_output_folder_names()
    for index, part in enumerate(parts):
        if part == "review" and index + 1 < len(parts):
            if parts[index + 1] in names:
                return True
    return False


def collect_final_output_files(
    session_workspace: str | None = None,
) -> list[str] | None:
    """
    Collect deliverable files from ``review/output_review_final/`` (and aliases)
    anywhere under the session workspace, newest first.
    """
    root = workspace_root_from(session_workspace)
    if not root.is_dir():
        return None

    candidates: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or not _is_file_path(path.name):
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if not _is_under_final_output_dir(relative):
                continue
            try:
                path.resolve(strict=False).relative_to(root)
            except ValueError:
                continue
            candidates.append(path)
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [str(path.resolve()) for path in candidates]


def workspace_root_from(session_workspace: str | None = None) -> Path:
    if session_workspace and str(session_workspace).strip():
        return Path(session_workspace).resolve()
    return workspace_base_dir().resolve()


def _is_file_path(path: str) -> bool:
    if not path or not path.strip():
        return False
    name = Path(path.rstrip("/\\")).name
    if not name or "." not in name:
        return False
    ext = name.rsplit(".", 1)[-1]
    return bool(ext and len(ext) <= 10 and ext.isalnum())


def _is_safe_workspace_relative_path(path: str) -> bool:
    """Reject absolute paths and traversal segments before joining under workspace."""
    if not path or not path.strip():
        return False
    candidate = Path(path.strip())
    if candidate.is_absolute() or candidate.anchor:
        return False
    return all(part not in ("", ".", "..") for part in candidate.parts)


def _resolve_under_workspace(
    path: str,
    *,
    workspace_root: Path | None = None,
) -> Path | None:
    if not path or not path.strip():
        return None

    root = (workspace_root or workspace_base_dir()).resolve()
    stripped = path.strip()
    try:
        user_path = Path(stripped)
        if user_path.is_absolute():
            # Gradio FileExplorer may return absolute paths already under root_dir.
            resolved = user_path.resolve(strict=False)
        elif _is_safe_workspace_relative_path(stripped):
            resolved = root.joinpath(*user_path.parts).resolve(strict=False)
        else:
            return None
        resolved.relative_to(root)
    except (ValueError, OSError):
        return None
    return resolved if resolved.is_file() else None


def load_workspace_output_files(session_workspace: str = ""):
    root = workspace_root_from(session_workspace or None)
    root.mkdir(parents=True, exist_ok=True)
    return gr.FileExplorer(root_dir=str(root))


def refresh_workspace_output_files_stub():
    return gr.FileExplorer(root_dir=str(REFRESH_STUB_DIR.resolve()))


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio may serve via gr.File (must include the shared workspace)."""
    paths: list[str] = []
    for raw in (
        workspace_base_dir(),
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
    for raw in gradio_example_allowed_paths():
        if raw not in paths:
            paths.append(raw)
    return paths


def refresh_workspace_panel(
    session_workspace: str = "",
) -> tuple[Any, list[str] | None]:
    """Refresh file explorer and auto-detected final deliverables."""
    return (
        load_workspace_output_files(session_workspace),
        collect_final_output_files(session_workspace),
    )


def workspace_files_download_fn(
    selected: list[str] | None,
    session_workspace: str = "",
) -> list[str] | None:
    """Return only file paths under the session workspace (for gr.File download)."""
    if not selected:
        return None
    root = workspace_root_from(session_workspace or None)
    downloads: list[str] = []
    for raw in selected:
        if not _is_file_path(raw):
            continue
        resolved = _resolve_under_workspace(raw, workspace_root=root)
        if resolved is not None:
            downloads.append(str(resolved))
    return downloads or None
