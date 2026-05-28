"""Browse and download files from the Pi agent shared workspace."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import gradio as gr
from pi_examples import gradio_example_allowed_paths
from session_workspace import (
    WORKSPACE_BASE_DIR,
    session_workspace_dir,
    workspace_base_dir,
)

# Back-compat alias for modules that import WORKSPACE_DIR.
WORKSPACE_DIR = WORKSPACE_BASE_DIR
REFRESH_STUB_DIR = Path(os.environ.get("PI_FILEEXPLORER_STUB_DIR", "/tmp"))

# Folder names under ``.../review/`` where Pass 1 deliverables are saved (see partnership prompt).
_DEFAULT_FINAL_OUTPUT_FOLDER_NAMES = ("output_review_final", "output_final")
_DEFAULT_FINAL_DOWNLOAD_FOLDER = "output_final_download"
_DEFAULT_GRADIO_PREFIX_MIN_LEN = 16


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


def final_download_folder_name() -> str:
    raw = os.environ.get("PI_FINAL_DOWNLOAD_FOLDER", _DEFAULT_FINAL_DOWNLOAD_FOLDER)
    stripped = raw.strip() if raw else ""
    return stripped or _DEFAULT_FINAL_DOWNLOAD_FOLDER


def _gradio_prefix_min_len() -> int:
    raw = os.environ.get(
        "PI_GRADIO_FILENAME_PREFIX_MIN_LEN",
        str(_DEFAULT_GRADIO_PREFIX_MIN_LEN),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_GRADIO_PREFIX_MIN_LEN


def strip_gradio_cache_prefix(filename: str) -> str:
    """
    Remove a leading Gradio cache id prefix (``{alphanumeric}_{name}``).

    Gradio client downloads often prefix filenames with a long hash so repeated
    exports do not collide; users expect the original basename instead.
    """
    pattern = re.compile(rf"^[A-Za-z0-9]{{{_gradio_prefix_min_len()},}}_(.+)$")
    match = pattern.match(filename)
    if match:
        return match.group(1)
    return filename


def _file_created_timestamp(path: Path) -> float:
    stat = path.stat()
    birth = getattr(stat, "st_birthtime", None)
    if birth is not None and birth > 0:
        return float(birth)
    return float(stat.st_mtime)


def _collect_raw_final_output_files(
    session_hash: str | None = None,
) -> list[Path] | None:
    """
    Collect deliverable files from ``review/output_review_final/`` (and aliases)
    anywhere under the session workspace.
    """
    root = workspace_root_from(session_hash)
    if not root.is_dir():
        return None

    download_folder = final_download_folder_name()
    candidates: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or not _is_file_path(path.name):
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if download_folder in relative.parts:
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
    return candidates


def build_final_download_files(
    session_hash: str | None = None,
) -> list[str] | None:
    """
    Stage cleaned deliverables under ``output_final_download/``.

    Copies files from agent final-output folders, strips Gradio cache prefixes,
    deduplicates by basename (newest file wins), and returns paths for ``gr.File``.
    """
    root = workspace_root_from(session_hash)
    raw_files = _collect_raw_final_output_files(session_hash)
    if not raw_files:
        return None

    download_dir = root / final_download_folder_name()
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(raw_files, key=_file_created_timestamp)
    latest_by_name: dict[str, Path] = {}
    for path in ordered:
        latest_by_name[strip_gradio_cache_prefix(path.name)] = path

    staged: list[str] = []
    for name in sorted(latest_by_name):
        source = latest_by_name[name]
        destination = download_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged.append(str(destination.resolve()))
    return staged or None


def collect_final_output_files(
    session_hash: str | None = None,
) -> list[str] | None:
    """Return deduplicated, prefix-stripped deliverables for download and S3 export."""
    return build_final_download_files(session_hash)


def workspace_root_from(session_hash: str | None = None) -> Path:
    """Resolve the session workspace from a sanitized Gradio session hash only."""
    if not session_hash or not str(session_hash).strip():
        return workspace_base_dir().resolve()
    return session_workspace_dir(str(session_hash).strip())


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


def load_workspace_output_files(session_hash: str = ""):
    root = workspace_root_from(session_hash or None)
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
    session_hash: str = "",
) -> tuple[Any, list[str] | None]:
    """Refresh file explorer and auto-detected final deliverables."""
    return (
        load_workspace_output_files(session_hash),
        collect_final_output_files(session_hash),
    )


def workspace_files_download_fn(
    selected: list[str] | None,
    session_hash: str = "",
) -> list[str] | None:
    """Return only file paths under the session workspace (for gr.File download)."""
    if not selected:
        return None
    root = workspace_root_from(session_hash or None)
    downloads: list[str] = []
    for raw in selected:
        if not _is_file_path(raw):
            continue
        resolved = _resolve_under_workspace(raw, workspace_root=root)
        if resolved is not None:
            downloads.append(str(resolved))
    return downloads or None
