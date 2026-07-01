"""Browse and download files from the Pi agent shared workspace."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import gradio as gr
from bootstrap_pi_config import pi_repo_root_path
from pi_examples import gradio_example_allowed_paths
from session_logs import gradio_session_log_allowed_paths
from session_workspace import (
    sanitize_session_id,
    session_workspace_dir,
    workspace_base_dir,
)

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


def final_download_dir(session_hash: str | None = None) -> Path:
    """
    Per-session staging folder for ``gr.File`` downloads.

    Always ``{PI_WORKSPACE_DIR}/{session_id}/output_final_download/`` when a session
    id is known, even if the broader workspace is shared (``PI_SESSION_WORKSPACE=false``).
    """
    base = workspace_base_dir().resolve()
    folder = final_download_folder_name()
    if not session_hash or not str(session_hash).strip():
        return base / folder
    safe_id = sanitize_session_id(str(session_hash))
    return base / safe_id / folder


def _remove_path(path: Path) -> None:
    """Best-effort delete (handles read-only / OneDrive locks on Windows)."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        if not path.exists():
            return
        try:
            os.chmod(path, 0o666)
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _reset_download_dir(download_dir: Path) -> None:
    """Clear staged downloads without removing the directory inode (safer on Windows)."""
    download_dir.mkdir(parents=True, exist_ok=True)
    for child in download_dir.iterdir():
        _remove_path(child)


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
    Stage cleaned deliverables under ``{session_id}/output_final_download/``.

    Copies files from agent final-output folders, strips Gradio cache prefixes,
    deduplicates by basename (newest file wins), and returns paths for ``gr.File``.
    """
    raw_files = _collect_raw_final_output_files(session_hash)
    if not raw_files:
        return None

    download_dir = final_download_dir(session_hash)
    _reset_download_dir(download_dir)

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


_REDACTED_PDF_SUFFIX = "_redacted.pdf"
_REVIEW_PDF_MARKER = "_redactions_for_review"
_PREVIEW_DIRNAME = "preview"
_LEGACY_PREVIEW_DIRNAME = ".pi/preview"
_PREVIEW_FILENAME = "latest_redacted.pdf"
_MIN_PDF_BYTES = 1024


def _is_redacted_pdf_candidate(path: Path) -> bool:
    """True for deliverable ``*_redacted.pdf`` names (not review-only copies)."""
    name = path.name.lower()
    if not name.endswith(_REDACTED_PDF_SUFFIX):
        return False
    if _REVIEW_PDF_MARKER in name:
        return False
    return True


def _is_valid_pdf_file(path: Path, *, min_bytes: int = _MIN_PDF_BYTES) -> bool:
    """Reject empty, partial, or non-PDF files (e.g. HTML error bodies from failed downloads)."""
    try:
        if not path.is_file():
            return False
        if path.stat().st_size < min_bytes:
            return False
        with path.open("rb") as handle:
            header = handle.read(5)
            if not header.startswith(b"%PDF-"):
                return False
            size = path.stat().st_size
            if size < 256:
                handle.seek(max(0, size - 32))
                return b"%%EOF" in handle.read()
            return True
    except OSError:
        return False


def _find_newest_valid_redacted_pdf(session_hash: str | None) -> Path | None:
    """Newest readable ``*_redacted.pdf`` under the session workspace.

    Prefer deliverables under ``review/output_final`` (and aliases) over intermediate
    ``output_redact`` copies so the preview matches the final download.
    """
    root = workspace_root_from(session_hash)
    if not root.is_dir():
        return None

    final_candidates: list[tuple[float, Path]] = []
    other_candidates: list[tuple[float, Path]] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or not _is_redacted_pdf_candidate(path):
                continue
            if not _is_valid_pdf_file(path):
                continue
            try:
                relative = path.resolve(strict=False).relative_to(root.resolve())
            except ValueError:
                continue
            timestamp = _file_created_timestamp(path)
            bucket = (
                final_candidates
                if _is_under_final_output_dir(relative)
                else other_candidates
            )
            bucket.append((timestamp, path))
    except OSError:
        return None

    pool = final_candidates or other_candidates
    if not pool:
        return None
    return max(pool, key=lambda item: item[0])[1]


def _staged_preview_pdf_path(session_hash: str | None) -> Path:
    root = workspace_root_from(session_hash)
    return root / _PREVIEW_DIRNAME / _PREVIEW_FILENAME


def _legacy_staged_preview_pdf_path(session_hash: str | None) -> Path:
    root = workspace_root_from(session_hash)
    return root / _LEGACY_PREVIEW_DIRNAME / _PREVIEW_FILENAME


def _gradio_pdf_path(path: Path) -> str:
    """POSIX absolute path for Gradio File/PDF components (Windows-safe URLs)."""
    return path.resolve().as_posix()


def _stage_preview_pdf(source: Path, session_hash: str | None) -> Path:
    """
    Copy *source* into a stable preview path under the session workspace.

    The Gradio PDF component reads a single file path; staging avoids serving
    files that are still being written in ``output_redact/`` and gives a
    consistent path under ``allowed_paths``.
    """
    dest = _staged_preview_pdf_path(session_hash)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    # copyfile only: copy2/copystat can raise EPERM on OneDrive bind mounts.
    shutil.copyfile(source, tmp)
    tmp.replace(dest)
    return dest.resolve()


def latest_redacted_pdf_path(session_hash: str | None = None) -> str | None:
    """
    Return the newest valid ``*_redacted.pdf`` for the Gradio PDF preview.

    Copies the chosen file to ``{session}/preview/latest_redacted.pdf`` so the
    component always receives a complete PDF under the workspace root.
    """
    source = _find_newest_valid_redacted_pdf(session_hash)
    staged = _staged_preview_pdf_path(session_hash)
    legacy_staged = _legacy_staged_preview_pdf_path(session_hash)
    if source is None:
        for candidate in (staged, legacy_staged):
            if _is_valid_pdf_file(candidate):
                return _gradio_pdf_path(candidate)
        return None

    try:
        if staged.is_file():
            src_mtime = _file_created_timestamp(source)
            staged_mtime = _file_created_timestamp(staged)
            if (
                src_mtime <= staged_mtime
                and staged.stat().st_size == source.stat().st_size
                and _is_valid_pdf_file(staged)
            ):
                return _gradio_pdf_path(staged)
    except OSError:
        pass

    return _gradio_pdf_path(_stage_preview_pdf(source, session_hash))


def preview_pdf_path_for_gradio(session_hash: str | None = None) -> str | None:
    """Return a Gradio-safe preview path, or ``None`` when no valid PDF exists."""
    return latest_redacted_pdf_path(session_hash)


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
        str(pi_repo_root_path()),
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
    for raw in gradio_session_log_allowed_paths():
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
