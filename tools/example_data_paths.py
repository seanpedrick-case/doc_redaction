"""Resolve bundled Gradio example file locations (repo, package, Docker)."""

from __future__ import annotations

import filecmp
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def resolve_example_data_dirs() -> list[Path]:
    """Return all existing ``example_data`` roots (may be multiple layouts)."""
    roots: list[Path] = []

    def _add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)

    try:
        import doc_redaction as _doc_redaction_pkg

        _add(Path(_doc_redaction_pkg.__file__).resolve().parent / "example_data")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    for base in (repo_root, cwd):
        for rel in ("doc_redaction/example_data", "example_data"):
            _add(base / rel)

    return roots


@lru_cache(maxsize=256)
def bundled_example_candidate_path(file_name: str) -> Path | None:
    """Resolved path to a bundled example file by basename, if it exists."""
    if not file_name or file_name in {".", ".."}:
        return None
    for root in resolve_example_data_dirs():
        candidate = (root / file_name).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
        name_only = Path(file_name).name
        if name_only != file_name:
            direct = (root / name_only).resolve()
            try:
                direct.relative_to(root.resolve())
                if direct.is_file():
                    return direct
            except ValueError:
                pass
        for match in root.rglob(name_only):
            if not match.is_file():
                continue
            try:
                match.relative_to(root.resolve())
            except ValueError:
                continue
            return match.resolve()
    return None


def is_lfs_pointer(path: Path | str) -> bool:
    """True when ``path`` is a Git LFS pointer stub, not real file content."""
    try:
        p = Path(path)
        first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except (OSError, IndexError):
        return False
    return first_line.startswith("version https://git-lfs.github.com/spec/v1")


def bundled_example_asset_available(path: str) -> bool:
    """True when ``path`` is a readable bundled asset (exists and not an LFS pointer)."""
    if not path or not str(path).strip():
        return False
    try:
        resolved = Path(os.path.abspath(path)).resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    return not is_lfs_pointer(resolved)


def gradio_example_allowed_paths() -> list[str]:
    """Directories to pass to Gradio ``allowed_paths`` for bundled demo files."""
    return [str(root.resolve()) for root in resolve_example_data_dirs()]


def resolve_gradio_example_file_path(local_path: str) -> str | None:
    """
    Map Gradio ``/tmp/gradio*`` example paths back to stable ``example_data`` files.

    Gradio ``Examples`` copies demos into a content-hash cache that may disappear on
    Hugging Face; bundled paths under ``example_data`` remain valid for postprocess.
    """
    if not local_path or not str(local_path).strip():
        return None
    try:
        abspath = Path(os.path.abspath(local_path)).resolve()
    except OSError:
        abspath = None

    if abspath is not None and abspath.is_file() and not is_lfs_pointer(abspath):
        if is_trusted_bundled_example_path(str(abspath)):
            return str(abspath)
        bundled = bundled_example_candidate_path(abspath.name)
        if bundled is not None and not is_lfs_pointer(bundled):
            return str(bundled.resolve())
        return str(abspath)

    bundled = bundled_example_candidate_path(Path(local_path).name)
    if bundled is not None and bundled.is_file() and not is_lfs_pointer(bundled):
        return str(bundled.resolve())
    return None


def resolve_gradio_example_file_paths(file_input: Any) -> list[str]:
    """Normalize a Gradio File/State value to stable bundled example paths."""
    from tools.malware_scan import normalize_gradio_file_paths

    resolved: list[str] = []
    for raw in normalize_gradio_file_paths(file_input):
        path = resolve_gradio_example_file_path(raw)
        if path:
            resolved.append(path)
    return resolved


def is_trusted_bundled_example_path(local_path: str) -> bool:
    """
    True when ``local_path`` is a file under a resolved bundled ``example_data`` dir.

    Used to skip GuardDuty malware staging for Gradio ``Examples`` demo files shipped
    with the app (not end-user uploads).
    """
    if not local_path or not str(local_path).strip():
        return False
    try:
        abspath = Path(os.path.abspath(local_path)).resolve()
    except OSError:
        return False
    if not abspath.is_file():
        return False
    for root in resolve_example_data_dirs():
        try:
            abspath.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def is_bundled_example_file(local_path: str) -> bool:
    """
    True for bundled demo files, including Gradio temp copies of ``example_data`` assets.

    ``gr.Examples`` often copies files into ``/tmp/gradio*`` with a UUID prefix on the
    basename; those paths are accepted when the bytes match any file under
    ``example_data``.
    """
    if is_trusted_bundled_example_path(local_path):
        return True
    try:
        abspath = Path(os.path.abspath(local_path)).resolve()
    except OSError:
        return False
    if not abspath.is_file():
        return False

    basename = abspath.name
    candidate = bundled_example_candidate_path(basename)
    if candidate is not None:
        try:
            if filecmp.cmp(abspath, candidate, shallow=False):
                return True
        except OSError:
            pass

    for root in resolve_example_data_dirs():
        for bundled in root.iterdir():
            if not bundled.is_file():
                continue
            if basename != bundled.name and not basename.endswith(f"_{bundled.name}"):
                continue
            try:
                if filecmp.cmp(abspath, bundled, shallow=False):
                    return True
            except OSError:
                continue
    return False
