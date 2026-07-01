"""Curated LangGraph tools for doc_redaction orchestration (no shell)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_PI_DIR = Path(__file__).resolve().parents[1] / "pi"
if str(_PI_DIR) not in sys.path:
    sys.path.insert(0, str(_PI_DIR))

from remote_redaction import (  # noqa: E402
    call_doc_redact,
    extract_server_paths,
    fetch_redaction_files,
    make_redaction_client,
)
from session_workspace import session_workspace_dir  # noqa: E402

_MAX_TEXT_BYTES = int(os.environ.get("LANGGRAPH_MAX_WORKSPACE_TEXT_BYTES", "1500000"))
_MAX_SCRIPT_SECONDS = int(os.environ.get("LANGGRAPH_WORKSPACE_SCRIPT_TIMEOUT", "300"))


def _session_root(session_hash: str | None) -> Path:
    if session_hash:
        return session_workspace_dir(session_hash)
    from session_workspace import workspace_base_dir

    return workspace_base_dir()


_MAX_TEXT_BYTES = int(os.environ.get("LANGGRAPH_MAX_WORKSPACE_TEXT_BYTES", "1500000"))
_MAX_SCRIPT_SECONDS = int(os.environ.get("LANGGRAPH_WORKSPACE_SCRIPT_TIMEOUT", "300"))
_TOOL_ARG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOC_REDACT_PDF_KEYS = (
    "pdf_relative_path",
    "pdf_path",
    "pdf",
    "document_file",
)
_DOC_REDACT_DEST_KEYS = (
    "dest_relative_dir",
    "dest_dir",
    "dest",
    "output_dir",
)
_SCRIPT_PATH_KEYS = (
    "relative_path",
    "path",
    "script",
    "script_path",
    "file",
    "filename",
)
_PATH_ONLY_TOOL_KEYS = frozenset(_SCRIPT_PATH_KEYS)


def _merge_tool_arg_dicts(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _sanitize_tool_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop hallucinated tool-arg keys from weak local models (URLs, JSON fragments)."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str) and _TOOL_ARG_KEY_RE.fullmatch(key):
            clean[key] = value
    return clean


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _default_dest_for_pdf(pdf_relative_path: str) -> str:
    stem = Path(pdf_relative_path.replace("\\", "/")).stem
    return f"redact/{stem or 'document'}/output_redact"


def _default_review_apply_dest_for_pdf(pdf_relative_path: str) -> str:
    stem = Path(pdf_relative_path.replace("\\", "/")).stem
    return f"redact/{stem or 'document'}/review/output_review_final"


def _default_review_apply_dest_for_review_csv(review_csv_relative_path: str) -> str:
    normalized = review_csv_relative_path.replace("\\", "/")
    parts = Path(normalized).parts
    if "output_redact" in parts:
        idx = parts.index("output_redact")
        doc = Path(*parts[:idx])
        return str(doc / "review" / "output_review_final").replace("\\", "/")
    return _default_review_apply_dest_for_pdf(review_csv_relative_path)


_OUTPUT_FILE_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".csv",
        ".json",
        ".txt",
        ".py",
        ".zip",
        ".png",
        ".jpg",
        ".jpeg",
        ".xlsx",
    }
)


def _looks_like_file_relative_path(rel: str) -> bool:
    ext = Path(rel.replace("\\", "/")).suffix.lower()
    return bool(ext) and ext in _OUTPUT_FILE_EXTENSIONS


def _ensure_workspace_output_dir(
    session_hash: str | None,
    dest_relative_dir: Any,
    *,
    pdf_relative_path: str | None = None,
    review_csv_relative_path: str | None = None,
    default_for: str = "doc_redact",
) -> Path:
    """
    Resolve an output directory under the session workspace.

    Weak local models often pass the PDF path (or another file) as dest_relative_dir;
    on Windows ``Path.mkdir()`` then raises WinError 183 when that path is an existing file.
    """
    rel = ""
    if dest_relative_dir is not None and dest_relative_dir != "":
        try:
            rel = _coerce_relative_path(dest_relative_dir, label="dest_relative_dir")
        except ValueError:
            rel = ""

    pdf_rel = ""
    if pdf_relative_path:
        try:
            pdf_rel = _coerce_relative_path(
                pdf_relative_path, label="pdf_relative_path"
            )
        except ValueError:
            pdf_rel = str(pdf_relative_path).strip().replace("\\", "/")

    review_rel = ""
    if review_csv_relative_path:
        try:
            review_rel = _coerce_relative_path(
                review_csv_relative_path, label="review_csv_relative_path"
            )
        except ValueError:
            review_rel = str(review_csv_relative_path).strip().replace("\\", "/")

    if not rel or _looks_like_file_relative_path(rel):
        if default_for == "review_apply":
            if review_rel:
                rel = _default_review_apply_dest_for_review_csv(review_rel)
            elif pdf_rel:
                rel = _default_review_apply_dest_for_pdf(pdf_rel)
        elif pdf_rel:
            rel = _default_dest_for_pdf(pdf_rel)

    if not rel:
        raise ValueError(
            "dest_relative_dir must be an output directory path, not a document file."
        )

    candidate = _resolve_workspace_path(session_hash, rel)
    if candidate.is_file():
        if default_for == "review_apply":
            rel = (
                _default_review_apply_dest_for_review_csv(review_rel)
                if review_rel
                else _default_review_apply_dest_for_pdf(pdf_rel or candidate.name)
            )
        else:
            rel = _default_dest_for_pdf(pdf_rel or candidate.name)
        candidate = _resolve_workspace_path(session_hash, rel)

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _coerce_relative_path(value: Any, *, label: str = "path") -> str:
    """
    Normalize tool path arguments.

    Local OpenAI-compatible models sometimes emit nested dicts or pass the full
    tool-args object as a single value; ``Path / dict`` then fails at runtime.
    """
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        payload = _sanitize_tool_dict(value)
        text = _first_string(
            payload,
            (
                label,
                "relative_path",
                "path",
                *_DOC_REDACT_PDF_KEYS,
                *_DOC_REDACT_DEST_KEYS,
                "review_csv_relative_path",
                "redacted_pdf_relative_path",
                "ocr_words_csv_relative_path",
                "script",
                "script_path",
                "file",
                "filename",
                "value",
            ),
        )
        if not text and len(payload) == 1:
            return _coerce_relative_path(next(iter(payload.values())), label=label)
        if not text:
            for key in ("relative_path", label, "path"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    return _coerce_relative_path(nested, label=label)
        if not text:
            raise ValueError(f"Tool {label} must be a string path, got dict: {value!r}")
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_relative_path(value[0], label=label)
    else:
        text = str(value).strip()
    if not text:
        raise ValueError(f"Tool {label} is empty.")
    return text.replace("\\", "/")


def _coerce_tool_text_content(value: Any, *, label: str = "content") -> str:
    """Normalize write_workspace_text body from messy local-model tool calls."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, dict):
        for key in (label, "content", "text", "body", "data", "source"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
            if isinstance(nested, dict):
                return _coerce_tool_text_content(nested, label=label)
        str_values = [item for item in value.values() if isinstance(item, str)]
        if len(str_values) > 1:
            return max(str_values, key=len)
        if len(str_values) == 1:
            return str_values[0]
        payload = _sanitize_tool_dict(value)
        for key in (label, "content", "text", "body", "script", "data", "source"):
            nested = payload.get(key)
            if isinstance(nested, str):
                return nested
            if isinstance(nested, dict):
                return _coerce_tool_text_content(nested, label=label)
        str_values = [item for item in payload.values() if isinstance(item, str)]
        if len(str_values) == 1:
            return str_values[0]
        if len(payload) == 1:
            return _coerce_tool_text_content(next(iter(payload.values())), label=label)
        raise ValueError(f"Tool {label} must be text, got dict: {value!r}")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_tool_text_content(value[0], label=label)
    raise ValueError(
        f"Tool {label} must be text, got {type(value).__name__}: {value!r}"
    )


def _should_resolve_script_path(payload: dict[str, Any], rel_raw: str) -> bool:
    """Only remap bare script names; leave explicit paths and non-.py files alone."""
    if _first_string(payload, ("script", "script_path")):
        return True
    rel = rel_raw.replace("\\", "/")
    if "/" in rel:
        return False
    name = Path(rel).name
    if name.lower().endswith(".py"):
        return True
    return "." not in name


def _parse_write_workspace_text_input(
    relative_path: Any,
    content: Any,
) -> tuple[str, str]:
    """Merge/normalize write_workspace_text args from messy local-model tool calls."""
    merged = _merge_tool_arg_dicts(relative_path, content)
    payload = _sanitize_tool_dict(merged)

    rel_raw = _first_string(payload, _SCRIPT_PATH_KEYS)
    if not rel_raw:
        nested = payload.get("relative_path")
        if isinstance(nested, dict):
            rel_raw = _coerce_relative_path(nested, label="relative_path")
    if not rel_raw and isinstance(relative_path, str):
        rel_raw = relative_path.strip()
    if not rel_raw:
        raise ValueError(
            "write_workspace_text requires relative_path or script (e.g. fix_policy.py)."
        )
    rel_raw = rel_raw.replace("\\", "/")

    content_raw: Any = merged.get("content")
    if isinstance(content_raw, dict):
        content_raw = _coerce_tool_text_content(content_raw)
    if content_raw is None and isinstance(content, str):
        content_raw = content
    if content_raw is None:
        for key, value in merged.items():
            if key in _PATH_ONLY_TOOL_KEYS:
                continue
            if isinstance(value, dict):
                content_raw = _coerce_tool_text_content(value)
                break
            content_raw = value
            break
    if content_raw is None:
        raise ValueError("write_workspace_text requires content text.")
    return rel_raw, _coerce_tool_text_content(content_raw)


def _resolve_script_relative_path(session_hash: str | None, script: str) -> str:
    """Map a script filename or relative path to a workspace-relative .py path."""
    rel = script.replace("\\", "/").strip()
    if "/" in rel:
        return rel
    name = Path(rel).name
    if not name.lower().endswith(".py"):
        name = f"{name}.py" if name else "fix_policy.py"
    root = _session_root(session_hash).resolve()
    matches = sorted(
        (path for path in root.rglob(name) if path.is_file()),
        key=lambda path: len(path.relative_to(root).parts),
    )
    if matches:
        return str(matches[0].relative_to(root)).replace("\\", "/")
    output_dirs = sorted(
        (path for path in root.rglob("output_redact") if path.is_dir()),
        key=lambda path: len(path.relative_to(root).parts),
    )
    if output_dirs:
        target = output_dirs[0]
        return str((target / name).relative_to(root)).replace("\\", "/")
    return f"scripts/{name}"


def _parse_doc_redact_tool_input(
    pdf_relative_path: Any,
    dest_relative_dir: Any | None,
    *,
    ocr_method: str | None,
    pii_method: str | None,
) -> tuple[str, str, str | None, str | None]:
    """Merge/normalize doc_redact tool args from messy local-model tool calls."""
    payload: dict[str, Any] = {}
    for value in (pdf_relative_path, dest_relative_dir):
        if isinstance(value, dict):
            payload.update(_sanitize_tool_dict(value))

    pdf_raw = _first_string(payload, _DOC_REDACT_PDF_KEYS)
    if not pdf_raw and isinstance(pdf_relative_path, str):
        pdf_raw = pdf_relative_path.strip()
    if not pdf_raw:
        raise ValueError(
            "doc_redact requires a PDF path (pdf_relative_path or pdf_path)."
        )
    pdf_rel = _coerce_relative_path(pdf_raw, label="pdf_relative_path")

    dest_raw = _first_string(payload, _DOC_REDACT_DEST_KEYS)
    if not dest_raw and isinstance(dest_relative_dir, str):
        dest_raw = dest_relative_dir.strip()
    dest_rel = (
        _coerce_relative_path(dest_raw, label="dest_relative_dir")
        if dest_raw
        else _default_dest_for_pdf(pdf_rel)
    )

    ocr = ocr_method or _first_string(payload, ("ocr_method",)) or None
    pii = pii_method or _first_string(payload, ("pii_method",)) or None
    return pdf_rel, dest_rel, ocr, pii


def _parse_review_apply_tool_input(
    pdf_relative_path: Any,
    review_csv_relative_path: Any,
    dest_relative_dir: Any | None,
) -> tuple[str, str, str]:
    """Merge/normalize review_apply tool args from messy local-model tool calls."""
    payload: dict[str, Any] = {}
    for value in (pdf_relative_path, review_csv_relative_path, dest_relative_dir):
        if isinstance(value, dict):
            payload.update(_sanitize_tool_dict(value))

    pdf_raw = _first_string(payload, _DOC_REDACT_PDF_KEYS)
    if not pdf_raw and isinstance(pdf_relative_path, str):
        pdf_raw = pdf_relative_path.strip()
    if not pdf_raw:
        raise ValueError(
            "review_apply requires a PDF path (pdf_relative_path or pdf_path)."
        )
    pdf_rel = _coerce_relative_path(pdf_raw, label="pdf_relative_path")

    review_raw = _first_string(
        payload,
        (
            "review_csv_relative_path",
            "review_csv",
            "csv_path",
            "csv",
            "review_file",
        ),
    )
    if not review_raw and isinstance(review_csv_relative_path, str):
        review_raw = review_csv_relative_path.strip()
    if not review_raw:
        raise ValueError(
            "review_apply requires a review CSV path (review_csv_relative_path)."
        )
    review_rel = _coerce_relative_path(review_raw, label="review_csv_relative_path")

    dest_raw = _first_string(payload, _DOC_REDACT_DEST_KEYS)
    if not dest_raw and isinstance(dest_relative_dir, str):
        dest_raw = dest_relative_dir.strip()
    dest_rel = (
        _coerce_relative_path(dest_raw, label="dest_relative_dir") if dest_raw else ""
    )
    return pdf_rel, review_rel, dest_rel


def _resolve_workspace_path(session_hash: str | None, relative_path: Any) -> Path:
    rel = _coerce_relative_path(relative_path)
    root = _session_root(session_hash).resolve()
    candidate = (root / rel).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"Path escapes session workspace: {rel}")
    return candidate


def _resolve_workspace_pdf(session_hash: str | None, pdf_relative_path: str) -> Path:
    """Resolve a PDF under the session workspace; fall back to unique basename match."""
    try:
        candidate = _resolve_workspace_path(session_hash, pdf_relative_path)
        if candidate.is_file():
            return candidate
    except ValueError:
        candidate = None

    root = _session_root(session_hash).resolve()
    basename = Path(pdf_relative_path.replace("\\", "/")).name
    if not basename:
        raise FileNotFoundError(f"PDF not found in workspace: {pdf_relative_path}")
    matches = sorted(
        (path for path in root.rglob(basename) if path.is_file()),
        key=lambda path: len(path.relative_to(root).parts),
    )
    if not matches:
        missing = candidate or (root / pdf_relative_path)
        raise FileNotFoundError(f"PDF not found in workspace: {missing}")
    if len(matches) > 1:
        rels = [str(path.relative_to(root)).replace("\\", "/") for path in matches[:5]]
        raise ValueError(
            "Multiple PDFs match "
            f"{basename!r} in the workspace; use a relative path. Matches: {rels}"
        )
    return matches[0].resolve()


def list_workspace_files(session_hash: str | None = None) -> str:
    """List files under the current session workspace."""
    root = _session_root(session_hash)
    if not root.is_dir():
        return json.dumps({"files": [], "root": str(root)})
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root)).replace("\\", "/"))
    return json.dumps({"root": str(root), "files": files[:500]})


def run_doc_redact(
    pdf_relative_path: str,
    dest_relative_dir: str,
    *,
    session_hash: str | None = None,
    ocr_method: str | None = None,
    pii_method: str | None = None,
    deny_list: list[str] | None = None,
    allow_list: list[str] | None = None,
) -> str:
    """Run Pass 1 redaction via /doc_redact and download artifacts into the session workspace."""
    pdf_rel, dest_rel, ocr_from_tool, pii_from_tool = _parse_doc_redact_tool_input(
        pdf_relative_path,
        dest_relative_dir,
        ocr_method=ocr_method,
        pii_method=pii_method,
    )
    pdf = _resolve_workspace_pdf(session_hash, pdf_rel)
    dest = _ensure_workspace_output_dir(
        session_hash,
        dest_rel,
        pdf_relative_path=pdf_rel,
        default_for="doc_redact",
    )
    result, saved = call_doc_redact(
        pdf,
        dest,
        ocr_method=ocr_from_tool or os.environ.get("PI_DEFAULT_OCR_METHOD"),
        pii_method=pii_from_tool or os.environ.get("PI_DEFAULT_PII_METHOD"),
        deny_list=deny_list,
        allow_list=allow_list,
    )
    message = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else ""
    payload = {
        "message": str(message or "doc_redact completed."),
        "saved_paths": [str(p) for p in saved],
        "server_paths": extract_server_paths(result),
    }
    return json.dumps(payload, indent=2)


def _discover_ocr_words_csv(review_csv: Path) -> Path | None:
    """Find the word-level OCR CSV sibling of a *_review_file.csv."""
    parent = review_csv.parent
    review_csv.name.lower()
    patterns = (
        "*word*ocr*.csv",
        "*ocr*word*.csv",
        "*_words.csv",
        "*words*.csv",
    )
    for pattern in patterns:
        for candidate in sorted(parent.glob(pattern)):
            if candidate.resolve() == review_csv.resolve():
                continue
            if "_review_file" in candidate.name.lower():
                continue
            return candidate
    for candidate in sorted(parent.glob("*.csv")):
        if candidate.resolve() == review_csv.resolve():
            continue
        name = candidate.name.lower()
        if "_review_file" in name:
            continue
        if "word" in name or "ocr" in name:
            return candidate
    return None


def read_workspace_text(
    relative_path: Any,
    *,
    session_hash: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Read a UTF-8 text file from the session workspace (CSV, JSON, Python script)."""
    try:
        rel = _coerce_relative_path(relative_path, label="relative_path")
        path = _resolve_workspace_path(session_hash, rel)
    except ValueError as exc:
        return json.dumps({"error": str(exc), "relative_path": str(relative_path)})
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc), "relative_path": str(relative_path)})
    if not path.is_file():
        return json.dumps({"error": f"File not found: {rel}"})
    limit = max_bytes if max_bytes is not None else _MAX_TEXT_BYTES
    size = path.stat().st_size
    if size > limit:
        return json.dumps(
            {
                "error": (
                    f"File too large to read ({size} bytes > {limit}). "
                    "Use run_workspace_python_script on a .py file instead."
                )
            }
        )
    text = path.read_text(encoding="utf-8-sig")
    max_lines = int(os.environ.get("LANGGRAPH_READ_CSV_MAX_LINES", "60"))
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv"):
        lines = text.splitlines()
        if len(lines) > max_lines:
            preview = "\n".join(lines[:max_lines])
            return (
                f"CSV preview for {rel} (lines 1-{max_lines} of {len(lines)}). "
                "Edit the full file with write_workspace_text or run_workspace_python_script.\n\n"
                f"{preview}"
            )
    return text


def write_workspace_text(
    relative_path: Any,
    content: Any,
    *,
    session_hash: str | None = None,
) -> str:
    """Write UTF-8 text into the session workspace (preserve utf-8-sig for review CSVs)."""
    try:
        merged = _merge_tool_arg_dicts(relative_path, content)
        rel, body = _parse_write_workspace_text_input(relative_path, content)
        if _should_resolve_script_path(_sanitize_tool_dict(merged), rel):
            rel = _resolve_script_relative_path(session_hash, rel)
        path = _resolve_workspace_path(session_hash, rel)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    if len(body.encode("utf-8")) > _MAX_TEXT_BYTES:
        return json.dumps({"error": f"Content too large (>{_MAX_TEXT_BYTES} bytes)."})
    path.parent.mkdir(parents=True, exist_ok=True)
    unchanged = False
    if path.is_file():
        try:
            unchanged = path.read_text(encoding="utf-8-sig") == body
        except OSError:
            unchanged = False
    if not unchanged:
        path.write_text(body, encoding="utf-8-sig")
    root = _session_root(session_hash)
    rel_written = str(path.relative_to(root)).replace("\\", "/")
    payload: dict[str, Any] = {
        "written": rel_written,
        "bytes": path.stat().st_size,
    }
    if unchanged:
        payload["unchanged"] = True
    if path.suffix.lower() == ".py":
        payload["next_step"] = (
            "Script already saved. Call run_workspace_python_script with "
            f"relative_path={rel_written!r} now — do not call write_workspace_text "
            "again unless the script body must change."
        )
    return json.dumps(payload)


def run_workspace_python_script(
    relative_path: Any,
    content: Any = None,
    *,
    session_hash: str | None = None,
) -> str:
    """Run a Python script already saved under the session workspace."""
    merged = _merge_tool_arg_dicts(relative_path, content)
    written_path: str | None = None
    if isinstance(merged.get("content"), str):
        write_out = write_workspace_text(
            relative_path, content, session_hash=session_hash
        )
        write_payload = json.loads(write_out)
        if write_payload.get("error"):
            return write_out
        written_path = write_payload.get("written")
    try:
        if written_path:
            rel = written_path
        else:
            payload = _sanitize_tool_dict(merged)
            rel = _first_string(payload, _SCRIPT_PATH_KEYS)
            if not rel:
                nested = payload.get("relative_path")
                if isinstance(nested, dict):
                    rel = _coerce_relative_path(nested, label="relative_path")
            if not rel and not isinstance(relative_path, dict):
                rel = _coerce_relative_path(relative_path, label="relative_path")
            if not rel:
                raise ValueError(
                    "run_workspace_python_script requires relative_path or script "
                    "(e.g. fix_policy.py)."
                )
            rel = rel.replace("\\", "/")
            if _should_resolve_script_path(payload, rel):
                rel = _resolve_script_relative_path(session_hash, rel)
        path = _resolve_workspace_path(session_hash, rel)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    if path.suffix.lower() != ".py":
        return json.dumps({"error": "Only .py scripts are allowed."})
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(path.parent),
        capture_output=True,
        text=True,
        timeout=_MAX_SCRIPT_SECONDS,
        check=False,
    )
    return json.dumps(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        },
        indent=2,
    )


_REVIEW_APPROVED: dict[str, bool] = {}


def approve_review_apply(session_hash: str | None = None) -> str:
    """Mark review_apply as approved for human-in-the-loop gating."""
    key = session_hash or ""
    _REVIEW_APPROVED[key] = True
    return json.dumps({"approved": True, "session": key})


def run_review_apply(
    pdf_relative_path: str,
    review_csv_relative_path: str,
    dest_relative_dir: str,
    *,
    session_hash: str | None = None,
) -> str:
    """Apply an edited review CSV via /review_apply and download outputs."""
    if os.environ.get("LANGGRAPH_REQUIRE_REVIEW_APPROVAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        key = session_hash or ""
        if not _REVIEW_APPROVED.pop(key, False):
            return json.dumps(
                {
                    "error": (
                        "Human approval required before review_apply. "
                        "Set LANGGRAPH_REQUIRE_REVIEW_APPROVAL=false to disable, or call "
                        "approve_review_apply first."
                    )
                }
            )
    from gradio_client import handle_file

    try:
        pdf_rel, review_rel, dest_rel = _parse_review_apply_tool_input(
            pdf_relative_path,
            review_csv_relative_path,
            dest_relative_dir,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    pdf = _resolve_workspace_pdf(session_hash, pdf_rel)
    review_csv = _resolve_workspace_path(session_hash, review_rel)
    dest = _ensure_workspace_output_dir(
        session_hash,
        dest_rel,
        pdf_relative_path=pdf_rel,
        review_csv_relative_path=review_rel,
        default_for="review_apply",
    )

    client = make_redaction_client()
    result = client.predict(
        api_name="/review_apply",
        pdf_file=handle_file(str(pdf)),
        review_csv_file=handle_file(str(review_csv)),
    )
    server_paths = extract_server_paths(result)
    saved = fetch_redaction_files(server_paths, dest)
    message = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else ""
    return json.dumps(
        {
            "message": str(message or "review_apply completed."),
            "saved_paths": [str(p) for p in saved],
            "server_paths": server_paths,
        },
        indent=2,
    )


def run_verify_coverage(
    review_csv_relative_path: str,
    *,
    session_hash: str | None = None,
    redacted_pdf_relative_path: str | None = None,
    ocr_words_csv_relative_path: str | None = None,
    must_redact: list[str] | None = None,
    must_not_redact: list[str] | None = None,
) -> str:
    """Run Pass 1 coverage verification on workspace-local CSV/PDF paths."""
    from redaction_langgraph.verify_coverage_lib import verify_redaction_coverage

    review_csv = _resolve_workspace_path(session_hash, review_csv_relative_path)
    if ocr_words_csv_relative_path:
        ocr_words_csv = _resolve_workspace_path(
            session_hash, ocr_words_csv_relative_path
        )
    else:
        discovered = _discover_ocr_words_csv(review_csv)
        if discovered is None:
            return json.dumps(
                {
                    "error": (
                        "Could not find word-level OCR CSV beside the review CSV. "
                        "Pass ocr_words_csv_relative_path explicitly."
                    ),
                    "review_csv": str(review_csv),
                }
            )
        ocr_words_csv = discovered
    redacted_pdf = (
        _resolve_workspace_path(session_hash, redacted_pdf_relative_path)
        if redacted_pdf_relative_path
        else None
    )
    try:
        report = verify_redaction_coverage(
            review_csv,
            ocr_words_csv,
            must_redact=must_redact,
            must_not_redact=must_not_redact,
            redacted_pdf_path=redacted_pdf,
        )
    except (ValueError, re.error) as exc:
        return json.dumps(
            {
                "error": str(exc),
                "review_csv": str(review_csv),
                "ocr_words_csv": str(ocr_words_csv),
            },
            indent=2,
        )
    payload = report.to_dict()
    payload["ocr_words_csv"] = str(ocr_words_csv)
    return json.dumps(payload, indent=2, default=str)


def build_langgraph_tools(session_hash: str | None):
    """Return LangChain tools bound to *session_hash* workspace."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            name="list_workspace_files",
            description="List files in the current session workspace.",
            func=lambda: list_workspace_files(session_hash),
        ),
        StructuredTool.from_function(
            name="doc_redact",
            description=(
                "Run initial document redaction (Pass 1) via /doc_redact. "
                "Paths are relative to the session workspace."
            ),
            func=lambda pdf_relative_path, dest_relative_dir, ocr_method=None, pii_method=None: run_doc_redact(
                pdf_relative_path,
                dest_relative_dir,
                session_hash=session_hash,
                ocr_method=ocr_method,
                pii_method=pii_method,
            ),
        ),
        StructuredTool.from_function(
            name="approve_review_apply",
            description="Approve review_apply when LANGGRAPH_REQUIRE_REVIEW_APPROVAL is enabled.",
            func=lambda: approve_review_apply(session_hash),
        ),
        StructuredTool.from_function(
            name="review_apply",
            description=(
                "Apply an edited *_review_file.csv to the source PDF via /review_apply. "
                "Paths are relative to the session workspace."
            ),
            func=lambda pdf_relative_path, review_csv_relative_path, dest_relative_dir: run_review_apply(
                pdf_relative_path,
                review_csv_relative_path,
                dest_relative_dir,
                session_hash=session_hash,
            ),
        ),
        StructuredTool.from_function(
            name="verify_coverage",
            description=(
                "Verify Pass 1 redaction coverage on workspace-local review CSV and word OCR CSV. "
                "Returns pass_strict and pages needing fixes. Auto-discovers OCR words CSV when omitted. "
                "must_redact and must_not_redact: list of regex strings (one term per item), e.g. "
                '["Hyde", "Lauren\\\\s+Lilley", "Poss\\\\b"]. A single pipe-separated string is also accepted.'
            ),
            func=lambda review_csv_relative_path, redacted_pdf_relative_path=None, ocr_words_csv_relative_path=None, must_redact=None, must_not_redact=None: run_verify_coverage(
                review_csv_relative_path,
                session_hash=session_hash,
                redacted_pdf_relative_path=redacted_pdf_relative_path,
                ocr_words_csv_relative_path=ocr_words_csv_relative_path,
                must_redact=must_redact,
                must_not_redact=must_not_redact,
            ),
        ),
        StructuredTool.from_function(
            name="read_workspace_text",
            description="Read a text file (CSV, JSON, .py) from the session workspace.",
            func=lambda relative_path: read_workspace_text(
                relative_path, session_hash=session_hash
            ),
        ),
        StructuredTool.from_function(
            name="write_workspace_text",
            description=(
                "Write UTF-8 text into the session workspace (use utf-8-sig for review CSV edits)."
            ),
            func=lambda relative_path, content: write_workspace_text(
                relative_path, content, session_hash=session_hash
            ),
        ),
        StructuredTool.from_function(
            name="run_workspace_python_script",
            description=(
                "Execute a .py script saved in the session workspace (for pandas CSV policy edits)."
            ),
            func=lambda relative_path, content=None: run_workspace_python_script(
                relative_path, content, session_hash=session_hash
            ),
        ),
    ]
