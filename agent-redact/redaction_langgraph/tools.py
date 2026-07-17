"""Curated LangGraph tools for doc_redaction orchestration (no shell)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from remote_redaction import (  # noqa: E402
    call_doc_redact,
    extract_server_paths,
    fetch_redaction_files,
    make_redaction_client,
)
from session_workspace import session_workspace_dir  # noqa: E402

_MAX_TEXT_BYTES = int(os.environ.get("LANGGRAPH_MAX_WORKSPACE_TEXT_BYTES", "1500000"))
_MAX_SCRIPT_SECONDS = int(os.environ.get("LANGGRAPH_WORKSPACE_SCRIPT_TIMEOUT", "300"))
# Soft cap for write_workspace_text bodies so tool-call JSON stays parseable on local models.
_MAX_WRITE_CONTENT_BYTES = int(
    os.environ.get("LANGGRAPH_MAX_WRITE_CONTENT_BYTES", "24000")
)
_MAX_PY_WRITES_WITHOUT_RUN = int(
    os.environ.get("LANGGRAPH_MAX_PY_WRITES_WITHOUT_RUN", "2")
)
_TOOL_ARG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# session_hash -> {path, count} for consecutive .py writes without a run
_PY_WRITE_STORM: dict[str, dict[str, Any]] = {}
# session_hash -> list of (tool_name, error_signature) for loop breaking
_RECENT_TOOL_ERRORS: dict[str, list[tuple[str, str]]] = {}
# session_hash -> last known Pass 1 artifact paths (for empty {} arg autofill)
_SESSION_ARTIFACTS: dict[str, dict[str, Any]] = {}


def _session_root(session_hash: str | None) -> Path:
    if session_hash:
        return session_workspace_dir(session_hash)
    from session_workspace import workspace_base_dir

    return workspace_base_dir()


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


def _session_key(session_hash: str | None) -> str:
    return session_hash or ""


def _error_signature(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower())[:240]


def _record_tool_error(session_hash: str | None, tool_name: str, message: str) -> int:
    """Record a tool error; return consecutive identical streak for this tool."""
    key = _session_key(session_hash)
    sig = _error_signature(message)
    recent = _RECENT_TOOL_ERRORS.setdefault(key, [])
    recent.append((tool_name, sig))
    del recent[:-8]
    streak = 0
    for name, prev in reversed(recent):
        if name != tool_name or prev != sig:
            break
        streak += 1
    return streak


def _clear_tool_errors(session_hash: str | None, tool_name: str | None = None) -> None:
    key = _session_key(session_hash)
    if tool_name is None:
        _RECENT_TOOL_ERRORS.pop(key, None)
        return
    recent = _RECENT_TOOL_ERRORS.get(key) or []
    _RECENT_TOOL_ERRORS[key] = [
        (name, sig) for name, sig in recent if name != tool_name
    ]


def _note_python_write(session_hash: str | None, rel: str) -> int:
    key = _session_key(session_hash)
    state = _PY_WRITE_STORM.setdefault(key, {"path": None, "count": 0})
    if state.get("path") == rel:
        state["count"] = int(state.get("count") or 0) + 1
    else:
        state["path"] = rel
        state["count"] = 1
    return int(state["count"])


def _clear_python_write_storm(session_hash: str | None) -> None:
    _PY_WRITE_STORM.pop(_session_key(session_hash), None)


def reset_langgraph_tool_session_state(session_hash: str | None = None) -> None:
    """Clear write-storm / error-loop counters (tests and new Gradio sessions)."""
    if session_hash is None:
        _PY_WRITE_STORM.clear()
        _RECENT_TOOL_ERRORS.clear()
        _SESSION_ARTIFACTS.clear()
        return
    _clear_python_write_storm(session_hash)
    _clear_tool_errors(session_hash)
    _SESSION_ARTIFACTS.pop(_session_key(session_hash), None)


def _remember_session_artifacts(
    session_hash: str | None, **paths: str | None
) -> dict[str, Any]:
    key = _session_key(session_hash)
    state = _SESSION_ARTIFACTS.setdefault(key, {})
    for name, value in paths.items():
        if isinstance(value, str) and value.strip():
            state[name] = value.strip().replace("\\", "/")
    return state


def _session_artifacts(session_hash: str | None) -> dict[str, Any]:
    return _SESSION_ARTIFACTS.get(_session_key(session_hash), {})


def _path_arg_is_empty(value: Any) -> bool:
    """True when the model omitted a path or passed ``{}`` / nested empties."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or text in {"{}", "null", "None", "none"}
    if isinstance(value, dict):
        if not value:
            return True
        try:
            return not bool(_coerce_relative_path(value))
        except ValueError:
            return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or (len(value) == 1 and _path_arg_is_empty(value[0]))
    return False


def _list_workspace_review_csvs(session_hash: str | None) -> list[str]:
    root = _session_root(session_hash).resolve()
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in sorted(root.rglob("*.csv")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if _looks_like_review_csv(rel):
            found.append(rel)
    return found


def _list_workspace_ocr_words_csvs(session_hash: str | None) -> list[str]:
    root = _session_root(session_hash).resolve()
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in sorted(root.rglob("*ocr_results_with_words*.csv")):
        if path.is_file():
            found.append(str(path.relative_to(root)).replace("\\", "/"))
    return found


def _auto_discover_workspace_text_path(session_hash: str | None) -> str:
    """
    Pick a useful text/CSV path when read_workspace_text gets ``{}``.

    Rotates through known review → OCR words paths so repeated empty calls still
    make progress instead of identical-error looping.
    """
    arts = _session_artifacts(session_hash)
    candidates: list[str] = []
    for key in ("review_csv_relative_path", "ocr_words_csv_relative_path"):
        value = arts.get(key)
        if isinstance(value, str) and value.strip() and value not in candidates:
            candidates.append(value.strip())
    for rel in _list_workspace_review_csvs(session_hash):
        if rel not in candidates:
            candidates.append(rel)
    for rel in _list_workspace_ocr_words_csvs(session_hash):
        if rel not in candidates:
            candidates.append(rel)
    if not candidates:
        raise ValueError(
            "read_workspace_text requires relative_path as a plain string "
            '(e.g. "redact/doc/output_redact/doc_review_file.csv"), not {}. '
            "Call list_workspace_files first, then retry with the exact path."
        )
    state = _SESSION_ARTIFACTS.setdefault(_session_key(session_hash), {})
    idx = int(state.get("empty_read_idx") or 0)
    chosen = candidates[idx % len(candidates)]
    state["empty_read_idx"] = idx + 1
    # Persist discovered paths for later tools (verify_coverage, etc.).
    if _looks_like_review_csv(chosen):
        state.setdefault("review_csv_relative_path", chosen)
    elif "ocr_results_with_words" in chosen.lower():
        state.setdefault("ocr_words_csv_relative_path", chosen)
    return chosen


def _tool_error_payload(
    session_hash: str | None,
    tool_name: str,
    message: str,
    *,
    extra: dict[str, Any] | None = None,
    fix_example: dict[str, Any] | None = None,
) -> str:
    streak = _record_tool_error(session_hash, tool_name, message)
    payload: dict[str, Any] = {"error": message}
    if extra:
        payload.update(extra)
    if fix_example is not None:
        payload["fix_example"] = fix_example
        payload["hint"] = (
            "Tool arguments must be flat strings "
            '(e.g. {"pdf_relative_path": "file.pdf"}), never nested objects or {}.'
        )
    if streak >= 2:
        payload["loop_breaker"] = True
        payload["identical_error_streak"] = streak
        payload["next_step"] = (
            "STOP retrying with the same args. Re-read fix_example, use flat string "
            "values, then take a different next step (list_workspace_files or a "
            "corrected call)."
        )
    return json.dumps(payload, indent=2)


def normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested local-model tool args into plain string parameters."""
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if not isinstance(key, str):
            continue
        if key in {
            "must_redact",
            "must_not_redact",
            "deny_list",
            "allow_list",
        }:
            out[key] = value
            continue
        if key in {"content", "ocr_method", "pii_method"}:
            if isinstance(value, dict):
                try:
                    out[key] = _coerce_tool_text_content(value, label=key)
                except ValueError:
                    out[key] = value
            else:
                out[key] = value
            continue
        if isinstance(value, dict):
            try:
                out[key] = _coerce_relative_path(value, label=key)
            except ValueError:
                # Keep non-path dicts (rare) for downstream parsers.
                nested = _sanitize_tool_dict(value)
                if key in nested and not isinstance(nested[key], dict):
                    out[key] = nested[key]
                else:
                    out[key] = value
        else:
            out[key] = value
    # Cross-key rescue: wrong inner key names often leave empty strings.
    for key, value in list(out.items()):
        if value == "" or value == {}:
            raw = args.get(key)
            if isinstance(raw, dict):
                try:
                    out[key] = _coerce_relative_path(raw, label=key)
                except ValueError:
                    pass
    _ = tool_name  # reserved for tool-specific remaps
    return out


def _looks_like_review_csv(path: str) -> bool:
    name = Path(path.replace("\\", "/")).name.lower()
    return name.endswith(".csv") and (
        "review_file" in name or name.endswith("_review.csv")
    )


_DEFAULT_REVIEW_COLOR = "(0, 0, 0)"


def _normalize_review_color_cell(raw: Any) -> str:
    """Coerce a review-CSV color cell to ``'(R, G, B)'`` with 0–255 ints."""
    import ast

    if raw is None:
        return _DEFAULT_REVIEW_COLOR
    if isinstance(raw, (tuple, list)) and len(raw) == 3:
        try:
            rgb = tuple(int(part) for part in raw)
            if all(0 <= c <= 255 for c in rgb):
                return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
        except (TypeError, ValueError):
            return _DEFAULT_REVIEW_COLOR
    s = str(raw).strip()
    if not s or s.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "black",
        "placeholder",
        "",
    }:
        return _DEFAULT_REVIEW_COLOR
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 3:
            rgb = tuple(int(part) for part in parsed)
            if all(0 <= c <= 255 for c in rgb):
                return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    except (SyntaxError, ValueError, TypeError):
        pass
    match = re.match(
        r"^\(?\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*\)?$",
        s,
    )
    if match:
        rgb = tuple(int(match.group(i)) for i in range(1, 4))
        if all(0 <= c <= 255 for c in rgb):
            return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", s)
    if hex_match:
        h = hex_match.group(1)
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    return _DEFAULT_REVIEW_COLOR


def _repair_review_csv_body(body: str) -> tuple[str, int]:
    """
    Normalize review CSV ``color`` cells to ``(R, G, B)`` strings.

    Returns (repaired_body, number_of_cells_changed).
    """
    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(body))
    except csv.Error:
        return body, 0
    if not reader.fieldnames:
        return body, 0
    color_key = None
    for name in reader.fieldnames:
        if name and name.strip().lower() == "color":
            color_key = name
            break
    if color_key is None:
        return body, 0
    rows = list(reader)
    changed = 0
    for row in rows:
        original = row.get(color_key, "")
        fixed = _normalize_review_color_cell(original)
        if str(original).strip() != fixed:
            changed += 1
        row[color_key] = fixed
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), changed


def _repair_review_csv_file(path: Path) -> int:
    """Repair color column on disk; return number of cells changed."""
    body = path.read_text(encoding="utf-8-sig")
    repaired, changed = _repair_review_csv_body(body)
    if changed:
        path.write_text(repaired, encoding="utf-8-sig")
    return changed


def _validate_review_csv_body(body: str) -> str | None:
    """Return an error message if review CSV bbox cells are non-numeric placeholders."""
    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(body))
    except csv.Error as exc:
        return f"Invalid review CSV: {exc}"
    if not reader.fieldnames:
        return "Invalid review CSV: missing header row."
    fields = {name.strip().lower(): name for name in reader.fieldnames if name}
    bbox_keys = []
    for wanted in ("xmin", "xmax", "ymin", "ymax"):
        if wanted in fields:
            bbox_keys.append(fields[wanted])
    if len(bbox_keys) < 4:
        return None  # schema variants without bbox — leave to verify_coverage
    placeholder_tokens = {
        "",
        "placeholder",
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "?",
    }
    for row_idx, row in enumerate(reader, start=2):
        for key in bbox_keys:
            raw = str(row.get(key, "")).strip()
            if raw.lower() in placeholder_tokens:
                return (
                    f"Invalid review CSV at row {row_idx}: column {key!r} is "
                    f"{raw!r}. Bbox values must be numeric floats in [0, 1]. "
                    "Look up coordinates from the OCR words CSV, or omit the row."
                )
            try:
                val = float(raw)
            except ValueError:
                return (
                    f"Invalid review CSV at row {row_idx}: column {key!r}={raw!r} "
                    "is not a number. Use floats in [0, 1] from the OCR words CSV."
                )
            if val < -0.05 or val > 1.05:
                return (
                    f"Invalid review CSV at row {row_idx}: column {key!r}={val} "
                    "out of expected [0, 1] range (normalized PDF coordinates)."
                )
        if "color" in fields:
            color_raw = str(row.get(fields["color"], "")).strip()
            fixed = _normalize_review_color_cell(color_raw)
            # Only error when the cell is clearly a placeholder and we cannot
            # interpret it — repair handles most cases on write/apply.
            if (
                color_raw.lower() in placeholder_tokens
                and fixed == _DEFAULT_REVIEW_COLOR
            ):
                # Allow — write path will repair to (0, 0, 0).
                pass
    return None


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


def _looks_like_filesystem_path(text: str) -> bool:
    normalized = text.strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized.startswith(("/", "~")):
        return True
    if re.match(r"^[A-Za-z]:/", normalized):
        return True
    return "/" in normalized or normalized.lower().endswith(".pdf")


def _deep_flatten_tool_payload(*values: Any) -> dict[str, Any]:
    """
    Recursively collect tool-arg fields from nested local-model structures.

    Weak models often nest the real args under an absolute path key, e.g.
    ``{"pdf_relative_path": {"/abs/path/doc.pdf": {"pdf_relative_path": "doc.pdf"}}}``.
    """
    merged: dict[str, Any] = {}

    def absorb(key: str, val: Any) -> None:
        if isinstance(val, str) and val.strip():
            merged[key] = val.strip()
        elif val is not None and not isinstance(val, (dict, list, tuple)):
            merged[key] = val

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(key, str) and _TOOL_ARG_KEY_RE.fullmatch(key):
                    if isinstance(val, dict):
                        walk(val)
                    else:
                        absorb(key, val)
                elif isinstance(key, str) and _looks_like_filesystem_path(key):
                    if isinstance(val, dict):
                        walk(val)
                    elif isinstance(val, str) and val.strip():
                        absorb("pdf_path", val)
                    else:
                        key_path = Path(key.replace("\\", "/"))
                        if key_path.suffix.lower() in _OUTPUT_FILE_EXTENSIONS:
                            name = key_path.name
                            if name:
                                absorb("pdf_path", name)
                else:
                    walk(val)
        elif isinstance(node, str) and node.strip():
            absorb("_literal", node)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    for value in values:
        walk(value)
    return merged


def _normalize_workspace_relative_path(path: str, session_hash: str | None) -> str:
    """Strip a session workspace prefix from absolute paths; keep basename as fallback."""
    text = path.strip().replace("\\", "/")
    if not text:
        return text
    root = _session_root(session_hash).resolve()
    try:
        raw_path = Path(path.strip())
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
            rel = os.path.relpath(str(resolved), str(root))
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
    except (OSError, ValueError):
        pass
    root_posix = root.as_posix().rstrip("/")
    lowered = text.lower()
    root_lower = root_posix.lower()
    idx = lowered.find(root_lower)
    if idx != -1:
        suffix = text[idx + len(root_posix) :].lstrip("/\\")
        if suffix:
            return suffix.replace("\\", "/")
    if "/" in text or ":" in text:
        name = Path(text).name
        if name:
            return name
    return text


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
            for nested in value.values():
                try:
                    return _coerce_relative_path(nested, label=label)
                except ValueError:
                    continue
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


def _list_workspace_source_pdfs(session_hash: str | None) -> list[str]:
    """Workspace-relative PDFs suitable as doc_redact inputs (exclude outputs)."""
    root = _session_root(session_hash).resolve()
    if not root.is_dir():
        return []
    skip_parts = {"output_redact", "output_review_final", "review"}
    found: list[tuple[int, str]] = []
    for path in root.rglob("*.pdf"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts_lower = {part.lower() for part in rel.parts}
        if parts_lower & skip_parts:
            continue
        name_lower = path.name.lower()
        if name_lower.endswith("_redacted.pdf") or "_redacted." in name_lower:
            continue
        found.append((len(rel.parts), str(rel).replace("\\", "/")))
    found.sort(key=lambda item: (item[0], item[1].lower()))
    return [rel for _, rel in found]


def _auto_discover_workspace_pdf(session_hash: str | None) -> str:
    """
    Pick a source PDF when the model omits the path or passes an empty object {}.

    Local models often emit ``{"pdf_relative_path": {}}`` after compaction drops
    the filename from context. If exactly one source PDF exists, use it.
    """
    pdfs = _list_workspace_source_pdfs(session_hash)
    if len(pdfs) == 1:
        return pdfs[0]
    if not pdfs:
        raise ValueError(
            "doc_redact requires pdf_relative_path as a plain string "
            '(e.g. "document.pdf"), not {}. No source PDF found in the workspace — '
            "call list_workspace_files first."
        )
    preview = ", ".join(pdfs[:8])
    more = f" (+{len(pdfs) - 8} more)" if len(pdfs) > 8 else ""
    raise ValueError(
        "doc_redact requires pdf_relative_path as a plain string filename "
        f'(e.g. "{pdfs[0]}"), not {{}} or a nested object. '
        f"Available source PDFs: {preview}{more}. "
        "Call list_workspace_files if unsure, then retry with the exact string path."
    )


def _looks_like_dest_dir_path(rel: str) -> bool:
    """True when a path looks like an output directory, not a source PDF."""
    text = (rel or "").strip().replace("\\", "/")
    if not text:
        return True
    path = Path(text)
    name = path.name.lower()
    if name in {"output_review_final", "output_redact", "review", "redact"}:
        return True
    if path.suffix:
        return False
    lowered = text.lower()
    return (
        "output_review_final" in lowered
        or lowered.endswith("/output_redact")
        or "/output_redact/" in lowered
        or lowered.endswith("/review")
    )


def _looks_like_pdf_filename(rel: str) -> bool:
    return (rel or "").strip().lower().replace("\\", "/").endswith(".pdf")


def _looks_like_review_csv_path(rel: str) -> bool:
    return _looks_like_review_csv(rel or "")


def _fix_example_source_pdf(session_hash: str | None) -> str:
    """Best-known source PDF path for tool error fix_example payloads."""
    arts = _session_artifacts(session_hash)
    remembered = arts.get("pdf_relative_path")
    if isinstance(remembered, str) and remembered.strip().lower().endswith(".pdf"):
        return remembered.strip().replace("\\", "/")
    pdfs = _list_workspace_source_pdfs(session_hash)
    if pdfs:
        return pdfs[0]
    return "document.pdf"


def _resolve_source_pdf_arg(
    pdf_raw: str,
    *,
    session_hash: str | None,
    tool_name: str = "doc_redact",
) -> str:
    """
    Resolve a source PDF path, repairing dest-dir mix-ups like ``output_redact``.
    """
    text = (pdf_raw or "").strip()
    if text and _looks_like_pdf_filename(text) and not _looks_like_dest_dir_path(text):
        return _normalize_workspace_relative_path(text, session_hash)

    arts = _session_artifacts(session_hash)
    remembered = str(arts.get("pdf_relative_path") or "").strip()
    if remembered and _looks_like_pdf_filename(remembered):
        return _normalize_workspace_relative_path(remembered, session_hash)

    try:
        return _auto_discover_workspace_pdf(session_hash)
    except ValueError as exc:
        bad = text or "{}"
        raise ValueError(
            f"{tool_name} requires pdf_relative_path as a source .pdf file, "
            f"not {bad!r}. " + str(exc).replace("doc_redact", tool_name)
        ) from exc


def _parse_doc_redact_tool_input(
    pdf_relative_path: Any,
    dest_relative_dir: Any | None,
    *,
    ocr_method: str | None,
    pii_method: str | None,
    session_hash: str | None = None,
) -> tuple[str, str, str | None, str | None]:
    """Merge/normalize doc_redact tool args from messy local-model tool calls."""
    payload = _deep_flatten_tool_payload(pdf_relative_path, dest_relative_dir)

    # Prefer explicit string args first (avoid _literal overwrite by dest strings).
    pdf_from_arg = ""
    if isinstance(pdf_relative_path, str) and pdf_relative_path.strip():
        pdf_from_arg = pdf_relative_path.strip()
    if not pdf_from_arg and not _path_arg_is_empty(pdf_relative_path):
        try:
            pdf_from_arg = _coerce_relative_path(
                pdf_relative_path, label="pdf_relative_path"
            )
        except ValueError:
            pdf_from_arg = ""
    if _path_arg_is_empty(pdf_relative_path) and not isinstance(pdf_relative_path, str):
        pdf_from_arg = ""

    pdf_from_keys = _first_string(payload, _DOC_REDACT_PDF_KEYS)
    literal = str(payload.get("_literal") or "").strip()

    dest_raw = ""
    if isinstance(dest_relative_dir, str) and dest_relative_dir.strip():
        dest_raw = dest_relative_dir.strip()
    if not dest_raw:
        dest_raw = _first_string(payload, _DOC_REDACT_DEST_KEYS)
    if (
        not dest_raw
        and dest_relative_dir not in (None, "")
        and not _path_arg_is_empty(dest_relative_dir)
    ):
        try:
            dest_raw = _coerce_relative_path(
                dest_relative_dir, label="dest_relative_dir"
            )
        except ValueError:
            dest_raw = ""

    # Prefer named PDF keys / real .pdf strings over dest-dir mix-ups like output_redact.
    pdf_raw = ""
    for candidate in (pdf_from_keys, pdf_from_arg, literal):
        if not candidate:
            continue
        if _looks_like_dest_dir_path(candidate) or not _looks_like_pdf_filename(
            candidate
        ):
            if not dest_raw and _looks_like_dest_dir_path(candidate):
                dest_raw = candidate
            continue
        pdf_raw = candidate
        break

    if literal and not dest_raw and _looks_like_dest_dir_path(literal):
        dest_raw = literal
    if pdf_from_arg and not dest_raw and _looks_like_dest_dir_path(pdf_from_arg):
        dest_raw = pdf_from_arg

    pdf_raw = _resolve_source_pdf_arg(
        pdf_raw, session_hash=session_hash, tool_name="doc_redact"
    )
    pdf_rel = _coerce_relative_path(pdf_raw, label="pdf_relative_path")

    dest_rel = (
        _coerce_relative_path(dest_raw, label="dest_relative_dir")
        if dest_raw
        else _default_dest_for_pdf(pdf_rel)
    )
    if dest_rel and _looks_like_file_relative_path(dest_rel):
        dest_rel = _default_dest_for_pdf(pdf_rel)

    ocr = ocr_method or _first_string(payload, ("ocr_method",)) or None
    pii = pii_method or _first_string(payload, ("pii_method",)) or None
    return pdf_rel, dest_rel, ocr, pii


def _parse_review_apply_tool_input(
    pdf_relative_path: Any,
    review_csv_relative_path: Any,
    dest_relative_dir: Any | None,
    *,
    session_hash: str | None = None,
) -> tuple[str, str, str]:
    """Merge/normalize review_apply tool args from messy local-model tool calls."""
    payload = _deep_flatten_tool_payload(
        pdf_relative_path, review_csv_relative_path, dest_relative_dir
    )
    arts = _session_artifacts(session_hash)

    # Prefer explicit string parameters first. Deep-flatten's ``_literal`` is a single
    # slot and later string args (review CSV) can overwrite an earlier dest/PDF literal.
    pdf_raw = ""
    if isinstance(pdf_relative_path, str) and pdf_relative_path.strip():
        pdf_raw = pdf_relative_path.strip()
    if not pdf_raw:
        pdf_raw = _first_string(payload, _DOC_REDACT_PDF_KEYS)
    if not pdf_raw and not _path_arg_is_empty(pdf_relative_path):
        try:
            pdf_raw = _coerce_relative_path(
                pdf_relative_path, label="pdf_relative_path"
            )
        except ValueError:
            pdf_raw = ""
    if _path_arg_is_empty(pdf_relative_path) and not isinstance(pdf_relative_path, str):
        pdf_raw = ""

    review_raw = ""
    if isinstance(review_csv_relative_path, str) and review_csv_relative_path.strip():
        review_raw = review_csv_relative_path.strip()
    if not review_raw:
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
    if not review_raw and not _path_arg_is_empty(review_csv_relative_path):
        try:
            review_raw = _coerce_relative_path(
                review_csv_relative_path, label="review_csv_relative_path"
            )
        except ValueError:
            review_raw = ""
    if _path_arg_is_empty(review_csv_relative_path) and not isinstance(
        review_csv_relative_path, str
    ):
        review_raw = ""

    dest_raw = ""
    if isinstance(dest_relative_dir, str) and dest_relative_dir.strip():
        dest_raw = dest_relative_dir.strip()
    if not dest_raw:
        dest_raw = _first_string(payload, _DOC_REDACT_DEST_KEYS)
    if (
        not dest_raw
        and dest_relative_dir not in (None, "")
        and not _path_arg_is_empty(dest_relative_dir)
    ):
        try:
            dest_raw = _coerce_relative_path(
                dest_relative_dir, label="dest_relative_dir"
            )
        except ValueError:
            dest_raw = ""

    literal = str(payload.get("_literal") or "").strip()
    if literal:
        if not pdf_raw and _looks_like_pdf_filename(literal):
            pdf_raw = literal
        elif not review_raw and _looks_like_review_csv_path(literal):
            review_raw = literal
        elif not dest_raw and _looks_like_dest_dir_path(literal):
            dest_raw = literal

    # Models often put dest_relative_dir (e.g. output_review_final) in the PDF slot,
    # or swap PDF/CSV. Repair before resolving files.
    if pdf_raw and _looks_like_dest_dir_path(pdf_raw):
        if not dest_raw:
            dest_raw = pdf_raw
        pdf_raw = ""
    if pdf_raw and _looks_like_review_csv_path(pdf_raw):
        if not review_raw:
            review_raw = pdf_raw
        pdf_raw = ""
    if review_raw and _looks_like_pdf_filename(review_raw) and not pdf_raw:
        pdf_raw, review_raw = review_raw, ""
    if dest_raw and _looks_like_pdf_filename(dest_raw) and not pdf_raw:
        pdf_raw, dest_raw = dest_raw, ""
    if dest_raw and _looks_like_review_csv_path(dest_raw) and not review_raw:
        review_raw, dest_raw = dest_raw, ""

    pdf_raw = _resolve_source_pdf_arg(
        pdf_raw, session_hash=session_hash, tool_name="review_apply"
    )
    if not review_raw:
        review_raw = str(arts.get("review_csv_relative_path") or "").strip()
    if not review_raw:
        reviews = _list_workspace_review_csvs(session_hash)
        if len(reviews) == 1:
            review_raw = reviews[0]
        elif reviews:
            raise ValueError(
                "review_apply requires review_csv_relative_path as a plain string. "
                f"Candidates: {', '.join(reviews[:8])}."
            )
        else:
            raise ValueError(
                "review_apply requires a review CSV path (review_csv_relative_path)."
            )

    pdf_rel = _coerce_relative_path(pdf_raw, label="pdf_relative_path")
    if not _looks_like_pdf_filename(pdf_rel) or _looks_like_dest_dir_path(pdf_rel):
        raise ValueError(
            "review_apply pdf_relative_path must be a source PDF filename "
            f'(e.g. "{_fix_example_source_pdf(session_hash)}"), not a directory '
            f"like {pdf_rel!r}."
        )

    review_rel = _coerce_relative_path(review_raw, label="review_csv_relative_path")
    dest_rel = (
        _coerce_relative_path(dest_raw, label="dest_relative_dir") if dest_raw else ""
    )
    if dest_rel and _looks_like_file_relative_path(dest_rel):
        dest_rel = ""
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
    if _looks_like_dest_dir_path(pdf_relative_path) or not _looks_like_pdf_filename(
        pdf_relative_path
    ):
        raise ValueError(
            "pdf_relative_path must be a .pdf file, not "
            f"{pdf_relative_path!r}. Use the source document PDF."
        )
    try:
        candidate = _resolve_workspace_path(session_hash, pdf_relative_path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            raise ValueError(
                "pdf_relative_path points to a directory, not a PDF: "
                f"{pdf_relative_path!r}"
            )
    except ValueError as exc:
        if "directory" in str(exc).lower() or "must be a .pdf" in str(exc).lower():
            raise
        candidate = None

    root = _session_root(session_hash).resolve()
    basename = Path(pdf_relative_path.replace("\\", "/")).name
    if not basename or not basename.lower().endswith(".pdf"):
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
    dest_relative_dir: str = "",
    *,
    session_hash: str | None = None,
    ocr_method: str | None = None,
    pii_method: str | None = None,
    deny_list: list[str] | None = None,
    allow_list: list[str] | None = None,
) -> str:
    """Run Pass 1 redaction via /doc_redact and download artifacts into the session workspace."""
    try:
        pdf_rel, dest_rel, ocr_from_tool, pii_from_tool = _parse_doc_redact_tool_input(
            pdf_relative_path,
            dest_relative_dir,
            ocr_method=ocr_method,
            pii_method=pii_method,
            session_hash=session_hash,
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
            ocr_method=ocr_from_tool or os.environ.get("AGENT_DEFAULT_OCR_METHOD"),
            pii_method=pii_from_tool or os.environ.get("AGENT_DEFAULT_PII_METHOD"),
            deny_list=deny_list,
            allow_list=allow_list,
        )
    except (ValueError, FileNotFoundError) as exc:
        return _tool_error_payload(
            session_hash,
            "doc_redact",
            str(exc),
            fix_example={
                "pdf_relative_path": _fix_example_source_pdf(session_hash),
            },
            extra={
                "available_source_pdfs": _list_workspace_source_pdfs(session_hash)[:8],
            },
        )
    root = _session_root(session_hash).resolve()
    saved_rels: list[str] = []
    review_csv_rel: str | None = None
    ocr_words_rel: str | None = None
    for path in saved:
        p = Path(path)
        try:
            rel = str(p.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(path).replace("\\", "/")
        saved_rels.append(rel)
        name_lower = Path(rel).name.lower()
        if "ocr_results_with_words" in name_lower and name_lower.endswith(".csv"):
            ocr_words_rel = rel
        if "_review_file" in name_lower and name_lower.endswith(".csv"):
            review_csv_rel = rel
    if ocr_words_rel is None and review_csv_rel:
        try:
            discovered = _discover_ocr_words_csv(
                _resolve_workspace_path(session_hash, review_csv_rel)
            )
        except (ValueError, FileNotFoundError, OSError):
            discovered = None
        if discovered is not None:
            try:
                ocr_words_rel = str(discovered.resolve().relative_to(root)).replace(
                    "\\", "/"
                )
            except ValueError:
                ocr_words_rel = str(discovered)
    message = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else ""
    payload: dict[str, Any] = {
        "message": str(message or "doc_redact completed."),
        "saved_paths": saved_rels,
        "server_paths": extract_server_paths(result),
    }
    if review_csv_rel:
        payload["review_csv_relative_path"] = review_csv_rel
    if ocr_words_rel:
        payload["ocr_words_csv_relative_path"] = ocr_words_rel
    _remember_session_artifacts(
        session_hash,
        pdf_relative_path=pdf_rel,
        review_csv_relative_path=review_csv_rel,
        ocr_words_csv_relative_path=ocr_words_rel,
        dest_relative_dir=dest_rel,
    )
    _clear_tool_errors(session_hash, "doc_redact")
    return json.dumps(payload, indent=2)


def _discover_ocr_words_csv(review_csv: Path) -> Path | None:
    """Find the word-level OCR CSV sibling of a *_review_file.csv."""
    parent = review_csv.parent
    review_resolved = review_csv.resolve()

    def _is_candidate(candidate: Path) -> bool:
        if candidate.resolve() == review_resolved:
            return False
        name = candidate.name.lower()
        if "_review_file" in name:
            return False
        return name.endswith(".csv")

    # Prefer the canonical Pass 1 artifact name first.
    prioritized_patterns = (
        "*ocr_results_with_words*.csv",
        "*word*ocr*.csv",
        "*ocr*word*.csv",
        "*_words.csv",
        "*words*.csv",
    )
    search_dirs = [parent]
    # Also check one level up when review CSV sits beside other outputs.
    if parent.parent != parent:
        search_dirs.append(parent.parent)
    for directory in search_dirs:
        for pattern in prioritized_patterns:
            for candidate in sorted(directory.glob(pattern)):
                if _is_candidate(candidate):
                    return candidate
    for directory in search_dirs:
        for candidate in sorted(directory.glob("*.csv")):
            if not _is_candidate(candidate):
                continue
            name = candidate.name.lower()
            if "ocr_results_with_words" in name or "word" in name or "ocr" in name:
                return candidate
    return None


def read_workspace_text(
    relative_path: Any,
    *,
    session_hash: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Read a UTF-8 text file from the session workspace (CSV, JSON, Python script)."""
    auto_filled = False
    try:
        if _path_arg_is_empty(relative_path):
            rel = _auto_discover_workspace_text_path(session_hash)
            auto_filled = True
        else:
            rel = _coerce_relative_path(relative_path, label="relative_path")
        path = _resolve_workspace_path(session_hash, rel)
    except ValueError as exc:
        arts = _session_artifacts(session_hash)
        example_path = (
            arts.get("review_csv_relative_path")
            or next(iter(_list_workspace_review_csvs(session_hash)), None)
            or "redact/doc/output_redact/doc_review_file.csv"
        )
        return _tool_error_payload(
            session_hash,
            "read_workspace_text",
            str(exc),
            extra={"relative_path": str(relative_path)},
            fix_example={"relative_path": example_path},
        )
    except FileNotFoundError as exc:
        return _tool_error_payload(
            session_hash,
            "read_workspace_text",
            str(exc),
            extra={"relative_path": str(relative_path)},
        )
    if not path.is_file():
        return _tool_error_payload(
            session_hash,
            "read_workspace_text",
            f"File not found: {rel}",
            fix_example={"relative_path": rel},
        )
    limit = max_bytes if max_bytes is not None else _MAX_TEXT_BYTES
    size = path.stat().st_size
    if size > limit:
        return _tool_error_payload(
            session_hash,
            "read_workspace_text",
            (
                f"File too large to read ({size} bytes > {limit}). "
                "Use run_workspace_python_script on a .py file instead."
            ),
        )
    text = path.read_text(encoding="utf-8-sig")
    _clear_tool_errors(session_hash, "read_workspace_text")
    prefix = ""
    if auto_filled:
        arts = _session_artifacts(session_hash)
        extras = []
        for key, label in (
            ("review_csv_relative_path", "review"),
            ("ocr_words_csv_relative_path", "ocr_words"),
        ):
            value = arts.get(key)
            if isinstance(value, str) and value and value != rel:
                extras.append(f"{label}={value!r}")
        extra_note = f" Other paths: {', '.join(extras)}." if extras else ""
        prefix = (
            f"[auto-filled relative_path={rel!r} because the model passed {{}} "
            f"— use a plain string path next time.{extra_note}]\n\n"
        )
    max_lines = int(os.environ.get("LANGGRAPH_READ_CSV_MAX_LINES", "60"))
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv"):
        lines = text.splitlines()
        if len(lines) > max_lines:
            preview = "\n".join(lines[:max_lines])
            return (
                f"{prefix}"
                f"CSV preview for {rel} (lines 1-{max_lines} of {len(lines)}). "
                "Edit the full file with write_workspace_text or run_workspace_python_script.\n\n"
                f"{preview}"
            )
    return f"{prefix}{text}" if prefix else text


def write_workspace_text(
    relative_path: Any,
    content: Any,
    *,
    session_hash: str | None = None,
) -> str:
    """Write UTF-8 text into the session workspace (preserve utf-8-sig for review CSVs)."""
    try:
        merged = _merge_tool_arg_dicts(relative_path, content)
        # Flat-arg normalize when the model nests the whole payload in one arg.
        if isinstance(relative_path, dict) and content is None:
            merged = normalize_tool_args("write_workspace_text", merged) or merged
        rel, body = _parse_write_workspace_text_input(relative_path, content)
        if _should_resolve_script_path(_sanitize_tool_dict(merged), rel):
            rel = _resolve_script_relative_path(session_hash, rel)
        path = _resolve_workspace_path(session_hash, rel)
    except ValueError as exc:
        return _tool_error_payload(
            session_hash,
            "write_workspace_text",
            str(exc),
            fix_example={
                "relative_path": "fix_review.py",
                "content": "import csv\nprint('ok')\n",
            },
        )
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > _MAX_TEXT_BYTES:
        return _tool_error_payload(
            session_hash,
            "write_workspace_text",
            f"Content too large (>{_MAX_TEXT_BYTES} bytes).",
        )
    if body_bytes > _MAX_WRITE_CONTENT_BYTES:
        return _tool_error_payload(
            session_hash,
            "write_workspace_text",
            (
                f"Content too large for a single tool call ({body_bytes} bytes > "
                f"{_MAX_WRITE_CONTENT_BYTES}). Write a shorter .py script that reads "
                "the review/OCR CSV and derives rows programmatically, or split into "
                "smaller scripts."
            ),
            fix_example={
                "relative_path": "fix_review.py",
                "content": (
                    "import csv\n"
                    "from pathlib import Path\n"
                    "review = Path('doc_review_file.csv')\n"
                    "# filter/add rows from OCR CSV — do not hard-code bboxes\n"
                ),
            },
        )
    if _looks_like_review_csv(rel):
        body, color_fixed = _repair_review_csv_body(body)
        csv_err = _validate_review_csv_body(body)
        if csv_err:
            return _tool_error_payload(
                session_hash,
                "write_workspace_text",
                csv_err,
            )
    else:
        color_fixed = 0
    # Write-storm gate: refuse a 3rd consecutive rewrite of the same .py without run.
    if path.suffix.lower() == ".py" or rel.lower().endswith(".py"):
        # Count prospective write before applying storm limit (includes this call).
        storm_key = _session_key(session_hash)
        prior = _PY_WRITE_STORM.get(storm_key) or {}
        prior_count = int(prior.get("count") or 0) if prior.get("path") == rel else 0
        if prior_count >= _MAX_PY_WRITES_WITHOUT_RUN:
            return _tool_error_payload(
                session_hash,
                "write_workspace_text",
                (
                    f"Refusing another write to {rel!r} — already saved "
                    f"{prior_count} time(s) without running. Call "
                    f"run_workspace_python_script with relative_path={rel!r} now."
                ),
                extra={"written": rel, "blocked_write_storm": True},
                fix_example={"relative_path": rel},
            )
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
    if color_fixed:
        payload["color_cells_repaired"] = color_fixed
        payload["note"] = (
            f"Normalized {color_fixed} review CSV color cell(s) to '(R, G, B)' "
            f"(default {_DEFAULT_REVIEW_COLOR})."
        )
    if path.suffix.lower() == ".py":
        write_count = _note_python_write(session_hash, rel_written)
        payload["write_count_without_run"] = write_count
        payload["next_step"] = (
            "Script already saved. Call run_workspace_python_script with "
            f"relative_path={rel_written!r} now — do not call write_workspace_text "
            "again unless the script body must change."
        )
        if write_count >= _MAX_PY_WRITES_WITHOUT_RUN:
            payload["warning"] = (
                f"This is write #{write_count} without a run. The next write to "
                "this script will be refused until you call run_workspace_python_script."
            )
    _clear_tool_errors(session_hash, "write_workspace_text")
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
        return _tool_error_payload(
            session_hash,
            "run_workspace_python_script",
            str(exc),
            fix_example={"relative_path": "fix_review.py"},
        )
    if path.suffix.lower() != ".py":
        return _tool_error_payload(
            session_hash,
            "run_workspace_python_script",
            "Only .py scripts are allowed.",
            fix_example={"relative_path": "fix_review.py"},
        )
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(path.parent),
        capture_output=True,
        text=True,
        timeout=_MAX_SCRIPT_SECONDS,
        check=False,
    )
    _clear_python_write_storm(session_hash)
    _clear_tool_errors(session_hash, "run_workspace_python_script")
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
            session_hash=session_hash,
        )
        pdf = _resolve_workspace_pdf(session_hash, pdf_rel)
        review_csv = _resolve_workspace_path(session_hash, review_rel)
        if not review_csv.is_file():
            raise FileNotFoundError(f"Review CSV not found: {review_rel}")
        color_repaired = _repair_review_csv_file(review_csv)
        dest = _ensure_workspace_output_dir(
            session_hash,
            dest_rel,
            pdf_relative_path=pdf_rel,
            review_csv_relative_path=review_rel,
            default_for="review_apply",
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        arts = _session_artifacts(session_hash)
        return _tool_error_payload(
            session_hash,
            "review_apply",
            str(exc),
            fix_example={
                "pdf_relative_path": arts.get("pdf_relative_path")
                or _fix_example_source_pdf(session_hash),
                "review_csv_relative_path": arts.get("review_csv_relative_path")
                or "redact/doc/output_redact/doc_review_file.csv",
                "dest_relative_dir": (
                    arts.get("dest_relative_dir")
                    or "redact/doc/review/output_review_final"
                ),
            },
        )

    try:
        client = make_redaction_client()
        result = client.predict(
            api_name="/review_apply",
            pdf_file=handle_file(str(pdf)),
            review_csv_file=handle_file(str(review_csv)),
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 — Gradio AppError must not kill the tools node
        return _tool_error_payload(
            session_hash,
            "review_apply",
            str(exc),
            extra={
                "hint": (
                    "If the error mentions column 'color', each color cell must be "
                    f"a string like '{_DEFAULT_REVIEW_COLOR}' with integers 0–255. "
                    "Invalid colors are auto-repaired when possible; re-check the "
                    "review CSV and call review_apply again."
                ),
                "color_cells_repaired_before_call": color_repaired,
            },
            fix_example={
                "pdf_relative_path": pdf_rel,
                "review_csv_relative_path": review_rel,
                "dest_relative_dir": (
                    _default_review_apply_dest_for_review_csv(review_rel)
                ),
            },
        )
    server_paths = extract_server_paths(result)
    saved = fetch_redaction_files(server_paths, dest)
    message = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else ""
    root = _session_root(session_hash).resolve()
    saved_rels: list[str] = []
    redacted_rel: str | None = None
    for path in saved:
        p = Path(path)
        try:
            rel = str(p.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(path).replace("\\", "/")
        saved_rels.append(rel)
        if rel.lower().endswith(".pdf") and "_redacted" in Path(rel).name.lower():
            redacted_rel = rel
    _remember_session_artifacts(
        session_hash,
        pdf_relative_path=pdf_rel,
        review_csv_relative_path=review_rel,
        redacted_pdf_relative_path=redacted_rel,
    )
    _clear_tool_errors(session_hash, "review_apply")
    payload: dict[str, Any] = {
        "message": str(message or "review_apply completed."),
        "saved_paths": saved_rels,
        "server_paths": server_paths,
    }
    if redacted_rel:
        payload["redacted_pdf_relative_path"] = redacted_rel
    if color_repaired:
        payload["color_cells_repaired"] = color_repaired
    return json.dumps(payload, indent=2)


def _resolve_optional_redacted_pdf(
    session_hash: str | None,
    redacted_pdf_relative_path: Any,
    *,
    review_csv: Path,
) -> Path | None:
    """Resolve optional post-apply PDF; reject CSV / non-PDF mix-ups from the model."""
    if redacted_pdf_relative_path is None:
        return None
    if (
        isinstance(redacted_pdf_relative_path, str)
        and not redacted_pdf_relative_path.strip()
    ):
        return None
    rel = _coerce_relative_path(
        redacted_pdf_relative_path, label="redacted_pdf_relative_path"
    )
    if not rel:
        return None
    lower = rel.lower().replace("\\", "/")
    name = Path(lower).name
    if (
        lower.endswith((".csv", ".json", ".py", ".txt", ".md"))
        or "review_file" in name
        or name.endswith("_review.csv")
    ):
        raise ValueError(
            "redacted_pdf_relative_path must be a PDF (e.g. *_redacted.pdf). "
            f"Got {rel!r}. For pre-apply verify_coverage, omit "
            "redacted_pdf_relative_path entirely. For post-apply checks, pass the "
            "*_redacted.pdf produced by review_apply."
        )
    if not lower.endswith(".pdf"):
        raise ValueError(
            "redacted_pdf_relative_path must end with .pdf "
            f"(got {rel!r}). Omit it for pre-apply checks."
        )
    path = _resolve_workspace_path(session_hash, rel)
    if path.resolve() == review_csv.resolve():
        raise ValueError(
            "redacted_pdf_relative_path must not be the review CSV. "
            "Omit it for pre-apply verify_coverage, or pass *_redacted.pdf."
        )
    if not path.is_file():
        raise FileNotFoundError(f"redacted PDF not found: {rel}")
    return path


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

    try:
        arts = _session_artifacts(session_hash)
        if _path_arg_is_empty(review_csv_relative_path):
            review_rel = arts.get("review_csv_relative_path") or next(
                iter(_list_workspace_review_csvs(session_hash)), None
            )
            if not review_rel:
                raise ValueError(
                    "verify_coverage requires review_csv_relative_path as a plain "
                    'string (e.g. "..._review_file.csv"), not {}.'
                )
        else:
            review_rel = _coerce_relative_path(
                review_csv_relative_path, label="review_csv_relative_path"
            )
        review_csv = _resolve_workspace_path(session_hash, review_rel)
        if _path_arg_is_empty(ocr_words_csv_relative_path):
            ocr_words_csv_relative_path = arts.get("ocr_words_csv_relative_path")
        if ocr_words_csv_relative_path:
            ocr_rel = _coerce_relative_path(
                ocr_words_csv_relative_path, label="ocr_words_csv_relative_path"
            )
            ocr_words_csv = _resolve_workspace_path(session_hash, ocr_rel)
        else:
            discovered = _discover_ocr_words_csv(review_csv)
            if discovered is None:
                return _tool_error_payload(
                    session_hash,
                    "verify_coverage",
                    (
                        "Could not find word-level OCR CSV beside the review CSV. "
                        "Pass ocr_words_csv_relative_path explicitly "
                        "(look for *ocr_results_with_words*.csv under output_redact/)."
                    ),
                    extra={"review_csv": str(review_csv)},
                    fix_example={
                        "review_csv_relative_path": review_rel,
                        "ocr_words_csv_relative_path": (
                            "redact/doc/output_redact/"
                            "doc_ocr_results_with_words_local_ocr.csv"
                        ),
                        "must_redact": ["ExampleName"],
                    },
                )
            ocr_words_csv = discovered
        redacted_pdf = _resolve_optional_redacted_pdf(
            session_hash,
            redacted_pdf_relative_path,
            review_csv=review_csv,
        )
        report = verify_redaction_coverage(
            review_csv,
            ocr_words_csv,
            must_redact=must_redact,
            must_not_redact=must_not_redact,
            redacted_pdf_path=redacted_pdf,
        )
    except (ValueError, re.error, FileNotFoundError, OSError) as exc:
        return _tool_error_payload(
            session_hash,
            "verify_coverage",
            str(exc),
            extra={
                "hint": (
                    "verify_coverage args: review_csv_relative_path (required), "
                    "optional redacted_pdf_relative_path (*_redacted.pdf only; "
                    "omit for pre-apply), optional ocr_words_csv_relative_path "
                    "(*ocr_results_with_words*.csv; auto-discovered when omitted)."
                )
            },
            fix_example={
                "review_csv_relative_path": "redact/doc/output_redact/doc_review_file.csv",
                "must_redact": ["ExampleName"],
            },
        )
    except Exception as exc:  # noqa: BLE001 — keep LangGraph tool node alive
        return _tool_error_payload(
            session_hash,
            "verify_coverage",
            f"{type(exc).__name__}: {exc}",
            extra={
                "hint": (
                    "verify_coverage failed unexpectedly. Check paths: review CSV vs "
                    "optional *_redacted.pdf (never pass the review CSV as the PDF)."
                )
            },
        )
    payload = report.to_dict()
    payload["ocr_words_csv"] = str(ocr_words_csv)
    if redacted_pdf is not None:
        payload["redacted_pdf"] = str(redacted_pdf)
    _clear_tool_errors(session_hash, "verify_coverage")
    return json.dumps(payload, indent=2, default=str)


def build_langgraph_tools(session_hash: str | None):
    """Return LangChain tools bound to *session_hash* workspace."""
    from langchain_core.tools import StructuredTool

    def _doc_redact(
        pdf_relative_path,
        dest_relative_dir="",
        ocr_method=None,
        pii_method=None,
    ):
        args = normalize_tool_args(
            "doc_redact",
            {
                "pdf_relative_path": pdf_relative_path,
                "dest_relative_dir": dest_relative_dir,
                "ocr_method": ocr_method,
                "pii_method": pii_method,
            },
        )
        return run_doc_redact(
            args.get("pdf_relative_path", pdf_relative_path),
            args.get("dest_relative_dir", dest_relative_dir) or "",
            session_hash=session_hash,
            ocr_method=args.get("ocr_method", ocr_method),
            pii_method=args.get("pii_method", pii_method),
        )

    def _review_apply(
        pdf_relative_path,
        review_csv_relative_path,
        dest_relative_dir,
    ):
        args = normalize_tool_args(
            "review_apply",
            {
                "pdf_relative_path": pdf_relative_path,
                "review_csv_relative_path": review_csv_relative_path,
                "dest_relative_dir": dest_relative_dir,
            },
        )
        return run_review_apply(
            args.get("pdf_relative_path", pdf_relative_path),
            args.get("review_csv_relative_path", review_csv_relative_path),
            args.get("dest_relative_dir", dest_relative_dir),
            session_hash=session_hash,
        )

    def _verify_coverage(
        review_csv_relative_path,
        redacted_pdf_relative_path=None,
        ocr_words_csv_relative_path=None,
        must_redact=None,
        must_not_redact=None,
    ):
        args = normalize_tool_args(
            "verify_coverage",
            {
                "review_csv_relative_path": review_csv_relative_path,
                "redacted_pdf_relative_path": redacted_pdf_relative_path,
                "ocr_words_csv_relative_path": ocr_words_csv_relative_path,
                "must_redact": must_redact,
                "must_not_redact": must_not_redact,
            },
        )
        return run_verify_coverage(
            args.get("review_csv_relative_path", review_csv_relative_path),
            session_hash=session_hash,
            redacted_pdf_relative_path=args.get(
                "redacted_pdf_relative_path", redacted_pdf_relative_path
            ),
            ocr_words_csv_relative_path=args.get(
                "ocr_words_csv_relative_path", ocr_words_csv_relative_path
            ),
            must_redact=args.get("must_redact", must_redact),
            must_not_redact=args.get("must_not_redact", must_not_redact),
        )

    def _read_workspace_text(relative_path):
        args = normalize_tool_args(
            "read_workspace_text", {"relative_path": relative_path}
        )
        return read_workspace_text(
            args.get("relative_path", relative_path), session_hash=session_hash
        )

    def _write_workspace_text(relative_path, content):
        # Preserve merge semantics when the model nests the full payload in one arg.
        if isinstance(relative_path, dict) and content is None:
            merged = dict(relative_path)
            if "content" in merged or "relative_path" in merged:
                return write_workspace_text(
                    relative_path, None, session_hash=session_hash
                )
        args = normalize_tool_args(
            "write_workspace_text",
            {"relative_path": relative_path, "content": content},
        )
        return write_workspace_text(
            args.get("relative_path", relative_path),
            args.get("content", content),
            session_hash=session_hash,
        )

    def _run_workspace_python_script(relative_path, content=None):
        args = normalize_tool_args(
            "run_workspace_python_script",
            {"relative_path": relative_path, "content": content},
        )
        return run_workspace_python_script(
            args.get("relative_path", relative_path),
            args.get("content", content),
            session_hash=session_hash,
        )

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
                "pdf_relative_path MUST be a plain string path relative to the session "
                'workspace (e.g. "filename.pdf") — never {}, never a nested object. '
                "Call list_workspace_files first if you do not know the filename. "
                "dest_relative_dir is optional. Returns review_csv_relative_path and "
                "ocr_words_csv_relative_path when available."
            ),
            func=_doc_redact,
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
                "Paths are relative to the session workspace. Use flat string args only."
            ),
            func=_review_apply,
        ),
        StructuredTool.from_function(
            name="verify_coverage",
            description=(
                "Verify Pass 1 redaction coverage on a *_review_file.csv. "
                "Word-level OCR CSV (*ocr_results_with_words*) is auto-discovered beside "
                "the review CSV when ocr_words_csv_relative_path is omitted. "
                "Returns pass_strict and pages needing fixes. "
                "For pre-apply checks, pass only review_csv_relative_path (omit "
                "redacted_pdf_relative_path). For post-apply checks, pass "
                "redacted_pdf_relative_path as the *_redacted.pdf from review_apply — "
                "never the review CSV. "
                "must_redact and must_not_redact: list of regex strings (one term per item), e.g. "
                '["Hyde", "Lauren\\\\s+Lilley", "Poss\\\\b"]. A single pipe-separated string is also accepted.'
            ),
            func=_verify_coverage,
        ),
        StructuredTool.from_function(
            name="read_workspace_text",
            description=(
                "Read a text file (CSV, JSON, .py) from the session workspace. "
                'Args: {"relative_path": "path/to/file.csv"} — flat string, not nested.'
            ),
            func=_read_workspace_text,
        ),
        StructuredTool.from_function(
            name="write_workspace_text",
            description=(
                "Write UTF-8 text into the session workspace (use utf-8-sig for review CSV edits). "
                "Keep content compact (roughly under 24KB / ~80 lines) — prefer short .py scripts "
                "that read OCR/review CSVs and add rows programmatically; avoid huge hard-coded "
                "lists (large/quote-heavy payloads often break tool-call JSON on local models). "
                "After saving a .py file, call run_workspace_python_script immediately — do not "
                "rewrite the same script repeatedly. Review CSV bbox columns must be numeric [0,1]."
            ),
            func=_write_workspace_text,
        ),
        StructuredTool.from_function(
            name="run_workspace_python_script",
            description=(
                "Execute a .py script saved in the session workspace (for pandas CSV policy edits). "
                "Prefer writing the script with write_workspace_text first, then call this with "
                "relative_path only (omit content) so tool args stay small. "
                'Flat args only: {"relative_path": "fix_review.py"}.'
            ),
            func=_run_workspace_python_script,
        ),
    ]
