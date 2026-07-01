"""Curated LangGraph tools for doc_redaction orchestration (no shell)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

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


def _resolve_workspace_path(session_hash: str | None, relative_path: str) -> Path:
    root = _session_root(session_hash).resolve()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"Path escapes session workspace: {relative_path}")
    return candidate


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
    pdf = _resolve_workspace_path(session_hash, pdf_relative_path)
    dest = _resolve_workspace_path(session_hash, dest_relative_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result, saved = call_doc_redact(
        pdf,
        dest,
        ocr_method=ocr_method or os.environ.get("PI_DEFAULT_OCR_METHOD"),
        pii_method=pii_method or os.environ.get("PI_DEFAULT_PII_METHOD"),
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
    relative_path: str,
    *,
    session_hash: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Read a UTF-8 text file from the session workspace (CSV, JSON, Python script)."""
    try:
        path = _resolve_workspace_path(session_hash, relative_path)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"error": str(exc), "relative_path": relative_path})
    if not path.is_file():
        return json.dumps({"error": f"File not found: {relative_path}"})
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
    return path.read_text(encoding="utf-8-sig")


def write_workspace_text(
    relative_path: str,
    content: str,
    *,
    session_hash: str | None = None,
) -> str:
    """Write UTF-8 text into the session workspace (preserve utf-8-sig for review CSVs)."""
    path = _resolve_workspace_path(session_hash, relative_path)
    if len(content.encode("utf-8")) > _MAX_TEXT_BYTES:
        return json.dumps({"error": f"Content too large (>{_MAX_TEXT_BYTES} bytes)."})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig")
    root = _session_root(session_hash)
    return json.dumps(
        {
            "written": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
        }
    )


def run_workspace_python_script(
    relative_path: str,
    *,
    session_hash: str | None = None,
) -> str:
    """Run a Python script already saved under the session workspace."""
    path = _resolve_workspace_path(session_hash, relative_path)
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

    pdf = _resolve_workspace_path(session_hash, pdf_relative_path)
    review_csv = _resolve_workspace_path(session_hash, review_csv_relative_path)
    dest = _resolve_workspace_path(session_hash, dest_relative_dir)
    dest.mkdir(parents=True, exist_ok=True)

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
            func=lambda relative_path: run_workspace_python_script(
                relative_path, session_hash=session_hash
            ),
        ),
    ]
