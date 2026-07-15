"""Detect incomplete redaction turns and build LangGraph auto-continue nudges."""

from __future__ import annotations

import json
import os
from typing import Any

_WORKFLOW_CONTINUE_PROMPT = """Redaction work is NOT complete yet. Continue now:
1. Edit the *_review_file.csv for the user requirements (write_workspace_text or run_workspace_python_script)
2. Run verify_coverage until pass_strict is true
3. Run review_apply once on the source PDF and edited review CSV
Call the next required tool — do not stop after read_workspace_text or write_workspace_text.
Keep write_workspace_text / script bodies compact (prefer short Python that derives rows from OCR/CSV — avoid dozens of hard-coded dict literals)."""

_TOOL_CALL_JSON_RETRY_PROMPT = """Your previous tool call failed: the inference server could not parse the tool arguments as JSON (usually a truncated or unescaped string inside write_workspace_text content).

Retry with a SHORT approach — do not paste a huge hard-coded row list into tool args:
1. write_workspace_text a compact .py script (ideally under ~80 lines) that reads the review/OCR CSV and adds/filters rows programmatically, OR make a small targeted CSV edit
2. run_workspace_python_script on that script (if you wrote one)
3. verify_coverage until pass_strict, then review_apply once

Use plain JSON string arguments only. Avoid triple-quoted Python docstrings and nested quote-heavy literals in content when possible."""


def build_tool_call_json_retry_prompt() -> str:
    """Nudge after llama.cpp / OpenAI-compatible tool-arg JSON parse failures."""
    return _TOOL_CALL_JSON_RETRY_PROMPT


def _parse_write_workspace_payload(output: str) -> dict[str, Any] | None:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def last_written_python_script(tool_outputs: list[tuple[str, str]]) -> str | None:
    for name, output in reversed(tool_outputs):
        if name != "write_workspace_text":
            continue
        data = _parse_write_workspace_payload(output)
        if not data:
            continue
        written = str(data.get("written") or "")
        if written.lower().endswith(".py"):
            return written
    return None


def wrote_review_csv(tool_outputs: list[tuple[str, str]]) -> bool:
    for name, output in tool_outputs:
        if name != "write_workspace_text":
            continue
        data = _parse_write_workspace_payload(output)
        if not data:
            continue
        written = str(data.get("written") or "").lower().replace("\\", "/")
        if written.endswith(".csv") and (
            "review_file" in written or written.endswith("_review.csv")
        ):
            return True
    return False


def pending_python_script(
    tool_names_seen: set[str],
    tool_outputs: list[tuple[str, str]],
) -> str | None:
    if "run_workspace_python_script" in tool_names_seen:
        return None
    return last_written_python_script(tool_outputs)


def build_workflow_continue_prompt(
    tool_names_seen: set[str],
    tool_outputs: list[tuple[str, str]],
) -> str:
    script_path = pending_python_script(tool_names_seen, tool_outputs)
    if script_path:
        return (
            "Redaction work is NOT complete. The Python script is already saved at "
            f"`{script_path}` — do NOT call write_workspace_text again. "
            f"Call run_workspace_python_script with relative_path={script_path!r} "
            "now, then verify_coverage and review_apply."
        )
    return _WORKFLOW_CONTINUE_PROMPT


def langgraph_auto_continue_enabled() -> bool:
    return os.environ.get(
        "LANGGRAPH_AUTO_CONTINUE_WORKFLOW", "true"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def langgraph_max_continuations() -> int:
    raw = os.environ.get("LANGGRAPH_WORKFLOW_CONTINUATIONS", "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def redaction_workflow_incomplete(
    tool_names: set[str],
    tool_outputs: list[tuple[str, str]] | None = None,
) -> bool:
    """True when this turn started redaction work but stopped before review_apply.

    Covers Pass 1 (doc_redact without review_apply) and follow-ups that write a
    pending .py script, run a workspace script, or edit a review CSV without apply.
    Explore-only turns (list/read) are treated as complete.
    """
    if "review_apply" in tool_names:
        return False
    if "doc_redact" in tool_names:
        return True
    outputs = tool_outputs or []
    if pending_python_script(tool_names, outputs):
        return True
    if "run_workspace_python_script" in tool_names:
        return True
    if wrote_review_csv(outputs):
        return True
    return False
