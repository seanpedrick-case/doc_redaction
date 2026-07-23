"""Detect incomplete redaction turns and build LangGraph auto-continue nudges."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_WORKFLOW_CONTINUE_PROMPT = """Redaction work is NOT complete yet. Continue now:
1. Edit the *_review_file.csv for the user requirements (write ONE .py script, then run_workspace_python_script once)
2. Run verify_coverage until pass_strict is true
3. Run review_apply once on the source PDF and edited review CSV
Call the next required tool — do not stop after read_workspace_text or write_workspace_text.
Keep write_workspace_text / script bodies compact (prefer short Python that derives rows from OCR/CSV — avoid dozens of hard-coded dict literals).
Tool args are flat strings: {"relative_path": "fix_review.py"} — never nested {"relative_path": {"relative_path": "..."}}."""

_TOOL_CALL_JSON_RETRY_PROMPT = """Your previous tool call failed: the inference server could not parse the tool arguments as JSON (usually a truncated or unescaped string inside write_workspace_text content).

Retry with a SHORT approach — do not paste a huge hard-coded row list into tool args:
1. write_workspace_text a compact .py script (ideally under ~80 lines) that reads the review/OCR CSV and adds/filters rows programmatically, OR make a small targeted CSV edit
2. run_workspace_python_script on that script (if you wrote one)
3. verify_coverage until pass_strict, then review_apply once

Use plain flat JSON string arguments only, e.g. {"relative_path": "fix_review.py", "content": "import csv\\n..."}.
Avoid triple-quoted Python docstrings and nested quote-heavy literals in content when possible."""

_IDENTICAL_ERROR_BREAKER_PROMPT = """STOP — you are stuck in a tool-error retry loop.

Your last tool calls returned the same error repeatedly. Do NOT call that tool again with the same (or empty/nested) arguments.

Tool argument format (flat strings — nesting is WRONG):
  Correct:  {{"pdf_relative_path": "file.pdf"}}
  Correct:  {{"relative_path": "fix_review.py", "content": "import csv\\n..."}}
  Wrong:    {{"pdf_relative_path": {{}}}}
  Wrong:    {{"pdf_relative_path": {{"relative_path": "file.pdf"}}}}
  Wrong:    {{"relative_path": {{"relative_path": "fix_review.py"}}}}

If the error says the script was already saved / write-storm: call run_workspace_python_script (NOT write_workspace_text) with only relative_path.

Last repeated error from `{tool_name}`:
{error_preview}

Before any further tool call: write out the exact flat JSON you will use (inner values are plain strings, not objects), then call ONE different next step toward review_apply."""


def build_tool_call_json_retry_prompt() -> str:
    """Nudge after llama.cpp / OpenAI-compatible tool-arg JSON parse failures."""
    return _TOOL_CALL_JSON_RETRY_PROMPT


def _parse_write_workspace_payload(output: str) -> dict[str, Any] | None:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def tool_output_error_signature(output: str) -> str | None:
    """Stable signature for identical tool-error detection, or None if not an error."""
    data = _parse_write_workspace_payload(output)
    if data is None:
        text = (output or "").strip()
        if not text:
            return None
        lowered = text.lower()
        if "error" not in lowered and "failed" not in lowered:
            return None
        return re.sub(r"\s+", " ", text.lower())[:240]
    if data.get("loop_breaker"):
        return f"loop_breaker:{data.get('error') or ''}"[:240]
    err = data.get("error")
    if err is None:
        return None
    return re.sub(r"\s+", " ", str(err).lower())[:240]


def identical_tool_error_streak(
    tool_outputs: list[tuple[str, str]],
    *,
    min_streak: int = 2,
) -> tuple[str, str] | None:
    """Return (tool_name, signature) when the last min_streak errors match."""
    if min_streak < 2 or len(tool_outputs) < min_streak:
        return None
    recent = tool_outputs[-min_streak:]
    names = [name for name, _ in recent]
    if len(set(names)) != 1:
        return None
    sigs: list[str] = []
    for _, output in recent:
        sig = tool_output_error_signature(output)
        if not sig:
            return None
        sigs.append(sig)
    if len(set(sigs)) != 1:
        return None
    return names[0], sigs[0]


def build_identical_error_breaker_prompt(
    tool_name: str,
    tool_outputs: list[tuple[str, str]],
) -> str:
    """Nudge after repeated identical tool errors."""
    preview = ""
    fix_example = ""
    for name, output in reversed(tool_outputs):
        if name != tool_name:
            continue
        data = _parse_write_workspace_payload(output)
        if isinstance(data, dict) and data.get("error"):
            preview = str(data["error"])[:500]
            example = data.get("fix_example")
            if isinstance(example, dict) and example:
                fix_example = (
                    "\n\nUse this exact flat JSON next (copy the path strings):\n"
                    + json.dumps(example)
                )
        else:
            preview = (output or "")[:500]
        break
    # Prefer concrete paths from the latest successful doc_redact if present.
    artifact_hint = ""
    for name, output in reversed(tool_outputs):
        if name != "doc_redact":
            continue
        data = _parse_write_workspace_payload(output)
        if not isinstance(data, dict) or data.get("error"):
            continue
        paths = {
            key: data[key]
            for key in (
                "review_csv_relative_path",
                "ocr_words_csv_relative_path",
            )
            if isinstance(data.get(key), str) and data.get(key)
        }
        if paths:
            artifact_hint = (
                "\n\nKnown paths from doc_redact (use as plain strings):\n"
                + json.dumps(paths)
            )
        break
    return _IDENTICAL_ERROR_BREAKER_PROMPT.format(
        tool_name=tool_name,
        error_preview=(preview or "(see prior tool output)")
        + fix_example
        + artifact_hint,
    )


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
        # Only pending if a newer write happened after the last run.
        last_run_idx = -1
        last_write_idx = -1
        last_write_path: str | None = None
        for idx, (name, output) in enumerate(tool_outputs):
            if name == "run_workspace_python_script":
                last_run_idx = idx
            elif name == "write_workspace_text":
                data = _parse_write_workspace_payload(output)
                written = str((data or {}).get("written") or "")
                if written.lower().endswith(".py"):
                    last_write_idx = idx
                    last_write_path = written
        if last_write_path and last_write_idx > last_run_idx:
            return last_write_path
        return None
    return last_written_python_script(tool_outputs)


def consecutive_python_writes_without_run(
    tool_outputs: list[tuple[str, str]],
) -> tuple[str | None, int]:
    """Count trailing write_workspace_text .py saves since the last script run."""
    count = 0
    path: str | None = None
    for name, output in reversed(tool_outputs):
        if name == "run_workspace_python_script":
            break
        if name != "write_workspace_text":
            if name in {
                "doc_redact",
                "verify_coverage",
                "review_apply",
                "read_workspace_text",
                "list_workspace_files",
            }:
                break
            continue
        data = _parse_write_workspace_payload(output)
        if not data or data.get("error"):
            continue
        written = str(data.get("written") or "")
        if not written.lower().endswith(".py"):
            break
        count += 1
        path = path or written
    return path, count


def build_workflow_continue_prompt(
    tool_names_seen: set[str],
    tool_outputs: list[tuple[str, str]],
) -> str:
    streak = identical_tool_error_streak(tool_outputs, min_streak=2)
    if streak:
        return build_identical_error_breaker_prompt(streak[0], tool_outputs)
    script_path = pending_python_script(tool_names_seen, tool_outputs)
    if script_path:
        return (
            "Redaction work is NOT complete. The Python script is already saved at "
            f"`{script_path}` — do NOT call write_workspace_text again. "
            f"Call run_workspace_python_script with relative_path={script_path!r} "
            "now, then verify_coverage and review_apply."
        )
    path, write_count = consecutive_python_writes_without_run(tool_outputs)
    if path and write_count >= 2:
        return (
            "Redaction work is NOT complete. You have rewritten the same Python script "
            f"({path}) {write_count} times without running it. "
            f"Call run_workspace_python_script with relative_path={path!r} NOW — "
            "do not write again."
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


def langgraph_identical_error_stop_streak() -> int:
    """Stop / break after this many identical consecutive tool errors (default 3)."""
    raw = os.environ.get("LANGGRAPH_IDENTICAL_ERROR_STOP", "3").strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return 3


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
