"""Persist active Gradio redaction task state for multi-turn post-run eval."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from session_workspace import (
    read_active_redaction_task_file,
    session_workspace_dir,
    write_active_redaction_task_file,
)

_LOCK = threading.Lock()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def _ensure_eval_imports() -> None:
    text = str(_EVAL_DIR)
    if text not in sys.path:
        sys.path.insert(0, text)


@dataclass
class ActiveRedactionTask:
    document_name: str
    task_instructions: str
    follow_up_steer_count: int = 0
    cumulative_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    eval_gold_path: str | None = None
    eval_gold_match_level: str = "none"
    eval_gold_match_reason: str = ""
    encourage_vlm_faces: bool = False
    encourage_vlm_signatures: bool = False


def _task_from_raw(raw: dict[str, Any]) -> ActiveRedactionTask | None:
    doc = str(raw.get("document_name") or "").strip()
    if not doc:
        return None
    instructions = str(raw.get("task_instructions") or "").strip()
    count = raw.get("follow_up_steer_count", 0)
    try:
        follow_up_steer_count = max(0, int(count))
    except (TypeError, ValueError):
        follow_up_steer_count = 0
    calls = raw.get("cumulative_tool_calls")
    if not isinstance(calls, list):
        calls = []
    cumulative_tool_calls = [
        c for c in calls if isinstance(c, dict) and str(c.get("tool") or "").strip()
    ]
    eval_gold_path = str(raw.get("eval_gold_path") or "").strip() or None
    return ActiveRedactionTask(
        document_name=doc,
        task_instructions=instructions,
        follow_up_steer_count=follow_up_steer_count,
        cumulative_tool_calls=cumulative_tool_calls,
        eval_gold_path=eval_gold_path,
        eval_gold_match_level=str(raw.get("eval_gold_match_level") or "none"),
        eval_gold_match_reason=str(raw.get("eval_gold_match_reason") or ""),
        encourage_vlm_faces=bool(raw.get("encourage_vlm_faces")),
        encourage_vlm_signatures=bool(raw.get("encourage_vlm_signatures")),
    )


def _read_task(session_hash: str) -> ActiveRedactionTask | None:
    raw = read_active_redaction_task_file(session_hash)
    if raw is None:
        return None
    return _task_from_raw(raw)


def _write_task(session_hash: str, task: ActiveRedactionTask) -> bool:
    payload = {
        "document_name": task.document_name,
        "task_instructions": task.task_instructions,
        "follow_up_steer_count": task.follow_up_steer_count,
        "cumulative_tool_calls": task.cumulative_tool_calls,
        "eval_gold_path": task.eval_gold_path,
        "eval_gold_match_level": task.eval_gold_match_level,
        "eval_gold_match_reason": task.eval_gold_match_reason,
        "encourage_vlm_faces": bool(task.encourage_vlm_faces),
        "encourage_vlm_signatures": bool(task.encourage_vlm_signatures),
    }
    return write_active_redaction_task_file(session_hash, payload)


def _resolve_pdf_path(
    session_hash: str, document_name: str, pdf_path: Path | None
) -> Path | None:
    if pdf_path is not None and pdf_path.is_file():
        return pdf_path
    candidate = session_workspace_dir(session_hash) / document_name
    return candidate if candidate.is_file() else None


def begin_redaction_task(
    session_hash: str,
    document_name: str,
    task_instructions: str,
    *,
    pdf_path: Path | None = None,
    encourage_vlm_faces: bool = False,
    encourage_vlm_signatures: bool = False,
) -> str:
    """
    Start a new redaction task in *session_hash* (resets steer count and tool trace).

    Returns a short status line for the activity log (gold match result when auto-gold on).
    """
    doc = (document_name or "").strip()
    if not doc or not (session_hash or "").strip():
        return ""
    task = ActiveRedactionTask(
        document_name=doc,
        task_instructions=(task_instructions or "").strip(),
        follow_up_steer_count=0,
        cumulative_tool_calls=[],
        encourage_vlm_faces=bool(encourage_vlm_faces),
        encourage_vlm_signatures=bool(encourage_vlm_signatures),
    )
    log_line = ""
    try:
        _ensure_eval_imports()
        from agent_eval_env import agent_eval_auto_gold_enabled, resolve_gold_root
        from gold_match import resolve_gold_case_for_task

        if agent_eval_auto_gold_enabled():
            upload = _resolve_pdf_path(session_hash, doc, pdf_path)
            if upload is not None:
                result = resolve_gold_case_for_task(
                    pdf_path=upload,
                    document_name=doc,
                    task_instructions=task.task_instructions,
                    gold_root=resolve_gold_root(_REPO_ROOT),
                )
                task.eval_gold_match_level = result.match_level
                task.eval_gold_match_reason = result.reason
                if result.gold_dir is not None:
                    task.eval_gold_path = str(result.gold_dir.resolve())
                    log_line = (
                        f"**Eval gold case:** `{result.case_id or result.gold_dir.name}` "
                        f"({result.match_level})"
                    )
                else:
                    log_line = (
                        f"**Eval skipped:** {result.reason or 'no matching gold case'}"
                    )
            else:
                task.eval_gold_match_reason = (
                    "upload PDF not found in session workspace"
                )
                log_line = f"**Eval skipped:** {task.eval_gold_match_reason}"
    except Exception as exc:
        task.eval_gold_match_reason = f"gold match error: {exc}"
        log_line = f"**Eval skipped:** {task.eval_gold_match_reason}"

    with _LOCK:
        _write_task(session_hash, task)
    return log_line


def note_follow_up_or_steer_message(session_hash: str) -> bool:
    """
    Record one user follow-up or in-run steer message for the active task.

    Returns True when an active task was updated.
    """
    if not (session_hash or "").strip():
        return False
    with _LOCK:
        task = _read_task(session_hash)
        if task is None:
            return False
        task.follow_up_steer_count += 1
        return _write_task(session_hash, task)


def _tool_calls_from_events(events: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        kind = str(getattr(event, "kind", "") or "")
        if kind != "tool_start":
            continue
        name = str(getattr(event, "tool_name", None) or "tool")
        args = getattr(event, "tool_args", None)
        if not isinstance(args, dict):
            args = {}
        calls.append({"tool": name, "args": args})
    return calls


def append_turn_tool_calls(session_hash: str, events: list[Any]) -> None:
    """Merge tool-start events from the latest agent turn into the task trace."""
    if not (session_hash or "").strip() or not events:
        return
    new_calls = _tool_calls_from_events(events)
    if not new_calls:
        return
    with _LOCK:
        task = _read_task(session_hash)
        if task is None:
            return
        task.cumulative_tool_calls.extend(new_calls)
        _write_task(session_hash, task)


def get_active_task_for_eval(session_hash: str) -> ActiveRedactionTask | None:
    """Return the current task snapshot for post-run eval (or None)."""
    if not (session_hash or "").strip():
        return None
    return _read_task(session_hash)
