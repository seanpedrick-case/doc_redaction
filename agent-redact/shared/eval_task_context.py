"""Persist active Gradio redaction task state for multi-turn post-run eval."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from session_workspace import (
    read_active_redaction_task_file,
    write_active_redaction_task_file,
)

_LOCK = threading.Lock()


@dataclass
class ActiveRedactionTask:
    document_name: str
    task_instructions: str
    follow_up_steer_count: int = 0
    cumulative_tool_calls: list[dict[str, Any]] = field(default_factory=list)


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
    return ActiveRedactionTask(
        document_name=doc,
        task_instructions=instructions,
        follow_up_steer_count=follow_up_steer_count,
        cumulative_tool_calls=cumulative_tool_calls,
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
    }
    return write_active_redaction_task_file(session_hash, payload)


def begin_redaction_task(
    session_hash: str,
    document_name: str,
    task_instructions: str,
) -> None:
    """Start a new redaction task in *session_hash* (resets steer count and tool trace)."""
    doc = (document_name or "").strip()
    if not doc or not (session_hash or "").strip():
        return
    task = ActiveRedactionTask(
        document_name=doc,
        task_instructions=(task_instructions or "").strip(),
        follow_up_steer_count=0,
        cumulative_tool_calls=[],
    )
    with _LOCK:
        _write_task(session_hash, task)


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
