"""Persist active Gradio redaction task state for multi-turn post-run eval."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_ACTIVE_REL = Path("eval") / "active_redaction_task.json"


@dataclass
class ActiveRedactionTask:
    document_name: str
    task_instructions: str
    follow_up_steer_count: int = 0
    cumulative_tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _active_task_path(session_hash: str) -> Path:
    from session_workspace import session_workspace_dir

    return session_workspace_dir(session_hash) / _ACTIVE_REL


def _read_task(path: Path) -> ActiveRedactionTask | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
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


def _write_task(path: Path, task: ActiveRedactionTask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_name": task.document_name,
        "task_instructions": task.task_instructions,
        "follow_up_steer_count": task.follow_up_steer_count,
        "cumulative_tool_calls": task.cumulative_tool_calls,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    path = _active_task_path(session_hash)
    with _LOCK:
        _write_task(path, task)


def note_follow_up_or_steer_message(session_hash: str) -> bool:
    """
    Record one user follow-up or in-run steer message for the active task.

    Returns True when an active task was updated.
    """
    if not (session_hash or "").strip():
        return False
    path = _active_task_path(session_hash)
    with _LOCK:
        task = _read_task(path)
        if task is None:
            return False
        task.follow_up_steer_count += 1
        _write_task(path, task)
        return True


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
    path = _active_task_path(session_hash)
    with _LOCK:
        task = _read_task(path)
        if task is None:
            return
        task.cumulative_tool_calls.extend(new_calls)
        _write_task(path, task)


def get_active_task_for_eval(session_hash: str) -> ActiveRedactionTask | None:
    """Return the current task snapshot for post-run eval (or None)."""
    if not (session_hash or "").strip():
        return None
    return _read_task(_active_task_path(session_hash))
