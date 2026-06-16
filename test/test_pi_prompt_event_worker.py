"""Tests for Pi prompt event worker startup (rate-limit retry support)."""

import queue
import sys
import time
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_rpc_client import PiStreamEvent, start_pi_prompt_event_worker


def test_start_pi_prompt_event_worker_feeds_queue():
    """Worker thread must enqueue events so quota retries can resume."""

    events = [
        PiStreamEvent(kind="status", text="Turn started."),
        PiStreamEvent(kind="done", text="Agent finished."),
    ]

    class _Client:
        def prompt_events(self, message: str):
            assert "Continue the redaction task" in message
            yield from events

    event_queue: queue.Queue = queue.Queue()
    start_pi_prompt_event_worker(_Client(), event_queue, "Continue the redaction task")

    collected: list[PiStreamEvent | None] = []
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            item = event_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        collected.append(item)
        if item is None:
            break

    assert collected[:-1] == events
    assert collected[-1] is None
