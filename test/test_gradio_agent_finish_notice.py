"""Finish-banner copy for LangGraph clarification pauses."""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "agent-redact" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import gradio_app as ga


def test_clarification_finish_notice():
    text = ga._agent_finish_chat_notice(awaiting_clarification=True)
    assert "your answer is needed" in text.lower()
    assert "follow-up / steer chat" in text.lower()
    assert "agent finished" not in text.lower()


def test_complete_finish_notice_unchanged():
    text = ga._agent_finish_chat_notice()
    assert "**Agent finished**" in text
