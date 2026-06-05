"""Tests for Pi Gradio chat segment deduplication (in-progress tool snapshots)."""

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

import gradio as gr
from gradio_app import (
    _CHAT_OUTPUT_COMPONENT_COUNT,
    _append_chat_segment,
    _apply_event,
    _chat_segment_tool_label,
    _format_queue_update_activity,
    _passthrough_chat_outputs,
)
from pi_rpc_client import PiStreamEvent


def test_chat_segment_tool_label_bash_and_bare():
    assert _chat_segment_tool_label("**bash:** `ls`") == "bash"
    assert _chat_segment_tool_label("**tool**") == "tool"


def test_append_chat_segment_replaces_growing_bash_snapshot():
    done: list[str] = []
    stream = ""
    snapshots = [
        "**bash:** `cd /ho`",
        "**bash:** `cd /home/user`",
        "**bash:** `cd /home/user/app && ls`",
    ]
    for snap in snapshots:
        done, stream = _append_chat_segment(done, stream, snap)
    assert done == ["**bash:** `cd /home/user/app && ls`"]


def test_append_chat_segment_skips_empty_command_until_content():
    done: list[str] = []
    stream = ""
    done, stream = _append_chat_segment(done, stream, '**bash:** `{"command": ""}`')
    assert done == []
    done, stream = _append_chat_segment(done, stream, "**bash:** `ls`")
    assert len(done) == 1
    assert "ls" in done[0]


def test_append_chat_segment_replaces_bare_tool_with_named_tool():
    done: list[str] = []
    stream = ""
    done, stream = _append_chat_segment(done, stream, "**tool**")
    done, stream = _append_chat_segment(done, stream, "**bash:** `pwd`")
    assert done == ["**bash:** `pwd`"]


def test_append_chat_segment_keeps_distinct_tools():
    done: list[str] = []
    stream = ""
    done, stream = _append_chat_segment(done, stream, "**read:** `skills/foo.md`")
    done, stream = _append_chat_segment(done, stream, "**bash:** `ls`")
    assert len(done) == 2
    assert done[0].startswith("**read:**")
    assert done[1].startswith("**bash:**")


def test_passthrough_chat_outputs_returns_all_values():
    values = tuple(range(_CHAT_OUTPUT_COMPONENT_COUNT))
    assert _passthrough_chat_outputs(*values) == values


def test_passthrough_chat_outputs_empty_returns_skip_tuple():
    result = _passthrough_chat_outputs()
    assert len(result) == _CHAT_OUTPUT_COMPONENT_COUNT
    assert all(v == gr.skip() for v in result)


def test_passthrough_chat_outputs_pads_partial_values():
    result = _passthrough_chat_outputs("a", "b")
    assert len(result) == _CHAT_OUTPUT_COMPONENT_COUNT
    assert result[0] == "a"
    assert result[1] == "b"


def test_format_queue_update_activity_steering_and_follow_up():
    lines = _format_queue_update_activity(
        ["Stop and fix page 3"],
        ["Summarise when done"],
    )
    assert len(lines) == 2
    assert "Steer queued" in lines[0]
    assert "Follow-up queued" in lines[1]


def test_apply_event_queue_update_appends_user_messages():
    history = [
        {"role": "user", "content": "Start task"},
        {"role": "assistant", "content": "Working…"},
    ]
    activity: list[str] = []
    event = PiStreamEvent(
        kind="queue_update",
        meta={"steering": ["Only redact names"], "follow_up": []},
    )
    history, activity, *_rest = _apply_event(
        event,
        history=history,
        activity=activity,
        thinking="",
        tool_output="",
        tool_heading="",
        completed_segments=[],
        streaming_text="",
    )
    assert history[-1]["role"] == "user"
    assert "Steer" in history[-1]["content"]
    assert "Only redact names" in history[-1]["content"]
    assert any("Steer queued" in line for line in activity)
