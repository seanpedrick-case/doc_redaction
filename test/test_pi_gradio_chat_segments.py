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
    _chat_segment_tool_label,
    _passthrough_chat_outputs,
)


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
    assert all(result[i] == gr.skip() for i in range(2, _CHAT_OUTPUT_COMPONENT_COUNT))
