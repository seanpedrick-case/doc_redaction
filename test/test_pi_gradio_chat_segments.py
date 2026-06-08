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
    _chat_yield,
    _ensure_pi_client_for_redaction,
    _finalize_assistant_chat,
    _format_queue_update_activity,
    _initial_chat_outputs_on_page_load,
    _is_agent_finish_notice_only,
    _passthrough_chat_outputs,
    _passthrough_chat_outputs_for_notify,
    _pi_agent_is_streaming,
    _pi_wait_until_idle,
    _refresh_pi_client_model,
    _reset_pi_on_page_load,
    _reset_pi_rpc_client,
    _should_queue_agent_message,
)
from pi_rpc_client import PiRpcClient, PiStreamEvent


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


class _FakePiClient:
    def __init__(
        self,
        *,
        running: bool = True,
        streaming: bool = False,
        prompt_stream_active: bool = False,
    ):
        self._running = running
        self._streaming = streaming
        self._prompt_stream_active = prompt_stream_active

    @property
    def running(self) -> bool:
        return self._running

    @property
    def prompt_stream_active(self) -> bool:
        return self._prompt_stream_active

    def get_state(self) -> dict:
        return {"isStreaming": self._streaming}


def test_chat_yield_leaves_message_box_unchanged_during_stream_updates():
    class _Client:
        running = True

        def get_state(self) -> dict:
            return {
                "sessionFile": "x.jsonl",
                "isStreaming": True,
                "isCompacting": False,
            }

    out = _chat_yield([], _Client(), [], "", "", "")
    # Third output is the follow-up message textbox.
    assert out[2] == gr.update()


def test_chat_yield_clears_message_box_when_requested():
    class _Client:
        running = True

        def get_state(self) -> dict:
            return {
                "sessionFile": "x.jsonl",
                "isStreaming": False,
                "isCompacting": False,
            }

    out = _chat_yield([], _Client(), [], "", "", "", msg="")
    assert out[2] == ""


def test_reset_pi_rpc_client_restarts_running_client(monkeypatch):
    class _Client(PiRpcClient):
        @property
        def running(self) -> bool:
            return True

        def get_state(self) -> dict:
            return {"isStreaming": False}

        def close(self) -> None:
            pass

    restarted = object()
    monkeypatch.setattr("gradio_app._coerce_client", lambda c: c)
    monkeypatch.setattr(
        "gradio_app._restart_pi_rpc_client",
        lambda session_hash, prior=None: restarted,
    )
    assert _reset_pi_rpc_client(_Client(), "sess") is restarted
    assert _reset_pi_on_page_load(_Client(), "sess") is restarted


def test_reset_pi_rpc_client_noop_when_client_none():
    assert _reset_pi_rpc_client(None, "sess") is None


def test_ensure_pi_client_for_redaction_uses_ensure_when_no_prior_client(
    monkeypatch,
):
    ensured = object()
    monkeypatch.setattr("gradio_app._reset_pi_rpc_client", lambda _c, _h: None)
    monkeypatch.setattr("gradio_app._ensure_client", lambda _c, _h: ensured)
    assert _ensure_pi_client_for_redaction(None, "sess") is ensured


def test_initial_chat_outputs_on_page_load_clears_chatbot(monkeypatch):
    class _Client:
        running = True

        def get_state(self) -> dict:
            return {
                "sessionFile": "x.jsonl",
                "isStreaming": False,
                "isCompacting": False,
            }

    monkeypatch.setattr("gradio_app._coerce_client", lambda c: c)
    monkeypatch.setattr(
        "gradio_app.collect_final_output_files",
        lambda _h: [],
    )
    monkeypatch.setattr(
        "gradio_app.latest_redacted_pdf_path",
        lambda _h: None,
    )
    client = _Client()
    outputs = _initial_chat_outputs_on_page_load(client, "sess")
    assert len(outputs) == _CHAT_OUTPUT_COMPONENT_COUNT
    assert outputs[0] == []
    assert outputs[1] is client


def test_is_agent_finish_notice_only():
    assert _is_agent_finish_notice_only("**Agent finished** — the task is complete.")
    assert not _is_agent_finish_notice_only("Here is a summary of what was redacted.")


def test_pi_wait_until_idle_waits_for_streaming_to_clear(monkeypatch):
    class _Client:
        running = True
        calls = 0

        @property
        def prompt_stream_active(self) -> bool:
            return False

        def get_state(self) -> dict:
            self.calls += 1
            return {"isStreaming": self.calls < 3}

    client = _Client()
    monkeypatch.setattr("gradio_app._coerce_client", lambda c: c)
    monkeypatch.setattr("gradio_app._PI_IDLE_POLL_INTERVAL_S", 0.0)
    assert _pi_wait_until_idle(client, max_wait_s=1.0) is True
    assert client.calls >= 3


def test_finalize_assistant_chat_llama_shows_failure_when_only_finish_banner(
    monkeypatch,
):
    monkeypatch.setattr("gradio_app.normalize_provider", lambda _p: "llama-cpp")

    class _Client:
        def get_messages(self) -> list:
            return [
                {"role": "user", "content": "What was redacted?"},
                {"role": "assistant", "content": []},
            ]

    history = [
        {"role": "user", "content": "What was redacted?"},
        {
            "role": "assistant",
            "content": "**Agent finished** — the task is complete.",
        },
    ]
    _finalize_assistant_chat(
        _Client(),
        history,
        completed_segments=[],
        streaming_text="",
        activity=["Prompt sent."],
    )
    assert "no response from the orchestration model" in history[-1]["content"]


def test_schedule_post_pi_task_runs_off_hot_path(monkeypatch):
    calls: list[dict] = []

    class _ImmediateThread:
        def __init__(self, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("gradio_app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr(
        "gradio_app._after_pi_task",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "gradio_app.usage_for_completed_turn",
        lambda _client, _baseline: type(
            "Usage",
            (),
            {"llm_input_tokens": 10, "llm_output_tokens": 5},
        )(),
    )
    from gradio_app import _schedule_post_pi_task

    _schedule_post_pi_task(
        session_hash="sess",
        client=None,
        s3_output_folder="s3://bucket/",
        save_outputs_to_s3=True,
        document_name="doc.pdf",
    )
    assert calls
    assert calls[0]["session_hash"] == "sess"
    assert calls[0]["llm_input_tokens"] == 10


def test_should_queue_only_while_pi_streaming(monkeypatch):
    monkeypatch.setattr(
        "gradio_app._coerce_client",
        lambda client: client,
    )
    live = _FakePiClient(streaming=True, prompt_stream_active=True)
    idle = _FakePiClient(streaming=False, prompt_stream_active=True)
    stale = _FakePiClient(streaming=True, prompt_stream_active=False)
    assert _should_queue_agent_message(live, message="tweak page 3") is True
    assert _should_queue_agent_message(idle, message="tweak page 3") is False
    assert _should_queue_agent_message(stale, message="tweak page 3") is False
    assert _should_queue_agent_message(idle, message="") is False
    assert _pi_agent_is_streaming(None) is False
    assert _pi_agent_is_streaming(_FakePiClient(running=False, streaming=True)) is False


def test_refresh_pi_client_model_calls_set_model(monkeypatch):
    class _Client:
        calls: list[tuple[str, str]] = []

        def set_model(self, provider: str, model: str) -> None:
            self.calls.append((provider, model))

    client = _Client()
    monkeypatch.setattr("gradio_app.normalize_provider", lambda _p: "llama-cpp")
    monkeypatch.setattr("gradio_app.resolved_default_model", lambda _p: "gemma_4_31b")
    _refresh_pi_client_model(client)
    assert client.calls == [("llama-cpp", "gemma_4_31b")]


def test_pi_wait_until_idle_skips_during_active_prompt_stream(monkeypatch):
    class _Client:
        running = True
        state_calls = 0

        @property
        def prompt_stream_active(self) -> bool:
            return True

        def get_state(self) -> dict:
            self.state_calls += 1
            return {"isStreaming": True}

    client = _Client()
    monkeypatch.setattr("gradio_app._coerce_client", lambda c: c)
    assert _pi_wait_until_idle(client, max_wait_s=0.0) is False
    assert client.state_calls == 0


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


def test_passthrough_chat_outputs_for_notify_skips_file_slots():
    non_file_count = _CHAT_OUTPUT_COMPONENT_COUNT - 3
    values = tuple(f"v{i}" for i in range(non_file_count))
    result = _passthrough_chat_outputs_for_notify(*values)
    assert len(result) == _CHAT_OUTPUT_COMPONENT_COUNT
    assert result[10] == gr.skip()
    assert result[11] == gr.skip()
    assert result[12] == gr.skip()
    assert result[0] == "v0"
    assert result[9] == "v9"
    assert result[13] == "v10"


def test_passthrough_chat_outputs_for_notify_pads_when_cancelled_empty():
    result = _passthrough_chat_outputs_for_notify()
    assert len(result) == _CHAT_OUTPUT_COMPONENT_COUNT
    assert result[10] == gr.skip()
    assert result[11] == gr.skip()
    assert result[12] == gr.skip()


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
    assert history[-2]["role"] == "user"
    assert "Steer" in history[-2]["content"]
    assert "Only redact names" in history[-2]["content"]
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == ""
    assert any("Steer queued" in line for line in activity)


def test_apply_event_text_snapshot_after_steer_updates_assistant_not_user():
    history = [
        {"role": "user", "content": "Start task"},
        {"role": "assistant", "content": "Working…"},
        {"role": "user", "content": "_**Steer:**_ Only redact names"},
        {"role": "assistant", "content": ""},
    ]
    activity: list[str] = []
    event = PiStreamEvent(kind="text_snapshot", text="Acknowledged — adjusting scope.")
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
    assert history[-2]["content"].startswith("_**Steer:**_")
    assert history[-1]["role"] == "assistant"
    assert "Acknowledged" in history[-1]["content"]
