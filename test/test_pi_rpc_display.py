"""Tests for Pi RPC assistant display extraction (Gemini reasoning blocks)."""

from pi_test_support import ensure_agent_redact_paths

ensure_agent_redact_paths()

from pi_rpc_client import (
    PiRpcClient,
    assistant_chat_text,
    assistant_text_since_last_user,
    chat_text_from_assistant_message,
    extract_assistant_display,
    format_assistant_message_for_chat,
    is_rate_limit_error,
    last_assistant_turn_error,
)


def test_extract_assistant_display_text_blocks():
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
    }
    visible, thinking = extract_assistant_display(message)
    assert visible == "Hello"
    assert thinking == ""


def test_extract_assistant_display_thinking_only():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "Planning redaction…"}],
    }
    visible, thinking = extract_assistant_display(message)
    assert visible == ""
    assert thinking == "Planning redaction…"
    assert assistant_chat_text(visible, thinking) == "Planning redaction…"


def test_extract_assistant_display_reasoning_block_with_text_field():
    message = {
        "role": "assistant",
        "content": [{"type": "reasoning", "text": "Step one complete."}],
    }
    visible, thinking = extract_assistant_display(message)
    assert visible == ""
    assert thinking == "Step one complete."


def test_assistant_chat_text_prefers_visible():
    assert assistant_chat_text("Answer", "Reasoning") == "Answer"


def test_format_assistant_message_for_chat_bash_commentary_as_prose():
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "name": "bash",
                "arguments": {
                    "command": "# Planning next step\n# Will call doc_redact"
                },
            },
        ],
    }
    rendered = format_assistant_message_for_chat(message)
    assert "**bash:**" not in rendered
    assert "Planning next step" in rendered


def test_format_assistant_message_for_chat_tool_only():
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "name": "bash",
                "arguments": {"command": "ls -F skills/doc-redaction-app/"},
            },
            {"type": "text", "text": ""},
        ],
    }
    rendered = format_assistant_message_for_chat(message)
    assert "**bash:**" in rendered
    assert "skills/doc-redaction-app" in rendered


def test_format_assistant_message_for_chat_skips_thinking_only():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "Planning redaction…"}],
    }
    assert format_assistant_message_for_chat(message) == ""


def test_chat_text_from_assistant_message_gemini_tool_turn():
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "id": "p74ovciq",
                "name": "read",
                "arguments": {"path": "skills/doc-redaction-app/SKILL.md"},
            },
            {"type": "text", "text": ""},
        ],
    }
    text = chat_text_from_assistant_message(message)
    assert "**read:**" in text
    assert "doc-redaction-app" in text


def test_assistant_text_since_last_user():
    messages = [
        {"role": "user", "content": "Redact this PDF"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "name": "bash",
                    "arguments": {"command": "env | grep DOC_REDACTION"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done — see workspace outputs."}],
        },
    ]
    text = assistant_text_since_last_user(messages)
    assert "DOC_REDACTION" in text
    assert "Done — see workspace outputs." in text


def test_is_rate_limit_error_detects_gemini_quota():
    err = (
        '{"error":{"code":429,"message":"You exceeded your current quota",'
        '"status":"RESOURCE_EXHAUSTED"}}'
    )
    assert is_rate_limit_error(err)


def test_is_rate_limit_error_detects_bedrock_throttling():
    err = "ThrottlingException: Too many requests, please wait before trying again."
    assert is_rate_limit_error(err)
    assert is_rate_limit_error("ServiceQuotaExceededException: Limit exceeded")


def test_is_rate_limit_error_rejects_unrelated():
    assert not is_rate_limit_error("connection refused")
    assert not is_rate_limit_error(None)


def test_last_assistant_turn_error_from_error_message():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "stopReason": "error",
            "errorMessage": "429 Too Many Requests quota exceeded",
            "content": [],
        },
    ]
    assert last_assistant_turn_error(messages) == "429 Too Many Requests quota exceeded"


def test_follow_up_increments_pending_delivery_counter(monkeypatch):
    client = PiRpcClient()
    commands: list[dict] = []
    monkeypatch.setattr(
        client,
        "_send_command",
        lambda command, **kwargs: commands.append(command),
    )
    client.follow_up("After you finish, run redaction")
    assert client._pending_follow_ups == 1
    assert commands[-1]["type"] == "follow_up"
    client.follow_up("Also verify coverage")
    assert client._pending_follow_ups == 2


def test_iter_agent_events_waits_for_agent_settled(monkeypatch):
    client = PiRpcClient()
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_GRACE_S", 5.0)
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_MAX_S", 30.0)
    client._events.put({"type": "agent_end", "messages": []})
    client._events.put({"type": "compaction_start", "reason": "context limit"})
    client._events.put({"type": "compaction_end", "result": {"tokensBefore": 1000}})
    client._events.put({"type": "agent_settled"})

    kinds = [event.kind for event in client._iter_agent_events()]
    assert kinds == ["compaction_start", "compaction_end", "done"]


def test_iter_agent_events_grace_finishes_without_settled(monkeypatch):
    client = PiRpcClient()
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_GRACE_S", 0.01)
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_MAX_S", 1.0)
    client._events.put({"type": "agent_end", "messages": []})

    events = list(client._iter_agent_events())
    assert len(events) == 1
    assert events[0].kind == "done"
    assert events[0].text == "Agent finished."


def test_iter_agent_events_will_retry_continues(monkeypatch):
    client = PiRpcClient()
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_GRACE_S", 0.01)
    monkeypatch.setattr("pi_rpc_client._AGENT_SETTLE_MAX_S", 1.0)
    client._events.put({"type": "agent_end", "willRetry": True, "messages": []})
    client._events.put({"type": "agent_start"})
    client._events.put({"type": "agent_end", "messages": []})
    client._events.put({"type": "agent_settled"})

    events = list(client._iter_agent_events())
    kinds = [event.kind for event in events]
    assert kinds[0] == "status"
    assert "retry" in events[0].text.lower()
    assert "Agent started" in events[1].text
    assert kinds[-1] == "done"
    assert kinds.count("done") == 1
