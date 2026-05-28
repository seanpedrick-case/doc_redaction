"""Tests for Pi RPC assistant display extraction (Gemini reasoning blocks)."""

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_rpc_client import (
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
