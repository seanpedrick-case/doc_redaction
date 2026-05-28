"""Tests for Pi RPC assistant display extraction (Gemini reasoning blocks)."""

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_rpc_client import (
    assistant_chat_text,
    assistant_text_since_last_user,
    extract_assistant_display,
    format_assistant_message_for_chat,
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
            }
        ],
    }
    rendered = format_assistant_message_for_chat(message)
    assert "**bash:**" in rendered
    assert "skills/doc-redaction-app" in rendered


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
