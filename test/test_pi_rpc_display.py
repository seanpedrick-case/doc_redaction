"""Tests for Pi RPC assistant display extraction (Gemini reasoning blocks)."""

from docker.pi.pi_rpc_client import assistant_chat_text, extract_assistant_display


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
