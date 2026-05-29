"""Tests for Pi session token usage aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_PI = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI) not in sys.path:
    sys.path.insert(0, str(_PI))

from pi_session_usage import (  # noqa: E402
    TokenUsageTotals,
    sum_usage_from_jsonl,
    sum_usage_from_messages,
    totals_from_stats_payload,
    totals_from_usage_dict,
    usage_for_completed_turn,
)


def test_totals_from_usage_dict_sums_cache_into_input_column():
    usage = totals_from_usage_dict(
        {
            "input": 100,
            "output": 40,
            "cacheRead": 500,
            "cacheWrite": 10,
        }
    )
    assert usage.llm_input_tokens == 610
    assert usage.llm_output_tokens == 40


def test_sum_usage_from_messages_since_last_user():
    messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0},
        },
        {"role": "user", "content": "second"},
        {
            "role": "assistant",
            "usage": {"input": 20, "output": 8, "cacheRead": 1, "cacheWrite": 0},
        },
    ]
    turn = sum_usage_from_messages(messages, since_last_user=True)
    assert turn.llm_input_tokens == 21
    assert turn.llm_output_tokens == 8


def test_sum_usage_from_jsonl(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "usage": {"input": 3, "output": 2},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    totals = sum_usage_from_jsonl(log)
    assert totals.llm_input_tokens == 3
    assert totals.llm_output_tokens == 2


def test_usage_for_completed_turn_prefers_stats_delta():
    client = MagicMock()
    client.running = True
    client.get_session_stats.side_effect = [
        {"tokens": {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0}},
        {"tokens": {"input": 250, "output": 55, "cacheRead": 0, "cacheWrite": 0}},
    ]
    client.get_messages.return_value = []

    baseline = totals_from_stats_payload(client.get_session_stats())
    usage = usage_for_completed_turn(client, baseline)
    assert usage.llm_input_tokens == 150
    assert usage.llm_output_tokens == 45


def test_totals_from_stats_payload_empty():
    assert totals_from_stats_payload(None) == TokenUsageTotals()
