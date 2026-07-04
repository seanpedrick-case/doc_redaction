"""Summarize Pi agent LLM token usage for usage-log CSV rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_rpc_client import PiRpcClient, PiRpcError  # noqa: E402

try:
    from agent_runtime import AgentRuntime
except ImportError:
    AgentRuntime = PiRpcClient  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class TokenUsageTotals:
    """Pi session usage (see Pi session-format ``Usage``)."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def llm_input_tokens(self) -> int:
        """Input-side tokens for the main-app usage log (input + cache)."""
        return self.input + self.cache_read + self.cache_write

    @property
    def llm_output_tokens(self) -> int:
        return self.output


def _int_field(raw: Any) -> int:
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def totals_from_usage_dict(usage: dict[str, Any] | None) -> TokenUsageTotals:
    if not usage:
        return TokenUsageTotals()
    return TokenUsageTotals(
        input=_int_field(usage.get("input")),
        output=_int_field(usage.get("output")),
        cache_read=_int_field(usage.get("cacheRead")),
        cache_write=_int_field(usage.get("cacheWrite")),
    )


def totals_from_stats_payload(data: dict[str, Any] | None) -> TokenUsageTotals:
    if not data:
        return TokenUsageTotals()
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        return totals_from_usage_dict(tokens)
    return TokenUsageTotals()


def subtract_usage(
    after: TokenUsageTotals, before: TokenUsageTotals
) -> TokenUsageTotals:
    return TokenUsageTotals(
        input=max(0, after.input - before.input),
        output=max(0, after.output - before.output),
        cache_read=max(0, after.cache_read - before.cache_read),
        cache_write=max(0, after.cache_write - before.cache_write),
    )


def add_usage(left: TokenUsageTotals, right: TokenUsageTotals) -> TokenUsageTotals:
    return TokenUsageTotals(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
    )


def sum_usage_from_messages(
    messages: list[dict[str, Any]],
    *,
    since_last_user: bool = False,
) -> TokenUsageTotals:
    """Sum ``usage`` on assistant messages (optional: only after the last user turn)."""
    last_user = -1
    if since_last_user:
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                last_user = index
        messages = messages[last_user + 1 :] if last_user >= 0 else messages

    total = TokenUsageTotals()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if isinstance(usage, dict):
            total = add_usage(total, totals_from_usage_dict(usage))
    return total


def sum_usage_from_jsonl(path: Path) -> TokenUsageTotals:
    """Parse a Pi session JSONL file and sum assistant ``usage`` blocks."""
    total = TokenUsageTotals()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return total
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if isinstance(usage, dict):
            total = add_usage(total, totals_from_usage_dict(usage))
    return total


def resolve_session_token_usage(client: AgentRuntime | None) -> TokenUsageTotals:
    """
    Best-effort session usage from Pi RPC ``get_session_stats``, live messages, or JSONL.
    """
    if client is None or not client.running:
        return TokenUsageTotals()

    try:
        stats = client.get_session_stats()
        totals = totals_from_stats_payload(stats)
        if totals.input or totals.output or totals.cache_read or totals.cache_write:
            return totals
    except PiRpcError:
        pass

    try:
        messages = client.get_messages()
        totals = sum_usage_from_messages(messages)
        if totals.input or totals.output or totals.cache_read or totals.cache_write:
            return totals
    except PiRpcError:
        pass

    from session_logs import pi_session_file_from_client

    session_file = pi_session_file_from_client(client)
    if session_file is not None:
        return sum_usage_from_jsonl(session_file)
    return TokenUsageTotals()


def usage_for_completed_turn(
    client: AgentRuntime | None,
    baseline: TokenUsageTotals | None,
) -> TokenUsageTotals:
    """
    Tokens consumed by the prompt that just finished.

    Prefers delta from *baseline* (captured before ``prompt_events``). Falls back to
    summing assistant ``usage`` since the last user message, then whole-session totals.
    """
    if client is None or not client.running:
        return TokenUsageTotals()

    current = resolve_session_token_usage(client)
    if baseline is not None:
        delta = subtract_usage(current, baseline)
        if delta.input or delta.output or delta.cache_read or delta.cache_write:
            return delta

    try:
        turn = sum_usage_from_messages(client.get_messages(), since_last_user=True)
        if turn.input or turn.output or turn.cache_read or turn.cache_write:
            return turn
    except PiRpcError:
        pass

    return current
