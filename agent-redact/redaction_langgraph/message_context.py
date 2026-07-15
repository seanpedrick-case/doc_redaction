"""Trim LangGraph message history before each LLM call (context compaction)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

_DEFAULT_CONTEXT_WINDOW = 114_688
_DEFAULT_RESERVE_TOKENS = 28_672
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192

_trim_stats = threading.local()
_aggressive_trim = threading.local()


@dataclass(frozen=True)
class TrimStats:
    """Snapshot of the last pre_model_hook trim (per-thread)."""

    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int

    @property
    def trimmed(self) -> bool:
        return self.messages_after < self.messages_before or (
            self.tokens_after < self.tokens_before
        )

    @property
    def messages_dropped(self) -> int:
        return max(0, self.messages_before - self.messages_after)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def langgraph_compaction_enabled() -> bool:
    """Whether to attach a pre_model_hook that trims LLM input (default: true)."""
    return _env_flag("LANGGRAPH_COMPACTION_ENABLED", default=True)


def langgraph_context_window() -> int:
    return max(1_024, _env_int("AGENT_LLAMA_CONTEXT_WINDOW", _DEFAULT_CONTEXT_WINDOW))


def langgraph_compaction_reserve_tokens() -> int:
    """Tokens reserved for tools/schemas/overhead.

    Prefers ``LANGGRAPH_COMPACTION_RESERVE_TOKENS``, then
    ``AGENT_COMPACTION_RESERVE_TOKENS``, else ``28672``.
    """
    if (os.environ.get("LANGGRAPH_COMPACTION_RESERVE_TOKENS") or "").strip():
        return max(0, _env_int("LANGGRAPH_COMPACTION_RESERVE_TOKENS", _DEFAULT_RESERVE_TOKENS))
    if (os.environ.get("AGENT_COMPACTION_RESERVE_TOKENS") or "").strip():
        return max(0, _env_int("AGENT_COMPACTION_RESERVE_TOKENS", _DEFAULT_RESERVE_TOKENS))
    return _DEFAULT_RESERVE_TOKENS


def langgraph_max_output_tokens() -> int:
    return max(256, _env_int("LANGGRAPH_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS))


def langgraph_llm_input_max_tokens(*, aggressive: bool = False) -> int:
    """
    Max tokens for the LLM input after reserve and generation headroom.

    ``aggressive=True`` halves the budget (overflow retry safety net).
    """
    window = langgraph_context_window()
    reserve = langgraph_compaction_reserve_tokens()
    max_out = langgraph_max_output_tokens()
    budget = window - reserve - max_out
    # Leave room for ChatOpenAI / llama.cpp tokenization differences vs approx counts.
    floor = max(2_048, window // 8)
    budget = max(floor, budget)
    if aggressive:
        budget = max(floor, budget // 2)
    return budget


def reset_trim_stats() -> None:
    _trim_stats.last = None  # type: ignore[attr-defined]


def get_trim_stats() -> TrimStats | None:
    return getattr(_trim_stats, "last", None)


def set_aggressive_trim(enabled: bool) -> None:
    """Thread-local override used for one-shot context-overflow retry."""
    _aggressive_trim.enabled = enabled  # type: ignore[attr-defined]


def is_aggressive_trim() -> bool:
    return bool(getattr(_aggressive_trim, "enabled", False))


def is_context_overflow_error(exc: BaseException) -> bool:
    """True when llama.cpp / OpenAI-compatible APIs reject an oversized prompt."""
    text = str(exc).lower()
    markers = (
        "exceed_context_size_error",
        "exceeds the available context size",
        "context length",
        "maximum context length",
        "prompt is too long",
        "n_prompt_tokens",
        "context_length_exceeded",
    )
    return any(marker in text for marker in markers)


def trim_messages_for_llm(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
) -> list[BaseMessage]:
    """Keep system + most recent human/tool tail within *max_tokens*."""
    if not messages:
        return messages
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False,
    )


def build_pre_model_hook(
    *,
    aggressive: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Return a ``create_react_agent`` pre_model_hook that trims via ``llm_input_messages``.

    Full graph ``messages`` state is left intact for tool routing.
    """

    def pre_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        use_aggressive = aggressive or is_aggressive_trim()
        max_tokens = langgraph_llm_input_max_tokens(aggressive=use_aggressive)
        before_tokens = count_tokens_approximately(messages) if messages else 0
        trimmed = trim_messages_for_llm(messages, max_tokens=max_tokens)
        after_tokens = count_tokens_approximately(trimmed) if trimmed else 0
        _trim_stats.last = TrimStats(  # type: ignore[attr-defined]
            tokens_before=before_tokens,
            tokens_after=after_tokens,
            messages_before=len(messages),
            messages_after=len(trimmed),
        )
        # Always provide at least a system message if the trim was empty somehow.
        if not trimmed and messages:
            systems = [m for m in messages if isinstance(m, SystemMessage)]
            trimmed = systems[-1:] if systems else messages[:1]
        return {"llm_input_messages": trimmed}

    return pre_model_hook
