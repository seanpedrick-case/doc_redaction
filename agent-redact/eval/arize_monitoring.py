"""Arize AX OpenTelemetry tracing for LangGraph agent runs.

Call :func:`setup_arize_ax_tracing` once **before** any ``langchain`` /
``langchain_core`` import so ``LangChainInstrumentor`` can patch the stack.

Multi-turn chats: wrap each ``graph.stream`` with :func:`arize_session_context`
and pass :func:`langgraph_trace_config` so turns share ``session.id`` (Gradio
``session_hash``) and appear under one Sessions row in Arize AX.

Environment (see ``config/agent.env.example``):

- ``ARIZE_TRACING_ENABLED`` — set ``true`` / ``1`` / ``yes`` / ``on`` to enable
- ``ARIZE_SPACE_ID`` / ``ARIZE_API_KEY`` — required when enabled
- ``ARIZE_PROJECT_NAME`` — default ``doc-redaction-langgraph``
- ``ARIZE_ENDPOINT`` — ``europe`` (default) or ``us``

Smoke check
-----------
1. Set ``AGENT_ORCHESTRATOR=langgraph`` and the ``ARIZE_*`` variables above.
2. Start the agent Gradio app; send one short chat that triggers an LLM call.
3. Confirm spans appear under the Arize AX EU (or US) project — LLM plus
   tool/agent hierarchy if tools run.
4. Send a follow-up in the same Gradio session; both turns should share one
   Sessions row (``session.id`` = Gradio ``session_hash``).
5. With tracing disabled or ``AGENT_ORCHESTRATOR=pi``: confirm no errors and
   no Arize traffic.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_INITIALIZED = False


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def arize_session_id(session_hash: str | None) -> str | None:
    """Return a non-empty session id for OpenInference, or None."""
    sid = (session_hash or "").strip()
    return sid or None


def langgraph_trace_config(
    session_hash: str | None,
    *,
    recursion_limit: int,
) -> dict[str, Any]:
    """LangGraph ``config`` with recursion limit and Arize/LangChain session keys."""
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    sid = arize_session_id(session_hash)
    if sid:
        # LangChainInstrumentor groups turns via any of these metadata keys.
        config["metadata"] = {
            "session_id": sid,
            "thread_id": sid,
            "conversation_id": sid,
        }
        config["configurable"] = {"thread_id": sid}
    return config


@contextmanager
def arize_session_context(session_hash: str | None) -> Iterator[None]:
    """Attach OpenInference ``session.id`` for the duration of a LangGraph turn.

    No-op when *session_hash* is empty or ``openinference`` is not installed.
    Safe to use inside generators (context stays active across ``yield``).
    """
    sid = arize_session_id(session_hash)
    if not sid:
        yield
        return
    try:
        from openinference.instrumentation import using_session
    except ImportError:
        yield
        return
    with using_session(session_id=sid):
        yield


def setup_arize_ax_tracing() -> bool:
    """Register Arize AX tracer + LangChainInstrumentor.

    Returns True if tracing was enabled (or already initialized). Soft-fails
    with a warning when partially configured or packages are missing.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    if not _env_truthy("ARIZE_TRACING_ENABLED"):
        return False

    space_id = (os.environ.get("ARIZE_SPACE_ID") or "").strip()
    api_key = (os.environ.get("ARIZE_API_KEY") or "").strip()
    if not space_id or not api_key:
        warnings.warn(
            "ARIZE_TRACING_ENABLED is set but ARIZE_SPACE_ID / ARIZE_API_KEY "
            "are missing; Arize AX tracing is skipped.",
            UserWarning,
            stacklevel=2,
        )
        return False

    project_name = (
        os.environ.get("ARIZE_PROJECT_NAME") or "doc-redaction-langgraph"
    ).strip()
    endpoint_raw = (os.environ.get("ARIZE_ENDPOINT") or "europe").strip().lower()

    try:
        from arize.otel import Endpoint, register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError as exc:
        warnings.warn(
            f"Arize AX tracing dependencies unavailable ({exc}); tracing skipped. "
            "Install arize-otel and openinference-instrumentation-langchain.",
            UserWarning,
            stacklevel=2,
        )
        return False

    if endpoint_raw in {"us", "arize", "default"}:
        endpoint = Endpoint.ARIZE
    else:
        # Default to Europe (draft / Lambeth deployment).
        endpoint = Endpoint.ARIZE_EUROPE

    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
        endpoint=endpoint,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _INITIALIZED = True
    return True
