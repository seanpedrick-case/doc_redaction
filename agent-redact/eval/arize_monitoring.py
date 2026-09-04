"""Arize / Phoenix OpenTelemetry tracing for agent runs.

Call :func:`setup_arize_ax_tracing` once at runtime startup (before LangChain
imports when using LangGraph) so exporters and optional
``LangChainInstrumentor`` are registered.

- **LangGraph**: ``instrument_langchain=True`` (default) patches LangChain;
  wrap ``graph.stream`` with :func:`arize_session_context` and
  :func:`langgraph_trace_config`.
- **Pi**: ``instrument_langchain=False`` registers the collector only; wrap
  RPC event streams with :func:`iter_pi_events_with_tracing` (custom AGENT /
  TOOL spans from Pi events — LLM HTTP stays inside the Node ``pi`` process).

Multi-turn chats: ``session.id`` = Gradio ``session_hash`` so turns share one
Sessions row in Arize AX or Phoenix.

Environment (see ``config/agent.env.example``):

- ``ARIZE_TRACING_ENABLED`` — set ``true`` / ``1`` / ``yes`` / ``on`` to enable
- ``ARIZE_BACKEND`` — ``ax`` (default, hosted Arize AX) or ``phoenix`` (local /
  self-hosted Phoenix)
- ``ARIZE_PROJECT_NAME`` — default ``doc-redaction-langgraph`` (Phoenix also
  accepts ``PHOENIX_PROJECT_NAME``)
- AX (``ARIZE_BACKEND=ax``): ``ARIZE_SPACE_ID`` / ``ARIZE_API_KEY`` required;
  ``ARIZE_ENDPOINT`` — ``europe`` (default) or ``us``
- Phoenix (``ARIZE_BACKEND=phoenix``): ``PHOENIX_COLLECTOR_ENDPOINT`` (default
  ``http://localhost:6006``); optional ``PHOENIX_API_KEY``

Smoke check
-----------
1. Set tracing env vars (``ARIZE_TRACING_ENABLED``, backend, endpoint/credentials).
2. LangGraph: ``AGENT_ORCHESTRATOR=langgraph``, send one Gradio chat turn.
3. Pi: ``AGENT_ORCHESTRATOR=pi``, send one Gradio chat turn; expect AGENT/TOOL
   spans (not full LLM message spans).
4. Confirm spans in Phoenix (``http://localhost:6006``) or the Arize AX project.
5. Send a follow-up in the same Gradio session; both turns should share one
   Sessions row (``session.id`` = Gradio ``session_hash``).
6. With tracing disabled: confirm no errors and no collector traffic.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_INITIALIZED = False
_TRACER_NAME = "doc_redaction.agent"
_MAX_ATTR_CHARS = 4000


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tracing_backend() -> str:
    """Return ``ax`` or ``phoenix`` from ``ARIZE_BACKEND`` (default ``ax``)."""
    raw = (os.environ.get("ARIZE_BACKEND") or "ax").strip().lower()
    if raw in {"phoenix", "local", "oss"}:
        return "phoenix"
    return "ax"


def _project_name() -> str:
    return (
        os.environ.get("PHOENIX_PROJECT_NAME")
        or os.environ.get("ARIZE_PROJECT_NAME")
        or "doc-redaction-langgraph"
    ).strip()


def _phoenix_otlp_endpoint(collector: str) -> str:
    """Normalize collector base URL to an HTTP OTLP traces endpoint when needed."""
    ep = collector.strip().rstrip("/")
    if not ep:
        ep = "http://localhost:6006"
    if ep.endswith("/v1/traces") or ep.endswith(":4317"):
        return ep
    return f"{ep}/v1/traces"


def _truncate(value: str, limit: int = _MAX_ATTR_CHARS) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_attr(value: Any) -> str:
    try:
        return _truncate(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return _truncate(str(value))


def tracing_initialized() -> bool:
    """True after a successful :func:`setup_arize_ax_tracing` call."""
    return _INITIALIZED


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


def _instrument_langchain(tracer_provider: Any) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


def _setup_phoenix_tracing(
    project_name: str, *, instrument_langchain: bool = True
) -> bool:
    """Register Phoenix OTEL exporter; optionally LangChainInstrumentor."""
    try:
        from phoenix.otel import register
    except ImportError as exc:
        warnings.warn(
            f"Phoenix tracing dependencies unavailable ({exc}); tracing skipped. "
            "Install arize-phoenix-otel.",
            UserWarning,
            stacklevel=3,
        )
        return False

    if instrument_langchain:
        try:
            from openinference.instrumentation.langchain import (  # noqa: F401
                LangChainInstrumentor,
            )
        except ImportError as exc:
            warnings.warn(
                f"LangChain OpenInference unavailable ({exc}); tracing skipped. "
                "Install openinference-instrumentation-langchain.",
                UserWarning,
                stacklevel=3,
            )
            return False

    collector = (
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006"
    ).strip()
    endpoint = _phoenix_otlp_endpoint(collector)
    register_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "endpoint": endpoint,
    }
    api_key = (os.environ.get("PHOENIX_API_KEY") or "").strip()
    if api_key:
        register_kwargs["api_key"] = api_key

    tracer_provider = register(**register_kwargs)
    if instrument_langchain:
        _instrument_langchain(tracer_provider)
    return True


def _setup_ax_tracing(project_name: str, *, instrument_langchain: bool = True) -> bool:
    """Register Arize AX tracer; optionally LangChainInstrumentor."""
    space_id = (os.environ.get("ARIZE_SPACE_ID") or "").strip()
    api_key = (os.environ.get("ARIZE_API_KEY") or "").strip()
    if not space_id or not api_key:
        warnings.warn(
            "ARIZE_TRACING_ENABLED is set but ARIZE_SPACE_ID / ARIZE_API_KEY "
            "are missing; Arize AX tracing is skipped.",
            UserWarning,
            stacklevel=3,
        )
        return False

    try:
        from arize.otel import Endpoint, register
    except ImportError as exc:
        warnings.warn(
            f"Arize AX tracing dependencies unavailable ({exc}); tracing skipped. "
            "Install arize-otel.",
            UserWarning,
            stacklevel=3,
        )
        return False

    if instrument_langchain:
        try:
            from openinference.instrumentation.langchain import (  # noqa: F401
                LangChainInstrumentor,
            )
        except ImportError as exc:
            warnings.warn(
                f"LangChain OpenInference unavailable ({exc}); tracing skipped. "
                "Install openinference-instrumentation-langchain.",
                UserWarning,
                stacklevel=3,
            )
            return False

    endpoint_raw = (os.environ.get("ARIZE_ENDPOINT") or "europe").strip().lower()
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
    if instrument_langchain:
        _instrument_langchain(tracer_provider)
    return True


def setup_arize_ax_tracing(*, instrument_langchain: bool = True) -> bool:
    """Register Arize AX or Phoenix tracer (+ optional LangChainInstrumentor).

    Backend is selected via ``ARIZE_BACKEND`` (``ax`` default, or ``phoenix``).

    Pass ``instrument_langchain=False`` for Pi (custom RPC spans only).

    Returns True if tracing was enabled (or already initialized). Soft-fails
    with a warning when partially configured or packages are missing.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    if not _env_truthy("ARIZE_TRACING_ENABLED"):
        return False

    project_name = _project_name()
    backend = _tracing_backend()
    if backend == "phoenix":
        ok = _setup_phoenix_tracing(
            project_name, instrument_langchain=instrument_langchain
        )
    else:
        ok = _setup_ax_tracing(project_name, instrument_langchain=instrument_langchain)

    if ok:
        _INITIALIZED = True
    return ok


def _apply_session_stats_to_span(span: Any, stats: dict[str, Any]) -> None:
    """Best-effort token / cost attributes from Pi ``get_session_stats``."""
    try:
        from openinference.semconv.trace import SpanAttributes
    except ImportError:
        return

    def _set_int(attr: str, *keys: str) -> None:
        for key in keys:
            val = stats.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and val >= 0:
                span.set_attribute(attr, int(val))
                return

    _set_int(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
        "input_tokens",
        "prompt_tokens",
        "tokens_prompt",
    )
    _set_int(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
        "output_tokens",
        "completion_tokens",
        "tokens_completion",
    )
    total = stats.get("total_tokens") or stats.get("tokens")
    if isinstance(total, (int, float)) and total >= 0:
        span.set_attribute("llm.token_count.total", int(total))


def iter_pi_events_with_tracing(
    events: Iterator[Any],
    *,
    session_hash: str | None,
    message: str,
    get_session_stats: Callable[[], dict[str, Any]] | None = None,
) -> Iterator[Any]:
    """Yield Pi stream events while emitting OpenInference AGENT/TOOL spans.

    No-op (passthrough) when tracing was not initialized. LLM HTTP calls remain
    inside the Node ``pi`` process and are not auto-instrumented.
    """
    if not _INITIALIZED:
        yield from events
        return

    try:
        from openinference.semconv.trace import (
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        yield from events
        return

    tracer = trace.get_tracer(_TRACER_NAME)
    tool_spans: dict[str, Any] = {}
    text_parts: list[str] = []
    had_error = False

    def _tool_key(event: Any) -> str:
        call_id = getattr(event, "tool_call_id", None)
        if call_id:
            return str(call_id)
        name = getattr(event, "tool_name", None) or "tool"
        return f"anon:{name}:{len(tool_spans)}"

    def _end_tool_span(key: str, event: Any) -> None:
        nonlocal had_error
        span = tool_spans.pop(key, None)
        if span is None:
            return
        output = getattr(event, "tool_output", None) or getattr(event, "text", "") or ""
        if output:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _truncate(str(output)))
        if getattr(event, "is_error", False):
            span.set_status(Status(StatusCode.ERROR))
            had_error = True
        span.end()

    def _observe(event: Any) -> None:
        nonlocal had_error
        kind = str(getattr(event, "kind", "") or "")
        if kind in {"text_delta", "text_replace", "thinking_delta"}:
            chunk = getattr(event, "text", None)
            if chunk:
                text_parts.append(str(chunk))
            return
        if kind == "tool_start":
            key = _tool_key(event)
            name = str(getattr(event, "tool_name", None) or "tool")
            span = tracer.start_span(
                f"pi.tool.{name}",
                context=trace.set_span_in_context(trace.get_current_span()),
            )
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.TOOL.value,
            )
            span.set_attribute(SpanAttributes.TOOL_NAME, name)
            call_id = getattr(event, "tool_call_id", None)
            if call_id:
                span.set_attribute(SpanAttributes.TOOL_ID, str(call_id))
            args = getattr(event, "tool_args", None)
            if args:
                span.set_attribute(SpanAttributes.TOOL_PARAMETERS, _json_attr(args))
                span.set_attribute(SpanAttributes.INPUT_VALUE, _json_attr(args))
            tool_spans[key] = span
            return
        if kind == "tool_end":
            # Prefer matching call id; fall back to most recent anon key for name.
            call_id = getattr(event, "tool_call_id", None)
            if call_id and str(call_id) in tool_spans:
                _end_tool_span(str(call_id), event)
                return
            name = str(getattr(event, "tool_name", None) or "")
            for key in list(tool_spans):
                if key.startswith(f"anon:{name}:") or (
                    not call_id and key.startswith("anon:")
                ):
                    _end_tool_span(key, event)
                    return
            return
        if getattr(event, "is_error", False) or kind == "error":
            had_error = True

    with arize_session_context(session_hash):
        with tracer.start_as_current_span("pi.agent") as root:
            root.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.AGENT.value,
            )
            root.set_attribute(SpanAttributes.AGENT_NAME, "pi")
            root.set_attribute(SpanAttributes.INPUT_VALUE, _truncate(message))
            sid = arize_session_id(session_hash)
            if sid:
                root.set_attribute(SpanAttributes.SESSION_ID, sid)

            try:
                for event in events:
                    try:
                        _observe(event)
                    except Exception:  # pragma: no cover - never break the UI stream
                        pass
                    yield event
            finally:
                for span in list(tool_spans.values()):
                    try:
                        span.end()
                    except Exception:  # pragma: no cover
                        pass
                tool_spans.clear()
                if text_parts:
                    root.set_attribute(
                        SpanAttributes.OUTPUT_VALUE, _truncate("".join(text_parts))
                    )
                if get_session_stats is not None:
                    try:
                        stats = get_session_stats() or {}
                        if isinstance(stats, dict):
                            _apply_session_stats_to_span(root, stats)
                    except Exception:  # pragma: no cover
                        pass
                if had_error:
                    root.set_status(Status(StatusCode.ERROR))
