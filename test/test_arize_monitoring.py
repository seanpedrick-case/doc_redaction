"""Tests for Arize AX / Phoenix tracing setup branching."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_AGENT_REDACT = Path(__file__).resolve().parents[1] / "agent-redact"
if str(_AGENT_REDACT) not in sys.path:
    sys.path.insert(0, str(_AGENT_REDACT))


def _reload_arize_monitoring():
    """Reload module so ``_INITIALIZED`` resets between tests."""
    import eval.arize_monitoring as mod

    importlib.reload(mod)
    return mod


@pytest.fixture(autouse=True)
def _clear_tracing_env(monkeypatch):
    for key in (
        "ARIZE_TRACING_ENABLED",
        "ARIZE_BACKEND",
        "ARIZE_SPACE_ID",
        "ARIZE_API_KEY",
        "ARIZE_PROJECT_NAME",
        "ARIZE_ENDPOINT",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_PROJECT_NAME",
        "PHOENIX_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_phoenix_otlp_endpoint_normalization():
    mod = _reload_arize_monitoring()
    assert (
        mod._phoenix_otlp_endpoint("http://localhost:6006")
        == "http://localhost:6006/v1/traces"
    )
    assert (
        mod._phoenix_otlp_endpoint("http://localhost:6006/v1/traces")
        == "http://localhost:6006/v1/traces"
    )
    assert mod._phoenix_otlp_endpoint("http://host:4317") == "http://host:4317"


def test_tracing_backend_defaults_and_aliases(monkeypatch):
    mod = _reload_arize_monitoring()
    assert mod._tracing_backend() == "ax"
    monkeypatch.setenv("ARIZE_BACKEND", "phoenix")
    assert mod._tracing_backend() == "phoenix"
    monkeypatch.setenv("ARIZE_BACKEND", "LOCAL")
    assert mod._tracing_backend() == "phoenix"
    monkeypatch.setenv("ARIZE_BACKEND", "ax")
    assert mod._tracing_backend() == "ax"


def test_setup_disabled_when_env_unset():
    mod = _reload_arize_monitoring()
    assert mod.setup_arize_ax_tracing() is False


def test_setup_ax_path_registers_with_space_and_key(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "ax")
    monkeypatch.setenv("ARIZE_SPACE_ID", "space-1")
    monkeypatch.setenv("ARIZE_API_KEY", "key-1")
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "proj-ax")
    monkeypatch.setenv("ARIZE_ENDPOINT", "europe")

    mod = _reload_arize_monitoring()
    mock_register = MagicMock(return_value=SimpleNamespace(name="provider"))
    mock_endpoint = SimpleNamespace(ARIZE="us", ARIZE_EUROPE="eu")
    mock_instrumentor_cls = MagicMock()
    mock_instrumentor = MagicMock()
    mock_instrumentor_cls.return_value = mock_instrumentor

    arize_otel = ModuleType("arize.otel")
    arize_otel.Endpoint = mock_endpoint
    arize_otel.register = mock_register
    arize_pkg = ModuleType("arize")
    arize_pkg.otel = arize_otel

    oi_lc = ModuleType("openinference.instrumentation.langchain")
    oi_lc.LangChainInstrumentor = mock_instrumentor_cls

    with patch.dict(
        sys.modules,
        {
            "arize": arize_pkg,
            "arize.otel": arize_otel,
            "openinference.instrumentation.langchain": oi_lc,
        },
    ):
        assert mod.setup_arize_ax_tracing() is True

    mock_register.assert_called_once_with(
        space_id="space-1",
        api_key="key-1",
        project_name="proj-ax",
        endpoint="eu",
    )
    mock_instrumentor.instrument.assert_called_once()
    assert mod.setup_arize_ax_tracing() is True  # already initialized
    assert mock_register.call_count == 1


def test_setup_ax_skips_without_credentials(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "ax")
    mod = _reload_arize_monitoring()
    with pytest.warns(UserWarning, match="ARIZE_SPACE_ID"):
        assert mod.setup_arize_ax_tracing() is False


def test_setup_phoenix_path_registers_local_collector(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "phoenix")
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "proj-px")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006")
    monkeypatch.setenv("PHOENIX_API_KEY", "px-key")

    mod = _reload_arize_monitoring()
    mock_register = MagicMock(return_value=SimpleNamespace(name="provider"))
    mock_instrumentor_cls = MagicMock()
    mock_instrumentor = MagicMock()
    mock_instrumentor_cls.return_value = mock_instrumentor

    phoenix_otel = ModuleType("phoenix.otel")
    phoenix_otel.register = mock_register
    phoenix_pkg = ModuleType("phoenix")
    phoenix_pkg.otel = phoenix_otel

    oi_lc = ModuleType("openinference.instrumentation.langchain")
    oi_lc.LangChainInstrumentor = mock_instrumentor_cls

    with patch.dict(
        sys.modules,
        {
            "phoenix": phoenix_pkg,
            "phoenix.otel": phoenix_otel,
            "openinference.instrumentation.langchain": oi_lc,
        },
    ):
        assert mod.setup_arize_ax_tracing() is True

    mock_register.assert_called_once_with(
        project_name="proj-px",
        endpoint="http://127.0.0.1:6006/v1/traces",
        api_key="px-key",
    )
    mock_instrumentor.instrument.assert_called_once()


def test_setup_phoenix_default_endpoint_no_api_key(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "phoenix")

    mod = _reload_arize_monitoring()
    mock_register = MagicMock(return_value=SimpleNamespace(name="provider"))
    mock_instrumentor_cls = MagicMock()
    mock_instrumentor_cls.return_value = MagicMock()

    phoenix_otel = ModuleType("phoenix.otel")
    phoenix_otel.register = mock_register
    phoenix_pkg = ModuleType("phoenix")
    phoenix_pkg.otel = phoenix_otel
    oi_lc = ModuleType("openinference.instrumentation.langchain")
    oi_lc.LangChainInstrumentor = mock_instrumentor_cls

    with patch.dict(
        sys.modules,
        {
            "phoenix": phoenix_pkg,
            "phoenix.otel": phoenix_otel,
            "openinference.instrumentation.langchain": oi_lc,
        },
    ):
        assert mod.setup_arize_ax_tracing() is True

    kwargs = mock_register.call_args.kwargs
    assert kwargs["endpoint"] == "http://localhost:6006/v1/traces"
    assert kwargs["project_name"] == "doc-redaction-langgraph"
    assert "api_key" not in kwargs


def test_setup_phoenix_without_langchain_instrumentor(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "phoenix")

    mod = _reload_arize_monitoring()
    mock_register = MagicMock(return_value=SimpleNamespace(name="provider"))
    mock_instrumentor_cls = MagicMock()
    mock_instrumentor_cls.return_value = MagicMock()

    phoenix_otel = ModuleType("phoenix.otel")
    phoenix_otel.register = mock_register
    phoenix_pkg = ModuleType("phoenix")
    phoenix_pkg.otel = phoenix_otel
    oi_lc = ModuleType("openinference.instrumentation.langchain")
    oi_lc.LangChainInstrumentor = mock_instrumentor_cls

    with patch.dict(
        sys.modules,
        {
            "phoenix": phoenix_pkg,
            "phoenix.otel": phoenix_otel,
            "openinference.instrumentation.langchain": oi_lc,
        },
    ):
        assert mod.setup_arize_ax_tracing(instrument_langchain=False) is True

    mock_register.assert_called_once()
    mock_instrumentor_cls.assert_not_called()
    assert mod.tracing_initialized() is True


def test_iter_pi_events_passthrough_when_not_initialized():
    mod = _reload_arize_monitoring()
    events = [
        SimpleNamespace(kind="text_delta", text="hi", is_error=False),
        SimpleNamespace(kind="done", text="done", is_error=False),
    ]
    out = list(
        mod.iter_pi_events_with_tracing(
            iter(events), session_hash="sess-1", message="hello"
        )
    )
    assert [e.kind for e in out] == ["text_delta", "done"]


def test_iter_pi_events_emits_agent_and_tool_spans(monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING_ENABLED", "true")
    monkeypatch.setenv("ARIZE_BACKEND", "phoenix")
    mod = _reload_arize_monitoring()

    mock_register = MagicMock(return_value=SimpleNamespace(name="provider"))
    phoenix_otel = ModuleType("phoenix.otel")
    phoenix_otel.register = mock_register
    phoenix_pkg = ModuleType("phoenix")
    phoenix_pkg.otel = phoenix_otel

    with patch.dict(
        sys.modules,
        {"phoenix": phoenix_pkg, "phoenix.otel": phoenix_otel},
    ):
        assert mod.setup_arize_ax_tracing(instrument_langchain=False) is True

    root_span = MagicMock()
    tool_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = root_span
    mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
    mock_tracer.start_span.return_value = tool_span

    span_attrs = SimpleNamespace(
        OUTPUT_VALUE="output.value",
        LLM_TOKEN_COUNT_PROMPT="llm.token_count.prompt",
        LLM_TOKEN_COUNT_COMPLETION="llm.token_count.completion",
        OPENINFERENCE_SPAN_KIND="openinference.span.kind",
        TOOL_NAME="tool.name",
        TOOL_ID="tool.id",
        TOOL_PARAMETERS="tool.parameters",
        INPUT_VALUE="input.value",
        SESSION_ID="session.id",
        AGENT_NAME="agent.name",
    )
    oi_semconv_trace = ModuleType("openinference.semconv.trace")
    oi_semconv_trace.SpanAttributes = span_attrs
    oi_semconv_trace.OpenInferenceSpanKindValues = SimpleNamespace(
        TOOL=SimpleNamespace(value="TOOL"),
        AGENT=SimpleNamespace(value="AGENT"),
    )
    oi_semconv = ModuleType("openinference.semconv")
    oi_semconv.trace = oi_semconv_trace
    oi_instrumentation = ModuleType("openinference.instrumentation")

    @contextmanager
    def _using_session(*, session_id: str):
        yield

    oi_instrumentation.using_session = _using_session
    oi_pkg = ModuleType("openinference")
    oi_pkg.semconv = oi_semconv
    oi_pkg.instrumentation = oi_instrumentation

    otel_trace = ModuleType("opentelemetry.trace")
    otel_trace.get_tracer = MagicMock(return_value=mock_tracer)
    otel_trace.Status = MagicMock(side_effect=lambda status: status)
    otel_trace.StatusCode = SimpleNamespace(ERROR="ERROR")
    otel_trace.get_current_span = MagicMock(return_value=MagicMock())
    otel_trace.set_span_in_context = MagicMock(side_effect=lambda span: span)
    otel_pkg = ModuleType("opentelemetry")
    otel_pkg.trace = otel_trace

    events = [
        SimpleNamespace(
            kind="tool_start",
            tool_name="bash",
            tool_call_id="c1",
            tool_args={"command": "ls"},
            text="",
            is_error=False,
            tool_output=None,
        ),
        SimpleNamespace(
            kind="tool_end",
            tool_name="bash",
            tool_call_id="c1",
            tool_output="ok",
            text="",
            is_error=False,
            tool_args=None,
        ),
        SimpleNamespace(
            kind="text_delta",
            text="hello",
            is_error=False,
            tool_name=None,
            tool_call_id=None,
            tool_args=None,
            tool_output=None,
        ),
        SimpleNamespace(
            kind="done",
            text="done",
            is_error=False,
            tool_name=None,
            tool_call_id=None,
            tool_args=None,
            tool_output=None,
        ),
    ]

    with patch.dict(
        sys.modules,
        {
            "opentelemetry": otel_pkg,
            "opentelemetry.trace": otel_trace,
            "openinference": oi_pkg,
            "openinference.semconv": oi_semconv,
            "openinference.semconv.trace": oi_semconv_trace,
            "openinference.instrumentation": oi_instrumentation,
        },
    ):
        out = list(
            mod.iter_pi_events_with_tracing(
                iter(events),
                session_hash="sess-abc",
                message="run ls",
                get_session_stats=lambda: {"input_tokens": 3, "output_tokens": 2},
            )
        )

    assert [e.kind for e in out] == ["tool_start", "tool_end", "text_delta", "done"]
    mock_tracer.start_as_current_span.assert_called_once_with("pi.agent")
    mock_tracer.start_span.assert_called_once()
    tool_span.end.assert_called_once()
    root_span.set_attribute.assert_any_call(span_attrs.OUTPUT_VALUE, "hello")
    root_span.set_attribute.assert_any_call(span_attrs.LLM_TOKEN_COUNT_PROMPT, 3)
    root_span.set_attribute.assert_any_call(span_attrs.LLM_TOKEN_COUNT_COMPLETION, 2)
