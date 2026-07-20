"""Tests for Arize AX / Phoenix tracing setup branching."""

from __future__ import annotations

import importlib
import sys
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
