"""DeepEval metric builder compatibility (optional deepeval)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_AGENT_REDACT = Path(__file__).resolve().parents[1] / "agent-redact"
_EVAL = _AGENT_REDACT / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

deepeval = pytest.importorskip("deepeval")

from deepeval_metrics import build_regression_deepeval_metrics  # noqa: E402


def test_judge_model_falls_back_to_agent_llama_url(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_MODEL", "1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_LLAMA_BASE_URL", "http://llama-inference:8080/v1")
    monkeypatch.setenv("AGENT_LLAMA_MODEL_ID", "test-gguf")
    monkeypatch.setenv("OPENAI_API_KEY", "local")

    from deepeval_metrics import _deepeval_judge_model

    model = _deepeval_judge_model()
    assert model is not None
    assert getattr(model, "base_url", None) == "http://llama-inference:8080/v1"


def test_build_tool_correctness_metric_deepeval_4(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_MODEL", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    gold_case = _EVAL / "gold" / "example_live"
    metrics = build_regression_deepeval_metrics(
        gold_case,
        include_policy_strict=False,
        include_tool_correctness=True,
        include_task_completion=False,
    )
    assert len(metrics) == 1
    assert metrics[0].__class__.__name__ == "ToolCorrectnessMetric"
