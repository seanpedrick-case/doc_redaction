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


def test_bedrock_judge_explicit_flag(monkeypatch):
    monkeypatch.setenv("USE_BEDROCK_JUDGE", "1")
    monkeypatch.setenv(
        "DEEPEVAL_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.delenv("USE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("USE_LOCAL_MODEL", raising=False)

    created: list[dict] = []

    class FakeBedrock:
        def __init__(self, model: str, region: str, **kwargs):
            created.append({"model": model, "region": region, **kwargs})

    import deepeval.models as deepeval_models

    monkeypatch.setattr(deepeval_models, "AmazonBedrockModel", FakeBedrock)

    from deepeval_metrics import _deepeval_judge_model

    model = _deepeval_judge_model()
    assert model is not None
    assert created[0]["model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert created[0]["region"] == "eu-west-2"


def test_bedrock_judge_auto_when_provider_amazon_bedrock(monkeypatch):
    monkeypatch.delenv("USE_BEDROCK_JUDGE", raising=False)
    monkeypatch.delenv("USE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("USE_LOCAL_MODEL", raising=False)
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "amazon-bedrock")
    monkeypatch.setenv(
        "AGENT_DEFAULT_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    created: list[dict] = []

    class FakeBedrock:
        def __init__(self, model: str, region: str, **kwargs):
            created.append({"model": model, "region": region})

    import deepeval.models as deepeval_models

    monkeypatch.setattr(deepeval_models, "AmazonBedrockModel", FakeBedrock)

    from deepeval_metrics import _deepeval_judge_model

    assert _deepeval_judge_model() is not None
    assert created[0]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_judge_model_falls_back_to_agent_llama_url(monkeypatch):
    monkeypatch.delenv("USE_BEDROCK_JUDGE", raising=False)
    monkeypatch.delenv("AGENT_DEFAULT_PROVIDER", raising=False)
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
