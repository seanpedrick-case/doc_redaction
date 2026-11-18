"""DeepEval metrics tied to deterministic agent-redact eval gates."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

try:
    from deepeval.metrics import BaseMetric, TaskCompletionMetric, ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall
except ImportError:  # pragma: no cover - optional dev dependency
    BaseMetric = object  # type: ignore[misc, assignment]
    LLMTestCase = Any  # type: ignore[misc, assignment]
    TaskCompletionMetric = None  # type: ignore[misc, assignment]
    ToolCorrectnessMetric = None  # type: ignore[misc, assignment]
    ToolCall = None  # type: ignore[misc, assignment]


class PolicyStrictMetric(BaseMetric):
    """
    Pass when ``run_policy_gate`` reports ``pass_strict`` for a fixture/run dir.

    ``LLMTestCase.expected_output`` must be JSON with either:
      - ``{"fixture": "<path>"}``  (gates-only on fixture artifacts)
      - ``{"run_dir": "<path>", "fixture": "<path>"}`` after a live run
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.verbose_mode = False

    def measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        payload = _parse_expected_payload(test_case)
        gold_key = payload.get("gold_case") or payload.get("fixture")
        if not gold_key:
            raise ValueError('expected_output JSON must include "gold_case" path')
        gold_case = Path(str(gold_key)).resolve()
        run_dir = Path(payload.get("run_dir") or gold_case).resolve()

        gate_path = run_dir / "policy_gate.json"
        if gate_path.is_file():
            try:
                data = json.loads(gate_path.read_text(encoding="utf-8"))
                passed = bool(data.get("pass_strict"))
            except (OSError, json.JSONDecodeError, TypeError):
                passed = False
        else:
            from run_agent_regression import run_deterministic_gates

            summary = run_deterministic_gates(
                run_dir,
                gold_case,
                require_pass_strict=True,
                compare_extraction=bool(payload.get("compare_extraction", True)),
            )
            passed = bool((summary.policy_gate or {}).get("pass_strict"))
        self.success = passed
        self.score = 1.0 if passed else 0.0
        if passed:
            self.reason = "pass_strict is true (policy gate)."
        else:
            self.reason = "; ".join(summary.messages) or "policy gate failed"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def load_expected_tool_names(fixture_dir: Path) -> list[str]:
    """Tool names for :class:`ToolCorrectnessMetric` from fixture metadata."""
    fixture_dir = fixture_dir.resolve()
    traj_path = fixture_dir / "expected_tool_trajectory.json"
    if traj_path.is_file():
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        tools = data.get("tools")
        if isinstance(tools, list):
            return [str(t).strip() for t in tools if str(t).strip()]
    tool_args = fixture_dir / "tool_args.json"
    if tool_args.is_file():
        from compare_policy_lists import load_observed_tool_args

        calls = load_observed_tool_args(tool_args).get("tool_calls") or []
        names: list[str] = []
        for call in calls:
            if isinstance(call, dict):
                name = str(call.get("tool") or call.get("name") or "").strip()
                if name:
                    names.append(name)
        return names
    return []


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _agent_llama_base_url() -> str:
    """Same OpenAI-compatible URL as Pi ``llama-cpp`` (``AGENT_LLAMA_BASE_URL``)."""
    for key in ("AGENT_LLAMA_BASE_URL", "AGENT_LLAMA_MODE_BASE_URL"):
        raw = (os.environ.get(key) or "").strip().rstrip("/")
        if raw:
            return raw if raw.endswith("/v1") else f"{raw}/v1"
    return ""


def _agent_llama_model_id() -> str:
    return (
        os.environ.get("AGENT_DEFAULT_MODEL")
        or os.environ.get("AGENT_LLAMA_MODEL_ID")
        or os.environ.get("OPENAI_MODEL_NAME")
        or "gpt-4"
    ).strip()


def deepeval_judge_setup_hint() -> str:
    """Human-readable hint when no TaskCompletion judge is configured."""
    return (
        "Configure a DeepEval judge: USE_BEDROCK_JUDGE=1 with AWS_REGION and "
        "AGENT_DEFAULT_MODEL (or DEEPEVAL_BEDROCK_MODEL); or USE_OPENAI_MODEL=1 "
        "with OPENAI_BASE_URL or AGENT_LLAMA_BASE_URL; or USE_LOCAL_MODEL=1."
    )


def _agent_default_provider() -> str:
    return (os.environ.get("AGENT_DEFAULT_PROVIDER") or "").strip().lower()


def _should_try_bedrock_judge() -> bool:
    if _env_truthy("USE_BEDROCK_JUDGE") or _env_truthy("DEEPEVAL_USE_BEDROCK"):
        return True
    if _env_truthy("USE_OPENAI_MODEL") or _env_truthy("USE_LOCAL_MODEL"):
        return False
    return _agent_default_provider() in {"amazon-bedrock", "bedrock"}


def _bedrock_region() -> str:
    for key in (
        "DEEPEVAL_BEDROCK_REGION",
        "AWS_BEDROCK_REGION",
        "AWS_REGION",
    ):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    try:
        from tools.config import AWS_REGION

        return str(AWS_REGION or "").strip()
    except ImportError:
        return ""


def _bedrock_model_id() -> str:
    for key in (
        "DEEPEVAL_BEDROCK_MODEL",
        "AWS_BEDROCK_MODEL_NAME",
        "AGENT_DEFAULT_MODEL",
    ):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    return ""


def _build_amazon_bedrock_judge() -> Any | None:
    """DeepEval ``AmazonBedrockModel`` (aiobotocore / default AWS credential chain)."""
    try:
        from deepeval.models import AmazonBedrockModel
    except ImportError:
        return None

    model_id = _bedrock_model_id()
    region = _bedrock_region()
    if not model_id or not region:
        return None

    kwargs: dict[str, Any] = {}
    for env_key, arg in (
        ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
        ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
        ("AWS_SESSION_TOKEN", "aws_session_token"),
    ):
        val = (os.environ.get(env_key) or "").strip()
        if val:
            kwargs[arg] = val
    if not kwargs:
        try:
            from tools.config import AWS_ACCESS_KEY, AWS_SECRET_KEY

            if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                kwargs["aws_access_key_id"] = str(AWS_ACCESS_KEY).strip()
                kwargs["aws_secret_access_key"] = str(AWS_SECRET_KEY).strip()
        except ImportError:
            pass

    try:
        return AmazonBedrockModel(model=model_id, region=region, **kwargs)
    except Exception:
        return None


def _deepeval_judge_model() -> Any | None:
    """Judge for DeepEval 4.x metrics (Bedrock, OpenAI-compatible local, or OpenAI)."""
    if _should_try_bedrock_judge():
        bedrock = _build_amazon_bedrock_judge()
        if bedrock is not None:
            return bedrock

    try:
        from deepeval.models import LocalModel, OpenAIModel
    except ImportError:
        return None

    model_name = (
        os.environ.get("LOCAL_MODEL_NAME")
        or os.environ.get("OPENAI_MODEL_NAME")
        or _agent_llama_model_id()
        or "gpt-4"
    ).strip()

    if _env_truthy("USE_LOCAL_MODEL"):
        base_url = (os.environ.get("LOCAL_MODEL_BASE_URL") or "").strip()
        if not base_url:
            base_url = _agent_llama_base_url()
        if not base_url:
            return None
        api_key = (os.environ.get("LOCAL_MODEL_API_KEY") or "local").strip()
        return LocalModel(model=model_name, base_url=base_url, api_key=api_key)

    if _env_truthy("USE_OPENAI_MODEL"):
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not base_url:
            base_url = _agent_llama_base_url()
        api_key = (os.environ.get("OPENAI_API_KEY") or "local").strip()
        if base_url:
            return LocalModel(model=model_name, base_url=base_url, api_key=api_key)
        if api_key:
            return OpenAIModel(model=model_name)

    return None


def expected_tool_calls(fixture_dir: Path) -> list[Any]:
    """DeepEval 4.x ``LLMTestCase.expected_tools`` from fixture metadata."""
    if ToolCall is None:
        return []
    return [ToolCall(name=n) for n in load_expected_tool_names(fixture_dir)]


def build_regression_deepeval_metrics(
    fixture_dir: Path,
    *,
    include_policy_strict: bool = True,
    include_tool_correctness: bool = True,
    include_task_completion: bool = True,
    task_completion_threshold: float = 0.7,
    tool_correctness_threshold: float = 0.5,
    task: str | None = None,
) -> list[Any]:
    """
    Metrics for local agent regression via ``assert_test``.

    - ``PolicyStrictMetric`` — deterministic policy gate (always safe offline).
    - ``ToolCorrectnessMetric`` — needs a LangGraph trace with tool spans
      (set ``AGENT_EVAL_DEEPEVAL_TRACING=1`` for LangGraph live runs).
    - ``TaskCompletionMetric`` — LLM judge over the full trace (Bedrock via
      ``USE_BEDROCK_JUDGE=1``, OpenAI-compatible local server, or OpenAI); not
      necessarily the same runtime as the orchestrator unless env aligns them.
    """
    if ToolCorrectnessMetric is None or TaskCompletionMetric is None:
        raise ImportError("deepeval is not installed")

    metrics: list[Any] = []
    if include_policy_strict:
        metrics.append(PolicyStrictMetric())

    expected_names = load_expected_tool_names(fixture_dir)
    if include_tool_correctness and expected_names:
        tc_kwargs: dict[str, Any] = {
            "threshold": tool_correctness_threshold,
            "should_consider_ordering": False,
            "should_exact_match": False,
            # Sync path scores tools_called vs expected_tools without LLM tool-selection.
            "async_mode": False,
        }
        judge = _deepeval_judge_model()
        if judge is not None:
            tc_kwargs["model"] = judge
        # DeepEval 3.x: expected tools on the metric; 4.x: on LLMTestCase.expected_tools.
        sig = inspect.signature(ToolCorrectnessMetric.__init__)
        if "expected_tools" in sig.parameters:
            tc_kwargs["expected_tools"] = expected_tool_calls(fixture_dir)
        try:
            metrics.append(ToolCorrectnessMetric(**tc_kwargs))
        except Exception:
            pass

    if include_task_completion:
        kwargs: dict[str, Any] = {"threshold": task_completion_threshold}
        if task and task.strip():
            kwargs["task"] = task.strip()
        judge = _deepeval_judge_model()
        if judge is not None:
            kwargs["model"] = judge
        try:
            metrics.append(TaskCompletionMetric(**kwargs))
        except Exception:
            pass

    return metrics


def _parse_expected_payload(test_case: LLMTestCase) -> dict[str, Any]:
    raw = getattr(test_case, "expected_output", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected_output must be JSON: {raw}") from exc
    if not isinstance(data, dict) or (
        "gold_case" not in data and "fixture" not in data
    ):
        raise ValueError('expected_output JSON must include "gold_case" path')
    return data
