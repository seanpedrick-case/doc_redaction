"""DeepEval agent tests (optional dev dependency)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

deepeval = pytest.importorskip("deepeval")
from deepeval import assert_test  # noqa: E402
from deepeval.dataset import Golden  # noqa: E402
from deepeval.test_case import LLMTestCaseParams  # noqa: E402
from deepeval.tracing import observe, update_current_trace  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
for path in (_REPO, _REPO / "agent-redact", _REPO / "agent-redact" / "shared"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from deepeval_metrics import (  # noqa: E402
    PolicyStrictMetric,
    build_regression_deepeval_metrics,
)
from run_agent_regression import run_regression  # noqa: E402

_SMOKE_FIXTURE = _EVAL / "fixtures" / "example_smoke"


@pytest.fixture
def smoke_golden() -> Golden:
    return Golden(
        input="Synthetic smoke fixture (offline policy gate).",
        expected_output=json.dumps({"fixture": str(_SMOKE_FIXTURE)}),
    )


def test_deepeval_policy_strict_gates_only(smoke_golden: Golden):
    """Offline: policy gate only (no ToolCorrectness / TaskCompletion — no agent trace)."""
    run_dir_holder: dict[str, Path] = {}

    @observe()
    def _run_offline_gate() -> str:
        summary, run_dir = run_regression(
            fixture_dir=_SMOKE_FIXTURE,
            gates_only=True,
            orchestrator="langgraph",
        )
        run_dir_holder["run_dir"] = run_dir
        update_current_trace(
            input=smoke_golden.input,
            output=json.dumps(
                {
                    "run_dir": str(run_dir),
                    "exit_code": summary.exit_code,
                    "pass_strict": (summary.policy_gate or {}).get("pass_strict"),
                }
            ),
        )
        return str(run_dir)

    _run_offline_gate()
    scored_golden = Golden(
        input=smoke_golden.input,
        expected_output=json.dumps(
            {
                "fixture": str(_SMOKE_FIXTURE),
                "run_dir": str(run_dir_holder["run_dir"]),
            }
        ),
    )
    assert_test(
        golden=scored_golden,
        metrics=[PolicyStrictMetric()],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )


@pytest.mark.live_agent
@pytest.mark.deepeval_judge
@pytest.mark.skip(
    reason="Requires compose, orchestrator LLM, AGENT_EVAL_DEEPEVAL_TRACING=1, judge API."
)
def test_deepeval_langgraph_live_tool_and_task_metrics():
    """Live LangGraph: PolicyStrict + ToolCorrectness + TaskCompletion (see example_commands.txt)."""
    pytest.importorskip("langchain_core")
    import os

    live_fixture = _EVAL / "fixtures" / "example_live"
    if not live_fixture.is_dir():
        pytest.skip("example_live fixture missing")

    instructions = (live_fixture / "instructions.txt").read_text(encoding="utf-8")
    golden = Golden(
        input=instructions,
        expected_output=json.dumps({"fixture": str(live_fixture)}),
    )

    os.environ["AGENT_EVAL_DEEPEVAL_TRACING"] = "1"
    run_dir_holder: dict[str, Path] = {}

    @observe()
    def _run_live() -> str:
        summary, run_dir = run_regression(
            fixture_dir=live_fixture,
            orchestrator="langgraph",
            gates_only=False,
        )
        run_dir_holder["run_dir"] = run_dir
        update_current_trace(
            input=golden.input,
            output=json.dumps(
                {
                    "run_dir": str(run_dir),
                    "exit_code": summary.exit_code,
                    "pass_strict": (summary.policy_gate or {}).get("pass_strict"),
                }
            ),
        )
        return str(run_dir)

    _run_live()
    run_dir_path = run_dir_holder["run_dir"]
    scored_golden = Golden(
        input=golden.input,
        expected_output=json.dumps(
            {"fixture": str(live_fixture), "run_dir": str(run_dir_path)}
        ),
    )
    metrics = build_regression_deepeval_metrics(
        live_fixture,
        include_policy_strict=True,
        include_tool_correctness=True,
        include_task_completion=True,
        task=instructions,
    )
    assert_test(
        golden=scored_golden,
        metrics=metrics,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )


@pytest.mark.live_agent
@pytest.mark.skip(reason="Requires Node pi CLI and orchestrator LLM.")
def test_deepeval_pi_live_observe_wrapper():
    """Pi path uses manual @observe at the Python orchestration boundary."""
    live_fixture = _EVAL / "fixtures" / "example_live"
    golden = Golden(
        input="Pi live regression",
        expected_output=json.dumps({"fixture": str(live_fixture)}),
    )

    @observe()
    def _run_pi() -> None:
        summary, run_dir = run_regression(
            fixture_dir=live_fixture,
            orchestrator="pi",
            gates_only=False,
        )
        update_current_trace(
            input=golden.input,
            output=json.dumps(
                {"run_dir": str(run_dir), "exit_code": summary.exit_code}
            ),
        )

    _run_pi()
    assert_test(
        golden=golden,
        metrics=build_regression_deepeval_metrics(
            live_fixture,
            include_task_completion=False,
            task=(live_fixture / "instructions.txt").read_text(encoding="utf-8"),
        ),
    )
