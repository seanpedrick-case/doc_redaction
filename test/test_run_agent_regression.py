"""Tests for agent-redact/eval/run_agent_regression.py (offline gates)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from run_agent_regression import main as regression_main  # noqa: E402
from run_agent_regression import run_regression  # noqa: E402

_GOLD = _EVAL / "gold" / "example_smoke"


def test_run_regression_gates_only_example_smoke():
    summary, run_dir = run_regression(
        gold_case_dir=_GOLD,
        gates_only=True,
        orchestrator="langgraph",
    )
    assert run_dir.is_dir()
    assert summary.exit_code == 0
    assert summary.policy_gate is not None
    assert summary.policy_gate.get("pass_strict") is True


def test_regression_cli_gates_only():
    code = regression_main(["--gold", str(_GOLD), "--gates-only"])
    assert code == 0
