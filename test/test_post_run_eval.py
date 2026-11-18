"""Tests for Gradio post-run eval summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
if str(_REPO / "agent-redact" / "shared") not in sys.path:
    sys.path.insert(0, str(_REPO / "agent-redact" / "shared"))

from post_run_eval import (
    score_post_run_metrics,
    write_session_post_run_eval,
)  # noqa: E402
from run_agent_regression import run_regression  # noqa: E402

_GOLD = _EVAL / "gold" / "example_smoke"


def test_write_session_post_run_eval_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_EVAL_DEEPEVAL_TRACING", raising=False)
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    path = write_session_post_run_eval(
        session_hash="testsession",
        tool_events=[],
        task_instructions="test",
        assistant_output="done",
    )
    assert path is None


def test_score_post_run_metrics_policy_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_EVAL_GOLD", str(_GOLD))
    summary, run_dir = run_regression(
        gold_case_dir=_GOLD,
        gates_only=True,
        orchestrator="langgraph",
    )
    assert summary.exit_code == 0
    tool_args_path = _GOLD / "tool_args.json"
    tool_args = (
        json.loads(tool_args_path.read_text(encoding="utf-8"))
        if tool_args_path.is_file()
        else {"tool_calls": []}
    )
    metrics = score_post_run_metrics(
        run_dir=run_dir,
        gold_case_dir=_GOLD,
        task_instructions="smoke",
        tool_args=tool_args,
        assistant_output="ok",
    )
    policy = next(m for m in metrics if m["name"] == "policy_strict")
    assert policy.get("skipped") is not True
    assert policy.get("pass") is True


def test_score_post_run_metrics_three_names(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_EVAL_GOLD", str(_GOLD))
    summary, run_dir = run_regression(
        gold_case_dir=_GOLD,
        gates_only=True,
        orchestrator="langgraph",
    )
    assert summary.exit_code == 0
    metrics = score_post_run_metrics(
        run_dir=run_dir,
        gold_case_dir=_GOLD,
        task_instructions="smoke task",
        tool_args={"tool_calls": []},
        assistant_output="finished",
    )
    assert {m["name"] for m in metrics} == {
        "policy_strict",
        "tool_correctness",
        "task_completion",
    }
