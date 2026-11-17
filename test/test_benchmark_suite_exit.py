"""Benchmark suite exit codes: technical vs QA."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from run_benchmark_suite import (  # noqa: E402
    _accumulate_surface_outcome,
    _apply_qa_outcome,
)


def test_apply_qa_outcome_poor_score_technical_zero():
    rec: dict = {"exit_code": 1}
    _apply_qa_outcome(
        rec, technical_exit=0, qa_pass=False, message="overall_pass=False"
    )
    assert rec["exit_code"] == 0
    assert rec["qa_pass"] is False
    assert rec["qa_completed"] is True


def test_accumulate_surface_qa_fail_does_not_raise_technical():
    rec = {"skipped": False, "exit_code": 0, "qa_completed": True, "qa_pass": False}
    max_t, any_qa = _accumulate_surface_outcome(rec, 0, False)
    assert max_t == 0
    assert any_qa is True
