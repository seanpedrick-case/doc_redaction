"""Tests for agent-redact/eval/benchmark_cases.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from benchmark_cases import (  # noqa: E402
    audit_benchmark_gold_cases,
    case_readiness,
    discover_benchmark_gold_cases,
)

_GOLD = _EVAL / "gold"


def test_example_smoke_readiness():
    smoke = _GOLD / "example_smoke"
    if not smoke.is_dir():
        return
    row = case_readiness(smoke)
    assert row.has_ideal_review is True
    assert row.benchmark_eligible is False  # no input PDF


def test_discover_benchmark_excludes_smoke_without_pdf():
    cases = discover_benchmark_gold_cases(_GOLD)
    names = {p.name for p in cases}
    assert "example_smoke" not in names


def test_audit_returns_smoke_row():
    reports = audit_benchmark_gold_cases(_GOLD)
    ids = {r.case_id for r in reports}
    assert "example_smoke" in ids


def test_harness_surface_matches_agent_requirements():
    smoke = _GOLD / "example_smoke"
    if not smoke.is_dir():
        return
    row = case_readiness(smoke)
    assert row.harness_surface_ready == row.agent_surface_ready
