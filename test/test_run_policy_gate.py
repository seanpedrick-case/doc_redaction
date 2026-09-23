"""Tests for agent-redact/eval/run_policy_gate.py against example_smoke."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from run_policy_gate import main as policy_main  # noqa: E402
from run_policy_gate import run_policy_gate  # noqa: E402

_FIXTURE = _EVAL / "fixtures" / "example_smoke"


def test_run_policy_gate_example_smoke():
    summary = run_policy_gate(
        review_csv=_FIXTURE / "agent_review_file.csv",
        ocr_words_csv=_FIXTURE / "ocr_results_with_words.csv",
        must_redact=["Alice"],
        must_not_redact=[r"Example\s+Council"],
    )
    assert summary["pass_strict"] is True


def test_policy_gate_cli_fixture():
    code = policy_main(["--fixture", str(_FIXTURE)])
    assert code == 0


def test_policy_gate_cli_dual():
    code = policy_main(["--fixture", str(_FIXTURE), "--dual"])
    assert code == 0
