"""Tests for main-app eval env helpers (agent-redact/eval/app_eval_env.py)."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from app_eval_env import app_eval_deepeval_metrics_enabled  # noqa: E402


def test_app_eval_deepeval_metrics_default_on_when_app_eval_on(monkeypatch):
    monkeypatch.delenv("APP_EVAL_DEEPEVAL_METRICS", raising=False)
    monkeypatch.setattr("app_eval_env.app_eval_enabled", lambda: True)
    assert app_eval_deepeval_metrics_enabled() is True


def test_app_eval_deepeval_metrics_explicit_off(monkeypatch):
    monkeypatch.setenv("APP_EVAL_DEEPEVAL_METRICS", "0")
    monkeypatch.setattr("app_eval_env.app_eval_enabled", lambda: True)
    assert app_eval_deepeval_metrics_enabled() is False


def test_app_eval_deepeval_metrics_false_when_app_eval_disabled(monkeypatch):
    monkeypatch.delenv("APP_EVAL_DEEPEVAL_METRICS", raising=False)
    monkeypatch.setattr("app_eval_env.app_eval_enabled", lambda: False)
    assert app_eval_deepeval_metrics_enabled() is False
