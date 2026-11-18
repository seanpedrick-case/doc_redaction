"""Tests for synthetic Task Completion actual_output (main app eval)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from app_task_completion_output import (
    build_app_task_completion_actual_output,
)  # noqa: E402
from derive_app_eval_instructions import AppEvalRedactionSettings  # noqa: E402
from run_agent_regression import GateSummary  # noqa: E402


def _settings(**kwargs) -> AppEvalRedactionSettings:
    return AppEvalRedactionSettings(
        text_extraction_method="Local model - selectable text",
        pii_identification_method="Local",
        **kwargs,
    )


def test_build_output_non_empty_and_omits_pii_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("APP_EVAL_TASK_COMPLETION_INCLUDE_TERM_SAMPLES", raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "agent_review_file.csv").write_text(
        "image,page,label,color,xmin,ymin,xmax,ymax,id,text\n"
        'x.png,1,NAME,"(0,0,0)",0,0,1,1,a,SecretName\n',
        encoding="utf-8",
    )
    (run_dir / "redacted.pdf").write_bytes(b"%PDF-1.4")
    (run_dir / "ocr_results_with_words.csv").write_text(
        "page,line,word_text\n1,1,hello\n", encoding="utf-8"
    )
    policy_gate = {
        "pass_strict": False,
        "must_redact_count": 2,
        "report": {
            "summary": {"pages_total": 1, "pages_with_issues": 1},
            "pages": {
                "1": {
                    "uncovered_terms": [{"text": "SecretUncovered", "page": 1}],
                }
            },
        },
    }
    (run_dir / "policy_gate.json").write_text(json.dumps(policy_gate), encoding="utf-8")
    (run_dir / "gold_geometry.json").write_text(
        json.dumps({"f1": 0.5, "iou_threshold": 0.85}), encoding="utf-8"
    )
    (run_dir / "ocr_gold.json").write_text(
        json.dumps({"f1": 0.9, "skipped": False}), encoding="utf-8"
    )

    summary = GateSummary()
    summary.policy_gate = policy_gate
    summary.gold_geometry = {"f1": 0.5, "iou_threshold": 0.85}
    summary.ocr_gold = {"f1": 0.9, "skipped": False}

    text = build_app_task_completion_actual_output(
        run_dir=run_dir,
        settings=_settings(),
        document_name="doc.pdf",
        gate_summary=summary,
        gold_case_id="test_case",
    )
    assert text
    assert "Main-app redaction run completed" in text
    assert "pass_strict: False" in text
    assert "SecretUncovered" not in text
    assert "SecretName" not in text


def test_include_term_samples_uses_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_EVAL_TASK_COMPLETION_INCLUDE_TERM_SAMPLES", "1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    policy_gate = {
        "pass_strict": False,
        "report": {
            "summary": {"pages_total": 1, "pages_with_issues": 1},
            "pages": {
                "1": {
                    "uncovered_terms": [{"text": "HiddenToken", "page": 1}],
                }
            },
        },
    }
    summary = GateSummary()
    summary.policy_gate = policy_gate

    text = build_app_task_completion_actual_output(
        run_dir=run_dir,
        settings=_settings(text_extraction_only=True),
        document_name="doc.pdf",
        gate_summary=summary,
        gold_case_id="test_case",
    )
    assert "HiddenToken" not in text
    assert "[REDACTED len=" in text


def test_app_overall_pass_excludes_task_completion_by_default(monkeypatch):
    monkeypatch.delenv("APP_EVAL_TASK_COMPLETION_AFFECTS_OVERALL", raising=False)
    from post_run_app_eval import _app_eval_overall_pass

    metrics = [
        {"name": "policy_strict", "pass": False, "skipped": False},
        {"name": "task_completion", "pass": False, "skipped": False},
    ]
    assert _app_eval_overall_pass(metrics) is False

    metrics_task_only_fail = [
        {"name": "policy_strict", "pass": True, "skipped": False},
        {"name": "task_completion", "pass": False, "skipped": False},
    ]
    assert _app_eval_overall_pass(metrics_task_only_fail) is True


def test_app_overall_pass_includes_task_when_env_set(monkeypatch):
    monkeypatch.setenv("APP_EVAL_TASK_COMPLETION_AFFECTS_OVERALL", "1")
    from post_run_app_eval import _app_eval_overall_pass

    metrics = [
        {"name": "policy_strict", "pass": True, "skipped": False},
        {"name": "task_completion", "pass": False, "skipped": False},
    ]
    assert _app_eval_overall_pass(metrics) is False
