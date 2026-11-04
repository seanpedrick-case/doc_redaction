"""Tests for eval_run_summary_csv append log."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from eval_run_summary_csv import (  # noqa: E402
    SUMMARY_CSV_COLUMNS,
    append_eval_run_summary_row,
    rebuild_eval_run_summary_csv,
)


def test_append_eval_run_summary_row(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_DEEPEVAL_TRACING", "1")
    monkeypatch.setenv("AGENT_EVAL_RUN_SUMMARY_CSV_PATH", str(tmp_path / "eval_run_summary.csv"))

    run_dir = tmp_path / "run_20261104_122626"
    run_dir.mkdir()
    summary = {
        "document_name": "doc.pdf",
        "fixture": "/fixtures/example_live",
        "overall_pass": False,
        "workflow_incomplete": False,
        "eval_turn_kind": "initial_task",
        "follow_up_steer_count": 0,
        "metrics": [
            {"name": "policy_strict", "pass": False, "score": 0.0, "skipped": False},
            {"name": "tool_correctness", "pass": True, "score": 1.0, "skipped": False},
            {"name": "task_completion", "pass": True, "score": 0.9, "skipped": False},
        ],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "policy_gate.json").write_text(
        json.dumps({"pass_strict": False}), encoding="utf-8"
    )

    out = append_eval_run_summary_row(
        summary=summary,
        run_dir=run_dir,
        task_instructions="Redact names",
        run_id="gradio_abc_20261104_122626",
    )
    assert out is not None
    assert out.is_file()

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "gradio_abc_20261104_122626"
    assert row["document_name"] == "doc.pdf"
    assert row["run_timestamp_utc"] == "2026-11-04T12:26:26Z"
    assert row["policy_strict_pass"] == "False"
    assert row["tool_correctness_pass"] == "True"
    assert row["pass_strict_from_gate"] == "false"
    assert "eval_summary.json" in row["output_files"]
    assert set(row.keys()) == set(SUMMARY_CSV_COLUMNS)


def test_rebuild_from_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_RUN_SUMMARY_CSV_PATH", str(tmp_path / "rebuilt.csv"))
    runs = tmp_path / "runs"
    run_dir = runs / "gradio_x_20261104_100000"
    run_dir.mkdir(parents=True)
    summary = {
        "document_name": "a.pdf",
        "overall_pass": True,
        "metrics": [],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    count = rebuild_eval_run_summary_csv(search_root=runs, dest=tmp_path / "rebuilt.csv")
    assert count == 1
    text = (tmp_path / "rebuilt.csv").read_text(encoding="utf-8")
    assert "a.pdf" in text
