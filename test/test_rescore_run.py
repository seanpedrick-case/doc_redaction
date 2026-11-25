"""Offline rescore_run against example_smoke artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from rescore_run import rescore_run  # noqa: E402

_SMOKE = _EVAL / "gold" / "example_smoke"


def test_rescore_smoke_no_deepeval(tmp_path: Path, monkeypatch):
    if not _SMOKE.is_dir():
        return
    run_dir = tmp_path / "prior_run"
    run_dir.mkdir()
    for name in (
        "agent_review_file.csv",
        "ocr_results_with_words.csv",
        "tool_args.json",
    ):
        src = _SMOKE / name
        if src.is_file():
            shutil.copy2(src, run_dir / name)

    out_dir = tmp_path / "rescore_out"
    summary, code = rescore_run(
        run_dir=run_dir,
        gold_case_dir=_SMOKE,
        out_dir=out_dir,
        in_place=False,
        deepeval=False,
        append_csv=False,
        fail_on_qa=False,
    )
    assert code == 0
    assert summary.get("run_source") == "rescore"
    assert summary.get("gold_case")
    assert (out_dir / "eval_summary.json").is_file()
    assert (out_dir / "policy_gate.json").is_file()
    assert (out_dir / "gold_geometry.json").is_file()

    csv_path = tmp_path / "eval_run_summary.csv"
    monkeypatch.setattr(
        "eval_run_summary_csv.default_summary_csv_path",
        lambda: csv_path,
    )
    summary2, code2 = rescore_run(
        run_dir=run_dir,
        gold_case_dir=_SMOKE,
        out_dir=tmp_path / "rescore_out2",
        deepeval=False,
        append_csv=True,
        fail_on_qa=False,
    )
    assert code2 == 0
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "rescore" in text

    data = json.loads((tmp_path / "rescore_out2" / "eval_summary.json").read_text())
    assert data.get("eval_turn_kind") == "rescore"
    metric_names = {m.get("name") for m in data.get("metrics") or []}
    assert "policy_strict" in metric_names
    assert "gold_geometry" in metric_names
    assert summary2.get("metrics")
