#!/usr/bin/env python3
"""Recompute gold_geometry.json for mirrored runs and optionally rebuild eval_run_summary.csv.

Use after changing core-coverage scoring or updating ideal_review_file.csv, without
re-running the agent.

Example::

    python agent-redact/eval/refresh_eval_geometry.py
    python agent-redact/eval/refresh_eval_geometry.py --run-id gradio_8s2ysucu_20261104_160453
    python agent-redact/eval/refresh_eval_geometry.py --no-rebuild-csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_gold_case_dir(repo_root: Path, gold_case: str) -> Path | None:
    raw = (gold_case or "").replace("\\", "/").strip()
    if not raw:
        return None
    if "agent-redact/" in raw:
        rel = raw.split("agent-redact/", 1)[1]
        candidate = (repo_root / "agent-redact" / rel).resolve()
        return candidate if candidate.is_dir() else None
    path = Path(raw)
    if path.is_dir():
        return path.resolve()
    if not path.is_absolute():
        candidate = (repo_root / raw).resolve()
        return candidate if candidate.is_dir() else None
    return None


def _gold_case_from_run(run_dir: Path) -> str:
    for name in ("eval_summary.json", "gradio_run_meta.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            gc = str(data.get("gold_case") or "").strip()
            if gc:
                return gc
    return ""


@dataclass
class RefreshResult:
    run_id: str
    scoring: str
    f1: float
    recall: float
    precision: float
    core_gold_count: int
    skipped_reason: str = ""


def refresh_run_geometry(
    run_dir: Path,
    *,
    repo_root: Path,
    geo_opts: dict,
) -> RefreshResult | None:
    from compare_review_csvs import compare_review_csvs

    run_dir = run_dir.resolve()
    run_id = run_dir.name
    agent_csv = run_dir / "agent_review_file.csv"
    if not agent_csv.is_file():
        return RefreshResult(run_id, "", 0.0, 0.0, 0.0, 0, "no agent_review_file.csv")

    gold_dir = _resolve_gold_case_dir(repo_root, _gold_case_from_run(run_dir))
    if gold_dir is None:
        return RefreshResult(
            run_id, "", 0.0, 0.0, 0.0, 0, "no resolvable gold_case dir"
        )

    ideal = gold_dir / "ideal_review_file.csv"
    if not ideal.is_file():
        return RefreshResult(
            run_id, "", 0.0, 0.0, 0.0, 0, "gold case has no ideal_review_file.csv"
        )

    report = compare_review_csvs(
        ideal,
        agent_csv,
        threshold=geo_opts["threshold"],
        mode=geo_opts["mode"],
        many_to_one=geo_opts["many_to_one"],
        scoring=geo_opts["scoring"],
    )
    (run_dir / "gold_geometry.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    summary_path = run_dir / "eval_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
        if isinstance(summary, dict):
            metrics = summary.get("metrics")
            if isinstance(metrics, list):
                for m in metrics:
                    if isinstance(m, dict) and m.get("name") == "gold_geometry":
                        m["score"] = report.f1
                        m["reason"] = (
                            f"precision={report.precision} recall={report.recall} "
                            f"scoring={report.scoring} core_gold={report.core_gold_count}"
                        )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return RefreshResult(
        run_id,
        report.scoring,
        float(report.f1),
        float(report.recall),
        float(report.precision),
        int(report.core_gold_count),
    )


def refresh_runs(
    *,
    runs_root: Path,
    repo_root: Path,
    run_id: str | None = None,
) -> list[RefreshResult]:
    from run_agent_regression import gold_geometry_compare_options

    geo_opts = gold_geometry_compare_options()
    results: list[RefreshResult] = []

    if run_id:
        run_dir = runs_root / run_id
        if not run_dir.is_dir():
            results.append(
                RefreshResult(run_id, "", 0.0, 0.0, 0.0, 0, "run directory not found")
            )
            return results
        out = refresh_run_geometry(run_dir, repo_root=repo_root, geo_opts=geo_opts)
        if out is not None:
            results.append(out)
        return results

    if not runs_root.is_dir():
        return results

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        out = refresh_run_geometry(run_dir, repo_root=repo_root, geo_opts=geo_opts)
        if out is not None:
            results.append(out)
    return results


def _build_parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Refresh gold_geometry.json under eval/runs and rebuild CSV."
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=repo / "agent-redact" / "eval" / "runs",
        help="Directory containing mirrored gradio_* run folders.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Refresh a single run folder name (default: all under --runs-root).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=repo / "agent-redact" / "eval" / "outputs" / "eval_run_summary.csv",
        help="Rebuild eval_run_summary.csv to this path.",
    )
    p.add_argument(
        "--no-rebuild-csv",
        action="store_true",
        help="Only update gold_geometry.json / eval_summary.json per run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = _repo_root()
    eval_dir = repo_root / "agent-redact" / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    run_id = (args.run_id or "").strip() or None
    results = refresh_runs(
        runs_root=args.runs_root.resolve(),
        repo_root=repo_root,
        run_id=run_id,
    )

    updated = [r for r in results if not r.skipped_reason]
    skipped = [r for r in results if r.skipped_reason]

    for r in updated:
        print(
            f"{r.run_id}: scoring={r.scoring} f1={r.f1:.4f} "
            f"recall={r.recall:.4f} precision={r.precision:.4f} "
            f"core_gold={r.core_gold_count}"
        )
    for r in skipped:
        print(f"{r.run_id}: skipped ({r.skipped_reason})", file=sys.stderr)

    if not args.no_rebuild_csv:
        from eval_run_summary_csv import rebuild_eval_run_summary_csv

        count = rebuild_eval_run_summary_csv(
            search_root=args.runs_root.resolve(),
            dest=args.csv_out.resolve(),
        )
        print(f"Rebuilt {args.csv_out} ({count} rows from eval_summary.json).")

    return 0 if updated or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
