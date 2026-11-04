"""Append-only CSV log of agent eval runs (one row per run).

Layout follows the topic-modeller ``eval_run_summary.csv`` pattern: run metadata,
timestamps, pass/score columns, then ``output_files`` / ``intermediate_files``.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_agent_regression import GateSummary, _ensure_agent_imports

_RUN_DIR_STAMP = re.compile(r"^run_(\d{8})_(\d{6})$")

SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "run_id",
    "document_name",
    "task_instructions_preview",
    "orchestrator",
    "model_name",
    "gold_case_path",
    "eval_turn_kind",
    "follow_up_steer_count",
    "run_source",
    "run_timestamp_utc",
    "summary_appended_at_utc",
    "workflow_incomplete",
    "overall_pass",
    "policy_strict_pass",
    "policy_strict_score",
    "policy_strict_skipped",
    "tool_correctness_pass",
    "tool_correctness_score",
    "tool_correctness_skipped",
    "task_completion_pass",
    "task_completion_score",
    "task_completion_skipped",
    "pass_strict_from_gate",
    "gold_geometry_f1",
    "gold_geometry_recall",
    "gold_geometry_precision",
    "gold_geometry_skipped",
    "ocr_gold_f1",
    "ocr_gold_recall",
    "ocr_gold_precision",
    "ocr_gold_skipped",
    "policy_extraction_must_redact_f1",
    "policy_extraction_deny_list_f1",
    "run_dir",
    "eval_summary_file",
    "output_files",
    "intermediate_files",
)


def agent_eval_run_summary_csv_enabled() -> bool:
    """
    Default **on** when ``AGENT_EVAL_DEEPEVAL_TRACING=1`` (opt out with ``AGENT_EVAL_RUN_SUMMARY_CSV=0``).

    Set ``AGENT_EVAL_RUN_SUMMARY_CSV=1`` without tracing to log regression CLI runs only.
    """
    raw = (os.environ.get("AGENT_EVAL_RUN_SUMMARY_CSV") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        from eval.arize_monitoring import agent_eval_deepeval_tracing_enabled
    except ImportError:
        from arize_monitoring import agent_eval_deepeval_tracing_enabled

    return agent_eval_deepeval_tracing_enabled()


def default_summary_csv_path() -> Path:
    custom = (os.environ.get("AGENT_EVAL_RUN_SUMMARY_CSV_PATH") or "").strip()
    _, _, eval_dir = _ensure_agent_imports()
    if custom:
        path = Path(custom)
        if not path.is_absolute():
            repo_root, _, _ = _ensure_agent_imports()
            path = (repo_root / custom).resolve()
        return path
    return (eval_dir / "outputs" / "eval_run_summary.csv").resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_timestamp_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    m = _RUN_DIR_STAMP.match(name)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"
    if name.startswith("gradio_") or "_" in name:
        parts = name.rsplit("_", 2)
        if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-1].isdigit():
            d, t = parts[-2], parts[-1]
            if len(t) == 6:
                return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"
    try:
        return datetime.fromtimestamp(
            run_dir.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return _utc_now_iso()


def _model_label() -> str:
    provider = (os.environ.get("AGENT_DEFAULT_PROVIDER") or "").strip()
    model = (
        (os.environ.get("AGENT_DEFAULT_MODEL") or "").strip()
        or (os.environ.get("AGENT_LLAMA_MODEL_ID") or "").strip()
        or (os.environ.get("OPENAI_MODEL_NAME") or "").strip()
    )
    if provider and model:
        return f"{provider}/{model}"
    return model or provider or ""


def _metric_field(metrics: list[dict[str, Any]], name: str, field: str) -> Any:
    for m in metrics:
        if m.get("name") == name:
            return m.get(field)
    return ""


def _read_run_json(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _is_truthy_metric(val: Any) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _fill_deterministic_tier_columns(
    row: dict[str, Any],
    *,
    run_dir: Path,
    metrics: list[dict[str, Any]],
) -> None:
    """Populate gold geometry, OCR gold, and policy extraction columns."""
    geo = _read_run_json(run_dir, "gold_geometry.json")
    if geo is not None:
        row["gold_geometry_f1"] = geo.get("f1", "")
        row["gold_geometry_recall"] = geo.get("recall", "")
        row["gold_geometry_precision"] = geo.get("precision", "")
        row["gold_geometry_skipped"] = False
    elif _is_truthy_metric(_metric_field(metrics, "gold_geometry", "skipped")):
        row["gold_geometry_skipped"] = True
    else:
        score = _metric_field(metrics, "gold_geometry", "score")
        if score != "" and score is not None:
            row["gold_geometry_f1"] = score
            row["gold_geometry_skipped"] = False
        else:
            row["gold_geometry_skipped"] = True

    ocr = _read_run_json(run_dir, "ocr_gold.json")
    if ocr is not None:
        if ocr.get("skipped"):
            row["ocr_gold_skipped"] = True
        else:
            row["ocr_gold_f1"] = ocr.get("f1", "")
            row["ocr_gold_recall"] = ocr.get("recall", "")
            row["ocr_gold_precision"] = ocr.get("precision", "")
            row["ocr_gold_skipped"] = False
    elif _is_truthy_metric(_metric_field(metrics, "ocr_gold", "skipped")):
        row["ocr_gold_skipped"] = True
    else:
        score = _metric_field(metrics, "ocr_gold", "score")
        if score != "" and score is not None:
            row["ocr_gold_f1"] = score
            row["ocr_gold_skipped"] = False
        else:
            row["ocr_gold_skipped"] = True

    ext = _read_run_json(run_dir, "policy_extraction.json")
    if ext is not None:
        must = ext.get("must_redact") or {}
        row["policy_extraction_must_redact_f1"] = must.get("f1", "")
        deny = ext.get("deny_list")
        if isinstance(deny, dict):
            row["policy_extraction_deny_list_f1"] = deny.get("f1", "")
    else:
        score = _metric_field(metrics, "policy_extraction", "score")
        if score != "" and score is not None:
            row["policy_extraction_must_redact_f1"] = score


def _read_pass_strict_from_gate(run_dir: Path) -> str:
    gate_path = run_dir / "policy_gate.json"
    if not gate_path.is_file():
        return ""
    try:
        data = json.loads(gate_path.read_text(encoding="utf-8"))
        val = data.get("pass_strict")
        if val is None:
            return ""
        return "true" if val else "false"
    except (OSError, json.JSONDecodeError, TypeError):
        return ""


def _list_run_files(run_dir: Path) -> tuple[str, str]:
    if not run_dir.is_dir():
        return "", ""
    names = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    primary = []
    intermediate = []
    for name in names:
        if name == "eval_summary.json":
            primary.append(name)
        elif name in {"redacted.pdf", "agent_review_file.csv"}:
            primary.append(name)
        else:
            intermediate.append(name)
    return ", ".join(primary), " ".join(intermediate)


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_CSV_COLUMNS})


def append_eval_run_summary_row(
    *,
    summary: dict[str, Any],
    run_dir: Path,
    task_instructions: str = "",
    run_id: str | None = None,
    run_source: str = "gradio",
    csv_paths: list[Path] | None = None,
    extra_csv_paths: list[Path] | None = None,
    skip_enabled_check: bool = False,
) -> Path | None:
    """Append one row from a Gradio ``eval_summary.json`` payload."""
    if not skip_enabled_check and not agent_eval_run_summary_csv_enabled():
        return None

    run_dir = run_dir.resolve()
    metrics = summary.get("metrics") or []
    if not isinstance(metrics, list):
        metrics = []

    eval_summary_file = run_dir / "eval_summary.json"
    output_files, intermediate_files = _list_run_files(run_dir)

    row: dict[str, Any] = {
        "run_id": run_id or run_dir.name,
        "document_name": summary.get("document_name") or "",
        "task_instructions_preview": (task_instructions or "")[:500],
        "orchestrator": (os.environ.get("AGENT_ORCHESTRATOR") or "").strip(),
        "model_name": _model_label(),
        "gold_case_path": summary.get("gold_case") or summary.get("fixture") or "",
        "eval_turn_kind": summary.get("eval_turn_kind") or "",
        "follow_up_steer_count": summary.get("follow_up_steer_count", ""),
        "run_source": run_source,
        "run_timestamp_utc": _run_timestamp_from_dir(run_dir),
        "summary_appended_at_utc": _utc_now_iso(),
        "workflow_incomplete": summary.get("workflow_incomplete", ""),
        "overall_pass": summary.get("overall_pass", ""),
        "policy_strict_pass": _metric_field(metrics, "policy_strict", "pass"),
        "policy_strict_score": _metric_field(metrics, "policy_strict", "score"),
        "policy_strict_skipped": _metric_field(metrics, "policy_strict", "skipped"),
        "tool_correctness_pass": _metric_field(metrics, "tool_correctness", "pass"),
        "tool_correctness_score": _metric_field(metrics, "tool_correctness", "score"),
        "tool_correctness_skipped": _metric_field(
            metrics, "tool_correctness", "skipped"
        ),
        "task_completion_pass": _metric_field(metrics, "task_completion", "pass"),
        "task_completion_score": _metric_field(metrics, "task_completion", "score"),
        "task_completion_skipped": _metric_field(metrics, "task_completion", "skipped"),
        "pass_strict_from_gate": _read_pass_strict_from_gate(run_dir),
        "run_dir": str(run_dir),
        "eval_summary_file": (
            str(eval_summary_file) if eval_summary_file.is_file() else ""
        ),
        "output_files": output_files,
        "intermediate_files": intermediate_files,
    }
    _fill_deterministic_tier_columns(row, run_dir=run_dir, metrics=metrics)

    if csv_paths is not None:
        paths = list(csv_paths)
    else:
        paths = [default_summary_csv_path()]
        if extra_csv_paths:
            paths.extend(extra_csv_paths)
    seen: set[str] = set()
    primary: Path | None = None
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        _append_row(path, row)
        if primary is None:
            primary = path.resolve()

    return primary or default_summary_csv_path()


def append_regression_run_row(
    *,
    gates: GateSummary,
    run_dir: Path,
    gold_case_dir: Path,
    orchestrator: str,
    run_source: str = "regression",
) -> Path | None:
    """Append one row after ``run_agent_regression`` (deterministic gates)."""
    if not agent_eval_run_summary_csv_enabled():
        return None

    run_dir = run_dir.resolve()
    policy = gates.policy_gate or {}
    pass_strict = policy.get("pass_strict")
    metrics = [
        {
            "name": "policy_strict",
            "pass": bool(pass_strict) if pass_strict is not None else False,
            "score": 1.0 if pass_strict else 0.0,
            "skipped": False,
        },
        {
            "name": "tool_correctness",
            "pass": None,
            "score": None,
            "skipped": True,
        },
        {
            "name": "task_completion",
            "pass": None,
            "score": None,
            "skipped": True,
        },
    ]
    overall = gates.exit_code == 0 and bool(pass_strict)
    summary = {
        "document_name": gold_case_dir.name,
        "gold_case": str(gold_case_dir.resolve()),
        "workflow_incomplete": False,
        "overall_pass": overall,
        "metrics": metrics,
        "eval_turn_kind": "regression",
        "follow_up_steer_count": 0,
    }
    instructions_path = gold_case_dir / "instructions.txt"
    task = ""
    if instructions_path.is_file():
        task = instructions_path.read_text(encoding="utf-8")[:500]

    prev_orch = os.environ.get("AGENT_ORCHESTRATOR")
    os.environ["AGENT_ORCHESTRATOR"] = orchestrator
    try:
        return append_eval_run_summary_row(
            summary=summary,
            run_dir=run_dir,
            task_instructions=task,
            run_id=run_dir.name,
            run_source=run_source,
        )
    finally:
        if prev_orch is None:
            os.environ.pop("AGENT_ORCHESTRATOR", None)
        else:
            os.environ["AGENT_ORCHESTRATOR"] = prev_orch


def rebuild_eval_run_summary_csv(
    *,
    search_root: Path | None = None,
    dest: Path | None = None,
) -> int:
    """Rebuild ``eval_run_summary.csv`` from ``eval_summary.json`` under *search_root*."""
    _, _, eval_dir = _ensure_agent_imports()
    root = (search_root or eval_dir / "runs").resolve()
    out = (dest or default_summary_csv_path()).resolve()
    if out.is_file():
        out.unlink()

    count = 0
    if not root.is_dir():
        return 0

    summaries = sorted(root.rglob("eval_summary.json"), key=lambda p: p.stat().st_mtime)
    for summary_path in summaries:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        meta_path = summary_path.parent / "gradio_run_meta.json"
        task = ""
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                task = str(meta.get("task_instructions_preview") or "")
            except (OSError, json.JSONDecodeError):
                pass
        run_id = summary_path.parent.name
        if summary_path.parent.name.startswith("gradio_"):
            run_id = summary_path.parent.name
        append_eval_run_summary_row(
            summary=data,
            run_dir=summary_path.parent,
            task_instructions=task,
            run_id=run_id,
            run_source="gradio",
            csv_paths=[out],
            skip_enabled_check=True,
        )
        count += 1
    return count
