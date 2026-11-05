"""Post-run gold eval for the main Gradio redaction app (APP_EVAL_ENABLED)."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from app_eval_env import (
    app_eval_auto_gold_enabled,
    app_eval_deepeval_enabled,
    app_eval_enabled,
    app_eval_mirror_runs_enabled,
    app_eval_run_summary_csv_enabled,
    app_gold_match_mode,
    resolve_app_eval_gold_root,
    resolve_fixed_app_eval_gold,
)
from derive_app_eval_instructions import (
    AppEvalRedactionSettings,
    derive_app_eval_instructions,
    synthetic_doc_redact_tool_args,
)
from gold_match import resolve_gold_case_for_task
from run_agent_regression import (
    GateSummary,
    _ensure_agent_imports,
    discover_artifacts,
    run_deterministic_gates,
    write_run_artifacts,
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    repo_root, _, _ = _ensure_agent_imports()
    return repo_root


def resolve_app_eval_gold_dir(
    *,
    pdf_path: Path,
    document_name: str,
    derived_instructions: str,
) -> tuple[Path | None, str]:
    """Return (gold_case_dir, status_message)."""
    repo = _repo_root()
    fixed = resolve_fixed_app_eval_gold(repo)
    if fixed is not None:
        return fixed, f"Using fixed gold case `{fixed.name}`."
    if not app_eval_auto_gold_enabled():
        return None, "APP_EVAL_AUTO_GOLD is off and APP_EVAL_GOLD is unset."
    result = resolve_gold_case_for_task(
        pdf_path=pdf_path,
        document_name=document_name,
        task_instructions=derived_instructions,
        gold_root=resolve_app_eval_gold_root(repo),
        match_mode=app_gold_match_mode(),
    )
    if result.gold_dir is not None:
        return (
            result.gold_dir,
            f"Matched gold `{result.case_id or result.gold_dir.name}` ({result.match_level}).",
        )
    return None, result.reason or "No matching gold case."


def _copy_discovered_artifacts(
    run_dir: Path, discovered: dict[str, Path | None]
) -> None:
    for key, src in discovered.items():
        if src is None or not src.is_file():
            continue
        if key == "review_csv":
            dest = run_dir / "agent_review_file.csv"
        elif key == "ocr_words_csv":
            dest = run_dir / "ocr_results_with_words.csv"
        elif key == "redacted_pdf":
            dest = run_dir / "redacted.pdf"
        else:
            dest = run_dir / src.name
        shutil.copy2(src, dest)


def run_app_deterministic_gates(
    run_dir: Path,
    gold_case_dir: Path,
    *,
    text_extraction_only: bool,
) -> GateSummary:
    """Full gates when review exists; OCR-only path for extract-text-only."""
    review = run_dir / "agent_review_file.csv"
    if review.is_file() and not text_extraction_only:
        return run_deterministic_gates(
            run_dir,
            gold_case_dir,
            require_pass_strict=False,
            compare_extraction=True,
        )

    from compare_ocr_words_csvs import compare_ocr_gold_for_run

    summary = GateSummary()
    summary.messages.append(
        "Extract-only or missing review CSV: skipped policy gate and geometry."
    )
    ocr_report = compare_ocr_gold_for_run(gold_case_dir=gold_case_dir, run_dir=run_dir)
    summary.ocr_gold = ocr_report.to_dict()
    (run_dir / "ocr_gold.json").write_text(
        json.dumps(summary.ocr_gold, indent=2, default=str), encoding="utf-8"
    )
    return summary


def score_app_post_run_metrics(
    *,
    run_dir: Path,
    gold_case_dir: Path | None,
    derived_instructions: str,
    tool_args: dict[str, Any],
    gate_summary: GateSummary | None,
) -> list[dict[str, Any]]:
    from post_run_eval import (
        _metrics_from_gate_summary,
        _score_deepeval_metric,
        _score_policy_strict,
        _skipped,
    )

    task = derived_instructions.strip() or "App redaction task"
    results: list[dict[str, Any]] = []

    if gold_case_dir is None:
        results.append(_skipped("policy_strict", "No gold case matched."))
    else:
        results.append(
            _score_policy_strict(
                run_dir=run_dir,
                gold_case_dir=gold_case_dir,
                task=task,
                gate_summary=gate_summary,
            )
        )

    results.append(
        _skipped(
            "tool_correctness",
            "Main app eval does not score tool correctness.",
        )
    )

    if app_eval_deepeval_enabled() and gold_case_dir is not None:
        results.append(
            _score_deepeval_metric(
                name="task_completion",
                gold_case_dir=gold_case_dir,
                run_dir=run_dir,
                task=task,
                tool_args=tool_args,
                assistant_output="",
                include_policy=False,
                include_tool=False,
                include_task=True,
            )
        )
    else:
        reason = (
            "Enable AGENT_EVAL_DEEPEVAL_TRACING=1 for task completion."
            if gold_case_dir
            else "No gold case."
        )
        results.append(_skipped("task_completion", reason))

    results.extend(
        _metrics_from_gate_summary(gate_summary, gold_case_dir=gold_case_dir)
    )
    return results


def write_app_post_run_eval(
    *,
    pdf_path: Path,
    document_name: str,
    settings: AppEvalRedactionSettings,
    search_root: Path | None = None,
) -> tuple[Path | None, str]:
    """
    Stage artifacts, run gates, write ``eval_summary.json``.

    Returns ``(summary_path, human_status_line)``.
    """
    if not app_eval_enabled():
        return None, ""

    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        return None, f"App eval skipped: PDF not found at {pdf_path}."

    derived = derive_app_eval_instructions(settings)
    gold_case_dir, match_msg = resolve_app_eval_gold_dir(
        pdf_path=pdf_path,
        document_name=document_name or pdf_path.name,
        derived_instructions=derived,
    )

    _, _, eval_dir = _ensure_agent_imports()
    root = (search_root or pdf_path.parent).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (root / "eval" / f"run_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_artifacts(root)
    tool_args = synthetic_doc_redact_tool_args(settings)
    (run_dir / "tool_args.json").write_text(
        json.dumps(tool_args, indent=2), encoding="utf-8"
    )
    (run_dir / "app_eval_settings.json").write_text(
        json.dumps(settings.to_settings_dict(), indent=2), encoding="utf-8"
    )

    if gold_case_dir is not None:
        write_run_artifacts(
            run_dir,
            turn=None,
            discovered=discovered,
            gold_case_dir=gold_case_dir,
        )
    else:
        _copy_discovered_artifacts(run_dir, discovered)

    meta = {
        "document_name": document_name or pdf_path.name,
        "pdf_path": str(pdf_path),
        "derived_instructions_preview": derived[:500],
        "derived_instructions_version": settings.to_settings_dict().get(
            "derived_instructions_version"
        ),
        "gold_case": str(gold_case_dir) if gold_case_dir else None,
        "text_extraction_only": settings.text_extraction_only,
        "ocr_method": settings.text_extraction_method or None,
        "pii_method": settings.pii_identification_method or None,
        "encourage_vlm_faces": None,
        "encourage_vlm_signatures": None,
        "eval_profile": "app_gradio",
    }
    (run_dir / "app_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    gate_summary: GateSummary | None = None
    if gold_case_dir is not None:
        gate_summary = run_app_deterministic_gates(
            run_dir,
            gold_case_dir,
            text_extraction_only=settings.text_extraction_only,
        )

    metrics = score_app_post_run_metrics(
        run_dir=run_dir,
        gold_case_dir=gold_case_dir,
        derived_instructions=derived,
        tool_args=tool_args,
        gate_summary=gate_summary,
    )
    active = [m for m in metrics if not m.get("skipped")]
    overall_pass: bool | None
    if not active:
        overall_pass = None
    elif gold_case_dir is None:
        overall_pass = None
    else:
        overall_pass = all(m.get("pass") is True for m in active)

    summary = {
        "document_name": document_name or pdf_path.name,
        "run_dir": str(run_dir),
        "gold_case": str(gold_case_dir) if gold_case_dir else None,
        "overall_pass": overall_pass,
        "metrics": metrics,
        "eval_turn_kind": "app_redact",
        "follow_up_steer_count": 0,
        "workflow_incomplete": False,
        "ocr_method": settings.text_extraction_method or None,
        "pii_method": settings.pii_identification_method or None,
        "derived_instructions": derived,
    }
    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mirror = app_eval_mirror_runs_enabled()
    if mirror and gold_case_dir is not None:
        base = pdf_path.stem[:40].replace(" ", "_")
        dest = eval_dir / "runs" / f"app_{base}_{stamp}"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(run_dir, dest)
            summary["mirrored_to"] = str(dest)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            run_dir = dest
        except OSError:
            pass

    status = match_msg
    if gold_case_dir is not None and gate_summary and gate_summary.policy_gate:
        ps = gate_summary.policy_gate.get("pass_strict")
        status += f" pass_strict={ps}."
    if gold_case_dir is not None and gate_summary and gate_summary.gold_geometry:
        status += f" geometry_f1={gate_summary.gold_geometry.get('f1')}."
    status += f" Summary: {summary_path}."

    if app_eval_run_summary_csv_enabled() and gold_case_dir is not None:
        try:
            from eval_run_summary_csv import append_eval_run_summary_row

            run_id = run_dir.name
            if summary.get("mirrored_to"):
                run_id = Path(summary["mirrored_to"]).name
            append_eval_run_summary_row(
                summary=summary,
                run_dir=run_dir,
                task_instructions=derived,
                run_id=run_id,
                run_source="app_gradio",
                skip_enabled_check=True,
            )
        except Exception:
            logger.debug("App eval CSV append failed", exc_info=True)

    return summary_path, status


def ensure_eval_import_path() -> None:
    eval_dir = Path(__file__).resolve().parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
