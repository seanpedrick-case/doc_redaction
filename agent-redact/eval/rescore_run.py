#!/usr/bin/env python3
"""Re-score an existing run folder against a gold case (no agent/redaction).

Example::

    python agent-redact/eval/rescore_run.py \\
      --run-dir agent-redact/eval/runs/bench_.../agent/example_emails_to_professor_agentic \\
      --gold agent-redact/eval/gold/example_emails_to_professor_agentic

Writes ``eval_summary.json`` (+ gate JSON), appends ``eval_run_summary.csv``
(``run_source=rescore``), and optionally runs DeepEval metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from run_agent_regression import (  # noqa: E402
    GateSummary,
    _ensure_agent_imports,
    run_deterministic_gates,
)

_ARTIFACT_NAMES = (
    "agent_review_file.csv",
    "ocr_results_with_words.csv",
    "tool_args.json",
    "redacted.pdf",
    "manifest.json",
    "turn_meta.json",
    "gradio_run_meta.json",
    "app_run_meta.json",
    "eval_summary.json",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def resolve_gold_from_run_meta(run_dir: Path) -> Path | None:
    """Prefer ``gold_case`` from eval_summary / gradio / app meta when the path exists."""
    for name in ("eval_summary.json", "gradio_run_meta.json", "app_run_meta.json"):
        data = _read_json(run_dir / name)
        if not data:
            continue
        raw = data.get("gold_case") or data.get("fixture")
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_dir():
            return path.resolve()
    return None


def _find_review_csv(run_dir: Path) -> Path | None:
    preferred = run_dir / "agent_review_file.csv"
    if preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob("*_review_file.csv"))
    return matches[0] if matches else None


def _find_ocr_csv(run_dir: Path) -> Path | None:
    preferred = run_dir / "ocr_results_with_words.csv"
    if preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob("*ocr_results_with_words*.csv"))
    return matches[0] if matches else None


def _find_redacted_pdf(run_dir: Path) -> Path | None:
    preferred = run_dir / "redacted.pdf"
    if preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob("*_redacted.pdf"))
    return matches[0] if matches else None


def collect_run_artifacts(run_dir: Path) -> dict[str, Path | None]:
    """Map canonical artifact keys to paths under *run_dir* (non-recursive)."""
    run_dir = run_dir.resolve()
    tool = run_dir / "tool_args.json"
    return {
        "review_csv": _find_review_csv(run_dir),
        "ocr_words_csv": _find_ocr_csv(run_dir),
        "redacted_pdf": _find_redacted_pdf(run_dir),
        "tool_args": tool if tool.is_file() else None,
    }


def prepare_out_dir(
    *,
    run_dir: Path,
    out_dir: Path,
    in_place: bool,
) -> Path:
    """Copy observed artifacts into *out_dir*, or return *run_dir* when in-place."""
    run_dir = run_dir.resolve()
    if in_place:
        return run_dir

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = collect_run_artifacts(run_dir)
    mapping = {
        "review_csv": "agent_review_file.csv",
        "ocr_words_csv": "ocr_results_with_words.csv",
        "redacted_pdf": "redacted.pdf",
        "tool_args": "tool_args.json",
    }
    for key, dest_name in mapping.items():
        src = artifacts.get(key)
        if src is None or not src.is_file():
            continue
        dest = out_dir / dest_name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    for name in _ARTIFACT_NAMES:
        if name in mapping.values():
            continue
        src = run_dir / name
        if not src.is_file():
            continue
        dest = out_dir / name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    return out_dir


def load_tool_args(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "tool_args.json")
    if data is None:
        return {"tool_calls": []}
    if "tool_calls" not in data:
        return {"tool_calls": []}
    return data


def load_task_instructions(gold_case_dir: Path, run_dir: Path) -> str:
    for name in ("instructions.txt", "instruction.txt"):
        text = _read_text(gold_case_dir / name)
        if text:
            return text
    for meta_name in ("gradio_run_meta.json", "app_run_meta.json", "eval_summary.json"):
        data = _read_json(run_dir / meta_name)
        if not data:
            continue
        preview = data.get("task_instructions_preview") or data.get("task_instructions")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()
    return ""


def load_assistant_output(run_dir: Path) -> str:
    for name in ("assistant_output.txt", "task_completion_actual_output.txt"):
        text = _read_text(run_dir / name)
        if text:
            return text
    for meta_name in ("turn_meta.json", "gradio_run_meta.json", "eval_summary.json"):
        data = _read_json(run_dir / meta_name)
        if not data:
            continue
        for key in ("assistant_output", "actual_output", "final_message"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def load_stack_meta(run_dir: Path) -> dict[str, Any]:
    """OCR/PII/orchestrator/model from prior meta or manifest."""
    out: dict[str, Any] = {
        "orchestrator": "",
        "model_name": "",
        "ocr_method": "",
        "pii_method": "",
        "document_name": "",
        "session_hash": "",
        "encourage_vlm_faces": False,
        "encourage_vlm_signatures": False,
    }
    for meta_name in (
        "gradio_run_meta.json",
        "app_run_meta.json",
        "eval_summary.json",
        "turn_meta.json",
        "manifest.json",
    ):
        data = _read_json(run_dir / meta_name)
        if not data:
            continue
        for key in (
            "orchestrator",
            "model_name",
            "ocr_method",
            "pii_method",
            "document_name",
            "session_hash",
        ):
            val = data.get(key) or data.get(f"run_{key}")
            if val and not out[key]:
                out[key] = str(val).strip()
        if "encourage_vlm_faces" in data:
            out["encourage_vlm_faces"] = bool(data["encourage_vlm_faces"])
        if "encourage_vlm_signatures" in data:
            out["encourage_vlm_signatures"] = bool(data["encourage_vlm_signatures"])
        if data.get("model") and not out["model_name"]:
            out["model_name"] = str(data["model"]).strip()
    return out


def _metrics_without_deepeval(
    *,
    run_dir: Path,
    gold_case_dir: Path,
    gate_summary: GateSummary,
) -> list[dict[str, Any]]:
    from post_run_eval import _metrics_from_gate_summary, _score_policy_strict

    metrics: list[dict[str, Any]] = [
        _score_policy_strict(
            run_dir=run_dir,
            gold_case_dir=gold_case_dir,
            task="rescore",
            gate_summary=gate_summary,
        ),
        {
            "name": "tool_correctness",
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": "Skipped (--no-deepeval).",
        },
        {
            "name": "task_completion",
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": "Skipped (--no-deepeval).",
        },
    ]
    metrics.extend(
        _metrics_from_gate_summary(
            gate_summary,
            gold_case_dir=gold_case_dir,
            session_hash="",
        )
    )
    return metrics


def _overall_pass_from_metrics(metrics: list[dict[str, Any]]) -> bool | None:
    active = [m for m in metrics if not m.get("skipped")]
    if not active:
        return None
    return all(m.get("pass") is True for m in active)


def rescore_run(
    *,
    run_dir: Path,
    gold_case_dir: Path,
    out_dir: Path | None = None,
    in_place: bool = False,
    deepeval: bool = True,
    append_csv: bool = True,
    fail_on_qa: bool = False,
) -> tuple[dict[str, Any], int]:
    """
    Score *run_dir* artifacts against *gold_case_dir*.

    Returns ``(eval_summary, exit_code)``. Exit is 0 when scoring completes;
    use *fail_on_qa* to exit 1 when ``overall_pass`` is false.
    """
    _, _, eval_dir = _ensure_agent_imports()
    run_dir = run_dir.resolve()
    gold_case_dir = gold_case_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if not gold_case_dir.is_dir():
        raise FileNotFoundError(f"gold case not found: {gold_case_dir}")

    artifacts = collect_run_artifacts(run_dir)
    if artifacts["review_csv"] is None:
        raise FileNotFoundError(f"No review CSV (*_review_file.csv) under {run_dir}")
    if artifacts["ocr_words_csv"] is None:
        raise FileNotFoundError(
            f"No OCR words CSV (*ocr_results_with_words*) under {run_dir}"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if in_place:
        target = prepare_out_dir(run_dir=run_dir, out_dir=run_dir, in_place=True)
    else:
        default_out = eval_dir / "runs" / f"rescore_{gold_case_dir.name}_{stamp}"
        target = prepare_out_dir(
            run_dir=run_dir,
            out_dir=(out_dir or default_out),
            in_place=False,
        )

    # Ensure canonical names for gates (in-place may use alternate filenames).
    if not in_place or not (target / "agent_review_file.csv").is_file():
        review = artifacts["review_csv"]
        if review is not None:
            dest = target / "agent_review_file.csv"
            if review.resolve() != dest.resolve():
                shutil.copy2(review, dest)
    if not in_place or not (target / "ocr_results_with_words.csv").is_file():
        ocr = artifacts["ocr_words_csv"]
        if ocr is not None:
            dest = target / "ocr_results_with_words.csv"
            if ocr.resolve() != dest.resolve():
                shutil.copy2(ocr, dest)

    gate_summary = run_deterministic_gates(
        target,
        gold_case_dir,
        require_pass_strict=False,
        compare_extraction=True,
    )

    task_instructions = load_task_instructions(gold_case_dir, run_dir)
    tool_args = load_tool_args(target)
    if not tool_args.get("tool_calls") and artifacts["tool_args"]:
        tool_args = load_tool_args(run_dir)
        if tool_args.get("tool_calls"):
            (target / "tool_args.json").write_text(
                json.dumps(tool_args, indent=2), encoding="utf-8"
            )
    assistant_output = load_assistant_output(run_dir)
    stack = load_stack_meta(run_dir)

    if deepeval:
        from post_run_eval import score_post_run_metrics

        metrics = score_post_run_metrics(
            run_dir=target,
            gold_case_dir=gold_case_dir,
            session_hash=str(stack.get("session_hash") or ""),
            task_instructions=task_instructions,
            tool_args=tool_args,
            assistant_output=assistant_output,
            gate_summary=gate_summary,
        )
    else:
        metrics = _metrics_without_deepeval(
            run_dir=target,
            gold_case_dir=gold_case_dir,
            gate_summary=gate_summary,
        )

    overall_pass = _overall_pass_from_metrics(metrics)
    summary: dict[str, Any] = {
        "session_hash": stack.get("session_hash") or f"rescore_{stamp}",
        "document_name": stack.get("document_name") or gold_case_dir.name,
        "workflow_incomplete": False,
        "run_dir": str(target),
        "source_run_dir": str(run_dir),
        "gold_case": str(gold_case_dir),
        "overall_pass": overall_pass,
        "metrics": metrics,
        "follow_up_steer_count": 0,
        "eval_turn_kind": "rescore",
        "run_source": "rescore",
        "orchestrator": stack.get("orchestrator") or None,
        "model_name": stack.get("model_name") or None,
        "ocr_method": stack.get("ocr_method") or None,
        "pii_method": stack.get("pii_method") or None,
        "encourage_vlm_faces": bool(stack.get("encourage_vlm_faces")),
        "encourage_vlm_signatures": bool(stack.get("encourage_vlm_signatures")),
        "gate_messages": list(gate_summary.messages),
    }
    summary_path = target / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["eval_summary"] = str(summary_path)

    if append_csv:
        from eval_run_summary_csv import append_eval_run_summary_row

        append_eval_run_summary_row(
            summary=summary,
            run_dir=target,
            task_instructions=task_instructions,
            run_id=target.name,
            run_source="rescore",
            skip_enabled_check=True,
        )

    exit_code = 0
    if fail_on_qa and overall_pass is False:
        exit_code = 1
    return summary, exit_code


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Prior run folder with review CSV + OCR words CSV.",
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold case directory (optional if gold_case in run meta still exists).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write rescored artifacts here (default: eval/runs/rescore_<case>_<ts>/).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write eval_summary / gate JSON into --run-dir (do not copy).",
    )
    p.add_argument(
        "--no-deepeval",
        action="store_true",
        help="Skip DeepEval tool/task metrics (deterministic gates still run).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not append eval/outputs/eval_run_summary.csv.",
    )
    p.add_argument(
        "--fail-on-qa",
        action="store_true",
        help="Exit 1 when overall_pass is false (default: exit 0 after scoring).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    gold = args.gold.resolve() if args.gold else resolve_gold_from_run_meta(run_dir)
    if gold is None:
        print(
            "Provide --gold or ensure run meta contains a resolvable gold_case path.",
            file=sys.stderr,
        )
        return 2
    if args.in_place and args.out_dir is not None:
        print("Use either --in-place or --out-dir, not both.", file=sys.stderr)
        return 2

    try:
        summary, code = rescore_run(
            run_dir=run_dir,
            gold_case_dir=gold,
            out_dir=args.out_dir,
            in_place=bool(args.in_place),
            deepeval=not bool(args.no_deepeval),
            append_csv=not bool(args.no_csv),
            fail_on_qa=bool(args.fail_on_qa),
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
