"""Write pass/fail eval summaries after Gradio redaction tasks (AGENT_EVAL_DEEPEVAL_TRACING=1)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from run_agent_regression import (
    _ensure_agent_imports,
    discover_artifacts,
    record_tool_calls_from_events,
    write_run_artifacts,
)


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


def resolve_eval_fixture_dir() -> Path | None:
    """Fixture with gold policy lists and optional ``expected_tool_trajectory.json``."""
    raw = (os.environ.get("AGENT_EVAL_FIXTURE") or "").strip()
    if not raw:
        return None
    repo_root, _, _ = _ensure_agent_imports()
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / raw).resolve()
    return path if path.is_dir() else None


def write_session_post_run_eval(
    *,
    session_hash: str,
    tool_events: list[Any],
    task_instructions: str,
    assistant_output: str,
    document_name: str = "",
    workflow_incomplete: bool = False,
) -> Path | None:
    """
    Persist artifacts under ``{session_workspace}/eval/run_<ts>/`` and ``eval_summary.json``.

    Runs when :func:`agent_eval_deepeval_tracing_enabled` is true (see ``arize_monitoring``).
    Policy / tool metrics need ``AGENT_EVAL_FIXTURE``; task completion uses *task_instructions*.
    """
    try:
        from eval.arize_monitoring import agent_eval_deepeval_tracing_enabled
    except ImportError:
        from arize_monitoring import agent_eval_deepeval_tracing_enabled

    if not agent_eval_deepeval_tracing_enabled():
        return None

    _, _, eval_dir = _ensure_agent_imports()
    from session_workspace import session_workspace_dir

    workspace = session_workspace_dir(session_hash)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (workspace / "eval" / f"run_{stamp}").resolve()
    fixture_dir = resolve_eval_fixture_dir()

    tool_args = record_tool_calls_from_events(tool_events)
    discovered = discover_artifacts(workspace)
    run_dir.mkdir(parents=True, exist_ok=True)
    if fixture_dir is not None:
        write_run_artifacts(
            run_dir,
            turn=None,
            discovered=discovered,
            fixture_dir=fixture_dir,
        )
    else:
        _copy_discovered_artifacts(run_dir, discovered)
    tool_path = run_dir / "tool_args.json"
    tool_path.write_text(json.dumps(tool_args, indent=2), encoding="utf-8")

    meta = {
        "session_hash": session_hash,
        "document_name": document_name,
        "workflow_incomplete": workflow_incomplete,
        "fixture": str(fixture_dir) if fixture_dir else None,
        "task_instructions_preview": (task_instructions or "")[:500],
    }
    (run_dir / "gradio_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    metrics = score_post_run_metrics(
        run_dir=run_dir,
        fixture_dir=fixture_dir,
        task_instructions=task_instructions,
        tool_args=tool_args,
        assistant_output=assistant_output,
    )
    active = [m for m in metrics if not m.get("skipped")]
    overall_pass: bool | None
    if not active:
        overall_pass = None
    else:
        overall_pass = all(m.get("pass") is True for m in active)

    summary = {
        "session_hash": session_hash,
        "document_name": document_name,
        "workflow_incomplete": workflow_incomplete,
        "run_dir": str(run_dir),
        "fixture": str(fixture_dir) if fixture_dir else None,
        "overall_pass": overall_pass,
        "metrics": metrics,
    }
    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mirror = (os.environ.get("AGENT_EVAL_MIRROR_RUNS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if mirror:
        dest = eval_dir / "runs" / f"gradio_{session_hash[:8]}_{stamp}"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(run_dir, dest)
            summary["mirrored_to"] = str(dest)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            pass

    return summary_path


def score_post_run_metrics(
    *,
    run_dir: Path,
    fixture_dir: Path | None,
    task_instructions: str,
    tool_args: dict[str, Any],
    assistant_output: str,
) -> list[dict[str, Any]]:
    """Score PolicyStrict, ToolCorrectness, and TaskCompletion (best effort)."""
    results: list[dict[str, Any]] = []
    task = (task_instructions or "").strip() or "Redaction task"

    if fixture_dir is None:
        results.append(
            _skipped("policy_strict", "Set AGENT_EVAL_FIXTURE for gold policy lists.")
        )
    else:
        results.append(
            _score_policy_strict(run_dir=run_dir, fixture_dir=fixture_dir, task=task)
        )

    if fixture_dir is None or not _fixture_has_expected_tools(fixture_dir):
        results.append(
            _skipped(
                "tool_correctness",
                "Set AGENT_EVAL_FIXTURE with expected_tool_trajectory.json (or tool_args.json).",
            )
        )
    else:
        results.append(
            _score_deepeval_metric(
                name="tool_correctness",
                fixture_dir=fixture_dir,
                run_dir=run_dir,
                task=task,
                tool_args=tool_args,
                assistant_output=assistant_output,
                include_policy=False,
                include_tool=True,
                include_task=False,
            )
        )

    results.append(
        _score_deepeval_metric(
            name="task_completion",
            fixture_dir=fixture_dir or Path(),
            run_dir=run_dir,
            task=task,
            tool_args=tool_args,
            assistant_output=assistant_output,
            include_policy=False,
            include_tool=False,
            include_task=True,
        )
    )
    return results


def _load_expected_tool_names(fixture_dir: Path) -> list[str]:
    """Same rules as ``deepeval_metrics.load_expected_tool_names`` (no deepeval import)."""
    fixture_dir = fixture_dir.resolve()
    traj_path = fixture_dir / "expected_tool_trajectory.json"
    if traj_path.is_file():
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        tools = data.get("tools")
        if isinstance(tools, list):
            return [str(t).strip() for t in tools if str(t).strip()]
    tool_args = fixture_dir / "tool_args.json"
    if tool_args.is_file():
        from compare_policy_lists import load_observed_tool_args

        calls = load_observed_tool_args(tool_args).get("tool_calls") or []
        names: list[str] = []
        for call in calls:
            if isinstance(call, dict):
                name = str(call.get("tool") or call.get("name") or "").strip()
                if name:
                    names.append(name)
        return names
    return []


def _fixture_has_expected_tools(fixture_dir: Path) -> bool:
    return bool(_load_expected_tool_names(fixture_dir))


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": None,
        "skipped": True,
        "score": None,
        "reason": reason,
    }


def _metric_row(name: str, metric: Any) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(metric.is_successful()),
        "skipped": False,
        "score": float(getattr(metric, "score", 0.0) or 0.0),
        "reason": str(getattr(metric, "reason", "") or ""),
    }


def _deepeval_import_ok() -> bool:
    try:
        import deepeval  # noqa: F401
    except Exception:
        return False
    return True


def _score_policy_strict(
    *, run_dir: Path, fixture_dir: Path, task: str
) -> dict[str, Any]:
    if not _deepeval_import_ok():
        return _deterministic_policy_only(run_dir=run_dir, fixture_dir=fixture_dir)
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval_metrics import PolicyStrictMetric
    except Exception:
        return _deterministic_policy_only(run_dir=run_dir, fixture_dir=fixture_dir)

    payload = json.dumps(
        {"fixture": str(fixture_dir.resolve()), "run_dir": str(run_dir)}
    )
    metric = PolicyStrictMetric()
    try:
        metric.measure(LLMTestCase(input=task, expected_output=payload))
        return _metric_row("policy_strict", metric)
    except Exception as exc:
        return {
            "name": "policy_strict",
            "pass": False,
            "skipped": False,
            "score": 0.0,
            "reason": str(exc),
        }


def _deterministic_policy_only(*, run_dir: Path, fixture_dir: Path) -> dict[str, Any]:
    from run_agent_regression import run_deterministic_gates

    summary = run_deterministic_gates(
        run_dir,
        fixture_dir,
        require_pass_strict=True,
        compare_extraction=False,
    )
    passed = summary.exit_code == 0 and bool(
        (summary.policy_gate or {}).get("pass_strict")
    )
    reason = (
        "pass_strict is true (policy gate)."
        if passed
        else "; ".join(summary.messages) or "policy gate failed"
    )
    return {
        "name": "policy_strict",
        "pass": passed,
        "skipped": False,
        "score": 1.0 if passed else 0.0,
        "reason": reason,
    }


def _score_deepeval_metric(
    *,
    name: str,
    fixture_dir: Path,
    run_dir: Path,
    task: str,
    tool_args: dict[str, Any],
    assistant_output: str,
    include_policy: bool,
    include_tool: bool,
    include_task: bool,
) -> dict[str, Any]:
    if not _deepeval_import_ok():
        if name == "tool_correctness":
            return _deterministic_tool_match(fixture_dir, tool_args)
        return {
            "name": name,
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": "Install requirements_agent_eval.txt for DeepEval metrics.",
        }
    try:
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval_metrics import (
            build_regression_deepeval_metrics,
            expected_tool_calls,
        )
    except Exception:
        if name == "tool_correctness":
            return _deterministic_tool_match(fixture_dir, tool_args)
        return {
            "name": name,
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": "DeepEval import failed.",
        }

    fixture_for_build = fixture_dir if fixture_dir.is_dir() else run_dir
    try:
        metrics = build_regression_deepeval_metrics(
            fixture_for_build,
            include_policy_strict=include_policy,
            include_tool_correctness=include_tool,
            include_task_completion=include_task,
            task=task,
        )
    except ImportError as exc:
        return {
            "name": name,
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": str(exc),
        }

    if not metrics:
        if name == "tool_correctness":
            return _deterministic_tool_match(fixture_dir, tool_args)
        if name == "task_completion":
            from deepeval_metrics import _deepeval_judge_model

            if _deepeval_judge_model() is None:
                return _skipped(
                    name,
                    "Set USE_OPENAI_MODEL=1 and OPENAI_BASE_URL or AGENT_LLAMA_BASE_URL "
                    "for the DeepEval task-completion judge.",
                )
        return _skipped(name, "No metric configured for this run.")

    metric = metrics[-1]
    observed_tools: list[Any] = []
    for call in tool_args.get("tool_calls") or []:
        if isinstance(call, dict):
            tname = str(call.get("tool") or call.get("name") or "").strip()
            if tname:
                observed_tools.append(ToolCall(name=tname))

    payload = json.dumps(
        {
            "fixture": str(fixture_dir.resolve()) if fixture_dir.is_dir() else "",
            "run_dir": str(run_dir),
        }
    )
    fixture_for_tools = fixture_dir if fixture_dir.is_dir() else run_dir
    expected_tools = expected_tool_calls(fixture_for_tools)
    test_case = LLMTestCase(
        input=task,
        actual_output=(assistant_output or "")[:12000],
        expected_output=payload,
        tools_called=observed_tools or None,
        expected_tools=expected_tools or None,
    )
    try:
        from deepeval.tracing import update_current_trace

        update_current_trace(input=task, output=assistant_output[:4000])
    except Exception:
        pass

    try:
        metric.measure(test_case)
        return _metric_row(name, metric)
    except Exception as exc:
        if name == "tool_correctness":
            fallback = _deterministic_tool_match(fixture_dir, tool_args)
            fallback["reason"] = (
                f"{fallback.get('reason', '')} (DeepEval: {exc})".strip()
            )
            return fallback
        return {
            "name": name,
            "pass": False,
            "skipped": False,
            "score": 0.0,
            "reason": str(exc),
        }


def _deterministic_tool_match(
    fixture_dir: Path, tool_args: dict[str, Any]
) -> dict[str, Any]:
    expected = _load_expected_tool_names(fixture_dir)
    observed: list[str] = []
    for call in tool_args.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool") or call.get("name") or "").strip()
            if name:
                observed.append(name)
    if not expected:
        return _skipped("tool_correctness", "No expected tool list in fixture.")
    missing = [t for t in expected if t not in observed]
    passed = not missing
    return {
        "name": "tool_correctness",
        "pass": passed,
        "skipped": False,
        "score": 1.0 if passed else max(0.0, 1.0 - len(missing) / len(expected)),
        "reason": (
            "All expected tools observed."
            if passed
            else f"Missing tools: {', '.join(missing)}"
        ),
    }
