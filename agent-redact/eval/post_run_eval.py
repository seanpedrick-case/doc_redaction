"""Write pass/fail eval summaries after Gradio redaction tasks (AGENT_EVAL_DEEPEVAL_TRACING=1)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from run_agent_regression import (
    GateSummary,
    _ensure_agent_imports,
    _env_float,
    _env_truthy,
    discover_artifacts,
    record_tool_calls_from_events,
    run_deterministic_gates,
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


def resolve_eval_gold_dir(session_hash: str | None = None) -> Path | None:
    """Gold case directory for post-run eval (session auto-match or env)."""
    repo_root, _, _ = _ensure_agent_imports()
    from agent_eval_env import agent_eval_auto_gold_enabled, resolve_env_gold_case_dir

    if agent_eval_auto_gold_enabled():
        if not session_hash:
            return None
        try:
            from eval_task_context import get_active_task_for_eval
        except ImportError:
            from shared.eval_task_context import get_active_task_for_eval  # type: ignore

        task = get_active_task_for_eval(session_hash)
        if task is None or not task.eval_gold_path:
            return None
        path = Path(task.eval_gold_path).resolve()
        return path if path.is_dir() else None

    return resolve_env_gold_case_dir(repo_root)


resolve_eval_fixture_dir = resolve_eval_gold_dir  # deprecated alias


def _merge_run_manifest_stack(
    run_dir: Path,
    *,
    ocr_method: str = "",
    pii_method: str = "",
    orchestrator: str = "",
    model_name: str = "",
    encourage_vlm_faces: bool | None = None,
    encourage_vlm_signatures: bool | None = None,
) -> None:
    """Overlay actual run stack onto ``manifest.json`` (gold copy may already exist)."""
    path = run_dir / "manifest.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            data = {}
    if ocr_method.strip():
        data["run_ocr_method"] = ocr_method.strip()
    if pii_method.strip():
        data["run_pii_method"] = pii_method.strip()
    if orchestrator.strip():
        data["run_orchestrator"] = orchestrator.strip()
    if model_name.strip():
        data["run_model_name"] = model_name.strip()
    if encourage_vlm_faces is not None:
        data["encourage_vlm_faces"] = bool(encourage_vlm_faces)
    if encourage_vlm_signatures is not None:
        data["encourage_vlm_signatures"] = bool(encourage_vlm_signatures)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_session_post_run_eval(
    *,
    session_hash: str,
    tool_events: list[Any],
    task_instructions: str,
    assistant_output: str,
    document_name: str = "",
    workflow_incomplete: bool = False,
    follow_up_steer_count: int = 0,
    eval_turn_kind: str = "initial_task",
    cumulative_tool_calls: list[dict[str, Any]] | None = None,
    ocr_method: str = "",
    pii_method: str = "",
    orchestrator: str = "",
    model_name: str = "",
    encourage_vlm_faces: bool = False,
    encourage_vlm_signatures: bool = False,
    workspace_root: Path | str | None = None,
    gold_case_dir: Path | str | None = None,
    force: bool = False,
) -> Path | None:
    """
    Persist artifacts under ``{session_workspace}/eval/run_<ts>/`` and ``eval_summary.json``.

    Runs when :func:`agent_eval_deepeval_tracing_enabled` is true (see ``arize_monitoring``).
    Policy / tool metrics need a gold case (``AGENT_EVAL_GOLD`` or auto-match); task completion uses *task_instructions*.

    Gradio emits a new ``run_<ts>/`` folder after each completed turn (initial **Start
    redaction task** or follow-up/steer). ``follow_up_steer_count`` counts user
    follow-up and in-run steer messages since the last task start (see
    ``eval_task_context``).
    """
    try:
        from eval.arize_monitoring import agent_eval_deepeval_tracing_enabled
    except ImportError:
        from arize_monitoring import agent_eval_deepeval_tracing_enabled

    if not force and not agent_eval_deepeval_tracing_enabled():
        return None

    _, _, eval_dir = _ensure_agent_imports()
    from session_workspace import session_workspace_dir

    if workspace_root is not None:
        workspace = Path(workspace_root).resolve()
    else:
        workspace = session_workspace_dir(session_hash)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (workspace / "eval" / f"run_{stamp}").resolve()
    if gold_case_dir is not None:
        gold_path = Path(gold_case_dir).resolve()
        gold_case_dir = gold_path if gold_path.is_dir() else None
    else:
        gold_case_dir = resolve_eval_gold_dir(session_hash)

    if cumulative_tool_calls is not None:
        tool_args = {"tool_calls": list(cumulative_tool_calls)}
    else:
        tool_args = record_tool_calls_from_events(tool_events)
    discovered = discover_artifacts(workspace)
    run_dir.mkdir(parents=True, exist_ok=True)
    if gold_case_dir is not None:
        write_run_artifacts(
            run_dir,
            turn=None,
            discovered=discovered,
            gold_case_dir=gold_case_dir,
        )
    else:
        _copy_discovered_artifacts(run_dir, discovered)
    tool_path = run_dir / "tool_args.json"
    tool_path.write_text(json.dumps(tool_args, indent=2), encoding="utf-8")

    stack_orchestrator = (
        orchestrator or os.environ.get("AGENT_ORCHESTRATOR") or ""
    ).strip()
    stack_ocr = (ocr_method or "").strip()
    stack_pii = (pii_method or "").strip()
    stack_model = (model_name or "").strip()

    meta = {
        "session_hash": session_hash,
        "document_name": document_name,
        "workflow_incomplete": workflow_incomplete,
        "gold_case": str(gold_case_dir) if gold_case_dir else None,
        "task_instructions_preview": (task_instructions or "")[:500],
        "follow_up_steer_count": max(0, int(follow_up_steer_count)),
        "eval_turn_kind": (eval_turn_kind or "initial_task").strip() or "initial_task",
        "orchestrator": stack_orchestrator or None,
        "model_name": stack_model or None,
        "ocr_method": stack_ocr or None,
        "pii_method": stack_pii or None,
        "encourage_vlm_faces": bool(encourage_vlm_faces),
        "encourage_vlm_signatures": bool(encourage_vlm_signatures),
    }
    (run_dir / "gradio_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    _merge_run_manifest_stack(
        run_dir,
        ocr_method=stack_ocr,
        pii_method=stack_pii,
        orchestrator=stack_orchestrator,
        model_name=stack_model,
        encourage_vlm_faces=encourage_vlm_faces,
        encourage_vlm_signatures=encourage_vlm_signatures,
    )

    gate_summary: GateSummary | None = None
    if gold_case_dir is not None:
        gate_summary = run_deterministic_gates(
            run_dir,
            gold_case_dir,
            require_pass_strict=False,
            compare_extraction=True,
        )

    metrics = score_post_run_metrics(
        run_dir=run_dir,
        gold_case_dir=gold_case_dir,
        session_hash=session_hash,
        task_instructions=task_instructions,
        tool_args=tool_args,
        assistant_output=assistant_output,
        gate_summary=gate_summary,
    )
    active = [m for m in metrics if not m.get("skipped")]
    overall_pass: bool | None
    if not active:
        overall_pass = None
    else:
        overall_pass = all(m.get("pass") is True for m in active)
    if overall_pass is True and _overall_pass_includes_gold(gate_summary):
        gold_ok = _deterministic_gold_pass(gate_summary)
        if gold_ok is False:
            overall_pass = False

    summary = {
        "session_hash": session_hash,
        "document_name": document_name,
        "workflow_incomplete": workflow_incomplete,
        "run_dir": str(run_dir),
        "gold_case": str(gold_case_dir) if gold_case_dir else None,
        "overall_pass": overall_pass,
        "metrics": metrics,
        "follow_up_steer_count": max(0, int(follow_up_steer_count)),
        "eval_turn_kind": (eval_turn_kind or "initial_task").strip() or "initial_task",
        "orchestrator": stack_orchestrator or None,
        "model_name": stack_model or None,
        "ocr_method": stack_ocr or None,
        "pii_method": stack_pii or None,
        "encourage_vlm_faces": bool(encourage_vlm_faces),
        "encourage_vlm_signatures": bool(encourage_vlm_signatures),
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

    try:
        from eval_run_summary_csv import append_eval_run_summary_row
    except ImportError:
        append_eval_run_summary_row = None  # type: ignore[misc, assignment]

    if append_eval_run_summary_row is not None:
        session_csv = workspace / "eval" / "eval_run_summary.csv"
        mirrored = summary.get("mirrored_to")
        run_id = (
            Path(mirrored).name if mirrored else f"gradio_{session_hash[:8]}_{stamp}"
        )
        append_eval_run_summary_row(
            summary=summary,
            run_dir=run_dir,
            task_instructions=task_instructions,
            run_id=run_id,
            run_source="agent_gradio",
            extra_csv_paths=[session_csv],
        )

    return summary_path


def _overall_pass_includes_gold(gate_summary: GateSummary | None) -> bool:
    if not _env_truthy("AGENT_EVAL_OVERALL_INCLUDE_GOLD"):
        return False
    return gate_summary is not None


def _deterministic_gold_pass(gate_summary: GateSummary | None) -> bool | None:
    """When AGENT_EVAL_OVERALL_INCLUDE_GOLD=1, require configured min F1s if not skipped."""
    if gate_summary is None:
        return None
    min_gold = _env_float("AGENT_EVAL_MIN_GOLD_F1")
    min_ocr = _env_float("AGENT_EVAL_MIN_OCR_F1")
    checks: list[bool] = []
    geo = gate_summary.gold_geometry
    if min_gold is not None and geo is not None:
        checks.append(float(geo.get("f1") or 0.0) >= min_gold)
    ocr = gate_summary.ocr_gold
    if min_ocr is not None and ocr is not None and not ocr.get("skipped"):
        checks.append(float(ocr.get("f1") or 0.0) >= min_ocr)
    if not checks:
        return None
    return all(checks)


def _read_pass_strict_from_policy_gate(run_dir: Path) -> bool | None:
    gate_path = run_dir / "policy_gate.json"
    if not gate_path.is_file():
        return None
    try:
        data = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    val = data.get("pass_strict")
    if val is None:
        return None
    return bool(val)


def _metrics_from_gate_summary(
    gate_summary: GateSummary | None,
    *,
    gold_case_dir: Path | None = None,
    session_hash: str = "",
) -> list[dict[str, Any]]:
    if gate_summary is None:
        if gold_case_dir is None:
            reason = _gold_eval_skip_reason(session_hash)
            return [
                _skipped("gold_geometry", reason),
                _skipped("ocr_gold", reason),
                _skipped("policy_extraction", reason),
            ]
        return []
    out: list[dict[str, Any]] = []
    min_gold = _env_float("AGENT_EVAL_MIN_GOLD_F1")
    min_ocr = _env_float("AGENT_EVAL_MIN_OCR_F1")

    if gate_summary.gold_geometry is not None:
        geo = gate_summary.gold_geometry
        f1 = float(geo.get("f1") or 0.0)
        passed: bool | None
        if min_gold is not None:
            passed = f1 >= min_gold
        else:
            passed = None
        out.append(
            {
                "name": "gold_geometry",
                "pass": passed,
                "skipped": False,
                "score": f1,
                "reason": f"precision={geo.get('precision')} recall={geo.get('recall')}",
            }
        )
    elif gold_case_dir is not None:
        ideal = gold_case_dir / "ideal_review_file.csv"
        if ideal.is_file():
            out.append(
                _skipped(
                    "gold_geometry",
                    "Agent review CSV missing for geometry compare.",
                )
            )
        else:
            out.append(
                _skipped("gold_geometry", "No ideal_review_file.csv in gold case.")
            )

    ocr = gate_summary.ocr_gold
    if ocr is not None:
        if ocr.get("skipped"):
            out.append(
                {
                    "name": "ocr_gold",
                    "pass": None,
                    "skipped": True,
                    "score": None,
                    "reason": str(
                        ocr.get("skip_reason") or "OCR gold compare skipped."
                    ),
                }
            )
        else:
            f1 = float(ocr.get("f1") or 0.0)
            passed = f1 >= min_ocr if min_ocr is not None else None
            out.append(
                {
                    "name": "ocr_gold",
                    "pass": passed,
                    "skipped": False,
                    "score": f1,
                    "reason": f"precision={ocr.get('precision')} recall={ocr.get('recall')}",
                }
            )

    ext = gate_summary.policy_extraction
    if ext is not None:
        mr = ext.get("must_redact") or {}
        f1 = float(mr.get("f1") or 0.0)
        out.append(
            {
                "name": "policy_extraction",
                "pass": bool(ext.get("guardrail", {}).get("pass", True)),
                "skipped": False,
                "score": f1,
                "reason": "must_redact phrase F1 (see policy_extraction.json).",
            }
        )
    return out


def score_post_run_metrics(
    *,
    run_dir: Path,
    gold_case_dir: Path | None,
    task_instructions: str,
    tool_args: dict[str, Any],
    assistant_output: str,
    gate_summary: GateSummary | None = None,
    session_hash: str = "",
) -> list[dict[str, Any]]:
    """Score PolicyStrict, ToolCorrectness, and TaskCompletion (best effort)."""
    results: list[dict[str, Any]] = []
    task = (task_instructions or "").strip() or "Redaction task"

    if gold_case_dir is None:
        skip_reason = _gold_eval_skip_reason(session_hash)
        results.append(_skipped("policy_strict", skip_reason))
    else:
        results.append(
            _score_policy_strict(
                run_dir=run_dir,
                gold_case_dir=gold_case_dir,
                task=task,
                gate_summary=gate_summary,
            )
        )

    if gold_case_dir is None or not _gold_case_has_expected_tools(gold_case_dir):
        skip_reason = _gold_eval_skip_reason(session_hash)
        results.append(
            _skipped(
                "tool_correctness",
                skip_reason
                or "Set AGENT_EVAL_GOLD or enable auto-gold with a matching task.",
            )
        )
    else:
        results.append(
            _score_deepeval_metric(
                name="tool_correctness",
                gold_case_dir=gold_case_dir,
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
            gold_case_dir=gold_case_dir or Path(),
            run_dir=run_dir,
            task=task,
            tool_args=tool_args,
            assistant_output=assistant_output,
            include_policy=False,
            include_tool=False,
            include_task=True,
        )
    )
    results.extend(
        _metrics_from_gate_summary(
            gate_summary,
            gold_case_dir=gold_case_dir,
            session_hash=session_hash,
        )
    )
    return results


def _gold_eval_skip_reason(session_hash: str) -> str:
    from agent_eval_env import agent_eval_auto_gold_enabled

    if agent_eval_auto_gold_enabled():
        if not session_hash:
            return "Auto-gold enabled but session_hash missing."
        try:
            from eval_task_context import get_active_task_for_eval
        except ImportError:
            return "Auto-gold: could not load eval_task_context."
        task = get_active_task_for_eval(session_hash)
        if task is None:
            return "No active redaction task for gold eval."
        return task.eval_gold_match_reason or "No matching gold case for this task."
    return "Set AGENT_EVAL_GOLD for gold policy lists (or enable AGENT_EVAL_AUTO_GOLD)."


def _load_expected_tool_names(gold_case_dir: Path) -> list[str]:
    """Same rules as ``deepeval_metrics.load_expected_tool_names`` (no deepeval import)."""
    gold_case_dir = gold_case_dir.resolve()
    traj_path = gold_case_dir / "expected_tool_trajectory.json"
    if traj_path.is_file():
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        tools = data.get("tools")
        if isinstance(tools, list):
            return [str(t).strip() for t in tools if str(t).strip()]
    tool_args = gold_case_dir / "tool_args.json"
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


def _gold_case_has_expected_tools(gold_case_dir: Path) -> bool:
    return bool(_load_expected_tool_names(gold_case_dir))


_fixture_has_expected_tools = _gold_case_has_expected_tools  # deprecated


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
    *,
    run_dir: Path,
    gold_case_dir: Path,
    task: str,
    gate_summary: GateSummary | None = None,
) -> dict[str, Any]:
    precomputed = None
    if gate_summary and gate_summary.policy_gate is not None:
        precomputed = bool(gate_summary.policy_gate.get("pass_strict"))
    else:
        from_gate = _read_pass_strict_from_policy_gate(run_dir)
        if from_gate is not None:
            precomputed = from_gate

    if precomputed is not None:
        return {
            "name": "policy_strict",
            "pass": precomputed,
            "skipped": False,
            "score": 1.0 if precomputed else 0.0,
            "reason": (
                "pass_strict is true (policy gate)."
                if precomputed
                else "pass_strict is false (policy gate)."
            ),
        }

    if not _deepeval_import_ok():
        return _deterministic_policy_only(run_dir=run_dir, gold_case_dir=gold_case_dir)
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval_metrics import PolicyStrictMetric
    except Exception:
        return _deterministic_policy_only(run_dir=run_dir, gold_case_dir=gold_case_dir)

    payload = json.dumps(
        {
            "gold_case": str(gold_case_dir.resolve()),
            "fixture": str(gold_case_dir.resolve()),
            "run_dir": str(run_dir),
        }
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


def _deterministic_policy_only(*, run_dir: Path, gold_case_dir: Path) -> dict[str, Any]:
    from_gate = _read_pass_strict_from_policy_gate(run_dir)
    if from_gate is None:
        summary = run_deterministic_gates(
            run_dir,
            gold_case_dir,
            require_pass_strict=True,
            compare_extraction=True,
        )
        from_gate = bool((summary.policy_gate or {}).get("pass_strict"))
    passed = bool(from_gate)
    reason = (
        "pass_strict is true (policy gate)."
        if passed
        else "pass_strict is false (policy gate)."
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
    gold_case_dir: Path,
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
            return _deterministic_tool_match(gold_case_dir, tool_args)
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
            return _deterministic_tool_match(gold_case_dir, tool_args)
        return {
            "name": name,
            "pass": None,
            "skipped": True,
            "score": None,
            "reason": "DeepEval import failed.",
        }

    gold_for_build = gold_case_dir if gold_case_dir.is_dir() else run_dir
    try:
        metrics = build_regression_deepeval_metrics(
            gold_for_build,
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
            return _deterministic_tool_match(gold_case_dir, tool_args)
        if name == "task_completion":
            from deepeval_metrics import _deepeval_judge_model

            if _deepeval_judge_model() is None:
                from deepeval_metrics import deepeval_judge_setup_hint

                return _skipped(name, deepeval_judge_setup_hint())
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
            "gold_case": str(gold_case_dir.resolve()) if gold_case_dir.is_dir() else "",
            "run_dir": str(run_dir),
        }
    )
    gold_for_tools = gold_case_dir if gold_case_dir.is_dir() else run_dir
    expected_tools = expected_tool_calls(gold_for_tools)
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
            fallback = _deterministic_tool_match(gold_case_dir, tool_args)
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
    gold_case_dir: Path, tool_args: dict[str, Any]
) -> dict[str, Any]:
    expected = _load_expected_tool_names(gold_case_dir)
    observed: list[str] = []
    for call in tool_args.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool") or call.get("name") or "").strip()
            if name:
                observed.append(name)
    if not expected:
        return _skipped("tool_correctness", "No expected tool list in gold case.")
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
