#!/usr/bin/env python3
"""Local agent regression: seed fixture → LangGraph/Pi turn → deterministic eval gates.

Example::

    python agent-redact/eval/run_agent_regression.py \\
        --fixture agent-redact/eval/fixtures/example_smoke --gates-only

    python agent-redact/eval/run_agent_regression.py \\
        --fixture agent-redact/eval/fixtures/example_live \\
        --orchestrator langgraph
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

OrchestratorChoice = Literal["langgraph", "pi", "both"]


def _repo_paths() -> tuple[Path, Path, Path]:
    eval_dir = Path(__file__).resolve().parent
    agent_redact = eval_dir.parent
    repo_root = agent_redact.parent
    return repo_root, agent_redact, eval_dir


def _ensure_agent_imports() -> tuple[Path, Path, Path]:
    repo_root, agent_redact, eval_dir = _repo_paths()
    shared = agent_redact / "shared"
    for path in (
        repo_root,
        agent_redact,
        shared,
        agent_redact / "pi",
        agent_redact / "agentcore",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    return repo_root, agent_redact, eval_dir


def _read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_fixture_pdf(fixture_dir: Path) -> Path:
    """Return PDF path for a fixture (input.pdf or manifest ``input_pdf``)."""
    root = fixture_dir.resolve()
    direct = root / "input.pdf"
    if direct.is_file():
        return direct
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        rel = (data.get("input_pdf") or "").strip()
        if rel:
            candidate = (root / rel).resolve()
            if candidate.is_file():
                return candidate
            repo_root, _, _ = _repo_paths()
            from_repo = (repo_root / rel).resolve()
            if from_repo.is_file():
                return from_repo
    matches = sorted(root.glob("*.pdf"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No input PDF for fixture {root}. Add input.pdf or manifest input_pdf."
    )


def discover_artifacts(search_root: Path) -> dict[str, Path | None]:
    """Find newest review CSV, word OCR, and redacted PDF under *search_root*."""
    root = search_root.resolve()

    def newest(glob_pattern: str) -> Path | None:
        files = [p for p in root.rglob(glob_pattern) if p.is_file()]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    return {
        "review_csv": newest("*_review_file.csv"),
        "ocr_words_csv": newest("*ocr_results_with_words*.csv"),
        "redacted_pdf": newest("*_redacted.pdf"),
    }


def record_tool_calls_from_events(events: list[Any]) -> dict[str, Any]:
    """Build ``tool_args.json`` payload from normalized agent stream events."""
    calls: list[dict[str, Any]] = []
    for event in events:
        kind = str(getattr(event, "kind", "") or "")
        if kind != "tool_start":
            continue
        name = str(getattr(event, "tool_name", None) or "tool")
        args = getattr(event, "tool_args", None)
        if not isinstance(args, dict):
            args = {}
        calls.append({"tool": name, "args": args})
    return {"tool_calls": calls}


@dataclass
class TurnResult:
    orchestrator: str
    session_hash: str
    workspace: Path
    events: list[Any] = field(default_factory=list)
    tool_args: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class GateSummary:
    policy_gate: dict[str, Any] | None = None
    policy_extraction: dict[str, Any] | None = None
    gold_geometry: dict[str, Any] | None = None
    exit_code: int = 0
    messages: list[str] = field(default_factory=list)


def _apply_orchestrator_env(orchestrator: str) -> None:
    os.environ["AGENT_ORCHESTRATOR"] = orchestrator.strip().lower()


def _apply_provider_env(provider: str | None, model: str | None) -> None:
    if provider and provider.strip():
        os.environ["AGENT_DEFAULT_PROVIDER"] = provider.strip()
    if model and model.strip():
        os.environ["AGENT_DEFAULT_MODEL"] = model.strip()


def _consume_prompt_events(
    runtime: Any, prompt: str, *, timeout_s: float
) -> TurnResult:
    from agent_runtime import AgentStreamEvent

    events: list[Any] = []
    deadline = time.monotonic() + max(1.0, timeout_s)
    error: str | None = None
    try:
        for event in runtime.prompt_events(prompt):
            if time.monotonic() > deadline:
                runtime.abort()
                error = f"Timed out after {timeout_s}s"
                break
            events.append(event)
            if isinstance(event, AgentStreamEvent) and event.kind == "error":
                error = event.text or "Agent error"
            if isinstance(event, AgentStreamEvent) and event.kind == "done":
                break
    except Exception as exc:
        error = str(exc)
    tool_args = record_tool_calls_from_events(events)
    return TurnResult(
        orchestrator=str(getattr(runtime, "orchestrator", "")),
        session_hash="",
        workspace=Path("."),
        events=events,
        tool_args=tool_args,
        error=error,
        elapsed_s=0.0,
    )


def run_agent_turn(
    *,
    orchestrator: str,
    session_hash: str,
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    timeout_s: float = 3600.0,
) -> TurnResult:
    """Run one redaction task prompt through Pi or LangGraph."""
    _apply_orchestrator_env(orchestrator)
    _apply_provider_env(provider, model)
    from agent_runtime import AgentRuntimeError, create_agent_runtime

    started = time.monotonic()
    runtime = create_agent_runtime(session_hash)
    runtime.start()
    if provider or model:
        prov = (provider or os.environ.get("AGENT_DEFAULT_PROVIDER") or "").strip()
        mod = (model or os.environ.get("AGENT_DEFAULT_MODEL") or "").strip()
        if prov and mod:
            runtime.apply_backend(prov, mod)
        elif prov:
            runtime.set_model(prov, mod or "")
    try:
        result = _consume_prompt_events(runtime, prompt, timeout_s=timeout_s)
    finally:
        try:
            runtime.close()
        except Exception:
            pass
    result.session_hash = session_hash
    result.elapsed_s = round(time.monotonic() - started, 2)
    if orchestrator == "pi" and not runtime.running and result.error is None:
        # Pi RPC may fail to start (missing node pi).
        pass
    if result.error and orchestrator == "pi":
        raise AgentRuntimeError(result.error)
    return result


def seed_fixture_workspace(
    fixture_dir: Path,
    session_hash: str,
) -> tuple[Path, str, str]:
    """
    Copy fixture PDF + build redaction prompt into the session workspace.

    Returns ``(workspace_dir, pdf_basename, full_prompt)``.
    """
    repo_root, _, _ = _ensure_agent_imports()
    from bootstrap_pi_config import ensure_pi_config_env

    ensure_pi_config_env(repo_root)
    from redaction_prompt import prepare_redaction_task
    from session_workspace import ensure_session_workspace

    fixture_dir = fixture_dir.resolve()
    instructions = _read_text(fixture_dir / "instructions.txt")
    if not instructions:
        instructions = _read_text(fixture_dir / "instruction.txt")
    pdf_path = resolve_fixture_pdf(fixture_dir)
    workspace = ensure_session_workspace(session_hash)
    file_name, prompt, _renamed = prepare_redaction_task(
        pdf_path,
        instructions,
        workspace_dir=workspace,
    )
    return workspace, file_name, prompt


def write_run_artifacts(
    run_dir: Path,
    *,
    turn: TurnResult | None,
    discovered: dict[str, Path | None],
    fixture_dir: Path,
) -> dict[str, Path | None]:
    """Persist agent outputs and metadata under *run_dir*."""
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {
        "review_csv": None,
        "ocr_words_csv": None,
        "redacted_pdf": None,
        "tool_args": None,
    }
    if turn is not None:
        meta = {
            "orchestrator": turn.orchestrator,
            "session_hash": turn.session_hash,
            "elapsed_s": turn.elapsed_s,
            "error": turn.error,
            "provider": os.environ.get("AGENT_DEFAULT_PROVIDER"),
            "model": os.environ.get("AGENT_DEFAULT_MODEL"),
        }
        (run_dir / "turn_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        tool_path = run_dir / "tool_args.json"
        tool_path.write_text(json.dumps(turn.tool_args, indent=2), encoding="utf-8")
        paths["tool_args"] = tool_path

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
        paths[key] = dest

    # Fall back to fixture offline artifacts for gates-only runs.
    from run_policy_gate import _discover_fixture_files

    fixture_files = _discover_fixture_files(fixture_dir)
    for key, fixture_key in (
        ("review_csv", "review_csv"),
        ("ocr_words_csv", "ocr_words_csv"),
        ("redacted_pdf", "redacted_pdf"),
    ):
        if paths.get(key) is not None:
            continue
        src = fixture_files.get(fixture_key)
        if src is not None and src.is_file():
            if key == "review_csv":
                dest = run_dir / "agent_review_file.csv"
            elif key == "ocr_words_csv":
                dest = run_dir / "ocr_results_with_words.csv"
            else:
                dest = run_dir / "redacted.pdf"
            if not dest.is_file():
                shutil.copy2(src, dest)
            paths[key] = dest

    manifest_src = fixture_dir / "manifest.json"
    if manifest_src.is_file():
        shutil.copy2(manifest_src, run_dir / "manifest.json")
    return paths


def run_deterministic_gates(
    run_dir: Path,
    fixture_dir: Path,
    *,
    min_gold_f1: float | None = None,
    require_pass_strict: bool = True,
    compare_extraction: bool = True,
) -> GateSummary:
    """Run policy gate, optional extraction F1, optional gold geometry compare."""
    from compare_policy_lists import compare_policy_extraction, load_observed_tool_args
    from compare_review_csvs import compare_review_csvs
    from run_policy_gate import (
        _discover_fixture_files,
        _read_pattern_file,
        run_policy_gate,
    )

    summary = GateSummary()
    exit_code = 0
    run_dir = run_dir.resolve()
    fixture_dir = fixture_dir.resolve()
    artifact_paths = {
        "review_csv": run_dir / "agent_review_file.csv",
        "ocr_words_csv": run_dir / "ocr_results_with_words.csv",
        "redacted_pdf": run_dir / "redacted.pdf",
        "tool_args": run_dir / "tool_args.json",
    }
    for key, path in list(artifact_paths.items()):
        if not path.is_file():
            artifact_paths[key] = None  # type: ignore[assignment]

    fixture_discovered = _discover_fixture_files(fixture_dir)
    review_csv = artifact_paths["review_csv"] or fixture_discovered.get("review_csv")
    ocr_csv = artifact_paths["ocr_words_csv"] or fixture_discovered.get("ocr_words_csv")
    redacted = artifact_paths["redacted_pdf"] or fixture_discovered.get("redacted_pdf")

    if review_csv is None or not Path(review_csv).is_file():
        summary.messages.append("Missing review CSV for policy gate.")
        summary.exit_code = 2
        return summary
    if ocr_csv is None or not Path(ocr_csv).is_file():
        summary.messages.append("Missing OCR words CSV for policy gate.")
        summary.exit_code = 2
        return summary

    must_redact = _read_pattern_file(fixture_discovered.get("must_redact_file"))
    must_not = _read_pattern_file(fixture_discovered.get("must_not_redact_file"))

    summary.policy_gate = run_policy_gate(
        review_csv=Path(review_csv),
        ocr_words_csv=Path(ocr_csv),
        must_redact=must_redact,
        must_not_redact=must_not,
        redacted_pdf=Path(redacted) if redacted and Path(redacted).is_file() else None,
    )
    (run_dir / "policy_gate.json").write_text(
        json.dumps(summary.policy_gate, indent=2), encoding="utf-8"
    )
    if require_pass_strict and not summary.policy_gate.get("pass_strict"):
        summary.messages.append("Policy gate: pass_strict is false.")
        exit_code = 1

    tool_args_path = artifact_paths["tool_args"] or fixture_discovered.get("tool_args")
    gold_must = fixture_discovered.get("must_redact_file")
    gold_not = fixture_discovered.get("must_not_redact_file")
    gold_deny = fixture_dir / "expected_deny_list.txt"
    if not gold_deny.is_file():
        gold_deny = fixture_dir / "deny_list.txt"

    if compare_extraction and tool_args_path and Path(tool_args_path).is_file():
        if gold_must and gold_must.is_file():
            from compare_policy_lists import extract_lists_from_tool_calls

            observed = extract_lists_from_tool_calls(
                load_observed_tool_args(Path(tool_args_path))
            )
            report = compare_policy_extraction(
                gold_must_redact=_read_pattern_file(gold_must),
                gold_must_not_redact=_read_pattern_file(gold_not),
                gold_deny_list=_read_pattern_file(
                    gold_deny if gold_deny.is_file() else None
                ),
                observed_must_redact=observed["must_redact"],
                observed_must_not_redact=observed["must_not_redact"],
                observed_deny_list=observed["deny_list"],
            )
            summary.policy_extraction = report.to_dict()
            (run_dir / "policy_extraction.json").write_text(
                json.dumps(summary.policy_extraction, indent=2, default=str),
                encoding="utf-8",
            )
            if not report.guardrail.get("pass", True):
                summary.messages.append(
                    f"Policy extraction guardrail failed: {report.guardrail.get('issues')}"
                )
                exit_code = max(exit_code, 1)

    ideal = fixture_dir / "ideal_review_file.csv"
    agent_review = artifact_paths["review_csv"] or review_csv
    if ideal.is_file() and agent_review and Path(agent_review).is_file():
        report = compare_review_csvs(ideal, agent_review)
        summary.gold_geometry = asdict(report)
        (run_dir / "gold_geometry.json").write_text(
            json.dumps(summary.gold_geometry, indent=2, default=str), encoding="utf-8"
        )
        if min_gold_f1 is not None and report.f1 < min_gold_f1:
            summary.messages.append(f"Gold F1 {report.f1} below {min_gold_f1}.")
            exit_code = 1

    summary.exit_code = exit_code
    return summary


def run_regression(
    *,
    fixture_dir: Path,
    orchestrator: OrchestratorChoice = "langgraph",
    session_hash: str | None = None,
    run_dir: Path | None = None,
    gates_only: bool = False,
    provider: str | None = None,
    model: str | None = None,
    timeout_s: float = 3600.0,
    min_gold_f1: float | None = None,
) -> tuple[GateSummary, Path]:
    """Full regression pipeline; returns gate summary and output directory."""
    repo_root, _, eval_dir = _ensure_agent_imports()
    fixture_dir = fixture_dir.resolve()
    case_id = fixture_dir.name
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    out = run_dir or (eval_dir / "runs" / f"{case_id}_{stamp}_{short_id}")
    out.mkdir(parents=True, exist_ok=True)

    session = session_hash or f"agent_eval_{case_id}_{short_id}"
    turn: TurnResult | None = None
    workspace_for_search = repo_root / "workspace"

    if not gates_only:
        workspace, _pdf_name, prompt = seed_fixture_workspace(fixture_dir, session)
        workspace_for_search = workspace
        orchestrators: list[str]
        if orchestrator == "both":
            orchestrators = ["langgraph", "pi"]
        else:
            orchestrators = [orchestrator]

        last_turn: TurnResult | None = None
        for orch in orchestrators:
            orch_out = out / orch if orchestrator == "both" else out
            orch_out.mkdir(parents=True, exist_ok=True)
            last_turn = run_agent_turn(
                orchestrator=orch,
                session_hash=session,
                prompt=prompt,
                provider=provider,
                model=model,
                timeout_s=timeout_s,
            )
            discovered = discover_artifacts(workspace)
            write_run_artifacts(
                orch_out, turn=last_turn, discovered=discovered, fixture_dir=fixture_dir
            )
            if last_turn.error:
                summary = GateSummary(
                    exit_code=1,
                    messages=[f"{orch} turn failed: {last_turn.error}"],
                )
                return summary, orch_out
        turn = last_turn
        if orchestrator == "both":
            # Score LangGraph run dir by default; Pi artifacts remain alongside.
            out = out / "langgraph"
    else:
        write_run_artifacts(
            out,
            turn=None,
            discovered=discover_artifacts(workspace_for_search),
            fixture_dir=fixture_dir,
        )

    gates = run_deterministic_gates(
        out,
        fixture_dir,
        min_gold_f1=min_gold_f1,
    )
    if turn and turn.error:
        gates.messages.append(f"Turn error (scored artifacts anyway): {turn.error}")
    return gates, out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local agent regression harness.")
    p.add_argument("--fixture", type=Path, required=True, help="Fixture directory.")
    p.add_argument(
        "--orchestrator",
        choices=("langgraph", "pi", "both"),
        default=os.environ.get("AGENT_ORCHESTRATOR", "langgraph"),
        help="Orchestrator backend (default: env or langgraph).",
    )
    p.add_argument(
        "--gates-only",
        action="store_true",
        help="Skip live agent; score artifacts already in fixture or workspace.",
    )
    p.add_argument("--provider", help="Override AGENT_DEFAULT_PROVIDER.")
    p.add_argument("--model", help="Override AGENT_DEFAULT_MODEL.")
    p.add_argument("--session-hash", help="Gradio-style session id for workspace.")
    p.add_argument(
        "--run-dir", type=Path, help="Output directory (default: eval/runs/...)."
    )
    p.add_argument(
        "--timeout-s", type=float, default=3600.0, help="Agent turn timeout."
    )
    p.add_argument(
        "--min-gold-f1", type=float, default=None, help="Minimum gold CSV F1."
    )
    p.add_argument(
        "--allow-fail", action="store_true", help="Exit 0 even when gates fail."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary, out_dir = run_regression(
            fixture_dir=args.fixture,
            orchestrator=args.orchestrator,  # type: ignore[arg-type]
            session_hash=args.session_hash,
            run_dir=args.run_dir,
            gates_only=args.gates_only,
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            min_gold_f1=args.min_gold_f1,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Regression failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {"run_dir": str(out_dir), "summary": asdict(summary)}, indent=2, default=str
        )
    )
    for msg in summary.messages:
        print(msg, file=sys.stderr)
    if args.allow_fail:
        return 0
    return int(summary.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
