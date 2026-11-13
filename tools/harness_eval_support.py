"""Post-run eval hooks for main-app AgentCore harness runs (reuses agent-redact/eval)."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENT_EVAL = REPO_ROOT / "agent-redact" / "eval"
_AGENT_SHARED = REPO_ROOT / "agent-redact" / "shared"

TOOL_NAME_ALIASES = {
    "verify_redaction_coverage": "verify_coverage",
    "preview_redaction_boxes": "preview_boxes",
}


def _ensure_agent_eval_paths() -> None:
    for folder in (_AGENT_EVAL, _AGENT_SHARED, REPO_ROOT):
        text = str(folder)
        if text not in sys.path:
            sys.path.insert(0, text)


def normalize_harness_tool_name(name: str) -> str:
    return TOOL_NAME_ALIASES.get((name or "").strip(), (name or "").strip())


def harness_eval_tracing_enabled() -> bool:
    """True when ``AGENT_EVAL_DEEPEVAL_TRACING`` is on (agent-redact Gradio path)."""
    try:
        _ensure_agent_eval_paths()
        from eval.arize_monitoring import agent_eval_deepeval_tracing_enabled
    except ImportError:
        try:
            from arize_monitoring import agent_eval_deepeval_tracing_enabled  # type: ignore
        except ImportError:
            return False
    return agent_eval_deepeval_tracing_enabled()


def harness_post_run_eval_enabled() -> bool:
    """Main-app harness eval: ``APP_EVAL_ENABLED`` or ``AGENT_EVAL_DEEPEVAL_TRACING``."""
    try:
        from tools.config import APP_EVAL_ENABLED

        if APP_EVAL_ENABLED:
            return True
    except ImportError:
        pass
    return harness_eval_tracing_enabled()


def resolve_harness_gold_case(
    pdf_path: Path,
    document_name: str,
    task_instructions: str,
) -> tuple[Path | None, str]:
    """Match upload + instructions to a gold case (``APP_EVAL_*`` or ``AGENT_EVAL_*``)."""
    _ensure_agent_eval_paths()
    try:
        from app_eval_env import app_eval_enabled
        from post_run_app_eval import resolve_app_eval_gold_dir
    except ImportError:
        def app_eval_enabled():
            return False  # type: ignore[assignment, misc]
        resolve_app_eval_gold_dir = None  # type: ignore[assignment, misc]

    if app_eval_enabled() and resolve_app_eval_gold_dir is not None:
        gold_dir, msg = resolve_app_eval_gold_dir(
            pdf_path=pdf_path,
            document_name=document_name,
            derived_instructions=(task_instructions or "").strip(),
        )
        if gold_dir is not None:
            return gold_dir, f"**Eval gold case:** {msg}"
        if msg:
            return None, f"**Eval skipped:** {msg}"

    try:
        from agent_eval_env import (
            agent_eval_auto_gold_enabled,
            resolve_env_gold_case_dir,
            resolve_gold_root,
        )
        from gold_match import resolve_gold_case_for_task
    except ImportError as exc:
        logger.debug("Gold match unavailable: %s", exc)
        return None, ""

    fixed = resolve_env_gold_case_dir(REPO_ROOT)
    if fixed is not None and fixed.is_dir():
        return fixed.resolve(), f"**Eval gold case:** `{fixed.name}` (AGENT_EVAL_GOLD)"

    if not agent_eval_auto_gold_enabled():
        return None, ""

    if not pdf_path.is_file():
        return None, ""
    result = resolve_gold_case_for_task(
        pdf_path=pdf_path,
        document_name=document_name,
        task_instructions=(task_instructions or "").strip(),
        gold_root=resolve_gold_root(REPO_ROOT),
    )
    if result.gold_dir is not None:
        return result.gold_dir.resolve(), (
            f"**Eval gold case:** `{result.case_id or result.gold_dir.name}` "
            f"({result.match_level})"
        )
    return None, f"**Eval skipped:** {result.reason or 'no matching gold case'}"


@dataclass
class HarnessEvalRecorder:
    """Collect inline tool calls and run ``write_session_post_run_eval`` after the harness run."""

    session_hash: str = ""
    cumulative_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    gold_case_dir: Path | None = None
    gold_log_line: str = ""
    text_extract_method: str = ""
    pii_method: str = ""
    ocr_method_short: str = ""
    started: bool = False

    def begin(
        self,
        *,
        workspace: Path,
        pdf_path: Path,
        document_name: str,
        instructions: str,
        text_extract_method: str,
        pii_method: str,
        ocr_method_short: str,
    ) -> str:
        """Resolve gold case and reset the tool-call trace. Returns optional activity line."""
        if not harness_post_run_eval_enabled():
            return ""
        self.cumulative_tool_calls = []
        self.text_extract_method = text_extract_method
        self.pii_method = pii_method
        self.ocr_method_short = ocr_method_short
        self.gold_case_dir, self.gold_log_line = resolve_harness_gold_case(
            pdf_path, document_name, instructions
        )
        self.started = True
        self._write_active_task_stub(workspace, document_name, instructions)
        return self.gold_log_line

    def record_tool(self, name: str, arguments: dict[str, Any]) -> None:
        if not self.started:
            return
        alias = normalize_harness_tool_name(name)
        self.cumulative_tool_calls.append(
            {"tool": alias, "args": dict(arguments or {})}
        )

    def write_post_run(
        self,
        *,
        workspace: Path,
        document_name: str,
        instructions: str,
        assistant_output: str,
        workflow_incomplete: bool,
        session_hash: str,
    ) -> Path | None:
        if not self.started or not harness_post_run_eval_enabled():
            return None
        _ensure_agent_eval_paths()
        try:
            from post_run_eval import write_session_post_run_eval
        except ImportError as exc:
            logger.warning("post_run_eval unavailable: %s", exc)
            return None
        sid = (session_hash or self.session_hash or workspace.name).strip() or "harness"
        use_app_eval = False
        try:
            from tools.config import APP_EVAL_ENABLED

            use_app_eval = bool(APP_EVAL_ENABLED)
        except ImportError:
            pass
        try:
            return write_session_post_run_eval(
                session_hash=sid,
                tool_events=[],
                task_instructions=instructions,
                assistant_output=assistant_output,
                document_name=document_name,
                workflow_incomplete=workflow_incomplete,
                cumulative_tool_calls=list(self.cumulative_tool_calls),
                ocr_method=self.text_extract_method or self.ocr_method_short,
                pii_method=self.pii_method,
                orchestrator="main-app-harness",
                model_name="agentcore-harness",
                workspace_root=workspace,
                gold_case_dir=self.gold_case_dir,
                force=use_app_eval,
            )
        except Exception:
            logger.exception("Harness post-run eval failed")
            return None

    def _write_active_task_stub(
        self, workspace: Path, document_name: str, instructions: str
    ) -> None:
        task_path = workspace / "eval" / "active_redaction_task.json"
        payload = {
            "document_name": document_name,
            "task_instructions": instructions,
            "eval_gold_path": str(self.gold_case_dir) if self.gold_case_dir else None,
            "cumulative_tool_calls": [],
        }
        try:
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
