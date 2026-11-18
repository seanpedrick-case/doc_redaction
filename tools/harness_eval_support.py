"""Post-run eval hooks for main-app AgentCore harness runs (reuses agent-redact/eval)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENT_EVAL = REPO_ROOT / "agent-redact" / "eval"
_AGENT_SHARED = REPO_ROOT / "agent-redact" / "shared"

TOOL_NAME_ALIASES = {
    "verify_redaction_coverage": "verify_coverage",
    "preview_redaction_boxes": "preview_boxes",
}

_TRACE_OUTPUT_MAX_CHARS = 4000
_ARIZE_SETUP_DONE = False


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
    raw = (os.environ.get("APP_EVAL_ENABLED") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        from tools.config import APP_EVAL_ENABLED

        if APP_EVAL_ENABLED:
            return True
    except ImportError:
        pass
    return harness_eval_tracing_enabled()


def maybe_setup_harness_arize_tracing() -> bool:
    """One-shot Phoenix/AX setup for main-app harness (no LangChain instrumentor)."""
    global _ARIZE_SETUP_DONE
    if _ARIZE_SETUP_DONE:
        return True
    raw = (os.environ.get("ARIZE_TRACING_ENABLED") or "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False
    try:
        _ensure_agent_eval_paths()
        from eval.arize_monitoring import setup_arize_ax_tracing, tracing_initialized
    except ImportError:
        try:
            from arize_monitoring import (  # type: ignore
                setup_arize_ax_tracing,
                tracing_initialized,
            )
        except ImportError:
            return False
    try:
        ok = bool(setup_arize_ax_tracing(instrument_langchain=False))
        _ARIZE_SETUP_DONE = ok or tracing_initialized()
        return _ARIZE_SETUP_DONE
    except Exception:  # pragma: no cover - never block harness startup
        logger.debug("Harness Arize setup failed", exc_info=True)
        return False


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


def _truncate_trace_text(value: Any, max_chars: int = _TRACE_OUTPUT_MAX_CHARS) -> str:
    text = str(value if value is not None else "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@contextmanager
def harness_agent_turn_span(
    *,
    session_hash: str,
    message: str,
) -> Iterator[dict[str, Any]]:
    """Root AGENT span for one main-app harness run (no-op when Arize unset)."""
    maybe_setup_harness_arize_tracing()
    try:
        _ensure_agent_eval_paths()
        from eval.arize_monitoring import agent_turn_span
    except ImportError:
        try:
            from arize_monitoring import agent_turn_span  # type: ignore
        except ImportError:
            yield {"span": None, "workflow_incomplete": None}
            return
    with agent_turn_span(
        "main_app.harness",
        session_hash=session_hash or None,
        message=message,
        agent_name="main-app-harness",
    ) as meta:
        yield meta


@contextmanager
def harness_inline_tool_span(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tool_use_id: str = "",
) -> Iterator[dict[str, Any]]:
    """OpenInference TOOL span around one inline dispatch (no-op without OTEL)."""
    meta: dict[str, Any] = {"span": None, "output": "", "ok": True}
    try:
        _ensure_agent_eval_paths()
        from eval.arize_monitoring import (
            maybe_annotate_coverage_from_tool_output,
            tracing_initialized,
        )
    except ImportError:
        try:
            from arize_monitoring import (  # type: ignore
                maybe_annotate_coverage_from_tool_output,
                tracing_initialized,
            )
        except ImportError:
            yield meta
            return

    if not tracing_initialized():
        yield meta
        return

    try:
        from openinference.semconv.trace import (
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        yield meta
        return

    alias = normalize_harness_tool_name(tool_name)
    tracer = trace.get_tracer("doc_redaction.harness")
    span = tracer.start_span(
        f"main_app.harness.tool.{alias}",
        context=trace.set_span_in_context(trace.get_current_span()),
    )
    meta["span"] = span
    try:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.TOOL.value,
        )
        span.set_attribute(SpanAttributes.TOOL_NAME, alias)
        if tool_use_id:
            span.set_attribute(SpanAttributes.TOOL_ID, str(tool_use_id))
        if arguments:
            params = json.dumps(arguments, default=str)[:_TRACE_OUTPUT_MAX_CHARS]
            span.set_attribute(SpanAttributes.TOOL_PARAMETERS, params)
            span.set_attribute(SpanAttributes.INPUT_VALUE, params)
    except Exception:  # pragma: no cover
        pass

    try:
        yield meta
    finally:
        output = meta.get("output") or ""
        ok = bool(meta.get("ok", True))
        try:
            if output:
                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, _truncate_trace_text(output)
                )
            maybe_annotate_coverage_from_tool_output(span, output, tool_name=alias)
            if not ok:
                span.set_status(Status(StatusCode.ERROR))
        except Exception:  # pragma: no cover
            pass
        try:
            span.end()
        except Exception:  # pragma: no cover
            pass


@dataclass
class HarnessEvalRecorder:
    """Collect inline tool calls and run ``write_session_post_run_eval`` after the harness run."""

    session_hash: str = ""
    cumulative_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    gold_case_dir: Path | None = None
    gold_log_line: str = ""
    text_extract_method: str = ""
    pii_method: str = ""
    ocr_method_short: str = ""
    started: bool = False
    run_source: str = "main_app_harness"

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
        self.trace_events = []
        self.text_extract_method = text_extract_method
        self.pii_method = pii_method
        self.ocr_method_short = ocr_method_short
        self.gold_case_dir, self.gold_log_line = resolve_harness_gold_case(
            pdf_path, document_name, instructions
        )
        self.started = True
        self._write_active_task_stub(workspace, document_name, instructions)
        return self.gold_log_line

    def record_event(self, kind: str, **payload: Any) -> None:
        if not self.started:
            return
        event = {"ts": time.time(), "kind": kind}
        event.update(payload)
        self.trace_events.append(event)

    def record_turn_start(self, *, round_no: int) -> None:
        self.record_event("turn_start", round_no=round_no)

    def record_turn_end(
        self,
        *,
        round_no: int,
        stop_reason: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        pending_tools: list[str] | None = None,
    ) -> None:
        self.record_event(
            "turn_end",
            round_no=round_no,
            stop_reason=stop_reason or "",
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            pending_tools=list(pending_tools or []),
        )

    def record_tool(self, name: str, arguments: dict[str, Any]) -> None:
        if not self.started:
            return
        alias = normalize_harness_tool_name(name)
        self.cumulative_tool_calls.append(
            {"tool": alias, "args": dict(arguments or {})}
        )
        self.record_event(
            "tool_start",
            tool=alias,
            tool_raw=name,
            args=dict(arguments or {}),
        )

    def record_tool_end(
        self,
        name: str,
        *,
        ok: bool,
        duration_ms: float,
        output: str = "",
        error: str = "",
        tool_use_id: str = "",
    ) -> None:
        alias = normalize_harness_tool_name(name)
        self.record_event(
            "tool_end",
            tool=alias,
            tool_raw=name,
            ok=bool(ok),
            duration_ms=round(float(duration_ms), 3),
            output=_truncate_trace_text(output),
            error=_truncate_trace_text(error) if error else "",
            tool_use_id=tool_use_id or "",
        )

    def record_stream_status(self, status: str, **payload: Any) -> None:
        self.record_event("stream_status", status=status, **payload)

    def write_trace_events_file(self, run_dir: Path) -> Path | None:
        if not self.trace_events:
            return None
        path = Path(run_dir) / "trace_events.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for event in self.trace_events:
                    fh.write(json.dumps(event, default=str) + "\n")
            return path
        except OSError as exc:
            logger.warning("Failed to write harness trace_events.jsonl: %s", exc)
            return None

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
        bench = (os.environ.get("BENCHMARK_SUITE_ID") or "").strip()
        try:
            summary_path = write_session_post_run_eval(
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
                force=use_app_eval or bool(bench),
                run_source=self.run_source or "main_app_harness",
            )
        except Exception:
            logger.exception("Harness post-run eval failed")
            return None
        if summary_path is not None:
            self.write_trace_events_file(Path(summary_path).parent)
        return summary_path

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
