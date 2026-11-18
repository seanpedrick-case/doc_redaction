"""Background AgentCore harness jobs for Gradio (survive page refresh).

Enabled when ``AGENTCORE_HARNESS_DETACHED_JOBS=True``. The worker runs
``run_agentic_redaction`` in a daemon thread, persists ``job_status.json``, and
uploads Output-files deliverables to S3 on completion when requested.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_JOBS: dict[str, "HarnessJob"] = {}
_JOBS_LOCK = threading.Lock()
_MAX_CONCURRENT = 1
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class HarnessJob:
    run_id: str
    session_dir: str
    document_name: str = ""
    status: str = "queued"
    activity: str = ""
    summary: str = ""
    output_files: list[str] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    stop_event: threading.Event = field(default_factory=threading.Event)
    s3_upload_done: bool = False
    post_complete_hooks_done: bool = False
    s3_message: str = ""
    # Gradio / Cognito identity — dropdown/attach must not leak across sessions.
    gradio_session_hash: str = ""

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_dir": self.session_dir,
            "document_name": self.document_name,
            "status": self.status,
            "activity": self.activity,
            "summary": self.summary,
            "output_files": list(self.output_files),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "s3_upload_done": self.s3_upload_done,
            "post_complete_hooks_done": self.post_complete_hooks_done,
            "s3_message": self.s3_message,
            "gradio_session_hash": self.gradio_session_hash,
        }


def job_status_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / "job_status.json"


def _write_status(job: HarnessJob) -> None:
    path = job_status_path(job.session_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(job.to_public_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write harness job_status.json: %s", exc)


def load_job_status_from_disk(session_dir: str | Path) -> dict[str, Any] | None:
    path = job_status_path(session_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def get_job(run_id: str) -> HarnessJob | None:
    with _JOBS_LOCK:
        return _JOBS.get((run_id or "").strip())


def mark_post_complete_hooks_done(run_id: str) -> None:
    job = get_job(run_id)
    if job is None:
        return
    with _JOBS_LOCK:
        job.post_complete_hooks_done = True
        _write_status(job)


def request_cancel(run_id: str) -> bool:
    job = get_job(run_id)
    if job is None or job.is_terminal():
        return False
    job.stop_event.set()
    with _JOBS_LOCK:
        if not job.is_terminal():
            job.status = "cancelled"
            job.summary = (job.summary or "") + "\n\nCancel requested."
            job.finished_at = time.time()
            _write_status(job)
    return True


def _active_count() -> int:
    return sum(1 for j in _JOBS.values() if j.status in {"queued", "running"})


def _normalize_session_hash(session_hash: str | None) -> str:
    return (session_hash or "").strip()


def _job_matches_session(row: dict[str, Any] | HarnessJob, session_hash: str) -> bool:
    """True when the job belongs to this Gradio/Cognito session (or session filter is off)."""
    want = _normalize_session_hash(session_hash)
    if not want:
        # No session identity yet — only show jobs that also have no owner hash.
        if isinstance(row, HarnessJob):
            return not (row.gradio_session_hash or "").strip()
        return not str(row.get("gradio_session_hash") or "").strip()
    if isinstance(row, HarnessJob):
        owned = (row.gradio_session_hash or "").strip()
        if owned:
            return owned == want
        # Legacy in-memory job: fall back to session_dir path segment.
        return want in Path(row.session_dir).parts
    owned = str(row.get("gradio_session_hash") or "").strip()
    if owned:
        return owned == want
    session_dir = str(row.get("session_dir") or "")
    return bool(session_dir) and want in Path(session_dir).parts


def list_jobs(*, limit: int = 20, session_hash: str = "") -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = sorted(
            _JOBS.values(),
            key=lambda j: j.started_at or 0.0,
            reverse=True,
        )
        filtered = [j for j in jobs if _job_matches_session(j, session_hash)]
        out = [j.to_public_dict() for j in filtered[: max(1, int(limit))]]
    return out


def _safe_output_root(output_root: str | Path | None) -> Path:
    """Resolve ``output_root`` only if contained under ``OUTPUT_FOLDER`` (CodeQL py/path-injection)."""
    from tools.config import OUTPUT_FOLDER
    from tools.secure_path_utils import validate_folder_containment

    fallback = Path(OUTPUT_FOLDER)
    if output_root is None or not str(output_root).strip():
        return fallback
    candidate = Path(str(output_root).strip())
    try:
        if validate_folder_containment(candidate, OUTPUT_FOLDER):
            return candidate
    except Exception:
        pass
    logger.warning(
        "Ignoring output_root outside OUTPUT_FOLDER (%r); using %s",
        str(output_root),
        fallback,
    )
    return fallback


def list_jobs_with_disk_scan(
    output_root: str | Path,
    *,
    limit: int = 20,
    session_hash: str = "",
) -> list[dict[str, Any]]:
    """Active/recent jobs for one Gradio session (memory + ``job_status.json`` under output_root)."""
    by_id: dict[str, dict[str, Any]] = {
        d["run_id"]: d
        for d in list_jobs(limit=limit * 2, session_hash=session_hash)
        if d.get("run_id")
    }
    root = _safe_output_root(output_root)
    if root.is_dir():
        found: list[tuple[float, dict[str, Any]]] = []
        try:
            for status_path in root.glob("**/agentcore/*/job_status.json"):
                data = load_job_status_from_disk(status_path.parent)
                if not data or not data.get("run_id"):
                    continue
                if not data.get("session_dir"):
                    data = {**data, "session_dir": str(status_path.parent)}
                if not _job_matches_session(data, session_hash):
                    continue
                rid = str(data["run_id"])
                if rid in by_id:
                    continue
                mtime = status_path.stat().st_mtime
                found.append((mtime, data))
        except OSError:
            found = []
        found.sort(key=lambda t: t[0], reverse=True)
        for _, data in found[:limit]:
            by_id[str(data["run_id"])] = data
    merged = sorted(
        by_id.values(),
        key=lambda d: float(d.get("started_at") or d.get("finished_at") or 0),
        reverse=True,
    )
    return merged[: max(1, int(limit))]


def job_choices_for_dropdown(
    output_root: str | Path,
    *,
    limit: int = 20,
    session_hash: str = "",
) -> list[str]:
    rows = list_jobs_with_disk_scan(output_root, limit=limit, session_hash=session_hash)
    choices: list[str] = []
    for row in rows:
        rid = str(row.get("run_id") or "")
        status = str(row.get("status") or "")
        doc = str(row.get("document_name") or "")
        label = f"{rid[:8]}… — {status}"
        if doc:
            label = f"{label} — {doc}"
        choices.append(f"{rid} | {label}")
    return choices


def parse_run_id_from_choice(choice: str) -> str:
    text = (choice or "").strip()
    if not text:
        return ""
    if " |" in text:
        return text.split(" |", 1)[0].strip()
    return text


def _run_worker(
    job: HarnessJob,
    *,
    document: str,
    instructions: str,
    defaults: Any,
    page_range: str,
    output_root: str,
    session: Any,
    eval_recorder: Any,
    session_hash: str,
    text_extract_method: str,
    pii_method: str,
    cost_code: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    progress_factory: Callable[..., Any] | None = None,
) -> None:
    from tools.agentcore_harness import (
        _pdf_page_count,
        run_agentic_redaction,
        upload_agentic_outputs_to_s3,
    )

    with _JOBS_LOCK:
        job.status = "running"
        job.started_at = time.time()
        job.summary = "Agent run started (detached)…"
        job.activity = "Detached harness job started."
        _write_status(job)

    final = None
    try:
        for progress in run_agentic_redaction(
            document,
            instructions,
            defaults=defaults,
            page_range=page_range,
            output_root=output_root,
            session=session,
            eval_recorder=eval_recorder,
            eval_session_hash=session_hash,
            should_stop=job.stop_event.is_set,
        ):
            final = progress
            with _JOBS_LOCK:
                job.activity = progress.activity or job.activity
                job.summary = progress.summary or job.summary
                job.output_files = list(progress.output_files or [])
                if job.stop_event.is_set() and not progress.done:
                    job.status = "cancelled"
                _write_status(job)
            if progress_factory is not None:
                progress_factory(progress)
            if job.stop_event.is_set() and not progress.done:
                break
    except Exception as exc:  # noqa: BLE001 — surface in job status
        logger.exception("Detached harness job %s failed", job.run_id)
        with _JOBS_LOCK:
            job.status = "failed"
            job.error = str(exc)
            job.summary = f"Agent run failed: {exc}"
            job.activity = (job.activity or "") + f"\n\n**Job error:** {exc}"
            job.finished_at = time.time()
            _write_status(job)
        return

    try:
        if final is not None and final.done and not job.stop_event.is_set():
            from tools.harness_usage_log import record_harness_agent_usage

            record_harness_agent_usage(
                session_hash=(session_hash or "").strip() or session.run_id,
                document_name=session.source_pdf.name,
                time_taken=final.elapsed_seconds,
                token_usage=final.token_usage,
                pii_method=pii_method,
                text_extraction_method=text_extract_method or defaults.ocr_method,
                cost_code=(cost_code or "").strip(),
                total_pages=_pdf_page_count(session.source_pdf),
            )
    except Exception:  # noqa: BLE001
        logger.exception("Harness usage log failed for job %s", job.run_id)

    outputs = list(final.output_files) if final is not None else list(job.output_files)
    if not outputs:
        try:
            outputs = list(session.output_files())
        except Exception:  # noqa: BLE001
            outputs = []

    s3_message = ""
    s3_done = False
    if save_outputs_to_s3 and outputs:
        try:
            from tools.agentcore_harness import resolve_agentic_s3_output_folder

            resolved_prefix = resolve_agentic_s3_output_folder(
                session_dir=session.session_dir,
                output_paths=outputs,
                fallback_s3_folder=s3_output_folder,
            )
            if not resolved_prefix:
                s3_message = (
                    "S3 upload skipped: no first-pass or Gradio S3 output folder."
                )
                s3_done = False
            else:
                s3_message = (
                    upload_agentic_outputs_to_s3(
                        outputs,
                        s3_output_folder=s3_output_folder,
                        save_outputs_to_s3=True,
                        base_document=document,
                        session_dir=session.session_dir,
                        s3_subfolder="agentic",
                    )
                    or ""
                )
                s3_done = not bool(s3_message)
                if s3_done:
                    logger.info(
                        "Harness job %s uploaded agentic outputs under s3://…/%s<stem>/agentic/",
                        job.run_id,
                        resolved_prefix,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("S3 upload failed for harness job %s", job.run_id)
            s3_message = f"S3 upload failed: {exc}"
            s3_done = False

    with _JOBS_LOCK:
        job.output_files = outputs
        job.s3_upload_done = s3_done
        job.s3_message = s3_message
        job.finished_at = time.time()
        activity_l = (job.activity or "").lower()
        summary_l = (job.summary or "").lower()
        stream_failed = (
            "harness stream error" in activity_l
            or "agent run failed" in summary_l
            or bool(job.error)
        )
        if job.stop_event.is_set():
            job.status = "cancelled"
            job.summary = job.summary or "Agent run cancelled."
        elif stream_failed:
            job.status = "failed"
            job.error = job.error or job.summary or "Agent run failed."
        elif final is not None and final.done:
            job.status = "succeeded"
        elif job.status not in _TERMINAL:
            job.status = "failed"
            job.error = job.error or "Agent run ended without completion."
        if s3_message:
            job.activity = (job.activity or "") + f"\n\n**S3:** {s3_message}"
        _write_status(job)


def start_harness_job(
    *,
    files: Any,
    instructions: str,
    text_extract_method: str,
    local_ocr_model: str,
    pii_method: str,
    local_entities: list[str] | None,
    comprehend_entities: list[str] | None,
    llm_entities: list[str] | None,
    allow_list: list[str] | None,
    deny_list: list[str] | None,
    handwrite_signature: list[str] | None,
    page_min: Any,
    page_max: Any,
    output_folder: str | None = None,
    session_hash: str = "",
    cost_code: str = "",
    s3_output_folder: str = "",
    save_outputs_to_s3: bool = False,
) -> HarnessJob:
    """Validate inputs, create session, start background worker. Raises ValueError."""
    from tools.agentcore_harness import (
        HarnessRunDefaults,
        HarnessSession,
        _first_document,
        _page_range_text,
        harness_feature_ready,
        ui_entities_for_pii_method,
        ui_ocr_method_to_tool_value,
    )
    from tools.config import OUTPUT_FOLDER
    from tools.harness_eval_support import (
        HarnessEvalRecorder,
        harness_post_run_eval_enabled,
    )

    ready, reason = harness_feature_ready()
    if not ready:
        raise ValueError(f"AgentCore agent redaction is not configured: {reason}")
    if not (instructions or "").strip():
        raise ValueError("Enter redaction instructions for the agent.")

    with _JOBS_LOCK:
        if _active_count() >= _MAX_CONCURRENT:
            raise ValueError(
                "Another detached harness job is already running. "
                "Wait for it to finish or cancel it before starting a new one."
            )

    document = _first_document(files)
    defaults = HarnessRunDefaults(
        ocr_method=ui_ocr_method_to_tool_value(text_extract_method, local_ocr_model),
        pii_method=pii_method or "AWS Comprehend",
        redact_entities=ui_entities_for_pii_method(
            pii_method, local_entities, comprehend_entities, llm_entities
        ),
        allow_list=[str(x) for x in (allow_list or []) if str(x).strip()],
        deny_list=[str(x) for x in (deny_list or []) if str(x).strip()],
        handwrite_signature_checkbox=list(handwrite_signature or []) or None,
        page_min=int(page_min or 0),
        page_max=int(page_max or 0),
    )
    out_root = output_folder or OUTPUT_FOLDER
    sid = (session_hash or "").strip()
    session = HarnessSession(
        document, defaults=defaults, output_root=out_root, session_hash=sid
    )
    eval_recorder: HarnessEvalRecorder | None = None
    if harness_post_run_eval_enabled():
        eval_recorder = HarnessEvalRecorder(
            session_hash=sid,
            run_source="main_app_harness",
        )
        eval_recorder.begin(
            workspace=session.session_dir,
            pdf_path=session.source_pdf,
            document_name=session.source_pdf.name,
            instructions=instructions,
            text_extract_method=text_extract_method,
            pii_method=pii_method or "AWS Comprehend",
            ocr_method_short=defaults.ocr_method,
        )

    job = HarnessJob(
        run_id=session.run_id,
        session_dir=str(session.session_dir),
        document_name=session.source_pdf.name,
        status="queued",
        summary="Queued detached agent run…",
        activity="Job queued.",
        started_at=time.time(),
        gradio_session_hash=sid,
    )
    with _JOBS_LOCK:
        _JOBS[job.run_id] = job
        _write_status(job)

    thread = threading.Thread(
        target=_run_worker,
        name=f"harness-job-{job.run_id[:8]}",
        kwargs={
            "job": job,
            "document": document,
            "instructions": instructions,
            "defaults": defaults,
            "page_range": _page_range_text(page_min, page_max),
            "output_root": out_root,
            "session": session,
            "eval_recorder": eval_recorder,
            "session_hash": (session_hash or "").strip(),
            "text_extract_method": text_extract_method,
            "pii_method": pii_method or "AWS Comprehend",
            "cost_code": cost_code,
            "s3_output_folder": (s3_output_folder or "").strip(),
            "save_outputs_to_s3": bool(save_outputs_to_s3),
        },
        daemon=True,
    )
    thread.start()
    return job


def poll_job_snapshot(run_id: str) -> dict[str, Any] | None:
    job = get_job(run_id)
    if job is not None:
        return job.to_public_dict()
    # Fallback: not in memory (shouldn't happen same process) — no session_dir known
    return None


def poll_job_snapshot_or_disk(
    run_id: str,
    session_dir: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any] | None:
    snap = poll_job_snapshot(run_id)
    if snap is not None:
        return snap
    if session_dir:
        disk = load_job_status_from_disk(session_dir)
        if disk is not None:
            return disk
    job = get_job(run_id)
    if job is not None:
        disk = load_job_status_from_disk(job.session_dir)
        if disk is not None:
            return disk
    if output_root:
        for row in list_jobs_with_disk_scan(output_root, limit=50):
            if row.get("run_id") == run_id:
                disk = load_job_status_from_disk(row.get("session_dir") or "")
                return disk or row
    for row in list_jobs(limit=50):
        if row.get("run_id") == run_id:
            return load_job_status_from_disk(row.get("session_dir") or "") or row
    return None


def _dropdown_update(
    choices: list[str],
    *,
    preferred_run_id: str = "",
) -> Any:
    """Return ``gr.update(choices=…, value=…)`` so Gradio does not treat the list as value."""
    import gradio as gr

    value = None
    rid = (preferred_run_id or "").strip()
    if rid:
        for choice in choices:
            if parse_run_id_from_choice(choice) == rid or choice.startswith(f"{rid} "):
                value = choice
                break
    if value is None and choices:
        value = choices[0]
    return gr.update(choices=choices, value=value)


def start_detached_agentic_redact_for_gradio(
    files: Any,
    instructions: str,
    text_extract_method: str,
    local_ocr_model: str,
    pii_method: str,
    local_entities: list[str] | None,
    comprehend_entities: list[str] | None,
    llm_entities: list[str] | None,
    allow_list: list[str] | None,
    deny_list: list[str] | None,
    handwrite_signature: list[str] | None,
    page_min: Any,
    page_max: Any,
    output_folder: str | None,
    session_hash: str,
    cost_code: str,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
) -> tuple[str, str, list[str], list[str], str, Any, list[str]]:
    """Gradio entry: start detached job; returns UI fields + run_id + timer active + choices."""
    import gradio as gr

    try:
        job = start_harness_job(
            files=files,
            instructions=instructions,
            text_extract_method=text_extract_method,
            local_ocr_model=local_ocr_model,
            pii_method=pii_method,
            local_entities=local_entities,
            comprehend_entities=comprehend_entities,
            llm_entities=llm_entities,
            allow_list=allow_list,
            deny_list=deny_list,
            handwrite_signature=handwrite_signature,
            page_min=page_min,
            page_max=page_max,
            output_folder=output_folder,
            session_hash=session_hash,
            cost_code=cost_code,
            s3_output_folder=s3_output_folder,
            save_outputs_to_s3=save_outputs_to_s3,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    from tools.config import OUTPUT_FOLDER

    sid = (session_hash or "").strip()
    choices = job_choices_for_dropdown(output_folder or OUTPUT_FOLDER, session_hash=sid)
    summary = f"Detached agent job `{job.run_id}` started. You can refresh this page; the run continues on the server."
    activity = job.activity or "Job started."
    return (
        summary,
        activity,
        [],
        [],
        job.run_id,
        gr.update(active=True),
        _dropdown_update(choices, preferred_run_id=job.run_id),
    )


def poll_detached_harness_job(
    run_id: str,
    output_folder: str | None = None,
    session_hash: str = "",
) -> tuple[str, str, list[str], list[str], str, Any, Any, bool]:
    """Poll job for Gradio Timer. Last value is ``just_finished_for_hooks``."""
    import gradio as gr

    from tools.config import OUTPUT_FOLDER

    rid = (run_id or "").strip()
    sid = (session_hash or "").strip()
    choices = job_choices_for_dropdown(output_folder or OUTPUT_FOLDER, session_hash=sid)
    drop = _dropdown_update(choices, preferred_run_id=rid)
    if not rid:
        return (
            "",
            "No agent run attached.",
            [],
            [],
            "",
            gr.update(active=False),
            drop,
            False,
        )

    snap = poll_job_snapshot_or_disk(rid, output_root=output_folder or OUTPUT_FOLDER)
    if snap is not None and not _job_matches_session(snap, sid):
        snap = None
    if snap is None:
        return (
            f"Unknown job `{rid}`.",
            f"No status for `{rid}`.",
            [],
            [],
            rid,
            gr.update(active=False),
            drop,
            False,
        )

    status = str(snap.get("status") or "")
    files = [p for p in (snap.get("output_files") or []) if p]
    summary = str(snap.get("summary") or f"Status: {status}")
    activity = str(snap.get("activity") or "")
    terminal = status in _TERMINAL
    hooks_done = bool(snap.get("post_complete_hooks_done"))
    just_finished = terminal and not hooks_done and status == "succeeded"
    # Also wire review tabs after failed/cancelled if files exist
    if terminal and not hooks_done and files and status in {"failed", "cancelled"}:
        just_finished = True
    return (
        summary,
        activity,
        files,
        files,
        rid,
        gr.update(active=not terminal),
        drop,
        just_finished,
    )


def cancel_detached_harness_job(run_id: str, session_hash: str = "") -> str:
    rid = (run_id or "").strip()
    if not rid:
        return "No job to cancel."
    snap = poll_job_snapshot_or_disk(rid)
    if snap is None or not _job_matches_session(snap, session_hash):
        return f"Could not cancel `{rid}` (missing, finished, or not in this session)."
    if request_cancel(rid):
        return f"Cancel requested for `{rid}`."
    return f"Could not cancel `{rid}` (missing or already finished)."


def attach_detached_harness_job(
    choice: str,
    output_folder: str | None = None,
    session_hash: str = "",
) -> tuple[str, str, list[str], list[str], str, Any, Any]:
    import gradio as gr

    from tools.config import OUTPUT_FOLDER

    rid = parse_run_id_from_choice(choice)
    sid = (session_hash or "").strip()
    choices = job_choices_for_dropdown(output_folder or OUTPUT_FOLDER, session_hash=sid)
    drop = _dropdown_update(choices, preferred_run_id=rid)
    if not rid:
        return (
            "",
            "Select a job to attach.",
            [],
            [],
            "",
            gr.update(active=False),
            drop,
        )
    snap = poll_job_snapshot_or_disk(rid, output_root=output_folder or OUTPUT_FOLDER)
    if snap is None or not _job_matches_session(snap, sid):
        # Try matching session_dir from dropdown scan (already session-filtered)
        snap = None
        for row in list_jobs_with_disk_scan(
            output_folder or OUTPUT_FOLDER, session_hash=sid
        ):
            if row.get("run_id") == rid:
                snap = row
                break
    if snap is None:
        return (
            f"Job `{rid}` not found.",
            f"Job `{rid}` not found.",
            [],
            [],
            rid,
            gr.update(active=False),
            drop,
        )
    status = str(snap.get("status") or "")
    files = [p for p in (snap.get("output_files") or []) if p]
    return (
        str(snap.get("summary") or f"Attached `{rid}` ({status})"),
        str(snap.get("activity") or ""),
        files,
        files,
        rid,
        gr.update(active=status not in _TERMINAL),
        drop,
    )


def refresh_detached_job_choices(
    output_folder: str | None = None,
    session_hash: str = "",
) -> Any:
    from tools.config import OUTPUT_FOLDER

    choices = job_choices_for_dropdown(
        output_folder or OUTPUT_FOLDER,
        session_hash=(session_hash or "").strip(),
    )
    return _dropdown_update(choices)


__all__ = [
    "HarnessJob",
    "attach_detached_harness_job",
    "cancel_detached_harness_job",
    "get_job",
    "job_choices_for_dropdown",
    "job_status_path",
    "list_jobs",
    "list_jobs_with_disk_scan",
    "load_job_status_from_disk",
    "mark_post_complete_hooks_done",
    "parse_run_id_from_choice",
    "poll_detached_harness_job",
    "poll_job_snapshot",
    "poll_job_snapshot_or_disk",
    "refresh_detached_job_choices",
    "request_cancel",
    "start_detached_agentic_redact_for_gradio",
    "start_harness_job",
]
