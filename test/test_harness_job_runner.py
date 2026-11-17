"""Tests for detached AgentCore harness job runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import harness_job_runner as hjr


@pytest.fixture(autouse=True)
def _clear_jobs():
    with hjr._JOBS_LOCK:
        hjr._JOBS.clear()
    yield
    with hjr._JOBS_LOCK:
        hjr._JOBS.clear()


def test_config_detached_jobs_default_false():
    from tools.config import AGENTCORE_HARNESS_DETACHED_JOBS

    assert AGENTCORE_HARNESS_DETACHED_JOBS is False or isinstance(
        AGENTCORE_HARNESS_DETACHED_JOBS, bool
    )


def test_write_and_load_job_status(tmp_path: Path):
    job = hjr.HarnessJob(
        run_id="abc123",
        session_dir=str(tmp_path / "sess"),
        document_name="doc.pdf",
        status="running",
        activity="hello",
        summary="sum",
        output_files=["/tmp/a.pdf"],
        started_at=1.0,
    )
    (tmp_path / "sess").mkdir()
    hjr._write_status(job)
    loaded = hjr.load_job_status_from_disk(tmp_path / "sess")
    assert loaded is not None
    assert loaded["run_id"] == "abc123"
    assert loaded["status"] == "running"
    assert loaded["output_files"] == ["/tmp/a.pdf"]


def test_request_cancel_sets_event(tmp_path: Path):
    job = hjr.HarnessJob(
        run_id="cancel-me",
        session_dir=str(tmp_path),
        status="running",
    )
    with hjr._JOBS_LOCK:
        hjr._JOBS[job.run_id] = job
    assert hjr.request_cancel("cancel-me") is True
    assert job.stop_event.is_set()
    assert job.status == "cancelled"


def test_second_concurrent_start_rejected(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "tools.agentcore_harness.harness_feature_ready",
        lambda: (True, ""),
    )

    existing = hjr.HarnessJob(
        run_id="busy",
        session_dir=str(tmp_path / "busy"),
        status="running",
    )
    with hjr._JOBS_LOCK:
        hjr._JOBS[existing.run_id] = existing

    with pytest.raises(ValueError, match="already running"):
        hjr.start_harness_job(
            files=str(tmp_path / "missing.pdf"),
            instructions="redact names",
            text_extract_method="AWS Textract",
            local_ocr_model="tesseract",
            pii_method="AWS Comprehend",
            local_entities=[],
            comprehend_entities=[],
            llm_entities=[],
            allow_list=[],
            deny_list=[],
            handwrite_signature=[],
            page_min=0,
            page_max=0,
        )


def test_worker_s3_upload_on_success(monkeypatch, tmp_path: Path):
    session_dir = tmp_path / "agentcore" / "run1"
    session_dir.mkdir(parents=True)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    uploaded: dict = {}

    def fake_upload(paths, **kwargs):
        uploaded["paths"] = list(paths)
        uploaded["kwargs"] = kwargs
        return None  # export_outputs_to_s3 returns None on success

    class FakeProgress:
        def __init__(self):
            self.activity = "done activity"
            self.summary = "All done"
            self.output_files = [str(tmp_path / "out.pdf")]
            self.done = True
            self.elapsed_seconds = 1.5
            self.token_usage = None

    def fake_run(*args, **kwargs):
        yield FakeProgress()

    session = SimpleNamespace(
        run_id="run1",
        session_dir=session_dir,
        source_pdf=pdf,
        output_files=lambda: [str(tmp_path / "out.pdf")],
    )
    defaults = SimpleNamespace(ocr_method="AWS Textract")

    monkeypatch.setattr(
        "tools.agentcore_harness.run_agentic_redaction",
        fake_run,
    )
    monkeypatch.setattr(
        "tools.agentcore_harness.upload_agentic_outputs_to_s3",
        fake_upload,
    )
    monkeypatch.setattr(
        "tools.agentcore_harness._pdf_page_count",
        lambda p: 1,
    )
    monkeypatch.setattr(
        "tools.harness_usage_log.record_harness_agent_usage",
        lambda **k: None,
    )

    job = hjr.HarnessJob(
        run_id="run1",
        session_dir=str(session_dir),
        document_name="doc.pdf",
        status="queued",
    )
    with hjr._JOBS_LOCK:
        hjr._JOBS[job.run_id] = job

    hjr._run_worker(
        job,
        document=str(pdf),
        instructions="redact",
        defaults=defaults,
        page_range="all",
        output_root=str(tmp_path),
        session=session,
        eval_recorder=None,
        session_hash="sess",
        text_extract_method="AWS Textract",
        pii_method="AWS Comprehend",
        cost_code="",
        s3_output_folder="s3://bucket/out",
        save_outputs_to_s3=True,
    )

    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert job.status == "succeeded"
    assert job.s3_upload_done is True
    assert uploaded["paths"] == [str(tmp_path / "out.pdf")]
    assert uploaded["kwargs"]["save_outputs_to_s3"] is True
    disk = json.loads((session_dir / "job_status.json").read_text(encoding="utf-8"))
    assert disk["status"] == "succeeded"
    assert disk["s3_upload_done"] is True


def test_parse_run_id_from_choice():
    assert (
        hjr.parse_run_id_from_choice("abcdef12 | abcdef12… — running — doc.pdf")
        == "abcdef12"
    )
    assert hjr.parse_run_id_from_choice("") == ""


def test_dropdown_update_sets_choices_and_value():
    update = hjr._dropdown_update(
        ["aaa | running — a.pdf", "bbb | succeeded — b.pdf"],
        preferred_run_id="bbb",
    )
    assert update["choices"][1].startswith("bbb")
    assert update["value"].startswith("bbb")
    assert update.get("__type__") == "update"


def test_job_choices_filtered_by_gradio_session(tmp_path: Path):
    a = hjr.HarnessJob(
        run_id="run-a",
        session_dir=str(tmp_path / "a" / "agentcore" / "run-a"),
        document_name="a.pdf",
        status="running",
        started_at=2.0,
        gradio_session_hash="user-a",
    )
    b = hjr.HarnessJob(
        run_id="run-b",
        session_dir=str(tmp_path / "b" / "agentcore" / "run-b"),
        document_name="b.pdf",
        status="succeeded",
        started_at=1.0,
        gradio_session_hash="user-b",
    )
    with hjr._JOBS_LOCK:
        hjr._JOBS[a.run_id] = a
        hjr._JOBS[b.run_id] = b
    choices_a = hjr.job_choices_for_dropdown(tmp_path, session_hash="user-a")
    choices_b = hjr.job_choices_for_dropdown(tmp_path, session_hash="user-b")
    assert len(choices_a) == 1 and "run-a" in choices_a[0]
    assert len(choices_b) == 1 and "run-b" in choices_b[0]
    assert "run-b" not in "".join(choices_a)
    assert "run-a" not in "".join(choices_b)


def test_mark_post_complete_hooks_done(tmp_path: Path):
    job = hjr.HarnessJob(
        run_id="hooks",
        session_dir=str(tmp_path),
        status="succeeded",
    )
    with hjr._JOBS_LOCK:
        hjr._JOBS[job.run_id] = job
    hjr.mark_post_complete_hooks_done("hooks")
    assert job.post_complete_hooks_done is True
