"""Tests for Pi session JSONL log helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


def _import_session_logs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PI_SESSION_DIR", str(tmp_path / "sessions"))
    import importlib
    import sys

    pi_dir = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
    if str(pi_dir) not in sys.path:
        sys.path.insert(0, str(pi_dir))
    import session_logs

    session_logs = importlib.reload(session_logs)
    repo = Path(__file__).resolve().parents[1]
    usage_dir = repo / "usage" / "_pi_session_log_tests"
    usage_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(session_logs, "USAGE_LOGS_FOLDER", str(usage_dir) + "/")
    return session_logs


def test_collect_session_log_download_resolves_under_session_dir(monkeypatch, tmp_path):
    session_logs = _import_session_logs(monkeypatch, tmp_path)
    sessions = session_logs.ensure_session_dir()
    log_file = sessions / "test-session.jsonl"
    log_file.write_text('{"type":"test"}\n', encoding="utf-8")

    client = MagicMock()
    client.running = True
    client.get_state.return_value = {"sessionFile": str(log_file)}

    assert session_logs.collect_session_log_download(client) == str(log_file.resolve())


def test_collect_session_log_download_rejects_outside_session_dir(
    monkeypatch, tmp_path
):
    session_logs = _import_session_logs(monkeypatch, tmp_path)
    session_logs.ensure_session_dir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}", encoding="utf-8")

    client = MagicMock()
    client.running = True
    client.get_state.return_value = {"sessionFile": str(outside)}

    assert session_logs.collect_session_log_download(client) is None


def test_copy_session_log_to_usage_folder(monkeypatch, tmp_path):
    session_logs = _import_session_logs(monkeypatch, tmp_path)
    sessions = session_logs.ensure_session_dir()
    log_file = sessions / "agent-run.jsonl"
    log_file.write_text('{"ok":true}\n', encoding="utf-8")

    archived = session_logs.copy_session_log_to_usage_folder(
        log_file,
        session_hash="abc123",
    )
    assert archived is not None
    assert archived.name == "abc123_agent-run.jsonl"
    assert archived.read_text(encoding="utf-8") == '{"ok":true}\n'


def test_persist_session_log_skips_when_csv_logging_disabled(monkeypatch, tmp_path):
    session_logs = _import_session_logs(monkeypatch, tmp_path)
    monkeypatch.setattr(session_logs, "SAVE_LOGS_TO_CSV", False)

    client = MagicMock()
    client.running = True

    assert session_logs.persist_session_log(client, session_hash="x") is None


def test_persist_session_log_uploads_archived_copy(monkeypatch, tmp_path):
    session_logs = _import_session_logs(monkeypatch, tmp_path)
    sessions = session_logs.ensure_session_dir()
    log_file = sessions / "upload-me.jsonl"
    log_file.write_text('{"ok":true}\n', encoding="utf-8")

    uploads: list[tuple[str, str]] = []

    def fake_upload(local_path, s3_prefix):
        uploads.append((local_path, s3_prefix))

    monkeypatch.setattr(session_logs, "RUN_AWS_FUNCTIONS", True)
    monkeypatch.setattr(session_logs, "SAVE_LOGS_TO_CSV", True)
    monkeypatch.setattr(session_logs, "S3_USAGE_LOGS_FOLDER", "usage/test/")
    monkeypatch.setattr(session_logs, "upload_log_file_to_s3", fake_upload)

    client = MagicMock()
    client.running = True
    client.get_state.return_value = {"sessionFile": str(log_file)}

    archived = session_logs.persist_session_log(client, session_hash="sess1")

    assert archived is not None
    assert archived.name == "sess1_upload-me.jsonl"
    assert uploads == [(str(archived), "usage/test/")]
