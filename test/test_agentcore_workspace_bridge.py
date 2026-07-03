"""Tests for AgentCore workspace upload bridge."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_test_support import ensure_gradio_importable

ensure_gradio_importable()

import agentcore_workspace_bridge as bridge


def test_discover_session_document_name_picks_newest_pdf(tmp_path, monkeypatch):
    import time

    base = tmp_path / "workspace"
    session = base / "sess"
    session.mkdir(parents=True)
    older = session / "old.pdf"
    older.write_bytes(b"%PDF-1.4 old")
    time.sleep(0.05)
    newer = session / "new.pdf"
    newer.write_bytes(b"%PDF-1.4 new")

    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "true")

    assert bridge.discover_session_document_name("sess") == "new.pdf"


def test_collect_session_files_for_agentcore_upload_includes_redact_tree(
    tmp_path, monkeypatch
):
    base = tmp_path / "workspace"
    session = base / "sess"
    doc = session / "report.pdf"
    review = (
        session / "redact" / "report.pdf" / "output_redact" / "report_review_file.csv"
    )
    review.parent.mkdir(parents=True)
    doc.write_bytes(b"%PDF-1.4 source")
    review.write_text("id,page\n", encoding="utf-8-sig")
    preview = session / "preview" / "latest_redacted.pdf"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"%PDF-1.4 preview")

    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "true")

    staged = bridge.collect_session_files_for_agentcore_upload("sess")
    paths = {item["relative_path"] for item in staged}
    assert "report.pdf" in paths
    assert "redact/report.pdf/output_redact/report_review_file.csv" in paths
    assert "preview/latest_redacted.pdf" not in paths


def test_build_agentcore_followup_context_mentions_review_csv(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session = base / "sess"
    csv = session / "redact" / "doc.pdf" / "output_redact" / "doc_review_file.csv"
    csv.parent.mkdir(parents=True)
    (session / "doc.pdf").write_bytes(b"%PDF")
    csv.write_text("id\n", encoding="utf-8-sig")

    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "true")

    text = bridge.build_agentcore_followup_context(
        "sess",
        [{"role": "user", "content": "Redact Dr Hyde"}],
    )
    assert "follow-up" in text.lower()
    assert "doc_review_file.csv" in text
    assert "Redact Dr Hyde" in text


def test_collect_upload_skips_oversized_files(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session = base / "sess"
    big = session / "redact" / "doc.pdf" / "output_redact" / "big.bin"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"x" * 20)
    small = session / "redact" / "doc.pdf" / "output_redact" / "notes.txt"
    small.write_text("ok")

    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "true")
    monkeypatch.setenv("AGENTCORE_MAX_UPLOAD_BYTES", "10")

    staged = bridge.collect_session_files_for_agentcore_upload("sess")
    assert staged
    skipped = staged[-1]
    assert skipped["relative_path"] == ".agentcore_upload_skipped.txt"
    note = base64.b64decode(skipped["content_base64"]).decode("utf-8")
    assert "big.bin" in note


def test_build_agentcore_invoke_runtime_config_uses_pi_env(monkeypatch):
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "http://host.docker.internal:7861")
    monkeypatch.setenv("AGENT_DEFAULT_OCR_METHOD", "paddle")
    monkeypatch.setenv("HF_TOKEN", "should-not-pass")
    config = bridge.build_agentcore_invoke_runtime_config()
    assert config["DOC_REDACTION_GRADIO_URL"] == "http://host.docker.internal:7861"
    assert config["AGENT_DEFAULT_OCR_METHOD"] == "paddle"
    assert "HF_TOKEN" not in config


def test_build_agentcore_invoke_runtime_config_includes_hf_token_for_space(
    monkeypatch,
):
    monkeypatch.setenv(
        "DOC_REDACTION_GRADIO_URL",
        "https://seanpedrickcase-document-redaction.hf.space",
    )
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    config = bridge.build_agentcore_invoke_runtime_config()
    assert config["HF_TOKEN"] == "hf-secret"


class _FakeS3Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _FakeS3Client:
    def __init__(self, data: bytes):
        self._data = data
        self.calls = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3 kwarg name
        self.calls.append((Bucket, Key))
        return {"Body": _FakeS3Body(self._data)}


def _install_fake_s3(monkeypatch, body: str):
    import boto3

    client = _FakeS3Client(body.encode("utf-8"))
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
    return client


def _reset_overlay(monkeypatch):
    monkeypatch.setattr(bridge, "_cloudfront_overlay_applied", False, raising=False)
    for key in (
        "AGENT_ORCHESTRATOR",
        "AGENTCORE_RUNTIME_URL",
        "AGENT_DEFAULT_MODEL",
        "DOC_REDACTION_GRADIO_URL",
        "DOC_REDACTION_AUTH_TOKEN",
        "DOC_REDACTION_CONFIG_S3_BUCKET",
        "DOC_REDACTION_CONFIG_S3_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_overlay_promotes_orchestrator_from_s3(monkeypatch):
    """Phase-2 recycle: S3 agent.env flips an un-set container to AgentCore."""
    _reset_overlay(monkeypatch)
    monkeypatch.setenv("DOC_REDACTION_CONFIG_S3_BUCKET", "cfg-bucket")
    monkeypatch.setenv("DOC_REDACTION_CONFIG_S3_KEY", "agent.env")
    _install_fake_s3(
        monkeypatch,
        "AGENT_ORCHESTRATOR=agentcore\n"
        "AGENTCORE_RUNTIME_URL=https://runtime.example.amazonaws.com\n"
        "AGENT_DEFAULT_MODEL=mistral.mistral-large-2407\n"
        "DOC_REDACTION_GRADIO_URL=https://cf.example.cloudfront.net/agent\n",
    )

    applied = bridge.apply_agentcore_cloudfront_config_overlay()

    assert os.environ["AGENT_ORCHESTRATOR"] == "agentcore"
    assert (
        os.environ["AGENTCORE_RUNTIME_URL"] == "https://runtime.example.amazonaws.com"
    )
    assert os.environ["AGENT_DEFAULT_MODEL"] == "mistral.mistral-large-2407"
    assert applied["AGENT_ORCHESTRATOR"] == "agentcore"


def test_overlay_noop_for_pi_deploy(monkeypatch):
    """A pi deploy sharing the config bucket must not be promoted."""
    _reset_overlay(monkeypatch)
    monkeypatch.setenv("DOC_REDACTION_CONFIG_S3_BUCKET", "cfg-bucket")
    _install_fake_s3(
        monkeypatch,
        "AGENT_ORCHESTRATOR=pi\nDOC_REDACTION_GRADIO_URL=http://redaction:7860\n",
    )

    applied = bridge.apply_agentcore_cloudfront_config_overlay()

    assert applied == {}
    assert "AGENT_ORCHESTRATOR" not in os.environ
    assert "AGENTCORE_RUNTIME_URL" not in os.environ


def test_overlay_noop_without_config_bucket(monkeypatch):
    _reset_overlay(monkeypatch)
    monkeypatch.setenv("AGENT_ORCHESTRATOR", "agentcore")

    applied = bridge.apply_agentcore_cloudfront_config_overlay()

    assert applied == {}


def test_apply_invoke_runtime_config_overrides_agentcore_env(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("langchain_core")
    _AGENTCORE = Path(__file__).resolve().parents[1] / "agent-redact" / "agentcore"
    if str(_AGENTCORE) not in sys.path:
        sys.path.insert(0, str(_AGENTCORE))
    from invoke_agent import apply_invoke_runtime_config

    monkeypatch.setenv(
        "DOC_REDACTION_GRADIO_URL",
        "https://seanpedrickcase-document-redaction.hf.space",
    )
    apply_invoke_runtime_config(
        {
            "runtime_config": {
                "DOC_REDACTION_GRADIO_URL": "http://host.docker.internal:7861",
            }
        }
    )
    assert (
        os.environ.get("DOC_REDACTION_GRADIO_URL") == "http://host.docker.internal:7861"
    )
