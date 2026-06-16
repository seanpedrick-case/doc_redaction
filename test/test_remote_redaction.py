"""Tests for Pi agent Gradio download helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from remote_redaction import (  # noqa: E402
    clear_redaction_client_cache,
    discover_redaction_outputs,
    extract_server_paths,
    fetch_redaction_files,
    is_gradio_file_path,
    is_gradio_rate_limit_error,
    make_redaction_client,
)


def test_make_redaction_client_uses_token_kwarg(monkeypatch):
    calls: list[tuple[str, str | None, tuple[str, str] | None]] = []

    class _FakeClient:
        def __init__(self, url, token=None, auth=None, **kwargs):
            calls.append((url, token, auth))

    monkeypatch.setattr("remote_redaction.Client", _FakeClient)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "https://example.hf.space")
    clear_redaction_client_cache()
    client = make_redaction_client()
    assert client is not None
    assert calls == [("https://example.hf.space", "hf_secret", None)]


def test_make_redaction_client_uses_gradio_auth(monkeypatch):
    calls: list[tuple[str, str | None, tuple[str, str] | None]] = []

    class _FakeClient:
        def __init__(self, url, token=None, auth=None, **kwargs):
            calls.append((url, token, auth))

    monkeypatch.setattr("remote_redaction.Client", _FakeClient)
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "https://redact.example.com")
    monkeypatch.setenv("DOC_REDACTION_GRADIO_AUTH_USER", "svc-user")
    monkeypatch.setenv("DOC_REDACTION_GRADIO_AUTH_PASSWORD", "svc-pass")
    clear_redaction_client_cache()
    make_redaction_client()
    assert calls == [
        ("https://redact.example.com", None, ("svc-user", "svc-pass")),
    ]


def test_make_redaction_client_retries_rate_limit(monkeypatch):
    attempts = {"n": 0}

    class _TooMany(Exception):
        pass

    class _FakeClient:
        def __init__(self, url, token=None, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _TooMany("429 Too Many Requests")

    monkeypatch.setattr("remote_redaction.Client", _FakeClient)
    monkeypatch.setattr("remote_redaction.is_gradio_rate_limit_error", lambda exc: True)
    monkeypatch.setattr("remote_redaction._quota_retry_attempts", lambda: 3)
    monkeypatch.setattr("remote_redaction._quota_retry_delay_s", lambda: 0)
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "https://example.hf.space")
    clear_redaction_client_cache()
    make_redaction_client()
    assert attempts["n"] == 2


def test_is_gradio_rate_limit_error_detects_too_many_requests():
    class TooManyRequestsError(Exception):
        pass

    assert is_gradio_rate_limit_error(TooManyRequestsError("x"))
    assert is_gradio_rate_limit_error(RuntimeError("429 Too Many Requests"))
    assert not is_gradio_rate_limit_error(RuntimeError("connection refused"))


def test_is_gradio_file_path_windows_and_unix():
    assert is_gradio_file_path("/tmp/gradio/foo.pdf")
    assert is_gradio_file_path(r"C:\Users\me\workspace\.gradio_uploads\a.pdf")
    assert is_gradio_file_path("C:/Users/me/output/doc_redacted.pdf")
    assert not is_gradio_file_path("relative/path.pdf")
    assert not is_gradio_file_path("")


def test_extract_server_paths_nested_windows(tmp_path):
    sample = tmp_path / "review.csv"
    sample.write_text("a,b\n", encoding="utf-8")
    win_path = str(sample)
    result = (
        "ok",
        [win_path, {"path": win_path}],
        ["/linux/only.pdf"],
    )
    paths = extract_server_paths(result)
    assert win_path in paths
    assert "/linux/only.pdf" in paths


def test_discover_redaction_outputs_skips_split_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    assert discover_redaction_outputs("example_doc", since=None) == []


def test_fetch_redaction_files_local_copy(tmp_path):
    src = tmp_path / "artifact.pdf"
    src.write_bytes(b"%PDF-1.4")
    dest = tmp_path / "out"
    copied = fetch_redaction_files([str(src)], dest, base_url="http://127.0.0.1:7860")
    assert len(copied) == 1
    assert copied[0].read_bytes() == b"%PDF-1.4"
