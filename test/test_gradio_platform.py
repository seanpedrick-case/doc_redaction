"""Tests for shared Gradio platform helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools import gradio_platform


class _FakeRequest:
    def __init__(
        self,
        *,
        username: str | None = None,
        session_hash: str = "abc123",
        headers: dict | None = None,
    ):
        self.username = username
        self.session_hash = session_hash
        self.headers = headers or {}


def test_resolve_session_identity_prefers_username():
    request = _FakeRequest(username="alice@example.com", session_hash="fallback")
    assert gradio_platform.resolve_session_identity(request) == "alice@example.com"


def test_resolve_session_identity_uses_cognito_header():
    request = _FakeRequest(headers={"x-cognito-id": "user-42"})
    assert gradio_platform.resolve_session_identity(request) == "user-42"


def test_resolve_session_identity_falls_back_to_session_hash():
    request = _FakeRequest(session_hash="sess-99")
    assert gradio_platform.resolve_session_identity(request) == "sess-99"


def test_validate_custom_header_raises_when_missing(monkeypatch):
    monkeypatch.setattr(gradio_platform, "CUSTOM_HEADER", "X-Route")
    monkeypatch.setattr(gradio_platform, "CUSTOM_HEADER_VALUE", "expected")
    request = _FakeRequest()
    with pytest.raises(ValueError, match="Custom header value not found"):
        gradio_platform.validate_custom_header(request)


def test_build_s3_outputs_prefix_adds_session_and_date(monkeypatch):
    monkeypatch.setattr(gradio_platform, "SAVE_OUTPUTS_TO_S3", True)
    with patch("tools.gradio_platform.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260528"
        prefix = gradio_platform.build_s3_outputs_prefix(
            "user1",
            "outputs/",
            session_scoped=True,
        )
    assert prefix == "outputs/user1/20260528/"


def test_build_s3_outputs_prefix_no_session_scope(monkeypatch):
    monkeypatch.setattr(gradio_platform, "SAVE_OUTPUTS_TO_S3", True)
    with patch("tools.gradio_platform.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260528"
        prefix = gradio_platform.build_s3_outputs_prefix(
            "user1",
            "outputs/",
            session_scoped=False,
        )
    assert prefix == "outputs/20260528/"


def test_log_platform_access_calls_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(gradio_platform, "get_access_logger", lambda: logger)
    monkeypatch.setattr(gradio_platform, "SAVE_LOGS_TO_CSV", True)
    monkeypatch.setattr(gradio_platform, "SAVE_LOGS_TO_DYNAMODB", False)
    gradio_platform.log_platform_access("user-a", "host-1")
    logger.log.assert_called_once_with("user-a", "host-1")


def test_log_pi_usage_event_skips_chat_when_disabled(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(gradio_platform, "get_pi_usage_logger", lambda: logger)
    monkeypatch.setattr(gradio_platform, "SAVE_LOGS_TO_CSV", True)
    monkeypatch.setattr(gradio_platform, "PI_LOG_CHAT_USAGE", False)
    gradio_platform.log_pi_usage_event(session_hash="u1", event="chat_message")
    logger.log.assert_not_called()


def test_create_fastapi_app_exposes_health():
    app = gradio_platform.create_fastapi_app()
    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths


def test_gradio_head_html_sets_base_href():
    html = gradio_platform.gradio_head_html("pi-agent")
    assert "<base href='/pi-agent/'>" in html
