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


def test_build_agent_usage_log_row_uses_main_app_schema(monkeypatch):
    monkeypatch.setattr(gradio_platform, "DISPLAY_FILE_NAMES_IN_LOGS", True)
    monkeypatch.setattr(gradio_platform, "DEFAULT_COST_CODE", "CC1")
    monkeypatch.setattr(gradio_platform, "HOST_NAME", "test-host")
    row = gradio_platform.build_agent_usage_log_row(
        session_hash="user-1",
        duration_seconds=12.5,
        document_name="report.pdf",
        total_page_count=3,
        ocr_method="hybrid-paddle-vlm",
        pii_method="LLM (AWS Bedrock)",
        llm_model_name="amazon-bedrock/anthropic.claude-sonnet-4-6",
        vlm_model_name="anthropic.claude-sonnet-4-6",
        task="agent",
    )
    assert row[0] == "user-1"
    assert row[1] == "report"
    assert row[3] == 12.5
    assert row[4] == 3
    assert row[6] == "LLM (AWS Bedrock)"
    assert row[11] == "hybrid-paddle-vlm"
    assert row[13] == "agent"
    assert row[17] == "amazon-bedrock/anthropic.claude-sonnet-4-6"


def test_log_agent_usage_event_writes_via_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(gradio_platform, "get_agent_usage_logger", lambda: logger)
    monkeypatch.setattr(gradio_platform, "SAVE_LOGS_TO_CSV", True)
    monkeypatch.setattr(gradio_platform, "SAVE_LOGS_TO_DYNAMODB", False)
    gradio_platform.log_agent_usage_event(
        session_hash="user-1",
        duration_seconds=5,
        llm_model_name="amazon-bedrock/claude",
        task="agent",
    )
    logger.log_row.assert_called_once()
    row = logger.log_row.call_args.args[0]
    assert row[13] == "agent"
    assert row[17] == "amazon-bedrock/claude"


def test_create_fastapi_app_exposes_health():
    app = gradio_platform.create_fastapi_app()
    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths


def test_gradio_head_html_sets_base_href():
    html = gradio_platform.gradio_head_html("pi-agent")
    assert "<base href='/pi-agent/'>" in html


def test_mount_or_launch_subpath_serves_200_not_404(monkeypatch):
    """Regression: mounting Gradio at /pi must not double-apply the prefix as FastAPI
    root_path (which strips /pi before routing and returns 404 for GET /pi/)."""
    import gradio as gr
    from starlette.testclient import TestClient

    monkeypatch.setattr(gradio_platform, "RUN_FASTAPI", True)
    monkeypatch.setattr(gradio_platform, "COGNITO_AUTH", False)

    with gr.Blocks() as demo:
        gr.Markdown("hi")

    app = gradio_platform.mount_or_launch(
        demo,
        root_path="/pi",
        fastapi_root_path="/pi",
    )
    client = TestClient(app)
    assert client.get("/pi/").status_code == 200
    assert client.get("/pi/config").status_code == 200
    # Health endpoint on the parent app is unaffected by the subpath mount.
    assert client.get("/health").status_code == 200


def _run_asgi(app, scope):
    import asyncio

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    asyncio.run(app(scope, receive, send))


def test_forwarded_host_middleware_rewrites_host_and_scheme():
    captured = {}

    async def inner(scope, receive, send):
        captured["scheme"] = scope["scheme"]
        captured["host"] = dict(scope["headers"]).get(b"host")

    mw = gradio_platform.ForwardedHostMiddleware(inner)
    scope = {
        "type": "http",
        "scheme": "http",
        "headers": [
            (b"host", b"internal.ecs.on.aws"),
            (b"x-forwarded-host", b"cf.example.net"),
            (b"x-forwarded-proto", b"https"),
        ],
    }
    _run_asgi(mw, scope)
    assert captured["host"] == b"cf.example.net"
    assert captured["scheme"] == "https"


def test_forwarded_host_middleware_passthrough_without_header():
    captured = {}

    async def inner(scope, receive, send):
        captured["host"] = dict(scope["headers"]).get(b"host")

    mw = gradio_platform.ForwardedHostMiddleware(inner)
    scope = {
        "type": "http",
        "scheme": "http",
        "headers": [(b"host", b"internal.ecs.on.aws")],
    }
    _run_asgi(mw, scope)
    assert captured["host"] == b"internal.ecs.on.aws"


def test_forwarded_host_middleware_fixes_trailing_slash_redirect_target():
    """The exact bug: GET /agent -> 307 /agent/ must redirect to the CloudFront host,
    not the internal origin host (which is unreachable once locked to CloudFront)."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    async def page(_request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/agent/", page)])
    app.add_middleware(gradio_platform.ForwardedHostMiddleware)
    client = TestClient(app)
    resp = client.get(
        "/agent",
        headers={
            "host": "internal.ecs.on.aws",
            "x-forwarded-host": "cf.example.net",
            "x-forwarded-proto": "https",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 307
    assert resp.headers["location"] == "https://cf.example.net/agent/"
