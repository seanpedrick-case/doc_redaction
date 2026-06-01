"""Shared Gradio deployment helpers: session identity, FastAPI mount, logging."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import gradio as gr
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from fastapi import FastAPI, status
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tools.auth import authenticate_user
from tools.aws_functions import upload_log_file_to_s3
from tools.config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    ACCESS_LOGS_FOLDER,
    ALLOWED_HOSTS,
    ALLOWED_ORIGINS,
    AWS_USER_POOL_ID,
    COGNITO_AUTH,
    CSV_ACCESS_LOG_HEADERS,
    CSV_USAGE_LOG_HEADERS,
    CUSTOM_HEADER,
    CUSTOM_HEADER_VALUE,
    DEFAULT_COST_CODE,
    DISPLAY_FILE_NAMES_IN_LOGS,
    DYNAMODB_ACCESS_LOG_HEADERS,
    DYNAMODB_USAGE_LOG_HEADERS,
    FASTAPI_ROOT_PATH,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    HOST_NAME,
    LOG_FILE_NAME,
    ROOT_PATH,
    RUN_FASTAPI,
    S3_ACCESS_LOGS_FOLDER,
    S3_OUTPUTS_FOLDER,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SAVE_OUTPUTS_TO_S3,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_FILE_NAME,
    USAGE_LOGS_FOLDER,
)
from tools.custom_csvlogger import CSVLogger_custom


def validate_custom_header(request: gr.Request) -> None:
    """Raise when CUSTOM_HEADER is configured but missing or wrong on the request."""
    if not CUSTOM_HEADER or not CUSTOM_HEADER_VALUE:
        return
    headers = getattr(request, "headers", None) or {}
    if CUSTOM_HEADER in headers:
        supplied = headers[CUSTOM_HEADER]
        if supplied == CUSTOM_HEADER_VALUE:
            print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
            return
        print("Custom header value does not match expected value.")
        raise ValueError("Custom header value does not match expected value.")
    print("Custom header value not found.")
    raise ValueError("Custom header value not found.")


def resolve_session_identity(request: gr.Request) -> str:
    """
    Resolve the session identifier from Gradio auth, Cognito/OIDC headers, or session hash.
    """
    validate_custom_header(request)
    headers = getattr(request, "headers", None) or {}

    if request.username:
        return request.username

    if "x-cognito-id" in headers:
        out_session_hash = headers["x-cognito-id"]
        print("Cognito ID found:", out_session_hash)
        return out_session_hash

    if "x-amzn-oidc-identity" in headers:
        out_session_hash = headers["x-amzn-oidc-identity"]

        if AWS_USER_POOL_ID:
            try:
                cognito_client = boto3.client("cognito-idp")
                response = cognito_client.admin_get_user(
                    UserPoolId=AWS_USER_POOL_ID,
                    Username=out_session_hash,
                )
                email = next(
                    attr["Value"]
                    for attr in response["UserAttributes"]
                    if attr["Name"] == "email"
                )
                print("Cognito email address found, will be used as session hash")
                out_session_hash = email
            except (
                ClientError,
                NoCredentialsError,
                PartialCredentialsError,
                BotoCoreError,
            ) as exc:
                print(f"Error fetching Cognito user details: {exc}")
                print("Falling back to using AWS ID as session hash")
            except Exception as exc:
                print(f"Unexpected error when fetching Cognito user details: {exc}")
                print("Falling back to using AWS ID as session hash")

        print("AWS ID found, will be used as username for session:", out_session_hash)
        return out_session_hash

    return request.session_hash


def build_s3_outputs_prefix(
    session_hash: str,
    base_folder: str = S3_OUTPUTS_FOLDER,
    *,
    session_scoped: bool = True,
) -> str:
    """Build the S3 key prefix for output uploads (optional session + date suffix)."""
    s3_outputs_folder = base_folder or ""

    if session_scoped and session_hash and s3_outputs_folder:
        if SAVE_OUTPUTS_TO_S3:
            s3_outputs_folder = s3_outputs_folder.rstrip("/") + "/" + session_hash + "/"
    elif not session_scoped:
        pass
    elif not session_hash or not s3_outputs_folder:
        s3_outputs_folder = base_folder or ""

    if SAVE_OUTPUTS_TO_S3 and s3_outputs_folder:
        today_suffix = datetime.now().strftime("%Y%m%d") + "/"
        s3_outputs_folder = s3_outputs_folder.rstrip("/") + "/" + today_suffix

    return s3_outputs_folder


def gradio_head_html(root_path: str = ROOT_PATH) -> str:
    """HTML head snippet with base href for reverse-proxy subpaths."""
    clean_path = f"/{root_path.strip('/')}"
    base_href = f"{clean_path}/" if clean_path != "/" else "/"
    if root_path:
        print(f"Setting HTML base href for Gradio to: '{base_href}'")
    return (
        f"<base href='{base_href}'>\n\n"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/'
        '4.3.1/iframeResizer.contentWindow.min.js" '
        'integrity="sha256-62pj+jS8t+leByFOFwjiY0T92YlWwowYgHnFRklgv0M=" '
        'crossorigin="anonymous"></script>'
    )


def create_fastapi_app() -> FastAPI:
    """Create FastAPI app with lifespan, optional CORS/trusted-host middleware, and /health."""
    from tools.helper_functions import lifespan

    clean_root = (
        f"/{FASTAPI_ROOT_PATH.strip('/')}" if FASTAPI_ROOT_PATH.strip("/") else ""
    )
    fastapi_app = FastAPI(lifespan=lifespan, root_path=clean_root)

    if ALLOWED_ORIGINS:
        print(f"CORS enabled. Allowing origins: {ALLOWED_ORIGINS}")
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if ALLOWED_HOSTS:
        fastapi_app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @fastapi_app.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        return {"status": "ok"}

    return fastapi_app


def _cognito_auth():
    return authenticate_user if COGNITO_AUTH else None


class _LogField:
    """Minimal stand-in for Gradio components in CSVLogger_custom."""

    def __init__(self, label: str) -> None:
        self.label = label

    def flag(self, sample: Any, flag_dir: Any = None) -> str:
        return str(sample) if sample is not None else ""


class PlatformAccessLogger:
    """Access log writer using CSVLogger_custom."""

    def __init__(self) -> None:
        self._callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)
        self._fields = [
            _LogField("session_hash"),
            _LogField("host_name"),
        ]
        self._callback.setup(self._fields, ACCESS_LOGS_FOLDER)

    def log(self, session_hash: str, host_name: str = HOST_NAME) -> None:
        self._callback.flag(
            [session_hash, host_name],
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS,
            replacement_headers=CSV_ACCESS_LOG_HEADERS or None,
        )
        upload_log_file_to_s3(
            ACCESS_LOGS_FOLDER + LOG_FILE_NAME,
            S3_ACCESS_LOGS_FOLDER,
        )


class PlatformAgentUsageLogger:
    """Agent orchestration usage log writer (main-app CSV / DynamoDB / S3 schema)."""

    _MAIN_USAGE_FIELD_LABELS: tuple[str, ...] = (
        "session_hash_textbox",
        "doc_full_file_name_textbox",
        "data_full_file_name_textbox",
        "actual_time_taken_number",
        "total_page_count",
        "textract_query_number",
        "pii_detection_method",
        "comprehend_query_number",
        "cost_code",
        "textract_handwriting_signature",
        "host_name_textbox",
        "text_extraction_method",
        "is_this_a_textract_api_call",
        "task",
        "vlm_model_name",
        "vlm_total_input_tokens",
        "vlm_total_output_tokens",
        "llm_model_name",
        "llm_total_input_tokens",
        "llm_total_output_tokens",
    )

    def __init__(self) -> None:
        self._callback = CSVLogger_custom(dataset_file_name=USAGE_LOG_FILE_NAME)
        labels = (
            list(CSV_USAGE_LOG_HEADERS)
            if CSV_USAGE_LOG_HEADERS
            else list(self._MAIN_USAGE_FIELD_LABELS)
        )
        self._fields = [_LogField(label) for label in labels]
        self._callback.setup(self._fields, USAGE_LOGS_FOLDER)

    def log_row(self, row: list[Any]) -> None:
        self._callback.flag(
            row,
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS or None,
        )
        upload_log_file_to_s3(
            USAGE_LOGS_FOLDER + USAGE_LOG_FILE_NAME,
            S3_USAGE_LOGS_FOLDER,
        )


def _doc_name_for_usage_log(document_name: str) -> str:
    if DISPLAY_FILE_NAMES_IN_LOGS:
        if not document_name:
            return ""
        return Path(document_name).stem
    return "document" if document_name else ""


def build_agent_usage_log_row(
    *,
    session_hash: str,
    duration_seconds: float | int | str = "",
    document_name: str = "",
    total_page_count: int | str = 0,
    ocr_method: str = "",
    pii_method: str = "",
    llm_model_name: str = "",
    vlm_model_name: str = "",
    llm_input_tokens: int | str = 0,
    llm_output_tokens: int | str = 0,
    vlm_input_tokens: int | str = 0,
    vlm_output_tokens: int | str = 0,
    task: str = "agent",
) -> list[Any]:
    """Build a usage log row matching the main redaction app schema."""
    return [
        session_hash,
        _doc_name_for_usage_log(document_name),
        "",
        duration_seconds,
        total_page_count,
        0,
        pii_method,
        0,
        DEFAULT_COST_CODE,
        False,
        HOST_NAME,
        ocr_method,
        False,
        task,
        vlm_model_name,
        vlm_input_tokens,
        vlm_output_tokens,
        llm_model_name,
        llm_input_tokens,
        llm_output_tokens,
    ]


# Module-level singletons for PI / lightweight callers
_access_logger: PlatformAccessLogger | None = None
_agent_usage_logger: PlatformAgentUsageLogger | None = None


def get_access_logger() -> PlatformAccessLogger:
    global _access_logger
    if _access_logger is None:
        _access_logger = PlatformAccessLogger()
    return _access_logger


def get_agent_usage_logger() -> PlatformAgentUsageLogger:
    global _agent_usage_logger
    if _agent_usage_logger is None:
        _agent_usage_logger = PlatformAgentUsageLogger()
    return _agent_usage_logger


def log_platform_access(session_hash: str, host_name: str = HOST_NAME) -> None:
    if not SAVE_LOGS_TO_CSV and not SAVE_LOGS_TO_DYNAMODB:
        return
    get_access_logger().log(session_hash, host_name)


def log_agent_usage_event(
    *,
    session_hash: str,
    duration_seconds: float | int | str = "",
    document_name: str = "",
    total_page_count: int | str = 0,
    ocr_method: str = "",
    pii_method: str = "",
    llm_model_name: str = "",
    vlm_model_name: str = "",
    llm_input_tokens: int | str = 0,
    llm_output_tokens: int | str = 0,
    vlm_input_tokens: int | str = 0,
    vlm_output_tokens: int | str = 0,
    task: str = "agent",
) -> None:
    """Log one Pi agent run to the main-app usage CSV / DynamoDB / S3 locations."""
    if not SAVE_LOGS_TO_CSV and not SAVE_LOGS_TO_DYNAMODB:
        return
    row = build_agent_usage_log_row(
        session_hash=session_hash,
        duration_seconds=duration_seconds,
        document_name=document_name,
        total_page_count=total_page_count,
        ocr_method=ocr_method,
        pii_method=pii_method,
        llm_model_name=llm_model_name,
        vlm_model_name=vlm_model_name,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        vlm_input_tokens=vlm_input_tokens,
        vlm_output_tokens=vlm_output_tokens,
        task=task,
    )
    get_agent_usage_logger().log_row(row)


def log_pi_usage_event(**kwargs: Any) -> None:
    """Back-compat alias — maps legacy Pi kwargs onto ``log_agent_usage_event``."""
    event = kwargs.pop("event", "")
    provider = kwargs.pop("provider", "")
    model = kwargs.pop("model", "")
    deployment_profile = kwargs.pop("deployment_profile", "")
    llm_model_name = kwargs.pop("llm_model_name", "")
    if not llm_model_name:
        parts = [part for part in (provider, model, deployment_profile, event) if part]
        llm_model_name = "/".join(parts) if parts else model
    log_agent_usage_event(
        llm_model_name=llm_model_name,
        **kwargs,
    )


def wire_pi_usage_logging(**kwargs: Any) -> None:
    """Log a Pi agent usage event (direct-call helper for Gradio handlers)."""
    log_agent_usage_event(**kwargs)


def wire_access_logging(
    session_hash_component: gr.Component,
    host_name_component: gr.Component,
    access_logs_state: gr.Component,
    access_s3_logs_loc_state: gr.Component,
    *,
    flag_output: gr.Component | None = None,
) -> gr.events.EventListener:
    """
    Wire access logging on session_hash change (main-app pattern).

    Returns the EventListener so callers can chain further .success handlers.
    """
    access_callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)
    access_callback.setup(
        [session_hash_component, host_name_component], ACCESS_LOGS_FOLDER
    )
    outputs = [flag_output] if flag_output is not None else []
    return session_hash_component.change(
        lambda *args: access_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS,
            replacement_headers=CSV_ACCESS_LOG_HEADERS or None,
        ),
        [session_hash_component, host_name_component],
        outputs=outputs,
        preprocess=False,
    ).success(
        fn=upload_log_file_to_s3,
        inputs=[access_logs_state, access_s3_logs_loc_state],
        outputs=[],
    )


def mount_or_launch(
    demo: gr.Blocks,
    *,
    fastapi_app: FastAPI | None = None,
    allowed_paths: list[str] | None = None,
    css: str | None = None,
    head_extra: str = "",
    theme: gr.themes.Base | None = None,
    server_name: str | None = None,
    server_port: int | None = None,
    show_error: bool = True,
    queue_kwargs: dict[str, Any] | None = None,
) -> FastAPI | None:
    """
    Mount Gradio on FastAPI when RUN_FASTAPI else launch directly.

    Returns the FastAPI app when mounted; None when launched in-process.
    """
    if theme is None:
        theme = gr.themes.Default(primary_hue="blue")
    if server_name is None:
        server_name = GRADIO_SERVER_NAME
    if server_port is None:
        server_port = GRADIO_SERVER_PORT
    if queue_kwargs:
        demo.queue(**queue_kwargs)

    head = gradio_head_html(ROOT_PATH) + (head_extra or "")
    auth = _cognito_auth()
    allowed = allowed_paths or []

    if RUN_FASTAPI:
        if fastapi_app is None:
            fastapi_app = create_fastapi_app()
        return gr.mount_gradio_app(
            fastapi_app,
            demo,
            path="",
            head=head,
            css=css,
            theme=theme,
            show_error=show_error,
            auth=auth,
            allowed_paths=allowed,
        )

    demo.launch(
        theme=theme,
        head=head,
        css=css,
        show_error=show_error,
        server_name=server_name,
        server_port=server_port,
        root_path=ROOT_PATH,
        auth=auth,
        allowed_paths=allowed,
    )
    return None
