"""Usage CSV/DynamoDB logging for main-app AgentCore harness runs."""

from __future__ import annotations

from tools.agentcore_harness_stream import HarnessTokenUsage
from tools.cli_usage_logger import create_cli_usage_logger, log_redaction_usage
from tools.config import (
    AGENTCORE_HARNESS_MODEL_ID,
    DEFAULT_COST_CODE,
    DISPLAY_FILE_NAMES_IN_LOGS,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
)

AGENT_USAGE_TASK = "agent"


def record_harness_agent_usage(
    *,
    session_hash: str,
    document_name: str,
    time_taken: float,
    token_usage: HarnessTokenUsage | None,
    pii_method: str = "",
    text_extraction_method: str = "",
    cost_code: str = "",
    total_pages: int = 0,
    usage_logs_folder: str | None = None,
    save_to_dynamodb: bool | None = None,
    save_to_s3: bool | None = None,
    s3_bucket: str | None = None,
    s3_key_prefix: str | None = None,
    llm_model_name: str | None = None,
) -> None:
    """
    Append one usage-log row for a harness agent run (``task=agent``).

    Token counts use ``llm_total_input_tokens`` / ``llm_total_output_tokens`` columns
    (harness Bedrock model), not VLM/PII detector fields.
    """
    if not SAVE_LOGS_TO_CSV and not SAVE_LOGS_TO_DYNAMODB:
        return
    usage = token_usage or HarnessTokenUsage()
    doc_name = (document_name or "").strip()
    if not DISPLAY_FILE_NAMES_IN_LOGS and doc_name:
        doc_name = "hidden"
    model = (
        llm_model_name or AGENTCORE_HARNESS_MODEL_ID or "agentcore-harness"
    ).strip()
    logger = create_cli_usage_logger(logs_folder=usage_logs_folder)
    log_redaction_usage(
        logger=logger,
        session_hash=(session_hash or "").strip() or "harness",
        doc_file_name=doc_name,
        data_file_name="",
        time_taken=float(time_taken),
        total_pages=int(total_pages),
        textract_queries=0,
        pii_method=pii_method or "",
        comprehend_queries=0,
        cost_code=(cost_code or DEFAULT_COST_CODE or "").strip(),
        handwriting_signature="",
        text_extraction_method=text_extraction_method or "",
        is_textract_call=False,
        task=AGENT_USAGE_TASK,
        save_to_dynamodb=save_to_dynamodb,
        save_to_s3=save_to_s3,
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_key_prefix,
        vlm_model_name="",
        vlm_total_input_tokens=0,
        vlm_total_output_tokens=0,
        llm_model_name=model,
        llm_total_input_tokens=int(usage.input_tokens),
        llm_total_output_tokens=int(usage.output_tokens),
    )
