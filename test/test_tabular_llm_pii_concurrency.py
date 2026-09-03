"""Tabular LLM PII cell queries: sequential for local transformers and inference server."""

from __future__ import annotations

from tools.config import (
    AWS_LLM_PII_OPTION,
    INFERENCE_SERVER_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
)
from tools.data_anonymise import tabular_llm_pii_max_workers


def test_tabular_llm_pii_is_sequential_for_local_and_inference_server():
    assert tabular_llm_pii_max_workers(LOCAL_TRANSFORMERS_LLM_PII_OPTION) == 1
    assert tabular_llm_pii_max_workers(INFERENCE_SERVER_PII_OPTION) == 1


def test_tabular_llm_pii_keeps_concurrency_for_bedrock():
    assert tabular_llm_pii_max_workers(AWS_LLM_PII_OPTION) >= 1
