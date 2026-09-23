"""Summarisation should reuse the loaded VLM when USE_TRANSFORMERS_VLM_MODEL_AS_LLM is True."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.summaries import (
    _load_local_model_for_summarisation,
    _local_summarisation_model_choice,
    get_model_choice_from_inference_method,
    get_model_source_from_model_choice,
)


def test_local_summarisation_model_choice_uses_vlm_when_flag_on(monkeypatch):
    monkeypatch.setattr("tools.summaries.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", True)
    monkeypatch.setattr(
        "tools.summaries.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    monkeypatch.setattr(
        "tools.summaries.LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "Gemma 3 12B"
    )
    assert _local_summarisation_model_choice() == "Qwen3.5-9B"
    assert get_model_choice_from_inference_method("local") == "Qwen3.5-9B"


def test_local_summarisation_model_choice_uses_pii_llm_when_flag_off(monkeypatch):
    monkeypatch.setattr("tools.summaries.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", False)
    monkeypatch.setattr(
        "tools.summaries.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    monkeypatch.setattr(
        "tools.summaries.LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "Gemma 3 12B"
    )
    assert _local_summarisation_model_choice() == "Gemma 3 12B"
    assert get_model_choice_from_inference_method("local") == "Gemma 3 12B"


def test_vlm_model_name_is_treated_as_local_source(monkeypatch):
    monkeypatch.setattr(
        "tools.summaries.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    monkeypatch.setattr(
        "tools.summaries.LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "Gemma 3 12B"
    )
    assert get_model_source_from_model_choice("Qwen3.5-9B") == "Local"
    assert get_model_source_from_model_choice("Gemma 3 12B") == "Local"


def test_load_local_model_reuses_vlm_and_skips_load_model(monkeypatch):
    monkeypatch.setattr("tools.summaries.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", True)
    monkeypatch.setattr(
        "tools.summaries.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    vlm = MagicMock(name="vlm")
    tokenizer = MagicMock(name="tokenizer")

    with (
        patch(
            "tools.summaries._get_loaded_vlm_model_and_tokenizer",
            return_value=(vlm, tokenizer),
        ) as get_vlm,
        patch("tools.summaries.load_model") as load_model,
    ):
        model, tok, assistant = _load_local_model_for_summarisation()

    assert model is vlm
    assert tok is tokenizer
    assert assistant is None
    get_vlm.assert_called_once()
    load_model.assert_not_called()


def test_load_local_model_raises_if_vlm_missing_when_flag_on(monkeypatch):
    monkeypatch.setattr("tools.summaries.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", True)
    monkeypatch.setattr(
        "tools.summaries.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    monkeypatch.setattr(
        "tools.summaries.LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "Gemma 3 12B"
    )

    with (
        patch(
            "tools.summaries._get_loaded_vlm_model_and_tokenizer",
            return_value=(None, None),
        ),
        patch("tools.summaries.load_model") as load_model,
    ):
        with pytest.raises(RuntimeError, match="VLM is not loaded"):
            _load_local_model_for_summarisation()
    load_model.assert_not_called()


def test_load_local_model_uses_load_model_when_flag_off(monkeypatch):
    monkeypatch.setattr("tools.summaries.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", False)
    pii_model = MagicMock(name="pii_model")
    pii_tok = MagicMock(name="pii_tok")
    assistant = MagicMock(name="assistant")

    with patch(
        "tools.summaries.load_model",
        return_value=(pii_model, pii_tok, assistant),
    ) as load_model:
        model, tok, asst = _load_local_model_for_summarisation()

    assert model is pii_model
    assert tok is pii_tok
    assert asst is assistant
    load_model.assert_called_once()


def test_local_llm_task_model_choice_uses_vlm_when_flag_on(monkeypatch):
    from tools.llm_funcs import local_llm_task_model_choice

    monkeypatch.setattr("tools.llm_funcs.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", True)
    monkeypatch.setattr(
        "tools.llm_funcs.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    monkeypatch.setattr(
        "tools.llm_funcs.LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "Gemma 3 12B"
    )
    assert local_llm_task_model_choice() == "Qwen3.5-9B"


def test_load_local_llm_for_task_reuses_vlm(monkeypatch):
    from tools.llm_funcs import load_local_llm_for_task

    monkeypatch.setattr("tools.llm_funcs.USE_TRANSFORMERS_VLM_MODEL_AS_LLM", True)
    monkeypatch.setattr(
        "tools.llm_funcs.SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3.5-9B"
    )
    vlm = MagicMock(name="vlm")
    tokenizer = MagicMock(name="tokenizer")

    with (
        patch(
            "tools.llm_funcs._get_loaded_vlm_model_and_tokenizer",
            return_value=(vlm, tokenizer),
        ),
        patch("tools.llm_funcs.load_model") as load_model,
    ):
        model, tok, assistant = load_local_llm_for_task()

    assert model is vlm
    assert tok is tokenizer
    assert assistant is None
    load_model.assert_not_called()
