"""Tests for derive_app_eval_instructions."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import (  # noqa: E402
    AppEvalRedactionSettings,
    derive_app_eval_instructions,
    settings_from_json,
    settings_from_legacy_redactor_kwargs,
    synthetic_doc_redact_tool_args,
)


def test_entities_and_deny_sorted_and_quoted():
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=["PHONE_NUMBER", "PERSON", "ADDRESS"],
        deny_list=["Mr Smith", "O'Brien"],
    )
    text = derive_app_eval_instructions(settings)
    assert "- Redact any instance of ADDRESS, PERSON, PHONE_NUMBER" in text
    assert "- Redact any instances of the term 'Mr Smith'" in text
    assert '"O\'Brien"' in text


def test_extract_text_only():
    settings = AppEvalRedactionSettings(text_extraction_only=True)
    text = derive_app_eval_instructions(settings)
    assert "Extract text only" in text
    assert "Redact any instance" not in text


def test_allow_list_bullet():
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=["PERSON"],
        allow_list=["Public Body"],
    )
    text = derive_app_eval_instructions(settings)
    assert "Do not redact" in text
    assert "Public Body" in text


def test_legacy_kwargs_mapping():
    flat = {
        "chosen_redact_entities": ["PERSON"],
        "in_deny_list": ["x"],
        "text_extraction_method": "PyMuPDF",
        "pii_identification_method": "Local",
        "text_extraction_only": False,
    }
    settings = settings_from_legacy_redactor_kwargs(flat)
    assert settings.merged_entities() == ["PERSON"]
    assert settings.deny_list == ["x"]
    tool = synthetic_doc_redact_tool_args(settings)
    assert tool["tool_calls"][0]["args"]["deny_list"] == ["x"]


def test_chosen_ocr_model_json_round_trip(tmp_path):
    import json

    settings = AppEvalRedactionSettings(
        chosen_local_ocr_model="hybrid-paddle-inference-server",
        chosen_ocr_model="qwen_3_5_27b",
    )
    assert settings.to_settings_dict()["chosen_ocr_model"] == "qwen_3_5_27b"
    path = tmp_path / "app_eval_settings.json"
    path.write_text(json.dumps(settings.to_settings_dict()), encoding="utf-8")
    loaded = settings_from_json(str(path))
    assert loaded.chosen_local_ocr_model == "hybrid-paddle-inference-server"
    assert loaded.chosen_ocr_model == "qwen_3_5_27b"


def test_chosen_ocr_model_legacy_kwargs_and_fallbacks(tmp_path):
    import json

    # explicit per-run model kwargs win
    settings = settings_from_legacy_redactor_kwargs(
        {
            "chosen_local_ocr_model": "hybrid-paddle-inference-server",
            "inference_server_vlm_model": "qwen_3_5_27b",
        }
    )
    assert settings.chosen_ocr_model == "qwen_3_5_27b"

    settings2 = settings_from_legacy_redactor_kwargs(
        {
            "chosen_local_ocr_model": "hybrid-paddle-vlm",
            "vlm_model_choice": "gemma-3-27b",
        }
    )
    assert settings2.chosen_ocr_model == "gemma-3-27b"

    # engines without a separate model resolve to ""
    settings3 = settings_from_legacy_redactor_kwargs(
        {"chosen_local_ocr_model": "tesseract"}
    )
    assert settings3.chosen_ocr_model == ""

    # settings_from_json accepts legacy key fallbacks
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"inference_server_vlm_model": "qwen_3_5_27b"}), encoding="utf-8"
    )
    assert settings_from_json(str(path)).chosen_ocr_model == "qwen_3_5_27b"


def test_pii_llm_model_mapping(tmp_path):
    import json

    # explicit per-run kwargs win
    settings = settings_from_legacy_redactor_kwargs(
        {
            "pii_identification_method": "LLM (AWS Bedrock)",
            "llm_model_choice": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        }
    )
    assert settings.pii_llm_model == "anthropic.claude-3-7-sonnet-20250219-v1:0"

    settings2 = settings_from_legacy_redactor_kwargs(
        {
            "pii_identification_method": "Inference server LLM",
            "inference_server_pii_model": "gemma_3_27b",
        }
    )
    assert settings2.pii_llm_model == "gemma_3_27b"

    # non-LLM PII methods resolve to ""
    settings3 = settings_from_legacy_redactor_kwargs(
        {"pii_identification_method": "Local"}
    )
    assert settings3.pii_llm_model == ""
    settings4 = settings_from_legacy_redactor_kwargs(
        {"pii_identification_method": "AWS Comprehend"}
    )
    assert settings4.pii_llm_model == ""

    # json round-trip incl. legacy fallback keys
    settings5 = AppEvalRedactionSettings(
        pii_identification_method="LLM (AWS Bedrock)",
        pii_llm_model="amazon.nova-pro-v1:0",
    )
    assert settings5.to_settings_dict()["pii_llm_model"] == "amazon.nova-pro-v1:0"
    path = tmp_path / "s2.json"
    path.write_text(json.dumps(settings5.to_settings_dict()), encoding="utf-8")
    assert settings_from_json(str(path)).pii_llm_model == "amazon.nova-pro-v1:0"
    path2 = tmp_path / "s3.json"
    path2.write_text(
        json.dumps({"llm_model_choice": "amazon.nova-pro-v1:0"}), encoding="utf-8"
    )
    assert settings_from_json(str(path2)).pii_llm_model == "amazon.nova-pro-v1:0"


def test_local_pii_ignores_llm_entity_checkboxes_for_gold_match():
    """Gradio keeps CHOSEN_LLM_ENTITIES in state; Local PII must not change instructions.txt hash."""
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=["TITLES", "PERSON"],
        chosen_redact_comprehend_entities=["EMAIL", "NAME"],
        chosen_llm_entities=["PERSON_NAME", "STREET_ADDRESS"],
        pii_identification_method="Local",
    )
    assert settings.merged_entities() == ["EMAIL", "NAME", "PERSON", "TITLES"]
    assert "PERSON_NAME" not in derive_app_eval_instructions(settings)


def test_show_ex_professor_example_matches_gold_instructions():
    repo = Path(__file__).resolve().parents[1]
    gold = repo / "agent-redact/eval/gold/show_ex_professor_emails_selectable_text"
    if not (gold / "instructions.txt").is_file():
        return
    import json

    settings = AppEvalRedactionSettings(
        **{
            k: v
            for k, v in json.loads(
                (gold / "app_eval_settings.json").read_text(encoding="utf-8")
            ).items()
            if k
            in {
                "chosen_redact_entities",
                "chosen_redact_comprehend_entities",
                "chosen_llm_entities",
                "deny_list",
                "allow_list",
                "text_extraction_method",
                "pii_identification_method",
                "page_min",
                "page_max",
                "text_extraction_only",
                "handwrite_signature_checkbox",
                "custom_llm_instructions",
            }
        }
    )
    # Simulate UI state: LLM entities still selected while PII is Local.
    settings = AppEvalRedactionSettings(
        chosen_redact_entities=settings.chosen_redact_entities,
        chosen_redact_comprehend_entities=settings.chosen_redact_comprehend_entities,
        chosen_llm_entities=[
            "EMAIL_ADDRESS",
            "STREET_ADDRESS",
            "PERSON_NAME",
            "PHONE_NUMBER",
        ],
        deny_list=settings.deny_list,
        allow_list=settings.allow_list,
        text_extraction_method=settings.text_extraction_method,
        pii_identification_method="Local",
        page_min=settings.page_min,
        page_max=settings.page_max,
        text_extraction_only=settings.text_extraction_only,
    )
    gold_text = (gold / "instructions.txt").read_text(encoding="utf-8-sig").strip()
    assert derive_app_eval_instructions(settings).strip() == gold_text
