"""Benchmark stack overrides (OCR / PII / local OCR model / app config env)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from derive_app_eval_instructions import AppEvalRedactionSettings


@dataclass(frozen=True)
class BenchmarkStackOverrides:
    """CLI/env overrides applied on top of gold ``app_eval_settings.json``."""

    text_extraction_method: str | None = None
    pii_identification_method: str | None = None
    pii_llm_model: str | None = None
    chosen_local_ocr_model: str | None = None
    chosen_ocr_model: str | None = None
    app_config_path: str | None = None


def _local_ocr_text_extract_option() -> str:
    """The text-extraction option required for a local OCR engine override."""
    try:
        from tools.config import LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION

        return LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION
    except ImportError:
        return "Local OCR model - PDFs without selectable text"


def apply_benchmark_overrides(
    settings: AppEvalRedactionSettings,
    overrides: BenchmarkStackOverrides | None,
) -> AppEvalRedactionSettings:
    if overrides is None:
        return settings
    patch: dict[str, str] = {}
    if overrides.text_extraction_method:
        patch["text_extraction_method"] = overrides.text_extraction_method.strip()
    if overrides.pii_identification_method:
        patch["pii_identification_method"] = overrides.pii_identification_method.strip()
    if overrides.pii_llm_model:
        patch["pii_llm_model"] = overrides.pii_llm_model.strip()
    if overrides.chosen_local_ocr_model:
        patch["chosen_local_ocr_model"] = overrides.chosen_local_ocr_model.strip()
        # A local OCR engine only takes effect under the local-OCR text extraction
        # method; when only the engine is overridden, switch the method too so the
        # override applies uniformly to every case.
        if not overrides.text_extraction_method:
            patch["text_extraction_method"] = _local_ocr_text_extract_option()
    if overrides.chosen_ocr_model:
        patch["chosen_ocr_model"] = overrides.chosen_ocr_model.strip()
    if overrides.app_config_path:
        patch["app_config_path"] = overrides.app_config_path.strip()
    if not patch:
        return settings
    return replace(settings, **patch)


def effective_ocr_method_label(settings: AppEvalRedactionSettings) -> str:
    """Resolved OCR label for manifests / CSV (includes local engine id when applicable)."""
    text = (settings.text_extraction_method or "").strip()
    local = (settings.chosen_local_ocr_model or "").strip()
    if not text:
        return local or "Local text"
    try:
        from tools.agentcore_harness import ui_ocr_method_to_tool_value
        from tools.config import LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION

        if text == LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION or (
            "local ocr" in text.casefold() and local
        ):
            return ui_ocr_method_to_tool_value(text, local)
    except ImportError:
        pass
    if local and "paddle" not in text.casefold() and "tesseract" not in text.casefold():
        return f"{text} ({local})"
    return text


def overrides_from_argv(
    *,
    ocr_method: str | None,
    pii_method: str | None,
    chosen_local_ocr_model: str | None,
    app_config_path: str | None,
    chosen_ocr_model: str | None = None,
    pii_llm_model: str | None = None,
    ocr_method_explicit: bool = True,
    local_ocr_model_explicit: bool = False,
) -> BenchmarkStackOverrides | None:
    if not any(
        (
            ocr_method,
            pii_method,
            pii_llm_model,
            chosen_local_ocr_model,
            chosen_ocr_model,
            app_config_path,
        )
    ):
        return None
    effective_ocr_method = ocr_method
    if chosen_local_ocr_model and local_ocr_model_explicit and not ocr_method_explicit:
        # Only the local OCR engine was chosen on the command line: map it back to
        # the local-OCR text-extraction method it requires, ignoring an env-default
        # --ocr-method value (e.g. DEFAULT_TEXT_EXTRACTION_MODEL from the app config).
        effective_ocr_method = _local_ocr_text_extract_option()
    return BenchmarkStackOverrides(
        text_extraction_method=effective_ocr_method,
        pii_identification_method=pii_method,
        pii_llm_model=pii_llm_model,
        chosen_local_ocr_model=chosen_local_ocr_model,
        chosen_ocr_model=chosen_ocr_model,
        app_config_path=app_config_path,
    )
