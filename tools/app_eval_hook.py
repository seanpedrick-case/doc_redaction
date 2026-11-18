"""Hook main-app redaction runs into agent-redact/eval (APP_EVAL_ENABLED)."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from tools.config import APP_EVAL_ENABLED
from tools.file_redaction import choose_and_run_redactor

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parents[1] / "agent-redact" / "eval"


def _report_app_eval_to_console(message: str, *, failed: bool = False) -> None:
    """Always print app eval status (Gradio console); also log when logging is configured."""
    prefix = "App eval failed" if failed else "App eval"
    line = message if message.startswith(prefix) else f"{prefix}: {message}"
    print(line, flush=True)
    if failed:
        logger.warning("%s", line)
    else:
        logger.info("%s", line)


def _schedule_app_eval(flat: dict[str, Any]) -> None:
    if not APP_EVAL_ENABLED:
        return

    def _work() -> None:
        try:
            if str(_EVAL_DIR) not in sys.path:
                sys.path.insert(0, str(_EVAL_DIR))
            from derive_app_eval_instructions import (
                settings_from_legacy_redactor_kwargs,
            )
            from post_run_app_eval import write_app_post_run_eval

            settings = settings_from_legacy_redactor_kwargs(flat)
            pii = (settings.pii_identification_method or "").strip() or "unknown"
            _report_app_eval_to_console(
                f"post-run eval started (PII={pii})…",
            )
            file_paths = flat.get("file_paths")
            if not file_paths:
                return
            first = file_paths[0] if isinstance(file_paths, list) else file_paths
            pdf_path = Path(str(first))
            output_folder = str(flat.get("output_folder") or pdf_path.parent)
            search_root = Path(output_folder)
            doc_name = pdf_path.name
            _, status = write_app_post_run_eval(
                pdf_path=pdf_path,
                document_name=doc_name,
                settings=settings,
                search_root=search_root,
            )
            if status:
                _report_app_eval_to_console(status)
        except Exception as exc:
            logger.warning("App post-run eval failed", exc_info=True)
            _report_app_eval_to_console(str(exc), failed=True)

    threading.Thread(target=_work, daemon=True).start()


def choose_and_run_redactor_with_app_eval(*args: Any, **kwargs: Any):
    """``choose_and_run_redactor`` plus background gold eval when enabled."""
    flat = {"file_paths": args[0] if args else kwargs.get("file_paths")}
    if args:
        param_names = [
            "file_paths",
            "prepared_pdf_file_paths",
            "pdf_image_file_paths",
            "chosen_redact_entities",
            "chosen_redact_comprehend_entities",
            "chosen_llm_entities",
            "text_extraction_method",
            "in_allow_list",
            "in_deny_list",
            "redact_whole_page_list",
            "latest_file_completed",
            "combined_out_message",
            "out_file_paths",
            "log_files_output_paths",
            "page_min",
            "page_max",
            "estimated_time_taken_state",
            "handwrite_signature_checkbox",
            "all_request_metadata_str",
            "annotations_all_pages",
            "all_page_line_level_ocr_results_df",
            "all_pages_decision_process_table",
            "pymupdf_doc",
            "pii_identification_method",
            "max_fuzzy_spelling_mistakes_num",
            "match_fuzzy_whole_phrase_bool",
            "aws_access_key_textbox",
            "aws_secret_key_textbox",
            "annotate_max_pages",
            "review_file_state",
            "output_folder",
            "document_cropboxes",
            "page_sizes",
            "textract_output_found",
            "text_extraction_only",
            "duplication_file_path_outputs",
            "review_file_path",
            "input_folder",
            "ocr_file_path",
            "all_page_line_level_ocr_results",
            "all_page_line_level_ocr_results_with_words",
            "all_page_line_level_ocr_results_with_words_df",
            "chosen_local_ocr_model",
            "language",
            "ocr_review_files",
            "custom_llm_instructions",
            "inference_server_vlm_model",
            "efficient_ocr",
            "efficient_ocr_min_words",
            "efficient_ocr_min_image_coverage_fraction",
            "efficient_ocr_min_embedded_image_px",
            "hybrid_textract_bedrock_vlm",
            "overwrite_existing_ocr_results",
            "save_page_ocr_visualisations",
            "ocr_first_pass_max_workers",
            "prepare_images",
            "RETURN_REDACTED_PDF",
            "RETURN_PDF_FOR_REVIEW",
            "post_redact_pass1_qa",
            "post_redact_pass1_auto_prune",
        ]
        for idx, name in enumerate(param_names):
            if idx < len(args):
                flat[name] = args[idx]
    flat.update(kwargs)
    if "file_paths" not in flat and args:
        flat["file_paths"] = args[0]

    try:
        return choose_and_run_redactor(*args, **kwargs)
    finally:
        _schedule_app_eval(flat)


def _load_list_file(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    import pandas as pd

    try:
        df = pd.read_csv(p, header=None)
        if df.empty:
            return []
        return [str(x).strip() for x in df.iloc[:, 0].dropna() if str(x).strip()]
    except Exception:
        return []


def run_cli_app_eval_if_requested(args: Any) -> None:
    """Run gold eval after CLI redaction when ``--app-eval`` or ``APP_EVAL_ENABLED``."""
    use_eval = bool(getattr(args, "app_eval", False)) or APP_EVAL_ENABLED
    if not use_eval:
        return
    input_files = getattr(args, "input_file", None) or []
    if not input_files:
        return
    pdf_path = Path(str(input_files[0]))
    output_dir = Path(str(getattr(args, "output_dir", pdf_path.parent)))

    try:
        if str(_EVAL_DIR) not in sys.path:
            sys.path.insert(0, str(_EVAL_DIR))
        from derive_app_eval_instructions import AppEvalRedactionSettings
        from post_run_app_eval import write_app_post_run_eval

        entities = getattr(args, "redact_entities", None) or []
        if isinstance(entities, str):
            entities = [e.strip() for e in entities.split(",") if e.strip()]
        settings = AppEvalRedactionSettings(
            chosen_redact_entities=list(entities),
            chosen_redact_comprehend_entities=list(
                getattr(args, "comprehend_entities", None) or []
            ),
            chosen_llm_entities=list(getattr(args, "llm_entities", None) or []),
            deny_list=_load_list_file(getattr(args, "deny_list_file", None)),
            allow_list=_load_list_file(getattr(args, "allow_list_file", None)),
            text_extraction_method=str(getattr(args, "ocr_method", "") or ""),
            pii_identification_method=str(getattr(args, "pii_detector", "") or ""),
            page_min=int(getattr(args, "page_min", 0) or 0),
            page_max=int(getattr(args, "page_max", 0) or 0),
            text_extraction_only=bool(getattr(args, "text_extraction_only", False)),
            handwrite_signature_checkbox=list(
                getattr(args, "handwrite_signature_extraction", None) or []
            ),
            custom_llm_instructions=str(
                getattr(args, "custom_llm_instructions", "") or ""
            ),
        )
        _, status = write_app_post_run_eval(
            pdf_path=pdf_path,
            document_name=pdf_path.name,
            settings=settings,
            search_root=output_dir,
        )
        if status:
            print(f"\n--- App eval ---\n{status}")
    except Exception as exc:
        print(f"Warning: app eval failed: {exc}")
