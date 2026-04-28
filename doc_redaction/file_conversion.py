"""
Public API wrappers for conversion/prepare functions.

These are used by Gradio endpoints in `app.py` (via `api_name` handlers).
"""

from __future__ import annotations

from tools.file_conversion import (
    combine_review_pdf_files,
    prepare_image_or_pdf_with_efficient_ocr,
)

__all__ = ["combine_review_pdf_files", "prepare_image_or_pdf_with_efficient_ocr"]
