"""Tests for Pi agent PDF page limit validation."""

import sys
from pathlib import Path
from types import ModuleType

import pymupdf
import pytest

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

import redaction_prompt as rp


@pytest.fixture
def tiny_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def test_max_pages_limit_prefers_pi_env(monkeypatch):
    monkeypatch.setenv("PI_MAX_PAGES", "42")
    monkeypatch.setenv("MAX_DOC_PAGES", "999")
    assert rp.max_pages_limit() == 42


def test_pages_to_process_count_all(tiny_pdf):
    assert rp.pages_to_process_count("all", 5) == 5
    assert rp.pages_to_process_count("", 5) == 5


def test_pages_to_process_count_range(tiny_pdf):
    assert rp.pages_to_process_count("2-4", 5) == 3
    assert rp.pages_to_process_count("5", 5) == 1


def test_validate_pdf_page_limit_allows_within_limit(tiny_pdf):
    rp.validate_pdf_page_limit(tiny_pdf, page_range="all", max_pages=5)


def test_validate_pdf_page_limit_rejects_over_limit(tiny_pdf):
    with pytest.raises(ValueError, match="exceeds the maximum allowed"):
        rp.validate_pdf_page_limit(tiny_pdf, page_range="all", max_pages=4)


def test_validate_pdf_page_limit_allows_subset_on_large_doc(tiny_pdf):
    rp.validate_pdf_page_limit(tiny_pdf, page_range="1-2", max_pages=2)


def test_validate_pdf_page_limit_skips_non_pdf(tmp_path):
    docx = tmp_path / "notes.docx"
    docx.write_bytes(b"not a pdf")
    rp.validate_pdf_page_limit(docx, page_range="all", max_pages=1)
