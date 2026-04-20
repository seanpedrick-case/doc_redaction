"""
Public API wrapper for redaction functions.

This module exists so consumers can use package-qualified imports like:

    from doc_redaction.file_redaction import choose_and_run_redactor

Internally it delegates to the existing implementation in `tools.file_redaction`
to preserve backwards compatibility with the current repo layout.
"""

from __future__ import annotations

from tools.file_redaction import choose_and_run_redactor, run_redaction

__all__ = ["choose_and_run_redactor", "run_redaction"]
