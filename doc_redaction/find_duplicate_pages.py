"""
Public API wrappers for duplicate page detection functions.
"""

from __future__ import annotations

from tools.find_duplicate_pages import (
    run_duplicate_analysis,
    run_search_with_regex_option,
)

__all__ = ["run_duplicate_analysis", "run_search_with_regex_option"]
