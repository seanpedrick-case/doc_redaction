"""
AWS Lambda entrypoint for packaging.

Re-exports the existing repo-root `lambda_entrypoint.py` handler so Lambda
handler strings can move to `doc_redaction.lambda_entrypoint.lambda_handler`
without changing behavior.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

_root_lambda = importlib.import_module("lambda_entrypoint")

lambda_handler = getattr(_root_lambda, "lambda_handler")

__all__ = ["lambda_handler"]

