"""Test package: fast pytest tests in ``test_*.py``; heavy CLI epilog suite in ``cli_epilog_suite``."""

from .cli_epilog_suite import run_all_tests, run_cli_redact

__all__ = ["run_all_tests", "run_cli_redact"]
