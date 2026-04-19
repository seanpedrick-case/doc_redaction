"""CLI entrypoint smoke tests (fast). Optional integration tests are marked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_redact_help_exits_zero():
    """Ensure the installed entrypoint responds to --help."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli_redact.py"), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in (result.stdout + result.stderr).lower()


def test_cli_redact_module_help_exits_zero():
    """Same as --help via python -m (packaging smoke)."""
    result = subprocess.run(
        [sys.executable, "-m", "cli_redact", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_cli_redact_smoke_pdf_local_text_optional(tmp_path):
    """
    End-to-end redact on a small example PDF (Local text path), only when enabled.

    Set PYTEST_CLI_INTEGRATION=1 and ensure the repo is installed with dependencies.
    Skips by default to keep CI fast unless the env var is set.
    """
    if os.environ.get("PYTEST_CLI_INTEGRATION") != "1":
        pytest.skip("Set PYTEST_CLI_INTEGRATION=1 to run CLI PDF smoke test")

    pdf = REPO_ROOT / "example_data" / "graduate-job-example-cover-letter.pdf"
    if not pdf.is_file():
        pytest.skip(f"Example PDF not found: {pdf}")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli_redact.py"),
            "--task",
            "redact",
            "--input_file",
            str(pdf),
            "--output_dir",
            str(out_dir),
            "--input_dir",
            str(REPO_ROOT / "example_data"),
            "--ocr_method",
            "Local text",
            "--pii_detector",
            "Local",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
