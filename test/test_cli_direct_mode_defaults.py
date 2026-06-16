"""Direct-mode CLI argument merging (headless / Lambda / app.py RUN_DIRECT_MODE)."""

from __future__ import annotations

from cli_redact import get_cli_default_args_dict


def test_cli_defaults_include_s3_output_flags():
    defaults = get_cli_default_args_dict()
    assert "save_outputs_to_s3" in defaults
    assert "s3_outputs_folder" in defaults
    assert "s3_outputs_bucket" in defaults


def test_partial_direct_mode_merge_keeps_defaults():
    partial = {"task": "redact", "input_file": "doc.pdf"}
    merged = {**get_cli_default_args_dict(), **partial}
    assert merged["task"] == "redact"
    assert merged["input_file"] == "doc.pdf"
    assert "save_outputs_to_s3" in merged
