"""Tests for loading config/app_config.env into Express container environment."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_functions import load_app_config_env_for_express


def test_load_app_config_env_for_express_reads_config_env():
    config_path = CDK_DIR / "config" / "app_config.env"
    env_vars = load_app_config_env_for_express(str(config_path))
    names = {p.name for p in env_vars}
    assert "RUN_FASTAPI" in names
    assert "DOCUMENT_REDACTION_BUCKET" in names
    assert "AWS_CLIENT_ID" not in names


def test_load_app_config_env_missing_file_returns_empty():
    assert (
        load_app_config_env_for_express(str(CDK_DIR / "config" / "nonexistent.env"))
        == []
    )
