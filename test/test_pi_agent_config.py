"""Tests for Pi runtime config (session directory)."""

import os
import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

import pi_agent_config as pac


def test_hf_profile_defaults_session_dir_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "hf-space")
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()
    assert Path(settings["sessionDir"]).resolve() == Path("/tmp/pi-sessions").resolve()
    assert Path(settings["sessionDir"]).is_dir()
    assert settings["retry"]["baseDelayMs"] == 60000
    assert settings["retry"]["maxRetries"] == 3


def test_pi_session_dir_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom-sessions"
    monkeypatch.setenv("PI_SESSION_DIR", str(custom))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()
    assert Path(settings["sessionDir"]) == custom.resolve()
    assert custom.is_dir()


def test_configure_aws_credentials_prioritises_sso_over_env_keys(monkeypatch):
    monkeypatch.setenv("RUN_AWS_FUNCTIONS", "True")
    monkeypatch.setenv("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", "True")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_PROFILE", "my-sso-profile")

    pac.configure_aws_credentials()

    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert os.environ["AWS_PROFILE"] == "my-sso-profile"


def test_configure_aws_credentials_discovers_sso_profile_from_aws_config(
    tmp_path, monkeypatch
):
    aws_dir = tmp_path / ".aws"
    aws_dir.mkdir()
    (aws_dir / "config").write_text(
        "[profile corp-sso]\n"
        "sso_session = corp\n"
        "sso_start_url = https://example.awsapps.com/start\n"
        "sso_region = eu-west-2\n"
        "sso_account_id = 123456789012\n"
        "sso_role_name = MyRole\n"
        "region = eu-west-2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RUN_AWS_FUNCTIONS", "True")
    monkeypatch.setenv("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", "True")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("PI_AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    pac.configure_aws_credentials()

    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert os.environ["AWS_PROFILE"] == "corp-sso"


def test_configure_aws_credentials_keeps_env_keys_without_run_aws(monkeypatch):
    monkeypatch.delenv("RUN_AWS_FUNCTIONS", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    pac.configure_aws_credentials()

    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"


def test_configure_aws_credentials_session_ui_keys_win(monkeypatch):
    monkeypatch.setenv("RUN_AWS_FUNCTIONS", "True")
    monkeypatch.setenv("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", "True")

    pac.configure_aws_credentials(
        session_access_key_id="AKIAUI",
        session_secret_access_key="ui-secret",
        session_session_token="token",
    )

    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAUI"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "ui-secret"
    assert os.environ["AWS_SESSION_TOKEN"] == "token"
