"""Tests for Pi runtime config (session directory)."""

import os
import sys
from pathlib import Path

import pytest

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

import pi_agent_config as pac


@pytest.fixture
def pi_workspace(tmp_path, monkeypatch):
    """Writable workspace for build_settings_config (skills sync, session dir)."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(ws))
    return ws


def test_build_settings_config_uses_pi_default_model_for_bedrock(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "amazon-bedrock")
    monkeypatch.setenv("PI_DEFAULT_MODEL", "anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    import importlib

    importlib.reload(pac)

    settings = pac.build_settings_config()
    assert settings["defaultProvider"] == "amazon-bedrock"
    assert settings["defaultModel"] == "anthropic.claude-sonnet-4-6"
    assert pac.default_model_for_provider(pac.PROVIDER_BEDROCK) == (
        "anthropic.claude-sonnet-4-6"
    )
    assert pac.resolved_default_model(pac.PROVIDER_LLAMA) == pac.LLAMA_MODEL_ID


def test_aws_ecs_profile_agent_dir_under_tmp(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

    import importlib

    importlib.reload(pac)

    assert pac.resolve_agent_dir() == Path("/tmp/pi-agent")


def test_hf_profile_agent_dir_under_tmp(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "hf-space")
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

    import importlib

    importlib.reload(pac)

    assert pac.resolve_agent_dir() == Path("/tmp/pi-agent")


def test_hf_profile_defaults_session_dir_to_tmp(tmp_path, monkeypatch, pi_workspace):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "hf-space")
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()
    assert Path(settings["sessionDir"]).resolve() == Path("/tmp/pi-sessions").resolve()
    assert Path(settings["sessionDir"]).is_dir()
    assert settings["retry"]["baseDelayMs"] == 60000
    assert settings["retry"]["maxRetries"] == 5
    assert settings["retry"]["provider"]["maxRetries"] == 5


def test_gemini_provider_applies_retry_settings(tmp_path, monkeypatch, pi_workspace):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "google-gemini")
    monkeypatch.setenv("PI_MAX_RETRIES", "7")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config(default_provider="google-gemini")
    assert settings["retry"]["maxRetries"] == 7
    assert settings["retry"]["provider"]["maxRetries"] == 7


def test_bedrock_provider_applies_quota_retry_settings(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "amazon-bedrock")
    monkeypatch.setenv("PI_QUOTA_RETRY_DELAY_S", "45")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config(default_provider="amazon-bedrock")
    assert settings["retry"]["baseDelayMs"] == 45000
    assert settings["retry"]["provider"]["maxRetryDelayMs"] == 67500
    assert settings["retry"]["maxRetries"] == 5


def test_aws_ecs_profile_applies_bedrock_retry_settings(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    monkeypatch.setenv("PI_BEDROCK_RETRY_BASE_DELAY_MS", "55000")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()
    assert settings["defaultProvider"] == "amazon-bedrock"
    assert settings["retry"]["baseDelayMs"] == 55000
    assert settings["retry"]["maxRetries"] == 5


def test_pi_session_dir_override(tmp_path, monkeypatch, pi_workspace):
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


def test_configure_aws_credentials_strips_empty_profile_and_uses_pi_alias(
    monkeypatch,
):
    monkeypatch.setenv("RUN_AWS_FUNCTIONS", "True")
    monkeypatch.setenv("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", "True")
    monkeypatch.setenv("AWS_PROFILE", "")
    monkeypatch.setenv("PI_AWS_PROFILE", "bedrock-sso")

    pac.configure_aws_credentials()

    assert os.environ["AWS_PROFILE"] == "bedrock-sso"


def test_configure_aws_credentials_sets_region_from_profile_config(
    tmp_path, monkeypatch
):
    aws_dir = tmp_path / ".aws"
    aws_dir.mkdir()
    (aws_dir / "config").write_text(
        "[profile corp-sso]\n" "sso_session = corp\n" "region = eu-west-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RUN_AWS_FUNCTIONS", "True")
    monkeypatch.setenv("AWS_PROFILE", "corp-sso")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    pac.configure_aws_credentials()

    assert os.environ["AWS_REGION"] == "eu-west-1"
    assert os.environ["AWS_DEFAULT_REGION"] == "eu-west-1"


def test_configure_aws_credentials_defaults_region_when_unset(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("PI_AWS_PROFILE", raising=False)

    pac.configure_aws_credentials()

    assert os.environ["AWS_REGION"] == "eu-west-2"
    assert os.environ["AWS_DEFAULT_REGION"] == "eu-west-2"


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


def test_build_settings_config_compaction_enabled_from_env(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("PI_COMPACTION_RESERVE_TOKENS", "4096")
    monkeypatch.setenv("PI_COMPACTION_KEEP_RECENT_TOKENS", "2048")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()

    assert settings["compaction"]["enabled"] is True
    assert settings["compaction"]["reserveTokens"] == 4096
    assert settings["compaction"]["keepRecentTokens"] == 2048


def test_build_settings_config_compaction_disabled_from_env(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_COMPACTION_ENABLED", "false")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()

    assert settings["compaction"]["enabled"] is False


def test_resolve_llama_base_url_prefers_pi_llama_base_url(monkeypatch):
    monkeypatch.setenv("PI_LLAMA_BASE_URL", "http://192.168.0.220:8080/v1")
    monkeypatch.setenv("PI_LLAMA_MODE_BASE_URL", "http://ignored:9999")

    import importlib

    importlib.reload(pac)

    assert pac.resolve_llama_base_url() == "http://192.168.0.220:8080/v1"
    assert pac.LLAMA_BASE_URL == "http://192.168.0.220:8080/v1"


def test_resolve_llama_base_url_accepts_legacy_alias_and_appends_v1(monkeypatch):
    monkeypatch.delenv("PI_LLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("PI_LLAMA_MODE_BASE_URL", "http://192.168.0.220:8080")

    import importlib

    importlib.reload(pac)

    assert pac.resolve_llama_base_url() == "http://192.168.0.220:8080/v1"
    assert pac.LLAMA_BASE_URL == "http://192.168.0.220:8080/v1"


def test_build_settings_config_compaction_uses_template_when_env_unset(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.delenv("PI_COMPACTION_ENABLED", raising=False)
    monkeypatch.delenv("PI_COMPACTION_RESERVE_TOKENS", raising=False)
    monkeypatch.delenv("PI_COMPACTION_KEEP_RECENT_TOKENS", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    settings = pac.build_settings_config()

    assert settings["compaction"]["enabled"] is True
    assert settings["compaction"]["reserveTokens"] == 32768
    assert settings["compaction"]["keepRecentTokens"] == 20000


def test_credential_status_markdown_llama_shows_endpoint_not_aws(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "llama-cpp")
    monkeypatch.setenv("PI_LLAMA_BASE_URL", "http://192.168.0.220:8000/v1")
    monkeypatch.setenv("PI_AWS_PROFILE", "AWSAdministratorAccess-460501890304")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    import importlib

    importlib.reload(pac)

    text = pac.credential_status_markdown(provider="llama-cpp")
    assert "local llama-cpp" in text
    assert "192.168.0.220:8000/v1" in text
    assert "AWSAdministratorAccess" not in text
    assert "Gemini `" not in text


def test_credential_status_markdown_bedrock_shows_aws_profile(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    monkeypatch.setenv("AWS_PROFILE", "corp-sso")
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    # CI runners often inject AWS_ACCESS_KEY_* for deployment; profile must win in UI text.
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    text = pac.credential_status_markdown(provider="amazon-bedrock")
    assert "AWS `profile corp-sso`" in text
    assert "region `eu-west-2`" in text


def test_normalize_backend_model_accepts_custom_llama_id(monkeypatch):
    monkeypatch.setenv("PI_LLAMA_MODEL_ID", "unsloth/Qwen3.6-27B-MTP-GGUF")
    assert pac.normalize_backend_model("llama-cpp", "my-custom-swap-model") == (
        "my-custom-swap-model"
    )


def test_normalize_backend_model_rejects_unknown_gemini_id(monkeypatch):
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "google-gemini")
    assert pac.normalize_backend_model(
        "google-gemini", "not-a-real-gemini-model"
    ) == pac.default_model_for_provider(pac.PROVIDER_GEMINI)


def test_resolved_default_model_uses_runtime_pi_default_for_active_provider(
    monkeypatch,
):
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "llama-cpp")
    monkeypatch.setenv("PI_DEFAULT_MODEL", "swap-model-v2")
    assert pac.resolved_default_model(pac.PROVIDER_LLAMA) == "swap-model-v2"


def test_resolved_default_model_ignores_gemini_env_on_bedrock(monkeypatch):
    """Cross-profile PI_DEFAULT_MODEL must not apply to amazon-bedrock."""
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "amazon-bedrock")
    monkeypatch.setenv("PI_DEFAULT_MODEL", "gemini-flash-latest")

    assert pac.resolved_default_model(pac.PROVIDER_BEDROCK) == (
        "anthropic.claude-sonnet-4-6"
    )
    assert pac.default_model_for_provider(pac.PROVIDER_BEDROCK) == (
        "anthropic.claude-sonnet-4-6"
    )


def test_get_default_provider_aws_ecs_without_env_defaults_to_bedrock(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    monkeypatch.delenv("PI_DEFAULT_PROVIDER", raising=False)

    import importlib

    importlib.reload(pac)

    assert pac.get_default_provider() == pac.PROVIDER_BEDROCK


def test_resolved_default_model_honours_override_without_catalog_entry():
    assert (
        pac.resolved_default_model(pac.PROVIDER_LLAMA, override="another-local-model")
        == "another-local-model"
    )


def test_write_runtime_config_persists_custom_llama_model(
    tmp_path, monkeypatch, pi_workspace
):
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_DEFAULT_PROVIDER", "llama-cpp")
    monkeypatch.setenv("PI_LLAMA_MODEL_ID", "unsloth/Qwen3.6-27B-MTP-GGUF")

    pac.write_runtime_config(
        agent_dir=agent_dir,
        default_provider="llama-cpp",
        default_model="custom-llama-model",
    )

    assert os.environ["PI_DEFAULT_PROVIDER"] == "llama-cpp"
    assert os.environ["PI_DEFAULT_MODEL"] == "custom-llama-model"
    assert os.environ["PI_LLAMA_MODEL_ID"] == "custom-llama-model"
    assert pac.models_for_provider(pac.PROVIDER_LLAMA) == ["custom-llama-model"]

    import json

    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    llama_models = models["providers"]["llama-cpp"]["models"]
    assert llama_models[0]["id"] == "custom-llama-model"

    settings = json.loads((agent_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["defaultModel"] == "custom-llama-model"


def test_build_settings_config_compaction_scales_for_small_llama_context(
    tmp_path, monkeypatch, pi_workspace
):
    monkeypatch.setenv("PI_LLAMA_CONTEXT_WINDOW", "65536")
    monkeypatch.delenv("PI_COMPACTION_RESERVE_TOKENS", raising=False)
    monkeypatch.delenv("PI_COMPACTION_KEEP_RECENT_TOKENS", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    import importlib

    importlib.reload(pac)

    settings = pac.build_settings_config()

    assert settings["compaction"]["reserveTokens"] == 16384
    assert settings["compaction"]["keepRecentTokens"] == 12288
