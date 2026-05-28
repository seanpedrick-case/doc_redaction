"""Tests for Pi runtime config (session directory)."""

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
