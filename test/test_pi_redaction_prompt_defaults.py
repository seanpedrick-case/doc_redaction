"""Tests for Pi redaction OCR/PII default resolution from environment."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

import pi_agent_config
import redaction_prompt as rp


def _reload_redaction_prompt(monkeypatch, *, profile: str = "local-docker"):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", profile)
    importlib.reload(pi_agent_config)
    return importlib.reload(rp)


def test_default_ocr_and_pii_from_pi_agent_env(monkeypatch):
    monkeypatch.setenv("PI_DEFAULT_OCR_METHOD", "hybrid-paddle-vlm")
    monkeypatch.setenv("PI_DEFAULT_PII_METHOD", "LLM (AWS Bedrock)")
    module = _reload_redaction_prompt(monkeypatch)
    assert module.DEFAULT_OCR_METHOD == "hybrid-paddle-vlm"
    assert module.DEFAULT_PII_METHOD == "LLM (AWS Bedrock)"


def test_local_fallback_when_env_unset(monkeypatch):
    monkeypatch.delenv("PI_DEFAULT_OCR_METHOD", raising=False)
    monkeypatch.delenv("PI_DEFAULT_PII_METHOD", raising=False)
    module = _reload_redaction_prompt(monkeypatch)
    assert module.DEFAULT_OCR_METHOD == "hybrid-paddle-inference-server"
    assert module.DEFAULT_PII_METHOD == "Local"


def test_hf_space_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("PI_DEFAULT_OCR_METHOD", raising=False)
    monkeypatch.delenv("PI_DEFAULT_PII_METHOD", raising=False)
    module = _reload_redaction_prompt(monkeypatch, profile="hf-space")
    assert module.DEFAULT_OCR_METHOD == module.HF_DEFAULT_OCR
    assert module.DEFAULT_PII_METHOD == module.HF_DEFAULT_PII


def test_build_redaction_prompt_omits_long_document_rules_for_small_pdfs(monkeypatch):
    module = _reload_redaction_prompt(monkeypatch)
    prompt = module.build_redaction_prompt(
        "short.pdf",
        "- Redact names",
        total_pages=5,
        workspace_dir=Path("/workspace"),
    )
    assert "Specific rules for long documents" not in prompt
    assert "User redaction requirements" in prompt


def test_aws_ecs_remote_guidance_forbids_workspace_output_grep(monkeypatch):
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "http://redaction:7860")
    module = _reload_redaction_prompt(monkeypatch, profile="aws-ecs")
    guidance = module.build_remote_backend_guidance(
        gradio_url="http://redaction:7860",
        output_base="/home/user/app/workspace/sess/redact/doc.pdf/",
        workspace_root="/home/user/app/workspace/sess",
    )
    assert "Split-container" in guidance
    assert "Do not" in guidance and "find /workspace" in guidance
    assert "/home/user/app/workspace/sess/redact/doc.pdf/output_redact/" in guidance
    assert ".pi/helpers/remote_redaction.py" in guidance


def test_build_redaction_prompt_keeps_long_document_rules_at_scale(monkeypatch):
    module = _reload_redaction_prompt(monkeypatch)
    prompt = module.build_redaction_prompt(
        "big.pdf",
        "- Redact names",
        total_pages=120,
        workspace_dir=Path("/workspace"),
    )
    assert "Specific rules for long documents" in prompt
