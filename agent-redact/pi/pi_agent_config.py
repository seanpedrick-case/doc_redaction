"""Generate Pi agent models.json and settings.json at runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

AGENT_DIR = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
TEMPLATE_DIR = Path(__file__).resolve().parent / "agent"
SETTINGS_TEMPLATE = TEMPLATE_DIR / "settings.json"

DEPLOYMENT_LOCAL = "local-docker"
DEPLOYMENT_HF_SPACE = "hf-space"
DEPLOYMENT_PROFILE = (
    os.environ.get("PI_DEPLOYMENT_PROFILE", DEPLOYMENT_LOCAL).strip().lower()
)

PROVIDER_LLAMA = "llama-cpp"
PROVIDER_GEMINI = "google-gemini"
PROVIDER_BEDROCK = "amazon-bedrock"

PROVIDER_LABELS: dict[str, str] = {
    PROVIDER_LLAMA: "Local (llama-cpp)",
    PROVIDER_GEMINI: "Gemini",
    PROVIDER_BEDROCK: "AWS Bedrock",
}


def is_hf_space_profile() -> bool:
    return DEPLOYMENT_PROFILE == DEPLOYMENT_HF_SPACE


def _default_provider() -> str:
    if is_hf_space_profile():
        return PROVIDER_GEMINI
    return os.environ.get("PI_DEFAULT_PROVIDER", PROVIDER_LLAMA)


DEFAULT_PROVIDER = _default_provider()

LLAMA_BASE_URL = os.environ.get("PI_LLAMA_BASE_URL", "http://llama-inference:8080/v1")
LLAMA_MODEL_ID = os.environ.get("PI_LLAMA_MODEL_ID", "unsloth/Qwen3.6-27B-MTP-GGUF")
LLAMA_CONTEXT = int(os.environ.get("PI_LLAMA_CONTEXT_WINDOW", "114688"))
LLAMA_MAX_TOKENS = int(os.environ.get("PI_LLAMA_MAX_TOKENS", "32768"))

GEMINI_MODELS: tuple[tuple[str, str, int, bool], ...] = (
    ("gemini-flash-lite-latest", "Gemini Flash Lite", 1048576, False),
    ("gemini-flash-latest", "Gemini Flash", 1048576, True),
    ("gemini-pro-latest", "Gemini Pro", 1048576, True),
)

BEDROCK_MODELS: tuple[tuple[str, str, int, bool], ...] = (
    (
        "anthropic.claude-3-haiku-20240307-v1:0",
        "Claude 3 Haiku (Bedrock)",
        200000,
        False,
    ),
    (
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude 3.7 Sonnet (Bedrock)",
        200000,
        True,
    ),
    (
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude Sonnet 4.5 (Bedrock)",
        200000,
        True,
    ),
    ("anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6 (Bedrock)", 200000, True),
    ("amazon.nova-micro-v1:0", "Amazon Nova Micro (Bedrock)", 128000, False),
    ("amazon.nova-lite-v1:0", "Amazon Nova Lite (Bedrock)", 300000, False),
    ("amazon.nova-pro-v1:0", "Amazon Nova Pro (Bedrock)", 300000, False),
)

PROVIDER_MODELS: dict[str, list[str]] = {
    PROVIDER_LLAMA: [LLAMA_MODEL_ID],
    PROVIDER_GEMINI: [model_id for model_id, _, _, _ in GEMINI_MODELS],
    PROVIDER_BEDROCK: [model_id for model_id, _, _, _ in BEDROCK_MODELS],
}

DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    PROVIDER_LLAMA: LLAMA_MODEL_ID,
    PROVIDER_GEMINI: GEMINI_MODELS[0][0],  # Gemini Flash Lite
    PROVIDER_BEDROCK: "anthropic.claude-sonnet-4-6",
}

DEFAULT_MODEL = os.environ.get(
    "PI_DEFAULT_MODEL",
    DEFAULT_MODEL_BY_PROVIDER.get(DEFAULT_PROVIDER, LLAMA_MODEL_ID),
)


def _zero_cost() -> dict[str, int]:
    return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}


def _model_entry(
    model_id: str,
    name: str,
    *,
    context_window: int,
    max_tokens: int,
    reasoning: bool,
    image_input: bool = True,
) -> dict[str, Any]:
    inputs = ["text", "image"] if image_input else ["text"]
    return {
        "id": model_id,
        "name": name,
        "reasoning": reasoning,
        "input": inputs,
        "contextWindow": context_window,
        "maxTokens": max_tokens,
        "cost": _zero_cost(),
    }


def _llama_provider() -> dict[str, Any]:
    return {
        "baseUrl": LLAMA_BASE_URL,
        "api": "openai-completions",
        "apiKey": "llama-cpp",
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": False,
            "maxTokensField": "max_tokens",
        },
        "models": [
            _model_entry(
                LLAMA_MODEL_ID,
                "Qwen 3.6 27B (local)",
                context_window=LLAMA_CONTEXT,
                max_tokens=LLAMA_MAX_TOKENS,
                reasoning=False,
            )
        ],
    }


def _gemini_provider() -> dict[str, Any]:
    return {
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "api": "google-generative-ai",
        "apiKey": "GEMINI_API_KEY",
        "models": [
            _model_entry(
                model_id, name, context_window=ctx, max_tokens=8192, reasoning=reasoning
            )
            for model_id, name, ctx, reasoning in GEMINI_MODELS
        ],
    }


def _bedrock_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "eu-west-2"
    )


def _bedrock_provider() -> dict[str, Any]:
    region = _bedrock_region()
    return {
        "baseUrl": f"https://bedrock-runtime.{region}.amazonaws.com",
        "api": "bedrock-converse-stream",
        "models": [
            _model_entry(
                model_id,
                name,
                context_window=ctx,
                max_tokens=8192,
                reasoning=reasoning,
            )
            for model_id, name, ctx, reasoning in BEDROCK_MODELS
        ],
    }


def build_models_config() -> dict[str, Any]:
    if is_hf_space_profile():
        return {"providers": {PROVIDER_GEMINI: _gemini_provider()}}
    return {
        "providers": {
            PROVIDER_LLAMA: _llama_provider(),
            PROVIDER_GEMINI: _gemini_provider(),
            PROVIDER_BEDROCK: _bedrock_provider(),
        }
    }


def _load_settings_template() -> dict[str, Any]:
    if SETTINGS_TEMPLATE.is_file():
        return json.loads(SETTINGS_TEMPLATE.read_text(encoding="utf-8"))
    return {
        "defaultThinkingLevel": "off",
        "hideThinkingBlock": True,
        "compaction": {
            "enabled": True,
            "reserveTokens": 32768,
            "keepRecentTokens": 20000,
        },
        "enableSkillCommands": True,
        "sessionDir": "sessions",
    }


def resolve_session_dir() -> str:
    """Pi session JSONL directory (absolute path or relative to ``AGENT_DIR``)."""
    explicit = os.environ.get("PI_SESSION_DIR", "").strip()
    if explicit:
        return explicit
    if is_hf_space_profile():
        return "/tmp/pi-sessions"
    return "sessions"


def ensure_session_dir(session_dir: str | None = None) -> Path:
    """Create the Pi session directory and return its resolved absolute path."""
    raw = (session_dir or resolve_session_dir()).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = (AGENT_DIR / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_settings_config(
    *,
    default_provider: str | None = None,
    default_model: str | None = None,
) -> dict[str, Any]:
    provider = default_provider or DEFAULT_PROVIDER
    if provider not in PROVIDER_MODELS:
        provider = PROVIDER_GEMINI if is_hf_space_profile() else PROVIDER_LLAMA
    model = default_model or DEFAULT_MODEL_BY_PROVIDER.get(provider, DEFAULT_MODEL)
    if model not in PROVIDER_MODELS.get(provider, []):
        model = DEFAULT_MODEL_BY_PROVIDER[provider]

    settings = _load_settings_template()
    settings["defaultProvider"] = provider
    settings["defaultModel"] = model
    session_path = ensure_session_dir(resolve_session_dir())
    settings["sessionDir"] = session_path.as_posix()
    return settings


def write_runtime_config(
    *,
    agent_dir: Path | None = None,
    default_provider: str | None = None,
    default_model: str | None = None,
) -> tuple[Path, Path]:
    """Write models.json and settings.json; return their paths."""
    target = Path(agent_dir or AGENT_DIR)
    target.mkdir(parents=True, exist_ok=True)

    models_path = target / "models.json"
    settings_path = target / "settings.json"

    models_path.write_text(
        json.dumps(build_models_config(), indent=2) + "\n",
        encoding="utf-8",
    )
    settings_path.write_text(
        json.dumps(
            build_settings_config(
                default_provider=default_provider,
                default_model=default_model,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return models_path, settings_path


def models_for_provider(provider: str) -> list[str]:
    if is_hf_space_profile():
        return list(PROVIDER_MODELS[PROVIDER_GEMINI])
    return list(PROVIDER_MODELS.get(provider, PROVIDER_MODELS[PROVIDER_LLAMA]))


def default_model_for_provider(provider: str) -> str:
    return DEFAULT_MODEL_BY_PROVIDER.get(provider, DEFAULT_MODEL)


def normalize_provider(provider: str) -> str:
    label_map = {label.lower(): key for key, label in PROVIDER_LABELS.items()}
    lowered = (provider or "").strip().lower()
    if lowered in PROVIDER_MODELS:
        return lowered
    if lowered in label_map:
        return label_map[lowered]
    return PROVIDER_GEMINI if is_hf_space_profile() else PROVIDER_LLAMA


def apply_session_credentials(
    *,
    gemini_api_key: str | None = None,
    hf_token: str | None = None,
    aws_region: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
) -> None:
    """Apply session-only credential overrides to os.environ."""
    if gemini_api_key and gemini_api_key.strip():
        os.environ["GEMINI_API_KEY"] = gemini_api_key.strip()
    if hf_token and hf_token.strip():
        token = hf_token.strip()
        os.environ["HF_TOKEN"] = token
        os.environ["DOC_REDACTION_HF_TOKEN"] = token
    if aws_region and aws_region.strip():
        os.environ["AWS_REGION"] = aws_region.strip()
        os.environ["AWS_DEFAULT_REGION"] = aws_region.strip()
    if aws_access_key_id and aws_access_key_id.strip():
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id.strip()
    if aws_secret_access_key and aws_secret_access_key.strip():
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key.strip()
    if aws_session_token and aws_session_token.strip():
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token.strip()


def mirror_hf_token_from_env() -> None:
    """Mirror DOC_REDACTION_HF_TOKEN or Space secret HF_TOKEN for Pi subprocess."""
    if os.environ.get("HF_TOKEN"):
        return
    doc_token = os.environ.get("DOC_REDACTION_HF_TOKEN", "").strip()
    if doc_token:
        os.environ["HF_TOKEN"] = doc_token


def _hf_token_status() -> str:
    if os.environ.get("HF_TOKEN"):
        source = (
            "UI session" if os.environ.get("_HF_TOKEN_FROM_UI") else "env/Space secret"
        )
        return f"set ({source})"
    return "missing"


def credential_status_markdown() -> str:
    gemini = (
        "set"
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        else "missing"
    )
    lines = [f"**Credentials:** Gemini `{gemini}`"]
    if is_hf_space_profile():
        lines.append(f"HF token (redaction backend) `{_hf_token_status()}`")
    else:
        aws = (
            "set"
            if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
            else "profile/role"
        )
        region = _bedrock_region()
        lines.append(f"AWS `{aws}` · region `{region}`")
    return " · ".join(lines)


def provider_choices() -> list[str]:
    if is_hf_space_profile():
        return [PROVIDER_GEMINI]
    return list(PROVIDER_LABELS.keys())


def gemini_api_key_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


if __name__ == "__main__":
    models_path, settings_path = write_runtime_config()
    print(f"Wrote {models_path}")
    print(f"Wrote {settings_path}")
