"""Generate Pi agent models.json and settings.json at runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def resolve_agent_dir() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))


# Back-compat alias; prefer resolve_agent_dir() when env may change after import.
AGENT_DIR = resolve_agent_dir()
TEMPLATE_DIR = Path(__file__).resolve().parent / "agent"
SETTINGS_TEMPLATE = TEMPLATE_DIR / "settings.json"

DEPLOYMENT_LOCAL = "local-docker"
DEPLOYMENT_HF_SPACE = "hf-space"
DEPLOYMENT_PROFILE = (
    os.environ.get("PI_DEPLOYMENT_PROFILE", DEPLOYMENT_LOCAL).strip().lower()
)


def pi_max_retries() -> int:
    """Max retries for Pi auto-retry and Gradio quota backoff (env: PI_MAX_RETRIES, default 5)."""
    raw = (
        os.environ.get("PI_QUOTA_RETRY_ATTEMPTS")
        or os.environ.get("PI_MAX_RETRIES")
        or "5"
    ).strip()
    return int(raw)


def _apply_retry_settings(
    settings: dict[str, Any],
    *,
    provider: str,
) -> None:
    """Write Pi ``settings.json`` retry block (Gemini uses longer delays)."""
    max_retries = pi_max_retries()
    gemini_delays = provider == PROVIDER_GEMINI or is_hf_space_profile()
    base_delay_ms = 2000
    max_delay_ms = 60000
    if gemini_delays:
        base_delay_ms = int(os.environ.get("PI_GEMINI_RETRY_BASE_DELAY_MS", "60000"))
        max_delay_ms = int(os.environ.get("PI_GEMINI_RETRY_MAX_DELAY_MS", "90000"))
    settings["retry"] = {
        "enabled": True,
        "maxRetries": max_retries,
        "baseDelayMs": base_delay_ms,
        "provider": {
            "timeoutMs": 3600000,
            "maxRetries": max_retries,
            "maxRetryDelayMs": max_delay_ms,
        },
    }


PROVIDER_LLAMA = "llama-cpp"
PROVIDER_GEMINI = "google-gemini"
PROVIDER_BEDROCK = "amazon-bedrock"

PROVIDER_LABELS: dict[str, str] = {
    PROVIDER_LLAMA: "Local (llama-cpp)",
    PROVIDER_GEMINI: "Gemini",
    PROVIDER_BEDROCK: "AWS Bedrock",
}


def is_hf_space_profile() -> bool:
    profile = os.environ.get("PI_DEPLOYMENT_PROFILE", DEPLOYMENT_LOCAL).strip().lower()
    return profile == DEPLOYMENT_HF_SPACE


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


def get_default_provider() -> str:
    """Current default Pi provider (reads ``PI_DEFAULT_PROVIDER`` from env each call)."""
    if is_hf_space_profile():
        return PROVIDER_GEMINI
    raw = (os.environ.get("PI_DEFAULT_PROVIDER") or PROVIDER_LLAMA).strip()
    if raw in PROVIDER_MODELS:
        return raw
    return PROVIDER_LLAMA


DEFAULT_PROVIDER = get_default_provider()

_env_default_model = (os.environ.get("PI_DEFAULT_MODEL") or "").strip()
DEFAULT_MODEL = _env_default_model or DEFAULT_MODEL_BY_PROVIDER.get(
    DEFAULT_PROVIDER, LLAMA_MODEL_ID
)


def resolved_default_model(provider: str, *, override: str | None = None) -> str:
    """
    Pick the default model id for a provider.

    Order: explicit override → ``PI_DEFAULT_MODEL`` (if listed for provider) →
    built-in per-provider default.
    """
    models = PROVIDER_MODELS.get(provider, [])
    if override and override in models:
        return override
    env_model = (os.environ.get("PI_DEFAULT_MODEL") or DEFAULT_MODEL or "").strip()
    if env_model and env_model in models:
        return env_model
    return DEFAULT_MODEL_BY_PROVIDER.get(provider, LLAMA_MODEL_ID)


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


_AWS_CREDENTIAL_ENV_KEYS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ACCESS_KEY",
    "AWS_SECRET_KEY",
)
_AWS_PROFILE_ENV_KEYS: tuple[str, ...] = ("AWS_PROFILE", "PI_AWS_PROFILE")


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_empty_env_vars(names: tuple[str, ...]) -> None:
    for name in names:
        if not (os.environ.get(name) or "").strip():
            os.environ.pop(name, None)


def _mirror_legacy_aws_key_env_vars() -> None:
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip():
        legacy = (os.environ.get("AWS_ACCESS_KEY") or "").strip()
        if legacy:
            os.environ["AWS_ACCESS_KEY_ID"] = legacy
    if not (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip():
        legacy = (os.environ.get("AWS_SECRET_KEY") or "").strip()
        if legacy:
            os.environ["AWS_SECRET_ACCESS_KEY"] = legacy


def _has_explicit_aws_access_keys() -> bool:
    access = (
        os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY") or ""
    ).strip()
    secret = (
        os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_KEY")
        or ""
    ).strip()
    return bool(access and secret)


def _aws_config_path() -> Path | None:
    explicit = (os.environ.get("AWS_CONFIG_FILE") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    home = Path(os.environ.get("HOME", "/home/node"))
    path = home / ".aws" / "config"
    return path if path.is_file() else None


def _discover_aws_profile_from_config() -> str | None:
    """Return an AWS profile name for Pi/Bedrock when only ~/.aws is mounted."""
    explicit = (os.environ.get("PI_AWS_PROFILE") or "").strip()
    if not explicit:
        explicit = (os.environ.get("AWS_PROFILE") or "").strip()
    if explicit:
        return explicit

    path = _aws_config_path()
    if not path:
        return None

    current_profile: str | None = None
    sso_profiles: list[str] = []
    all_profiles: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line == "[default]":
            current_profile = "default"
            all_profiles.append("default")
            continue
        if line.startswith("[profile ") and line.endswith("]"):
            current_profile = line[len("[profile ") : -1].strip()
            if current_profile:
                all_profiles.append(current_profile)
            continue
        if current_profile and line.startswith("sso_session"):
            sso_profiles.append(current_profile)

    if sso_profiles:
        return sso_profiles[0]
    if "default" in all_profiles:
        return "default"
    return all_profiles[0] if all_profiles else None


def _region_from_aws_config(profile: str | None = None) -> str | None:
    """Read ``region =`` from a profile block in ``~/.aws/config``."""
    path = _aws_config_path()
    if not path:
        return None

    target = (profile or _discover_aws_profile_from_config() or "").strip()
    if not target:
        return None

    current_profile: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line == "[default]":
            current_profile = "default"
            continue
        if line.startswith("[profile ") and line.endswith("]"):
            current_profile = line[len("[profile ") : -1].strip()
            continue
        if current_profile != target:
            continue
        if line.startswith("region"):
            _, _, value = line.partition("=")
            region = value.strip()
            if region:
                return region
    return None


def _ensure_aws_region_env() -> None:
    """Ensure AWS SDK env has a non-empty region (profile config, then eu-west-2)."""
    _strip_empty_env_vars(("AWS_REGION", "AWS_DEFAULT_REGION"))
    region = (
        os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    ).strip()
    if not region:
        profile = (os.environ.get("AWS_PROFILE") or "").strip()
        region = (_region_from_aws_config(profile) or "").strip()
    if not region:
        region = _bedrock_region()
    os.environ["AWS_REGION"] = region
    os.environ["AWS_DEFAULT_REGION"] = region


def _pi_bedrock_auth_visible() -> bool:
    """True when Pi's amazon-bedrock provider would detect configured auth."""
    if (os.environ.get("AWS_PROFILE") or "").strip():
        return True
    if _has_explicit_aws_access_keys():
        return True
    if (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip():
        return True
    return False


def _ensure_pi_bedrock_auth_env() -> None:
    """
    Pi checks env vars (not ~/.aws alone) before Bedrock is usable.

    When SSO credentials live in a mounted ``~/.aws`` tree, set ``AWS_PROFILE``
    so Pi passes its auth preflight and the AWS SDK loads the profile.
    """
    if _pi_bedrock_auth_visible():
        return
    profile = _discover_aws_profile_from_config()
    if profile:
        os.environ["AWS_PROFILE"] = profile


def configure_aws_credentials(
    *,
    session_access_key_id: str | None = None,
    session_secret_access_key: str | None = None,
    session_session_token: str | None = None,
) -> None:
    """
    Align Pi Bedrock AWS env with doc_redaction SSO/key priority.

    Mirrors ``tools/file_redaction.py``: when ``RUN_AWS_FUNCTIONS`` is enabled,
    prefer the default credential chain (SSO profile, instance role, etc.) over
    static env keys when ``PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS`` is true.
    Explicit UI session keys from **Apply backend** always win.
    """
    _strip_empty_env_vars(_AWS_CREDENTIAL_ENV_KEYS)
    _strip_empty_env_vars(_AWS_PROFILE_ENV_KEYS)
    _mirror_legacy_aws_key_env_vars()

    session_explicit = bool(
        session_access_key_id
        and session_access_key_id.strip()
        and session_secret_access_key
        and session_secret_access_key.strip()
    )
    if session_explicit:
        os.environ["AWS_ACCESS_KEY_ID"] = session_access_key_id.strip()
        os.environ["AWS_SECRET_ACCESS_KEY"] = session_secret_access_key.strip()
        if session_session_token and session_session_token.strip():
            os.environ["AWS_SESSION_TOKEN"] = session_session_token.strip()
        else:
            os.environ.pop("AWS_SESSION_TOKEN", None)
        _ensure_aws_region_env()
        return

    run_aws = _env_flag("RUN_AWS_FUNCTIONS")
    prioritise_sso = _env_flag("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", default=True)

    if run_aws and prioritise_sso:
        for key in _AWS_CREDENTIAL_ENV_KEYS:
            os.environ.pop(key, None)
        _ensure_pi_bedrock_auth_env()
    elif run_aws:
        for key in _AWS_CREDENTIAL_ENV_KEYS:
            os.environ.pop(key, None)
        _ensure_pi_bedrock_auth_env()

    # Propagate PI_AWS_PROFILE when only that alias is set (e.g. pi_agent.env).
    pi_profile = (os.environ.get("PI_AWS_PROFILE") or "").strip()
    if pi_profile and not (os.environ.get("AWS_PROFILE") or "").strip():
        os.environ["AWS_PROFILE"] = pi_profile

    _ensure_aws_region_env()


def _aws_credential_status() -> str:
    if _has_explicit_aws_access_keys():
        return "access keys"
    profile = (os.environ.get("AWS_PROFILE") or "").strip()
    if profile:
        return f"profile `{profile}`"
    if (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip():
        return "Bedrock bearer token"
    if _aws_config_path():
        return "SSO config mounted (profile not set)"
    if _env_flag("RUN_AWS_FUNCTIONS"):
        return "SSO/default chain (missing profile)"
    return "missing"


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


def _apply_compaction_settings(settings: dict[str, Any]) -> None:
    """
    Merge Pi session auto-compaction from env into ``settings.json``.

    ``PI_COMPACTION_ENABLED`` — when set, overrides the template ``compaction.enabled``
    flag (``true`` / ``false``). When unset, the template default applies (enabled).

    Optional tuning: ``PI_COMPACTION_RESERVE_TOKENS``, ``PI_COMPACTION_KEEP_RECENT_TOKENS``.
    """
    compaction = dict(
        settings.get("compaction")
        or {
            "enabled": True,
            "reserveTokens": 32768,
            "keepRecentTokens": 20000,
        }
    )
    if os.environ.get("PI_COMPACTION_ENABLED") is not None:
        compaction["enabled"] = _env_flag("PI_COMPACTION_ENABLED")
    reserve = (os.environ.get("PI_COMPACTION_RESERVE_TOKENS") or "").strip()
    if reserve:
        compaction["reserveTokens"] = int(reserve)
    keep = (os.environ.get("PI_COMPACTION_KEEP_RECENT_TOKENS") or "").strip()
    if keep:
        compaction["keepRecentTokens"] = int(keep)
    settings["compaction"] = compaction


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
        path = (resolve_agent_dir() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_settings_config(
    *,
    default_provider: str | None = None,
    default_model: str | None = None,
) -> dict[str, Any]:
    provider = default_provider or get_default_provider()
    if provider not in PROVIDER_MODELS:
        provider = PROVIDER_GEMINI if is_hf_space_profile() else PROVIDER_LLAMA
    model = resolved_default_model(provider, override=default_model)

    settings = _load_settings_template()
    settings["defaultProvider"] = provider
    settings["defaultModel"] = model
    _apply_compaction_settings(settings)
    session_path = ensure_session_dir(resolve_session_dir())
    settings["sessionDir"] = session_path.as_posix()
    if is_hf_space_profile() or provider == PROVIDER_GEMINI:
        _apply_retry_settings(settings, provider=provider)
    from pi_workspace_skills import ensure_workspace_skills, workspace_skills_dir

    ensure_workspace_skills()
    settings["skills"] = [workspace_skills_dir().as_posix()]
    return settings


def write_runtime_config(
    *,
    agent_dir: Path | None = None,
    default_provider: str | None = None,
    default_model: str | None = None,
) -> tuple[Path, Path]:
    """Write models.json and settings.json; return their paths."""
    target = Path(agent_dir or resolve_agent_dir())
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
    return resolved_default_model(provider)


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
    configure_aws_credentials(
        session_access_key_id=aws_access_key_id,
        session_secret_access_key=aws_secret_access_key,
        session_session_token=aws_session_token,
    )


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
        region = _bedrock_region()
        lines.append(f"AWS `{_aws_credential_status()}` · region `{region}`")
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
    configure_aws_credentials()
    models_path, settings_path = write_runtime_config()
    print(f"Wrote {models_path}")
    print(f"Wrote {settings_path}")
