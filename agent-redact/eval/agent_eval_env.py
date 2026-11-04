"""Environment helpers for agent eval (gold case paths, auto-match toggles)."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

_WARNED_LEGACY: set[str] = set()


def _warn_legacy_once(old_name: str, new_name: str) -> None:
    if old_name in _WARNED_LEGACY:
        return
    _WARNED_LEGACY.add(old_name)
    msg = (
        f"{old_name} is deprecated; use {new_name}. " "See agent-redact/eval/README.md."
    )
    logger.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def _env_raw(name: str, *, legacy: str | None = None) -> str:
    val = (os.environ.get(name) or "").strip()
    if val:
        return val
    if legacy:
        leg = (os.environ.get(legacy) or "").strip()
        if leg:
            _warn_legacy_once(legacy, name)
            return leg
    return ""


def _env_truthy(name: str, *, legacy: str | None = None, default: bool = False) -> bool:
    raw = _env_raw(name, legacy=legacy)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def default_gold_root(repo_root: Path) -> Path:
    return (repo_root / "agent-redact" / "eval" / "gold").resolve()


def resolve_gold_root(repo_root: Path) -> Path:
    raw = _env_raw("AGENT_EVAL_GOLD_ROOT", legacy="AGENT_EVAL_FIXTURES_ROOT")
    if not raw:
        return default_gold_root(repo_root)
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / raw).resolve()
    return path


def resolve_env_gold_case_dir(repo_root: Path) -> Path | None:
    """Single gold case directory from env (legacy fixed scoring when auto-gold off)."""
    raw = _env_raw("AGENT_EVAL_GOLD", legacy="AGENT_EVAL_FIXTURE")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / raw).resolve()
    return path if path.is_dir() else None


def agent_eval_auto_gold_enabled() -> bool:
    return _env_truthy("AGENT_EVAL_AUTO_GOLD", legacy="AGENT_EVAL_AUTO_FIXTURE")


def gold_match_mode() -> str:
    raw = _env_raw(
        "AGENT_EVAL_GOLD_MATCH_MODE",
        legacy="AGENT_EVAL_FIXTURE_MATCH_MODE",
    )
    mode = (raw or "strict_then_soft").strip().lower()
    if mode in {"strict_only", "soft_only", "strict_then_soft"}:
        return mode
    return "strict_then_soft"
