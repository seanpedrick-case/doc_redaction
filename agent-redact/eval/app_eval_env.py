"""Environment helpers for main-app (Gradio) gold eval."""

from __future__ import annotations

import os
from pathlib import Path

from agent_eval_env import gold_match_mode, resolve_gold_root


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def app_eval_enabled() -> bool:
    try:
        from tools.config import APP_EVAL_ENABLED

        return bool(APP_EVAL_ENABLED)
    except ImportError:
        return _env_truthy("APP_EVAL_ENABLED", default=False)


def app_eval_auto_gold_enabled() -> bool:
    try:
        from tools.config import APP_EVAL_AUTO_GOLD

        return bool(APP_EVAL_AUTO_GOLD)
    except ImportError:
        return _env_truthy("APP_EVAL_AUTO_GOLD", default=True)


def app_eval_mirror_runs_enabled() -> bool:
    raw = (os.environ.get("APP_EVAL_MIRROR_RUNS") or "").strip()
    if raw:
        return _env_truthy("APP_EVAL_MIRROR_RUNS")
    return _env_truthy("AGENT_EVAL_MIRROR_RUNS", default=False)


def resolve_app_eval_gold_root(repo_root: Path) -> Path:
    raw = (os.environ.get("APP_EVAL_GOLD_ROOT") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = (repo_root / raw).resolve()
        return path
    return resolve_gold_root(repo_root)


def resolve_fixed_app_eval_gold(repo_root: Path) -> Path | None:
    raw = (os.environ.get("APP_EVAL_GOLD") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / raw).resolve()
    return path if path.is_dir() else None


def app_eval_run_summary_csv_enabled() -> bool:
    if not app_eval_enabled():
        return False
    raw = (os.environ.get("APP_EVAL_RUN_SUMMARY_CSV") or "").strip()
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    return True


def app_eval_task_completion_affects_overall() -> bool:
    """When false (default), ``overall_pass`` ignores DeepEval task_completion."""
    raw = (os.environ.get("APP_EVAL_TASK_COMPLETION_AFFECTS_OVERALL") or "").strip()
    if not raw:
        return False
    return _env_truthy("APP_EVAL_TASK_COMPLETION_AFFECTS_OVERALL")


def app_eval_deepeval_enabled() -> bool:
    if not app_eval_enabled():
        return False
    try:
        from eval.arize_monitoring import agent_eval_deepeval_tracing_enabled
    except ImportError:
        from arize_monitoring import agent_eval_deepeval_tracing_enabled
    return agent_eval_deepeval_tracing_enabled()


def app_gold_match_mode() -> str:
    raw = (os.environ.get("APP_EVAL_GOLD_MATCH_MODE") or "").strip()
    if raw:
        return raw.lower()
    return gold_match_mode()
