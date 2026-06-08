"""Resolve Pi agent session JSONL logs for Gradio download and usage-log persistence."""

from __future__ import annotations

import shutil
from pathlib import Path

from pi_agent_config import ensure_session_dir
from pi_rpc_client import PiRpcClient, PiRpcError

from tools.aws_functions import upload_log_file_to_s3
from tools.config import (
    RUN_AWS_FUNCTIONS,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    USAGE_LOGS_FOLDER,
)


def _session_dir_root() -> Path:
    return ensure_session_dir()


def pi_session_file_from_client(client: PiRpcClient | None) -> Path | None:
    """Return the active Pi session JSONL path from RPC state, if readable."""
    if client is None or not client.running:
        return None
    try:
        state = client.get_state()
    except PiRpcError:
        return None
    raw = state.get("sessionFile")
    if not raw or str(raw).strip() in ("", "—"):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        return None
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(_session_dir_root())
    except ValueError:
        return None
    return resolved


def _usage_log_archive_name(source: Path, session_hash: str = "") -> str:
    if session_hash and str(session_hash).strip():
        return f"{str(session_hash).strip()}_{source.name}"
    return source.name


def copy_session_log_to_usage_folder(
    source: Path,
    *,
    session_hash: str = "",
) -> Path | None:
    """Copy a Pi session JSONL into ``USAGE_LOGS_FOLDER`` (beside ``usage_log.csv``)."""
    if not SAVE_LOGS_TO_CSV:
        return None
    usage_dir = Path(USAGE_LOGS_FOLDER)
    usage_dir.mkdir(parents=True, exist_ok=True)
    dest = usage_dir / _usage_log_archive_name(source, session_hash)
    try:
        shutil.copy2(source, dest)
    except OSError:
        return None
    return dest.resolve()


def collect_session_log_download(client: PiRpcClient | None) -> str | None:
    """Path suitable for ``gr.File`` download, or ``None`` if no log yet."""
    path = pi_session_file_from_client(client)
    if path is None:
        return None
    return str(path)


def persist_session_log(
    client: PiRpcClient | None,
    *,
    session_hash: str = "",
    source: Path | None = None,
) -> Path | None:
    """
    Archive the active Pi session JSONL when local usage logging is enabled.

    Copies into ``USAGE_LOGS_FOLDER`` when ``SAVE_LOGS_TO_CSV`` is true, then
    uploads that copy to ``S3_USAGE_LOGS_FOLDER`` when ``RUN_AWS_FUNCTIONS`` is true.

    When *source* is provided (resolved synchronously by the caller), it is used
    directly so this can run on a background thread without issuing an RPC read.
    """
    if not SAVE_LOGS_TO_CSV:
        return None
    if source is None:
        source = pi_session_file_from_client(client)
    if source is None:
        return None
    archived = copy_session_log_to_usage_folder(source, session_hash=session_hash)
    if archived is None:
        return None
    if RUN_AWS_FUNCTIONS:
        upload_log_file_to_s3(str(archived), S3_USAGE_LOGS_FOLDER)
    return archived


def export_session_log_to_s3(client: PiRpcClient | None) -> None:
    """Back-compat: persist session log (local archive + optional S3)."""
    persist_session_log(client)


def gradio_session_log_allowed_paths() -> list[str]:
    """Directories Gradio must allow to serve Pi session JSONL files."""
    paths: list[str] = []
    try:
        paths.append(str(_session_dir_root()))
    except OSError:
        pass
    if SAVE_LOGS_TO_CSV:
        try:
            paths.append(str(Path(USAGE_LOGS_FOLDER).resolve()))
        except OSError:
            pass
    return paths
