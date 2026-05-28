"""Gradio client helpers for remote doc_redaction HF Space backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from gradio_client import Client

DEFAULT_BASE_URL = "https://seanpedrickcase-document-redaction.hf.space"
DEFAULT_CONNECT_TIMEOUT = 120.0
DEFAULT_READ_TIMEOUT = 1800.0


def redaction_base_url() -> str:
    return os.environ.get("DOC_REDACTION_GRADIO_URL", DEFAULT_BASE_URL).rstrip("/")


def redaction_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("DOC_REDACTION_HF_TOKEN")
    return token.strip() if token and token.strip() else None


def httpx_timeout(
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT,
    read: float = DEFAULT_READ_TIMEOUT,
) -> httpx.Timeout:
    return httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)


def make_redaction_client(
    base_url: str | None = None,
    hf_token: str | None = None,
) -> Client:
    """Return a gradio_client for the remote doc_redaction Space."""
    url = (base_url or redaction_base_url()).rstrip("/")
    token = hf_token if hf_token is not None else redaction_hf_token()
    kwargs = {"httpx_kwargs": {"timeout": httpx_timeout()}}
    if token:
        return Client(url, hf_token=token, **kwargs)
    return Client(url, **kwargs)


def _collect_paths(value: Any, out: list[str]) -> None:
    if isinstance(value, str) and value.startswith("/"):
        out.append(value)
    elif isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path.startswith("/"):
            out.append(path)
        for item in value.values():
            _collect_paths(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_paths(item, out)


def extract_server_paths(result: Any) -> list[str]:
    """Walk a gradio_client predict result and collect server file paths."""
    paths: list[str] = []
    _collect_paths(result, paths)
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def download_gradio_files(
    paths: list[str],
    dest_dir: str | Path,
    *,
    base_url: str | None = None,
    hf_token: str | None = None,
) -> list[Path]:
    """Download server paths from a Gradio Space into dest_dir."""
    url = (base_url or redaction_base_url()).rstrip("/")
    token = hf_token if hf_token is not None else redaction_hf_token()
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    with httpx.Client(timeout=httpx_timeout(), headers=headers) as http:
        for path in paths:
            if not isinstance(path, str) or not path.startswith("/"):
                continue
            file_url = f"{url}/gradio_api/file={quote(path, safe='')}"
            local_path = dest / Path(path).name
            response = http.get(file_url)
            response.raise_for_status()
            local_path.write_bytes(response.content)
            downloaded.append(local_path)
    return downloaded
