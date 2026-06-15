"""Gradio client helpers for remote doc_redaction HF Space backends."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from gradio_client import Client

DEFAULT_CONNECT_TIMEOUT = 120.0
DEFAULT_READ_TIMEOUT = 1800.0
_DEFAULT_REDACT_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "STREETNAME",
    "UKPOSTCODE",
    "TITLES",
    "CUSTOM",
)

_CLIENT_CACHE: dict[tuple[str, str], Client] = {}


def split_redaction_backend() -> bool:
    """True when Pi and doc_redaction do not share a filesystem (ECS, HF Space, …)."""
    try:
        from pi_agent_config import uses_split_redaction_backend

        return uses_split_redaction_backend()
    except ImportError:
        return False


def redaction_base_url() -> str:
    raw = (os.environ.get("DOC_REDACTION_GRADIO_URL") or "").strip().rstrip("/")
    if raw:
        return raw
    try:
        from redaction_prompt import doc_redaction_gradio_url

        return doc_redaction_gradio_url()
    except ImportError:
        return "http://127.0.0.1:7860"


def redaction_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("DOC_REDACTION_HF_TOKEN")
    return token.strip() if token and token.strip() else None


def redaction_gradio_auth() -> tuple[str, str] | None:
    """
    Optional Gradio HTTP basic auth for doc_redaction when ``COGNITO_AUTH=True``.

    Set ``DOC_REDACTION_GRADIO_AUTH_USER`` and ``DOC_REDACTION_GRADIO_AUTH_PASSWORD``
    (e.g. a dedicated Cognito service account). Not the Pi UI user's session.
    """
    user = (os.environ.get("DOC_REDACTION_GRADIO_AUTH_USER") or "").strip()
    password = os.environ.get("DOC_REDACTION_GRADIO_AUTH_PASSWORD") or ""
    if user and password:
        return (user, password)
    return None


def httpx_timeout(
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT,
    read: float = DEFAULT_READ_TIMEOUT,
) -> httpx.Timeout:
    return httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)


def _quota_retry_attempts() -> int:
    for key in ("PI_QUOTA_RETRY_ATTEMPTS", "PI_MAX_RETRIES"):
        raw = (os.environ.get(key) or "").strip()
        if raw.isdigit():
            return max(1, int(raw))
    return 5


def _quota_retry_delay_s() -> int:
    raw = (os.environ.get("PI_QUOTA_RETRY_DELAY_S") or "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def is_gradio_rate_limit_error(exc: BaseException) -> bool:
    if type(exc).__name__ == "TooManyRequestsError":
        return True
    lowered = str(exc).lower()
    return any(
        marker in lowered
        for marker in ("429", "too many requests", "rate limit", "rate-limit")
    )


def clear_redaction_client_cache() -> None:
    """Drop cached gradio_client instances (tests or after credential rotation)."""
    _CLIENT_CACHE.clear()


def make_redaction_client(
    base_url: str | None = None,
    hf_token: str | None = None,
    *,
    force_new: bool = False,
    verbose: bool = False,
) -> Client:
    """
    Return a gradio_client for the remote doc_redaction Space.

    Uses ``token=`` (gradio_client 2.x). Retries ``TooManyRequestsError`` with
    ``PI_QUOTA_RETRY_DELAY_S`` backoff and caches one client per URL+token pair
    so agents do not re-fetch ``/gradio_api/info`` on every bash one-liner.
    """
    url = (base_url or redaction_base_url()).rstrip("/")
    token = hf_token if hf_token is not None else redaction_hf_token()
    auth = redaction_gradio_auth()
    cache_key = (url, token or "", auth or ())
    if not force_new and cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    client_kwargs: dict[str, Any] = {
        "httpx_kwargs": {"timeout": httpx_timeout()},
        "verbose": verbose,
    }
    max_attempts = _quota_retry_attempts()
    delay_s = _quota_retry_delay_s()
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if auth:
                client = Client(url, auth=auth, **client_kwargs)
            elif token:
                client = Client(url, token=token, **client_kwargs)
            else:
                client = Client(url, **client_kwargs)
            _CLIENT_CACHE[cache_key] = client
            return client
        except Exception as exc:
            if not is_gradio_rate_limit_error(exc):
                raise
            last_error = exc
            if attempt >= max_attempts:
                break
            time.sleep(delay_s)

    assert last_error is not None
    raise last_error


def call_doc_redact(
    pdf_path: str | Path,
    dest_dir: str | Path,
    *,
    ocr_method: str | None = None,
    pii_method: str | None = None,
    deny_list: list[str] | None = None,
    allow_list: list[str] | None = None,
    redact_entities: list[str] | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
) -> tuple[Any, list[Path]]:
    """
    Run ``/doc_redact`` and download outputs into *dest_dir*.

    Prefer this or ``run_doc_redact.py`` over hand-written Gradio scripts.
    """
    from gradio_client import handle_file

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    predict_kwargs: dict[str, Any] = {
        "api_name": "/doc_redact",
        "document_file": handle_file(str(pdf)),
        "redact_entities": list(redact_entities or _DEFAULT_REDACT_ENTITIES),
    }
    if ocr_method:
        predict_kwargs["ocr_method"] = ocr_method
    if pii_method:
        predict_kwargs["pii_method"] = pii_method
    if deny_list:
        predict_kwargs["deny_list"] = deny_list
    if allow_list:
        predict_kwargs["allow_list"] = allow_list
    if page_min is not None:
        predict_kwargs["page_min"] = page_min
    if page_max is not None:
        predict_kwargs["page_max"] = page_max

    client = make_redaction_client()
    result = client.predict(**predict_kwargs)
    paths = resolve_redaction_output_paths(result, document_stem=pdf.stem)
    saved = fetch_redaction_files(paths, dest_dir)
    return result, saved


def is_gradio_file_path(value: str) -> bool:
    """True for absolute Unix or Windows paths returned by Gradio predict."""
    s = (value or "").strip()
    if not s:
        return False
    if s.startswith("/") and len(s) > 1:
        return True
    return len(s) >= 3 and s[1] == ":" and s[0].isalpha() and s[2] in ("\\", "/")


def _collect_paths(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        if is_gradio_file_path(value):
            out.append(value.strip())
    elif isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and is_gradio_file_path(path):
            out.append(path.strip())
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


def doc_redaction_output_root() -> Path | None:
    """Resolved doc_redaction ``OUTPUT_FOLDER`` when the main app config is importable."""
    try:
        from tools.config import OUTPUT_FOLDER

        return Path(OUTPUT_FOLDER).resolve()
    except ImportError:
        raw = (os.environ.get("DOC_REDACTION_OUTPUT_FOLDER") or "").strip()
        if not raw:
            return None
        try:
            return Path(raw).resolve()
        except OSError:
            return None


def discover_redaction_outputs(
    document_stem: str,
    *,
    since: float | None = None,
) -> list[str]:
    """
    Fallback when ``/doc_redact`` returns ``[]``: glob the doc_redaction output tree.

    Matches filenames containing *document_stem* (e.g. ``example_of_emails``).
    When *since* is set, only files with ``mtime >= since`` are returned.
    """
    stem = (document_stem or "").strip()
    if not stem:
        return []
    if split_redaction_backend():
        return []

    root = doc_redaction_output_root()
    if root is None or not root.is_dir():
        return []

    threshold = since if since is not None else None
    found: list[str] = []
    try:
        for path in root.rglob(f"*{stem}*"):
            if not path.is_file():
                continue
            if threshold is not None:
                try:
                    if path.stat().st_mtime < threshold:
                        continue
                except OSError:
                    continue
            found.append(str(path.resolve()))
    except OSError:
        return []
    return sorted(found)


def resolve_redaction_output_paths(
    result: Any,
    *,
    document_stem: str = "",
    run_started_at: float | None = None,
) -> list[str]:
    """
    Collect output paths from a ``/doc_redact`` result, with on-disk fallback.

    Prefer paths embedded in the Gradio response; when empty, search
    ``OUTPUT_FOLDER`` (including per-user session subfolders).
    """
    paths = extract_server_paths(result)
    if paths:
        return paths
    if document_stem:
        discovered = discover_redaction_outputs(
            document_stem,
            since=run_started_at,
        )
        if discovered:
            return discovered
    return []


def _download_via_gradio_http(
    paths: list[str],
    dest: Path,
    *,
    base_url: str,
    hf_token: str | None,
) -> list[Path]:
    headers: dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token.strip()}"

    downloaded: list[Path] = []
    with httpx.Client(timeout=httpx_timeout(), headers=headers) as http:
        for path in paths:
            file_url = f"{base_url}/gradio_api/file={quote(path, safe='')}"
            local_path = dest / Path(path).name
            response = http.get(file_url)
            response.raise_for_status()
            local_path.write_bytes(response.content)
            downloaded.append(local_path)
    return downloaded


def fetch_redaction_files(
    paths: list[str],
    dest_dir: str | Path,
    *,
    base_url: str | None = None,
    hf_token: str | None = None,
) -> list[Path]:
    """
    Save redaction outputs into *dest_dir*.

    When Pi and doc_redaction share a host filesystem (typical local dev), copies
    directly from disk. Otherwise falls back to ``GET /gradio_api/file=``.
    """
    url = (base_url or redaction_base_url()).rstrip("/")
    token = hf_token if hf_token is not None else redaction_hf_token()

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    http_paths: list[str] = []

    use_http_only = split_redaction_backend()
    for path in paths:
        if not is_gradio_file_path(path):
            continue
        if not use_http_only:
            local = Path(path)
            try:
                if local.is_file():
                    out = dest / local.name
                    if local.resolve() != out.resolve():
                        shutil.copy2(local, out)
                    else:
                        out = local.resolve()
                    saved.append(out)
                    continue
            except OSError:
                pass
        http_paths.append(path)

    if http_paths:
        saved.extend(
            _download_via_gradio_http(http_paths, dest, base_url=url, hf_token=token)
        )
    return saved


def download_gradio_files(
    paths: list[str],
    dest_dir: str | Path,
    *,
    base_url: str | None = None,
    hf_token: str | None = None,
) -> list[Path]:
    """Backward-compatible alias for :func:`fetch_redaction_files`."""
    return fetch_redaction_files(
        paths,
        dest_dir,
        base_url=base_url,
        hf_token=hf_token,
    )
