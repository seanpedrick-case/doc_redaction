from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


def _norm_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("DOC_REDACTION_BASE_URL is required")
    if not base.startswith("https://"):
        raise ValueError("Only https:// base URLs are allowed by default")
    return base


def _auth_headers(hf_token: str | None) -> dict[str, str]:
    if hf_token and hf_token.strip():
        return {"Authorization": f"Bearer {hf_token.strip()}"}
    return {}


def extract_file_like_paths(value: Any) -> list[str]:
    """
    Recursively extract Gradio file paths from a completed payload.

    Handles:
    - strings that look like absolute paths (/tmp/gradio..., /home/user/..., etc.)
    - dicts with key 'path'
    - lists/tuples of nested values
    """
    out: list[str] = []

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("/") and len(s) > 1:
                out.append(s)
            return
        if isinstance(v, dict):
            p = v.get("path")
            if isinstance(p, str) and p.strip().startswith("/"):
                out.append(p.strip())
            for vv in v.values():
                walk(vv)
            return
        if isinstance(v, (list, tuple)):
            for vv in v:
                walk(vv)
            return

    walk(value)
    # de-dupe stable order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@dataclass(frozen=True)
class CompletedCall:
    api_name: str
    payload: dict[str, Any]


class GradioHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        hf_token: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = _norm_base_url(base_url)
        self._headers = _auth_headers(hf_token)
        self._timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s, headers=self._headers)
        self._info_cache: tuple[float, dict[str, Any]] | None = None

    def close(self) -> None:
        self._client.close()

    def info(self, ttl_s: float = 30.0) -> dict[str, Any]:
        now = time.time()
        if self._info_cache and (now - self._info_cache[0]) < ttl_s:
            return self._info_cache[1]
        r = self._client.get(f"{self.base_url}/gradio_api/info")
        r.raise_for_status()
        data = r.json()
        self._info_cache = (now, data)
        return data

    def endpoint_exists(self, api_name: str) -> bool:
        api = self.info().get("named_endpoints") or {}
        return str(api_name) in api

    def upload_bytes(self, filename: str, content: bytes) -> str:
        files = {"files": (filename, content)}
        r = self._client.post(f"{self.base_url}/gradio_api/upload", files=files)
        r.raise_for_status()
        paths = r.json()
        if not isinstance(paths, list) or not paths:
            raise RuntimeError(f"Unexpected upload response: {paths!r}")
        return str(paths[0])

    def call(self, api_name: str, data: list[Any]) -> str:
        r = self._client.post(
            f"{self.base_url}/gradio_api/call/{api_name.lstrip('/')}",
            content=json.dumps({"data": data}),
            headers={"Content-Type": "application/json", **self._headers},
        )
        r.raise_for_status()
        payload = r.json()
        event_id = (
            payload.get("event_id") or payload.get("id") or payload.get("eventId")
        )
        if not event_id:
            raise RuntimeError(f"Could not find event_id: {payload!r}")
        return str(event_id)

    def poll(
        self,
        api_name: str,
        event_id: str,
        *,
        timeout_s: float = 1800.0,
        initial_sleep_s: float = 0.5,
        max_sleep_s: float = 5.0,
    ) -> CompletedCall:
        start = time.time()
        sleep_s = initial_sleep_s
        while True:
            r = self._client.get(
                f"{self.base_url}/gradio_api/call/{api_name.lstrip('/')}/{event_id}"
            )
            r.raise_for_status()
            payload = r.json()
            status = str(payload.get("status") or payload.get("type") or "").lower()
            if status in ("complete", "completed", "success", "succeeded", "done"):
                return CompletedCall(api_name=api_name, payload=payload)
            if status in ("error", "failed", "canceled", "cancelled"):
                raise RuntimeError(f"Call failed: {payload!r}")
            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for {api_name}/{event_id}")
            time.sleep(sleep_s)
            sleep_s = min(max_sleep_s, sleep_s * 1.5)

    def download(self, internal_path: str) -> bytes:
        r = self._client.get(f"{self.base_url}/gradio_api/file={internal_path}")
        r.raise_for_status()
        return r.content
