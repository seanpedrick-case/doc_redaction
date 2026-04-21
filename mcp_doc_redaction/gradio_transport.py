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


def _parse_gradio_sse_final_payload(buffer: str) -> dict[str, Any] | None:
    """
    Parse Gradio server-sent events and return a payload dict when `event: complete`
    has been received with parsable `data:` JSON.

    Gradio signals success with SSE lines::

        event: complete
        data: [...]

    not with ``{\"status\": \"complete\"}``. Without this, HTTP clients poll forever
    while the Space has already finished.

    Returns None if the buffer does not yet contain a complete ``event: complete`` /
    ``data:`` pair (e.g. still streaming, or truncated JSON).

    Raises RuntimeError on ``event: error``.
    """
    lines = buffer.replace("\r\n", "\n").split("\n")
    i = 0
    last_complete: Any | None = None
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if not stripped.lower().startswith("event:"):
            i += 1
            continue
        ev = stripped.split(":", 1)[1].strip().lower()
        i += 1
        if i >= len(lines):
            break
        dline = lines[i].strip()
        i += 1
        if not dline.lower().startswith("data:"):
            continue
        raw = dline.split(":", 1)[1].strip() if ":" in dline else ""
        if ev == "heartbeat":
            continue
        if ev == "error":
            try:
                detail = json.loads(raw) if raw else raw
            except json.JSONDecodeError:
                detail = raw
            raise RuntimeError(f"Gradio event:error: {detail!r}")
        if ev == "complete" and raw and raw != "[DONE]":
            try:
                last_complete = json.loads(raw)
            except json.JSONDecodeError:
                # Truncated ``data:`` line while more chunks are incoming
                pass
    if last_complete is None:
        return None
    if isinstance(last_complete, dict):
        return last_complete
    return {"data": last_complete}


def _buffer_looks_like_gradio_sse(buf: str) -> bool:
    b = buf.lstrip().lower()
    return b.startswith("event:") or "\nevent:" in b


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
            url = f"{self.base_url}/gradio_api/call/{api_name.lstrip('/')}/{event_id}"
            payload = {}
            with self._client.stream("GET", url) as r:
                r.raise_for_status()

                ct = (r.headers.get("content-type") or "").lower()
                buf = ""
                max_chars = 2_000_000
                is_sse = "text/event-stream" in ct
                try:
                    for chunk in r.iter_text():
                        if chunk:
                            buf += chunk
                        if not is_sse and _buffer_looks_like_gradio_sse(buf):
                            is_sse = True
                        if is_sse:
                            try:
                                done = _parse_gradio_sse_final_payload(buf)
                            except RuntimeError:
                                raise
                            if done is not None:
                                return CompletedCall(api_name=api_name, payload=done)
                        elif not is_sse:
                            if '"status"' in buf or '"type"' in buf:
                                if len(buf) > 4096:
                                    break
                        if len(buf) >= max_chars:
                            break
                except Exception:
                    buf = buf or ""

                text = (buf or "").strip()
                if is_sse or _buffer_looks_like_gradio_sse(text):
                    try:
                        done = _parse_gradio_sse_final_payload(buf)
                    except RuntimeError:
                        raise
                    if done is not None:
                        return CompletedCall(api_name=api_name, payload=done)
                    payload = {}
                elif not text:
                    payload = {}
                elif text.startswith("data:"):
                    data_lines: list[str] = []
                    for line in text.splitlines():
                        if line.startswith("data:"):
                            data_lines.append(line[len("data:") :].strip())
                    candidate = next(
                        (x for x in reversed(data_lines) if x and x != "[DONE]"), ""
                    )
                    if candidate:
                        try:
                            parsed: Any = json.loads(candidate)
                        except json.JSONDecodeError:
                            parsed = {}
                        payload = (
                            parsed if isinstance(parsed, dict) else {"data": parsed}
                        )
                    else:
                        payload = {}
                else:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = {}
                    payload = parsed if isinstance(parsed, dict) else {"data": parsed}
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                payload = {"data": payload}
            status = str(payload.get("status") or payload.get("type") or "").lower()
            if status in ("complete", "completed", "success", "succeeded", "done"):
                return CompletedCall(api_name=api_name, payload=payload)
            if (
                "data" in payload
                and status == ""
                and isinstance(payload.get("data"), list)
            ):
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
