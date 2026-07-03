"""Bridge local Gradio session workspaces to AgentCore runtime invokes."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from session_workspace import session_workspace_dir

_SKIP_UPLOAD_PREFIXES = (
    "preview/",
    ".pi/preview/",
    "output_final_download/",
)
_DEFAULT_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_FILES = 80


def max_upload_bytes() -> int:
    raw = (os.environ.get("AGENTCORE_MAX_UPLOAD_BYTES") or "").strip()
    if raw.isdigit():
        return int(raw)
    return _DEFAULT_MAX_BYTES


def max_upload_files() -> int:
    raw = (os.environ.get("AGENTCORE_MAX_UPLOAD_FILES") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _DEFAULT_MAX_FILES


def discover_session_document_name(session_hash: str) -> str | None:
    """Return the newest PDF basename at the session workspace root."""
    root = session_workspace_dir(session_hash)
    if not root.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for path in root.glob("*.pdf"):
        if not path.is_file():
            continue
        try:
            candidates.append((path.stat().st_mtime, path.name))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _should_upload_relative_path(relative: str) -> bool:
    rel = relative.replace("\\", "/").lstrip("/")
    if any(rel.startswith(prefix) for prefix in _SKIP_UPLOAD_PREFIXES):
        return False
    if rel.lower().endswith(".pdf") and "/" not in rel:
        return True
    return rel.startswith("redact/")


def _upload_priority(relative: str) -> tuple[int, str]:
    rel = relative.replace("\\", "/")
    if rel.lower().endswith("_review_file.csv"):
        return (0, rel)
    if rel.lower().endswith(".pdf") and "/" not in rel:
        return (1, rel)
    if rel.startswith("redact/") and rel.lower().endswith("_redacted.pdf"):
        return (2, rel)
    if rel.startswith("redact/"):
        return (3, rel)
    return (9, rel)


def collect_session_files_for_agentcore_upload(
    session_hash: str,
    *,
    document_name: str | None = None,
) -> list[dict[str, str]]:
    """
    Collect local session files to seed the remote AgentCore workspace.

    Includes the source PDF and everything under ``redact/`` (review CSVs, OCR
    exports, etc.) so follow-up turns work after a runtime cold start.
    """
    root = session_workspace_dir(session_hash).resolve()
    if not root.is_dir():
        return []

    limit_bytes = max_upload_bytes()
    limit_files = max_upload_files()
    doc_name = (
        document_name or discover_session_document_name(session_hash) or ""
    ).strip()

    candidates: list[tuple[tuple[int, str], Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if doc_name and relative == doc_name:
            candidates.append(((1, relative), path))
            continue
        if not _should_upload_relative_path(relative):
            continue
        candidates.append((_upload_priority(relative), path))

    if doc_name:
        doc_path = (root / doc_name).resolve()
        if doc_path.is_file() and all(path != doc_path for _, path in candidates):
            candidates.append(((1, doc_name), doc_path))

    candidates.sort(key=lambda item: item[0])
    staged: list[dict[str, str]] = []
    skipped_large: list[str] = []

    for _, path in candidates:
        if len(staged) >= limit_files:
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > limit_bytes:
            skipped_large.append(path.relative_to(root).as_posix())
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        staged.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "content_base64": base64.b64encode(payload).decode("ascii"),
            }
        )

    if skipped_large and staged:
        staged.append(
            {
                "relative_path": ".agentcore_upload_skipped.txt",
                "content_base64": base64.b64encode(
                    (
                        "Some local files were not uploaded (over "
                        f"{limit_bytes:,} bytes):\n" + "\n".join(skipped_large[:20])
                    ).encode("utf-8")
                ).decode("ascii"),
            }
        )
    return staged


def _find_review_csv_paths(session_hash: str) -> list[str]:
    root = session_workspace_dir(session_hash)
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in sorted(root.rglob("*_review_file.csv")):
        if not path.is_file():
            continue
        try:
            found.append(path.relative_to(root).as_posix())
        except ValueError:
            continue
    return found[:5]


def build_agentcore_followup_context(
    session_hash: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Prompt prefix so follow-ups continue Pass 1 instead of restarting upload flow."""
    doc = discover_session_document_name(session_hash)
    review_csvs = _find_review_csv_paths(session_hash)
    lines = [
        "**AgentCore follow-up (mandatory):** Pass 1 redaction already ran in this "
        "session. The local UI synced workspace artifacts into your session folder "
        "before this message — call `list_workspace_files` first.",
        "Do **not** ask the user to re-upload the PDF unless `list_workspace_files` "
        "is empty after sync.",
        "Prefer editing the existing `*_review_file.csv` and running `review_apply` "
        "again (or `verify_coverage` first) rather than a full new `doc_redact`.",
    ]
    if doc:
        lines.append(f"**Source document:** `{doc}`")
        lines.append(f"**Redaction tree:** `redact/{doc}/`")
    if review_csvs:
        lines.append(
            "**Review CSV(s):** " + ", ".join(f"`{path}`" for path in review_csvs)
        )
    if history:
        excerpt = _format_history_excerpt(history)
        if excerpt:
            lines.append("**Prior chat (UI, for context):**\n" + excerpt)
    return "\n".join(lines) + "\n\n"


def _format_history_excerpt(
    history: list[dict[str, Any]],
    *,
    max_messages: int = 10,
    max_chars: int = 6000,
) -> str:
    chunks: list[str] = []
    total = 0
    for message in history[-max_messages:]:
        role = str(message.get("role") or "user").strip()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        line = f"- **{role}:** {content[:1500]}"
        if total + len(line) > max_chars:
            break
        chunks.append(line)
        total += len(line)
    return "\n".join(chunks)


_CLOUDFRONT_OVERLAY_KEYS = (
    "DOC_REDACTION_GRADIO_URL",
    "DOC_REDACTION_AUTH_TOKEN",
    "DOC_REDACTION_AUTH_COOKIE_NAME",
    # AgentCore runtime URL + model are created/finalised in deploy phase 2 and
    # re-uploaded to agent.env in S3; the container's task env is fixed at synth,
    # so pick them up here on (re)start rather than requiring a stack update.
    "AGENTCORE_RUNTIME_URL",
    "AGENT_DEFAULT_MODEL",
    "AGENT_DEFAULT_PROVIDER",
)
_cloudfront_overlay_applied = False


def apply_agentcore_cloudfront_config_overlay() -> dict[str, str]:
    """Overlay post-deploy AgentCore settings from S3 into ``os.environ`` (AgentCore only).

    The Pi Express task definition is fixed at synth time and cannot carry values
    that only exist after deploy phase 2 — the (later-created) CloudFront
    domain/magic-link token and the AgentCore runtime URL/model — and injecting
    them into the task env would create a dependency cycle. So when running as the
    AgentCore orchestrator, we fetch the post-deploy ``agent.env`` from S3 and
    override those keys (see ``_CLOUDFRONT_OVERLAY_KEYS``): the CloudFront backend
    URL + token cookie for ``build_agentcore_invoke_runtime_config``, plus
    ``AGENTCORE_RUNTIME_URL`` / ``AGENT_DEFAULT_MODEL`` so the running container
    reaches the runtime and shows the correct model without a stack update.

    Controlled by ``DOC_REDACTION_CONFIG_S3_BUCKET`` / ``DOC_REDACTION_CONFIG_S3_KEY``
    (set by CDK on every Express deploy). The overlay runs whenever that S3 config
    source is present and will **promote** the container to ``AGENT_ORCHESTRATOR=agentcore``
    if the S3 ``agent.env`` says so — this lets deploy phase 2 (which re-uploads
    ``agent.env`` and recycles the service) flip an already-running container to
    AgentCore without a stack update. It never demotes an already-agentcore
    container and is a no-op when the S3 config does not select AgentCore.
    Best-effort and idempotent; returns the keys it overrode.
    """
    global _cloudfront_overlay_applied
    applied: dict[str, str] = {}
    if _cloudfront_overlay_applied:
        return applied
    already_agentcore = (
        os.environ.get("AGENT_ORCHESTRATOR") or ""
    ).strip().lower() == "agentcore"
    bucket = (os.environ.get("DOC_REDACTION_CONFIG_S3_BUCKET") or "").strip()
    if not bucket:
        # No post-deploy S3 config source; rely solely on the task env.
        _cloudfront_overlay_applied = True
        return applied
    key = (os.environ.get("DOC_REDACTION_CONFIG_S3_KEY") or "agent.env").strip()

    try:
        import boto3

        region = (
            os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None
        )
        body = (
            boto3.client("s3", region_name=region)
            .get_object(Bucket=bucket, Key=key)["Body"]
            .read()
            .decode("utf-8", "replace")
        )
    except Exception as exc:  # noqa: BLE001 - best effort, keep the UI booting
        print(f"AgentCore config overlay skipped ({bucket}/{key}): {exc}")
        _cloudfront_overlay_applied = True
        return applied

    parsed: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        parsed[name.strip()] = value.strip().strip('"').strip("'")

    # Only act when this deployment actually selects AgentCore (task env already
    # says so, or the S3 config does). This keeps the overlay a no-op for pi /
    # langgraph deployments that share the same config bucket.
    s3_agentcore = (
        parsed.get("AGENT_ORCHESTRATOR") or ""
    ).strip().lower() == "agentcore"
    if not (already_agentcore or s3_agentcore):
        _cloudfront_overlay_applied = True
        return applied

    if s3_agentcore and not already_agentcore:
        os.environ["AGENT_ORCHESTRATOR"] = "agentcore"
        applied["AGENT_ORCHESTRATOR"] = "agentcore"

    for name in _CLOUDFRONT_OVERLAY_KEYS:
        value = parsed.get(name)
        if value:
            os.environ[name] = value
            applied[name] = value

    _cloudfront_overlay_applied = True
    if applied:
        print(
            "Applied AgentCore config overlay from S3 for: "
            + ", ".join(sorted(applied))
        )
    return applied


def build_agentcore_invoke_runtime_config() -> dict[str, str]:
    """
    Backend settings from the local Gradio process for each AgentCore invoke.

    Overrides ``agentcore.env`` on the AWS runtime so the remote agent uses the same
    ``DOC_REDACTION_GRADIO_URL`` shown in the Pi UI (not a baked-in HF Space default).
    """
    from redaction_prompt import doc_redaction_gradio_url

    url = doc_redaction_gradio_url().strip().rstrip("/")
    config: dict[str, str] = {}
    if url:
        config["DOC_REDACTION_GRADIO_URL"] = url
    for key in (
        "DOC_REDACTION_GRADIO_AUTH_USER",
        "DOC_REDACTION_GRADIO_AUTH_PASSWORD",
        # CloudFront magic-link: the AgentCore runtime runs outside the VPC and
        # reaches doc_redaction through CloudFront, so it must send the token
        # cookie on every request (Service Connect is not reachable from it).
        "DOC_REDACTION_AUTH_TOKEN",
        "DOC_REDACTION_AUTH_COOKIE_NAME",
        "AGENT_DEFAULT_OCR_METHOD",
        "AGENT_DEFAULT_PII_METHOD",
    ):
        value = (os.environ.get(key) or "").strip()
        if value:
            config[key] = value
    if "hf.space" in url.lower():
        token = (
            os.environ.get("HF_TOKEN") or os.environ.get("DOC_REDACTION_HF_TOKEN") or ""
        ).strip()
        if token:
            config["HF_TOKEN"] = token
    return config
