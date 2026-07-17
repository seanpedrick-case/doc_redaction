"""Upload task inputs for AgentCore Harness (S3 + presigned URL prompt prefix)."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from session_workspace import session_workspace_dir


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    prefix = (parsed.path or "").lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def harness_s3_input_uri(session_hash: str, file_name: str) -> tuple[str, str, str]:
    """
    Return ``(bucket, key, s3_uri)`` for a harness input object.

    Uses ``AGENTCORE_HARNESS_S3_INPUT_PREFIX`` (``s3://bucket/prefix/``) when set,
    otherwise ``s3://{S3_OUTPUTS_BUCKET}/harness-inputs/{session_hash}/``.
    """
    explicit = (os.environ.get("AGENTCORE_HARNESS_S3_INPUT_PREFIX") or "").strip()
    if explicit:
        bucket, prefix = _parse_s3_uri(explicit)
    else:
        bucket = (os.environ.get("S3_OUTPUTS_BUCKET") or "").strip()
        if not bucket:
            raise ValueError(
                "Set AGENTCORE_HARNESS_S3_INPUT_PREFIX or S3_OUTPUTS_BUCKET for harness file upload."
            )
        safe_session = (session_hash or "default").strip().replace("/", "_")[:128]
        prefix = f"harness-inputs/{safe_session}/"
    key = f"{prefix}{Path(file_name).name}"
    return bucket, key, f"s3://{bucket}/{key}"


def build_harness_document_prompt_prefix(
    session_hash: str,
    document_name: str,
) -> str | None:
    """
    Upload the task PDF to S3 and return a prompt prefix for the Harness to fetch it.

    Returns ``None`` when upload is disabled or the file is missing.
    """
    if not document_name:
        return None
    run_aws = (os.environ.get("RUN_AWS_FUNCTIONS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        not run_aws
        and not (os.environ.get("AGENTCORE_HARNESS_S3_INPUT_PREFIX") or "").strip()
    ):
        return (
            "**Harness file bridge:** RUN_AWS_FUNCTIONS is off and "
            "AGENTCORE_HARNESS_S3_INPUT_PREFIX is unset — upload the document to the Harness "
            "workspace manually or enable S3 upload."
        )

    root = session_workspace_dir(session_hash)
    src = root / document_name
    if not src.is_file():
        return None

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
        from pi_agent_config import configure_aws_credentials

        configure_aws_credentials()
        bucket, key, s3_uri = harness_s3_input_uri(session_hash, document_name)
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "eu-west-2"
        )
        client = boto3.client("s3", region_name=region)
        client.upload_file(str(src), bucket, key)
        presigned = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(os.environ.get("AGENTCORE_HARNESS_PRESIGN_SECONDS", "3600")),
        )
    except (BotoCoreError, ClientError, ValueError, OSError) as exc:
        return (
            f"**Harness file bridge error:** Could not upload `{document_name}` to S3 ({exc}). "
            "Place the file on the Harness workspace mount or fix AWS permissions."
        )

    mount_path = (
        os.environ.get("AGENTCORE_HARNESS_S3_MOUNT_PATH") or "/tmp/workspace"
    ).rstrip("/")
    dest = f"{mount_path}/{Path(document_name).name}"
    return (
        f"**Harness input file (download before Pass 1):**\n"
        f"- S3 object: `{s3_uri}`\n"
        f"- Presigned URL (expires in 1h): {presigned}\n"
        f"- Save to Harness workspace as: `{dest}`\n"
        f"- Example: `curl -fsSL -o {dest!r} '<presigned-url>'` then use `{dest}` as INPUT_PATH.\n"
    )
