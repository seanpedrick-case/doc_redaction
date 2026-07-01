"""Shared boto3 helpers for Bedrock AgentCore runtime and harness clients."""

from __future__ import annotations

from agent_runtime import AgentRuntimeError


def bedrock_agentcore_client(region: str):
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    from pi_agent_config import configure_aws_credentials

    configure_aws_credentials()
    session = boto3.Session(region_name=region)
    try:
        session.client("sts").get_caller_identity()
    except (ClientError, BotoCoreError, NoCredentialsError) as exc:
        raise AgentRuntimeError(
            "AWS credentials are required to invoke AgentCore. "
            "Set AWS_PROFILE / PI_AWS_PROFILE, mount ~/.aws into the pi-agent container, "
            "or paste session keys under **Agent backend** → **Apply backend**. "
            "For HTTP runtime auth with CUSTOM_JWT, set AGENTCORE_API_KEY instead."
        ) from exc
    return session.client("bedrock-agentcore", region_name=region)


def region_from_agentcore_arn(arn: str, *, resource_label: str) -> str:
    """Return AWS region from a bedrock-agentcore ARN."""
    normalized = (arn or "").strip()
    if not normalized.startswith("arn:"):
        raise AgentRuntimeError(
            f"Expected an AgentCore {resource_label} ARN, got: {arn!r}"
        )
    parts = normalized.split(":")
    if len(parts) < 6 or parts[2] != "bedrock-agentcore":
        raise AgentRuntimeError(f"Invalid AgentCore {resource_label} ARN: {arn!r}")
    region = parts[3].strip()
    if not region:
        raise AgentRuntimeError(
            f"Could not parse region from {resource_label} ARN: {arn!r}"
        )
    if f":{resource_label}/" not in normalized:
        raise AgentRuntimeError(
            f"ARN must be a {resource_label} resource (arn:...:bedrock-agentcore:...:{resource_label}/...), "
            f"got: {arn!r}"
        )
    return region
