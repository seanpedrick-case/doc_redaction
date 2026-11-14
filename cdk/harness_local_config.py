"""
Minimal configuration for the harness-only CDK stack (local ``app.py``).

Loads ``cdk/config/harness_local.env`` when present. Does not import the full
``cdk_config`` module (no VPC, ALB, or installer prerequisites).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

CDK_DIR = Path(__file__).resolve().parent
REPO_ROOT = CDK_DIR.parent
HARNESS_LOCAL_ENV = CDK_DIR / "config" / "harness_local.env"

load_dotenv(HARNESS_LOCAL_ENV, override=False)


def _env(name: str, default: str = "") -> str:
    value = (os.environ.get(name) or default).strip()
    return value


AWS_REGION = _env("AWS_REGION") or _env("CDK_DEFAULT_REGION", "eu-west-2")
AWS_ACCOUNT_ID = _env("AWS_ACCOUNT_ID") or _env("CDK_DEFAULT_ACCOUNT", "")

STACK_NAME = _env("HARNESS_LOCAL_STACK_NAME", "DocRedactionHarnessLocal")

HARNESS_NAME = _env("AGENTCORE_HARNESS_NAME", "LocalRedactionHarness")
HARNESS_MODEL_ID = _env("AGENTCORE_HARNESS_MODEL_ID", "")


def default_harness_model_id(model: str, region: str = "") -> str:
    """Return Bedrock model id as configured (no automatic ``eu.`` inference profile)."""
    _ = region  # kept for call-site compatibility
    return (model or "anthropic.claude-sonnet-5").strip()


if not HARNESS_MODEL_ID:
    HARNESS_MODEL_ID = default_harness_model_id(
        _env("AGENTCORE_BEDROCK_MODEL", "anthropic.claude-sonnet-5"), AWS_REGION
    )

HARNESS_MAX_ITERATIONS = int(_env("AGENTCORE_HARNESS_MAX_ITERATIONS", "150") or "150")
HARNESS_TIMEOUT_SECONDS = int(
    _env("AGENTCORE_HARNESS_TIMEOUT_SECONDS", "3600") or "3600"
)
HARNESS_MAX_TOKENS = int(_env("AGENTCORE_HARNESS_MAX_TOKENS", "16000") or "16000")
HARNESS_ENABLE_PAGE_REVIEW = _env("AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW", "True")
HARNESS_SKILLS_S3_PREFIX = _env(
    "AGENTCORE_HARNESS_SKILLS_S3_PREFIX", "agentcore-harness/skills/"
)
HARNESS_STAGING_S3_PREFIX = _env(
    "AGENTCORE_HARNESS_STAGING_S3_PREFIX", "harness-staging/"
)

STAGING_LIFECYCLE_DAYS = int(_env("HARNESS_LOCAL_STAGING_LIFECYCLE_DAYS", "7") or "7")


def _sanitize_bucket_label(raw: str) -> str:
    label = re.sub(r"[^a-z0-9-]", "-", raw.lower())
    label = re.sub(r"-+", "-", label).strip("-")
    return label[:40] or "local"


def default_skills_bucket_name(account: str, region: str) -> str:
    """Globally unique-ish name for the skills bucket (harness execution role reads it)."""
    acct = re.sub(r"\D", "", account or "000000000000")[-12:]
    region_bit = region.replace("_", "-").lower()
    name = f"doc-redact-harness-skills-{region_bit}-{acct}"
    return name[:63].rstrip("-")


def default_staging_bucket_name(account: str, region: str) -> str:
    """Bucket for presigned page-review uploads from a locally running app."""
    acct = re.sub(r"\D", "", account or "000000000000")[-12:]
    region_bit = region.replace("_", "-").lower()
    name = f"doc-redact-harness-stage-{region_bit}-{acct}"
    return name[:63].rstrip("-")


SKILLS_BUCKET_NAME = _env(
    "HARNESS_LOCAL_SKILLS_BUCKET_NAME"
) or default_skills_bucket_name(AWS_ACCOUNT_ID, AWS_REGION)
STAGING_BUCKET_NAME = _env("HARNESS_LOCAL_STAGING_BUCKET_NAME") or (
    default_staging_bucket_name(AWS_ACCOUNT_ID, AWS_REGION)
)

INLINE_TOOLS_JSON = REPO_ROOT / "tools" / "agentcore_harness_tools.json"
SKILL_SOURCE_DIR = REPO_ROOT / "skills" / "doc-redaction-harness"
