"""Minimal CDK stack: AgentCore Harness + S3 for skills and local page-review staging."""

from __future__ import annotations

import json
from typing import Any

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from cdk_functions import add_s3_enforce_ssl_policy, create_agentcore_harness
from constructs import Construct
from harness_local_config import (
    HARNESS_ENABLE_PAGE_REVIEW,
    HARNESS_MAX_ITERATIONS,
    HARNESS_MAX_TOKENS,
    HARNESS_MODEL_ID,
    HARNESS_NAME,
    HARNESS_SKILLS_S3_PREFIX,
    HARNESS_STAGING_S3_PREFIX,
    HARNESS_TIMEOUT_SECONDS,
    INLINE_TOOLS_JSON,
    SKILL_SOURCE_DIR,
    SKILLS_BUCKET_NAME,
    STAGING_BUCKET_NAME,
    STAGING_LIFECYCLE_DAYS,
)


class HarnessLocalStack(Stack):
    """
    Deploy only what a **locally running** ``app.py`` needs for AgentCore Harness redaction:

    - Bedrock AgentCore Harness (inline tool names + S3 skill; tools run in the local app)
    - S3 bucket for the ``doc-redaction-harness`` skill
    - S3 bucket for optional parallel page-review staging (presigned URLs)
    - IAM managed policy for your developer principal (``InvokeHarness`` + S3 staging)

    No ECS, VPC, Cognito, or main app infrastructure.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        skills_bucket = s3.Bucket(
            self,
            "HarnessSkillsBucket",
            bucket_name=SKILLS_BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        staging_bucket = s3.Bucket(
            self,
            "HarnessStagingBucket",
            bucket_name=STAGING_BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(expiration=Duration.days(STAGING_LIFECYCLE_DAYS))
            ],
        )
        add_s3_enforce_ssl_policy(skills_bucket)
        add_s3_enforce_ssl_policy(staging_bucket)

        inline_tool_specs = json.loads(INLINE_TOOLS_JSON.read_text(encoding="utf-8"))
        harness = create_agentcore_harness(
            self,
            "RedactionHarness",
            harness_name=HARNESS_NAME,
            model_id=HARNESS_MODEL_ID,
            inline_tool_specs=inline_tool_specs,
            skills_bucket=skills_bucket,
            skills_source_directory=str(SKILL_SOURCE_DIR),
            skills_s3_prefix=HARNESS_SKILLS_S3_PREFIX,
            region=self.region,
            account=self.account,
            kms_key=None,
            max_iterations=HARNESS_MAX_ITERATIONS,
            timeout_seconds=HARNESS_TIMEOUT_SECONDS,
            max_tokens=HARNESS_MAX_TOKENS,
        )
        harness_arn = harness.get_att("Arn").to_string()

        staging_prefix = HARNESS_STAGING_S3_PREFIX.strip("/") + "/"
        staging_uri = f"s3://{staging_bucket.bucket_name}/{staging_prefix}"

        iam.ManagedPolicy(
            self,
            "LocalAppInvokerPolicy",
            managed_policy_name=f"DocRedactionHarnessLocalInvoker-{self.region}"[:128],
            description=(
                "Attach to the IAM user or role you use when running app.py locally: "
                "InvokeHarness on this stack's harness and S3 read/write on the staging bucket."
            ),
            statements=[
                iam.PolicyStatement(
                    sid="InvokeHarness",
                    actions=["bedrock-agentcore:InvokeHarness"],
                    resources=[harness_arn],
                ),
                iam.PolicyStatement(
                    sid="StagingBucketList",
                    actions=["s3:ListBucket"],
                    resources=[staging_bucket.bucket_arn],
                    conditions={"StringLike": {"s3:prefix": [f"{staging_prefix}*"]}},
                ),
                iam.PolicyStatement(
                    sid="StagingObjectRW",
                    actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                    resources=[f"{staging_bucket.bucket_arn}/{staging_prefix}*"],
                ),
            ],
        )

        local_env_lines = "\n".join(
            [
                "USE_AGENTCORE=True",
                f"AGENTCORE_HARNESS_ARN={harness_arn}",
                f"AGENTCORE_REGION={self.region}",
                f"AGENTCORE_HARNESS_S3_STAGING_PREFIX={staging_uri}",
                f"AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW={HARNESS_ENABLE_PAGE_REVIEW}",
            ]
        )

        CfnOutput(
            self,
            "RedactionHarnessArn",
            value=harness_arn,
            description="Set AGENTCORE_HARNESS_ARN in config/app_config.env",
        )
        CfnOutput(
            self,
            "OutputHarnessSkillsBucket",
            value=skills_bucket.bucket_name,
            description="Harness execution role reads the skill from this bucket (not your local creds)",
        )
        CfnOutput(
            self,
            "OutputHarnessStagingBucket",
            value=staging_bucket.bucket_name,
            description="Local app presigns page-review bundles here",
        )
        CfnOutput(
            self,
            "HarnessStagingPrefix",
            value=staging_uri,
            description="AGENTCORE_HARNESS_S3_STAGING_PREFIX for local app",
        )
        CfnOutput(
            self,
            "LocalAppConfigSnippet",
            value=local_env_lines,
            description="Paste into config/app_config.env (or merge with deploy_harness_local.py)",
        )
        CfnOutput(
            self,
            "LocalInvokerPolicyName",
            value=f"DocRedactionHarnessLocalInvoker-{self.region}",
            description="Attach this managed policy to your SSO role / IAM user for local runs",
        )
