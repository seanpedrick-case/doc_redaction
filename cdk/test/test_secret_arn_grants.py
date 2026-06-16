"""Secrets Manager IAM grant ARN handling (name wildcard vs complete ARN)."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_from_secret_complete_arn_grant_uses_exact_resource():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_secretsmanager as sm

    app = App()
    stack = Stack(app, "SecretGrantTest")
    role = iam.Role(
        stack,
        "ExecutionRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    secret_arn = (
        "arn:aws:secretsmanager:eu-west-2:123456789012:secret:"
        "Lambeth-ParamCognitoSecret-AbCdEf"
    )
    secret = sm.Secret.from_secret_complete_arn(
        stack,
        "CognitoSecret",
        secret_complete_arn=secret_arn,
    )
    secret.grant_read(role)

    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": assertions.Match.object_like(
                {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Action": assertions.Match.array_with(
                                        ["secretsmanager:GetSecretValue"]
                                    ),
                                    "Resource": secret_arn,
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_from_secret_name_v2_grant_uses_wildcard_suffix():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_secretsmanager as sm

    app = App()
    stack = Stack(app, "SecretNameGrantTest")
    role = iam.Role(
        stack,
        "ExecutionRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    secret_name = "Lambeth-ParamCognitoSecret"
    secret = sm.Secret.from_secret_name_v2(
        stack, "CognitoSecret", secret_name=secret_name
    )
    secret.grant_read(role)

    template = assertions.Template.from_stack(stack)
    resources = template.find_resources("AWS::IAM::Policy")
    policy_text = str(resources)
    assert "??????" in policy_text or "???????" in policy_text
