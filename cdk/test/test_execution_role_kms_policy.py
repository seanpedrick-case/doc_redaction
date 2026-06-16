"""ECS task vs execution role KMS inline policy helpers."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_task_and_execution_role_kms_policies_use_different_key_arns():
    from cdk_functions import (
        build_ecs_execution_role_kms_policy,
        build_ecs_task_role_kms_policy,
    )

    s3_key = "arn:aws:kms:eu-west-2:123456789012:key/s3-shared-key"
    secret_key = "arn:aws:kms:eu-west-2:123456789012:key/secret-key"

    task_policy = build_ecs_task_role_kms_policy(shared_kms_key_arn=s3_key)
    exec_policy = build_ecs_execution_role_kms_policy(secret_kms_key_arn=secret_key)

    task_kms = next(
        s for s in task_policy["Statement"] if s.get("Sid") == "KMSS3Access"
    )
    secret_kms = next(
        s for s in exec_policy["Statement"] if s.get("Sid") == "KMSSecretDecrypt"
    )

    assert task_kms["Resource"] == s3_key
    assert secret_kms["Resource"] == secret_key
    assert "kms:GenerateDataKey" in task_kms["Action"]
    assert secret_kms["Action"] == ["kms:Decrypt"]


def test_execution_role_policy_without_custom_s3_key_uses_secret_kms_only():
    from cdk_functions import build_ecs_execution_role_kms_policy

    secret_key = "arn:aws:kms:eu-west-2:123456789012:key/aws/secretsmanager"
    policy = build_ecs_execution_role_kms_policy(secret_kms_key_arn=secret_key)
    kms_statements = [
        s for s in policy["Statement"] if s.get("Action") == ["kms:Decrypt"]
    ]
    assert len(kms_statements) == 1
    assert kms_statements[0]["Resource"] == secret_key


def test_get_secret_kms_key_arn_from_describe_secret(monkeypatch):
    from cdk_functions import get_secret_kms_key_arn

    class FakeSecretsManager:
        def describe_secret(self, SecretId):
            assert SecretId == "my-secret"
            return {
                "KmsKeyId": "arn:aws:kms:eu-west-2:123456789012:key/abc-123",
            }

    monkeypatch.setattr(
        "cdk_functions.boto3.client",
        lambda service, region_name=None: FakeSecretsManager(),
    )
    assert (
        get_secret_kms_key_arn("my-secret", region_name="eu-west-2")
        == "arn:aws:kms:eu-west-2:123456789012:key/abc-123"
    )
