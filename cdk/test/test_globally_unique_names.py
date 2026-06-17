"""Tests for globally unique AWS name checks (S3, Cognito)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_functions import (  # noqa: E402
    resolve_cognito_domain_prefix_availability,
    resolve_s3_bucket_availability,
)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadBucket")


def test_resolve_s3_bucket_available_on_404():
    s3 = MagicMock()
    s3.head_bucket.side_effect = _client_error("404")
    status, name = resolve_s3_bucket_availability("free-bucket", s3_client=s3)
    assert status == "available"
    assert name == "free-bucket"


def test_resolve_s3_bucket_owned_on_success():
    s3 = MagicMock()
    s3.head_bucket.return_value = {}
    status, name = resolve_s3_bucket_availability("my-bucket", s3_client=s3)
    assert status == "owned"
    assert name == "my-bucket"


def test_resolve_s3_bucket_globally_taken_on_403_not_listed():
    s3 = MagicMock()
    s3.head_bucket.side_effect = _client_error("403")
    s3.list_buckets.return_value = {"Buckets": [{"Name": "other-bucket"}]}
    status, name = resolve_s3_bucket_availability(
        "demo-redaction-s3-logs", s3_client=s3
    )
    assert status == "globally_taken"
    assert name == "demo-redaction-s3-logs"


def test_resolve_s3_bucket_owned_on_403_listed_in_account():
    s3 = MagicMock()
    s3.head_bucket.side_effect = _client_error("403")
    s3.list_buckets.return_value = {"Buckets": [{"Name": "my-bucket"}]}
    status, _ = resolve_s3_bucket_availability("my-bucket", s3_client=s3)
    assert status == "owned"


def test_resolve_cognito_domain_taken_when_owned_elsewhere():
    cognito = MagicMock()
    cognito.describe_user_pool_domain.side_effect = _client_error(
        "ResourceNotFoundException"
    )
    assert (
        resolve_cognito_domain_prefix_availability(
            "demo-redaction", region_name="eu-west-2", cognito_client=cognito
        )
        == "taken"
    )


def test_resolve_cognito_domain_available_when_empty_description():
    cognito = MagicMock()
    cognito.describe_user_pool_domain.return_value = {"DomainDescription": {}}
    assert (
        resolve_cognito_domain_prefix_availability(
            "my-prefix", region_name="eu-west-2", cognito_client=cognito
        )
        == "available"
    )


def test_resolve_cognito_domain_taken_when_user_pool_present():
    cognito = MagicMock()
    cognito.describe_user_pool_domain.return_value = {
        "DomainDescription": {"UserPoolId": "eu-west-2_abc"}
    }
    assert (
        resolve_cognito_domain_prefix_availability(
            "taken-prefix", region_name="eu-west-2", cognito_client=cognito
        )
        == "taken"
    )
