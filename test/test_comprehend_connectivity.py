"""Fast-fail AWS Comprehend connectivity helpers."""

from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from tools.data_anonymise import (
    _comprehend_connectivity_error_message,
    _is_non_retryable_aws_error,
    verify_comprehend_connectivity,
)


def test_is_non_retryable_aws_error_token_retrieval():
    err = botocore.exceptions.TokenRetrievalError(
        provider="sso",
        error_msg="Token has expired and refresh failed",
    )
    assert _is_non_retryable_aws_error(err) is True


def test_is_non_retryable_aws_error_access_denied_client_error():
    err = botocore.exceptions.ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "DetectPiiEntities",
    )
    assert _is_non_retryable_aws_error(err) is True


def test_verify_comprehend_connectivity_success():
    client = MagicMock()
    client.detect_pii_entities.return_value = {"Entities": []}
    verify_comprehend_connectivity(client, "en")
    client.detect_pii_entities.assert_called_once()


def test_verify_comprehend_connectivity_expired_sso_raises_clear_message():
    client = MagicMock()
    client.detect_pii_entities.side_effect = botocore.exceptions.TokenRetrievalError(
        provider="sso",
        error_msg="Token has expired and refresh failed",
    )
    with pytest.raises(Exception, match="SSO token has expired"):
        verify_comprehend_connectivity(client, "en")
    assert "aws sso login" in _comprehend_connectivity_error_message(
        client.detect_pii_entities.side_effect
    )
