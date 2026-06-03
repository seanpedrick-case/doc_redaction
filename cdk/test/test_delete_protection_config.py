"""Tests for ENABLE_RESOURCE_DELETE_PROTECTION helpers."""

import importlib

import pytest
from aws_cdk import RemovalPolicy


@pytest.mark.parametrize(
    ("env_value", "enabled"),
    [
        ("True", True),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("False", False),
        ("false", False),
        ("0", False),
    ],
)
def test_delete_protection_helpers(monkeypatch, env_value, enabled):
    monkeypatch.setenv("ENABLE_RESOURCE_DELETE_PROTECTION", env_value)

    import cdk_config

    importlib.reload(cdk_config)
    import cdk_functions

    importlib.reload(cdk_functions)

    assert cdk_functions.is_resource_delete_protection_enabled() is enabled
    assert cdk_functions.resource_deletion_protection_flag() is enabled
    assert cdk_functions.managed_resource_removal_policy() == (
        RemovalPolicy.RETAIN if enabled else RemovalPolicy.DESTROY
    )
    assert cdk_functions.s3_auto_delete_objects_on_stack_destroy() is (not enabled)
