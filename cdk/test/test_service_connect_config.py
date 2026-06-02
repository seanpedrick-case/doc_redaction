"""Config validation for ECS Service Connect options."""

import pytest
from cdk_config import (
    build_service_connect_client_security_group_names,
    parse_comma_separated_list,
)


def test_service_connect_express_mutual_exclusion():
    enable_sc = "True"
    use_express = "True"
    with pytest.raises(ValueError, match="ENABLE_ECS_SERVICE_CONNECT"):
        if enable_sc == "True" and use_express == "True":
            raise ValueError(
                "ENABLE_ECS_SERVICE_CONNECT=True is only supported on the legacy Fargate "
                "service path."
            )


def test_parse_client_security_group_ids():
    assert parse_comma_separated_list("sg-abc, sg-def") == ["sg-abc", "sg-def"]
    assert parse_comma_separated_list("") == []


def test_build_service_connect_client_sg_names_from_prefixes(monkeypatch):
    monkeypatch.setattr(
        "cdk_config.ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_LIST",
        [],
    )
    monkeypatch.setattr(
        "cdk_config.ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES_LIST",
        ["OtherApp-"],
    )
    monkeypatch.setattr(
        "cdk_config.ECS_SERVICE_CONNECT_CLIENT_SG_NAME_SUFFIX",
        "SecurityGroupECS",
    )
    assert build_service_connect_client_security_group_names() == [
        "OtherApp-SecurityGroupECS"
    ]


def test_resolve_service_connect_client_security_group_ids():
    from cdk_functions import resolve_service_connect_client_security_group_ids

    def _ctx(key, default=None):
        if key == "security_group_id:ClientSg":
            return "sg-ctx123"
        return default

    ids = resolve_service_connect_client_security_group_ids(
        ["sg-explicit"],
        ["ClientSg"],
        _ctx,
    )
    assert ids == ["sg-explicit", "sg-ctx123"]
