"""Tests for public subnet Internet Gateway audit and CDK wiring helpers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


@pytest.fixture(autouse=True)
def _minimal_cdk_config_env(monkeypatch):
    """Avoid loading the developer cdk_config.env validation errors during import."""
    monkeypatch.setenv("USE_ECS_EXPRESS_MODE", "False")
    monkeypatch.setenv("ACM_SSL_CERTIFICATE_ARN", "")
    monkeypatch.setenv("ENABLE_S3_BATCH_ECS_TRIGGER", "False")
    monkeypatch.setenv("ENABLE_PI_AGENT_ECS_SERVICE", "False")
    monkeypatch.setenv("ENABLE_PI_AGENT_EXPRESS_SERVICE", "False")
    monkeypatch.setenv("ENABLE_ECS_SERVICE_CONNECT", "False")
    for mod_name in list(sys.modules):
        if mod_name in ("cdk_config", "cdk_functions") or mod_name.startswith("cdk_"):
            del sys.modules[mod_name]


def test_route_table_default_internet_gateway_parses_igw():
    from cdk_functions import route_table_default_internet_gateway

    mock_client = MagicMock()
    mock_client.describe_route_tables.return_value = {
        "RouteTables": [
            {
                "Routes": [
                    {
                        "DestinationCidrBlock": "0.0.0.0/0",
                        "GatewayId": "igw-abc123",
                    }
                ]
            }
        ]
    }
    with patch("cdk_functions.boto3.client", return_value=mock_client):
        assert route_table_default_internet_gateway("rtb-1") == "igw-abc123"


def test_audit_needs_attachment_when_igw_detached():
    from cdk_functions import audit_public_subnet_internet_connectivity

    mock_ec2 = MagicMock()
    mock_ec2.describe_internet_gateways.return_value = {
        "InternetGateways": [{"Attachments": []}]
    }
    with (
        patch("cdk_functions.get_internet_gateways_attached_to_vpc", return_value=[]),
        patch("cdk_functions.internet_gateway_exists", return_value=True),
        patch("cdk_functions.boto3.client", return_value=mock_ec2),
        patch("cdk_functions.route_table_default_internet_gateway", return_value=None),
        patch(
            "cdk_functions.route_table_has_non_igw_default_route", return_value=False
        ),
    ):
        result = audit_public_subnet_internet_connectivity(
            "vpc-111",
            "igw-configured",
            [
                {
                    "name": "PublicSubnet1",
                    "subnet_id": "subnet-1",
                    "route_table_id": "rtb-1",
                }
            ],
        )
    assert result["internet_gateway_id"] == "igw-configured"
    assert result["internet_gateway_needs_vpc_attachment"] is True
    assert len(result["public_subnets_needing_igw_route"]) == 1


def test_audit_raises_when_default_route_conflicts():
    from cdk_functions import audit_public_subnet_internet_connectivity

    with (
        patch(
            "cdk_functions.get_internet_gateways_attached_to_vpc",
            return_value=["igw-attached"],
        ),
        patch("cdk_functions.internet_gateway_exists", return_value=True),
        patch(
            "cdk_functions.route_table_default_internet_gateway",
            return_value="igw-other",
        ),
    ):
        with pytest.raises(ValueError, match="0.0.0.0/0"):
            audit_public_subnet_internet_connectivity(
                "vpc-111",
                "igw-attached",
                [
                    {
                        "name": "PublicSubnet1",
                        "subnet_id": "subnet-1",
                        "route_table_id": "rtb-1",
                    }
                ],
            )


def test_wire_public_subnet_internet_access_synth():
    from aws_cdk import App, Environment, Stack, assertions
    from cdk_functions import wire_public_subnet_internet_access

    app = App()
    stack = Stack(
        app,
        "IgwWireTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    wire_public_subnet_internet_access(
        stack,
        "Test",
        vpc_id="vpc-111",
        internet_gateway_id="igw-abc123",
        needs_igw_vpc_attachment=True,
        subnets_needing_route=[
            {
                "name": "PublicSubnet1",
                "subnet_id": "subnet-1",
                "route_table_id": "rtb-aaa",
            }
        ],
    )
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::EC2::VPCGatewayAttachment", 1)
    template.resource_count_is("AWS::EC2::Route", 1)
    template.has_resource_properties(
        "AWS::EC2::Route",
        {
            "RouteTableId": "rtb-aaa",
            "DestinationCidrBlock": "0.0.0.0/0",
            "GatewayId": "igw-abc123",
        },
    )
