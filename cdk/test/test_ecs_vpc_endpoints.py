"""ECS VPC endpoint helpers (task-subnet aligned: public for Express + public subnets)."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_create_ecs_vpc_endpoints_synth_interface_and_s3_gateway():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import create_ecs_vpc_endpoints_for_private_subnets

    app = App()
    stack = Stack(app, "VpcEndpointTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    private = ec2.SubnetSelection(
        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
    )
    create_ecs_vpc_endpoints_for_private_subnets(
        stack,
        vpc=vpc,
        subnets=private,
        include_secrets_and_kms=True,
        aws_region="eu-west-2",
    )
    template = assertions.Template.from_stack(stack)
    # 5 interface endpoints + 1 S3 gateway endpoint
    template.resource_count_is("AWS::EC2::VPCEndpoint", 6)
    template.has_resource_properties(
        "AWS::EC2::VPCEndpoint",
        {"VpcEndpointType": "Interface", "PrivateDnsEnabled": True},
    )
    template.has_resource_properties(
        "AWS::EC2::VPCEndpoint",
        {"VpcEndpointType": "Gateway"},
    )


def test_imported_vpc_from_attributes_requires_cidr_block():
    """Imported VPCs must pass vpc_cidr_block or endpoint/SG helpers fail at synth."""
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2

    app = App()
    stack = Stack(app, "ImportedVpcTest")
    vpc = ec2.Vpc.from_vpc_attributes(
        stack,
        "ImportedVpc",
        vpc_id="vpc-12345678",
        availability_zones=["eu-west-2a", "eu-west-2b"],
        vpc_cidr_block="10.0.0.0/16",
    )
    assert vpc.vpc_cidr_block == "10.0.0.0/16"


def test_create_ecs_vpc_endpoints_skips_existing_service_names():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import create_ecs_vpc_endpoints_for_private_subnets

    app = App()
    stack = Stack(app, "SkipEndpointTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    private = ec2.SubnetSelection(
        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
    )
    skip = frozenset(
        {
            "com.amazonaws.eu-west-2.kms",
            "com.amazonaws.eu-west-2.secretsmanager",
        }
    )
    create_ecs_vpc_endpoints_for_private_subnets(
        stack,
        vpc=vpc,
        subnets=private,
        skip_service_names=skip,
        include_secrets_and_kms=True,
        aws_region="eu-west-2",
    )
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::EC2::VPCEndpoint", 4)


def test_create_ecs_vpc_endpoints_skips_precheck_shared_vpc_endpoints():
    """Regression: skip list must use AWS ServiceName strings, not CDK service.name tokens."""
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import create_ecs_vpc_endpoints_for_private_subnets

    app = App()
    stack = Stack(app, "SharedVpcSkipTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    private = ec2.SubnetSelection(
        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
    )
    precheck_skip = frozenset(
        {
            "com.amazonaws.eu-west-2.kms",
            "com.amazonaws.eu-west-2.s3",
            "com.amazonaws.eu-west-2.secretsmanager",
        }
    )
    create_ecs_vpc_endpoints_for_private_subnets(
        stack,
        vpc=vpc,
        subnets=private,
        skip_service_names=precheck_skip,
        include_secrets_and_kms=True,
        aws_region="eu-west-2",
    )
    template = assertions.Template.from_stack(stack)
    # ecr.api, ecr.dkr, logs only (kms, secretsmanager, s3 skipped)
    template.resource_count_is("AWS::EC2::VPCEndpoint", 3)


def test_list_existing_vpc_endpoint_service_names(monkeypatch):
    from cdk_functions import list_existing_vpc_endpoint_service_names

    class FakePaginator:
        def paginate(self, **kwargs):
            return [
                {
                    "VpcEndpoints": [
                        {
                            "ServiceName": "com.amazonaws.eu-west-2.kms",
                            "State": "available",
                        },
                        {
                            "ServiceName": "com.amazonaws.eu-west-2.s3",
                            "State": "available",
                        },
                        {
                            "ServiceName": "com.amazonaws.eu-west-2.ecr.api",
                            "State": "deleted",
                        },
                    ]
                }
            ]

    class FakeEc2:
        def get_paginator(self, name):
            assert name == "describe_vpc_endpoints"
            return FakePaginator()

    monkeypatch.setattr(
        "cdk_functions.boto3.client",
        lambda service, region_name=None: FakeEc2(),
    )
    names = list_existing_vpc_endpoint_service_names("vpc-123", region_name="eu-west-2")
    assert names == frozenset(
        {
            "com.amazonaws.eu-west-2.kms",
            "com.amazonaws.eu-west-2.s3",
        }
    )


def test_resolve_ecs_vpc_endpoint_subnet_selection_express_public():
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import resolve_ecs_vpc_endpoint_subnet_selection

    app = App()
    stack = Stack(app, "SubnetSelectTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    public_sel = resolve_ecs_vpc_endpoint_subnet_selection(
        use_express_ingress=True,
        express_use_public_subnets=True,
        public_subnets=vpc.public_subnets,
        private_subnets=vpc.private_subnets,
    )
    assert public_sel is not None
    assert {s.subnet_id for s in public_sel.subnets} == {
        s.subnet_id for s in vpc.public_subnets
    }


def test_resolve_ecs_vpc_endpoint_subnet_selection_express_private():
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import resolve_ecs_vpc_endpoint_subnet_selection

    app = App()
    stack = Stack(app, "SubnetSelectPrivateTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    private_sel = resolve_ecs_vpc_endpoint_subnet_selection(
        use_express_ingress=True,
        express_use_public_subnets=False,
        public_subnets=vpc.public_subnets,
        private_subnets=vpc.private_subnets,
    )
    assert private_sel is not None
    assert {s.subnet_id for s in private_sel.subnets} == {
        s.subnet_id for s in vpc.private_subnets
    }


def test_resolve_ecs_vpc_endpoint_subnet_selection_legacy_fargate():
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import resolve_ecs_vpc_endpoint_subnet_selection

    app = App()
    stack = Stack(app, "SubnetSelectLegacyTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    private_sel = resolve_ecs_vpc_endpoint_subnet_selection(
        use_express_ingress=False,
        express_use_public_subnets=True,
        public_subnets=vpc.public_subnets,
        private_subnets=vpc.private_subnets,
    )
    assert private_sel is not None
    assert {s.subnet_id for s in private_sel.subnets} == {
        s.subnet_id for s in vpc.private_subnets
    }


def test_resolve_ecs_vpc_endpoint_subnet_selection_public_only_legacy():
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import resolve_ecs_vpc_endpoint_subnet_selection

    app = App()
    stack = Stack(app, "SubnetSelectPublicOnlyTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    public_sel = resolve_ecs_vpc_endpoint_subnet_selection(
        use_express_ingress=False,
        express_use_public_subnets=True,
        public_subnets=vpc.public_subnets,
        private_subnets=[],
    )
    assert public_sel is not None
    assert {s.subnet_id for s in public_sel.subnets} == {
        s.subnet_id for s in vpc.public_subnets
    }


def test_create_ecs_vpc_endpoints_on_public_subnets():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import create_ecs_vpc_endpoints_for_private_subnets

    app = App()
    stack = Stack(app, "PublicEndpointTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    public = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
    create_ecs_vpc_endpoints_for_private_subnets(
        stack,
        vpc=vpc,
        subnets=public,
        include_secrets_and_kms=True,
        aws_region="eu-west-2",
    )
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::EC2::VPCEndpoint", 6)
    public_logical_ids = {
        stack.get_logical_id(s.node.default_child) for s in vpc.public_subnets
    }
    interface_endpoints = [
        r
        for r in template.find_resources("AWS::EC2::VPCEndpoint").values()
        if r.get("Properties", {}).get("VpcEndpointType") == "Interface"
    ]
    assert len(interface_endpoints) == 5
    for resource in interface_endpoints:
        subnet_ids = resource["Properties"]["SubnetIds"]
        assert len(subnet_ids) == len(vpc.public_subnets)
        for ref in subnet_ids:
            assert ref.get("Ref") in public_logical_ids


def test_resolve_ecs_s3_gateway_subnet_selection_all_stack_subnets():
    from aws_cdk import App, Stack
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import resolve_ecs_s3_gateway_subnet_selection

    app = App()
    stack = Stack(app, "S3SubnetSelectTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    s3_sel = resolve_ecs_s3_gateway_subnet_selection(
        public_subnets=vpc.public_subnets,
        private_subnets=vpc.private_subnets,
    )
    assert s3_sel is not None
    expected_ids = {s.subnet_id for s in vpc.public_subnets + vpc.private_subnets}
    assert {s.subnet_id for s in s3_sel.subnets} == expected_ids


def test_s3_gateway_endpoint_uses_all_stack_subnets_not_only_task_tier():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import (
        create_ecs_vpc_endpoints_for_private_subnets,
        resolve_ecs_s3_gateway_subnet_selection,
    )

    app = App()
    stack = Stack(app, "MixedS3GatewayTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    public = ec2.SubnetSelection(subnets=vpc.public_subnets)
    all_stack_subnets = resolve_ecs_s3_gateway_subnet_selection(
        public_subnets=vpc.public_subnets,
        private_subnets=vpc.private_subnets,
    )
    create_ecs_vpc_endpoints_for_private_subnets(
        stack,
        vpc=vpc,
        subnets=public,
        s3_gateway_subnets=all_stack_subnets,
        include_secrets_and_kms=True,
        aws_region="eu-west-2",
    )
    template = assertions.Template.from_stack(stack)
    gateway_endpoints = [
        r
        for r in template.find_resources("AWS::EC2::VPCEndpoint").values()
        if r.get("Properties", {}).get("VpcEndpointType") == "Gateway"
    ]
    assert len(gateway_endpoints) == 1
    route_table_ids = gateway_endpoints[0]["Properties"]["RouteTableIds"]
    assert len(route_table_ids) == len(vpc.public_subnets) + len(vpc.private_subnets)
    interface_endpoints = [
        r
        for r in template.find_resources("AWS::EC2::VPCEndpoint").values()
        if r.get("Properties", {}).get("VpcEndpointType") == "Interface"
    ]
    public_logical_ids = {
        stack.get_logical_id(s.node.default_child) for s in vpc.public_subnets
    }
    for resource in interface_endpoints:
        for ref in resource["Properties"]["SubnetIds"]:
            assert ref.get("Ref") in public_logical_ids


def test_list_vpc_associated_cidr_blocks_primary_and_secondary():
    from cdk_functions import list_vpc_associated_cidr_blocks

    vpc = {
        "CidrBlock": "10.0.0.0/16",
        "CidrBlockAssociationSet": [
            {
                "CidrBlock": "10.0.0.0/16",
                "CidrBlockState": {"State": "associated"},
            },
            {
                "CidrBlock": "10.1.0.0/16",
                "CidrBlockState": {"State": "associated"},
            },
        ],
    }
    assert list_vpc_associated_cidr_blocks(vpc) == [
        "10.0.0.0/16",
        "10.1.0.0/16",
    ]


def test_vpc_endpoint_security_group_ingress_covers_all_vpc_cidrs():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from cdk_functions import add_vpc_endpoint_https_ingress_from_vpc_cidrs

    app = App()
    stack = Stack(app, "DualCidrEndpointSgTest")
    vpc = ec2.Vpc.from_vpc_attributes(
        stack,
        "ImportedVpc",
        vpc_id="vpc-dual-cidr",
        availability_zones=["eu-west-2a"],
        vpc_cidr_block="10.0.0.0/16",
    )
    endpoint_sg = ec2.SecurityGroup(
        stack,
        "EndpointSg",
        vpc=vpc,
        description="test",
    )
    add_vpc_endpoint_https_ingress_from_vpc_cidrs(
        endpoint_sg,
        vpc_cidr_block="10.0.0.0/16",
        vpc_cidr_blocks=["10.0.0.0/16", "10.1.0.0/16"],
    )
    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::EC2::SecurityGroup",
        {
            "SecurityGroupIngress": assertions.Match.array_with(
                [
                    assertions.Match.object_like(
                        {"CidrIp": "10.0.0.0/16", "FromPort": 443, "ToPort": 443}
                    ),
                    assertions.Match.object_like(
                        {"CidrIp": "10.1.0.0/16", "FromPort": 443, "ToPort": 443}
                    ),
                ]
            )
        },
    )
