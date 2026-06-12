"""ECS private-subnet VPC endpoint helper."""

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
        private_subnets=private,
        include_secrets_and_kms=True,
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
