"""Tests for Pi on ECS Express Mode helpers and config rules."""

import sys
from pathlib import Path

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_pi_express_mutual_exclusion_with_legacy_pi():
    legacy = "True"
    express = "True"
    with pytest.raises(ValueError, match="at most one Pi deployment"):
        if legacy == "True" and express == "True":
            raise ValueError(
                "Enable at most one Pi deployment mode: ENABLE_PI_AGENT_ECS_SERVICE (legacy Fargate) "
                "or ENABLE_PI_AGENT_EXPRESS_SERVICE (Express), not both."
            )


def test_pi_express_requires_express_mode():
    express_pi = "True"
    use_express = "False"
    with pytest.raises(ValueError, match="ENABLE_PI_AGENT_EXPRESS_SERVICE"):
        if express_pi == "True" and use_express != "True":
            raise ValueError(
                "ENABLE_PI_AGENT_EXPRESS_SERVICE=True requires USE_ECS_EXPRESS_MODE=True "
                "(no ACM_SSL_CERTIFICATE_ARN)."
            )


def test_build_pi_express_container_environment():
    from cdk_functions import build_pi_express_container_environment

    env = build_pi_express_container_environment(
        service_connect_discovery_name="redaction",
        main_app_port=7860,
        pi_gradio_port=7862,
    )
    assert env["DOC_REDACTION_GRADIO_URL"] == "http://redaction:7860"
    assert env["PI_WORKSPACE_DIR"] == "/tmp/pi-workspace"
    assert env["PI_UPLOAD_ROOT"] == "/tmp/gradio"
    assert env["PI_DEPLOYMENT_PROFILE"] == "aws-ecs"


def test_express_service_connect_configuration_server_and_client():
    from cdk_functions import _express_service_connect_configuration

    server = _express_service_connect_configuration(
        namespace="demo-ns",
        port_name="port-7860",
        discovery_name="redaction",
        port=7860,
    )
    assert server["enabled"] is True
    assert server["namespace"] == "demo-ns"
    assert server["services"][0]["portName"] == "port-7860"
    assert server["services"][0]["discoveryName"] == "redaction"

    client = _express_service_connect_configuration(namespace="demo-ns")
    assert "services" not in client


def test_apply_service_connect_custom_resource_synth():
    from aws_cdk import App, Environment, Stack, assertions
    from aws_cdk import aws_ecs as ecs
    from cdk_functions import apply_service_connect_to_express_service

    app = App()
    stack = Stack(
        app,
        "ScExpressTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    express = ecs.CfnExpressGatewayService(
        stack,
        "MainExpress",
        service_name="main-express",
        cluster="test-cluster",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest",
            container_port=7860,
        ),
    )
    apply_service_connect_to_express_service(
        stack,
        "MainSc",
        cluster_name="test-cluster",
        service_name="main-express",
        namespace="test-ns",
        express_service=express,
        port_name="port-7860",
        discovery_name="redaction",
        port=7860,
    )
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("Custom::AWS", 1)
    template.has_resource_properties(
        "Custom::AWS",
        {
            "Create": assertions.Match.string_like_regexp(
                r'"portName":"port-7860".*"discoveryName":"redaction"'
            ),
        },
    )


def test_express_alb_ingress_uses_security_group_id_not_arn():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk import aws_ecs as ecs
    from cdk_functions import allow_express_load_balancer_to_ecs_security_group

    app = App()
    stack = Stack(app, "ExpressAlbIngressTest")
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    task_sg = ec2.SecurityGroup(stack, "TaskSg", vpc=vpc)
    express = ecs.CfnExpressGatewayService(
        stack,
        "Express",
        service_name="main-express",
        cluster="test-cluster",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest",
            container_port=7860,
        ),
    )
    allow_express_load_balancer_to_ecs_security_group(
        stack,
        "ExpressAlbToTask",
        express_service=express,
        ecs_security_group=task_sg,
        container_port=7860,
    )
    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::EC2::SecurityGroupIngress",
        {
            "SourceSecurityGroupId": {
                "Fn::Select": [
                    1,
                    {
                        "Fn::Split": [
                            "security-group/",
                            {
                                "Fn::Select": [
                                    0,
                                    {
                                        "Fn::GetAtt": [
                                            "Express",
                                            "ECSManagedResourceArns.IngressPath.LoadBalancerSecurityGroups",
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                ]
            }
        },
    )


def test_express_gateway_service_defaults_to_idle_scaling_target():
    from aws_cdk import App, Stack, assertions
    from aws_cdk import aws_ecs as ecs
    from cdk_functions import create_express_gateway_service

    app = App()
    stack = Stack(app, "ExpressIdleScalingTest")
    create_express_gateway_service(
        stack,
        "Express",
        service_name="main-express",
        cluster_name="test-cluster",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        task_role_arn="arn:aws:iam::123456789012:role/task",
        cpu="1024",
        memory="4096",
        health_check_path="/",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest",
            container_port=7860,
        ),
        subnet_ids=["subnet-abc"],
        security_group_ids=["sg-main"],
    )
    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::ECS::ExpressGatewayService",
        {
            "ScalingTarget": {
                "MinTaskCount": 0,
                "MaxTaskCount": 1,
                "AutoScalingMetric": "AVERAGE_CPU",
                "AutoScalingTargetValue": 60,
            }
        },
    )


def test_express_infrastructure_role_uses_service_role_managed_policy():
    from aws_cdk import App, Stack, assertions
    from cdk_functions import create_ecs_express_infrastructure_role

    app = App()
    stack = Stack(app, "ExpressInfraRoleTest")
    create_ecs_express_infrastructure_role(
        stack, "ExpressInfrastructureRole", "test-express-infra"
    )
    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "ManagedPolicyArns": assertions.Match.array_with(
                [
                    {
                        "Fn::Join": [
                            "",
                            [
                                "arn:",
                                {"Ref": "AWS::Partition"},
                                ":iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices",
                            ],
                        ]
                    }
                ]
            )
        },
    )


def test_express_listener_helpers_synth_without_reference_error():
    """Fn.select on Express list attrs must use typed attr_* list properties."""
    from aws_cdk import App, Environment, Stack, assertions
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk import aws_ecs as ecs
    from cdk_functions import (
        allow_express_load_balancer_to_ecs_security_group,
        configure_express_listener_cognito_and_cloudfront,
        configure_express_pi_listener_rules,
        create_express_gateway_service,
    )

    app = App()
    stack = Stack(
        app,
        "ExpressListenerHelpers",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    sg = ec2.SecurityGroup(stack, "TaskSg", vpc=vpc)
    main = create_express_gateway_service(
        stack,
        "Main",
        service_name="main-svc",
        cluster_name="cl",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        task_role_arn="arn:aws:iam::123456789012:role/task",
        cpu="1024",
        memory="2048",
        health_check_path="/",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest",
            container_port=7860,
        ),
        subnet_ids=["subnet-abc"],
        security_group_ids=["sg-main"],
    )
    pi = create_express_gateway_service(
        stack,
        "Pi",
        service_name="pi-svc",
        cluster_name="cl",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        task_role_arn="arn:aws:iam::123456789012:role/task",
        cpu="1024",
        memory="2048",
        health_check_path="/pi/",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/pi:latest",
            container_port=7862,
        ),
        subnet_ids=["subnet-abc"],
        security_group_ids=["sg-pi"],
    )
    allow_express_load_balancer_to_ecs_security_group(
        stack,
        "MainLbToTask",
        express_service=main,
        ecs_security_group=sg,
        container_port=7860,
    )
    configure_express_listener_cognito_and_cloudfront(
        stack,
        "MainListener",
        express_service=main,
        user_pool_arn="arn:aws:cognito-idp:eu-west-2:123456789012:userpool/pool",
        user_pool_client_id="client",
        user_pool_domain_prefix="demo-auth",
        use_cloudfront=False,
        cloudfront_host_header="",
    )
    configure_express_pi_listener_rules(
        stack,
        "PiRules",
        express_main_service=main,
        express_pi_service=pi,
        routing_mode="path",
        path_prefix="/pi",
        pi_host_header="",
        rule_priority=3,
        user_pool_arn="arn:aws:cognito-idp:eu-west-2:123456789012:userpool/pool",
        user_pool_client_id="client",
        user_pool_domain_prefix="demo-auth",
    )
    app.synth()
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::ECS::ExpressGatewayService", 2)
    # modifyListener (Custom::AWS) + Pi path rule (Custom::Elbv2ListenerRuleUpsert)
    template.resource_count_is("Custom::AWS", 1)
    template.resource_count_is("Custom::Elbv2ListenerRuleUpsert", 1)
    # authenticate-cognito on ALB requires DescribeUserPoolClient on the CR Lambda role
    cr_policy_actions: list[str] = []
    for props in template.find_resources("AWS::IAM::Policy").values():
        for stmt in (
            props.get("Properties", {}).get("PolicyDocument", {}).get("Statement", [])
        ):
            action = stmt.get("Action", [])
            if isinstance(action, str):
                cr_policy_actions.append(action)
            else:
                cr_policy_actions.extend(action)
    assert "cognito-idp:DescribeUserPoolClient" in cr_policy_actions
    assert "elasticloadbalancing:ModifyListener" in cr_policy_actions


def test_dual_express_gateway_services_synth():
    """Two ExpressGatewayService resources when wiring main + Pi helpers."""
    from aws_cdk import App, Environment, Stack, assertions
    from aws_cdk import aws_ecs as ecs
    from cdk_functions import (
        apply_service_connect_to_express_service,
        build_express_pi_primary_container,
        create_express_gateway_service,
    )

    app = App()
    stack = Stack(
        app,
        "DualExpressTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    main = create_express_gateway_service(
        stack,
        "Main",
        service_name="main-svc",
        cluster_name="cl",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        task_role_arn="arn:aws:iam::123456789012:role/task",
        cpu="1024",
        memory="2048",
        health_check_path="/",
        primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
            image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest",
            container_port=7860,
        ),
        subnet_ids=["subnet-abc"],
        security_group_ids=["sg-main"],
    )
    pi_container = build_express_pi_primary_container(
        image_uri="123456789012.dkr.ecr.eu-west-2.amazonaws.com/pi:latest",
        container_port=7862,
        log_group_name="/ecs/pi-logs",
        aws_region="eu-west-2",
        environment={"PI_WORKSPACE_DIR": "/tmp/pi-workspace"},
    )
    pi = create_express_gateway_service(
        stack,
        "Pi",
        service_name="pi-svc",
        cluster_name="cl",
        execution_role_arn="arn:aws:iam::123456789012:role/exec",
        infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
        task_role_arn="arn:aws:iam::123456789012:role/task",
        cpu="1024",
        memory="2048",
        health_check_path="/",
        primary_container=pi_container,
        subnet_ids=["subnet-abc"],
        security_group_ids=["sg-pi"],
    )
    apply_service_connect_to_express_service(
        stack,
        "MainSc",
        cluster_name="cl",
        service_name="main-svc",
        namespace="ns",
        express_service=main,
        port_name="port-7860",
        discovery_name="redaction",
        port=7860,
    )
    apply_service_connect_to_express_service(
        stack,
        "PiSc",
        cluster_name="cl",
        service_name="pi-svc",
        namespace="ns",
        express_service=pi,
    )
    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::ECS::ExpressGatewayService", 2)
    template.resource_count_is("Custom::AWS", 2)
