"""Synth assertions for optional Pi agent ECS service on shared ALB."""

import sys
from pathlib import Path

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_pi_agent_requires_service_connect():
    enable_pi = "True"
    enable_sc = "False"
    with pytest.raises(ValueError, match="ENABLE_PI_AGENT_ECS_SERVICE"):
        if enable_pi == "True" and enable_sc != "True":
            raise ValueError(
                "ENABLE_PI_AGENT_ECS_SERVICE=True requires ENABLE_ECS_SERVICE_CONNECT=True "
                "so the Pi task can reach the main app at http://<discovery>:7860."
            )


def test_build_pi_agent_container_environment():
    from cdk_functions import build_pi_agent_container_environment

    env = build_pi_agent_container_environment(
        service_connect_discovery_name="redaction",
        main_app_port=7860,
        pi_gradio_port=7862,
    )
    assert env["DOC_REDACTION_GRADIO_URL"] == "http://redaction:7860"
    assert env["PI_DEFAULT_PROVIDER"] == "amazon-bedrock"
    assert env["PI_GRADIO_PORT"] == "7862"


def test_pi_agent_alb_attachment_synth():
    from aws_cdk import App, Duration, Environment, Stack
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk import aws_ecs as ecs
    from aws_cdk import aws_elasticloadbalancingv2 as elbv2
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_s3 as s3
    from cdk_functions import (
        attach_pi_agent_to_shared_alb,
        create_pi_agent_ecs_resources,
    )

    app = App()
    stack = Stack(
        app,
        "PiAgentAlbTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    alb_sg = ec2.SecurityGroup(stack, "AlbSg", vpc=vpc)
    alb = elbv2.ApplicationLoadBalancer(
        stack,
        "Alb",
        vpc=vpc,
        internet_facing=True,
        security_group=alb_sg,
    )
    cluster = ecs.Cluster(stack, "Cluster", vpc=vpc)
    config_bucket = s3.Bucket(stack, "ConfigBucket")
    task_role = iam.Role(
        stack,
        "TaskRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    execution_role = iam.Role(
        stack,
        "ExecRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )

    pi_service, pi_sg, _ = create_pi_agent_ecs_resources(
        stack,
        "Pi",
        vpc=vpc,
        cluster=cluster,
        private_subnets=vpc.private_subnets,
        pi_ecr_image_uri="123456789012.dkr.ecr.eu-west-2.amazonaws.com/pi-agent",
        container_name="pi-agent",
        task_role=task_role,
        execution_role=execution_role,
        config_bucket=config_bucket,
        pi_agent_env_s3_key="",
        service_name="test-pi-service",
        task_family="test-pi-task",
        security_group_name="test-pi-sg",
        log_group_name="/ecs/test-pi-logs",
        cpu=1024,
        memory_mib=2048,
        pi_gradio_port=7862,
        service_connect_namespace="test-ns",
        service_connect_discovery_name="redaction",
        main_app_port=7860,
        use_fargate_spot="FARGATE",
    )

    http_listener = alb.add_listener("Http", port=80, open=True)
    http_listener.add_action(
        "DefaultDeny",
        action=elbv2.ListenerAction.fixed_response(
            status_code=403,
            content_type="text/plain",
            message_body="Access denied",
        ),
    )
    attach_pi_agent_to_shared_alb(
        stack,
        "PiAlb",
        vpc=vpc,
        alb_security_group=alb_sg,
        pi_security_group=pi_sg,
        pi_service=pi_service,
        pi_port=7862,
        pi_host_header="pi.example.com",
        listener_rule_priority=1,
        target_group_name="test-pi-tg",
        stickiness_cookie_duration=Duration.hours(8),
        https_listener=None,
        http_listener=http_listener,
        acm_certificate_arn="",
        enable_cognito_auth=False,
        cognito_user_pool=None,
        cognito_user_pool_client=None,
        cognito_user_pool_domain=None,
    )

    template = app.synth().get_stack_by_name("PiAgentAlbTest").template
    resources = template["Resources"]
    lb_count = sum(
        1
        for r in resources.values()
        if r["Type"] == "AWS::ElasticLoadBalancingV2::LoadBalancer"
    )
    tg_count = sum(
        1
        for r in resources.values()
        if r["Type"] == "AWS::ElasticLoadBalancingV2::TargetGroup"
    )
    assert lb_count == 1
    assert tg_count == 1
