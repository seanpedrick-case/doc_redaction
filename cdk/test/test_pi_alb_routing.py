"""Tests for Pi ALB path/host routing helpers."""

import sys
from pathlib import Path

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


@pytest.fixture(autouse=True)
def _minimal_cdk_config_env(monkeypatch):
    monkeypatch.setenv("USE_ECS_EXPRESS_MODE", "False")
    monkeypatch.setenv("ACM_SSL_CERTIFICATE_ARN", "")
    monkeypatch.setenv("ENABLE_S3_BATCH_ECS_TRIGGER", "False")
    monkeypatch.setenv("ENABLE_PI_AGENT_ECS_SERVICE", "False")
    monkeypatch.setenv("ENABLE_PI_AGENT_EXPRESS_SERVICE", "False")
    monkeypatch.setenv("ENABLE_ECS_SERVICE_CONNECT", "False")
    for mod_name in list(sys.modules):
        if mod_name in ("cdk_config", "cdk_functions") or mod_name.startswith("cdk_"):
            del sys.modules[mod_name]


def test_normalize_pi_alb_path_prefix():
    from cdk_functions import normalize_pi_alb_path_prefix

    assert normalize_pi_alb_path_prefix("/pi") == "/pi"
    assert normalize_pi_alb_path_prefix("pi") == "/pi"
    assert normalize_pi_alb_path_prefix("") == "/pi"


def test_pi_alb_path_patterns():
    from cdk_functions import pi_alb_path_patterns

    assert pi_alb_path_patterns("/pi") == ["/pi", "/pi/*"]


def test_format_pi_public_urls_path_on_cloudfront():
    from cdk_functions import format_pi_public_urls

    urls = format_pi_public_urls(
        routing_mode="path",
        path_prefix="/pi",
        host_header="",
        cloudfront_domain="d123.cloudfront.net",
        use_https=True,
    )
    assert urls == ["https://d123.cloudfront.net/pi/"]


def test_pi_listener_rule_count():
    from cdk_functions import pi_listener_rule_count

    assert pi_listener_rule_count("path") == 1
    assert pi_listener_rule_count("host") == 1
    assert pi_listener_rule_count("both") == 2


def test_attach_pi_path_rule_synth():
    from aws_cdk import App, Duration, Environment, Stack, assertions
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
        "PiPathAlbTest",
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
        pi_root_path="/pi",
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
        routing_mode="path",
        path_prefix="/pi",
        pi_host_header="",
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

    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        {
            "HealthCheckPath": "/pi/",
        },
    )
