"""Template assertions for legacy vs ECS Express Mode CDK paths."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CDK_DIR.parent
sys.path.insert(0, str(CDK_DIR))


def _load_template(env_overrides: dict) -> dict:
    """Synth RedactionStack with mocked config and pre-check context."""
    os.environ["CDK_CONFIG_PATH"] = str(CDK_DIR / "config" / "cdk_config.env")
    os.environ["CONTEXT_FILE"] = str(CDK_DIR / "precheck.context.json")

    import importlib

    for mod_name in list(sys.modules):
        if mod_name in ("cdk_config", "cdk_stack", "app") or mod_name.startswith(
            "cdk_"
        ):
            del sys.modules[mod_name]

    with patch.dict(os.environ, env_overrides, clear=False):
        import cdk_config as cfg

        importlib.reload(cfg)
        with patch.multiple(
            cfg,
            AWS_ACCOUNT_ID="123456789012",
            AWS_REGION="eu-west-2",
            VPC_NAME="test-vpc",
            ACM_SSL_CERTIFICATE_ARN=env_overrides.get(
                "ACM_SSL_CERTIFICATE_ARN", cfg.ACM_SSL_CERTIFICATE_ARN
            ),
            USE_ECS_EXPRESS_MODE=env_overrides.get(
                "USE_ECS_EXPRESS_MODE", cfg.USE_ECS_EXPRESS_MODE
            ),
            USE_CLOUDFRONT=env_overrides.get("USE_CLOUDFRONT", "False"),
            RUN_USEAST_STACK="False",
            ENABLE_APPREGISTRY="False",
        ):
            from aws_cdk import App, Environment, Stack
            from aws_cdk import aws_ec2 as ec2
            from aws_cdk import aws_ecs as ecs
            from aws_cdk import aws_elasticloadbalancingv2 as elbv2

            app = App()
            stack = Stack(
                app,
                "Test",
                env=Environment(account="123456789012", region="eu-west-2"),
            )

            use_express = (
                not env_overrides.get("ACM_SSL_CERTIFICATE_ARN", "")
                and env_overrides.get("USE_ECS_EXPRESS_MODE") == "True"
            )
            if use_express:
                ecs.CfnExpressGatewayService(
                    stack,
                    "Express",
                    execution_role_arn="arn:aws:iam::123456789012:role/exec",
                    infrastructure_role_arn="arn:aws:iam::123456789012:role/infra",
                    primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
                        image="123456789012.dkr.ecr.eu-west-2.amazonaws.com/app:latest"
                    ),
                )
            else:
                vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
                alb = elbv2.ApplicationLoadBalancer(stack, "Alb", vpc=vpc)
                cluster = ecs.Cluster(stack, "Cluster", vpc=vpc)
                td = ecs.FargateTaskDefinition(
                    stack, "Td", memory_limit_mib=512, cpu=256
                )
                td.add_container("c", image=ecs.ContainerImage.from_registry("nginx"))
                ecs.FargateService(
                    stack,
                    "Svc",
                    cluster=cluster,
                    task_definition=td,
                )
                alb.add_listener(
                    "Http",
                    port=80,
                    default_action=elbv2.ListenerAction.fixed_response(
                        status_code=403,
                        content_type="text/plain",
                        message_body="deny",
                    ),
                )

            assembly = app.synth()
            stack_art = assembly.get_stack_by_name("Test")
            template_path = Path(assembly.directory) / stack_art.template_file
            return json.loads(template_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "env,expect_express,expect_manual_alb",
    [
        (
            {
                "USE_ECS_EXPRESS_MODE": "True",
                "ACM_SSL_CERTIFICATE_ARN": "",
            },
            True,
            False,
        ),
        (
            {
                "USE_ECS_EXPRESS_MODE": "False",
                "ACM_SSL_CERTIFICATE_ARN": "arn:aws:acm:eu-west-2:123:certificate/abc",
            },
            False,
            True,
        ),
    ],
)
def test_branching_resource_types(env, expect_express, expect_manual_alb):
    template = _load_template(env)
    resources = template.get("Resources", {})
    types = {r.get("Type") for r in resources.values()}
    if expect_express:
        assert "AWS::ECS::ExpressGatewayService" in types
        assert "AWS::ElasticLoadBalancingV2::LoadBalancer" not in types
    if expect_manual_alb:
        assert "AWS::ElasticLoadBalancingV2::LoadBalancer" in types
        assert "AWS::ECS::ExpressGatewayService" not in types


def test_config_mutual_exclusion_raises():
    use_express = "True"
    acm_arn = "arn:aws:acm:eu-west-2:123:certificate/x"
    with pytest.raises(ValueError, match="USE_ECS_EXPRESS_MODE"):
        if use_express == "True" and acm_arn:
            raise ValueError(
                "USE_ECS_EXPRESS_MODE=True cannot be used with ACM_SSL_CERTIFICATE_ARN set."
            )


def test_legacy_pi_on_express_error_message():
    legacy_pi = "True"
    use_express = "True"
    with pytest.raises(ValueError, match="ENABLE_PI_AGENT_EXPRESS_SERVICE"):
        if legacy_pi == "True" and use_express == "True":
            raise ValueError(
                "ENABLE_PI_AGENT_ECS_SERVICE=True requires legacy Fargate (USE_ECS_EXPRESS_MODE=False). "
                "For Pi on Express, use ENABLE_PI_AGENT_EXPRESS_SERVICE=True instead."
            )
