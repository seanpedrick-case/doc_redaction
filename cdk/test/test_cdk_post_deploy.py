"""Unit tests for cdk_post_deploy.py (boto3 helpers, no live AWS)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

import cdk_post_deploy as post


def test_container_definitions_with_named_port_adds_mapping_to_first_container():
    containers = [{"name": "app", "image": "nginx"}]
    updated = post._container_definitions_with_named_port(
        containers,
        port_name="port-7860",
        container_port=7860,
    )
    assert updated[0]["portMappings"] == [
        {"name": "port-7860", "containerPort": 7860, "protocol": "tcp"}
    ]


def test_container_definitions_with_named_port_names_existing_mapping():
    containers = [
        {
            "name": "app",
            "image": "nginx",
            "portMappings": [{"containerPort": 7860, "protocol": "tcp"}],
        }
    ]
    updated = post._container_definitions_with_named_port(
        containers,
        port_name="port-7860",
        container_port=7860,
    )
    assert updated[0]["portMappings"][0]["name"] == "port-7860"


def test_resolve_service_task_definition_arn_from_describe_services():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.return_value = {
        "services": [
            {
                "serviceArn": "arn:aws:ecs:eu-west-2:123:service/cluster/app",
                "taskDefinition": "arn:aws:ecs:eu-west-2:123:task-definition/app:1",
            }
        ]
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        arn = post.resolve_service_task_definition_arn("cluster", "app")

    assert arn == "arn:aws:ecs:eu-west-2:123:task-definition/app:1"
    mock_ecs.describe_express_gateway_service.assert_not_called()


def test_resolve_service_task_definition_arn_from_express_service_revision():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.return_value = {
        "services": [
            {
                "serviceArn": "arn:aws:ecs:eu-west-2:123:service/cluster/express-app",
            }
        ]
    }
    mock_ecs.describe_express_gateway_service.return_value = {
        "service": {
            "activeConfigurations": [
                {
                    "serviceRevisionArn": "arn:aws:ecs:eu-west-2:123:service-revision/rev/1"
                }
            ]
        }
    }
    mock_ecs.describe_service_revisions.return_value = {
        "serviceRevisions": [
            {
                "taskDefinition": "arn:aws:ecs:eu-west-2:123:task-definition/express:3",
            }
        ]
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        arn = post.resolve_service_task_definition_arn("cluster", "express-app")

    assert arn == "arn:aws:ecs:eu-west-2:123:task-definition/express:3"
    mock_ecs.describe_express_gateway_service.assert_called_once()
    mock_ecs.describe_service_revisions.assert_called_once_with(
        serviceRevisionArns=["arn:aws:ecs:eu-west-2:123:service-revision/rev/1"]
    )


def test_start_express_gateway_service_updates_scaling_target():
    mock_ecs = MagicMock()
    mock_ecs.get_paginator.return_value.paginate.return_value = [
        {
            "serviceArns": [
                "arn:aws:ecs:eu-west-2:123456789012:service/my-cluster/my-express"
            ]
        }
    ]

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        result = post.start_express_gateway_service("my-cluster", "my-express")

    assert result["statusCode"] == 200
    mock_ecs.update_express_gateway_service.assert_called_once_with(
        serviceArn="arn:aws:ecs:eu-west-2:123456789012:service/my-cluster/my-express",
        scalingTarget=post.EXPRESS_GATEWAY_ACTIVE_SCALING_TARGET,
    )


def test_cognito_https_callback_urls():
    assert post.cognito_https_callback_urls(
        "https://abc123.eu-west-2.elb.amazonaws.com"
    ) == [
        "https://abc123.eu-west-2.elb.amazonaws.com",
        "https://abc123.eu-west-2.elb.amazonaws.com/oauth2/idpresponse",
    ]
    assert post.cognito_https_callback_urls("app.example.com")[0].startswith("https://")


def test_update_user_pool_client_callback_urls_preserves_oauth_settings():
    mock_cognito = MagicMock()
    mock_cognito.describe_user_pool_client.return_value = {
        "UserPoolClient": {
            "ClientName": "app-client",
            "CallbackURLs": ["https://old.example.com"],
            "AllowedOAuthFlows": ["code"],
            "AllowedOAuthScopes": ["openid", "email", "profile"],
            "AllowedOAuthFlowsUserPoolClient": True,
            "SupportedIdentityProviders": ["COGNITO"],
            "ExplicitAuthFlows": ["ALLOW_REFRESH_TOKEN_AUTH"],
        }
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_cognito):
        post.update_user_pool_client_callback_urls(
            "pool-1",
            "client-1",
            [
                "https://new.example.com",
                "https://new.example.com/oauth2/idpresponse",
            ],
            aws_region="eu-west-2",
        )

    mock_cognito.update_user_pool_client.assert_called_once()
    kwargs = mock_cognito.update_user_pool_client.call_args.kwargs
    assert kwargs["UserPoolId"] == "pool-1"
    assert kwargs["ClientId"] == "client-1"
    assert kwargs["CallbackURLs"] == [
        "https://new.example.com",
        "https://new.example.com/oauth2/idpresponse",
    ]
    assert kwargs["AllowedOAuthFlows"] == ["code"]
    assert kwargs["AllowedOAuthScopes"] == ["openid", "email", "profile"]


def test_apply_cognito_alb_callback_fixup_skips_when_already_correct():
    mock_cognito = MagicMock()
    mock_cognito.describe_user_pool_client.return_value = {
        "UserPoolClient": {
            "CallbackURLs": post.cognito_https_callback_urls("https://app.example.com"),
        }
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_cognito):
        changed = post.apply_cognito_alb_callback_fixup(
            user_pool_id="pool-1",
            client_id="client-1",
            redirect_base="https://app.example.com",
        )

    assert changed is False
    mock_cognito.update_user_pool_client.assert_not_called()


def test_target_group_arn_from_ecs_register_event():
    message = (
        "(service my-svc) registered 1 targets in "
        "(target-group arn:aws:elasticloadbalancing:eu-west-2:123:"
        "targetgroup/ecs-gateway-tg-abc/def)"
    )
    assert (
        post.target_group_arn_from_ecs_register_event(message)
        == "arn:aws:elasticloadbalancing:eu-west-2:123:targetgroup/ecs-gateway-tg-abc/def"
    )


def test_listener_actions_with_target_group_replaces_forward_arn():
    actions = [
        {"Type": "authenticate-cognito", "Order": 1},
        {
            "Type": "forward",
            "Order": 2,
            "TargetGroupArn": "arn:old",
            "ForwardConfig": {"TargetGroups": [{"TargetGroupArn": "arn:old"}]},
        },
    ]
    updated = post.listener_actions_with_target_group(actions, "arn:new")
    forward = next(action for action in updated if action["Type"] == "forward")
    assert forward["TargetGroupArn"] == "arn:new"
    assert forward["ForwardConfig"]["TargetGroups"][0]["TargetGroupArn"] == "arn:new"


def test_cognito_secret_payload_matches():
    desired = {
        "REDACTION_USER_POOL_ID": "eu-west-2_AAAA",
        "REDACTION_CLIENT_ID": "client",
        "REDACTION_CLIENT_SECRET": "secret",
    }
    current = json.dumps(
        {
            "REDACTION_USER_POOL_ID": "eu-west-2_OLD",
            "REDACTION_CLIENT_ID": "client",
            "REDACTION_CLIENT_SECRET": "secret",
        }
    )
    assert post.cognito_secret_payload_matches(current, desired) is False
    assert post.cognito_secret_payload_matches(json.dumps(desired), desired) is True


def test_listener_rule_has_cognito_auth():
    assert post.listener_rule_has_cognito_auth(
        [{"Type": "authenticate-cognito"}, {"Type": "forward"}]
    )
    assert not post.listener_rule_has_cognito_auth([{"Type": "forward"}])
