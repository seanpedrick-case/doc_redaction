"""Unit tests for cdk_post_deploy.py (boto3 helpers, no live AWS)."""

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
