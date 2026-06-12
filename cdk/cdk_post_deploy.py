"""
Post-deploy helpers (boto3 only).

Use this module from post_cdk_build_quickstart.py so you do not need Node.js or
aws-cdk-lib installed to start CodeBuild / ECS after deployment.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Union

import boto3
from cdk_config import (
    AWS_REGION,
)

_TASK_DEF_REGISTER_KEYS = (
    "family",
    "taskRoleArn",
    "executionRoleArn",
    "networkMode",
    "containerDefinitions",
    "volumes",
    "placementConstraints",
    "requiresCompatibilities",
    "cpu",
    "memory",
    "pidMode",
    "ipcMode",
    "proxyConfiguration",
    "inferenceAccelerators",
    "ephemeralStorage",
    "runtimePlatform",
)

_CONTAINER_REGISTER_OMIT_KEYS = frozenset(
    {
        "containerArn",
        "taskDefinitionArn",
        "status",
        "lastStatus",
        "managedAgents",
        "networkInterfaces",
        "healthStatus",
        "cpu",
        "memory",
        "gpu",
    }
)


def start_codebuild_build(project_name: str, aws_region: str = AWS_REGION) -> None:
    """Start an existing CodeBuild project build."""
    client = boto3.client("codebuild", region_name=aws_region)

    try:
        print(f"Attempting to start build for project: {project_name}")
        response = client.start_build(projectName=project_name)
        build_id = response["build"]["id"]
        print(f"Successfully started build with ID: {build_id}")
        print(f"Build ARN: {response['build']['arn']}")
        print(
            f"https://{aws_region}.console.aws.amazon.com/codesuite/codebuild/projects/"
            f"{project_name}/build/{build_id.split(':')[-1]}/detail"
        )
    except client.exceptions.ResourceNotFoundException:
        print(f"Error: Project '{project_name}' not found in region '{aws_region}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def upload_file_to_s3(
    local_file_paths: Union[str, List[str]],
    s3_key: str,
    s3_bucket: str,
    run_aws_functions: str = "1",
    aws_region: str = AWS_REGION,
) -> str:
    """Upload local file(s) to S3."""
    final_out_message: List[str] = []
    final_out_message_str = ""

    if run_aws_functions != "1":
        return "App not set to run AWS functions"

    try:
        if not (s3_bucket and local_file_paths):
            return "At least one essential variable is empty, could not upload to S3"

        s3_client = boto3.client("s3", region_name=aws_region)
        paths = (
            [local_file_paths]
            if isinstance(local_file_paths, str)
            else list(local_file_paths)
        )

        for file_path in paths:
            try:
                file_name = os.path.basename(file_path)
                s3_key_full = s3_key + file_name
                print("S3 key: ", s3_key_full)
                s3_client.upload_file(file_path, s3_bucket, s3_key_full)
                out_message = f"File {file_name} uploaded successfully!"
                print(out_message)
            except Exception as e:
                out_message = f"Error uploading file(s): {e}"
                print(out_message)
            final_out_message.append(out_message)

        final_out_message_str = "\n".join(final_out_message)
    except Exception as e:
        final_out_message_str = "Could not upload files to S3 due to: " + str(e)
        print(final_out_message_str)

    return final_out_message_str


def start_ecs_task(
    cluster_name: str,
    service_name: str,
    aws_region: str = AWS_REGION,
) -> dict:
    """Scale a legacy Fargate ECS service to one running task."""
    ecs_client = boto3.client("ecs", region_name=aws_region)

    try:
        ecs_client.update_service(
            cluster=cluster_name, service=service_name, desiredCount=1
        )
        return {
            "statusCode": 200,
            "body": (
                f"Service {service_name} in cluster {cluster_name} "
                "has been updated to 1 task."
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": f"Error updating service: {str(e)}"}


EXPRESS_GATEWAY_ACTIVE_SCALING_TARGET = {
    "minTaskCount": 1,
    "maxTaskCount": 1,
    "autoScalingMetric": "AVERAGE_CPU",
    "autoScalingTargetValue": 60,
}


def resolve_express_gateway_service_arn(
    cluster_name: str,
    service_name: str,
    aws_region: str = AWS_REGION,
) -> str:
    """Look up an Express gateway service ARN by cluster and service name."""
    ecs_client = boto3.client("ecs", region_name=aws_region)
    paginator = ecs_client.get_paginator("list_services")
    for page in paginator.paginate(cluster=cluster_name):
        for arn in page.get("serviceArns", []):
            if arn.rstrip("/").split("/")[-1] == service_name:
                return arn
    raise ValueError(
        f"Express gateway service '{service_name}' not found in cluster "
        f"'{cluster_name}'."
    )


def _task_definition_has_port_name(
    task_definition: Dict[str, Any], port_name: str
) -> bool:
    for container in task_definition.get("containerDefinitions", []):
        for mapping in container.get("portMappings") or []:
            if mapping.get("name") == port_name:
                return True
    return False


def _container_definitions_with_named_port(
    container_definitions: List[Dict[str, Any]],
    *,
    port_name: str,
    container_port: int,
) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    has_matching_port = any(
        mapping.get("containerPort") == container_port
        for container in container_definitions
        for mapping in container.get("portMappings") or []
    )
    for index, container in enumerate(container_definitions):
        container = {
            key: value
            for key, value in container.items()
            if key not in _CONTAINER_REGISTER_OMIT_KEYS
        }
        port_mappings = [
            dict(mapping) for mapping in container.get("portMappings") or []
        ]
        matched = False
        for mapping in port_mappings:
            if mapping.get("containerPort") == container_port:
                matched = True
                mapping["name"] = port_name
                mapping.setdefault("protocol", "tcp")
        if not matched and not has_matching_port and index == 0:
            port_mappings.append(
                {
                    "name": port_name,
                    "containerPort": container_port,
                    "protocol": "tcp",
                }
            )
        container["portMappings"] = port_mappings
        updated.append(container)
    return updated


def resolve_service_task_definition_arn(
    cluster_name: str,
    service_name: str,
    aws_region: str = AWS_REGION,
) -> str:
    """
    Resolve the task definition ARN for a Fargate or Express gateway ECS service.

    Express gateway services omit ``taskDefinition`` on ``describe_services``; use the
    active service revision from ``describe_express_gateway_service`` instead.
    """
    ecs_client = boto3.client("ecs", region_name=aws_region)
    services = ecs_client.describe_services(
        cluster=cluster_name, services=[service_name]
    ).get("services", [])
    if services:
        task_definition_arn = services[0].get("taskDefinition")
        if task_definition_arn:
            return task_definition_arn
        service_arn = services[0].get("serviceArn")
    else:
        service_arn = None

    if not service_arn:
        service_arn = resolve_express_gateway_service_arn(
            cluster_name, service_name, aws_region
        )

    express = ecs_client.describe_express_gateway_service(serviceArn=service_arn)
    active_configs = (express.get("service") or {}).get("activeConfigurations") or []
    if not active_configs:
        raise ValueError(
            f"Could not resolve task definition for service '{service_name}' in "
            f"cluster '{cluster_name}' (no active Express gateway configuration)."
        )
    revision_arn = active_configs[0].get("serviceRevisionArn")
    if not revision_arn:
        raise ValueError(
            f"Could not resolve task definition for service '{service_name}' "
            "(active Express configuration has no serviceRevisionArn)."
        )
    revisions = ecs_client.describe_service_revisions(
        serviceRevisionArns=[revision_arn]
    ).get("serviceRevisions", [])
    if not revisions:
        raise ValueError(
            f"Service revision '{revision_arn}' not found for service "
            f"'{service_name}'."
        )
    task_definition_arn = revisions[0].get("taskDefinition")
    if not task_definition_arn:
        raise ValueError(
            f"Service revision '{revision_arn}' has no taskDefinition for service "
            f"'{service_name}'."
        )
    return task_definition_arn


def ensure_ecs_service_port_mapping_name(
    cluster_name: str,
    service_name: str,
    port_name: str,
    container_port: int,
    aws_region: str = AWS_REGION,
) -> str:
    """
    Service Connect requires a named portMapping in the task definition.
    Express gateway services only set containerPort at create time.
    """
    ecs_client = boto3.client("ecs", region_name=aws_region)
    task_definition_arn = resolve_service_task_definition_arn(
        cluster_name, service_name, aws_region
    )
    task_definition = ecs_client.describe_task_definition(
        taskDefinition=task_definition_arn
    )["taskDefinition"]
    if _task_definition_has_port_name(task_definition, port_name):
        return task_definition_arn

    new_containers = _container_definitions_with_named_port(
        task_definition["containerDefinitions"],
        port_name=port_name,
        container_port=container_port,
    )
    register_kwargs = {
        key: copy.deepcopy(task_definition[key])
        for key in _TASK_DEF_REGISTER_KEYS
        if key in task_definition
    }
    register_kwargs["containerDefinitions"] = new_containers
    if task_definition.get("tags"):
        register_kwargs["tags"] = [
            {"key": tag["key"], "value": tag["value"]}
            for tag in task_definition["tags"]
        ]

    new_task_definition = ecs_client.register_task_definition(**register_kwargs)[
        "taskDefinition"
    ]
    new_arn = new_task_definition["taskDefinitionArn"]
    ecs_client.update_service(
        cluster=cluster_name,
        service=service_name,
        taskDefinition=new_arn,
        forceNewDeployment=True,
    )
    print(
        f"Registered task definition {new_arn} with Service Connect port "
        f"name {port_name!r} on container port {container_port}."
    )
    return new_arn


def apply_ecs_service_connect(
    cluster_name: str,
    service_name: str,
    service_connect_configuration: Dict[str, Any],
    aws_region: str = AWS_REGION,
) -> None:
    ecs_client = boto3.client("ecs", region_name=aws_region)
    ecs_client.update_service(
        cluster=cluster_name,
        service=service_name,
        serviceConnectConfiguration=service_connect_configuration,
        forceNewDeployment=True,
    )
    print(f"Applied Service Connect to {service_name} in cluster {cluster_name}.")


def configure_express_pi_service_connect(
    cluster_name: str,
    main_service_name: str,
    pi_service_name: str,
    namespace: str,
    main_port_name: str,
    discovery_name: str,
    main_port: int,
    aws_region: str = AWS_REGION,
) -> None:
    """Enable Service Connect for Pi Express -> main Express (post image build)."""
    ensure_ecs_service_port_mapping_name(
        cluster_name,
        main_service_name,
        main_port_name,
        main_port,
        aws_region=aws_region,
    )
    apply_ecs_service_connect(
        cluster_name,
        main_service_name,
        {
            "enabled": True,
            "namespace": namespace,
            "services": [
                {
                    "portName": main_port_name,
                    "discoveryName": discovery_name,
                    "clientAliases": [
                        {"port": int(main_port), "dnsName": discovery_name}
                    ],
                }
            ],
        },
        aws_region=aws_region,
    )
    apply_ecs_service_connect(
        cluster_name,
        pi_service_name,
        {"enabled": True, "namespace": namespace},
        aws_region=aws_region,
    )


def start_express_gateway_service(
    cluster_name: str,
    service_name: str,
    aws_region: str = AWS_REGION,
) -> dict:
    """Scale an ECS Express gateway service to one running task after image build."""
    ecs_client = boto3.client("ecs", region_name=aws_region)

    try:
        service_arn = resolve_express_gateway_service_arn(
            cluster_name, service_name, aws_region=aws_region
        )
        ecs_client.update_express_gateway_service(
            serviceArn=service_arn,
            scalingTarget=EXPRESS_GATEWAY_ACTIVE_SCALING_TARGET,
        )
        return {
            "statusCode": 200,
            "body": (
                f"Express service {service_name} in cluster {cluster_name} "
                "has been updated to run 1 task."
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error updating Express gateway service: {str(e)}",
        }
