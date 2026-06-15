"""
Post-deploy helpers (boto3 only).

Use this module from post_cdk_build_quickstart.py so you do not need Node.js or
aws-cdk-lib installed to start CodeBuild / ECS after deployment.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import boto3
from cdk_config import (
    AWS_REGION,
    CLOUDFRONT_DOMAIN,
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


_ALB_COGNITO_CALLBACK_SUFFIX = "/oauth2/idpresponse"

# Fields preserved from describe_user_pool_client when updating CallbackURLs only.
_USER_POOL_CLIENT_UPDATE_PASSTHROUGH_KEYS = (
    "ClientName",
    "RefreshTokenValidity",
    "AccessTokenValidity",
    "IdTokenValidity",
    "TokenValidityUnits",
    "ReadAttributes",
    "WriteAttributes",
    "ExplicitAuthFlows",
    "SupportedIdentityProviders",
    "DefaultRedirectURI",
    "AllowedOAuthFlows",
    "AllowedOAuthScopes",
    "AllowedOAuthFlowsUserPoolClient",
    "AnalyticsConfiguration",
    "PreventUserExistenceErrors",
    "EnableTokenRevocation",
    "EnablePropagateAdditionalUserContextData",
    "AuthSessionValidity",
    "RefreshTokenRotation",
)


def cognito_https_callback_urls(redirect_base: str) -> List[str]:
    """
    ALB authenticate-cognito requires the app URL and ``/oauth2/idpresponse``.
    """
    base = (redirect_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("redirect_base is required for Cognito callback URLs")
    if not base.startswith("https://"):
        base = f"https://{base.lstrip('/')}"
    return [base, f"{base}{_ALB_COGNITO_CALLBACK_SUFFIX}"]


def cognito_callback_urls_match(
    existing_callbacks: List[str],
    desired_callbacks: List[str],
) -> bool:
    return set(existing_callbacks or []) == set(desired_callbacks)


def get_user_pool_client_callback_urls(
    user_pool_id: str,
    client_id: str,
    *,
    aws_region: str = AWS_REGION,
) -> List[str]:
    cognito_client = boto3.client("cognito-idp", region_name=aws_region)
    existing = cognito_client.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=client_id,
    )["UserPoolClient"]
    return list(existing.get("CallbackURLs") or [])


def cognito_alb_callbacks_need_update(
    user_pool_id: str,
    client_id: str,
    redirect_base: str,
    *,
    aws_region: str = AWS_REGION,
) -> bool:
    desired = cognito_https_callback_urls(redirect_base)
    current = get_user_pool_client_callback_urls(
        user_pool_id, client_id, aws_region=aws_region
    )
    return not cognito_callback_urls_match(current, desired)


def update_user_pool_client_callback_urls(
    user_pool_id: str,
    client_id: str,
    callback_urls: List[str],
    *,
    aws_region: str = AWS_REGION,
) -> None:
    """
    Set Cognito app client callback URLs without a CDK redeploy.

    Merges existing client settings from ``describe_user_pool_client`` so OAuth
    flows/scopes and token validity are not reset.
    """
    cognito_client = boto3.client("cognito-idp", region_name=aws_region)
    existing = cognito_client.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=client_id,
    )["UserPoolClient"]

    update_kwargs: Dict[str, Any] = {
        "UserPoolId": user_pool_id,
        "ClientId": client_id,
        "CallbackURLs": callback_urls,
    }
    for key in _USER_POOL_CLIENT_UPDATE_PASSTHROUGH_KEYS:
        value = existing.get(key)
        if value is not None:
            update_kwargs[key] = value
    logout_urls = existing.get("LogoutURLs")
    if logout_urls:
        update_kwargs["LogoutURLs"] = logout_urls

    cognito_client.update_user_pool_client(**update_kwargs)
    print("Updated Cognito app client callback URLs: " + ", ".join(callback_urls))


def apply_cognito_alb_callback_fixup(
    *,
    user_pool_id: str,
    client_id: str,
    redirect_base: str,
    aws_region: str = AWS_REGION,
) -> bool:
    """
    Update Cognito callbacks when they differ from ``redirect_base``.

    Returns True if URLs were updated, False if already correct.
    """
    desired = cognito_https_callback_urls(redirect_base)
    cognito_client = boto3.client("cognito-idp", region_name=aws_region)
    existing = cognito_client.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=client_id,
    )["UserPoolClient"]
    current = existing.get("CallbackURLs") or []
    if cognito_callback_urls_match(current, desired):
        print("Cognito app client callback URLs already match the target endpoint.")
        return False
    update_user_pool_client_callback_urls(
        user_pool_id,
        client_id,
        desired,
        aws_region=aws_region,
    )
    return True


_TARGET_GROUP_REGISTER_EVENT = re.compile(
    r"target-group (arn:aws:elasticloadbalancing:[^\s)]+)",
    re.IGNORECASE,
)

EXPRESS_TARGET_GROUP_RESOLVE_MAX_WAIT_SECONDS = 600
EXPRESS_TARGET_GROUP_RESOLVE_POLL_INTERVAL_SECONDS = 15


def target_group_arn_from_ecs_register_event(message: str) -> Optional[str]:
    """Parse target group ARN from ECS ``registered N targets in (target-group ...)``."""
    if "registered" not in (message or "").lower():
        return None
    match = _TARGET_GROUP_REGISTER_EVENT.search(message)
    return match.group(1) if match else None


def _target_group_arn_from_service_events(
    events: List[Dict[str, Any]],
) -> Optional[str]:
    for event in events:
        target_group_arn = target_group_arn_from_ecs_register_event(
            event.get("message", "")
        )
        if target_group_arn:
            return target_group_arn
    return None


def _task_private_ipv4_addresses(
    ecs_client: Any,
    cluster_name: str,
    service_name: str,
) -> set[str]:
    task_arns = ecs_client.list_tasks(
        cluster=cluster_name, serviceName=service_name
    ).get("taskArns", [])
    if not task_arns:
        return set()
    tasks = ecs_client.describe_tasks(cluster=cluster_name, tasks=task_arns).get(
        "tasks", []
    )
    addresses: set[str] = set()
    for task in tasks:
        for attachment in task.get("attachments", []):
            for detail in attachment.get("details", []):
                if detail.get("name") == "privateIPv4Address" and detail.get("value"):
                    addresses.add(detail["value"])
    return addresses


def _target_group_arn_from_task_ips(
    elbv2_client: Any,
    load_balancer_arn: str,
    task_ips: set[str],
) -> Optional[str]:
    if not task_ips:
        return None
    target_groups = elbv2_client.describe_target_groups(
        LoadBalancerArn=load_balancer_arn
    ).get("TargetGroups", [])
    for target_group in target_groups:
        health = elbv2_client.describe_target_health(
            TargetGroupArn=target_group["TargetGroupArn"]
        ).get("TargetHealthDescriptions", [])
        for description in health:
            target_id = (description.get("Target") or {}).get("Id")
            if target_id in task_ips:
                return target_group["TargetGroupArn"]
    return None


def resolve_express_service_target_group_arn(
    cluster_name: str,
    service_name: str,
    *,
    aws_region: str = AWS_REGION,
    load_balancer_arn: Optional[str] = None,
    max_wait_seconds: int = EXPRESS_TARGET_GROUP_RESOLVE_MAX_WAIT_SECONDS,
    poll_interval_seconds: int = EXPRESS_TARGET_GROUP_RESOLVE_POLL_INTERVAL_SECONDS,
) -> str:
    """
    Target group where Express most recently registered tasks.

    After post-deploy scaling, this ARN can differ from the TG baked into the CDK
    Cognito listener custom resource at deploy time. Polls until a registration
    event appears or running tasks are visible in an Express ALB target group.
    """
    ecs_client = boto3.client("ecs", region_name=aws_region)
    elbv2_client = (
        boto3.client("elbv2", region_name=aws_region) if load_balancer_arn else None
    )
    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    while True:
        attempt += 1
        services = ecs_client.describe_services(
            cluster=cluster_name, services=[service_name]
        ).get("services", [])
        if not services:
            raise ValueError(
                f"ECS service '{service_name}' not found in cluster '{cluster_name}'."
            )
        service = services[0]
        target_group_arn = _target_group_arn_from_service_events(
            service.get("events", [])
        )
        if target_group_arn:
            if attempt > 1:
                print(
                    f"Resolved target group for '{service_name}' "
                    f"after {attempt} poll(s)."
                )
            return target_group_arn

        if elbv2_client and load_balancer_arn:
            task_ips = _task_private_ipv4_addresses(
                ecs_client, cluster_name, service_name
            )
            target_group_arn = _target_group_arn_from_task_ips(
                elbv2_client, load_balancer_arn, task_ips
            )
            if target_group_arn:
                print(
                    f"Resolved target group for '{service_name}' from running task IPs."
                )
                return target_group_arn

        if time.monotonic() >= deadline:
            running = service.get("runningCount", 0)
            desired = service.get("desiredCount", 0)
            raise ValueError(
                f"No target group registration event found for service "
                f"'{service_name}' after {max_wait_seconds}s "
                f"(running {running}/{desired} tasks). "
                "Ensure CodeBuild finished and the service scaled to at least one task."
            )

        if attempt == 1:
            print(
                f"Waiting for target group registration for '{service_name}' "
                f"(up to {max_wait_seconds}s)..."
            )
        time.sleep(poll_interval_seconds)


def find_express_gateway_https_listener(
    *,
    aws_region: str = AWS_REGION,
) -> Dict[str, str]:
    """Return Express-managed ALB HTTPS listener metadata."""
    elbv2 = boto3.client("elbv2", region_name=aws_region)
    for load_balancer in elbv2.describe_load_balancers().get("LoadBalancers", []):
        if not load_balancer["LoadBalancerName"].startswith("ecs-express-gateway-alb"):
            continue
        listeners = elbv2.describe_listeners(
            LoadBalancerArn=load_balancer["LoadBalancerArn"]
        ).get("Listeners", [])
        https_listener = next(
            (listener for listener in listeners if listener.get("Port") == 443),
            None,
        )
        if https_listener:
            return {
                "load_balancer_arn": load_balancer["LoadBalancerArn"],
                "listener_arn": https_listener["ListenerArn"],
                "dns_name": load_balancer["DNSName"],
            }
    raise ValueError(
        "Express gateway ALB (ecs-express-gateway-alb-*) with HTTPS listener not found."
    )


def listener_actions_with_target_group(
    existing_actions: List[Dict[str, Any]],
    target_group_arn: str,
) -> List[Dict[str, Any]]:
    """Copy listener/rule actions, replacing the forward target group ARN."""
    updated_actions: List[Dict[str, Any]] = []
    for action in sorted(existing_actions, key=lambda item: item.get("Order", 0)):
        action_copy = copy.deepcopy(action)
        if action_copy.get("Type") == "forward":
            action_copy["TargetGroupArn"] = target_group_arn
            forward_config = action_copy.setdefault("ForwardConfig", {})
            forward_config["TargetGroups"] = [
                {"TargetGroupArn": target_group_arn, "Weight": 1}
            ]
        updated_actions.append(action_copy)
    return updated_actions


def apply_express_alb_listener_target_group_fixup(
    *,
    cluster_name: str,
    main_service_name: str,
    pi_service_name: Optional[str] = None,
    pi_path_prefixes: Optional[List[str]] = None,
    aws_region: str = AWS_REGION,
) -> bool:
    """
    Point ALB Cognito listener actions at the target groups Express tasks use.

    Express creates fresh target groups when a service scales up after deploy; the
    CDK custom resource may still forward authenticated traffic to an empty TG.
    """
    ingress = find_express_gateway_https_listener(aws_region=aws_region)
    load_balancer_arn = ingress["load_balancer_arn"]
    main_target_group_arn = resolve_express_service_target_group_arn(
        cluster_name,
        main_service_name,
        aws_region=aws_region,
        load_balancer_arn=load_balancer_arn,
    )
    pi_target_group_arn = None
    if pi_service_name:
        try:
            pi_target_group_arn = resolve_express_service_target_group_arn(
                cluster_name,
                pi_service_name,
                aws_region=aws_region,
                load_balancer_arn=load_balancer_arn,
            )
        except ValueError as exc:
            print(f"Note: skipping Pi listener rule TG fixup: {exc}")

    elbv2 = boto3.client("elbv2", region_name=aws_region)
    listener_arn = ingress["listener_arn"]
    listener = elbv2.describe_listeners(ListenerArns=[listener_arn])["Listeners"][0]
    current_default = listener.get("DefaultActions", [])
    current_forward_arn = next(
        (
            action.get("TargetGroupArn")
            for action in current_default
            if action.get("Type") == "forward"
        ),
        None,
    )
    changed = current_forward_arn != main_target_group_arn

    if changed:
        elbv2.modify_listener(
            ListenerArn=listener_arn,
            DefaultActions=listener_actions_with_target_group(
                current_default, main_target_group_arn
            ),
        )
        print(
            "Updated Express ALB default listener forward target group to "
            f"{main_target_group_arn}."
        )
    else:
        print(
            "Express ALB default listener already forwards to the active target group."
        )

    if pi_target_group_arn and pi_path_prefixes:
        rules = elbv2.describe_rules(ListenerArn=listener_arn).get("Rules", [])
        prefixes = {prefix.rstrip("/") for prefix in pi_path_prefixes}
        for rule in rules:
            if rule.get("IsDefault"):
                continue
            path_values = []
            for condition in rule.get("Conditions", []):
                if condition.get("Field") == "path-pattern":
                    path_values.extend(condition.get("Values", []))
            if not prefixes.intersection({value.rstrip("/") for value in path_values}):
                continue
            current_actions = rule.get("Actions", [])
            current_pi_forward = next(
                (
                    action.get("TargetGroupArn")
                    for action in current_actions
                    if action.get("Type") == "forward"
                ),
                None,
            )
            if current_pi_forward == pi_target_group_arn:
                continue
            elbv2.modify_rule(
                RuleArn=rule["RuleArn"],
                Actions=listener_actions_with_target_group(
                    current_actions, pi_target_group_arn
                ),
            )
            print(
                "Updated Pi ALB listener rule forward target group to "
                f"{pi_target_group_arn}."
            )
            changed = True

    return changed


def listener_rule_has_cognito_auth(actions: List[Dict[str, Any]]) -> bool:
    return any(action.get("Type") == "authenticate-cognito" for action in actions)


def _listener_rule_is_cloudfront_bypass_without_cognito(
    rule: Dict[str, Any],
    *,
    cloudfront_host_header: str,
) -> bool:
    """True for legacy forward-only host-header rules matching the CloudFront domain."""
    if listener_rule_has_cognito_auth(rule.get("Actions", [])):
        return False
    host = (cloudfront_host_header or "").strip()
    if not host or host == "cloudfront_placeholder.net":
        return False
    for condition in rule.get("Conditions") or []:
        if condition.get("Field") != "host-header":
            continue
        values = (condition.get("HostHeaderConfig") or {}).get("Values") or []
        if host in values:
            return True
    return False


def remove_express_listener_rules_without_cognito(
    *,
    aws_region: str = AWS_REGION,
    cloudfront_host_header: Optional[str] = None,
) -> bool:
    """
    Delete legacy Express ALB rules that forward CloudFront host traffic without Cognito.

    Only removes host-header rules whose value matches ``CLOUDFRONT_DOMAIN`` (or the
    supplied override). ECS Express-managed ``*.ecs.*.on.aws`` host rules are left
    intact.
    """
    cloudfront_host = (cloudfront_host_header or CLOUDFRONT_DOMAIN or "").strip()
    ingress = find_express_gateway_https_listener(aws_region=aws_region)
    elbv2 = boto3.client("elbv2", region_name=aws_region)
    listener_arn = ingress["listener_arn"]
    changed = False
    for rule in elbv2.describe_rules(ListenerArn=listener_arn).get("Rules", []):
        if rule.get("IsDefault"):
            continue
        if not _listener_rule_is_cloudfront_bypass_without_cognito(
            rule, cloudfront_host_header=cloudfront_host
        ):
            continue
        rule_arn = rule["RuleArn"]
        elbv2.delete_rule(RuleArn=rule_arn)
        print(
            "Removed legacy Express CloudFront bypass ALB listener rule "
            f"(host {cloudfront_host!r}): {rule_arn}"
        )
        changed = True
    return changed


def build_cognito_secret_payload(
    user_pool_id: str,
    client_id: str,
    *,
    aws_region: str = AWS_REGION,
) -> Dict[str, str]:
    """Build Secrets Manager JSON for REDACTION_* Cognito keys."""
    cognito_client = boto3.client("cognito-idp", region_name=aws_region)
    client = cognito_client.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=client_id,
    )["UserPoolClient"]
    client_secret = client.get("ClientSecret") or ""
    return {
        "REDACTION_USER_POOL_ID": user_pool_id,
        "REDACTION_CLIENT_ID": client_id,
        "REDACTION_CLIENT_SECRET": client_secret,
    }


def cognito_secret_payload_matches(
    existing_secret_string: str,
    desired_payload: Dict[str, str],
) -> bool:
    try:
        current = json.loads(existing_secret_string or "{}")
    except json.JSONDecodeError:
        return False
    return all(current.get(key) == value for key, value in desired_payload.items())


def apply_cognito_secret_fixup(
    *,
    secret_name: str,
    user_pool_id: str,
    client_id: str,
    aws_region: str = AWS_REGION,
    recycle_express_service: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Sync imported Secrets Manager JSON with the stack's Cognito pool and app client.

    Express tasks read ``AWS_USER_POOL_ID`` / ``AWS_CLIENT_*`` from this secret.
    When the secret predates a redeploy, values can reference a deleted user pool.
    """
    desired_payload = build_cognito_secret_payload(
        user_pool_id, client_id, aws_region=aws_region
    )
    secrets_client = boto3.client("secretsmanager", region_name=aws_region)
    current = secrets_client.get_secret_value(SecretId=secret_name)
    current_string = current.get("SecretString") or ""
    if cognito_secret_payload_matches(current_string, desired_payload):
        print(
            "Cognito secret already matches Cognito user pool ID "
            "and Cognito app client ID."
        )
        return False

    secrets_client.put_secret_value(
        SecretId=secret_name,
        SecretString=json.dumps(desired_payload),
    )
    print(
        "Updated Cognito secret for Cognito user pool ID " "and Cognito app client ID."
    )
    if recycle_express_service:
        recycle_express_gateway_tasks(
            recycle_express_service["cluster_name"],
            recycle_express_service["service_name"],
            aws_region=aws_region,
        )
    return True


def recycle_express_gateway_tasks(
    cluster_name: str,
    service_name: str,
    *,
    aws_region: str = AWS_REGION,
) -> None:
    """Stop running Express tasks so replacements pick up updated secrets/env."""
    ecs_client = boto3.client("ecs", region_name=aws_region)
    task_arns = ecs_client.list_tasks(
        cluster=cluster_name,
        serviceName=service_name,
    ).get("taskArns", [])
    for task_arn in task_arns:
        ecs_client.stop_task(
            cluster=cluster_name,
            task=task_arn,
            reason="Recycle task after Cognito secret/config sync",
        )
    if task_arns:
        print(
            f"Stopped {len(task_arns)} task(s) for {service_name} to pick up Cognito updates."
        )


def apply_express_disable_in_app_cognito_auth(
    cluster_name: str,
    service_name: str,
    *,
    aws_region: str = AWS_REGION,
) -> bool:
    """
    Set ``COGNITO_AUTH=False`` on a running Express service revision.

    ALB ``authenticate-cognito`` already gates access; in-app Gradio login is redundant
    and fails when Secrets Manager still references an old user pool.
    """
    ecs_client = boto3.client("ecs", region_name=aws_region)
    service_arn = resolve_express_gateway_service_arn(
        cluster_name, service_name, aws_region=aws_region
    )
    express = ecs_client.describe_express_gateway_service(serviceArn=service_arn)[
        "service"
    ]
    active_configs = express.get("activeConfigurations") or []
    if not active_configs:
        raise ValueError(
            f"No active configuration for Express service '{service_name}'."
        )
    active = active_configs[0]
    primary = copy.deepcopy(active.get("primaryContainer") or {})
    environment = {
        item["name"]: item["value"]
        for item in primary.get("environment") or []
        if item.get("name")
    }
    if environment.get("COGNITO_AUTH") == "False":
        print(f"{service_name} already has COGNITO_AUTH=False.")
        return False
    environment["COGNITO_AUTH"] = "False"
    primary["environment"] = [
        {"name": name, "value": value} for name, value in sorted(environment.items())
    ]
    update_kwargs: Dict[str, Any] = {
        "serviceArn": service_arn,
        "primaryContainer": primary,
    }
    for key in (
        "executionRoleArn",
        "taskRoleArn",
        "cpu",
        "memory",
        "healthCheckPath",
        "networkConfiguration",
    ):
        value = active.get(key)
        if value is not None:
            update_kwargs[key] = value
    scaling = express.get("scalingTarget") or active.get("scalingTarget")
    if scaling is not None:
        update_kwargs["scalingTarget"] = scaling
    ecs_client.update_express_gateway_service(**update_kwargs)
    print(f"Set COGNITO_AUTH=False on Express service {service_name}.")
    return True


def apply_cognito_secret_fixup_from_stack(
    *,
    stack_name: str,
    secret_name: str,
    cluster_name: str,
    main_service_name: str,
    aws_region: str = AWS_REGION,
    recycle_tasks: bool = True,
) -> bool:
    """Read Cognito outputs from CloudFormation and sync the app client secret."""
    cfn_client = boto3.client("cloudformation", region_name=aws_region)
    stacks = cfn_client.describe_stacks(StackName=stack_name).get("Stacks", [])
    outputs = {
        item["OutputKey"]: item["OutputValue"]
        for item in (stacks[0].get("Outputs") or [])
    }
    user_pool_id = outputs.get("CognitoPoolId")
    client_id = outputs.get("CognitoAppClientId")
    if not user_pool_id or not client_id:
        raise ValueError(
            f"Stack '{stack_name}' is missing CognitoPoolId or CognitoAppClientId outputs."
        )
    return apply_cognito_secret_fixup(
        secret_name=secret_name,
        user_pool_id=user_pool_id,
        client_id=client_id,
        aws_region=aws_region,
        recycle_express_service=(
            {"cluster_name": cluster_name, "service_name": main_service_name}
            if recycle_tasks
            else None
        ),
    )


def get_stack_output(
    stack_name: str,
    output_key: str,
    region: str,
) -> Optional[str]:
    """Return a CloudFormation stack output value, or None if missing."""
    from botocore.exceptions import ClientError

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        response = cfn.describe_stacks(StackName=stack_name)
    except ClientError:
        return None
    for stack in response.get("Stacks", []):
        for output in stack.get("Outputs", []):
            if output.get("OutputKey") == output_key:
                return output.get("OutputValue")
    return None


def print_express_mode_next_steps(
    values: Dict[str, str],
    *,
    stack_name: str = "RedactionStack",
    region: Optional[str] = None,
) -> None:
    """Print user-facing next steps after Express mode deploy + quickstart."""
    from cdk_config import normalize_https_redirect_url
    from cdk_functions import format_express_pi_public_url

    aws_region = region or values.get("AWS_REGION") or AWS_REGION

    main_raw = (values.get("ECS_EXPRESS_COGNITO_REDIRECT_BASE") or "").strip()
    if not main_raw:
        main_raw = (
            get_stack_output(stack_name, "ExpressServiceEndpoint", aws_region) or ""
        )
    main_url = normalize_https_redirect_url(main_raw) if main_raw else ""

    print("\nDone. Next steps:")
    print("  - Wait 10 minutes for app deployment to finish.")
    pi_express = values.get("ENABLE_PI_AGENT_EXPRESS_SERVICE") == "True"
    if pi_express:
        print(
            "  - Register a Cognito user in AWS Console and sign in at the Pi agent URL "
            "below. The main redaction backend runs without in-app login so the Pi agent "
            "can call it over Service Connect."
        )
    else:
        print(
            "  - If you have enabled Cognito authorisation in app, register a new user "
            "to your Cognito user pool in AWS Console and complete sign up with the app "
            "client login. If you do not want Cognito login, then set COGNITO_AUTH to "
            "False in the ECS task definition / ECS service options."
        )
    if main_url:
        print(f"  - The main redaction app can be accessed at {main_url}")
    else:
        print("  - The main redaction app URL: see ExpressServiceEndpoint stack output")

    if values.get("ENABLE_PI_AGENT_EXPRESS_SERVICE") == "True":
        pi_raw = get_stack_output(stack_name, "PiExpressEndpoint", aws_region) or ""
        pi_url = (
            format_express_pi_public_url(normalize_https_redirect_url(pi_raw))
            if pi_raw
            else ""
        )
        if pi_url:
            print(f"  - The Pi agentic redaction app can be accessed at {pi_url}")
        else:
            print(
                "  - The Pi agentic redaction app URL: see PiExpressEndpoint stack output"
            )


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def seed_headless_batch_s3_layout(
    output_bucket: str,
    *,
    input_prefix: str = "input/",
    env_prefix: str = "input/config/",
    example_env_local_path: str,
    example_env_basename: str = "example_headless_env_file.env",
    aws_region: str = AWS_REGION,
) -> None:
    """
    Ensure headless batch prefixes and example job .env exist on the output bucket.

    Idempotent: existing keys are left unchanged (safe to run from quickstart).
    """
    if not output_bucket:
        print("Skipping headless S3 layout seed: output bucket name is empty.")
        return

    input_prefix_norm = (
        input_prefix if input_prefix.endswith("/") else f"{input_prefix}/"
    )
    env_prefix_norm = env_prefix if env_prefix.endswith("/") else f"{env_prefix}/"
    if not env_prefix_norm.startswith(input_prefix_norm):
        env_prefix_norm = f"{input_prefix_norm}{env_prefix_norm.lstrip('/')}"
        if not env_prefix_norm.endswith("/"):
            env_prefix_norm += "/"

    s3_client = boto3.client("s3", region_name=aws_region)
    markers = (
        input_prefix_norm,
        env_prefix_norm,
    )
    for key in markers:
        if _s3_object_exists(s3_client, output_bucket, key):
            print(f"S3 prefix marker already present: s3://{output_bucket}/{key}")
            continue
        s3_client.put_object(Bucket=output_bucket, Key=key, Body=b"")
        print(f"Created S3 prefix marker: s3://{output_bucket}/{key}")

    example_key = f"{env_prefix_norm}{example_env_basename}"
    if _s3_object_exists(s3_client, output_bucket, example_key):
        print(f"Example job .env already present: s3://{output_bucket}/{example_key}")
        return

    if not os.path.isfile(example_env_local_path):
        print(f"Skipping example job .env upload: {example_env_local_path} not found.")
        return

    with open(example_env_local_path, "rb") as handle:
        s3_client.put_object(Bucket=output_bucket, Key=example_key, Body=handle.read())
    print(f"Uploaded example job .env to s3://{output_bucket}/{example_key}")


def print_headless_deployment_next_steps(
    values: Dict[str, str],
    *,
    stack_name: str = "RedactionStack",
    region: Optional[str] = None,
) -> None:
    """Print user-facing next steps after headless deploy + quickstart."""
    aws_region = region or values.get("AWS_REGION") or AWS_REGION
    output_bucket = (values.get("S3_OUTPUT_BUCKET_NAME") or "").strip()
    input_prefix = (values.get("S3_BATCH_INPUT_PREFIX") or "input/").strip("/")
    config_prefix = (values.get("S3_BATCH_ENV_PREFIX") or "input/config/").strip("/")

    lambda_name = (values.get("S3_BATCH_LAMBDA_FUNCTION_NAME") or "").strip()
    if not lambda_name:
        lambda_name = (
            get_stack_output(stack_name, "BatchEcsTriggerLambdaName", aws_region) or ""
        ).strip()
    if not lambda_name:
        prefix = (values.get("CDK_PREFIX") or "").strip()
        lambda_name = f"{prefix}S3BatchEcsTrigger" if prefix else "S3BatchEcsTrigger"

    input_uri = (
        f"s3://{output_bucket}/{input_prefix}/"
        if output_bucket
        else f"<output-bucket>/{input_prefix}/"
    )
    config_uri = (
        f"s3://{output_bucket}/{config_prefix}/"
        if output_bucket
        else f"<output-bucket>/{config_prefix}/"
    )
    output_uri = (
        f"s3://{output_bucket}/output/<session-folder>/"
        if output_bucket
        else "<output-bucket>/output/<session-folder>/"
    )
    example_name = "example_headless_env_file.env"

    print("\nDone. Next steps:")
    print(f"  - Upload a PDF file for redaction to {input_uri}")
    print(
        f"  - Create a .env file for submitting a task. You can use '{example_name}' "
        f"(also at {config_uri}{example_name}) and change the PDF filename inside. "
        "Further direct-mode config variables are in tools/config.py (DIRECT_MODE_*)."
    )
    print(f"  - Upload your job .env file to {config_uri}")
    print(
        f"  - AWS Lambda function {lambda_name} will start an ECS task to redact your "
        "document according to your instructions."
    )
    print(
        f"  - Redaction outputs are written to {output_uri} when SAVE_OUTPUTS_TO_S3=True "
        "in your job .env (see the example file)."
    )
    log_group = (values.get("ECS_LOG_GROUP_NAME") or "").strip()
    if log_group:
        print(
            f"  - Batch task logs: CloudWatch log group {log_group} "
            "(streams appear once the container starts)."
        )
