"""
Post-deploy helpers (boto3 only).

Use this module from post_cdk_build_quickstart.py so you do not need Node.js or
aws-cdk-lib installed to start CodeBuild / ECS after deployment.
"""

from __future__ import annotations

import os
from typing import List, Union

import boto3
from cdk_config import (
    AWS_REGION,
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
    """Scale an ECS service to one running task."""
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
