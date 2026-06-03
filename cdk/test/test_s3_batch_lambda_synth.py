"""Synth assertions for optional S3 batch ECS trigger Lambda."""

import sys
from pathlib import Path

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_batch_trigger_express_mutual_exclusion():
    enable_batch = "True"
    use_express = "True"
    with pytest.raises(ValueError, match="ENABLE_S3_BATCH_ECS_TRIGGER"):
        if enable_batch == "True" and use_express == "True":
            raise ValueError(
                "ENABLE_S3_BATCH_ECS_TRIGGER=True requires the legacy Fargate task definition "
                "for ecs.run_task. Set USE_ECS_EXPRESS_MODE=False or disable the batch trigger."
            )


def test_s3_batch_lambda_synth_resources():
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk import aws_ecs as ecs
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_s3 as s3
    from cdk_functions import create_s3_batch_ecs_trigger_lambda

    app = App()
    stack = Stack(
        app,
        "BatchLambdaTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
    output_bucket = s3.Bucket(stack, "OutputBucket")
    config_bucket = s3.Bucket(stack, "ConfigBucket")
    execution_role = iam.Role(
        stack,
        "ExecutionRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    task_role = iam.Role(
        stack,
        "TaskRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    task_def = ecs.FargateTaskDefinition(
        stack, "TaskDef", memory_limit_mib=2048, cpu=1024
    )
    task_def.add_container(
        "docredaction",
        image=ecs.ContainerImage.from_registry("nginx:latest"),
    )

    lambda_asset = str(CDK_DIR / "config" / "lambda")
    create_s3_batch_ecs_trigger_lambda(
        stack,
        "S3BatchEcsTrigger",
        function_name=None,
        lambda_asset_path=lambda_asset,
        output_bucket=output_bucket,
        config_bucket=config_bucket,
        cluster_name="test-cluster",
        task_definition_arn=task_def.task_definition_arn,
        container_name="docredaction",
        subnet_ids=[vpc.private_subnets[0].subnet_id],
        security_group_id=ec2.SecurityGroup(
            stack, "EcsSg", vpc=vpc, allow_all_outbound=True
        ).security_group_id,
        execution_role=execution_role,
        task_role=task_role,
        env_prefix="input/config/",
        env_suffix=".env",
        input_prefix="input/",
        config_prefix="",
        default_params_key="general-config/batch_defaults.env",
    )

    template = app.synth().get_stack_by_name("BatchLambdaTest").template
    resources = template["Resources"]
    types = {r["Type"] for r in resources.values()}

    assert "AWS::Lambda::Function" in types
    assert "AWS::Lambda::Permission" in types
    assert "Custom::S3BucketNotifications" in types or any(
        r.get("Type") == "AWS::S3::Bucket"
        and "NotificationConfiguration" in r.get("Properties", {})
        for r in resources.values()
    )

    batch_lambda = next(
        r
        for r in resources.values()
        if r["Type"] == "AWS::Lambda::Function"
        and r["Properties"].get("Handler") == "lambda_function.lambda_handler"
    )
    env_vars = batch_lambda["Properties"]["Environment"]["Variables"]
    assert env_vars.get("DEFAULT_DIRECT_MODE_TASK") == "redact"
    assert "RUN_DIRECT_MODE" not in env_vars
    assert env_vars.get("ENV_PREFIX") == "input/config/"
