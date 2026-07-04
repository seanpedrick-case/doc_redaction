"""Tests for custom IAM policy file lists on ECS task / execution roles."""

import json
import sys
from pathlib import Path

from aws_cdk import App, Stack
from aws_cdk import aws_iam as iam

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_config import (  # noqa: E402
    ECS_EXECUTION_ROLE_MANAGED_POLICIES,
    POLICY_FILE_LOCATIONS,
    parse_comma_separated_list,
)
from cdk_functions import (  # noqa: E402
    add_custom_policies,
    resolve_policy_file_paths,
)


def test_parse_comma_separated_list_json_and_plain():
    assert parse_comma_separated_list('["a.json", "b.json"]') == ["a.json", "b.json"]
    assert parse_comma_separated_list("textract_policy.json") == [
        "textract_policy.json"
    ]
    assert parse_comma_separated_list("a.json, b.json") == ["a.json", "b.json"]
    assert parse_comma_separated_list("") == []


def test_policy_file_locations_default_is_list():
    assert isinstance(POLICY_FILE_LOCATIONS, list)
    assert len(POLICY_FILE_LOCATIONS) >= 1
    assert all(str(p).endswith(".json") for p in POLICY_FILE_LOCATIONS)


def test_pi_agentcore_invoke_policy_file_exists():
    policy_path = CDK_DIR / "policies" / "pi_agentcore_invoke_policy.json"
    assert policy_path.is_file()
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    actions = {
        action
        for stmt in data.get("Statement", [])
        for action in stmt.get("Action", [])
    }
    assert "bedrock-agentcore:InvokeAgentRuntime" in actions


def test_execution_role_managed_policies_default_minimal():
    assert "service-role/AmazonECSTaskExecutionRolePolicy" in (
        ECS_EXECUTION_ROLE_MANAGED_POLICIES
    )
    assert "AmazonS3FullAccess" not in ECS_EXECUTION_ROLE_MANAGED_POLICIES


def test_resolve_policy_file_paths_relative_to_cdk_folder(tmp_path):
    policy = tmp_path / "custom.json"
    policy.write_text("{}", encoding="utf-8")
    resolved = resolve_policy_file_paths(["custom.json"], cdk_folder=str(tmp_path))
    assert resolved == [str(policy.resolve())]


def test_add_custom_policies_loads_json_statements(tmp_path):
    policy_path = tmp_path / "extra.json"
    policy_path.write_text(
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "TestAllow",
                        "Effect": "Allow",
                        "Action": "s3:ListBucket",
                        "Resource": "*",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app = App()
    stack = Stack(app, "TestStack")
    role = iam.Role(
        stack,
        "TaskRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    add_custom_policies(stack, role, policy_file_locations=[str(policy_path)])

    template = app.synth().get_stack_by_name("TestStack").template
    policies = template["Resources"]
    inline = [
        r
        for r in policies.values()
        if r.get("Type") == "AWS::IAM::Policy" and "s3:ListBucket" in json.dumps(r)
    ]
    assert inline, "expected inline policy statement from JSON file"
