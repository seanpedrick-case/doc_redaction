"""Synth assertions for the main-app AgentCore Harness (inline tools + S3 skill)."""

import json
import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CDK_DIR.parent
sys.path.insert(0, str(CDK_DIR))


def _tool_specs():
    return json.loads(
        (REPO_ROOT / "tools" / "agentcore_harness_tools.json").read_text(
            encoding="utf-8"
        )
    )


def _synth_harness(*, with_kms: bool = False):
    from aws_cdk import App, Environment, Stack
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_kms as kms
    from aws_cdk import aws_s3 as s3
    from aws_cdk.assertions import Template
    from cdk_functions import create_agentcore_harness

    app = App()
    stack = Stack(
        app,
        "HarnessTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    bucket = s3.Bucket(stack, "ConfigBucket")
    key = kms.Key(stack, "Key") if with_kms else None
    harness = create_agentcore_harness(
        stack,
        "RedactionHarness",
        harness_name="demo-Redaction-Harness",
        model_id="eu.anthropic.claude-sonnet-4-6",
        inline_tool_specs=_tool_specs(),
        skills_bucket=bucket,
        skills_source_directory=str(REPO_ROOT / "skills" / "doc-redaction-harness"),
        skills_s3_prefix="agentcore-harness/skills/",
        region="eu-west-2",
        account="123456789012",
        kms_key=key,
    )
    task_role = iam.Role(
        stack, "TaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
    )
    task_role.add_to_principal_policy(
        iam.PolicyStatement(
            actions=["bedrock-agentcore:InvokeHarness"],
            resources=[harness.get_att("Arn").to_string()],
        )
    )
    return Template.from_stack(stack)


def _all_actions(template) -> set[str]:
    actions: set[str] = set()
    for policy in template.find_resources("AWS::IAM::Policy").values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            value = statement.get("Action", [])
            actions.update([value] if isinstance(value, str) else value)
    return actions


def test_harness_resource_properties():
    template = _synth_harness()
    template.resource_count_is("AWS::BedrockAgentCore::Harness", 1)
    harness = next(
        iter(template.find_resources("AWS::BedrockAgentCore::Harness").values())
    )
    props = harness["Properties"]

    assert props["HarnessName"] == "demo_Redaction_Harness"
    assert props["Model"]["BedrockModelConfig"]["ModelId"] == (
        "eu.anthropic.claude-sonnet-4-6"
    )
    assert props["Memory"] == {"Disabled": {}}

    tool_names = {t["Name"] for t in props["Tools"]}
    assert {"doc_redact", "review_apply", "finish_task"} <= tool_names
    for tool in props["Tools"]:
        assert tool["Type"] == "inline_function"
        inline = tool["Config"]["InlineFunction"]
        assert inline["Description"]
        assert inline["InputSchema"]["type"] == "object"

    assert "subagent" in props["AllowedTools"]
    assert "web_fetch" not in props["AllowedTools"]
    assert tool_names <= set(props["AllowedTools"])

    skills_uri = json.dumps(props["Skills"])
    assert "agentcore-harness/skills/doc-redaction-harness/" in skills_uri


def test_harness_roles_and_skill_deployment():
    template = _synth_harness()
    template.resource_count_is("Custom::CDKBucketDeployment", 1)
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "AssumeRolePolicyDocument": {
                "Statement": [
                    {
                        "Action": "sts:AssumeRole",
                        "Effect": "Allow",
                        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    }
                ]
            }
        },
    )
    actions = _all_actions(template)
    assert "bedrock:InvokeModelWithResponseStream" in actions
    assert "bedrock-agentcore:InvokeHarness" in actions
    assert "s3:GetObject" in actions


def test_harness_role_gets_kms_decrypt_with_custom_key():
    assert "kms:Decrypt" in _all_actions(_synth_harness(with_kms=True))


def test_harness_inline_tools_to_cfn_preserves_schema():
    from cdk_functions import harness_inline_tools_to_cfn

    specs = _tool_specs()
    cfn = harness_inline_tools_to_cfn(specs)
    assert len(cfn) == len(specs)
    assert cfn[0]["Config"]["InlineFunction"]["InputSchema"] == (
        specs[0]["config"]["inlineFunction"]["inputSchema"]
    )


def test_sanitize_harness_name():
    from cdk_functions import sanitize_harness_name

    assert sanitize_harness_name("demo-RedactionHarness") == "demo_RedactionHarness"
    assert sanitize_harness_name("1abc")[0].isalpha()
    assert len(sanitize_harness_name("x" * 80)) == 40


def test_default_harness_model_id_passes_through():
    from cdk_config import default_harness_model_id

    assert default_harness_model_id("anthropic.claude-sonnet-5", "eu-west-2") == (
        "anthropic.claude-sonnet-5"
    )
    assert default_harness_model_id("", "us-east-1") == "anthropic.claude-sonnet-5"
    assert default_harness_model_id("eu.anthropic.x", "eu-west-2") == "eu.anthropic.x"


def test_installer_writes_harness_flag_and_model():
    import argparse

    import cdk_install as inst

    answers = inst.InstallAnswers(profile="production")
    assert inst.build_env_values(answers)["ENABLE_AGENTCORE_HARNESS"] == "False"

    args = argparse.Namespace(
        enable_agentcore_harness=True,
        agentcore_harness_model="eu.anthropic.claude-opus-4-6",
    )
    inst.configure_agentcore_harness_options(answers, args, interactive=False)
    values = inst.build_env_values(answers)
    assert values["ENABLE_AGENTCORE_HARNESS"] == "True"
    assert values["AGENTCORE_HARNESS_MODEL_ID"] == "eu.anthropic.claude-opus-4-6"


def test_installer_parser_accepts_harness_flags():
    import cdk_install as inst

    args = inst.build_arg_parser().parse_args(
        ["--enable-agentcore-harness", "--agentcore-harness-model", "eu.x"]
    )
    assert args.enable_agentcore_harness is True
    assert args.agentcore_harness_model == "eu.x"


def test_agentcore_harness_invoke_policy_file():
    data = json.loads(
        (CDK_DIR / "policies" / "agentcore_harness_invoke_policy.json").read_text(
            encoding="utf-8"
        )
    )
    actions = {a for s in data["Statement"] for a in s["Action"]}
    assert actions == {"bedrock-agentcore:InvokeHarness"}
