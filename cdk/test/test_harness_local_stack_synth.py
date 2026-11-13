"""Synth tests for the harness-only stack (local app.py)."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_harness_local_stack_synth():
    from aws_cdk import App, Environment
    from aws_cdk.assertions import Template
    from harness_local_stack import HarnessLocalStack

    app = App()
    stack = HarnessLocalStack(
        app,
        "DocRedactionHarnessLocal",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::BedrockAgentCore::Harness", 1)
    template.resource_count_is("AWS::S3::Bucket", 2)
    template.resource_count_is("Custom::CDKBucketDeployment", 1)
    template.resource_count_is("AWS::IAM::ManagedPolicy", 1)

    policies = template.find_resources("AWS::IAM::ManagedPolicy")
    doc = next(iter(policies.values()))["Properties"]["PolicyDocument"]
    actions = {
        a
        for stmt in doc["Statement"]
        for a in (
            [stmt["Action"]] if isinstance(stmt["Action"], str) else stmt["Action"]
        )
    }
    assert "bedrock-agentcore:InvokeHarness" in actions
    assert "s3:PutObject" in actions

    template.has_output("RedactionHarnessArn", {})
    template.has_output("LocalAppConfigSnippet", {})


def test_harness_local_config_default_model():
    import harness_local_config as cfg

    assert (
        cfg.default_harness_model_id("anthropic.claude-sonnet-5", "eu-west-2")
        == "anthropic.claude-sonnet-5"
    )
    assert cfg.default_harness_model_id("", "eu-west-2") == "anthropic.claude-sonnet-5"
