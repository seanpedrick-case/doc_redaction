"""Stack-wide tagging: value derived from CDK_PREFIX and propagated to resources."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_config import _default_stack_tag_value


def test_default_stack_tag_value_lowercases_and_trims():
    assert _default_stack_tag_value("Demo-Redaction-") == "demo-redaction"
    assert _default_stack_tag_value("Production-Redaction-") == "production-redaction"


def test_default_stack_tag_value_normalises_separators():
    assert _default_stack_tag_value("Demo_Redaction") == "demo-redaction"
    assert _default_stack_tag_value("My Org Redaction ") == "my-org-redaction"


def test_default_stack_tag_value_empty_prefix():
    assert _default_stack_tag_value("") == ""
    assert _default_stack_tag_value(None) == ""


def test_stack_tag_propagates_to_taggable_resources():
    from aws_cdk import App, Environment, Stack, Tags, assertions
    from aws_cdk import aws_s3 as s3

    app = App()
    Tags.of(app).add("Project", "demo-redaction")
    stack = Stack(
        app,
        "TagTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    s3.Bucket(stack, "Bucket")

    template = assertions.Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "Tags": assertions.Match.array_with(
                [{"Key": "Project", "Value": "demo-redaction"}]
            )
        },
    )
