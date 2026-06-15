"""CodeBuild public GitHub source helpers."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from aws_cdk import App, Stack
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_iam as iam
from aws_cdk.assertions import Template
from cdk_functions import (
    configure_public_github_codebuild_source,
    public_github_codebuild_source,
    public_github_repository_url,
)


def test_public_github_repository_url():
    assert (
        public_github_repository_url("seanpedrick-case", "doc_redaction")
        == "https://github.com/seanpedrick-case/doc_redaction.git"
    )


def test_public_github_codebuild_source_synth_without_auth():
    app = App()
    stack = Stack(app, "Test")
    role = iam.Role(
        stack, "Role", assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com")
    )
    project = codebuild.Project(
        stack,
        "Proj",
        role=role,
        source=public_github_codebuild_source("owner", "repo", "main"),
        build_spec=codebuild.BuildSpec.from_object(
            {"version": "0.2", "phases": {"build": {"commands": ["echo hi"]}}}
        ),
    )
    configure_public_github_codebuild_source(project, "owner", "repo")

    template = Template.from_stack(stack)
    resources = template.to_json()["Resources"]
    project_resource = next(
        value
        for value in resources.values()
        if value["Type"] == "AWS::CodeBuild::Project"
    )
    source = project_resource["Properties"]["Source"]
    triggers = project_resource["Properties"]["Triggers"]

    assert "Auth" not in source
    assert source["Location"] == "https://github.com/owner/repo.git"
    assert source["ReportBuildStatus"] is False
    assert source["Type"] == "GITHUB"
    assert triggers["Webhook"] is False
