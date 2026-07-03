"""Tests for CDK-native Bedrock AgentCore runtime (ECR + CfnRuntime)."""

import argparse
import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))


def test_derive_agentcore_runtime_url_encodes_arn():
    from cdk_functions import derive_agentcore_runtime_url

    arn = "arn:aws:bedrock-agentcore:eu-west-2:123456789012:runtime/RedactionAgent"
    url = derive_agentcore_runtime_url(arn, "eu-west-2")
    assert url.startswith("https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/")
    assert "%3A" in url
    assert "%2F" in url
    assert "/invocations" not in url


def test_normalize_agentcore_runtime_url_strips_invocations():
    from cdk_functions import normalize_agentcore_runtime_url

    assert (
        normalize_agentcore_runtime_url("https://runtime.example/invocations")
        == "https://runtime.example"
    )


def test_build_env_values_agentcore_cdk_deploy_flags():
    import cdk_install as inst

    answers = inst.InstallAnswers(profile="demo", enable_agentic_express=True)
    answers.agent_orchestrator = "agentcore"
    answers.enable_agentcore_cdk_deploy = True
    values = inst.build_env_values(answers)
    assert values["AGENTCORE_CDK_DEPLOY"] == "True"
    assert values["ENABLE_AGENTCORE_RUNTIME"] == "True"
    assert "AGENTCORE_RUNTIME_URL" not in values or not values["AGENTCORE_RUNTIME_URL"]


def test_build_env_values_agentcore_cdk_runtime_flags():
    import cdk_install as inst

    answers = inst.InstallAnswers(profile="demo", enable_agentic_express=True)
    answers.enable_agentcore_cdk_runtime = True
    values = inst.build_env_values(answers)
    assert values["ENABLE_AGENTCORE_CDK_RUNTIME"] == "True"
    assert values["AGENTCORE_CDK_DEPLOY"] == "True"


def test_validate_env_values_allows_cdk_deploy_without_url():
    import cdk_install as inst

    answers = inst.InstallAnswers(
        profile="demo",
        aws_account_id="123456789012",
        aws_region="eu-west-2",
        cdk_prefix="Test-Redaction-",
        cognito_domain_prefix="test-redaction",
        vpc_mode="existing",
        vpc_name="test-vpc",
        enable_agentic_express=True,
    )
    answers.agent_orchestrator = "agentcore"
    answers.enable_agentcore_cdk_deploy = True
    values = inst.build_env_values(answers)
    assert values["AGENTCORE_CDK_DEPLOY"] == "True"
    assert inst.validate_env_values(values, allow_empty_agentcore_url=False) == []


def test_validate_install_answers_allows_cdk_deploy_without_url():
    import cdk_install as inst

    answers = inst.InstallAnswers(profile="demo", enable_agentic_express=True)
    answers.agent_orchestrator = "agentcore"
    answers.enable_agentcore_cdk_deploy = True
    assert inst.validate_install_answers(answers) == []


def test_patch_env_key_values_merges_updates(tmp_path):
    from cdk_post_deploy import _patch_env_key_values

    env_file = tmp_path / "agent.env"
    env_file.write_text(
        "AGENT_ORCHESTRATOR=pi\nDOC_REDACTION_GRADIO_URL=https://old.example\n",
        encoding="utf-8",
    )
    _patch_env_key_values(
        env_file,
        {
            "AGENT_ORCHESTRATOR": "agentcore",
            "AGENTCORE_RUNTIME_URL": "https://runtime.example",
        },
    )
    text = env_file.read_text(encoding="utf-8")
    assert "AGENT_ORCHESTRATOR=agentcore" in text
    assert "AGENTCORE_RUNTIME_URL=https://runtime.example" in text
    assert "DOC_REDACTION_GRADIO_URL=https://old.example" in text


def test_wait_for_codebuild_build_succeeds(monkeypatch):
    from cdk_post_deploy import wait_for_codebuild_build

    statuses = iter(["IN_PROGRESS", "SUCCEEDED"])

    class FakeClient:
        def batch_get_builds(self, *, ids):
            return {"builds": [{"buildStatus": next(statuses)}]}

    monkeypatch.setattr(
        "cdk_post_deploy.boto3.client",
        lambda *a, **k: FakeClient(),
    )
    monkeypatch.setattr("cdk_post_deploy.time.sleep", lambda _s: None)

    assert wait_for_codebuild_build("proj:build-id", timeout_sec=60) is True


def test_wait_for_agentcore_ecr_image_uses_existing_image(monkeypatch):
    from cdk_post_deploy import wait_for_agentcore_ecr_image

    monkeypatch.setattr(
        "cdk_post_deploy.ecr_image_with_tag_exists",
        lambda *a, **k: True,
    )
    assert (
        wait_for_agentcore_ecr_image(
            repository_name="repo",
            codebuild_project="proj",
        )
        is True
    )


def test_maybe_complete_agentcore_skips_when_image_not_ready(monkeypatch, tmp_path):
    import cdk_install as inst

    env_path = tmp_path / "cdk_config.env"
    env_path.write_text("AGENTCORE_CDK_DEPLOY=True\n", encoding="utf-8")
    monkeypatch.setattr(inst, "ENV_PATH", env_path)
    monkeypatch.setattr(
        "cdk_post_deploy.wait_for_agentcore_ecr_image",
        lambda **k: False,
    )
    monkeypatch.setattr(inst, "run_cdk_command", lambda *a, **k: None)

    inst.maybe_complete_agentcore_cdk_deploy(
        {"AGENTCORE_CDK_DEPLOY": "True", "USE_ECS_EXPRESS_MODE": "True"},
        argparse.Namespace(config_only=False),
        assume_yes=False,
    )
    assert "ENABLE_AGENTCORE_CDK_RUNTIME" not in env_path.read_text(encoding="utf-8")


def test_sync_agentcore_runtime_url_from_stack_patches_env(tmp_path, monkeypatch):
    import cdk_post_deploy as post

    pi_env = tmp_path / "agent.env"
    cdk_env = tmp_path / "cdk_config.env"
    pi_env.write_text("AGENT_ORCHESTRATOR=pi\n", encoding="utf-8")
    cdk_env.write_text("ENABLE_AGENTCORE_RUNTIME=False\n", encoding="utf-8")

    arn = "arn:aws:bedrock-agentcore:eu-west-2:123:runtime/TestAgent"
    monkeypatch.setattr(post, "get_stack_output", lambda *a, **k: arn)
    monkeypatch.setattr(post, "upload_file_to_s3", lambda *a, **k: None)
    monkeypatch.setattr(post, "recycle_express_gateway_tasks", lambda *a, **k: None)
    monkeypatch.setattr(
        "cdk_config.ENABLE_AGENTCORE_CDK_RUNTIME",
        "True",
        raising=False,
    )
    monkeypatch.setattr(
        "cdk_config.ENABLE_PI_AGENT_EXPRESS_SERVICE",
        "False",
        raising=False,
    )
    monkeypatch.setattr(
        "cdk_config.S3_LOG_CONFIG_BUCKET_NAME",
        "",
        raising=False,
    )

    url = post.sync_agentcore_runtime_url_from_stack(
        pi_agent_env_path=pi_env,
        cdk_env_path=cdk_env,
        recycle_agent_service=False,
    )
    assert url
    assert "bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/" in url
    assert "AGENTCORE_RUNTIME_URL=" in pi_env.read_text(encoding="utf-8")
    assert "ENABLE_AGENTCORE_CDK_RUNTIME=True" in cdk_env.read_text(encoding="utf-8")
