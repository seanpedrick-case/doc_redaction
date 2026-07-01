"""Unit tests for cdk_install.py (no live AWS or cdk deploy)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

import cdk_install as inst


def _demo_answers() -> inst.InstallAnswers:
    return inst.InstallAnswers(
        profile="demo",
        aws_account_id="123456789012",
        aws_region="eu-west-2",
        cdk_prefix="Test-Redaction-",
        cognito_domain_prefix="test-redaction",
        vpc_mode="existing",
        vpc_name="test-vpc",
        public_subnet_mode="auto",
        private_subnet_mode="auto",
    )


def _production_answers() -> inst.InstallAnswers:
    a = _demo_answers()
    a.profile = "production"
    a.acm_cert_arn = "arn:aws:acm:eu-west-2:123:certificate/abc"
    a.ssl_domain = "redaction.example.com"
    a.cloudfront_geo = "GB"
    return a


def _headless_answers() -> inst.InstallAnswers:
    return inst.InstallAnswers(
        profile="headless",
        aws_account_id="123456789012",
        aws_region="eu-west-2",
        cdk_prefix="Headless-Redaction-",
        cognito_domain_prefix="headless-redaction",
        vpc_mode="existing",
        vpc_name="test-vpc",
        public_subnet_mode="auto",
        private_subnet_mode="auto",
        enable_s3_batch=True,
    )


def test_default_cognito_domain_prefix_strips_reserved_words_from_cdk_prefix():
    prefix = inst.default_cognito_domain_prefix_from_cdk_prefix(
        "Lambeth-AWS-SharedServices-Dev-Redaction-"
    )
    assert prefix == "lambeth-sharedservic"
    assert "aws" not in prefix
    assert "amazon" not in prefix
    assert "cognito" not in prefix


def test_sanitize_cognito_domain_prefix_removes_reserved_substrings():
    assert (
        inst.sanitize_cognito_domain_prefix("lambeth-aws-sharedse")
        == "lambeth-sharedse"
    )
    assert inst.sanitize_cognito_domain_prefix("my-cognito-login") == "my-login"
    assert (
        inst.sanitize_cognito_domain_prefix("amazon-redaction-dev") == "redaction-dev"
    )
    assert inst.sanitize_cognito_domain_prefix("aws") == "redaction-app"


def test_validate_cognito_domain_prefix_rejects_reserved_words():
    assert inst.validate_cognito_domain_prefix("lambeth-aws-sharedse") is not None
    assert inst.validate_cognito_domain_prefix("lambeth-sharedse-dev") is None


def test_build_env_values_demo():
    values = inst.build_env_values(_demo_answers())
    assert values["USE_ECS_EXPRESS_MODE"] == "True"
    assert values["ECS_EXPRESS_USE_PUBLIC_SUBNETS"] == "True"
    assert values["PRIVATE_SUBNETS_TO_USE"] == ""
    assert values["USE_CLOUDFRONT"] == "False"
    assert values["ENABLE_RESOURCE_DELETE_PROTECTION"] == "False"
    assert values["VPC_NAME"] == "test-vpc"
    assert values["CONTEXT_FILE"] == "precheck.context.json"
    assert values["CDK_FOLDER"].endswith("/cdk/")


def test_build_env_values_production():
    values = inst.build_env_values(_production_answers())
    assert values["USE_ECS_EXPRESS_MODE"] == "False"
    assert values["USE_CLOUDFRONT"] == "True"
    assert values["RUN_USEAST_STACK"] == "True"
    assert values["ENABLE_RESOURCE_DELETE_PROTECTION"] == "True"
    assert values["ACM_SSL_CERTIFICATE_ARN"].startswith("arn:aws:acm:")
    assert values["SSL_CERTIFICATE_DOMAIN"] == "redaction.example.com"
    assert values["APPREGISTRY_STACK_NAME"] == "Test-Redaction-AppRegistryStack"


def test_build_env_values_uses_custom_s3_bucket_names():
    answers = _demo_answers()
    answers.s3_log_bucket_name = "063418083240-demo-redaction-s3-logs"
    answers.s3_output_bucket_name = "063418083240-demo-redaction-s3-output"
    values = inst.build_env_values(answers)
    assert values["S3_LOG_CONFIG_BUCKET_NAME"] == "063418083240-demo-redaction-s3-logs"
    assert values["S3_OUTPUT_BUCKET_NAME"] == "063418083240-demo-redaction-s3-output"


def test_normalize_s3_bucket_name_truncates_and_sanitizes():
    long_name = "A" * 80 + "/underscores"
    normalized = inst.normalize_s3_bucket_name(long_name)
    assert normalized == normalized.lower()
    assert len(normalized) <= inst.S3_BUCKET_NAME_MAX_LEN
    assert "_" not in normalized


def test_suggest_available_s3_bucket_name_prefers_account_prefix(monkeypatch):
    calls: list[str] = []

    def fake_resolve(name: str, **kwargs):
        calls.append(name)
        if name.startswith("123456789012-"):
            return ("available", name)
        return ("globally_taken", name)

    monkeypatch.setattr("cdk_functions.resolve_s3_bucket_availability", fake_resolve)
    suggested = inst.suggest_available_s3_bucket_name(
        "demo-redaction-s3-logs", "123456789012"
    )
    assert suggested == "123456789012-demo-redaction-s3-logs"
    assert calls[0] == "demo-redaction-s3-logs"


def test_validate_globally_unique_env_values_flags_taken_bucket(monkeypatch):
    monkeypatch.setattr(
        "cdk_functions.resolve_s3_bucket_availability",
        lambda name, **kwargs: ("globally_taken", name),
    )
    values = inst.build_env_values(_demo_answers())
    errors = inst.validate_globally_unique_env_values(values)
    assert any("taken globally" in err for err in errors)


def test_prompt_globally_unique_cognito_prefix_suggests_when_taken(monkeypatch):
    monkeypatch.setattr(
        "cdk_functions.resolve_cognito_domain_prefix_availability",
        lambda prefix, **kwargs: "taken" if prefix == "demo-redaction" else "available",
    )
    monkeypatch.setattr(
        inst,
        "suggest_available_cognito_domain_prefix",
        lambda preferred, account_id, region, **kwargs: "063418083240-demo-redaction",
    )
    monkeypatch.setattr(inst, "ask_yes_no", lambda *args, **kwargs: True)
    result = inst._prompt_globally_unique_cognito_prefix(
        "demo-redaction",
        "063418083240",
        "eu-west-2",
        interactive=True,
        assume_yes=False,
    )
    assert result == "063418083240-demo-redaction"


def test_prompt_globally_unique_cognito_prefix_rejects_taken_cli_override(monkeypatch):
    monkeypatch.setattr(
        "cdk_functions.resolve_cognito_domain_prefix_availability",
        lambda prefix, **kwargs: "taken",
    )
    with pytest.raises(SystemExit):
        inst._prompt_globally_unique_cognito_prefix(
            "demo-redaction",
            "063418083240",
            "eu-west-2",
            interactive=False,
            assume_yes=False,
            cli_override="demo-redaction",
        )


def test_subnet_cidr_prefix_len_for_express_demo():
    answers = _demo_answers()
    assert inst.subnet_cidr_prefix_len_for_tier(answers, "public") == 27
    assert inst.subnet_cidr_prefix_len_for_tier(answers, "private") == 28


def test_validate_public_subnet_cidr_for_express_rejects_small_blocks():
    assert inst.validate_public_subnet_cidr_for_express("10.0.0.0/28") is not None
    assert inst.validate_public_subnet_cidr_for_express("10.0.0.0/27") is None
    assert inst.validate_public_subnet_cidr_for_express("10.0.0.0/26") is None


def test_validate_env_values_rejects_express_public_subnets_too_small():
    values = inst.build_env_values(_demo_answers())
    values["PUBLIC_SUBNET_CIDR_BLOCKS"] = "['10.0.0.0/28', '10.0.0.16/28']"
    errors = inst.validate_env_values(values)
    assert any("too small for ECS Express" in err for err in errors)


def test_build_env_values_headless():
    values = inst.build_env_values(_headless_answers())
    assert values["USE_ECS_EXPRESS_MODE"] == "False"
    assert values["USE_CLOUDFRONT"] == "False"
    assert values["RUN_USEAST_STACK"] == "False"
    assert values["ENABLE_HEADLESS_DEPLOYMENT"] == "True"
    assert values["ENABLE_S3_BATCH_ECS_TRIGGER"] == "True"
    assert values["COGNITO_AUTH"] == "False"
    assert values["ECS_EXPRESS_USE_PUBLIC_SUBNETS"] == "True"
    assert values["PRIVATE_SUBNETS_TO_USE"] == ""
    assert values["S3_LOG_CONFIG_BUCKET_NAME"] == "headless-redaction-s3-logs"
    assert values["S3_OUTPUT_BUCKET_NAME"] == "headless-redaction-s3-output"


def test_validate_env_values_rejects_bare_s3_bucket_names():
    values = inst.build_env_values(_headless_answers())
    values["S3_LOG_CONFIG_BUCKET_NAME"] = "s3-logs"
    errors = inst.validate_env_values(values)
    assert any("S3_LOG_CONFIG_BUCKET_NAME" in e for e in errors)


def test_headless_profile_uses_public_subnets_only():
    assert inst.answers_use_public_subnets_only(_headless_answers()) is True


def test_build_env_values_production_headless():
    answers = _production_answers()
    answers.enable_headless = True
    values = inst.build_env_values(answers)
    assert values["USE_ECS_EXPRESS_MODE"] == "False"
    assert values["USE_CLOUDFRONT"] == "False"
    assert values["ENABLE_HEADLESS_DEPLOYMENT"] == "True"
    assert values["ENABLE_RESOURCE_DELETE_PROTECTION"] == "True"
    assert values.get("ECS_EXPRESS_USE_PUBLIC_SUBNETS") != "True"
    assert "PRIVATE_SUBNET_CIDR_BLOCKS" in values


def test_validate_install_answers_rejects_demo_headless():
    answers = _demo_answers()
    answers.enable_headless = True
    errors = inst.validate_install_answers(answers)
    assert any("Demonstration" in err for err in errors)


def test_validate_install_answers_rejects_custom_express_headless():
    answers = inst.InstallAnswers(profile="custom", enable_headless=True)
    answers.custom_overrides["USE_ECS_EXPRESS_MODE"] = "True"
    errors = inst.validate_install_answers(answers)
    assert any("USE_ECS_EXPRESS_MODE=False" in err for err in errors)


def test_profile_allows_headless_add_on():
    assert inst.profile_allows_headless_add_on(_demo_answers()) is False
    assert inst.profile_allows_headless_add_on(_production_answers()) is True
    assert inst.profile_allows_headless_add_on(_headless_answers()) is True
    custom = inst.InstallAnswers(profile="custom")
    assert inst.profile_allows_headless_add_on(custom) is True
    custom.custom_overrides["USE_ECS_EXPRESS_MODE"] = "True"
    assert inst.profile_allows_headless_add_on(custom) is False


def test_validate_install_answers_skips_cognito_prefix_for_headless():
    values = inst.build_env_values(_headless_answers())
    values["COGNITO_USER_POOL_DOMAIN_PREFIX"] = ""
    assert inst.validate_env_values(values) == []


def test_validate_headless_rejects_pi():
    values = inst.build_env_values(_headless_answers())
    values["ENABLE_ECS_SERVICE_CONNECT"] = "True"
    errors = inst.validate_env_values(values)
    assert any("HEADLESS" in e for e in errors)


def test_validate_rejects_express_with_acm():
    values = inst.build_env_values(_demo_answers())
    values["ACM_SSL_CERTIFICATE_ARN"] = "arn:aws:acm:eu-west-2:123:certificate/x"
    errors = inst.validate_env_values(values)
    assert any("ACM_SSL_CERTIFICATE_ARN" in e for e in errors)


def test_build_env_values_mixed_subnet_tiers():
    answers = _production_answers()
    answers.public_subnet_mode = "existing"
    answers.private_subnet_mode = "create"
    answers.public_subnet_names = ["existing-public-a", "existing-public-b"]
    answers.public_subnet_cidrs = ["10.244.1.0/28", "10.244.2.0/28"]
    answers.public_subnet_azs = ["eu-west-2a", "eu-west-2b"]
    answers.private_subnet_names = ["Demo-Redaction-PrivateSubnet1"]
    answers.private_subnet_cidrs = ["10.0.10.0/28"]
    answers.private_subnet_azs = ["eu-west-2a"]
    values = inst.build_env_values(answers)
    assert (
        values["PUBLIC_SUBNETS_TO_USE"] == '["existing-public-a", "existing-public-b"]'
    )
    assert values["PUBLIC_SUBNET_CIDR_BLOCKS"] == "['10.244.1.0/28', '10.244.2.0/28']"
    assert values["PUBLIC_SUBNET_AVAILABILITY_ZONES"] == "['eu-west-2a', 'eu-west-2b']"
    assert values["PRIVATE_SUBNETS_TO_USE"] == '["Demo-Redaction-PrivateSubnet1"]'
    assert values["PRIVATE_SUBNET_CIDR_BLOCKS"] == "['10.0.10.0/28']"
    assert values["PRIVATE_SUBNET_AVAILABILITY_ZONES"] == "['eu-west-2a']"


def test_enrich_existing_subnet_details_from_aws(monkeypatch):
    answers = _demo_answers()
    answers.vpc_name = "test-vpc"
    answers.public_subnet_mode = "existing"
    answers.public_subnet_names = ["pub-a", "pub-b"]
    answers.private_subnet_mode = "create"

    monkeypatch.setattr(
        inst,
        "list_vpcs",
        lambda _region: [{"name": "test-vpc", "id": "vpc-123", "cidr": "10.0.0.0/16"}],
    )
    monkeypatch.setattr(
        inst,
        "list_subnets_in_vpc",
        lambda _vpc_id, _region: [
            {"name": "pub-a", "cidr": "10.244.1.0/28", "az": "eu-west-2a"},
            {"name": "pub-b", "cidr": "10.244.2.0/28", "az": "eu-west-2b"},
        ],
    )

    errors = inst.enrich_existing_subnet_details_from_aws(answers)
    assert errors == []
    assert answers.public_subnet_cidrs == ["10.244.1.0/28", "10.244.2.0/28"]
    assert answers.public_subnet_azs == ["eu-west-2a", "eu-west-2b"]


def test_validate_subnet_answers_mixed_requires_public_names():
    answers = _demo_answers()
    answers.public_subnet_mode = "existing"
    answers.private_subnet_mode = "create"
    answers.private_subnet_names = ["new-private"]
    errors = inst.validate_subnet_answers(answers)
    assert any("Public subnets" in err for err in errors)


def test_resolve_subnet_tier_modes_per_tier_override():
    args = argparse.Namespace(
        subnet_mode="auto",
        public_subnet_mode="existing",
        private_subnet_mode="create",
    )
    public, private = inst.resolve_subnet_tier_modes(args)
    assert public == "existing"
    assert private == "create"


def test_suggest_vpc_cidr_block_empty_region():
    assert inst.suggest_vpc_cidr_block([]) == "10.0.0.0/24"


def test_suggest_vpc_cidr_block_skips_overlaps():
    assert inst.suggest_vpc_cidr_block(["10.0.0.0/24"]) == "10.0.1.0/24"
    assert inst.suggest_vpc_cidr_block(["10.0.0.0/16"]) == "10.1.0.0/24"


def test_validate_new_vpc_cidr_rejects_overlap_and_public_space():
    err = inst.validate_new_vpc_cidr("10.0.0.0/24", ["10.0.0.0/16"])
    assert err is not None
    assert "overlaps" in err
    err = inst.validate_new_vpc_cidr("8.8.8.0/24", [])
    assert err is not None
    assert "RFC1918" in err


def test_validate_new_vpc_cidr_accepts_available_block():
    assert inst.validate_new_vpc_cidr("10.2.0.0/24", ["10.0.0.0/16"]) is None


def test_prompt_new_vpc_cidr_validates_cli_override(monkeypatch):
    answers = inst.InstallAnswers(aws_region="eu-west-2", new_vpc_cidr="10.0.0.0/24")
    monkeypatch.setattr(
        inst, "list_vpc_cidr_blocks_in_region", lambda _r: ["10.0.0.0/16"]
    )
    with pytest.raises(SystemExit):
        inst.prompt_new_vpc_cidr(answers, interactive=False)


def test_prompt_new_vpc_cidr_auto_select_noninteractive(monkeypatch):
    answers = inst.InstallAnswers(aws_region="eu-west-2")
    monkeypatch.setattr(inst, "list_vpc_cidr_blocks_in_region", lambda _r: [])
    inst.prompt_new_vpc_cidr(answers, interactive=False)
    assert answers.new_vpc_cidr == "10.0.0.0/24"


def test_suggest_subnet_cidr_blocks_lowest_available():
    blocks = inst.suggest_subnet_cidr_blocks("10.0.0.0/24", ["10.0.0.0/28"], 2)
    assert blocks == ["10.0.0.16/28", "10.0.0.32/28"]


def test_suggest_subnet_cidr_blocks_respects_reserved():
    blocks = inst.suggest_subnet_cidr_blocks(
        "10.0.0.0/24",
        [],
        1,
        reserved_cidrs=["10.0.0.0/28"],
    )
    assert blocks == ["10.0.0.16/28"]


def test_vpc_cidr_blocks_from_describe_includes_associations():
    vpc = {
        "CidrBlock": "10.0.0.0/16",
        "CidrBlockAssociationSet": [{"CidrBlock": "10.1.0.0/16"}],
    }
    assert inst.vpc_cidr_blocks_from_describe(vpc) == [
        "10.0.0.0/16",
        "10.1.0.0/16",
    ]


def test_build_app_config_env_values_express_uses_in_app_cognito():
    values = inst.build_env_values(_demo_answers())
    updates = inst.build_app_config_env_values(values)
    assert updates["COGNITO_AUTH"] == "True"
    assert updates["RUN_AWS_FUNCTIONS"] == "True"
    assert updates["DOCUMENT_REDACTION_BUCKET"].endswith("s3-logs")


def test_build_app_config_env_values_express_pi_disables_main_cognito():
    answers = _demo_answers()
    answers.enable_pi_express = True
    values = inst.build_env_values(answers)
    updates = inst.build_app_config_env_values(values)
    assert updates["COGNITO_AUTH"] == "False"


def test_build_app_config_env_values_headless():
    answers = _headless_answers()
    values = inst.build_env_values(answers)
    updates = inst.build_app_config_env_values(values)
    assert updates["COGNITO_AUTH"] == "False"


def test_write_app_config_env_from_example(tmp_path, monkeypatch):
    example = tmp_path / "app_config.env.example"
    example.write_text(
        "RUN_FASTAPI=True\nDOCUMENT_REDACTION_BUCKET=placeholder\n",
        encoding="utf-8",
    )
    target = tmp_path / "app_config.env"
    monkeypatch.setattr(inst, "APP_CONFIG_ENV_EXAMPLE", example)
    monkeypatch.setattr(inst, "APP_CONFIG_ENV_PATH", target)

    answers = _demo_answers()
    answers.write_app_config_env = True
    values = inst.build_env_values(answers)
    inst.write_app_config_env_file(answers, values)

    written = inst.read_env_file(target)
    assert written["RUN_FASTAPI"] == "True"
    assert written["DOCUMENT_REDACTION_BUCKET"].endswith("s3-logs")
    assert written["COGNITO_AUTH"] == "True"


def test_format_list_env():
    assert inst.format_list_env(["a", "b"]) == '["a", "b"]'
    assert (
        inst.format_list_env(["10.0.0.0/28"], use_single_quotes=True)
        == "['10.0.0.0/28']"
    )
    assert inst.format_list_env([]) == "[]"


def test_write_env_file_backs_up(tmp_path):
    env_path = tmp_path / "cdk_config.env"
    env_path.write_text("OLD=1\n", encoding="utf-8")
    inst.write_env_file(env_path, {"NEW": "2"})
    backups = list(tmp_path.glob("cdk_config.env.bak.*"))
    assert len(backups) == 1
    assert "NEW=2" in env_path.read_text(encoding="utf-8")


def test_write_cdk_json_preserves_context(tmp_path, monkeypatch):
    cdk_json = tmp_path / "cdk.json"
    cdk_json.write_text(
        json.dumps(
            {
                "app": "old-python app.py",
                "context": {"@aws-cdk/custom:flag": True, "keep": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inst, "CDK_JSON_PATH", cdk_json)
    monkeypatch.setattr(inst, "CDK_JSON_EXAMPLE", tmp_path / "missing.example")

    py = Path(sys.executable)
    inst.write_cdk_json(py, force=True)

    data = json.loads(cdk_json.read_text(encoding="utf-8"))
    assert data["context"]["@aws-cdk/custom:flag"] is True
    assert data["context"]["keep"] == 1
    assert data["app"].endswith("app.py")
    assert "output" in data
    backups = list(tmp_path.glob("cdk.json.bak.*"))
    assert len(backups) == 1


def test_format_cdk_app_command():
    cmd = inst.format_cdk_app_command(Path(sys.executable))
    assert cmd.endswith("app.py")
    assert str(sys.executable).replace("/", "\\") in cmd or sys.executable in cmd


def test_venv_python_paths_includes_sys_executable():
    paths = inst._venv_python_paths()
    assert Path(sys.executable).resolve() == paths[0].resolve()


def test_merge_preset_custom():
    merged = inst.merge_preset("demo", {"ECS_TASK_MEMORY_SIZE": "8192"})
    assert merged["USE_ECS_EXPRESS_MODE"] == "True"
    assert merged["ECS_TASK_MEMORY_SIZE"] == "8192"


def test_build_env_values_pi_express():
    answers = _demo_answers()
    answers.enable_pi_express = True
    values = inst.build_env_values(answers)
    assert values["ENABLE_PI_AGENT_EXPRESS_SERVICE"] == "True"
    assert "PI_ALB_PATH_PREFIX" not in values
    assert "PI_ALB_ROUTING" not in values
    assert values["ECS_SERVICE_CONNECT_DISCOVERY_NAME"] == "redaction"
    assert values["ECS_PI_EXPRESS_SC_PORT_NAME"] == "port-7862"
    assert values["ECS_EXPRESS_SC_PORT_NAME"] == "port-7860"


def test_build_pi_agent_env_values_express_skips_root_path():
    answers = _demo_answers()
    answers.enable_pi_express = True
    env = inst.build_pi_agent_env_values(answers)
    assert env["RUN_FASTAPI"] == "True"
    assert "PI_ROOT_PATH" not in env


def test_build_env_values_pi_production_host():
    answers = _production_answers()
    answers.enable_pi_legacy = True
    answers.enable_service_connect = True
    answers.pi_alb_routing = "host"
    answers.pi_alb_host_header = "agent.redaction.example.com"
    values = inst.build_env_values(answers)
    assert values["ENABLE_PI_AGENT_ECS_SERVICE"] == "True"
    assert values["ENABLE_ECS_SERVICE_CONNECT"] == "True"
    assert values["PI_ALB_HOST_HEADER"] == "agent.redaction.example.com"
    assert values["PI_ALB_LISTENER_RULE_PRIORITY"] == "3"


def test_validate_pi_host_requires_header():
    answers = _production_answers()
    answers.enable_pi_legacy = True
    answers.enable_service_connect = True
    answers.pi_alb_routing = "host"
    answers.pi_alb_host_header = ""
    values = inst.build_env_values(answers)
    errors = inst.validate_env_values(values)
    assert any("PI_ALB_HOST_HEADER" in e for e in errors)


def test_validate_pi_express_skips_alb_routing():
    answers = _demo_answers()
    answers.enable_pi_express = True
    values = inst.build_env_values(answers)
    assert inst.validate_env_values(values) == []


def test_build_pi_agent_env_values():
    answers = _demo_answers()
    answers.enable_pi_express = False
    answers.enable_pi_legacy = True
    answers.pi_alb_routing = "path"
    answers.pi_alb_path_prefix = "/pi"
    values = inst.build_pi_agent_env_values(answers)
    assert values["PI_DEPLOYMENT_PROFILE"] == "aws-ecs"
    assert values["DOC_REDACTION_GRADIO_URL"] == "http://redaction:7860"
    assert values["PI_ROOT_PATH"] == "/pi"


def test_apply_pi_cli_flags_enable_pi_demo():
    answers = inst.InstallAnswers(profile="demo")
    args = argparse.Namespace(
        enable_pi=True,
        enable_pi_express=False,
        enable_pi_legacy=False,
        pi_alb_routing=None,
        pi_path_prefix="",
        pi_host_header="",
        pi_listener_priority="",
        pi_gradio_port="",
        sc_discovery_name="",
        pi_provider="",
        skip_pi_agent_env=False,
        agent_orchestrator=None,
        enable_agentcore_runtime=False,
        agentcore_runtime_url="",
        agentcore_api_key="",
    )
    inst.apply_pi_cli_flags(args, answers)
    assert answers.enable_pi_express is True
    assert answers.agent_orchestrator == "agentcore"
    assert answers.enable_agentcore_runtime is True


def test_demo_default_orchestrator_is_agentcore():
    answers = inst.InstallAnswers(profile="demo", enable_pi_express=True)
    assert inst.default_agent_orchestrator_for_answers(answers) == "agentcore"
    answers.profile = "production"
    assert inst.default_agent_orchestrator_for_answers(answers) == "pi"


def test_merge_policy_file_locations_adds_invoke_policy():
    raw = inst.merge_policy_file_locations(agentcore_enabled=False)
    assert inst.PI_AGENTCORE_INVOKE_POLICY_FILE not in raw
    raw_agentcore = inst.merge_policy_file_locations(agentcore_enabled=True)
    assert inst.PI_AGENTCORE_INVOKE_POLICY_FILE in raw_agentcore
    assert "textract_policy.json" in raw_agentcore


def test_build_env_values_agentcore_includes_invoke_policy():
    answers = _demo_answers()
    answers.enable_pi_express = True
    answers.agent_orchestrator = "agentcore"
    answers.enable_agentcore_runtime = True
    answers.agentcore_runtime_url = "https://runtime.example"
    values = inst.build_env_values(answers)
    assert inst.PI_AGENTCORE_INVOKE_POLICY_FILE in values["POLICY_FILE_LOCATIONS"]


def test_validate_agentcore_allows_deferred_url():
    answers = _demo_answers()
    answers.enable_pi_express = True
    answers.agent_orchestrator = "agentcore"
    answers.allow_empty_agentcore_url = True
    assert inst.validate_install_answers(answers) == []
    values = inst.build_env_values(answers)
    assert inst.validate_env_values(values, allow_empty_agentcore_url=True) == []


def test_build_env_values_agentcore_orchestrator():
    answers = _demo_answers()
    answers.enable_pi_express = True
    answers.agent_orchestrator = "agentcore"
    answers.enable_agentcore_runtime = True
    answers.agentcore_runtime_url = "https://runtime.example"
    values = inst.build_env_values(answers)
    assert values["AGENT_ORCHESTRATOR"] == "agentcore"
    assert values["ENABLE_AGENTCORE_RUNTIME"] == "True"
    assert values["AGENTCORE_RUNTIME_URL"] == "https://runtime.example"


def test_build_pi_agent_env_values_agentcore():
    answers = _demo_answers()
    answers.enable_pi_express = True
    answers.agent_orchestrator = "agentcore"
    answers.agentcore_runtime_url = "https://runtime.example"
    answers.agentcore_api_key = "secret-token"
    values = inst.build_pi_agent_env_values(answers)
    assert values["AGENT_ORCHESTRATOR"] == "agentcore"
    assert values["AGENTCORE_RUNTIME_URL"] == "https://runtime.example"
    assert values["AGENTCORE_API_KEY"] == "secret-token"


def test_validate_agentcore_requires_runtime_url():
    answers = _demo_answers()
    answers.enable_pi_express = True
    answers.agent_orchestrator = "agentcore"
    errors = inst.validate_install_answers(answers)
    assert any("AGENTCORE_RUNTIME_URL" in err for err in errors)

    answers.agentcore_runtime_url = "https://runtime.example"
    assert inst.validate_install_answers(answers) == []

    values = inst.build_env_values(answers)
    assert inst.validate_env_values(values) == []


def test_apply_agent_orchestrator_cli_flags():
    answers = inst.InstallAnswers(profile="demo", enable_pi_express=True)
    args = argparse.Namespace(
        agent_orchestrator="langgraph",
        enable_agentcore_runtime=False,
        agentcore_runtime_url="",
        agentcore_api_key="",
    )
    inst.apply_agent_orchestrator_cli_flags(args, answers)
    assert answers.agent_orchestrator == "langgraph"

    args.enable_agentcore_runtime = True
    args.agentcore_runtime_url = "https://runtime.example"
    inst.apply_agent_orchestrator_cli_flags(args, answers)
    assert answers.agent_orchestrator == "agentcore"
    assert answers.enable_agentcore_runtime is True
    assert answers.agentcore_runtime_url == "https://runtime.example"


def test_stacks_to_check_includes_appregistry_when_enabled():
    checks = inst.stacks_to_check(
        "eu-west-2",
        {
            "ENABLE_APPREGISTRY": "True",
            "APPREGISTRY_STACK_NAME": "Demo-Redaction-AppRegistryStack",
        },
    )
    names = [name for name, _ in checks]
    assert names == [
        inst.CLOUDFRONT_STACK,
        "Demo-Redaction-AppRegistryStack",
        inst.REGIONAL_STACK,
    ]
    assert checks[0][1] == inst.CLOUDFRONT_STACK_REGION
    assert checks[-1] == (inst.REGIONAL_STACK, "eu-west-2")


def test_stacks_to_check_without_appregistry():
    checks = inst.stacks_to_check("eu-west-2", {"ENABLE_APPREGISTRY": "False"})
    assert [name for name, _ in checks] == [
        inst.CLOUDFRONT_STACK,
        inst.REGIONAL_STACK,
    ]


def test_derived_appregistry_stack_name_from_cdk_prefix():
    name = inst.derived_appregistry_stack_name(
        {"CDK_PREFIX": "Demo-Redaction-", "ENABLE_APPREGISTRY": "True"}
    )
    assert name == "Demo-Redaction-AppRegistryStack"


def test_stacks_to_check_derives_appregistry_from_cdk_prefix():
    checks = inst.stacks_to_check(
        "eu-west-2",
        {
            "CDK_PREFIX": "Demo-Redaction-",
            "ENABLE_APPREGISTRY": "True",
        },
    )
    assert "Demo-Redaction-AppRegistryStack" in [name for name, _ in checks]


def test_stacks_to_check_skips_cloudfront_when_disabled():
    checks = inst.stacks_to_check(
        "eu-west-2",
        {
            "USE_CLOUDFRONT": "False",
            "RUN_USEAST_STACK": "False",
            "ENABLE_APPREGISTRY": "False",
        },
    )
    assert checks == [(inst.REGIONAL_STACK, "eu-west-2")]


def test_discover_existing_doc_redaction_stacks_order(monkeypatch):
    monkeypatch.setattr(
        inst, "list_regional_appregistry_stack_names", lambda _region: []
    )

    def fake_describe(stack_name: str, region: str):
        if stack_name == inst.REGIONAL_STACK and region == "eu-west-2":
            return inst.ExistingStack(
                name=stack_name,
                region=region,
                status="UPDATE_COMPLETE",
            )
        if (
            stack_name == inst.CLOUDFRONT_STACK
            and region == inst.CLOUDFRONT_STACK_REGION
        ):
            return inst.ExistingStack(
                name=stack_name,
                region=region,
                status="CREATE_COMPLETE",
                termination_protection=True,
            )
        return None

    monkeypatch.setattr(inst, "describe_existing_stack", fake_describe)
    found = inst.discover_existing_doc_redaction_stacks("eu-west-2")
    assert [s.name for s in found] == [inst.CLOUDFRONT_STACK, inst.REGIONAL_STACK]
    assert found[0].termination_protection is True


def test_discover_existing_stacks_continues_after_access_denied(monkeypatch):
    from botocore.exceptions import ClientError

    monkeypatch.setattr(
        inst, "list_regional_appregistry_stack_names", lambda _region: []
    )

    def fake_describe(stack_name: str, region: str):
        if (
            stack_name == inst.CLOUDFRONT_STACK
            and region == inst.CLOUDFRONT_STACK_REGION
        ):
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDenied",
                        "Message": "explicit deny in SCP",
                    }
                },
                "DescribeStacks",
            )
        if stack_name == inst.REGIONAL_STACK and region == "eu-west-2":
            return inst.ExistingStack(
                name=stack_name,
                region=region,
                status="UPDATE_COMPLETE",
            )
        return None

    monkeypatch.setattr(inst, "describe_existing_stack", fake_describe)
    found = inst.discover_existing_doc_redaction_stacks("eu-west-2")
    assert [s.name for s in found] == [inst.REGIONAL_STACK]


def test_discover_includes_orphan_appregistry_when_disabled_in_config(monkeypatch):
    monkeypatch.setattr(
        inst,
        "list_regional_appregistry_stack_names",
        lambda _region: ["Old-Redaction-AppRegistryStack"],
    )

    def fake_describe(stack_name: str, region: str):
        if stack_name == "Old-Redaction-AppRegistryStack":
            return inst.ExistingStack(
                name=stack_name,
                region=region,
                status="CREATE_COMPLETE",
            )
        return None

    monkeypatch.setattr(inst, "describe_existing_stack", fake_describe)
    found = inst.discover_existing_doc_redaction_stacks(
        "eu-west-2",
        {"ENABLE_APPREGISTRY": "False"},
    )
    assert [s.name for s in found] == ["Old-Redaction-AppRegistryStack"]


def test_handle_existing_stacks_force_delete(monkeypatch):
    stacks = [
        inst.ExistingStack(
            name=inst.CLOUDFRONT_STACK,
            region=inst.CLOUDFRONT_STACK_REGION,
            status="CREATE_COMPLETE",
        )
    ]
    deleted: list = []

    monkeypatch.setattr(
        inst,
        "discover_existing_doc_redaction_stacks",
        lambda *_a, **_k: stacks,
    )
    monkeypatch.setattr(
        inst,
        "force_delete_cloudformation_stacks",
        lambda s, **kwargs: deleted.extend(s),
    )

    args = argparse.Namespace(
        skip_stack_check=False,
        config_only=False,
        synth_only=False,
        force_delete_stacks=True,
        yes=True,
    )
    inst.handle_existing_stacks_at_start(args, "eu-west-2")
    assert deleted == stacks


def test_handle_existing_stacks_yes_without_force_skips_delete(monkeypatch):
    monkeypatch.setattr(
        inst,
        "discover_existing_doc_redaction_stacks",
        lambda *_a, **_k: [
            inst.ExistingStack(
                name=inst.REGIONAL_STACK,
                region="eu-west-2",
                status="CREATE_COMPLETE",
            )
        ],
    )
    monkeypatch.setattr(
        inst,
        "force_delete_cloudformation_stacks",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not delete")),
    )

    args = argparse.Namespace(
        skip_stack_check=False,
        config_only=False,
        synth_only=False,
        force_delete_stacks=False,
        yes=True,
    )
    inst.handle_existing_stacks_at_start(args, "eu-west-2")


def test_write_pi_agent_env_file_minimal(tmp_path, monkeypatch):
    answers = _demo_answers()
    answers.enable_pi_express = True
    target = tmp_path / "pi_agent.env"
    monkeypatch.setattr(inst, "PI_AGENT_ENV_PATH", target)
    monkeypatch.setattr(inst, "PI_AGENT_ENV_EXAMPLE", tmp_path / "missing.example")
    inst.write_pi_agent_env_file(answers)
    text = target.read_text(encoding="utf-8")
    assert "PI_DEPLOYMENT_PROFILE=aws-ecs" in text
    assert "DOC_REDACTION_GRADIO_URL=http://redaction:7860" in text


def test_run_cdk_command_invokes_cdk_cli_not_python(monkeypatch):
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(inst, "resolve_cdk_executable", lambda: r"C:\npm\cdk.cmd")
    monkeypatch.setattr(inst.subprocess, "run", fake_run)
    inst.run_cdk_command(["synth"], check=False)
    cmd, kwargs = calls[0]
    assert cmd == [r"C:\npm\cdk.cmd", "synth"]
    assert "executable" not in kwargs


def test_build_cdk_subprocess_env_overrides_stale_defaults(tmp_path, monkeypatch):
    env_path = tmp_path / "config" / "cdk_config.env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "VPC_NAME=my-vpc\nCDK_PREFIX=MyPrefix-\nAWS_REGION=eu-west-2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(inst, "ENV_PATH", env_path)
    monkeypatch.setattr(inst, "CDK_DIR", tmp_path)
    monkeypatch.setenv("VPC_NAME", "")
    monkeypatch.setenv("CDK_PREFIX", "")

    env = inst.build_cdk_subprocess_env()
    assert env["VPC_NAME"] == "my-vpc"
    assert env["CDK_PREFIX"] == "MyPrefix-"
    assert env["CDK_CONFIG_PATH"] == str(env_path)


def test_apply_cdk_runtime_env_sets_paths(tmp_path, monkeypatch):
    env_path = tmp_path / "cdk_config.env"
    monkeypatch.setattr(inst, "ENV_PATH", env_path)
    inst.apply_cdk_runtime_env(
        {"CDK_FOLDER": "C:/example/cdk/", "AWS_REGION": "eu-west-2"}
    )
    assert os.environ["CDK_CONFIG_PATH"] == str(env_path)
    assert os.environ["CDK_FOLDER"] == "C:/example/cdk/"


def test_run_smoke_test_if_needed_skips_config_only(monkeypatch):
    called = []

    def fake_smoke(_python_exe):
        called.append(True)

    monkeypatch.setattr(inst, "smoke_test_python_app", fake_smoke)
    args = argparse.Namespace(config_only=True, skip_cdk_json=False)
    inst.run_smoke_test_if_needed(Path(sys.executable), args)
    assert called == []


def test_main_writes_config_before_smoke_test(monkeypatch, tmp_path):
    call_order: list[str] = []
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    env_path = config_dir / "cdk_config.env"
    cdk_json = tmp_path / "cdk.json"

    monkeypatch.setattr(inst, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(inst, "ENV_PATH", env_path)
    monkeypatch.setattr(inst, "CDK_JSON_PATH", cdk_json)
    monkeypatch.setattr(inst, "CDK_DIR", tmp_path)
    monkeypatch.setattr(inst, "check_cdk_cli", lambda: "2.0.0")
    monkeypatch.setattr(
        inst,
        "handle_existing_stacks_at_start",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        inst,
        "resolve_python_executable",
        lambda **_kw: Path(sys.executable),
    )
    monkeypatch.setattr(inst, "write_cdk_json", lambda *_a, **_k: cdk_json)

    def fake_wizard(_args):
        call_order.append("wizard")
        return _demo_answers()

    def fake_write_env_file(path, values):
        call_order.append("write_config")
        path.write_text("VPC_NAME=test-vpc\n", encoding="utf-8")
        return values

    def fake_smoke(_python_exe):
        call_order.append("smoke")

    monkeypatch.setattr(inst, "run_wizard", fake_wizard)
    monkeypatch.setattr(inst, "write_env_file", fake_write_env_file)
    monkeypatch.setattr(inst, "smoke_test_python_app", fake_smoke)
    monkeypatch.setattr(inst, "print_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(inst, "ask_yes_no", lambda *_a, **_k: True)
    monkeypatch.setattr(inst, "cdk_bootstrap_needed", lambda *_a, **_k: False)
    monkeypatch.setattr(inst, "run_cdk_command", lambda *_a, **_k: None)

    inst.main(
        [
            "--yes",
            "--profile",
            "demo",
            "--vpc-name",
            "test-vpc",
            "--synth-only",
        ]
    )

    assert call_order.index("wizard") < call_order.index("write_config")
    assert call_order.index("write_config") < call_order.index("smoke")


def test_resolve_fixup_env_values_derives_service_from_prefix():
    values = {"CDK_PREFIX": "Demo-Redaction-", "USE_ECS_EXPRESS_MODE": "True"}
    resolved = inst.resolve_fixup_env_values(values)
    assert resolved["ECS_EXPRESS_SERVICE_NAME"] == "Demo-Redaction-ECSService"
    assert resolved["CLUSTER_NAME"] == "Demo-Redaction-Cluster"
    assert resolved["ECS_PI_EXPRESS_SERVICE_NAME"] == "Demo-Redaction-PiExpressService"


def test_apply_post_deploy_fixup_express_syncs_cognito_secret_not_alb(monkeypatch):
    import cdk_post_deploy as post

    values = {
        "USE_ECS_EXPRESS_MODE": "True",
        "USE_CLOUDFRONT": "False",
        "AWS_REGION": "eu-west-2",
        "CDK_PREFIX": "Demo-Redaction-",
        "ECS_EXPRESS_COGNITO_REDIRECT_BASE": "",
    }
    outputs = {
        "ExpressServiceEndpoint": "https://express.example.com",
        "CognitoPoolId": "pool-1",
        "CognitoAppClientId": "client-1",
    }
    monkeypatch.setattr(
        inst,
        "fetch_stack_output",
        lambda _stack, key, _region: outputs.get(key),
    )
    monkeypatch.setattr(inst, "patch_env_file", lambda *_a, **_k: None)
    monkeypatch.setattr(inst, "ask_yes_no", lambda *_a, **_k: True)
    redeploy_calls: list = []
    monkeypatch.setattr(
        inst,
        "run_cdk_command",
        lambda *args, **_k: redeploy_calls.append(args),
    )
    monkeypatch.setattr(
        post,
        "cognito_alb_callbacks_need_update",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        post,
        "apply_cognito_alb_callback_fixup",
        lambda *_a, **_k: True,
    )
    secret_fixup_calls: list = []
    monkeypatch.setattr(
        post,
        "apply_cognito_secret_fixup_from_stack",
        lambda **_k: secret_fixup_calls.append(_k) or True,
    )

    assert inst.apply_post_deploy_fixup(values, assume_yes=False) is True
    assert redeploy_calls == []
    assert secret_fixup_calls
    assert secret_fixup_calls[0]["main_service_name"] == "Demo-Redaction-ECSService"
    assert secret_fixup_calls[0]["cluster_name"] == "Demo-Redaction-Cluster"


def test_jsii_import_failure_hint_detects_json_decode_error():
    hint = inst._jsii_import_failure_hint(
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
    )
    assert "JSII Node.js helper" in hint
    assert "node --version" in hint


def test_jsii_import_failure_hint_empty_for_other_errors():
    assert inst._jsii_import_failure_hint("ModuleNotFoundError: aws_cdk") == ""
