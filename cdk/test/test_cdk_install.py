"""Unit tests for cdk_install.py (no live AWS or cdk deploy)."""

import argparse
import json
import sys
from pathlib import Path

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
        subnet_mode="auto",
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
        subnet_mode="auto",
        enable_s3_batch=True,
    )


def test_build_env_values_demo():
    values = inst.build_env_values(_demo_answers())
    assert values["USE_ECS_EXPRESS_MODE"] == "True"
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


def test_build_env_values_headless():
    values = inst.build_env_values(_headless_answers())
    assert values["USE_ECS_EXPRESS_MODE"] == "False"
    assert values["USE_CLOUDFRONT"] == "False"
    assert values["RUN_USEAST_STACK"] == "False"
    assert values["ENABLE_HEADLESS_DEPLOYMENT"] == "True"
    assert values["ENABLE_S3_BATCH_ECS_TRIGGER"] == "True"
    assert values["COGNITO_AUTH"] == "False"


def test_validate_headless_rejects_pi():
    answers = _headless_answers()
    answers.enable_pi_legacy = True
    answers.enable_service_connect = True
    values = inst.build_env_values(answers)
    errors = inst.validate_env_values(values)
    assert any("HEADLESS" in e for e in errors)


def test_validate_rejects_express_with_acm():
    values = inst.build_env_values(_demo_answers())
    values["ACM_SSL_CERTIFICATE_ARN"] = "arn:aws:acm:eu-west-2:123:certificate/x"
    errors = inst.validate_env_values(values)
    assert any("ACM_SSL_CERTIFICATE_ARN" in e for e in errors)


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
    answers.pi_alb_routing = "path"
    answers.pi_alb_path_prefix = "/pi"
    values = inst.build_env_values(answers)
    assert values["ENABLE_PI_AGENT_EXPRESS_SERVICE"] == "True"
    assert values["PI_ALB_ROUTING"] == "path"
    assert values["PI_ALB_PATH_PREFIX"] == "/pi"
    assert values["PI_ALB_LISTENER_RULE_PRIORITY"] == "1"
    assert values["ECS_SERVICE_CONNECT_DISCOVERY_NAME"] == "redaction"
    assert values["ECS_PI_EXPRESS_SC_PORT_NAME"] == "port-7862"


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
    assert values["PI_ALB_LISTENER_RULE_PRIORITY"] == "2"


def test_validate_pi_host_requires_header():
    answers = _production_answers()
    answers.enable_pi_legacy = True
    answers.enable_service_connect = True
    answers.pi_alb_routing = "host"
    answers.pi_alb_host_header = ""
    values = inst.build_env_values(answers)
    errors = inst.validate_env_values(values)
    assert any("PI_ALB_HOST_HEADER" in e for e in errors)


def test_build_pi_agent_env_values():
    answers = _demo_answers()
    answers.enable_pi_express = True
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
    )
    inst.apply_pi_cli_flags(args, answers)
    assert answers.enable_pi_express is True


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


def test_discover_existing_doc_redaction_stacks_order(monkeypatch):
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
