"""S3 batch Lambda handler behaviour (assignPublicIp, app_config merge)."""

import importlib.util
import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
LAMBDA_PATH = CDK_DIR / "config" / "lambda" / "lambda_function.py"


def _load_lambda_module():
    spec = importlib.util.spec_from_file_location("batch_lambda", LAMBDA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["batch_lambda"] = module
    spec.loader.exec_module(module)
    return module


def test_assign_public_ip_reads_env(monkeypatch):
    _load_lambda_module()
    monkeypatch.setenv("ECS_ASSIGN_PUBLIC_IP", "true")
    reloaded = importlib.util.spec_from_file_location("batch_lambda2", LAMBDA_PATH)
    module = importlib.util.module_from_spec(reloaded)
    assert reloaded.loader is not None
    reloaded.loader.exec_module(module)
    assert module.ASSIGN_PUBLIC_IP is True


def test_build_environment_includes_app_config(monkeypatch):
    mod = _load_lambda_module()
    monkeypatch.setattr(
        mod,
        "_load_app_config_env",
        lambda: {"RUN_AWS_FUNCTIONS": "True", "SHOW_COSTS": "False"},
    )
    monkeypatch.setattr(mod, "_load_default_env", lambda _bucket: {})
    monkeypatch.setattr(
        mod,
        "s3",
        type(
            "S3",
            (),
            {
                "get_object": staticmethod(
                    lambda Bucket, Key: {
                        "Body": type(
                            "Body",
                            (),
                            {
                                "read": staticmethod(
                                    lambda: b"DIRECT_MODE_INPUT_FILE=doc.pdf\n"
                                )
                            },
                        )()
                    }
                ),
                "exceptions": type("E", (), {"ClientError": Exception})(),
            },
        )(),
    )
    monkeypatch.setattr(
        mod,
        "ecs",
        type(
            "ECS",
            (),
            {"run_task": staticmethod(lambda **kwargs: {"tasks": [], "failures": []})},
        )(),
    )

    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "output-bucket"},
                    "object": {"key": "input/config/job.env"},
                }
            }
        ]
    }
    mod.ENV_PREFIX = "input/config/"
    mod.ENV_SUFFIX = ".env"
    mod.CONTAINER_NAME = "app"
    mod.CLUSTER = "cluster"
    mod.TASK_DEF = "arn:aws:ecs:eu-west-2:123:task-definition/app:1"
    mod.SUBNETS = ["subnet-1"]
    mod.SECURITY_GROUPS = ["sg-1"]

    result = mod.lambda_handler(event, None)
    assert result["runs"]
    assert result["runs"][0]["envCount"] >= 3
