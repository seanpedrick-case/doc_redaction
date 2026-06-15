"""Unit tests for ALB listener rule upsert Lambda helpers."""

import importlib.util
import sys
from pathlib import Path

_LAMBDA_PATH = (
    Path(__file__).resolve().parents[1]
    / "lambda_elbv2_listener_rule_upsert"
    / "lambda_function.py"
)
_spec = importlib.util.spec_from_file_location(
    "elbv2_listener_rule_upsert", _LAMBDA_PATH
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


def test_conditions_match_path_pattern():
    expected = [
        {
            "Field": "path-pattern",
            "PathPatternConfig": {"Values": ["/agent", "/agent/*"]},
        }
    ]
    from_api = [
        {
            "Field": "path-pattern",
            "PathPatternConfig": {"Values": ["/agent", "/agent/*"]},
            "Values": ["/agent", "/agent/*"],
        }
    ]
    assert _mod._conditions_match(from_api, expected)


def test_conditions_do_not_match_different_path():
    expected = [
        {
            "Field": "path-pattern",
            "PathPatternConfig": {"Values": ["/agent", "/agent/*"]},
        }
    ]
    other = [
        {
            "Field": "path-pattern",
            "PathPatternConfig": {"Values": ["/other", "/other/*"]},
        }
    ]
    assert not _mod._conditions_match(other, expected)


def test_normalize_listener_rule_payload_coerces_cfn_string_types():
    raw_actions = [
        {
            "Type": "authenticate-cognito",
            "Order": "1",
            "AuthenticateCognitoConfig": {
                "UserPoolArn": "arn:aws:cognito-idp:eu-west-2:1:userpool/pool",
                "UserPoolClientId": "client",
                "UserPoolDomain": "demo",
                "SessionTimeout": "28800",
            },
        },
        {
            "Type": "forward",
            "Order": "2",
            "ForwardConfig": {
                "TargetGroups": [
                    {"TargetGroupArn": "arn:aws:elasticloadbalancing:1:tg"}
                ],
                "TargetGroupStickinessConfig": {
                    "Enabled": "true",
                    "DurationSeconds": "28800",
                },
            },
        },
    ]
    _, actions = _mod._normalize_listener_rule_payload([], raw_actions)
    assert actions[0]["Order"] == 1
    assert actions[0]["AuthenticateCognitoConfig"]["SessionTimeout"] == 28800
    assert actions[1]["Order"] == 2
    stickiness = actions[1]["ForwardConfig"]["TargetGroupStickinessConfig"]
    assert stickiness["Enabled"] is True
    assert stickiness["DurationSeconds"] == 28800
