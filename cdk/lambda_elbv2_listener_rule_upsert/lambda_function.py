"""
CloudFormation custom resource: create or update an ALB listener rule (upsert by priority).

If a rule already exists at the requested priority with matching conditions, it is
modified in place. This supports stack retries after partial deploys left orphaned rules.
If priority is taken by a rule with different conditions, the operation fails with a
clear error so operators can choose another PI_ALB_LISTENER_RULE_PRIORITY.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

elbv2 = boto3.client("elbv2")

_DELETE_NOT_FOUND = frozenset(
    {
        "RuleNotFound",
        "ResourceNotFound",
        "ValidationError",
    }
)
# CloudFormation custom-resource properties are strings; ELBv2 expects typed fields.
_ELBV2_INT_KEYS = frozenset(
    {
        "Order",
        "Priority",
        "SessionTimeout",
        "DurationSeconds",
        "SessionCookieName",
    }
)
_ELBV2_BOOL_KEYS = frozenset({"Enabled"})


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _coerce_elbv2_api_value(key: str, value: Any) -> Any:
    """Coerce CloudFormation string properties to types boto3 elbv2 expects."""
    if isinstance(value, dict):
        return {k: _coerce_elbv2_api_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_elbv2_api_value(key, item) for item in value]
    if isinstance(value, str):
        if key in _ELBV2_INT_KEYS:
            try:
                return int(value)
            except ValueError:
                return value
        if key in _ELBV2_BOOL_KEYS:
            lowered = value.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
    return value


def _normalize_listener_rule_payload(
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [_coerce_elbv2_api_value("", c) for c in conditions],
        [_coerce_elbv2_api_value("", a) for a in actions],
    )


def _condition_values(condition: dict[str, Any]) -> list[str]:
    field = condition.get("Field")
    if field == "path-pattern":
        cfg = condition.get("PathPatternConfig") or {}
        return sorted(cfg.get("Values") or condition.get("Values") or [])
    if field == "host-header":
        cfg = condition.get("HostHeaderConfig") or {}
        return sorted(cfg.get("Values") or condition.get("Values") or [])
    return sorted(condition.get("Values") or [])


def _conditions_match(
    rule_conditions: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> bool:
    if len(rule_conditions) != len(expected):
        return False
    for exp in expected:
        exp_field = exp.get("Field")
        exp_vals = _condition_values(exp)
        if not any(
            rc.get("Field") == exp_field and _condition_values(rc) == exp_vals
            for rc in rule_conditions
        ):
            return False
    return True


def _rule_at_priority(
    rules: list[dict[str, Any]], priority: int
) -> dict[str, Any] | None:
    target = str(priority)
    for rule in rules:
        if rule.get("Priority") == target:
            return rule
    return None


def _upsert_rule(
    *,
    listener_arn: str,
    priority: int,
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> str:
    described = elbv2.describe_rules(ListenerArn=listener_arn)
    existing = _rule_at_priority(described.get("Rules", []), priority)
    if existing:
        if _conditions_match(existing.get("Conditions", []), conditions):
            rule_arn = existing["RuleArn"]
            LOGGER.info(
                "Modifying existing listener rule %s at priority %s", rule_arn, priority
            )
            elbv2.modify_rule(RuleArn=rule_arn, Conditions=conditions, Actions=actions)
            return rule_arn
        raise RuntimeError(
            f"Listener rule priority {priority} is already in use by {existing['RuleArn']} "
            "with different conditions. Delete that rule in the ELB console or set "
            "PI_ALB_LISTENER_RULE_PRIORITY to a free value, then redeploy."
        )
    LOGGER.info("Creating listener rule at priority %s on %s", priority, listener_arn)
    created = elbv2.create_rule(
        ListenerArn=listener_arn,
        Priority=priority,
        Conditions=conditions,
        Actions=actions,
    )
    return created["Rules"][0]["RuleArn"]


def _delete_rule(rule_arn: str) -> None:
    if not rule_arn:
        return
    try:
        elbv2.delete_rule(RuleArn=rule_arn)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in _DELETE_NOT_FOUND:
            LOGGER.info("Rule %s already deleted (%s)", rule_arn, code)
            return
        raise


def handler(event, context):
    request_type = event["RequestType"]
    props = event["ResourceProperties"]
    listener_arn = props["ListenerArn"]
    priority = int(props["Priority"])
    conditions, actions = _normalize_listener_rule_payload(
        _parse_json(props["Conditions"]),
        _parse_json(props["Actions"]),
    )

    if request_type == "Delete":
        _delete_rule(event.get("PhysicalResourceId", ""))
        return {"PhysicalResourceId": event.get("PhysicalResourceId", "deleted")}

    rule_arn = _upsert_rule(
        listener_arn=listener_arn,
        priority=priority,
        conditions=conditions,
        actions=actions,
    )
    if request_type == "Update":
        prior_arn = event.get("PhysicalResourceId", "")
        if prior_arn and prior_arn != rule_arn:
            _delete_rule(prior_arn)
    return {"PhysicalResourceId": rule_arn}
