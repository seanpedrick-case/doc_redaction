"""Unit tests for cdk_post_deploy.py (boto3 helpers, no live AWS)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

import cdk_post_deploy as post


def test_container_definitions_with_named_port_adds_mapping_to_first_container():
    containers = [{"name": "app", "image": "nginx"}]
    updated = post._container_definitions_with_named_port(
        containers,
        port_name="port-7860",
        container_port=7860,
    )
    assert updated[0]["portMappings"] == [
        {"name": "port-7860", "containerPort": 7860, "protocol": "tcp"}
    ]


def test_container_definitions_with_named_port_names_existing_mapping():
    containers = [
        {
            "name": "app",
            "image": "nginx",
            "portMappings": [{"containerPort": 7860, "protocol": "tcp"}],
        }
    ]
    updated = post._container_definitions_with_named_port(
        containers,
        port_name="port-7860",
        container_port=7860,
    )
    assert updated[0]["portMappings"][0]["name"] == "port-7860"


def test_resolve_service_task_definition_arn_from_describe_services():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.return_value = {
        "services": [
            {
                "serviceArn": "arn:aws:ecs:eu-west-2:123:service/cluster/app",
                "taskDefinition": "arn:aws:ecs:eu-west-2:123:task-definition/app:1",
            }
        ]
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        arn = post.resolve_service_task_definition_arn("cluster", "app")

    assert arn == "arn:aws:ecs:eu-west-2:123:task-definition/app:1"
    mock_ecs.describe_express_gateway_service.assert_not_called()


def test_resolve_service_task_definition_arn_from_express_service_revision():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.return_value = {
        "services": [
            {
                "serviceArn": "arn:aws:ecs:eu-west-2:123:service/cluster/express-app",
            }
        ]
    }
    mock_ecs.describe_express_gateway_service.return_value = {
        "service": {
            "activeConfigurations": [
                {
                    "serviceRevisionArn": "arn:aws:ecs:eu-west-2:123:service-revision/rev/1"
                }
            ]
        }
    }
    mock_ecs.describe_service_revisions.return_value = {
        "serviceRevisions": [
            {
                "taskDefinition": "arn:aws:ecs:eu-west-2:123:task-definition/express:3",
            }
        ]
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        arn = post.resolve_service_task_definition_arn("cluster", "express-app")

    assert arn == "arn:aws:ecs:eu-west-2:123:task-definition/express:3"
    mock_ecs.describe_express_gateway_service.assert_called_once()
    mock_ecs.describe_service_revisions.assert_called_once_with(
        serviceRevisionArns=["arn:aws:ecs:eu-west-2:123:service-revision/rev/1"]
    )


def test_start_express_gateway_service_updates_scaling_target():
    mock_ecs = MagicMock()
    mock_ecs.get_paginator.return_value.paginate.return_value = [
        {
            "serviceArns": [
                "arn:aws:ecs:eu-west-2:123456789012:service/my-cluster/my-express"
            ]
        }
    ]

    with patch("cdk_post_deploy.boto3.client", return_value=mock_ecs):
        result = post.start_express_gateway_service("my-cluster", "my-express")

    assert result["statusCode"] == 200
    mock_ecs.update_express_gateway_service.assert_called_once_with(
        serviceArn="arn:aws:ecs:eu-west-2:123456789012:service/my-cluster/my-express",
        scalingTarget=post.EXPRESS_GATEWAY_ACTIVE_SCALING_TARGET,
    )


def test_cognito_https_callback_urls():
    assert post.cognito_https_callback_urls(
        "https://abc123.eu-west-2.elb.amazonaws.com"
    ) == [
        "https://abc123.eu-west-2.elb.amazonaws.com",
        "https://abc123.eu-west-2.elb.amazonaws.com/oauth2/idpresponse",
    ]
    assert post.cognito_https_callback_urls("app.example.com")[0].startswith("https://")


def test_update_user_pool_client_callback_urls_preserves_oauth_settings():
    mock_cognito = MagicMock()
    mock_cognito.describe_user_pool_client.return_value = {
        "UserPoolClient": {
            "ClientName": "app-client",
            "CallbackURLs": ["https://old.example.com"],
            "AllowedOAuthFlows": ["code"],
            "AllowedOAuthScopes": ["openid", "email", "profile"],
            "AllowedOAuthFlowsUserPoolClient": True,
            "SupportedIdentityProviders": ["COGNITO"],
            "ExplicitAuthFlows": ["ALLOW_REFRESH_TOKEN_AUTH"],
        }
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_cognito):
        post.update_user_pool_client_callback_urls(
            "pool-1",
            "client-1",
            [
                "https://new.example.com",
                "https://new.example.com/oauth2/idpresponse",
            ],
            aws_region="eu-west-2",
        )

    mock_cognito.update_user_pool_client.assert_called_once()
    kwargs = mock_cognito.update_user_pool_client.call_args.kwargs
    assert kwargs["UserPoolId"] == "pool-1"
    assert kwargs["ClientId"] == "client-1"
    assert kwargs["CallbackURLs"] == [
        "https://new.example.com",
        "https://new.example.com/oauth2/idpresponse",
    ]
    assert kwargs["AllowedOAuthFlows"] == ["code"]
    assert kwargs["AllowedOAuthScopes"] == ["openid", "email", "profile"]


def test_apply_cognito_alb_callback_fixup_skips_when_already_correct():
    mock_cognito = MagicMock()
    mock_cognito.describe_user_pool_client.return_value = {
        "UserPoolClient": {
            "CallbackURLs": post.cognito_https_callback_urls("https://app.example.com"),
        }
    }

    with patch("cdk_post_deploy.boto3.client", return_value=mock_cognito):
        changed = post.apply_cognito_alb_callback_fixup(
            user_pool_id="pool-1",
            client_id="client-1",
            redirect_base="https://app.example.com",
        )

    assert changed is False
    mock_cognito.update_user_pool_client.assert_not_called()


def test_target_group_arn_from_ecs_register_event():
    message = (
        "(service my-svc) registered 1 targets in "
        "(target-group arn:aws:elasticloadbalancing:eu-west-2:123:"
        "targetgroup/ecs-gateway-tg-abc/def)"
    )
    assert (
        post.target_group_arn_from_ecs_register_event(message)
        == "arn:aws:elasticloadbalancing:eu-west-2:123:targetgroup/ecs-gateway-tg-abc/def"
    )


def test_resolve_express_service_target_group_arn_waits_for_registration_event():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.side_effect = [
        {
            "services": [
                {
                    "events": [],
                    "runningCount": 0,
                    "desiredCount": 1,
                }
            ]
        },
        {
            "services": [
                {
                    "events": [
                        {
                            "message": (
                                "(service Demo-Redaction-ECSService) registered 1 targets in "
                                "(target-group arn:aws:elasticloadbalancing:eu-west-2:123:"
                                "targetgroup/ecs-gateway-tg-abc/def)"
                            )
                        }
                    ],
                    "runningCount": 1,
                    "desiredCount": 1,
                }
            ]
        },
    ]

    with (
        patch("cdk_post_deploy.boto3.client", return_value=mock_ecs),
        patch("cdk_post_deploy.time.sleep"),
        patch("cdk_post_deploy.time.monotonic", side_effect=[0, 0, 600]),
    ):
        arn = post.resolve_express_service_target_group_arn(
            "cluster",
            "Demo-Redaction-ECSService",
            max_wait_seconds=600,
            poll_interval_seconds=15,
        )

    assert arn == (
        "arn:aws:elasticloadbalancing:eu-west-2:123:targetgroup/ecs-gateway-tg-abc/def"
    )
    assert mock_ecs.describe_services.call_count == 2


def test_resolve_express_service_target_group_arn_from_task_ips():
    mock_ecs = MagicMock()
    mock_ecs.describe_services.return_value = {
        "services": [
            {
                "events": [],
                "runningCount": 1,
                "desiredCount": 1,
            }
        ]
    }
    mock_ecs.list_tasks.return_value = {"taskArns": ["arn:task/1"]}
    mock_ecs.describe_tasks.return_value = {
        "tasks": [
            {
                "attachments": [
                    {"details": [{"name": "privateIPv4Address", "value": "10.0.1.42"}]}
                ]
            }
        ]
    }
    mock_elbv2 = MagicMock()
    mock_elbv2.describe_target_groups.return_value = {
        "TargetGroups": [{"TargetGroupArn": "arn:tg/main"}]
    }
    mock_elbv2.describe_target_health.return_value = {
        "TargetHealthDescriptions": [{"Target": {"Id": "10.0.1.42"}}]
    }

    def client_factory(service_name, **_kwargs):
        return mock_elbv2 if service_name == "elbv2" else mock_ecs

    with (
        patch("cdk_post_deploy.boto3.client", side_effect=client_factory),
        patch("cdk_post_deploy.time.monotonic", return_value=0),
    ):
        arn = post.resolve_express_service_target_group_arn(
            "cluster",
            "Demo-Redaction-ECSService",
            load_balancer_arn="arn:lb/express",
            max_wait_seconds=600,
            poll_interval_seconds=15,
        )

    assert arn == "arn:tg/main"


def test_listener_actions_with_target_group_replaces_forward_arn():
    actions = [
        {"Type": "authenticate-cognito", "Order": 1},
        {
            "Type": "forward",
            "Order": 2,
            "TargetGroupArn": "arn:old",
            "ForwardConfig": {"TargetGroups": [{"TargetGroupArn": "arn:old"}]},
        },
    ]
    updated = post.listener_actions_with_target_group(actions, "arn:new")
    forward = next(action for action in updated if action["Type"] == "forward")
    assert forward["TargetGroupArn"] == "arn:new"
    assert forward["ForwardConfig"]["TargetGroups"][0]["TargetGroupArn"] == "arn:new"


def test_cognito_secret_payload_matches():
    desired = {
        "REDACTION_USER_POOL_ID": "eu-west-2_AAAA",
        "REDACTION_CLIENT_ID": "client",
        "REDACTION_CLIENT_SECRET": "secret",
    }
    current = json.dumps(
        {
            "REDACTION_USER_POOL_ID": "eu-west-2_OLD",
            "REDACTION_CLIENT_ID": "client",
            "REDACTION_CLIENT_SECRET": "secret",
        }
    )
    assert post.cognito_secret_payload_matches(current, desired) is False
    assert post.cognito_secret_payload_matches(json.dumps(desired), desired) is True


def test_listener_rule_has_cognito_auth():
    assert post.listener_rule_has_cognito_auth(
        [{"Type": "authenticate-cognito"}, {"Type": "forward"}]
    )
    assert not post.listener_rule_has_cognito_auth([{"Type": "forward"}])


def test_print_express_mode_next_steps(capsys, monkeypatch):
    def fake_get_stack_output(stack_name, output_key, region):
        outputs = {
            "ExpressServiceEndpoint": "main.example.ecs.eu-west-2.on.aws",
            "AgenticExpressEndpoint": "pi.example.ecs.eu-west-2.on.aws",
        }
        return outputs.get(output_key)

    monkeypatch.setattr(post, "get_stack_output", fake_get_stack_output)
    post.print_express_mode_next_steps(
        {
            "AWS_REGION": "eu-west-2",
            "ENABLE_PI_AGENT_EXPRESS_SERVICE": "True",
        }
    )
    out = capsys.readouterr().out
    assert "Wait 10 minutes for app deployment to finish." in out
    assert "sign in at the agentic redaction URL" in out
    assert "agentic redaction app can be accessed at" in out
    assert "https://main.example.ecs.eu-west-2.on.aws" in out
    assert "https://pi.example.ecs.eu-west-2.on.aws/" in out
    assert "agent.env" not in out


def test_print_express_mode_next_steps_magic_link_uses_cloudfront(capsys, monkeypatch):
    outputs = {
        "CloudFrontDistributionURL": "d123.cloudfront.net",
        "RedactionLoginUrl": "https://d123.cloudfront.net/?key=secret",
        "ExpressServiceEndpoint": "main.example.ecs.eu-west-2.on.aws",
        "AgenticExpressEndpoint": "pi.example.ecs.eu-west-2.on.aws",
    }
    monkeypatch.setattr(post, "get_stack_output", lambda _s, key, _r: outputs.get(key))
    post.print_express_mode_next_steps(
        {
            "AWS_REGION": "eu-west-2",
            "ENABLE_PI_AGENT_EXPRESS_SERVICE": "True",
            "USE_CLOUDFRONT": "True",
            "CLOUDFRONT_AUTH_MODE": "magic-link",
            "AGENT_ALB_PATH_PREFIX": "/agent",
        }
    )
    out = capsys.readouterr().out
    # Magic-link guidance must point at the CloudFront URL (with ?key= unlock), not the
    # direct ECS endpoints.
    assert "https://d123.cloudfront.net/?key=secret" in out
    assert "https://d123.cloudfront.net" in out
    assert "/agent/" in out
    assert "main.example.ecs.eu-west-2.on.aws" not in out
    assert "pi.example.ecs.eu-west-2.on.aws" not in out


def test_sync_pi_agent_doc_redaction_url_for_agentcore(tmp_path, monkeypatch):
    env_file = tmp_path / "agent.env"
    env_file.write_text(
        "AGENT_ORCHESTRATOR=agentcore\nDOC_REDACTION_GRADIO_URL=http://redaction:7860\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cdk_config.ENABLE_AGENTCORE_RUNTIME",
        "True",
        raising=False,
    )
    monkeypatch.setattr(
        post,
        "get_stack_output",
        lambda *_a, **_k: "main.example.ecs.eu-west-2.on.aws",
    )
    url = post.sync_pi_agent_doc_redaction_url_for_agentcore(
        pi_agent_env_path=env_file,
    )
    assert url == "https://main.example.ecs.eu-west-2.on.aws"
    text = env_file.read_text(encoding="utf-8")
    assert "DOC_REDACTION_GRADIO_URL=https://main.example.ecs.eu-west-2.on.aws" in text
    assert "http://redaction:7860" not in text


def test_print_headless_deployment_next_steps(capsys):
    post.print_headless_deployment_next_steps(
        {
            "AWS_REGION": "eu-west-2",
            "S3_OUTPUT_BUCKET_NAME": "my-output-bucket",
            "S3_BATCH_INPUT_PREFIX": "input/",
            "S3_BATCH_ENV_PREFIX": "input/config/",
            "S3_BATCH_LAMBDA_FUNCTION_NAME": "Headless-Redaction-S3BatchEcsTrigger",
            "ECS_LOG_GROUP_NAME": "/ecs/headless-redaction-ecsservice-logs",
        }
    )
    out = capsys.readouterr().out
    assert "Upload a PDF file for redaction to s3://my-output-bucket/input/" in out
    assert "example_headless_env_file.env" in out
    assert "s3://my-output-bucket/input/config/" in out
    assert "Headless-Redaction-S3BatchEcsTrigger" in out
    assert "s3://my-output-bucket/output/<session-folder>/" in out
    assert "tools/config.py" in out


def test_seed_headless_batch_s3_layout_creates_prefixes(tmp_path):
    from botocore.exceptions import ClientError

    example = tmp_path / "example_headless_env_file.env"
    example.write_text("DIRECT_MODE_TASK=redact\n", encoding="utf-8")

    missing = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "HeadObject",
    )
    s3 = MagicMock()
    s3.head_object.side_effect = missing

    with patch("cdk_post_deploy.boto3.client", return_value=s3):
        post.seed_headless_batch_s3_layout(
            "my-bucket",
            example_env_local_path=str(example),
            aws_region="eu-west-2",
        )

    put_calls = s3.put_object.call_args_list
    assert len(put_calls) == 3
    keys = [call.kwargs["Key"] for call in put_calls]
    assert "input/" in keys
    assert "input/config/" in keys
    assert "input/config/example_headless_env_file.env" in keys


def _sg_with_open_ingress(sg_id="sg-abc", port=443):
    return {
        "SecurityGroups": [
            {
                "GroupId": sg_id,
                "IpPermissions": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                        "Ipv6Ranges": [{"CidrIpv6": "::/0"}],
                        "PrefixListIds": [],
                    }
                ],
            }
        ]
    }


def test_restrict_single_alb_sg_replaces_open_ingress_with_prefix_list():
    order = []
    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = _sg_with_open_ingress("sg-abc", 443)
    ec2.revoke_security_group_ingress.side_effect = lambda **_k: order.append("revoke")
    ec2.authorize_security_group_ingress.side_effect = lambda **_k: order.append(
        "authorize"
    )

    status = post._restrict_single_alb_sg_to_cloudfront(ec2, "sg-abc", "pl-cloudfront")

    assert status == "ok"
    # Open rules are revoked BEFORE the prefix list is authorized (frees rule quota).
    assert order == ["revoke", "authorize"]

    auth = ec2.authorize_security_group_ingress.call_args
    assert auth.kwargs["GroupId"] == "sg-abc"
    auth_perm = auth.kwargs["IpPermissions"][0]
    assert auth_perm["FromPort"] == 443
    assert auth_perm["PrefixListIds"] == [
        {"PrefixListId": "pl-cloudfront", "Description": "CloudFront origin-facing"}
    ]

    revoke = ec2.revoke_security_group_ingress.call_args
    assert revoke.kwargs["GroupId"] == "sg-abc"
    revoke_perm = revoke.kwargs["IpPermissions"][0]
    assert revoke_perm["IpRanges"] == [{"CidrIp": "0.0.0.0/0"}]
    assert revoke_perm["Ipv6Ranges"] == [{"CidrIpv6": "::/0"}]


def test_restrict_single_alb_sg_restores_open_when_rules_limit_exceeded():
    from botocore.exceptions import ClientError

    limit_err = ClientError(
        {
            "Error": {
                "Code": "RulesPerSecurityGroupLimitExceeded",
                "Message": "The maximum number of rules per security group has been reached.",
            }
        },
        "AuthorizeSecurityGroupIngress",
    )

    calls = {"n": 0}

    def _authorize(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # first authorize = prefix list attempt -> fails
            raise limit_err
        return {}  # second authorize = restore open ingress -> succeeds

    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = _sg_with_open_ingress("sg-abc", 443)
    ec2.authorize_security_group_ingress.side_effect = _authorize
    ec2.describe_managed_prefix_lists.return_value = {
        "PrefixLists": [{"PrefixListId": "pl-cloudfront", "MaxEntries": 55}]
    }

    status = post._restrict_single_alb_sg_to_cloudfront(ec2, "sg-abc", "pl-cloudfront")

    assert status == "quota"
    # Revoked open, tried prefix (failed), then restored the open ingress.
    assert ec2.revoke_security_group_ingress.call_count == 1
    assert ec2.authorize_security_group_ingress.call_count == 2
    restore_perm = ec2.authorize_security_group_ingress.call_args_list[1].kwargs[
        "IpPermissions"
    ][0]
    assert restore_perm["IpRanges"] == [{"CidrIp": "0.0.0.0/0"}]
    assert restore_perm["Ipv6Ranges"] == [{"CidrIpv6": "::/0"}]


def test_restrict_single_alb_sg_is_idempotent_when_already_locked_down():
    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-abc",
                "IpPermissions": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "IpRanges": [],
                        "Ipv6Ranges": [],
                        "PrefixListIds": [{"PrefixListId": "pl-cloudfront"}],
                    }
                ],
            }
        ]
    }

    status = post._restrict_single_alb_sg_to_cloudfront(ec2, "sg-abc", "pl-cloudfront")

    assert status == "noop"
    ec2.authorize_security_group_ingress.assert_not_called()
    ec2.revoke_security_group_ingress.assert_not_called()


def _mock_aws_for_split(sg_to_arn, *, authorize_side_effect=None, cf_prefix="sg-cf"):
    """MagicMock serving both the ec2 and elbv2 clients for the split lockdown flow."""
    m = MagicMock()
    counter = {"n": 0}

    def _describe_sgs(**kwargs):
        if "GroupIds" in kwargs:  # managed SG lookup (has open ingress + VpcId)
            gid = kwargs["GroupIds"][0]
            data = _sg_with_open_ingress(gid, 443)
            data["SecurityGroups"][0]["VpcId"] = "vpc-123"
            return data
        return {"SecurityGroups": []}  # dedicated-SG reuse lookup -> none exists yet

    m.describe_security_groups.side_effect = _describe_sgs

    def _create_sg(**_kwargs):
        counter["n"] += 1
        return {"GroupId": f"{cf_prefix}-{counter['n']}"}

    m.create_security_group.side_effect = _create_sg
    if authorize_side_effect is not None:
        m.authorize_security_group_ingress.side_effect = authorize_side_effect

    lbs = [
        {
            "LoadBalancerArn": arn,
            "VpcId": "vpc-123",
            "SecurityGroups": [sg_id],
        }
        for sg_id, arn in sg_to_arn.items()
    ]
    paginator = MagicMock()
    paginator.paginate.return_value = [{"LoadBalancers": lbs}]
    m.get_paginator.return_value = paginator
    m.describe_managed_prefix_lists.return_value = {
        "PrefixLists": [{"PrefixListId": "pl-cloudfront", "MaxEntries": 55}]
    }
    return m


def test_restrict_express_albs_auto_raises_quota_on_limit(monkeypatch):
    from botocore.exceptions import ClientError

    limit_err = ClientError(
        {
            "Error": {
                "Code": "RulesPerSecurityGroupLimitExceeded",
                "Message": "The maximum number of rules per security group has been reached.",
            }
        },
        "AuthorizeSecurityGroupIngress",
    )
    # Even a single dedicated per-port SG can't fit the prefix list -> quota fallback.
    m = _mock_aws_for_split(
        {"sg-main": "arn-main"},
        authorize_side_effect=lambda **_k: (_ for _ in ()).throw(limit_err),
    )

    outputs = {"AlbSecurityGroupIdOutput": "sg-main"}
    monkeypatch.setattr(post, "get_stack_output", lambda _s, key, _r: outputs.get(key))
    monkeypatch.setattr(post, "get_security_group_rules_quota", lambda _r: 60.0)

    requested = {}

    def _request(region, desired_value):
        requested["region"] = region
        requested["desired"] = desired_value
        return True

    monkeypatch.setattr(post, "request_security_group_rules_quota_increase", _request)

    with patch("cdk_post_deploy.boto3.client", return_value=m):
        post.restrict_express_albs_to_cloudfront(
            stack_name="RedactionStack",
            region="eu-west-2",
            prefix_list_id="pl-cloudfront",
            auto_raise_quota=True,
        )

    assert requested["region"] == "eu-west-2"
    assert requested["desired"] >= 65
    m.set_security_groups.assert_not_called()


def test_restrict_express_albs_no_quota_request_when_disabled(monkeypatch):
    from botocore.exceptions import ClientError

    limit_err = ClientError(
        {"Error": {"Code": "RulesPerSecurityGroupLimitExceeded", "Message": "x"}},
        "AuthorizeSecurityGroupIngress",
    )
    m = _mock_aws_for_split(
        {"sg-main": "arn-main"},
        authorize_side_effect=lambda **_k: (_ for _ in ()).throw(limit_err),
    )

    monkeypatch.setattr(
        post,
        "get_stack_output",
        lambda _s, key, _r: {"AlbSecurityGroupIdOutput": "sg-main"}.get(key),
    )

    requested = {"called": False}

    def _request(region, desired_value):
        requested["called"] = True
        return True

    monkeypatch.setattr(post, "request_security_group_rules_quota_increase", _request)

    with patch("cdk_post_deploy.boto3.client", return_value=m):
        post.restrict_express_albs_to_cloudfront(
            stack_name="RedactionStack",
            region="eu-west-2",
            prefix_list_id="pl-cloudfront",
            auto_raise_quota=False,
        )

    assert requested["called"] is False


def test_restrict_express_albs_reads_both_stack_outputs(monkeypatch):
    m = _mock_aws_for_split({"sg-main": "arn-main", "sg-agent": "arn-agent"})

    outputs = {
        "AlbSecurityGroupIdOutput": "sg-main",
        "AgenticAlbSecurityGroupIdOutput": "sg-agent",
    }
    monkeypatch.setattr(post, "get_stack_output", lambda _s, key, _r: outputs.get(key))
    monkeypatch.setattr(
        post,
        "resolve_cloudfront_origin_prefix_list_id",
        lambda *_a, **_k: "pl-cloudfront",
    )

    with patch("cdk_post_deploy.boto3.client", return_value=m):
        processed = post.restrict_express_albs_to_cloudfront(
            stack_name="RedactionStack", region="eu-west-2"
        )

    assert processed == ["sg-main", "sg-agent"]
    # One dedicated CloudFront SG created per ALB (one open port each) and attached.
    assert m.create_security_group.call_count == 2
    assert m.set_security_groups.call_count == 2
    # Open internet ingress revoked from each managed SG after CF SGs are attached.
    assert m.revoke_security_group_ingress.call_count == 2


def test_restrict_single_alb_split_attaches_then_revokes(monkeypatch):
    """The dedicated CF SG must be attached to the LB BEFORE open ingress is revoked
    (union semantics => no downtime window)."""
    order = []
    m = _mock_aws_for_split({"sg-main": "arn-main"})
    m.set_security_groups.side_effect = lambda **_k: order.append("attach")
    m.revoke_security_group_ingress.side_effect = lambda **_k: order.append("revoke")

    status = post._restrict_single_alb_to_cloudfront_split(
        m, m, "sg-main", "pl-cloudfront"
    )

    assert status == "ok"
    assert order == ["attach", "revoke"]
    attached = m.set_security_groups.call_args.kwargs["SecurityGroups"]
    assert "sg-main" in attached and any(s.startswith("sg-cf") for s in attached)


def test_restrict_single_alb_split_noop_when_already_locked():
    """No open ingress + extra SG already attached => idempotent no-op."""
    m = MagicMock()
    m.describe_security_groups.side_effect = lambda **kw: (
        {
            "SecurityGroups": [
                {"GroupId": kw["GroupIds"][0], "VpcId": "vpc-1", "IpPermissions": []}
            ]
        }
        if "GroupIds" in kw
        else {"SecurityGroups": []}
    )
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "LoadBalancers": [
                {
                    "LoadBalancerArn": "arn-main",
                    "VpcId": "vpc-1",
                    "SecurityGroups": ["sg-main", "sg-cf-1"],
                }
            ]
        }
    ]
    m.get_paginator.return_value = paginator

    status = post._restrict_single_alb_to_cloudfront_split(
        m, m, "sg-main", "pl-cloudfront"
    )

    assert status == "noop"
    m.set_security_groups.assert_not_called()
    m.create_security_group.assert_not_called()


def test_restrict_single_alb_split_errors_when_no_load_balancer():
    m = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"LoadBalancers": []}]
    m.get_paginator.return_value = paginator

    status = post._restrict_single_alb_to_cloudfront_split(
        m, m, "sg-main", "pl-cloudfront"
    )

    assert status == "error"
    m.set_security_groups.assert_not_called()


def test_restrict_express_albs_skips_when_no_sg_outputs(monkeypatch, capsys):
    monkeypatch.setattr(post, "get_stack_output", lambda *_a, **_k: None)
    monkeypatch.setattr(
        post,
        "resolve_cloudfront_origin_prefix_list_id",
        lambda *_a, **_k: "pl-cloudfront",
    )

    with patch("cdk_post_deploy.boto3.client", return_value=MagicMock()) as client:
        processed = post.restrict_express_albs_to_cloudfront(region="eu-west-2")

    assert processed == []
    client.assert_not_called()
    assert "no ALB security group ids" in capsys.readouterr().out
