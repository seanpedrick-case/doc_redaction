"""CloudFront in RedactionStack: magic-link auth and distribution synth."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_cloudfront_auth import (
    build_forwarded_host_viewer_request_js,
    build_magic_link_viewer_request_js,
)
from cdk_cloudfront_distribution import (
    express_endpoint_hostname,
    parse_geo_restriction_locations,
)


def test_express_endpoint_hostname():
    assert (
        express_endpoint_hostname("https://abc123.eu-west-2.on.aws")
        == "abc123.eu-west-2.on.aws"
    )
    assert express_endpoint_hostname("abc123.eu-west-2.on.aws") == (
        "abc123.eu-west-2.on.aws"
    )


def test_parse_geo_restriction_locations():
    assert parse_geo_restriction_locations("GB") == ["GB"]
    assert parse_geo_restriction_locations("gb, us") == ["GB", "US"]
    assert parse_geo_restriction_locations("") is None


def test_magic_link_viewer_request_js_embeds_token_and_cookie():
    js = build_magic_link_viewer_request_js(
        token="a1b2c3d4e5f6789012345678901234ab",
        cookie_name="doc-redaction-auth",
        cookie_max_age_sec=604800,
    )
    assert "a1b2c3d4e5f6789012345678901234ab" in js
    assert "doc-redaction-auth" in js
    assert "604800" in js
    assert "statusCode: 302" in js
    assert "statusCode: 401" in js


def test_magic_link_viewer_request_js_forwards_viewer_host():
    """Authorized requests must forward the viewer host so Gradio builds asset URLs
    against the CloudFront domain, not the *.on.aws origin host. The proto is NOT set
    here: x-forwarded-proto is a CloudFront-disallowed edge-function header (HTTP 502).
    """
    js = build_magic_link_viewer_request_js(
        token="a1b2c3d4e5f6789012345678901234ab",
        cookie_name="doc-redaction-auth",
        cookie_max_age_sec=604800,
    )
    assert "x-forwarded-host" in js
    assert "x-forwarded-proto" not in js


def test_forwarded_host_viewer_request_js_sets_headers():
    js = build_forwarded_host_viewer_request_js()
    assert "x-forwarded-host" in js
    # x-forwarded-proto is CloudFront-disallowed for edge functions; supplied via a
    # static custom origin header instead (see distribution config).
    assert "x-forwarded-proto" not in js
    assert "request.headers.host.value" in js
    assert "return request;" in js


def test_cloudfront_without_magic_link_still_forwards_host():
    """auth_mode='none' must still attach a viewer-request function that forwards
    the viewer host (otherwise Gradio assets 404/time out behind CloudFront)."""
    template = _synth_cloudfront_with_headers(attach=False)
    template.resource_count_is("AWS::CloudFront::Function", 1)


def test_express_cloudfront_synth_no_waf():
    from aws_cdk import App, Environment, Stack, assertions
    from cdk_cloudfront_distribution import create_redaction_cloudfront_distribution
    from cdk_functions import managed_resource_removal_policy

    app = App()
    stack = Stack(
        app,
        "ExpressCloudFrontTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )

    create_redaction_cloudfront_distribution(
        stack,
        "Cf",
        distribution_comment="test-dist",
        cognito_redirection_url="https://main.example.on.aws",
        cloudfront_domain="d111.cloudfront.net",
        cognito_user_pool_domain_prefix="demo",
        aws_region="eu-west-2",
        cognito_user_pool_login_url="",
        ssl_certificate_domain="",
        enable_secure_response_headers=False,
        geo_restriction_raw="GB",
        enable_cloudfront_waf=False,
        web_acl_name="test-waf",
        auth_mode="magic-link",
        magic_link_cookie_name="doc-redaction-auth",
        magic_link_cookie_max_age_sec=604800,
        custom_header_name="",
        custom_header_value="",
        cdk_prefix="Test",
        resource_removal_policy=managed_resource_removal_policy(),
        main_express_endpoint="https://main.example.on.aws",
        agentic_express_endpoint="https://agentic.example.on.aws",
        agentic_path_prefix="/agent",
    )

    template = assertions.Template.from_stack(stack)
    template.resource_count_is("AWS::WAFv2::WebACL", 0)
    template.resource_count_is("AWS::CloudFront::Distribution", 1)
    template.resource_count_is("AWS::CloudFront::Function", 1)
    resources = template.to_json()["Resources"]
    dist = next(
        r for r in resources.values() if r["Type"] == "AWS::CloudFront::Distribution"
    )
    dist_config = dist["Properties"]["DistributionConfig"]
    behaviors = dist_config["CacheBehaviors"]
    assert len(behaviors) >= 2

    # Express origins must NOT forward the viewer Host header: the ECS managed ALB routes
    # on each service's own *.on.aws host, so CloudFront sends the origin domain as Host.
    # ALL_VIEWER_EXCEPT_HOST_HEADER managed policy id (stable AWS constant).
    all_viewer_except_host = "b689b0a8-53d0-40ab-baf2-68738e2966ac"
    assert (
        dist_config["DefaultCacheBehavior"]["OriginRequestPolicyId"]
        == all_viewer_except_host
    )
    for behavior in behaviors:
        assert behavior["OriginRequestPolicyId"] == all_viewer_except_host


def test_express_origins_set_static_forwarded_proto_header():
    """Every Express origin must carry a static X-Forwarded-Proto: https custom header
    so Gradio emits https asset URLs (the edge function can't set this header)."""
    from aws_cdk import App, Environment, Stack, assertions
    from cdk_cloudfront_distribution import create_redaction_cloudfront_distribution
    from cdk_functions import managed_resource_removal_policy

    app = App()
    stack = Stack(
        app,
        "ExpressFwdProtoTest",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    create_redaction_cloudfront_distribution(
        stack,
        "Cf",
        distribution_comment="test-dist",
        cognito_redirection_url="https://main.example.on.aws",
        cloudfront_domain="d111.cloudfront.net",
        cognito_user_pool_domain_prefix="demo",
        aws_region="eu-west-2",
        cognito_user_pool_login_url="",
        ssl_certificate_domain="",
        enable_secure_response_headers=False,
        geo_restriction_raw="GB",
        enable_cloudfront_waf=False,
        web_acl_name="test-waf",
        auth_mode="magic-link",
        magic_link_cookie_name="doc-redaction-auth",
        magic_link_cookie_max_age_sec=604800,
        custom_header_name="",
        custom_header_value="",
        cdk_prefix="Test",
        resource_removal_policy=managed_resource_removal_policy(),
        main_express_endpoint="https://main.example.on.aws",
        agentic_express_endpoint="https://agentic.example.on.aws",
        agentic_path_prefix="/agent",
    )
    template = assertions.Template.from_stack(stack)
    dist = next(
        r
        for r in template.to_json()["Resources"].values()
        if r["Type"] == "AWS::CloudFront::Distribution"
    )
    origins_cfg = dist["Properties"]["DistributionConfig"]["Origins"]
    assert origins_cfg, "expected at least one origin"
    for origin in origins_cfg:
        header_pairs = {
            h["HeaderName"]: h["HeaderValue"]
            for h in origin.get("OriginCustomHeaders", [])
        }
        assert header_pairs.get("X-Forwarded-Proto") == "https"


def _synth_cloudfront_with_headers(attach: bool):
    from aws_cdk import App, Environment, Stack, assertions
    from cdk_cloudfront_distribution import create_redaction_cloudfront_distribution
    from cdk_functions import managed_resource_removal_policy

    app = App()
    stack = Stack(
        app,
        f"HeadersTest{'Attach' if attach else 'Detach'}",
        env=Environment(account="123456789012", region="eu-west-2"),
    )
    create_redaction_cloudfront_distribution(
        stack,
        "Cf",
        distribution_comment="test-dist",
        cognito_redirection_url="https://main.example.on.aws",
        cloudfront_domain="d111.cloudfront.net",
        cognito_user_pool_domain_prefix="demo",
        aws_region="eu-west-2",
        cognito_user_pool_login_url="",
        ssl_certificate_domain="",
        enable_secure_response_headers=True,
        attach_secure_response_headers=attach,
        geo_restriction_raw="GB",
        enable_cloudfront_waf=False,
        web_acl_name="test-waf",
        auth_mode="none",
        magic_link_cookie_name="doc-redaction-auth",
        magic_link_cookie_max_age_sec=604800,
        custom_header_name="",
        custom_header_value="",
        cdk_prefix="Test",
        resource_removal_policy=managed_resource_removal_policy(),
        main_express_endpoint="https://main.example.on.aws",
        agentic_express_endpoint="https://agentic.example.on.aws",
        agentic_path_prefix="/agent",
    )
    return assertions.Template.from_stack(stack)


def _behavior_response_headers_ids(template) -> list:
    dist = next(
        r
        for r in template.to_json()["Resources"].values()
        if r["Type"] == "AWS::CloudFront::Distribution"
    )
    config = dist["Properties"]["DistributionConfig"]
    behaviors = [config["DefaultCacheBehavior"], *config.get("CacheBehaviors", [])]
    return [b.get("ResponseHeadersPolicyId") for b in behaviors]


def test_response_headers_policy_created_but_detached_by_default():
    template = _synth_cloudfront_with_headers(attach=False)
    # The policy resource is created (appears in the account's policy list)...
    template.resource_count_is("AWS::CloudFront::ResponseHeadersPolicy", 1)
    # ...but is not attached to any distribution behavior.
    ids = _behavior_response_headers_ids(template)
    assert all(policy_id is None for policy_id in ids), ids


def test_response_headers_policy_attached_when_requested():
    template = _synth_cloudfront_with_headers(attach=True)
    template.resource_count_is("AWS::CloudFront::ResponseHeadersPolicy", 1)
    ids = _behavior_response_headers_ids(template)
    assert ids, ids
    assert all(policy_id is not None for policy_id in ids), ids
