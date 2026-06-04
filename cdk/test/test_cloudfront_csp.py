"""Tests for CloudFront CSP / response headers helpers."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_cloudfront_headers import (
    build_content_security_policy,
    cognito_hosted_ui_base_url,
    normalize_https_origin,
    resolve_cloudfront_csp_urls,
)


def test_normalize_https_origin():
    assert (
        normalize_https_origin("d111.cloudfront.net") == "https://d111.cloudfront.net"
    )
    assert normalize_https_origin("https://redaction.example.com/") == (
        "https://redaction.example.com"
    )


def test_build_content_security_policy_substitutes_hosts():
    csp = build_content_security_policy(
        app_origin="https://d111.cloudfront.net",
        cognito_login_url="https://my-prefix.auth.eu-west-2.amazoncognito.com",
    )
    assert "wss://d111.cloudfront.net" in csp
    assert "https://my-prefix.auth.eu-west-2.amazoncognito.com" in csp
    assert "cdnjs.cloudflare.com" in csp


def test_resolve_cloudfront_csp_urls_prefers_cognito_redirection():
    app_origin, login = resolve_cloudfront_csp_urls(
        cognito_redirection_url="https://redaction.example.com",
        cloudfront_domain="d111.cloudfront.net",
        cognito_user_pool_domain_prefix="my-prefix",
        aws_region="eu-west-2",
    )
    assert app_origin == "https://redaction.example.com"
    assert login == "https://my-prefix.auth.eu-west-2.amazoncognito.com"


def test_resolve_cloudfront_csp_urls_login_override():
    _, login = resolve_cloudfront_csp_urls(
        cognito_redirection_url="https://app.example.com",
        cloudfront_domain="d111.cloudfront.net",
        cognito_user_pool_domain_prefix="my-prefix",
        aws_region="eu-west-2",
        cognito_user_pool_login_url="https://custom-login.example.com",
    )
    assert login == "https://custom-login.example.com"


def test_cognito_hosted_ui_base_url():
    assert cognito_hosted_ui_base_url("redaction-123", "eu-west-1") == (
        "https://redaction-123.auth.eu-west-1.amazoncognito.com"
    )
