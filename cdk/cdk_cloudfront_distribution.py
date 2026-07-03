"""CloudFront distribution for doc_redaction (RedactionStack; optional WAF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

from aws_cdk import CfnOutput
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from cdk_cloudfront_auth import (
    MagicLinkAuthResources,
    create_forwarded_host_function,
    create_magic_link_auth,
    emit_magic_link_outputs,
    magic_link_function_association,
)
from cdk_cloudfront_headers import (
    create_secure_cloudfront_response_headers_policy,
    resolve_cloudfront_csp_urls,
)
from cdk_functions import create_web_acl_with_common_rules
from constructs import Construct


def express_endpoint_hostname(endpoint: str) -> str:
    """Hostname from ExpressServiceEndpoint / attr_endpoint (URL or host)."""
    value = (endpoint or "").strip()
    if "://" in value:
        parsed = urlparse(value)
        return (parsed.hostname or value).strip()
    return value.split("/")[0].strip()


def parse_geo_restriction_locations(raw: str) -> Optional[List[str]]:
    locations = [
        part.strip().upper() for part in (raw or "").split(",") if part.strip()
    ]
    return locations or None


def _behavior_options(
    origin: cloudfront.IOrigin,
    *,
    response_headers_policy: Optional[cloudfront.IResponseHeadersPolicy],
    function_associations: Optional[Sequence[cloudfront.FunctionAssociation]] = None,
    origin_request_policy: Optional[cloudfront.IOriginRequestPolicy] = None,
) -> cloudfront.BehaviorOptions:
    return cloudfront.BehaviorOptions(
        origin=origin,
        viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
        cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
        origin_request_policy=(
            origin_request_policy or cloudfront.OriginRequestPolicy.ALL_VIEWER
        ),
        response_headers_policy=response_headers_policy,
        function_associations=list(function_associations or ()),
    )


@dataclass(frozen=True)
class RedactionCloudFrontResources:
    distribution: cloudfront.Distribution
    magic_link: Optional[MagicLinkAuthResources] = None


def create_redaction_cloudfront_distribution(
    scope: Construct,
    construct_id: str,
    *,
    distribution_comment: str,
    cognito_redirection_url: str,
    cloudfront_domain: str,
    cognito_user_pool_domain_prefix: str,
    aws_region: str,
    cognito_user_pool_login_url: str,
    ssl_certificate_domain: str,
    enable_secure_response_headers: bool,
    attach_secure_response_headers: bool = False,
    geo_restriction_raw: str,
    enable_cloudfront_waf: bool,
    web_acl_name: str,
    auth_mode: str,
    magic_link_cookie_name: str,
    magic_link_cookie_max_age_sec: int,
    custom_header_name: str,
    custom_header_value: str,
    cdk_prefix: str,
    resource_removal_policy,
    # Origin mode: exactly one of alb or main express endpoint required.
    alb: Optional[elbv2.IApplicationLoadBalancer] = None,
    main_express_endpoint: str = "",
    agentic_express_endpoint: str = "",
    agentic_path_prefix: str = "/agent",
) -> RedactionCloudFrontResources:
    """Create CloudFront distribution in RedactionStack (no WAF unless opted in)."""
    geo_locations = parse_geo_restriction_locations(geo_restriction_raw)
    geo_restrict = (
        cloudfront.GeoRestriction.allowlist(*geo_locations) if geo_locations else None
    )

    # The secure response headers policy is created as a standalone resource (so it
    # appears in the account's CloudFront policy list and can be attached manually), but
    # it is only wired onto the distribution's behaviors when ``attach_secure_response_
    # headers`` is set. Attaching the CSP/security headers tends to break demonstration
    # mode (e.g. Cognito redirect flows and mixed Express origins), so it is created but
    # left detached by default.
    response_headers_policy = None
    if enable_secure_response_headers:
        app_origin, cognito_login_url = resolve_cloudfront_csp_urls(
            cognito_redirection_url=cognito_redirection_url,
            cloudfront_domain=cloudfront_domain,
            cognito_user_pool_domain_prefix=cognito_user_pool_domain_prefix,
            aws_region=aws_region,
            cognito_user_pool_login_url=cognito_user_pool_login_url,
            ssl_certificate_domain=ssl_certificate_domain,
        )
        policy_name = f"{cdk_prefix}SecureResponseHeaders"[:128]
        response_headers_policy = create_secure_cloudfront_response_headers_policy(
            scope,
            f"{construct_id}SecureResponseHeadersPolicy",
            policy_name=policy_name,
            app_origin=app_origin,
            cognito_login_url=cognito_login_url,
        )

    attached_response_headers_policy = (
        response_headers_policy if attach_secure_response_headers else None
    )

    magic_link: Optional[MagicLinkAuthResources] = None
    function_associations: List[cloudfront.FunctionAssociation] = []
    if auth_mode == "magic-link":
        magic_link = create_magic_link_auth(
            scope,
            f"{construct_id}MagicLink",
            cookie_name=magic_link_cookie_name,
            cookie_max_age_sec=magic_link_cookie_max_age_sec,
        )
        # The magic-link viewer-request function also forwards viewer host/proto.
        function_associations.append(
            magic_link_function_association(magic_link.auth_function)
        )
    else:
        # No magic-link function on this behavior, so attach a lightweight
        # viewer-request function that forwards the viewer host/proto to the
        # origin. Without it, CloudFront (ALL_VIEWER_EXCEPT_HOST_HEADER) presents
        # the origin's own *.ecs.on.aws host and Gradio emits absolute origin URLs
        # for its assets — which the browser can't reach once the origin SG is
        # locked to CloudFront-only.
        forwarded_host_function = create_forwarded_host_function(
            scope, f"{construct_id}Fwd"
        )
        function_associations.append(
            magic_link_function_association(forwarded_host_function)
        )

    custom_headers: Dict[str, str] = {}
    if custom_header_name and custom_header_value:
        custom_headers[custom_header_name] = custom_header_value

    additional_behaviors: Dict[str, cloudfront.BehaviorOptions] = {}

    if alb is not None:
        origin = origins.LoadBalancerV2Origin(
            alb,
            custom_headers=custom_headers or None,
            origin_shield_enabled=False,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
        )
        default_origin = origin
        # Legacy ALB listener rule matches the CloudFront distribution domain, so the
        # viewer Host header must be forwarded to the origin unchanged.
        origin_request_policy: cloudfront.IOriginRequestPolicy = (
            cloudfront.OriginRequestPolicy.ALL_VIEWER
        )
    else:
        main_host = express_endpoint_hostname(main_express_endpoint)
        if not main_host:
            raise ValueError(
                "main_express_endpoint is required for Express CloudFront origin."
            )
        # ECS Express managed ALB listener rules route on the per-service *.on.aws host
        # header. CloudFront must present each origin's own hostname (not the viewer's
        # CloudFront domain), so strip the viewer Host and let CloudFront set the origin
        # domain as Host instead.
        origin_request_policy = (
            cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER
        )
        default_origin = origins.HttpOrigin(
            main_host,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
            custom_headers=custom_headers or None,
        )
        agentic_host = express_endpoint_hostname(agentic_express_endpoint)
        prefix = (agentic_path_prefix or "/agent").strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        if agentic_host and agentic_host != main_host:
            agentic_origin = origins.HttpOrigin(
                agentic_host,
                protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
                custom_headers=custom_headers or None,
            )
            additional_behaviors[prefix] = _behavior_options(
                agentic_origin,
                response_headers_policy=attached_response_headers_policy,
                function_associations=function_associations,
                origin_request_policy=origin_request_policy,
            )
            additional_behaviors[f"{prefix}*"] = _behavior_options(
                agentic_origin,
                response_headers_policy=attached_response_headers_policy,
                function_associations=function_associations,
                origin_request_policy=origin_request_policy,
            )

    default_behavior = _behavior_options(
        default_origin,
        response_headers_policy=attached_response_headers_policy,
        function_associations=function_associations,
        origin_request_policy=origin_request_policy,
    )

    web_acl_id = None
    if enable_cloudfront_waf:
        web_acl = create_web_acl_with_common_rules(scope, web_acl_name)
        web_acl_id = web_acl.attr_arn

    distribution_kwargs = {
        "comment": distribution_comment,
        "geo_restriction": geo_restrict,
        "default_behavior": default_behavior,
        "additional_behaviors": additional_behaviors or None,
    }
    if web_acl_id:
        distribution_kwargs["web_acl_id"] = web_acl_id

    distribution = cloudfront.Distribution(
        scope,
        f"{construct_id}Distribution",
        **distribution_kwargs,
    )
    distribution.apply_removal_policy(resource_removal_policy)

    CfnOutput(
        scope,
        "CloudFrontDistributionURL",
        value=distribution.domain_name,
        description="CloudFront distribution domain name",
    )

    if magic_link is not None:
        emit_magic_link_outputs(
            scope,
            distribution_domain_name=distribution.domain_name,
            auth_token=magic_link.auth_token,
        )

    return RedactionCloudFrontResources(
        distribution=distribution, magic_link=magic_link
    )
