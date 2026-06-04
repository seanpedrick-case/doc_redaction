"""CloudFront response headers policy (CSP and related security headers)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from aws_cdk import Duration
from aws_cdk import aws_cloudfront as cloudfront
from constructs import Construct

# Template exported from AWS; placeholders {APP-URL} and {COGNITO-APP-CLIENT-LOGIN-URL}.
_CSP_TEMPLATE = (
    "default-src 'self'; script-src 'self' cdnjs.cloudflare.com 'unsafe-inline'; "
    "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self' https://fonts.gstatic.com data:; "
    "connect-src 'self' wss://{app_hostname} https://cdnjs.cloudflare.com; "
    "form-action 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'; "
    "manifest-src 'self' {cognito_login_url}; upgrade-insecure-requests;"
)

RESPONSE_HEADERS_POLICY_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "config" / "response-headers-policy-config.json"
)


def normalize_https_origin(url: str) -> str:
    """Return a canonical https origin (scheme + host, no path)."""
    value = (url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.hostname:
        return value.rstrip("/")
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.hostname
    return f"{scheme}://{netloc}".rstrip("/")


def hostname_from_origin(origin: str) -> str:
    parsed = urlparse(origin if "://" in origin else f"https://{origin}")
    return parsed.hostname or origin.strip()


def cognito_hosted_ui_base_url(domain_prefix: str, region: str) -> str:
    prefix = (domain_prefix or "").strip()
    if not prefix:
        return ""
    return f"https://{prefix}.auth.{region}.amazoncognito.com"


def build_content_security_policy(
    *,
    app_origin: str,
    cognito_login_url: str,
) -> str:
    origin = normalize_https_origin(app_origin)
    hostname = hostname_from_origin(origin)
    cognito_url = normalize_https_origin(cognito_login_url)
    return _CSP_TEMPLATE.format(app_hostname=hostname, cognito_login_url=cognito_url)


def create_secure_cloudfront_response_headers_policy(
    scope: Construct,
    construct_id: str,
    *,
    policy_name: str,
    app_origin: str,
    cognito_login_url: str,
    comment: str = "Secure response headers with CSP for doc_redaction",
) -> cloudfront.ResponseHeadersPolicy:
    """Response headers policy aligned with config/response-headers-policy-config.json."""
    cors_origin = normalize_https_origin(app_origin)
    csp = build_content_security_policy(
        app_origin=app_origin,
        cognito_login_url=cognito_login_url,
    )

    return cloudfront.ResponseHeadersPolicy(
        scope,
        construct_id,
        response_headers_policy_name=policy_name,
        comment=comment,
        cors_behavior=cloudfront.ResponseHeadersCorsBehavior(
            access_control_allow_credentials=False,
            access_control_allow_headers=["*"],
            access_control_allow_methods=["ALL"],
            access_control_allow_origins=[cors_origin],
            access_control_max_age=Duration.seconds(600),
            origin_override=True,
        ),
        security_headers_behavior=cloudfront.ResponseSecurityHeadersBehavior(
            content_security_policy=cloudfront.ResponseHeadersContentSecurityPolicy(
                content_security_policy=csp,
                override=True,
            ),
            content_type_options=cloudfront.ResponseHeadersContentTypeOptions(
                override=True
            ),
            frame_options=cloudfront.ResponseHeadersFrameOptions(
                frame_option=cloudfront.HeadersFrameOption.SAMEORIGIN,
                override=True,
            ),
            referrer_policy=cloudfront.ResponseHeadersReferrerPolicy(
                referrer_policy=cloudfront.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
                override=True,
            ),
            strict_transport_security=cloudfront.ResponseHeadersStrictTransportSecurity(
                access_control_max_age=Duration.seconds(31536000),
                include_subdomains=True,
                preload=False,
                override=True,
            ),
            xss_protection=cloudfront.ResponseHeadersXSSProtection(
                protection=True,
                mode_block=True,
                override=True,
            ),
        ),
        custom_headers_behavior=cloudfront.ResponseCustomHeadersBehavior(
            custom_headers=[
                cloudfront.ResponseCustomHeader(
                    header="Permissions-Policy",
                    value=(
                        "accelerometer=(), autoplay=(), camera=(), "
                        "cross-origin-isolated=(), display-capture=(), encrypted-media=(), "
                        "fullscreen=(), geolocation=(), gyroscope=(), keyboard-map=(), "
                        "magnetometer=(), microphone=(), midi=(), payment=(), "
                        "picture-in-picture=(), publickey-credentials-get=(), "
                        "screen-wake-lock=(), sync-xhr=(), usb=(), web-share=(), "
                        "xr-spatial-tracking=()"
                    ),
                    override=True,
                )
            ]
        ),
        remove_headers=["Server"],
    )


def resolve_cloudfront_csp_urls(
    *,
    cognito_redirection_url: str,
    cloudfront_domain: str,
    cognito_user_pool_domain_prefix: str,
    aws_region: str,
    cognito_user_pool_login_url: str = "",
    ssl_certificate_domain: str = "",
) -> tuple[str, str]:
    """
    Return (app_origin, cognito_login_url) for CSP/CORS substitution.

    App origin prefers COGNITO_REDIRECTION_URL (canonical browser URL), then
    https://SSL_CERTIFICATE_DOMAIN, then https://CLOUDFRONT_DOMAIN.
    Cognito login URL uses COGNITO_USER_POOL_LOGIN_URL when set, else the
    hosted UI base URL derived from COGNITO_USER_POOL_DOMAIN_PREFIX.
    """
    app_origin = normalize_https_origin(cognito_redirection_url)
    if not app_origin or "placeholder" in app_origin.lower():
        if ssl_certificate_domain.strip():
            app_origin = normalize_https_origin(ssl_certificate_domain)
        elif cloudfront_domain.strip():
            app_origin = normalize_https_origin(cloudfront_domain)

    login_url = (cognito_user_pool_login_url or "").strip()
    if not login_url:
        login_url = cognito_hosted_ui_base_url(
            cognito_user_pool_domain_prefix, aws_region
        )
    return app_origin, normalize_https_origin(login_url)
