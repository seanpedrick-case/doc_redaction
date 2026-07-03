"""CloudFront magic-link edge auth (demo / NDX:Try-style shared secret)."""

from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import CfnOutput, CustomResource, Duration
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_lambda as lambda_
from aws_cdk import custom_resources as cr
from constructs import Construct

# CloudFront (viewer-request) snippet: forward the real viewer host to the origin so
# Gradio builds asset URLs (config.root, theme.css, custom components) against the
# CloudFront domain instead of the origin's own *.ecs.on.aws host.
# Needed because the origin request policy is ALL_VIEWER_EXCEPT_HOST_HEADER — the
# Host header is stripped so ECS Express can route on its own hostname, which would
# otherwise make Gradio emit absolute origin URLs the browser can't reach once the
# origin is locked to CloudFront-only. ``x-forwarded-host`` is NOT stripped.
#
# NOTE: ``x-forwarded-proto`` is a CloudFront-disallowed header for edge functions —
# adding it here makes CloudFront reject the request with HTTP 502 ("tried to add a
# disallowed header"). The https scheme is instead supplied to the origin via a static
# ``X-Forwarded-Proto: https`` custom origin header on the distribution (see
# cdk_cloudfront_distribution.py), which is safe because viewers always reach
# CloudFront over HTTPS (REDIRECT_TO_HTTPS).
_FORWARDED_HOST_INJECTION_JS = (
    "if (request.headers.host && request.headers.host.value) {\n"
    "      request.headers['x-forwarded-host'] = "
    "{ value: request.headers.host.value };\n"
    "    }"
)


def build_forwarded_host_viewer_request_js() -> str:
    """CloudFront Function (viewer-request) JS: forward viewer host/proto only.

    Used on CloudFront deployments *without* magic-link auth (a behavior can only
    have one viewer-request function; magic-link injects the same headers itself).
    """
    return f"""function handler(event) {{
  var request = event.request;
  {_FORWARDED_HOST_INJECTION_JS}
  return request;
}}
"""


def build_magic_link_viewer_request_js(
    *,
    token: str,
    cookie_name: str,
    cookie_max_age_sec: int,
    login_url_hint: str = "RedactionLoginUrl",
) -> str:
    """CloudFront Function (viewer-request) JS; token is embedded at deploy time."""
    # Token is hex-only from secrets.token_hex; cookie name is config-controlled alnum+dash.
    return f"""function handler(event) {{
  var request = event.request;
  var TOKEN = '{token}';
  var COOKIE_NAME = '{cookie_name}';
  var MAX_AGE = {int(cookie_max_age_sec)};
  var LOGIN_HINT = '{login_url_hint}';

  var qs = request.querystring || {{}};
  if (qs.key && qs.key.value === TOKEN) {{
    var clean = 'https://' + request.headers.host.value + request.uri;
    return {{
      statusCode: 302,
      statusDescription: 'Found',
      headers: {{
        location: {{ value: clean }},
        'cache-control': {{ value: 'no-store' }}
      }},
      cookies: {{
        [COOKIE_NAME]: {{
          value: TOKEN,
          attributes: 'Max-Age=' + MAX_AGE + '; Secure; HttpOnly; SameSite=Lax; Path=/'
        }}
      }}
    }};
  }}

  var cookies = request.cookies || {{}};
  if (cookies[COOKIE_NAME] && cookies[COOKIE_NAME].value === TOKEN) {{
    {_FORWARDED_HOST_INJECTION_JS}
    return request;
  }}

  return {{
    statusCode: 401,
    statusDescription: 'Unauthorized',
    headers: {{
      'content-type': {{ value: 'text/html; charset=utf-8' }},
      'cache-control': {{ value: 'no-store' }}
    }},
    body: {{
      encoding: 'text',
      data: '<!DOCTYPE html><html><body><h1>Access denied</h1><p>Use the '
        + LOGIN_HINT
        + ' stack output (URL with <code>?key=</code>) to unlock this demo for 7 days.</p></body></html>'
    }}
  }};
}}
"""


@dataclass(frozen=True)
class MagicLinkAuthResources:
    auth_function: cloudfront.Function
    auth_token: str
    token_custom_resource: CustomResource


def create_magic_link_auth(
    scope: Construct,
    construct_id: str,
    *,
    cookie_name: str,
    cookie_max_age_sec: int,
) -> MagicLinkAuthResources:
    """Generate a deploy-time token and CloudFront Function for viewer-request auth."""
    auth_token_handler = lambda_.Function(
        scope,
        f"{construct_id}AuthTokenFn",
        runtime=lambda_.Runtime.PYTHON_3_12,
        handler="index.handler",
        timeout=Duration.seconds(30),
        code=lambda_.Code.from_inline("""
import secrets

def handler(event, context):
    req_type = event.get("RequestType", "")
    physical_id = event.get("PhysicalResourceId") or "RedactionAuthToken"
    if req_type == "Delete":
        return {"PhysicalResourceId": physical_id}
    token = secrets.token_hex(16)
    return {
        "PhysicalResourceId": physical_id,
        "Data": {"Token": token},
    }
"""),
    )
    provider = cr.Provider(
        scope,
        f"{construct_id}AuthTokenProvider",
        on_event_handler=auth_token_handler,
    )
    token_cr = CustomResource(
        scope,
        f"{construct_id}AuthToken",
        service_token=provider.service_token,
    )
    token = token_cr.get_att_string("Token")
    function_code = cloudfront.FunctionCode.from_inline(
        build_magic_link_viewer_request_js(
            token=token,
            cookie_name=cookie_name,
            cookie_max_age_sec=cookie_max_age_sec,
        )
    )
    auth_function = cloudfront.Function(
        scope,
        f"{construct_id}MagicLinkFunction",
        code=function_code,
        comment="Magic-link demo auth for doc_redaction",
    )
    auth_function.node.add_dependency(token_cr)
    return MagicLinkAuthResources(
        auth_function=auth_function,
        auth_token=token,
        token_custom_resource=token_cr,
    )


def magic_link_function_association(
    auth_function: cloudfront.Function,
) -> cloudfront.FunctionAssociation:
    return cloudfront.FunctionAssociation(
        function=auth_function,
        event_type=cloudfront.FunctionEventType.VIEWER_REQUEST,
    )


def create_forwarded_host_function(
    scope: Construct,
    construct_id: str,
) -> cloudfront.Function:
    """Viewer-request function that forwards the viewer host/proto to the origin.

    For CloudFront deployments without magic-link auth, so Gradio still builds
    asset URLs against the CloudFront domain (see ``_FORWARDED_HOST_INJECTION_JS``).
    """
    return cloudfront.Function(
        scope,
        f"{construct_id}ForwardedHostFunction",
        code=cloudfront.FunctionCode.from_inline(
            build_forwarded_host_viewer_request_js()
        ),
        comment="Forward viewer host/proto to origin for doc_redaction",
    )


def emit_magic_link_outputs(
    scope: Construct,
    *,
    distribution_domain_name: str,
    auth_token: str,
) -> None:
    """Stack outputs for demo unlock URL and normal app URL."""
    domain = distribution_domain_name
    CfnOutput(
        scope,
        "RedactionLoginUrl",
        value=f"https://{domain}/?key={auth_token}",
        description="Paste this URL into your browser to unlock the demo (7-day cookie)",
    )
    CfnOutput(
        scope,
        "RedactionAuthToken",
        value=auth_token,
        description="Magic-link token (the value after ?key= in RedactionLoginUrl)",
    )
    CfnOutput(
        scope,
        "RedactionUrl",
        value=f"https://{domain}/",
        description="Normal HTTPS URL — requires cookie from RedactionLoginUrl first",
    )
