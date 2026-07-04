# CDK deployment (doc_redaction)

Interactive installer: [`cdk_install.py`](cdk_install.py). Configuration: [`config/cdk_config.env`](config/cdk_config.env).

Human-oriented walkthrough: [Installation Guide](https://seanpedrick-case.github.io/doc_redaction/src/installation_guide.html).

## CloudFront response headers (CSP / CORS)

When `USE_CLOUDFRONT=True` and `CLOUDFRONT_ENABLE_SECURE_RESPONSE_HEADERS=True`, the stack **creates** a CloudFront **response headers policy** (see [`cdk_cloudfront_headers.py`](cdk_cloudfront_headers.py) and [`config/response-headers-policy-config.json`](config/response-headers-policy-config.json)). The policy is created as a standalone resource so it appears in the account's CloudFront policy list and can be attached manually.

Attaching the policy to the distribution's behaviors is controlled separately by `CLOUDFRONT_ATTACH_SECURE_RESPONSE_HEADERS` (**default `False`**). Attaching the CSP/security headers tends to break demonstration mode (Cognito redirect flows and mixed Express origins), so by default the policy is created but **left detached**. Set `CLOUDFRONT_ATTACH_SECURE_RESPONSE_HEADERS=True` to wire it onto the distribution.

The policy’s **CORS origin** and **Content-Security-Policy** `connect-src` (Gradio WebSocket: `wss://…`) must match the browser-facing app URL.

### Why `cloudfront_placeholder.net` appears on first deploy

The real `*.cloudfront.net` domain is only assigned **after** the distribution is created. The response headers policy is a **separate** CloudFormation resource referenced by the distribution; it cannot reference `distribution.domain_name` without a **circular dependency** (distribution → policy → distribution).

On the first deploy, `CLOUDFRONT_DOMAIN` and `COGNITO_REDIRECTION_URL` therefore use a synth-time placeholder (`cloudfront_placeholder.net`). CSP/CORS are built from those values.

### Automatic fix (installer)

After the initial `cdk deploy`, [`cdk_install.py`](cdk_install.py):

1. Reads `CloudFrontDistributionURL` from stack outputs.
2. Writes the real domain to `CLOUDFRONT_DOMAIN` and `COGNITO_REDIRECTION_URL` in `cdk_config.env` (and updates Cognito callback URLs via API).
3. Prompts for a **one-off refresh deploy** of `RedactionStack` so the response headers policy and related config use the real domain.

Use `--yes` to accept the refresh deploy without prompting. If you skip it, run manually:

```bash
cd cdk
cdk deploy RedactionStack
```

### Single-pass alternative (custom domain)

If you set `SSL_CERTIFICATE_DOMAIN` and `ACM_SSL_CERTIFICATE_ARN` (production profile), the app origin is known at synth time and CSP/CORS are correct on the **first** deploy—no placeholder and no refresh deploy.

### Verify after deploy

In the AWS console: **CloudFront** → your distribution → **Behaviors** → response headers policy. CORS should list `https://<your-domain>` and CSP `connect-src` should include `wss://<your-domain>` (not `cloudfront_placeholder.net`).
