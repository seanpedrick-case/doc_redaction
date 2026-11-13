# Minimal AgentCore Harness stack (local `app.py`)

Use this when you want **only** the Bedrock AgentCore Harness and its supporting AWS resources—not ECS, VPC, Cognito, or the production redaction stack. Your machine runs [`app.py`](../../app.py); AWS runs the agent loop; inline tools execute in your local Python process.

## What gets created

| Resource | Purpose |
|----------|---------|
| `AWS::BedrockAgentCore::Harness` | Agent loop, Bedrock model, inline tool **names** (implementations are local), S3 skill |
| Skills S3 bucket | `doc-redaction-harness` skill for the harness execution role |
| Staging S3 bucket | Presigned URLs for optional parallel page review (`stage_page_review_bundles`) |
| IAM managed policy | `InvokeHarness` + staging-bucket S3 for **your** developer principal |

Bucket names default to `doc-redact-harness-skills-<region>-<account>` and `doc-redact-harness-stage-<region>-<account>` (override in `config/harness_local.env`).

## Prerequisites

- AWS CDK CLI and Python deps from [`cdk/requirements.txt`](../requirements.txt)
- `cdk bootstrap aws://ACCOUNT/REGION` once per account/region
- Bedrock access to the chosen inference profile (e.g. `eu.anthropic.claude-sonnet-4-6`)
- Local app: `boto3>=1.44.3`, Tesseract/Poppler as for normal redaction

## Deploy

From the repository root (recommended):

```bash
python cdk/deploy_harness_local.py --profile YOUR_PROFILE --yes
```

Optional: merge stack outputs into `config/app_config.env`:

```bash
python cdk/deploy_harness_local.py --profile YOUR_PROFILE --yes --write-app-config
```

Manual CDK (from `cdk/`):

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=eu-west-2
cdk deploy DocRedactionHarnessLocal -a "python app_harness_local.py"
```

Copy `cdk.json.harness_local.example` to `cdk.json` if you want a dedicated output directory (`cdk.out.harness_local`) without touching the main app's `cdk.json`.

## After deploy

1. **IAM** — In the IAM console, attach managed policy `DocRedactionHarnessLocalInvoker-<region>` to the user or role your AWS CLI / SSO session uses locally (output `LocalInvokerPolicyName`).
2. **App config** — Set (or use `--write-app-config`):
   - `USE_AGENTCORE=True`
   - `AGENTCORE_HARNESS_ARN` (output `RedactionHarnessArn`)
   - `AGENTCORE_REGION`
   - `AGENTCORE_HARNESS_S3_STAGING_PREFIX` (output `HarnessStagingPrefix`)
   - `AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW=True` (or `False` to skip subagent page review)
3. Run the app and use the **AI agent redaction instructions** box on the Redact tab.

## Configuration

Copy [`config/harness_local.env.example`](../config/harness_local.env.example) to `cdk/config/harness_local.env` to override harness name, model id, iteration limits, bucket names, or lifecycle days on the staging bucket.

## Teardown

```bash
python cdk/deploy_harness_local.py --profile YOUR_PROFILE --destroy --yes
```

Skills and staging buckets use `RemovalPolicy.DESTROY` and `auto_delete_objects` for dev convenience.

## Related

- Main-app harness design: [agent-redact/agentcore/OPTIONS.md](../../agent-redact/agentcore/OPTIONS.md) (Option 3)
- Full production harness (same harness helper, inside `RedactionStack`): `ENABLE_AGENTCORE_HARNESS` in `cdk_install.py`
