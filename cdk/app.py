import os

from aws_cdk import App, Environment
from cdk_config import (
    AWS_ACCOUNT_ID,
    AWS_REGION,
    CDK_CONTEXT_FILE,
    RUN_USEAST_STACK,
    USE_CLOUDFRONT,
)
from cdk_functions import (
    create_basic_config_env,
    load_context_from_file,
    log_aws_credential_context,
    purge_cdk_lookup_context,
)
from cdk_stack import CdkStack, CdkStackCloudfront  # , CdkStackMain
from check_resources import CONTEXT_FILE, check_and_set_context

# Initialize the CDK app
app = App()

log_aws_credential_context(
    expected_account_id=AWS_ACCOUNT_ID,
    expected_region=AWS_REGION,
)

# Drop stale CDK lookup cache entries (require bootstrap lookup role in target account).
purge_cdk_lookup_context(CDK_CONTEXT_FILE)

# --- Pre-check context (boto3) — written to precheck.context.json, NOT cdk.context.json ---
print(f"Pre-check context file: {CONTEXT_FILE}")
print(f"CDK lookup cache file: {CDK_CONTEXT_FILE}")
if os.path.basename(CONTEXT_FILE.replace("\\", "/")) == os.path.basename(
    CDK_CONTEXT_FILE.replace("\\", "/")
):
    raise RuntimeError(
        f"CONTEXT_FILE and CDK_CONTEXT_FILE must differ (got '{CONTEXT_FILE}' for both). "
        "Set CONTEXT_FILE=precheck.context.json in config/cdk_config.env."
    )

print("Running pre-check script to generate application context...")
try:
    check_and_set_context()
    if not os.path.exists(CONTEXT_FILE):
        raise RuntimeError(
            f"check_and_set_context() finished, but {CONTEXT_FILE} was not created."
        )
    print(f"Context generated successfully at {CONTEXT_FILE}.")
except Exception as e:
    raise RuntimeError(f"Failed to generate context via check_and_set_context(): {e}")

# Pre-check must not repopulate CDK lookup keys; purge again if paths were ever shared.
purge_cdk_lookup_context(CDK_CONTEXT_FILE)

if os.path.exists(CONTEXT_FILE):
    load_context_from_file(app, CONTEXT_FILE)
else:
    raise RuntimeError(f"Could not find {CONTEXT_FILE}.")

create_basic_config_env("config")

aws_env_regional = Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION)

regional_stack = CdkStack(
    app, "RedactionStack", env=aws_env_regional, cross_region_references=True
)

if USE_CLOUDFRONT == "True" and RUN_USEAST_STACK == "True":
    aws_env_us_east_1 = Environment(account=AWS_ACCOUNT_ID, region="us-east-1")

    cloudfront_stack = CdkStackCloudfront(
        app,
        "RedactionStackCloudfront",
        env=aws_env_us_east_1,
        alb_arn=regional_stack.params["alb_arn_output"],
        alb_sec_group_id=regional_stack.params["alb_security_group_id"],
        alb_dns_name=regional_stack.params["alb_dns_name"],
        cross_region_references=True,
    )

# CDK CLI (deploy/synth/diff) calls app.synth() itself — do not call it here or
# the app (and check_and_set_context) runs twice per command.
