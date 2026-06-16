import os

from aws_cdk import App, Environment
from cdk_appregistry import register_doc_redaction_application
from cdk_config import (
    ALB_NAME,
    APPREGISTRY_APPLICATION_NAME,
    APPREGISTRY_ATTRIBUTE_GROUP_NAME,
    APPREGISTRY_DESCRIPTION,
    APPREGISTRY_REPOSITORY_URL,
    APPREGISTRY_STACK_NAME,
    AWS_ACCOUNT_ID,
    AWS_REGION,
    CDK_CONTEXT_FILE,
    CDK_PREFIX,
    ENABLE_APPREGISTRY,
    RUN_USEAST_STACK,
    USE_CLOUDFRONT,
)
from cdk_functions import (
    create_basic_config_env,
    is_resource_delete_protection_enabled,
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

_stack_delete_protection = is_resource_delete_protection_enabled()

regional_stack = CdkStack(
    app, "RedactionStack", env=aws_env_regional, cross_region_references=True
)
regional_stack.termination_protection = _stack_delete_protection

if ENABLE_APPREGISTRY == "True":
    # Use pre-check context only — not regional_stack.params (avoids AppRegistry
    # -> RedactionStack dependency cycle during synth).
    _alb_dns_context = app.node.try_get_context(f"dns:{ALB_NAME}")
    _alb_dns_name = (
        _alb_dns_context.strip()
        if isinstance(_alb_dns_context, str) and _alb_dns_context.strip()
        else None
    )
    appregistry_stack = register_doc_redaction_application(
        app,
        aws_account_id=AWS_ACCOUNT_ID,
        aws_region=AWS_REGION,
        application_name=APPREGISTRY_APPLICATION_NAME,
        application_description=APPREGISTRY_DESCRIPTION,
        appregistry_stack_name=APPREGISTRY_STACK_NAME,
        attribute_group_name=APPREGISTRY_ATTRIBUTE_GROUP_NAME,
        repository_url=APPREGISTRY_REPOSITORY_URL,
        cdk_prefix=CDK_PREFIX,
        use_cloudfront=USE_CLOUDFRONT,
        alb_dns_name=_alb_dns_name,
    )
    appregistry_stack.termination_protection = _stack_delete_protection

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
    cloudfront_stack.termination_protection = _stack_delete_protection

# CDK CLI invokes this script and expects a cloud assembly in cdk.out.
# Without app.synth(), Python defines constructs but never writes manifest.json
# (ENOENT on deploy). See: https://github.com/aws/aws-cdk/issues/11023
app.synth()
