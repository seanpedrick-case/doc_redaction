import os

from aws_cdk import App, Environment
from cdk_config import AWS_ACCOUNT_ID, AWS_REGION, RUN_USEAST_STACK, USE_CLOUDFRONT
from cdk_functions import create_basic_config_env, load_context_from_file
from cdk_stack import CdkStack, CdkStackCloudfront  # , CdkStackMain

# Assuming these are still relevant for you
from check_resources import CONTEXT_FILE, check_and_set_context

# Initialize the CDK app
app = App()

# --- ENHANCED CONTEXT GENERATION AND LOADING ---
# 1. Always ensure the old context file is removed before generation
if os.path.exists(CONTEXT_FILE):
    try:
        os.remove(CONTEXT_FILE)
        print(f"Removed stale context file: {CONTEXT_FILE}")
    except OSError as e:
        print(f"Warning: Could not remove old context file {CONTEXT_FILE}: {e}")
        # Proceed anyway, check_and_set_context might handle overwriting

# 2. Always run the pre-check script to generate fresh context
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

if os.path.exists(CONTEXT_FILE):
    load_context_from_file(app, CONTEXT_FILE)
else:
    raise RuntimeError(f"Could not find {CONTEXT_FILE}.")

# Create basic config.env file that user can use to run the app later. Input is the folder it is saved into.
create_basic_config_env("config")

# Define the environment for the regional stack (where ALB resides)
aws_env_regional = Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION)

# Create the regional stack (ALB, SGs, etc.)
# regional_stack = CdkStack(app,
#                           "RedactionStackSubnets",
#                           env=aws_env_regional,
#                           cross_region_references=True)

# regional_stack_main = CdkStackMain(app,
#                         "RedactionStackMain",
#                         env=aws_env_regional,
#                         private_subnets=regional_stack.params["private_subnets"],
#                         private_route_tables=regional_stack.params["private_route_tables"],
#                         public_subnets=regional_stack.params["public_subnets"],
#                         public_route_tables=regional_stack.params["public_route_tables"],
#                         cross_region_references=True)

regional_stack = CdkStack(
    app, "RedactionStack", env=aws_env_regional, cross_region_references=True
)

if USE_CLOUDFRONT == "True" and RUN_USEAST_STACK == "True":
    # Define the environment for the CloudFront stack (always us-east-1 for CF-level resources like WAFv2 WebACLs for CF)
    aws_env_us_east_1 = Environment(account=AWS_ACCOUNT_ID, region="us-east-1")

    # Create the CloudFront stack, passing the outputs from the regional stack
    cloudfront_stack = CdkStackCloudfront(
        app,
        "RedactionStackCloudfront",
        env=aws_env_us_east_1,
        alb_arn=regional_stack.params["alb_arn_output"],
        alb_sec_group_id=regional_stack.params["alb_security_group_id"],
        alb_dns_name=regional_stack.params["alb_dns_name"],
        cross_region_references=True,
    )


# Synthesize the CloudFormation template
app.synth(validate_on_synthesis=True)
