import os
import time

from cdk_config import (
    AWS_REGION,
    CDK_PREFIX,
    CLUSTER_NAME,
    CODEBUILD_PI_PROJECT_NAME,
    CODEBUILD_PROJECT_NAME,
    COGNITO_USER_POOL_CLIENT_SECRET_NAME,
    ECS_EXPRESS_COGNITO_REDIRECT_BASE,
    ECS_EXPRESS_SC_PORT_NAME,
    ECS_EXPRESS_SERVICE_NAME,
    ECS_LOG_GROUP_NAME,
    ECS_PI_EXPRESS_SERVICE_NAME,
    ECS_PI_SERVICE_NAME,
    ECS_SERVICE_CONNECT_DISCOVERY_NAME,
    ECS_SERVICE_CONNECT_NAMESPACE,
    ECS_SERVICE_NAME,
    ENABLE_HEADLESS_DEPLOYMENT,
    ENABLE_PI_AGENT_ECS_SERVICE,
    ENABLE_PI_AGENT_EXPRESS_SERVICE,
    GRADIO_SERVER_PORT,
    PI_AGENT_ENV_S3_KEY,
    S3_BATCH_ENV_PREFIX,
    S3_BATCH_INPUT_PREFIX,
    S3_BATCH_LAMBDA_FUNCTION_NAME,
    S3_LOG_CONFIG_BUCKET_NAME,
    S3_OUTPUT_BUCKET_NAME,
    USE_ECS_EXPRESS_MODE,
)
from cdk_functions import create_basic_config_env

# boto3-only module (does not import aws-cdk / Node.js)
from cdk_post_deploy import (
    apply_cognito_secret_fixup_from_stack,
    configure_express_pi_service_connect,
    print_headless_deployment_next_steps,
    seed_headless_batch_s3_layout,
    start_codebuild_build,
    start_ecs_task,
    start_express_gateway_service,
    upload_file_to_s3,
)
from tqdm import tqdm

# Create basic app_config.env file that user can use to run the app later. Input is the folder it is saved into.
create_basic_config_env(
    "config",
    headless=ENABLE_HEADLESS_DEPLOYMENT == "True",
    pi_express_backend=ENABLE_PI_AGENT_EXPRESS_SERVICE == "True",
)

# Start CodeBuild for the main app image
print("Starting main app CodeBuild project.")
start_codebuild_build(project_name=CODEBUILD_PROJECT_NAME)

_enable_pi_image_build = (
    ENABLE_PI_AGENT_ECS_SERVICE == "True" or ENABLE_PI_AGENT_EXPRESS_SERVICE == "True"
)
if _enable_pi_image_build:
    print("Starting Pi agent CodeBuild project.")
    start_codebuild_build(project_name=CODEBUILD_PI_PROJECT_NAME)

# Upload app_config.env file to S3 bucket
upload_file_to_s3(
    local_file_paths="config/app_config.env",
    s3_key="",
    s3_bucket=S3_LOG_CONFIG_BUCKET_NAME,
)

if _enable_pi_image_build:
    pi_env_local = os.path.join("config", "pi_agent.env")
    if os.path.isfile(pi_env_local):
        print(
            f"Uploading {pi_env_local} to s3://{S3_LOG_CONFIG_BUCKET_NAME}/{PI_AGENT_ENV_S3_KEY}"
        )
        upload_file_to_s3(
            local_file_paths=pi_env_local,
            s3_key="",
            s3_bucket=S3_LOG_CONFIG_BUCKET_NAME,
        )
    else:
        print(
            f"Skipping Pi env upload: {pi_env_local} not found. "
            f"Create it (from config/pi_agent.env.example) and upload to "
            f"s3://{S3_LOG_CONFIG_BUCKET_NAME}/{PI_AGENT_ENV_S3_KEY} before scaling the Pi service."
        )
    try:
        from cdk_config import ENABLE_AGENTCORE_RUNTIME

        if ENABLE_AGENTCORE_RUNTIME == "True":
            print(
                "\n--- AgentCore runtime checklist (required before agent UI works) ---"
            )
            print(
                "1. Package: python agent-redact/agentcore/package_runtime.py "
                "--target <AgentCoreProject>/app/RedactionAgent"
            )
            print("   Windows/OneDrive: set UV_LINK_MODE=copy before agentcore deploy")
            print("2. Deploy: cd <AgentCoreProject> && agentcore deploy")
            print(
                "3. URL: copy invocationUrl from `agentcore status` "
                "(base only, no /invocations suffix)"
            )
            print(
                "4. Wire: set AGENTCORE_RUNTIME_URL in config/pi_agent.env and "
                "config/cdk_config.env, upload pi_agent.env to S3, restart Pi Express"
            )
            print(
                "   DOC_REDACTION_GRADIO_URL: main Express HTTPS (ExpressServiceEndpoint "
                "or PiDocRedactionBackendUrl stack output) — set automatically on Pi "
                "Express when ENABLE_AGENTCORE_RUNTIME=True"
            )
            print(
                "   Or re-run: python cdk/cdk_install.py --config-only "
                "--agentcore-runtime-url <URL>"
            )
            print("See agent-redact/agentcore/README.md for full steps.\n")
    except ImportError:
        pass

total_seconds = 480  # 8 minutes
update_interval = 1  # Update every second

print("Waiting 8 minutes for CodeBuild container image(s) to build.")

# tqdm iterates over a range, and you perform a small sleep in each iteration
for i in tqdm(range(total_seconds), desc="Building container"):
    time.sleep(update_interval)

# Scale main ECS service to one task (skipped for headless batch-only deployments)
if ENABLE_HEADLESS_DEPLOYMENT != "True":
    if USE_ECS_EXPRESS_MODE == "True":
        print(f"Starting Express ECS service {ECS_EXPRESS_SERVICE_NAME}")
        start_express_gateway_service(
            cluster_name=CLUSTER_NAME, service_name=ECS_EXPRESS_SERVICE_NAME
        )
        if ENABLE_PI_AGENT_EXPRESS_SERVICE == "True":
            print(
                "Configuring Service Connect for main Express (server) and "
                "Pi Express (client)."
            )
            try:
                configure_express_pi_service_connect(
                    cluster_name=CLUSTER_NAME,
                    main_service_name=ECS_EXPRESS_SERVICE_NAME,
                    pi_service_name=ECS_PI_EXPRESS_SERVICE_NAME,
                    namespace=ECS_SERVICE_CONNECT_NAMESPACE,
                    main_port_name=ECS_EXPRESS_SC_PORT_NAME,
                    discovery_name=ECS_SERVICE_CONNECT_DISCOVERY_NAME,
                    main_port=int(GRADIO_SERVER_PORT),
                )
            except Exception as exc:
                print("Warning: could not configure Express Service Connect: " f"{exc}")
        try:
            from cdk_config import ENABLE_AGENTCORE_RUNTIME
            from cdk_post_deploy import sync_pi_agent_doc_redaction_url_for_agentcore

            if ENABLE_AGENTCORE_RUNTIME == "True":
                sync_pi_agent_doc_redaction_url_for_agentcore(
                    stack_name="RedactionStack",
                    region=AWS_REGION,
                )
        except ImportError:
            pass
        print("Syncing Cognito app client secret for in-app authentication.")
        apply_cognito_secret_fixup_from_stack(
            stack_name="RedactionStack",
            secret_name=COGNITO_USER_POOL_CLIENT_SECRET_NAME,
            cluster_name=CLUSTER_NAME,
            main_service_name=ECS_EXPRESS_SERVICE_NAME,
            recycle_tasks=False,
        )
    else:
        print(f"Starting ECS service {ECS_SERVICE_NAME}")
        start_ecs_task(cluster_name=CLUSTER_NAME, service_name=ECS_SERVICE_NAME)
else:
    print(
        "Headless deployment: skipping always-on ECS service start "
        "(tasks are started by the S3 batch Lambda)."
    )
    seed_headless_batch_s3_layout(
        S3_OUTPUT_BUCKET_NAME,
        input_prefix=S3_BATCH_INPUT_PREFIX,
        env_prefix=S3_BATCH_ENV_PREFIX,
        example_env_local_path=os.path.join("config", "example_headless_env_file.env"),
    )
    print_headless_deployment_next_steps(
        {
            "AWS_REGION": AWS_REGION,
            "S3_OUTPUT_BUCKET_NAME": S3_OUTPUT_BUCKET_NAME,
            "S3_BATCH_INPUT_PREFIX": S3_BATCH_INPUT_PREFIX,
            "S3_BATCH_ENV_PREFIX": S3_BATCH_ENV_PREFIX,
            "ECS_LOG_GROUP_NAME": ECS_LOG_GROUP_NAME,
            "S3_BATCH_LAMBDA_FUNCTION_NAME": S3_BATCH_LAMBDA_FUNCTION_NAME,
            "CDK_PREFIX": CDK_PREFIX,
        }
    )

if ENABLE_PI_AGENT_ECS_SERVICE == "True":
    print(f"Starting Pi agent ECS service {ECS_PI_SERVICE_NAME}")
    start_ecs_task(cluster_name=CLUSTER_NAME, service_name=ECS_PI_SERVICE_NAME)

if ENABLE_PI_AGENT_EXPRESS_SERVICE == "True":
    print(f"Starting Pi Express ECS service {ECS_PI_EXPRESS_SERVICE_NAME}")
    start_express_gateway_service(
        cluster_name=CLUSTER_NAME, service_name=ECS_PI_EXPRESS_SERVICE_NAME
    )

if USE_ECS_EXPRESS_MODE == "True" and ENABLE_HEADLESS_DEPLOYMENT != "True":
    from cdk_post_deploy import print_express_mode_next_steps

    print_express_mode_next_steps(
        {
            "AWS_REGION": AWS_REGION,
            "ECS_EXPRESS_COGNITO_REDIRECT_BASE": ECS_EXPRESS_COGNITO_REDIRECT_BASE,
            "ENABLE_PI_AGENT_EXPRESS_SERVICE": ENABLE_PI_AGENT_EXPRESS_SERVICE,
        }
    )
