import os
import tempfile
from typing import List

from dotenv import load_dotenv

# Set or retrieve configuration variables for CDK redaction deployment


def convert_string_to_boolean(value: str) -> bool:
    """Convert string to boolean, handling various formats."""
    if isinstance(value, bool):
        return value
    elif value in ["True", "1", "true", "TRUE"]:
        return True
    elif value in ["False", "0", "false", "FALSE"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def parse_comma_separated_list(value: str) -> List[str]:
    """Parse a comma-separated env value into a list of non-empty strings."""
    if not value or not str(value).strip():
        return []
    cleaned = str(value).strip().strip("[]")
    return [
        part.strip().strip('"').strip("'")
        for part in cleaned.split(",")
        if part.strip()
    ]


def get_or_create_env_var(var_name: str, default_value: str, print_val: bool = False):
    """
    Get an environmental variable, and set it to a default value if it doesn't exist
    """
    # Get the environment variable if it exists
    value = os.environ.get(var_name)

    # If it doesn't exist, set the environment variable to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    if print_val is True:
        print(f"The value of {var_name} is {value}")

    return value


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


###
# LOAD CONFIG FROM ENV FILE
###
CONFIG_FOLDER = get_or_create_env_var("CONFIG_FOLDER", "config/")

ensure_folder_exists(CONFIG_FOLDER)

# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. 'config/cdk_config.env'
CDK_CONFIG_PATH = get_or_create_env_var(
    "CDK_CONFIG_PATH", "config/cdk_config.env"
)  # e.g. config/cdk_config.env

if CDK_CONFIG_PATH:
    if os.path.exists(CDK_CONFIG_PATH):
        print(f"Loading CDK variables from config file {CDK_CONFIG_PATH}")
        load_dotenv(CDK_CONFIG_PATH)
    else:
        print("CDK config file not found at location:", CDK_CONFIG_PATH)

###
# AWS OPTIONS
###
AWS_REGION = get_or_create_env_var("AWS_REGION", "")
AWS_ACCOUNT_ID = get_or_create_env_var("AWS_ACCOUNT_ID", "")

###
# CDK OPTIONS
###
CDK_PREFIX = get_or_create_env_var("CDK_PREFIX", "")

# When True (default): CloudFormation stack termination protection, AWS
# deletion_protection on ALB/DynamoDB/Cognito, RemovalPolicy.RETAIN, and no S3
# auto-delete on stack-created resources. Set False for dev sandboxes where
# cdk destroy should remove resources cleanly.
ENABLE_RESOURCE_DELETE_PROTECTION = get_or_create_env_var(
    "ENABLE_RESOURCE_DELETE_PROTECTION", "True"
)

# AWS Console myApplications (Service Catalog AppRegistry)
ENABLE_APPREGISTRY = get_or_create_env_var("ENABLE_APPREGISTRY", "True")
APPREGISTRY_APPLICATION_NAME = get_or_create_env_var(
    "APPREGISTRY_APPLICATION_NAME", f"{CDK_PREFIX}doc-redaction"
)
APPREGISTRY_DESCRIPTION = get_or_create_env_var(
    "APPREGISTRY_DESCRIPTION",
    "PII document redaction app (ALB, ECS Fargate, Cognito, S3)",
)
APPREGISTRY_STACK_NAME = get_or_create_env_var(
    "APPREGISTRY_STACK_NAME", f"{CDK_PREFIX}AppRegistryStack"
)
APPREGISTRY_ATTRIBUTE_GROUP_NAME = get_or_create_env_var(
    "APPREGISTRY_ATTRIBUTE_GROUP_NAME",
    f"{APPREGISTRY_APPLICATION_NAME}-metadata",
)
APPREGISTRY_REPOSITORY_URL = get_or_create_env_var(
    "APPREGISTRY_REPOSITORY_URL",
    "https://github.com/seanpedrick-case/doc_redaction",
)

_precheck_context_file = get_or_create_env_var("CONTEXT_FILE", "precheck.context.json")
# Never write boto3 pre-check output into CDK's lookup cache file (causes stale
# vpc-provider / load-balancer entries and wrong-account lookup validation errors).
if os.path.basename(_precheck_context_file.replace("\\", "/")) == "cdk.context.json":
    print(
        "WARNING: CONTEXT_FILE must not be 'cdk.context.json' (that file is CDK's "
        "lookup cache). Using 'precheck.context.json' instead. Update "
        "config/cdk_config.env and remove CONTEXT_FILE=cdk.context.json if set."
    )
    _precheck_context_file = "precheck.context.json"
CONTEXT_FILE = _precheck_context_file
CDK_CONTEXT_FILE = get_or_create_env_var("CDK_CONTEXT_FILE", "cdk.context.json")
CDK_FOLDER = get_or_create_env_var(
    "CDK_FOLDER", ""
)  # FULL_PATH_TO_CDK_FOLDER_HERE (with forward slash)

# App runtime config (uploaded to S3 for legacy Fargate; inlined for ECS Express Mode)
_app_config_rel = os.path.join(CONFIG_FOLDER, "config.env").replace("\\", "/")
APP_CONFIG_ENV_FILE = get_or_create_env_var(
    "APP_CONFIG_ENV_FILE",
    (
        os.path.normpath(os.path.join(CDK_FOLDER, _app_config_rel))
        if CDK_FOLDER
        else os.path.normpath(_app_config_rel)
    ),
)
RUN_USEAST_STACK = get_or_create_env_var("RUN_USEAST_STACK", "False")

### VPC and connections
VPC_NAME = get_or_create_env_var("VPC_NAME", "")
NEW_VPC_DEFAULT_NAME = get_or_create_env_var("NEW_VPC_DEFAULT_NAME", f"{CDK_PREFIX}vpc")
NEW_VPC_CIDR = get_or_create_env_var("NEW_VPC_CIDR", "")  # "10.0.0.0/24"


EXISTING_IGW_ID = get_or_create_env_var("EXISTING_IGW_ID", "")
SINGLE_NAT_GATEWAY_ID = get_or_create_env_var("SINGLE_NAT_GATEWAY_ID", "")

### SUBNETS / ROUTE TABLES / NAT GATEWAY
PUBLIC_SUBNETS_TO_USE = get_or_create_env_var(
    "PUBLIC_SUBNETS_TO_USE", ""
)  # e.g. ['PublicSubnet1', 'PublicSubnet2']
PUBLIC_SUBNET_CIDR_BLOCKS = get_or_create_env_var(
    "PUBLIC_SUBNET_CIDR_BLOCKS", ""
)  # e.g. ["10.0.1.0/24", "10.0.2.0/24"]
PUBLIC_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var(
    "PUBLIC_SUBNET_AVAILABILITY_ZONES", ""
)  # e.g. ["eu-east-1b", "eu-east1b"]

PRIVATE_SUBNETS_TO_USE = get_or_create_env_var(
    "PRIVATE_SUBNETS_TO_USE", ""
)  # e.g. ['PrivateSubnet1', 'PrivateSubnet2']
PRIVATE_SUBNET_CIDR_BLOCKS = get_or_create_env_var(
    "PRIVATE_SUBNET_CIDR_BLOCKS", ""
)  # e.g. ["10.0.1.0/24", "10.0.2.0/24"]
PRIVATE_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var(
    "PRIVATE_SUBNET_AVAILABILITY_ZONES", ""
)  # e.g. ["eu-east-1b", "eu-east1b"]

ROUTE_TABLE_BASE_NAME = get_or_create_env_var(
    "ROUTE_TABLE_BASE_NAME", f"{CDK_PREFIX}PrivateRouteTable"
)
NAT_GATEWAY_EIP_NAME = get_or_create_env_var(
    "NAT_GATEWAY_EIP_NAME", f"{CDK_PREFIX}NatGatewayEip"
)
NAT_GATEWAY_NAME = get_or_create_env_var("NAT_GATEWAY_NAME", f"{CDK_PREFIX}NatGateway")

# IAM roles
AWS_MANAGED_TASK_ROLES_LIST = get_or_create_env_var(
    "AWS_MANAGED_TASK_ROLES_LIST",
    '["AmazonCognitoReadOnly", "service-role/AmazonECSTaskExecutionRolePolicy", "AmazonS3FullAccess", "AmazonTextractFullAccess", "ComprehendReadOnly", "AmazonDynamoDBFullAccess", "service-role/AWSAppSyncPushToCloudWatchLogs", "AmazonBedrockFullAccess"]',
)
POLICY_FILE_LOCATIONS = get_or_create_env_var(
    "POLICY_FILE_LOCATIONS", ""
)  # e.g. '["config/sts_permissions.json"]'
POLICY_FILE_ARNS = get_or_create_env_var("POLICY_FILE_ARNS", "")

# GITHUB REPO
GITHUB_REPO_USERNAME = get_or_create_env_var("GITHUB_REPO_USERNAME", "seanpedrick-case")
GITHUB_REPO_NAME = get_or_create_env_var("GITHUB_REPO_NAME", "doc_redaction")
GITHUB_REPO_BRANCH = get_or_create_env_var("GITHUB_REPO_BRANCH", "main")

### CODEBUILD
CODEBUILD_ROLE_NAME = get_or_create_env_var(
    "CODEBUILD_ROLE_NAME", f"{CDK_PREFIX}CodeBuildRole"
)
CODEBUILD_PROJECT_NAME = get_or_create_env_var(
    "CODEBUILD_PROJECT_NAME", f"{CDK_PREFIX}CodeBuildProject"
)

### ECR
ECR_REPO_NAME = get_or_create_env_var(
    "ECR_REPO_NAME", "doc-redaction"
)  # Beware - cannot have underscores and must be lower case
ECR_CDK_REPO_NAME = get_or_create_env_var(
    "ECR_CDK_REPO_NAME", f"{CDK_PREFIX}{ECR_REPO_NAME}".lower()
)

### S3
S3_LOG_CONFIG_BUCKET_NAME = get_or_create_env_var(
    "S3_LOG_CONFIG_BUCKET_NAME", f"{CDK_PREFIX}s3-logs".lower()
)  # S3 bucket names need to be lower case
S3_OUTPUT_BUCKET_NAME = get_or_create_env_var(
    "S3_OUTPUT_BUCKET_NAME", f"{CDK_PREFIX}s3-output".lower()
)

### KMS KEYS FOR S3 AND SECRETS MANAGER
USE_CUSTOM_KMS_KEY = get_or_create_env_var("USE_CUSTOM_KMS_KEY", "1")
CUSTOM_KMS_KEY_NAME = get_or_create_env_var(
    "CUSTOM_KMS_KEY_NAME", f"alias/{CDK_PREFIX}kms-key".lower()
)

### ECS
FARGATE_TASK_DEFINITION_NAME = get_or_create_env_var(
    "FARGATE_TASK_DEFINITION_NAME", f"{CDK_PREFIX}FargateTaskDefinition"
)
TASK_DEFINITION_FILE_LOCATION = get_or_create_env_var(
    "TASK_DEFINITION_FILE_LOCATION", CDK_FOLDER + CONFIG_FOLDER + "task_definition.json"
)

CLUSTER_NAME = get_or_create_env_var("CLUSTER_NAME", f"{CDK_PREFIX}Cluster")
ECS_SERVICE_NAME = get_or_create_env_var("ECS_SERVICE_NAME", f"{CDK_PREFIX}ECSService")
# Second Fargate service when ENABLE_PI_AGENT_ECS_SERVICE=True (legacy path only).
ECS_PI_SERVICE_NAME = get_or_create_env_var(
    "ECS_PI_SERVICE_NAME", f"{CDK_PREFIX}PiAgentService"
)
ECS_TASK_ROLE_NAME = get_or_create_env_var(
    "ECS_TASK_ROLE_NAME", f"{CDK_PREFIX}TaskRole"
)
ECS_TASK_EXECUTION_ROLE_NAME = get_or_create_env_var(
    "ECS_TASK_EXECUTION_ROLE_NAME", f"{CDK_PREFIX}ExecutionRole"
)
ECS_SECURITY_GROUP_NAME = get_or_create_env_var(
    "ECS_SECURITY_GROUP_NAME", f"{CDK_PREFIX}SecurityGroupECS"
)
ECS_LOG_GROUP_NAME = get_or_create_env_var(
    "ECS_LOG_GROUP_NAME", f"/ecs/{ECS_SERVICE_NAME}-logs".lower()
)

ECS_TASK_CPU_SIZE = get_or_create_env_var("ECS_TASK_CPU_SIZE", "1024")
ECS_TASK_MEMORY_SIZE = get_or_create_env_var("ECS_TASK_MEMORY_SIZE", "4096")
ECS_USE_FARGATE_SPOT = get_or_create_env_var("USE_FARGATE_SPOT", "False")
ECS_READ_ONLY_FILE_SYSTEM = get_or_create_env_var("ECS_READ_ONLY_FILE_SYSTEM", "True")

### Cognito
COGNITO_USER_POOL_NAME = get_or_create_env_var(
    "COGNITO_USER_POOL_NAME", f"{CDK_PREFIX}UserPool"
)
COGNITO_USER_POOL_CLIENT_NAME = get_or_create_env_var(
    "COGNITO_USER_POOL_CLIENT_NAME", f"{CDK_PREFIX}UserPoolClient"
)
COGNITO_USER_POOL_CLIENT_SECRET_NAME = get_or_create_env_var(
    "COGNITO_USER_POOL_CLIENT_SECRET_NAME", f"{CDK_PREFIX}ParamCognitoSecret"
)
COGNITO_USER_POOL_DOMAIN_PREFIX = get_or_create_env_var(
    "COGNITO_USER_POOL_DOMAIN_PREFIX", "redaction-app-domain"
)  # Should change this to something unique or you'll probably hit an error

COGNITO_REFRESH_TOKEN_VALIDITY = int(
    get_or_create_env_var("COGNITO_REFRESH_TOKEN_VALIDITY", "480")
)  # Minutes
COGNITO_ID_TOKEN_VALIDITY = int(
    get_or_create_env_var("COGNITO_ID_TOKEN_VALIDITY", "60")
)  # Minutes
COGNITO_ACCESS_TOKEN_VALIDITY = int(
    get_or_create_env_var("COGNITO_ACCESS_TOKEN_VALIDITY", "60")
)  # Minutes

# Application load balancer
ALB_NAME = get_or_create_env_var(
    "ALB_NAME", f"{CDK_PREFIX}Alb"[-32:]
)  # Application load balancer name can be max 32 characters, so taking the last 32 characters of the suggested name
ALB_NAME_SECURITY_GROUP_NAME = get_or_create_env_var(
    "ALB_SECURITY_GROUP_NAME", f"{CDK_PREFIX}SecurityGroupALB"
)
ALB_TARGET_GROUP_NAME = get_or_create_env_var(
    "ALB_TARGET_GROUP_NAME", f"{CDK_PREFIX}-tg"[-32:]
)  # Max 32 characters
EXISTING_LOAD_BALANCER_ARN = get_or_create_env_var("EXISTING_LOAD_BALANCER_ARN", "")
EXISTING_LOAD_BALANCER_DNS = get_or_create_env_var(
    "EXISTING_LOAD_BALANCER_DNS", "placeholder_load_balancer_dns.net"
)

## CLOUDFRONT
USE_CLOUDFRONT = get_or_create_env_var("USE_CLOUDFRONT", "True")
CLOUDFRONT_PREFIX_LIST_ID = get_or_create_env_var(
    "CLOUDFRONT_PREFIX_LIST_ID", "pl-93a247fa"
)
CLOUDFRONT_GEO_RESTRICTION = get_or_create_env_var(
    "CLOUDFRONT_GEO_RESTRICTION", ""
)  # A country that Cloudfront restricts access to. See here: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/georestrictions.html
CLOUDFRONT_DISTRIBUTION_NAME = get_or_create_env_var(
    "CLOUDFRONT_DISTRIBUTION_NAME", f"{CDK_PREFIX}CfDist"
)
CLOUDFRONT_DOMAIN = get_or_create_env_var(
    "CLOUDFRONT_DOMAIN", "cloudfront_placeholder.net"
)


# Certificate for Application load balancer (optional, for HTTPS and logins through the ALB)
ACM_SSL_CERTIFICATE_ARN = get_or_create_env_var("ACM_SSL_CERTIFICATE_ARN", "")
SSL_CERTIFICATE_DOMAIN = get_or_create_env_var(
    "SSL_CERTIFICATE_DOMAIN", ""
)  # e.g. example.com or www.example.com

# ECS Express Mode (opt-in HTTPS ingress without supplying ACM_SSL_CERTIFICATE_ARN).
# Pilot/dev: Express PrimaryContainer does not support S3 environmentFiles or Fargate mount points.
USE_ECS_EXPRESS_MODE = get_or_create_env_var("USE_ECS_EXPRESS_MODE", "False")
ECS_EXPRESS_SERVICE_NAME = get_or_create_env_var(
    "ECS_EXPRESS_SERVICE_NAME", ECS_SERVICE_NAME
)
ECS_EXPRESS_HEALTH_CHECK_PATH = get_or_create_env_var(
    "ECS_EXPRESS_HEALTH_CHECK_PATH", "/"
)
ECS_EXPRESS_INFRASTRUCTURE_ROLE_NAME = get_or_create_env_var(
    "ECS_EXPRESS_INFRASTRUCTURE_ROLE_NAME", f"{CDK_PREFIX}ExpressInfraRole"
)
# After first deploy, set to ExpressServiceEndpoint output (https://...) if not using CloudFront.
ECS_EXPRESS_COGNITO_REDIRECT_BASE = get_or_create_env_var(
    "ECS_EXPRESS_COGNITO_REDIRECT_BASE", ""
)

if USE_ECS_EXPRESS_MODE == "True" and ACM_SSL_CERTIFICATE_ARN:
    raise ValueError(
        "USE_ECS_EXPRESS_MODE=True cannot be used with ACM_SSL_CERTIFICATE_ARN set. "
        "Clear ACM_SSL_CERTIFICATE_ARN or set USE_ECS_EXPRESS_MODE=False."
    )

# ECS Service Connect (legacy Fargate only): VPC service-to-service HTTP to Gradio/FastAPI.
ENABLE_ECS_SERVICE_CONNECT = get_or_create_env_var(
    "ENABLE_ECS_SERVICE_CONNECT", "False"
)
ECS_SERVICE_CONNECT_NAMESPACE = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_NAMESPACE",
    (f"{CDK_PREFIX}local".lower().replace("_", "-").strip("-") or "redaction-local"),
)
ECS_SERVICE_CONNECT_DISCOVERY_NAME = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_DISCOVERY_NAME", "redaction"
)
# Optional friendly DNS label; defaults to discovery name when empty.
ECS_SERVICE_CONNECT_DNS_NAME = get_or_create_env_var("ECS_SERVICE_CONNECT_DNS_NAME", "")
# Client task security groups (at least one of IDs, names, or CDK prefixes required when SC on).
ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS", ""
)
ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS_LIST = parse_comma_separated_list(
    ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS
)
ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES", ""
)
ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES", ""
)

# This should be the CloudFront domain, the domain linked to your ACM certificate, or the DNS of your application load balancer in console afterwards
if USE_CLOUDFRONT == "True":
    COGNITO_REDIRECTION_URL = get_or_create_env_var(
        "COGNITO_REDIRECTION_URL", "https://" + CLOUDFRONT_DOMAIN
    )
elif SSL_CERTIFICATE_DOMAIN:
    COGNITO_REDIRECTION_URL = get_or_create_env_var(
        "COGNITO_REDIRECTION_URL", "https://" + SSL_CERTIFICATE_DOMAIN
    )
elif USE_ECS_EXPRESS_MODE == "True":
    _express_redirect_default = ECS_EXPRESS_COGNITO_REDIRECT_BASE or (
        "https://" + EXISTING_LOAD_BALANCER_DNS
    )
    COGNITO_REDIRECTION_URL = get_or_create_env_var(
        "COGNITO_REDIRECTION_URL", _express_redirect_default
    )
else:
    COGNITO_REDIRECTION_URL = get_or_create_env_var(
        "COGNITO_REDIRECTION_URL", "https://" + EXISTING_LOAD_BALANCER_DNS
    )

# Custom headers e.g. if routing traffic through Cloudfront
CUSTOM_HEADER = get_or_create_env_var(
    "CUSTOM_HEADER", ""
)  # Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER_VALUE = get_or_create_env_var(
    "CUSTOM_HEADER_VALUE", ""
)  # Retrieving or setting CUSTOM_HEADER_VALUE

# Firewall on top of load balancer
LOAD_BALANCER_WEB_ACL_NAME = get_or_create_env_var(
    "LOAD_BALANCER_WEB_ACL_NAME", f"{CDK_PREFIX}alb-web-acl"
)

# Firewall on top of CloudFront
WEB_ACL_NAME = get_or_create_env_var("WEB_ACL_NAME", f"{CDK_PREFIX}cloudfront-web-acl")

###
# File I/O options
###

OUTPUT_FOLDER = get_or_create_env_var("GRADIO_OUTPUT_FOLDER", "output/")  # 'output/'
INPUT_FOLDER = get_or_create_env_var("GRADIO_INPUT_FOLDER", "input/")  # 'input/'

# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP":
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")

        if OUTPUT_FOLDER == "TEMP":
            OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP":
            INPUT_FOLDER = temp_dir + "/"

###
# LOGGING OPTIONS
###

SAVE_LOGS_TO_CSV = get_or_create_env_var("SAVE_LOGS_TO_CSV", "True")

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table
SAVE_LOGS_TO_DYNAMODB = get_or_create_env_var("SAVE_LOGS_TO_DYNAMODB", "True")
ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "ACCESS_LOG_DYNAMODB_TABLE_NAME", f"{CDK_PREFIX}dynamodb-access-logs".lower()
)
FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "FEEDBACK_LOG_DYNAMODB_TABLE_NAME", f"{CDK_PREFIX}dynamodb-feedback-logs".lower()
)
USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "USAGE_LOG_DYNAMODB_TABLE_NAME", f"{CDK_PREFIX}dynamodb-usage-logs".lower()
)

###
# REDACTION OPTIONS
###

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var("COGNITO_AUTH", "0")

GRADIO_SERVER_PORT = int(get_or_create_env_var("GRADIO_SERVER_PORT", "7860"))

# Must match the named port mapping on the Fargate container (see cdk_stack.py).
ECS_SERVICE_CONNECT_PORT_MAPPING_NAME = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_PORT_MAPPING_NAME", f"port-{GRADIO_SERVER_PORT}"
)

# Suffix used with ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES (matches this stack's ECS SG name).
if ECS_SECURITY_GROUP_NAME.startswith(CDK_PREFIX):
    _default_sc_client_sg_suffix = ECS_SECURITY_GROUP_NAME[len(CDK_PREFIX) :]
else:
    _default_sc_client_sg_suffix = "SecurityGroupECS"
ECS_SERVICE_CONNECT_CLIENT_SG_NAME_SUFFIX = get_or_create_env_var(
    "ECS_SERVICE_CONNECT_CLIENT_SG_NAME_SUFFIX", _default_sc_client_sg_suffix
)

ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_LIST = parse_comma_separated_list(
    ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES
)
ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES_LIST = parse_comma_separated_list(
    ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES
)


def build_service_connect_client_security_group_names() -> List[str]:
    """Explicit SG names plus {prefix}{suffix} for each client CDK_PREFIX."""
    names: List[str] = list(ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_LIST)
    for prefix in ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES_LIST:
        names.append(f"{prefix}{ECS_SERVICE_CONNECT_CLIENT_SG_NAME_SUFFIX}")
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_TO_LOOKUP = (
    build_service_connect_client_security_group_names()
)

if ENABLE_ECS_SERVICE_CONNECT == "True" and USE_ECS_EXPRESS_MODE == "True":
    raise ValueError(
        "ENABLE_ECS_SERVICE_CONNECT=True is only supported on the legacy Fargate "
        "service path. Set USE_ECS_EXPRESS_MODE=False or disable Service Connect."
    )

# S3-uploaded job .env files trigger one-shot ECS Fargate tasks (direct mode / cli_redact).
ENABLE_S3_BATCH_ECS_TRIGGER = get_or_create_env_var(
    "ENABLE_S3_BATCH_ECS_TRIGGER", "False"
)
S3_BATCH_ENV_PREFIX = get_or_create_env_var("S3_BATCH_ENV_PREFIX", "input/config/")
S3_BATCH_ENV_SUFFIX = get_or_create_env_var("S3_BATCH_ENV_SUFFIX", ".env")
S3_BATCH_INPUT_PREFIX = get_or_create_env_var("S3_BATCH_INPUT_PREFIX", "input/")
S3_BATCH_CONFIG_PREFIX = get_or_create_env_var("S3_BATCH_CONFIG_PREFIX", "")
S3_BATCH_DEFAULT_PARAMS_KEY = get_or_create_env_var(
    "S3_BATCH_DEFAULT_PARAMS_KEY", "general-config/batch_defaults.env"
)
S3_BATCH_LAMBDA_FUNCTION_NAME = get_or_create_env_var(
    "S3_BATCH_LAMBDA_FUNCTION_NAME", ""
)

if ENABLE_S3_BATCH_ECS_TRIGGER == "True" and USE_ECS_EXPRESS_MODE == "True":
    raise ValueError(
        "ENABLE_S3_BATCH_ECS_TRIGGER=True requires the legacy Fargate task definition "
        "for ecs.run_task. Set USE_ECS_EXPRESS_MODE=False or disable the batch trigger."
    )

# Pi agent Gradio UI (second Fargate service; shared legacy ALB + Service Connect to main app).
ENABLE_PI_AGENT_ECS_SERVICE = get_or_create_env_var(
    "ENABLE_PI_AGENT_ECS_SERVICE", "False"
)
ECR_PI_REPO_NAME = get_or_create_env_var(
    "ECR_PI_REPO_NAME", f"{CDK_PREFIX}pi-agent".lower()
)
CODEBUILD_PI_PROJECT_NAME = get_or_create_env_var(
    "CODEBUILD_PI_PROJECT_NAME", f"{CDK_PREFIX}CodeBuildPiAgent"
)
ECS_PI_TASK_DEFINITION_NAME = get_or_create_env_var(
    "ECS_PI_TASK_DEFINITION_NAME", f"{CDK_PREFIX}PiAgentTaskDefinition"
)
ECS_PI_SECURITY_GROUP_NAME = get_or_create_env_var(
    "ECS_PI_SECURITY_GROUP_NAME", f"{CDK_PREFIX}SecurityGroupPiAgent"
)
ECS_PI_LOG_GROUP_NAME = get_or_create_env_var(
    "ECS_PI_LOG_GROUP_NAME", f"/ecs/{ECS_PI_SERVICE_NAME}-logs".lower()
)
ECS_PI_TASK_CPU_SIZE = get_or_create_env_var("ECS_PI_TASK_CPU_SIZE", "1024")
ECS_PI_TASK_MEMORY_SIZE = get_or_create_env_var("ECS_PI_TASK_MEMORY_SIZE", "2048")
PI_GRADIO_PORT = get_or_create_env_var("PI_GRADIO_PORT", "7862")
PI_ALB_HOST_HEADER = get_or_create_env_var("PI_ALB_HOST_HEADER", "")
PI_ALB_TARGET_GROUP_NAME = get_or_create_env_var(
    "PI_ALB_TARGET_GROUP_NAME", f"{CDK_PREFIX}PiAgentTG"[-32:]
)
PI_ALB_LISTENER_RULE_PRIORITY = int(
    get_or_create_env_var("PI_ALB_LISTENER_RULE_PRIORITY", "1")
)
PI_AGENT_ENV_S3_KEY = get_or_create_env_var("PI_AGENT_ENV_S3_KEY", "pi_agent.env")

if ENABLE_PI_AGENT_ECS_SERVICE == "True" and USE_ECS_EXPRESS_MODE == "True":
    raise ValueError(
        "ENABLE_PI_AGENT_ECS_SERVICE=True requires legacy Fargate (USE_ECS_EXPRESS_MODE=False)."
    )
if ENABLE_PI_AGENT_ECS_SERVICE == "True" and ENABLE_ECS_SERVICE_CONNECT != "True":
    raise ValueError(
        "ENABLE_PI_AGENT_ECS_SERVICE=True requires ENABLE_ECS_SERVICE_CONNECT=True "
        "so the Pi task can reach the main app at http://<discovery>:7860."
    )
if ENABLE_PI_AGENT_ECS_SERVICE == "True" and not PI_ALB_HOST_HEADER.strip():
    raise ValueError(
        "ENABLE_PI_AGENT_ECS_SERVICE=True requires PI_ALB_HOST_HEADER "
        "(host-header rule on the shared ALB, e.g. pi.redaction.example.com)."
    )

###
# WHOLE DOCUMENT API OPTIONS
###

DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS = get_or_create_env_var(
    "DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS", "7"
)  # How many days into the past should whole document Textract jobs be displayed? After that, the data is not deleted from the Textract jobs csv, but it is just filtered out. Included to align with S3 buckets where the file outputs will be automatically deleted after X days.
