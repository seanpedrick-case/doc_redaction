import os
import tempfile
from dotenv import load_dotenv

# Set or retrieve configuration variables for CDK redaction deployment

def get_or_create_env_var(var_name:str, default_value:str, print_val:bool=False):
    '''
    Get an environmental variable, and set it to a default value if it doesn't exist
    '''
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set the environment variable to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    if print_val == True:
        print(f'The value of {var_name} is {value}')
    
    return value

def ensure_folder_exists(output_folder:str):
    """Checks if the specified folder exists, creates it if not."""   

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")

def add_folder_to_path(folder_path: str):
    '''
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    '''

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ['PATH']
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ['PATH'] = full_path_extension
            #print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")

###
# LOAD CONFIG FROM ENV FILE
###
CONFIG_FOLDER = get_or_create_env_var('CONFIG_FOLDER', "config/")

ensure_folder_exists(CONFIG_FOLDER)

# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. 'config/cdk_config.env'
CDK_CONFIG_PATH = get_or_create_env_var('CDK_CONFIG_PATH', 'config/cdk_config.env') # e.g. config/cdk_config.env

if CDK_CONFIG_PATH:
    if os.path.exists(CDK_CONFIG_PATH):
        print(f"Loading CDK variables from config file {CDK_CONFIG_PATH}")
        load_dotenv(CDK_CONFIG_PATH)
    else: print("CDK config file not found at location:", CDK_CONFIG_PATH)

###
# AWS OPTIONS
###
AWS_REGION = get_or_create_env_var('AWS_REGION', '')
AWS_ACCOUNT_ID = get_or_create_env_var('AWS_ACCOUNT_ID', '')

###
# CDK OPTIONS
###
CDK_PREFIX = get_or_create_env_var('CDK_PREFIX', '')
CONTEXT_FILE = get_or_create_env_var('CONTEXT_FILE', 'cdk.context.json') # Define the CDK output context file name
CDK_FOLDER = get_or_create_env_var('CDK_FOLDER', '') # FULL_PATH_TO_CDK_FOLDER_HERE (with forward slash)
RUN_USEAST_STACK = get_or_create_env_var('RUN_USEAST_STACK', 'False')

### VPC
VPC_NAME = get_or_create_env_var('VPC_NAME', '')
EXISTING_IGW_ID = get_or_create_env_var('EXISTING_IGW_ID', '')
SINGLE_NAT_GATEWAY_ID = get_or_create_env_var('SINGLE_NAT_GATEWAY_ID', '')

### SUBNETS / ROUTE TABLES / NAT GATEWAY
PUBLIC_SUBNETS_TO_USE = get_or_create_env_var('PUBLIC_SUBNETS_TO_USE', '') # e.g. ['PublicSubnet1', 'PublicSubnet2']
PUBLIC_SUBNET_CIDR_BLOCKS = get_or_create_env_var('PUBLIC_SUBNET_CIDR_BLOCKS', '') # e.g. ["10.0.1.0/24", "10.0.2.0/24"]
PUBLIC_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var('PUBLIC_SUBNET_AVAILABILITY_ZONES', '') # e.g. ["eu-east-1b", "eu-east1b"]

PRIVATE_SUBNETS_TO_USE = get_or_create_env_var('PRIVATE_SUBNETS_TO_USE', '') # e.g. ['PrivateSubnet1', 'PrivateSubnet2']
PRIVATE_SUBNET_CIDR_BLOCKS = get_or_create_env_var('PRIVATE_SUBNET_CIDR_BLOCKS', '') # e.g. ["10.0.1.0/24", "10.0.2.0/24"]
PRIVATE_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var('PRIVATE_SUBNET_AVAILABILITY_ZONES', '') # e.g. ["eu-east-1b", "eu-east1b"]

ROUTE_TABLE_BASE_NAME = get_or_create_env_var('ROUTE_TABLE_BASE_NAME', f'{CDK_PREFIX}PrivateRouteTable')
NAT_GATEWAY_EIP_NAME = get_or_create_env_var('NAT_GATEWAY_EIP_NAME', f"{CDK_PREFIX}NatGatewayEip")
NAT_GATEWAY_NAME = get_or_create_env_var('NAT_GATEWAY_NAME', f"{CDK_PREFIX}NatGateway")

# IAM roles
AWS_MANAGED_TASK_ROLES_LIST = get_or_create_env_var('AWS_MANAGED_TASK_ROLES_LIST', '["AmazonCognitoReadOnly", "service-role/AmazonECSTaskExecutionRolePolicy", "AmazonS3FullAccess", "AmazonTextractFullAccess", "ComprehendReadOnly", "AmazonDynamoDBFullAccess", "service-role/AWSAppSyncPushToCloudWatchLogs"]')
POLICY_FILE_LOCATIONS = get_or_create_env_var('POLICY_FILE_LOCATIONS', '') # e.g. '["config/sts_permissions.json"]'
POLICY_FILE_ARNS = get_or_create_env_var('POLICY_FILE_ARNS', '')

# GITHUB REPO
GITHUB_REPO_USERNAME = get_or_create_env_var('GITHUB_REPO_USERNAME', 'seanpedrick-case')
GITHUB_REPO_NAME = get_or_create_env_var('GITHUB_REPO_NAME', 'doc_redaction')
GITHUB_REPO_BRANCH = get_or_create_env_var('GITHUB_REPO_BRANCH', 'main')

### CODEBUILD
CODEBUILD_ROLE_NAME = get_or_create_env_var('CODEBUILD_ROLE_NAME', f"{CDK_PREFIX}CodeBuildRole")
CODEBUILD_PROJECT_NAME = get_or_create_env_var('CODEBUILD_PROJECT_NAME', f"{CDK_PREFIX}CodeBuildProject")

### ECR
ECR_REPO_NAME = get_or_create_env_var('ECR_REPO_NAME', 'doc-redaction') # Beware - cannot have underscores and must be lower case
ECR_CDK_REPO_NAME = get_or_create_env_var('ECR_CDK_REPO_NAME', f"{CDK_PREFIX}{ECR_REPO_NAME}".lower()) 

### S3
S3_LOG_CONFIG_BUCKET_NAME = get_or_create_env_var('S3_LOG_CONFIG_BUCKET_NAME', f"{CDK_PREFIX}s3-logs".lower()) # S3 bucket names need to be lower case
S3_OUTPUT_BUCKET_NAME = get_or_create_env_var('S3_OUTPUT_BUCKET_NAME', f"{CDK_PREFIX}s3-output".lower())

### ECS
FARGATE_TASK_DEFINITION_NAME = get_or_create_env_var('FARGATE_TASK_DEFINITION_NAME', f"{CDK_PREFIX}FargateTaskDefinition")
TASK_DEFINITION_FILE_LOCATION = get_or_create_env_var('TASK_DEFINITION_FILE_LOCATION', CDK_FOLDER + CONFIG_FOLDER + "task_definition.json") 

CLUSTER_NAME = get_or_create_env_var('CLUSTER_NAME', f"{CDK_PREFIX}Cluster")
ECS_SERVICE_NAME = get_or_create_env_var('ECS_SERVICE_NAME', f"{CDK_PREFIX}ECSService")
ECS_TASK_ROLE_NAME = get_or_create_env_var('ECS_TASK_ROLE_NAME', f"{CDK_PREFIX}TaskRole")
ECS_TASK_EXECUTION_ROLE_NAME = get_or_create_env_var('ECS_TASK_EXECUTION_ROLE_NAME', f"{CDK_PREFIX}ExecutionRole")
ECS_SECURITY_GROUP_NAME = get_or_create_env_var('ECS_SECURITY_GROUP_NAME', f"{CDK_PREFIX}SecurityGroupECS")
ECS_LOG_GROUP_NAME = get_or_create_env_var('ECS_LOG_GROUP_NAME', f"/ecs/{ECS_SERVICE_NAME}-logs".lower())

ECS_TASK_CPU_SIZE = get_or_create_env_var('ECS_TASK_CPU_SIZE', '1024')
ECS_TASK_MEMORY_SIZE = get_or_create_env_var('ECS_TASK_MEMORY_SIZE', '4096')
ECS_USE_FARGATE_SPOT = get_or_create_env_var('USE_FARGATE_SPOT', 'False')
ECS_READ_ONLY_FILE_SYSTEM = get_or_create_env_var('ECS_READ_ONLY_FILE_SYSTEM', 'True')

### Cognito
COGNITO_USER_POOL_NAME = get_or_create_env_var('COGNITO_USER_POOL_NAME', f"{CDK_PREFIX}UserPool")
COGNITO_USER_POOL_CLIENT_NAME = get_or_create_env_var('COGNITO_USER_POOL_CLIENT_NAME', f"{CDK_PREFIX}UserPoolClient")
COGNITO_USER_POOL_CLIENT_SECRET_NAME = get_or_create_env_var('COGNITO_USER_POOL_CLIENT_SECRET_NAME', f"{CDK_PREFIX}ParamCognitoSecret")
COGNITO_USER_POOL_DOMAIN_PREFIX = get_or_create_env_var('COGNITO_USER_POOL_DOMAIN_PREFIX', "redaction-app-domain") # Should change this to something unique or you'll probably hit an error

# Application load balancer
ALB_NAME = get_or_create_env_var('ALB_NAME', f"{CDK_PREFIX}Alb"[-32:]) # Application load balancer name can be max 32 characters, so taking the last 32 characters of the suggested name
ALB_NAME_SECURITY_GROUP_NAME = get_or_create_env_var('ALB_SECURITY_GROUP_NAME', f"{CDK_PREFIX}SecurityGroupALB")
ALB_TARGET_GROUP_NAME = get_or_create_env_var('ALB_TARGET_GROUP_NAME', f"{CDK_PREFIX}-tg"[-32:]) # Max 32 characters
EXISTING_LOAD_BALANCER_ARN = get_or_create_env_var('EXISTING_LOAD_BALANCER_ARN', '')
EXISTING_LOAD_BALANCER_DNS = get_or_create_env_var('EXISTING_LOAD_BALANCER_ARN', 'placeholder_load_balancer_dns.net')

## CLOUDFRONT
USE_CLOUDFRONT = get_or_create_env_var('USE_CLOUDFRONT', 'True')
CLOUDFRONT_PREFIX_LIST_ID = get_or_create_env_var('CLOUDFRONT_PREFIX_LIST_ID', 'pl-93a247fa')
CLOUDFRONT_GEO_RESTRICTION = get_or_create_env_var('CLOUDFRONT_GEO_RESTRICTION', '') # A country that Cloudfront restricts access to. See here: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/georestrictions.html
CLOUDFRONT_DISTRIBUTION_NAME = get_or_create_env_var('CLOUDFRONT_DISTRIBUTION_NAME', f"{CDK_PREFIX}CfDist")
CLOUDFRONT_DOMAIN = get_or_create_env_var('CLOUDFRONT_DOMAIN', "cloudfront_placeholder.net")


# Certificate for Application load balancer (optional, for HTTPS and logins through the ALB)
ACM_CERTIFICATE_ARN = get_or_create_env_var('ACM_CERTIFICATE_ARN', '')
SSL_CERTIFICATE_DOMAIN = get_or_create_env_var('SSL_CERTIFICATE_DOMAIN', '') # e.g. example.com or www.example.com

# This should be the CloudFront domain, the domain linked to your ACM certificate, or the DNS of your application load balancer in console afterwards
if USE_CLOUDFRONT == "True":
    COGNITO_REDIRECTION_URL = get_or_create_env_var('COGNITO_REDIRECTION_URL', "https://" + CLOUDFRONT_DOMAIN) 
elif SSL_CERTIFICATE_DOMAIN:
    COGNITO_REDIRECTION_URL = get_or_create_env_var('COGNITO_REDIRECTION_URL', "https://" + SSL_CERTIFICATE_DOMAIN)
else:
    COGNITO_REDIRECTION_URL = get_or_create_env_var('COGNITO_REDIRECTION_URL', "https://" + EXISTING_LOAD_BALANCER_DNS)

# Custom headers e.g. if routing traffic through Cloudfront
CUSTOM_HEADER = get_or_create_env_var('CUSTOM_HEADER', '') # Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER_VALUE = get_or_create_env_var('CUSTOM_HEADER_VALUE', '') # Retrieving or setting CUSTOM_HEADER_VALUE

# Firewall on top of load balancer
LOAD_BALANCER_WEB_ACL_NAME = get_or_create_env_var('LOAD_BALANCER_WEB_ACL_NAME', f"{CDK_PREFIX}alb-web-acl")

# Firewall on top of CloudFront
WEB_ACL_NAME = get_or_create_env_var('WEB_ACL_NAME', f"{CDK_PREFIX}cloudfront-web-acl")

###
# File I/O options
###

OUTPUT_FOLDER = get_or_create_env_var('GRADIO_OUTPUT_FOLDER', 'output/') # 'output/'
INPUT_FOLDER = get_or_create_env_var('GRADIO_INPUT_FOLDER', 'input/') # 'input/'

# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP": 
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f'Temporary directory created at: {temp_dir}')

        if OUTPUT_FOLDER == "TEMP": OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP": INPUT_FOLDER = temp_dir + "/"

###
# LOGGING OPTIONS
###

SAVE_LOGS_TO_CSV = get_or_create_env_var('SAVE_LOGS_TO_CSV', 'True')

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table
SAVE_LOGS_TO_DYNAMODB = get_or_create_env_var('SAVE_LOGS_TO_DYNAMODB', 'True')
ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('ACCESS_LOG_DYNAMODB_TABLE_NAME', f"{CDK_PREFIX}dynamodb-access-log".lower())
FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('FEEDBACK_LOG_DYNAMODB_TABLE_NAME', f"{CDK_PREFIX}dynamodb-feedback".lower())
USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('USAGE_LOG_DYNAMODB_TABLE_NAME', f"{CDK_PREFIX}dynamodb-usage".lower())

###
# REDACTION OPTIONS
###

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7860'))

###
# WHOLE DOCUMENT API OPTIONS
###

DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS = get_or_create_env_var('DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS', '7') # How many days into the past should whole document Textract jobs be displayed? After that, the data is not deleted from the Textract jobs csv, but it is just filtered out. Included to align with S3 buckets where the file outputs will be automatically deleted after X days.