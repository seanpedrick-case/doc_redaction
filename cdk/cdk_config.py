import os
import tempfile
import socket
import logging
from datetime import datetime
from dotenv import load_dotenv
from tldextract import TLDExtract

today_rev = datetime.now().strftime("%Y%m%d")
HOST_NAME = socket.gethostname()

# Set or retrieve configuration variables for the redaction app

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
# AWS ROLE IDS
###
AWS_ACCOUNT_ID = get_or_create_env_var('AWS_ACCOUNT_ID', '')
ECR_REPO_NAME = get_or_create_env_var('ECR_REPO_NAME', 'doc-redaction') # Beware - cannot have underscores and must be lower case

###
# GITHUB REPO
###

GITHUB_REPO_USERNAME = get_or_create_env_var('GITHUB_REPO_USERNAME', 'spedrickcase')
GITHUB_REPO_NAME = get_or_create_env_var('GITHUB_REPO_NAME', 'doc_redaction')
GITHUB_REPO_BRANCH = get_or_create_env_var('GITHUB_REPO_BRANCH', 'main')

###
# CDK OPTIONS
###
CDK_PREFIX = get_or_create_env_var('CDK_PREFIX', '')

# Define the CDK output context file name
CONTEXT_FILE = get_or_create_env_var('CONTEXT_FILE', 'cdk.context.json')

SECRETS_MANAGER_ID = get_or_create_env_var('SECRETS_MANAGER_ID', '')

VPC_NAME = get_or_create_env_var('VPC_NAME', '')

CLOUDFRONT_PREFIX_LIST_ID = get_or_create_env_var('CLOUDFRONT_PREFIX_LIST_ID', '')

PUBLIC_SUBNETS_TO_USE = get_or_create_env_var('PUBLIC_SUBNETS_TO_USE', f'["{CDK_PREFIX}PublicSubnet1", "{CDK_PREFIX}PublicSubnet2", "{CDK_PREFIX}PublicSubnet3"]') # ["PublicSubnet1", "PublicSubnet2"]
PUBLIC_SUBNET_CIDR_BLOCKS = get_or_create_env_var('PUBLIC_SUBNET_CIDR_BLOCKS', '') # e.g. ["10.0.1.0/24", "10.0.2.0/24"]
PUBLIC_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var('PUBLIC_SUBNET_AVAILABILITY_ZONES', '') # e.g. ["A", "B"]

PRIVATE_SUBNETS_TO_USE = get_or_create_env_var('PRIVATE_SUBNETS_TO_USE', f'["{CDK_PREFIX}PrivateSubnet1", "{CDK_PREFIX}PrivateSubnet2", "{CDK_PREFIX}PrivateSubnet3"]') # ["PrivateSubnet1", "PrivateSubnet2"]
PRIVATE_SUBNET_CIDR_BLOCKS = get_or_create_env_var('PRIVATE_SUBNET_CIDR_BLOCKS', '')
PRIVATE_SUBNET_AVAILABILITY_ZONES = get_or_create_env_var('PRIVATE_SUBNET_AVAILABILITY_ZONES', '') # e.g. ["A", "B"]

CLOUDFRONT_GEO_RESTRICTION = get_or_create_env_var('CLOUDFRONT_GEO_RESTRICTION', 'GB') # A list of countries that Cloudfront restricts access to. See here: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/georestrictions.html

EXISTING_LOAD_BALANCER_ARN = get_or_create_env_var('EXISTING_LOAD_BALANCER_ARN', '')

###
# POLICY FILE OPTIONS
###

AWS_MANAGED_TASK_ROLES_LIST = get_or_create_env_var('AWS_MANAGED_TASK_ROLES_LIST', '["AmazonCognitoReadOnly", "AmazonECSTaskExecutionRolePolicy", "AmazonS3FullAccess", "AmazonTextractFullAccess", "ComprehendReadOnly", "AWSAppSyncPushToCloudWatchLogs"]')
POLICY_FILE_LOCATIONS = get_or_create_env_var('POLICY_FILE_LOCATIONS', '["cdk/config/sts_permissions.json"]')
POLICY_FILE_ARNS = get_or_create_env_var('POLICY_FILE_ARNS', '')

###
# ECS OPTIONS
###
ECS_TASK_CPU_SIZE = get_or_create_env_var('ECS_TASK_CPU_SIZE', '1024')
ECS_TASK_MEMORY_SIZE = get_or_create_env_var('ECS_TASK_MEMORY_SIZE', '4096')

###
# AWS OPTIONS
###

AWS_REGION = get_or_create_env_var('AWS_REGION', '')
AWS_ACCOUNT_ID = get_or_create_env_var('AWS_ACCOUNT_ID', '')

AWS_ACCESS_KEY = get_or_create_env_var('AWS_ACCESS_KEY', '')
if AWS_ACCESS_KEY: print(f'AWS_ACCESS_KEY found in environment variables')

AWS_SECRET_KEY = get_or_create_env_var('AWS_SECRET_KEY', '')
if AWS_SECRET_KEY: print(f'AWS_SECRET_KEY found in environment variables')

# Custom headers e.g. if routing traffic through Cloudfront
# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var('CUSTOM_HEADER', '')

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var('CUSTOM_HEADER_VALUE', '')

EXISTING_LOAD_BALANCER_ARN = get_or_create_env_var('EXISTING_LOAD_BALANCER_ARN', '')


###
# CDK TEMPLATE NAME OPTIONS
###
prefix = CDK_PREFIX
CODEBUILD_ROLE_NAME = get_or_create_env_var('CODEBUILD_ROLE_NAME', f"{prefix}CodeBuildRole")
CODEBUILD_PROJECT_NAME = get_or_create_env_var('CODEBUILD_PROJECT_NAME', f"{prefix}CodeBuildProject")

ECS_TASK_ROLE_NAME = get_or_create_env_var('ECS_TASK_ROLE_NAME', f"{prefix}TaskRole")
ECS_TASK_EXECUTION_ROLE_NAME = get_or_create_env_var('ECS_TASK_EXECUTION_ROLE_NAME', f"{prefix}ExecutionRole")

S3_LOG_CONFIG_BUCKET_NAME = get_or_create_env_var('S3_LOG_CONFIG_BUCKET_NAME', f"{prefix}s3-logs".lower()) # S3 bucket names need to be lower case
S3_OUTPUT_BUCKET_NAME = get_or_create_env_var('S3_OUTPUT_BUCKET_NAME', f"{prefix}s3-output".lower())

ECR_CDK_REPO_NAME = get_or_create_env_var('ECR_CDK_REPO_NAME', f"{prefix}{ECR_REPO_NAME}".lower()) 

ALB_NAME = get_or_create_env_var('ALB_NAME', f"{prefix}Alb"[-32:]) # Application load balancer name can be max 32 characters, so taking the last 32 characters of the suggested name

COGNITO_USER_POOL_NAME = get_or_create_env_var('COGNITO_USER_POOL_NAME', f"{prefix}UserPool")

COGNITO_USER_POOL_CLIENT_NAME = get_or_create_env_var('COGNITO_USER_POOL_CLIENT_NAME', f"{prefix}UserPoolClient")

COGNITO_USER_POOL_CLIENT_SECRET_NAME = get_or_create_env_var('COGNITO_USER_POOL_CLIENT_SECRET_NAME', f"{prefix}ParamCognitoSecret")

WEB_ACL_NAME = get_or_create_env_var('WEB_ACL_NAME', f"{prefix}web_acl")

NAT_GATEWAY_EIP_NAME = get_or_create_env_var('NAT_GATEWAY_EIP_NAME', f"{prefix}NatGatewayEip")

ECS_SECURITY_GROUP_NAME = get_or_create_env_var('ECS_SECURITY_GROUP_NAME', f"{prefix}SecurityGroupECS")

ALB_NAME_SECURITY_GROUP_NAME = get_or_create_env_var('ALB_SECURITY_GROUP_NAME', f"{prefix}SecurityGroupALB")

ALB_TARGET_GROUP_NAME = get_or_create_env_var('ALB_TARGET_GROUP_NAME', f"{prefix}-tg"[-32:]) # Max 32 characters

##
FARGATE_TASK_DEFINITION_NAME = get_or_create_env_var('FARGATE_TASK_DEFINITION_NAME', f"{prefix}FargateTaskDefinition")

CLUSTER_NAME = get_or_create_env_var('CLUSTER_NAME', f"{prefix}Cluster")

ECS_SERVICE_NAME = get_or_create_env_var('ECS_SERVICE_NAME', f"{prefix}ECSService")

CLOUDFRONT_DISTRIBUTION_NAME = get_or_create_env_var('CLOUDFRONT_DISTRIBUTION_NAME', f"{prefix}CfDist")

ACM_CERTIFICATE_ARN = get_or_create_env_var('ACM_CERTIFICATE_ARN', '')


###
# Image options
###
IMAGES_DPI = get_or_create_env_var('IMAGES_DPI', '300.0')
LOAD_TRUNCATED_IMAGES = get_or_create_env_var('LOAD_TRUNCATED_IMAGES', 'True')
MAX_IMAGE_PIXELS = get_or_create_env_var('MAX_IMAGE_PIXELS', '') # Changed to None if blank in file_conversion.py

###
# File I/O options
###

SESSION_OUTPUT_FOLDER = get_or_create_env_var('SESSION_OUTPUT_FOLDER', 'False') # i.e. do you want your input and output folders saved within a subfolder based on session hash value within output/input folders 

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

# By default, logs are put into a subfolder of today's date and the host name of the instance running the app. This is to avoid at all possible the possibility of log files from one instance overwriting the logs of another instance on S3. If running the app on one system always, or just locally, it is not necessary to make the log folders so specific.
# Another way to address this issue would be to write logs to another type of storage, e.g. database such as dynamodb. I may look into this in future.

SAVE_LOGS_TO_CSV = get_or_create_env_var('SAVE_LOGS_TO_CSV', 'True')

USE_LOG_SUBFOLDERS = get_or_create_env_var('USE_LOG_SUBFOLDERS', 'True')

if USE_LOG_SUBFOLDERS == "True":
    day_log_subfolder = today_rev + '/'
    host_name_subfolder = HOST_NAME + '/'
    full_log_subfolder = day_log_subfolder + host_name_subfolder
else:
    full_log_subfolder = ""

FEEDBACK_LOGS_FOLDER = get_or_create_env_var('FEEDBACK_LOGS_FOLDER', 'feedback/' + full_log_subfolder)
ACCESS_LOGS_FOLDER = get_or_create_env_var('ACCESS_LOGS_FOLDER', 'logs/' + full_log_subfolder)
USAGE_LOGS_FOLDER = get_or_create_env_var('USAGE_LOGS_FOLDER', 'usage/' + full_log_subfolder)


# Should the redacted file name be included in the logs? In some instances, the names of the files themselves could be sensitive, and should not be disclosed beyond the app. So, by default this is false.
DISPLAY_FILE_NAMES_IN_LOGS = get_or_create_env_var('DISPLAY_FILE_NAMES_IN_LOGS', 'False')

# Further customisation options for CSV logs

CSV_ACCESS_LOG_HEADERS = get_or_create_env_var('CSV_ACCESS_LOG_HEADERS', '') # If blank, uses component labels
CSV_FEEDBACK_LOG_HEADERS = get_or_create_env_var('CSV_FEEDBACK_LOG_HEADERS', '') # If blank, uses component labels
CSV_USAGE_LOG_HEADERS = get_or_create_env_var('CSV_USAGE_LOG_HEADERS', '["session_hash_textbox",	"doc_full_file_name_textbox",	"data_full_file_name_textbox",	"actual_time_taken_number",	"total_page_count",	"textract_query_number", "pii_detection_method", "comprehend_query_number",  "cost_code", "textract_handwriting_signature", "host_name_textbox", "text_extraction_method", "is_this_a_textract_api_call"]') # If blank, uses component labels

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table

SAVE_LOGS_TO_DYNAMODB = get_or_create_env_var('SAVE_LOGS_TO_DYNAMODB', 'False')

ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('ACCESS_LOG_DYNAMODB_TABLE_NAME', 'redaction_access_log')
DYNAMODB_ACCESS_LOG_HEADERS = get_or_create_env_var('DYNAMODB_ACCESS_LOG_HEADERS', '')

FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('FEEDBACK_LOG_DYNAMODB_TABLE_NAME', 'redaction_feedback')
DYNAMODB_FEEDBACK_LOG_HEADERS = get_or_create_env_var('DYNAMODB_FEEDBACK_LOG_HEADERS', '')

USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('USAGE_LOG_DYNAMODB_TABLE_NAME', 'redaction_usage')
DYNAMODB_USAGE_LOG_HEADERS = get_or_create_env_var('DYNAMODB_USAGE_LOG_HEADERS', '')

# Report logging to console?
LOGGING = get_or_create_env_var('LOGGING', 'False')

if LOGGING == 'True':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




###
# REDACTION OPTIONS
###

# Create Tesseract and Poppler folders if you have installed them locally
TESSERACT_FOLDER = get_or_create_env_var('TESSERACT_FOLDER', "") # e.g. tesseract/
POPPLER_FOLDER = get_or_create_env_var('POPPLER_FOLDER', "") # e.g. poppler/poppler-24.02.0/Library/bin/

if TESSERACT_FOLDER: add_folder_to_path(TESSERACT_FOLDER)
if POPPLER_FOLDER: add_folder_to_path(POPPLER_FOLDER)


# Number of pages to loop through before breaking the function and restarting from the last finished page (not currently activated).
PAGE_BREAK_VALUE = get_or_create_env_var('PAGE_BREAK_VALUE', '99999')

MAX_TIME_VALUE = get_or_create_env_var('MAX_TIME_VALUE', '999999')

CUSTOM_BOX_COLOUR = get_or_create_env_var("CUSTOM_BOX_COLOUR", "") # only "grey" is currently supported as a custom box colour

REDACTION_LANGUAGE = get_or_create_env_var("REDACTION_LANGUAGE", "en") # Currently only English is supported by the app

RETURN_PDF_END_OF_REDACTION = get_or_create_env_var("RETURN_PDF_END_OF_REDACTION", "True") # Return a redacted PDF at the end of the redaction task. Could be useful to set this to "False" if you want to ensure that the user always goes to the 'Review Redactions' tab before getting the final redacted PDF product.

COMPRESS_REDACTED_PDF = get_or_create_env_var("COMPRESS_REDACTED_PDF","False") # On low memory systems, the compression options in pymupdf can cause the app to crash if the PDF is longer than 500 pages or so. Setting this to False will save the PDF only with a basic cleaning option enabled




###
# APP RUN OPTIONS
###

TLDEXTRACT_CACHE = get_or_create_env_var('TLDEXTRACT_CACHE', 'tld/.tld_set_snapshot')
try:
    extract = TLDExtract(cache_dir=TLDEXTRACT_CACHE)
except:
    extract = TLDExtract(cache_dir=None)

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')

RUN_DIRECT_MODE = get_or_create_env_var('RUN_DIRECT_MODE', '0')

MAX_QUEUE_SIZE = int(get_or_create_env_var('MAX_QUEUE_SIZE', '5'))

MAX_FILE_SIZE = get_or_create_env_var('MAX_FILE_SIZE', '250mb')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7860'))

ROOT_PATH = get_or_create_env_var('ROOT_PATH', '')

DEFAULT_CONCURRENCY_LIMIT = get_or_create_env_var('DEFAULT_CONCURRENCY_LIMIT', '3')

GET_DEFAULT_ALLOW_LIST = get_or_create_env_var('GET_DEFAULT_ALLOW_LIST', '')

ALLOW_LIST_PATH = get_or_create_env_var('ALLOW_LIST_PATH', '') # config/default_allow_list.csv

S3_ALLOW_LIST_PATH = get_or_create_env_var('S3_ALLOW_LIST_PATH', '') # default_allow_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

if ALLOW_LIST_PATH: OUTPUT_ALLOW_LIST_PATH = ALLOW_LIST_PATH
else: OUTPUT_ALLOW_LIST_PATH = 'config/default_allow_list.csv'

###
# COST CODE OPTIONS
###

SHOW_COSTS = get_or_create_env_var('SHOW_COSTS', 'False')

GET_COST_CODES = get_or_create_env_var('GET_COST_CODES', 'False')

DEFAULT_COST_CODE = get_or_create_env_var('DEFAULT_COST_CODE', '')

COST_CODES_PATH = get_or_create_env_var('COST_CODES_PATH', '') # 'config/COST_CENTRES.csv' # file should be a csv file with a single table in it that has two columns with a header. First column should contain cost codes, second column should contain a name or description for the cost code

S3_COST_CODES_PATH = get_or_create_env_var('S3_COST_CODES_PATH', '') # COST_CENTRES.csv # This is a path within the DOCUMENT_REDACTION_BUCKET  
    
# A default path in case s3 cost code location is provided but no local cost code location given
if COST_CODES_PATH: OUTPUT_COST_CODES_PATH = COST_CODES_PATH
else: OUTPUT_COST_CODES_PATH = 'config/cost_codes.csv'

ENFORCE_COST_CODES = get_or_create_env_var('ENFORCE_COST_CODES', 'False') # If you have cost codes listed, is it compulsory to choose one before redacting?

if ENFORCE_COST_CODES == 'True': GET_COST_CODES = 'True'


###
# WHOLE DOCUMENT API OPTIONS
###

SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS = get_or_create_env_var('SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS', 'False') # This feature not currently implemented

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET = get_or_create_env_var('TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET', '')

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER = get_or_create_env_var('TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER', 'input')

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER = get_or_create_env_var('TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER', 'output')

LOAD_PREVIOUS_TEXTRACT_JOBS_S3 = get_or_create_env_var('LOAD_PREVIOUS_TEXTRACT_JOBS_S3', 'False') # Whether or not to load previous Textract jobs from S3

TEXTRACT_JOBS_S3_LOC = get_or_create_env_var('TEXTRACT_JOBS_S3_LOC', 'output') # Subfolder in the DOCUMENT_REDACTION_BUCKET where the Textract jobs are stored

TEXTRACT_JOBS_S3_INPUT_LOC = get_or_create_env_var('TEXTRACT_JOBS_S3_INPUT_LOC', 'input') # Subfolder in the DOCUMENT_REDACTION_BUCKET where the Textract jobs are stored

TEXTRACT_JOBS_LOCAL_LOC = get_or_create_env_var('TEXTRACT_JOBS_LOCAL_LOC', 'output') # Local subfolder where the Textract jobs are stored

DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS = get_or_create_env_var('DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS', '7') # How many days into the past should whole document Textract jobs be displayed? After that, the data is not deleted from the Textract jobs csv, but it is just filtered out. Included to align with S3 buckets where the file outputs will be automatically deleted after X days.