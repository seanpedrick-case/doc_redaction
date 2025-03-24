import os
from dotenv import load_dotenv

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


# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. '/env/app_config.env'
APP_CONFIG_PATH = get_or_create_env_var('APP_CONFIG_PATH', '', print_val=True)


if os.path.exists(APP_CONFIG_PATH):
    print(f"Loading APP variables from config file {APP_CONFIG_PATH}")
    load_dotenv(APP_CONFIG_PATH)

###
# AWS CONFIG
###

# If you have an aws_config env file in the config folder, you can load in AWS keys this way, e.g. '/env/aws_config.env'
AWS_CONFIG_PATH = get_or_create_env_var('AWS_CONFIG_PATH', '', print_val=True)

if os.path.exists(AWS_CONFIG_PATH):
    print(f"Loading AWS variables from config file {AWS_CONFIG_PATH}")
    load_dotenv(AWS_CONFIG_PATH)

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "0")

AWS_REGION = get_or_create_env_var('AWS_REGION', 'eu-west-2')

client_id = get_or_create_env_var('AWS_CLIENT_ID', '')

client_secret = get_or_create_env_var('AWS_CLIENT_SECRET', '')

user_pool_id = get_or_create_env_var('AWS_USER_POOL_ID', '')

AWS_ACCESS_KEY = get_or_create_env_var('AWS_ACCESS_KEY', '')
if AWS_ACCESS_KEY: print(f'AWS_ACCESS_KEY found in environment variables')

AWS_SECRET_KEY = get_or_create_env_var('AWS_SECRET_KEY', '')
if AWS_SECRET_KEY: print(f'AWS_SECRET_KEY found in environment variables')

DOCUMENT_REDACTION_BUCKET = get_or_create_env_var('DOCUMENT_REDACTION_BUCKET', '')

# Custom headers e.g. if routing traffic through Cloudfront
# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var('CUSTOM_HEADER', '')
if CUSTOM_HEADER: print(f'CUSTOM_HEADER found')

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var('CUSTOM_HEADER_VALUE', '')
if CUSTOM_HEADER_VALUE: print(f'CUSTOM_HEADER_VALUE found')

###
# Images config
###
IMAGES_DPI = get_or_create_env_var('IMAGES_DPI', '300.0')
LOAD_TRUNCATED_IMAGES = get_or_create_env_var('LOAD_TRUNCATED_IMAGES', 'True')
MAX_IMAGE_PIXELS = get_or_create_env_var('MAX_IMAGE_PIXELS', '') # Changed to None if blank in file_conversion.py

###
# File I/O config
###

output_folder = get_or_create_env_var('GRADIO_OUTPUT_FOLDER', 'output/')
print(f'The value of GRADIO_OUTPUT_FOLDER is {output_folder}')

session_output_folder = get_or_create_env_var('SESSION_OUTPUT_FOLDER', 'False')
print(f'The value of SESSION_OUTPUT_FOLDER is {session_output_folder}')

input_folder = get_or_create_env_var('GRADIO_INPUT_FOLDER', 'input/')
print(f'The value of GRADIO_INPUT_FOLDER is {input_folder}')

###
# REDACTION CONFIG
###
# Number of pages to loop through before breaking the function and restarting from the last finished page.
page_break_value = get_or_create_env_var('page_break_value', '50000')

max_time_value = get_or_create_env_var('max_time_value', '999999')

CUSTOM_BOX_COLOUR = get_or_create_env_var("CUSTOM_BOX_COLOUR", "")

###
# APP RUN CONFIG
###
# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')

RUN_DIRECT_MODE = get_or_create_env_var('RUN_DIRECT_MODE', '0')

MAX_QUEUE_SIZE = int(get_or_create_env_var('MAX_QUEUE_SIZE', '5'))

MAX_FILE_SIZE = get_or_create_env_var('MAX_FILE_SIZE', '250mb')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7860'))

ROOT_PATH = get_or_create_env_var('ROOT_PATH', '')

DEFAULT_CONCURRENCY_LIMIT = get_or_create_env_var('DEFAULT_CONCURRENCY_LIMIT', '5')

GET_DEFAULT_ALLOW_LIST = get_or_create_env_var('GET_DEFAULT_ALLOW_LIST', 'False')

DEFAULT_ALLOW_LIST_PATH = get_or_create_env_var('DEFAULT_ALLOW_LIST_PATH', '')