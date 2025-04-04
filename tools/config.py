import os
import tempfile
import socket
from datetime import datetime
from dotenv import load_dotenv
from tldextract import TLDExtract

today_rev = datetime.now().strftime("%Y%m%d")
host_name = socket.gethostname()

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
APP_CONFIG_PATH = get_or_create_env_var('APP_CONFIG_PATH', '')


if os.path.exists(APP_CONFIG_PATH):
    print(f"Loading APP variables from config file {APP_CONFIG_PATH}")
    load_dotenv(APP_CONFIG_PATH)

###
# AWS CONFIG
###

# If you have an aws_config env file in the config folder, you can load in AWS keys this way, e.g. '/env/aws_config.env'
AWS_CONFIG_PATH = get_or_create_env_var('AWS_CONFIG_PATH', '')

if os.path.exists(AWS_CONFIG_PATH):
    print(f"Loading AWS variables from config file {AWS_CONFIG_PATH}")
    load_dotenv(AWS_CONFIG_PATH)

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "0")

AWS_REGION = get_or_create_env_var('AWS_REGION', 'eu-west-2')

AWS_CLIENT_ID = get_or_create_env_var('AWS_CLIENT_ID', '')

AWS_CLIENT_SECRET = get_or_create_env_var('AWS_CLIENT_SECRET', '')

AWS_USER_POOL_ID = get_or_create_env_var('AWS_USER_POOL_ID', '')

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

FEEDBACK_LOGS_FOLDER = get_or_create_env_var('FEEDBACK_LOGS_FOLDER', 'feedback/' + today_rev + '/' + host_name + '/')

USAGE_LOGS_FOLDER = get_or_create_env_var('USAGE_LOGS_FOLDER', 'logs/' + today_rev + '/' + host_name + '/')

ACCESS_LOGS_FOLDER = get_or_create_env_var('ACCESS_LOGS_FOLDER', 'usage/' + today_rev + '/' + host_name + '/')

DISPLAY_FILE_NAMES_IN_LOGS = get_or_create_env_var('DISPLAY_FILE_NAMES_IN_LOGS', 'False')

###
# REDACTION CONFIG
###
TESSERACT_FOLDER = get_or_create_env_var('TESSERACT_FOLDER', "tesseract/")

POPPLER_FOLDER = get_or_create_env_var('POPPLER_FOLDER', "poppler/poppler-24.02.0/Library/bin/")

SHOW_BULK_TEXTRACT_CALL_OPTIONS = get_or_create_env_var('SHOW_BULK_TEXTRACT_CALL_OPTIONS', 'False') # This feature not currently implemented

# Number of pages to loop through before breaking the function and restarting from the last finished page (not currently activated).
PAGE_BREAK_VALUE = get_or_create_env_var('PAGE_BREAK_VALUE', '99999')

MAX_TIME_VALUE = get_or_create_env_var('MAX_TIME_VALUE', '999999')

CUSTOM_BOX_COLOUR = get_or_create_env_var("CUSTOM_BOX_COLOUR", "")

REDACTION_LANGUAGE = get_or_create_env_var("REDACTION_LANGUAGE", "en") # Currently only English is supported by the app

###
# APP RUN CONFIG
###

TLDEXTRACT_CACHE = get_or_create_env_var('TLDEXTRACT_CACHE', 'tld/.tld_set_snapshot')
extract = TLDExtract(cache_dir=TLDEXTRACT_CACHE)

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')

RUN_DIRECT_MODE = get_or_create_env_var('RUN_DIRECT_MODE', '0')

MAX_QUEUE_SIZE = int(get_or_create_env_var('MAX_QUEUE_SIZE', '5'))

MAX_FILE_SIZE = get_or_create_env_var('MAX_FILE_SIZE', '250mb')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7860'))

ROOT_PATH = get_or_create_env_var('ROOT_PATH', '')

DEFAULT_CONCURRENCY_LIMIT = get_or_create_env_var('DEFAULT_CONCURRENCY_LIMIT', '5')

GET_DEFAULT_ALLOW_LIST = get_or_create_env_var('GET_DEFAULT_ALLOW_LIST', 'True')

ALLOW_LIST_PATH = get_or_create_env_var('ALLOW_LIST_PATH', 'config/default_allow_list.csv') #

S3_ALLOW_LIST_PATH = get_or_create_env_var('S3_ALLOW_LIST_PATH', 'default_allow_list.csv') # This is a path within the DOCUMENT_REDACTION_BUCKET

SHOW_COSTS = get_or_create_env_var('SHOW_COSTS', 'True')

GET_COST_CODES = get_or_create_env_var('GET_COST_CODES', 'True')

COST_CODES_PATH = get_or_create_env_var('COST_CODES_PATH', 'config/COST_CENTRES.csv') # file should be a csv file with a single table in it that has two columns with a header. First column should contain cost codes, second column should contain a name or description for the cost code

S3_COST_CODES_PATH = get_or_create_env_var('S3_COST_CODES_PATH', 'COST_CENTRES.csv') # This is a path within the DOCUMENT_REDACTION_BUCKET

ENFORCE_COST_CODES = get_or_create_env_var('ENFORCE_COST_CODES', 'False') # If you have cost codes listed, are they compulsory?

if ENFORCE_COST_CODES == 'True': GET_COST_CODES = 'True'
if GET_COST_CODES == 'True': ENFORCE_COST_CODES = 'False'