import logging
import os
import socket
import tempfile
import urllib.parse
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from tldextract import TLDExtract

today_rev = datetime.now().strftime("%Y%m%d")
HOST_NAME = socket.gethostname()


def _get_env_list(env_var_name: str) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('"', "").replace("'", "")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(",") if s.strip()]


# Set or retrieve configuration variables for the redaction app


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


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            pass
            # print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


###
# LOAD CONFIG FROM ENV FILE
###

CONFIG_FOLDER = get_or_create_env_var("CONFIG_FOLDER", "config/")

# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. 'config/app_config.env'
APP_CONFIG_PATH = get_or_create_env_var(
    "APP_CONFIG_PATH", CONFIG_FOLDER + "app_config.env"
)  # e.g. config/app_config.env

if APP_CONFIG_PATH:
    if os.path.exists(APP_CONFIG_PATH):
        print(f"Loading app variables from config file {APP_CONFIG_PATH}")
        load_dotenv(APP_CONFIG_PATH)
    else:
        print("App config file not found at location:", APP_CONFIG_PATH)

###
# AWS OPTIONS
###

# If you have an aws_config env file in the config folder, you can load in AWS keys this way, e.g. 'env/aws_config.env'
AWS_CONFIG_PATH = get_or_create_env_var(
    "AWS_CONFIG_PATH", ""
)  # e.g. config/aws_config.env

if AWS_CONFIG_PATH:
    if os.path.exists(AWS_CONFIG_PATH):
        print(f"Loading AWS variables from config file {AWS_CONFIG_PATH}")
        load_dotenv(AWS_CONFIG_PATH)
    else:
        print("AWS config file not found at location:", AWS_CONFIG_PATH)

RUN_AWS_FUNCTIONS = convert_string_to_boolean(
    get_or_create_env_var("RUN_AWS_FUNCTIONS", "False")
)

AWS_REGION = get_or_create_env_var("AWS_REGION", "")

AWS_CLIENT_ID = get_or_create_env_var("AWS_CLIENT_ID", "")

AWS_CLIENT_SECRET = get_or_create_env_var("AWS_CLIENT_SECRET", "")

AWS_USER_POOL_ID = get_or_create_env_var("AWS_USER_POOL_ID", "")

AWS_ACCESS_KEY = get_or_create_env_var("AWS_ACCESS_KEY", "")
# if AWS_ACCESS_KEY: print(f'AWS_ACCESS_KEY found in environment variables')

AWS_SECRET_KEY = get_or_create_env_var("AWS_SECRET_KEY", "")
# if AWS_SECRET_KEY: print(f'AWS_SECRET_KEY found in environment variables')

DOCUMENT_REDACTION_BUCKET = get_or_create_env_var("DOCUMENT_REDACTION_BUCKET", "")

# Should the app prioritise using AWS SSO over using API keys stored in environment variables/secrets (defaults to yes)
PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS = convert_string_to_boolean(
    get_or_create_env_var("PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS", "True")
)

# Custom headers e.g. if routing traffic through Cloudfront
# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var("CUSTOM_HEADER", "")

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var("CUSTOM_HEADER_VALUE", "")

###
# Image options
###
IMAGES_DPI = float(get_or_create_env_var("IMAGES_DPI", "300.0"))
LOAD_TRUNCATED_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("LOAD_TRUNCATED_IMAGES", "True")
)
MAX_IMAGE_PIXELS = get_or_create_env_var(
    "MAX_IMAGE_PIXELS", ""
)  # Changed to None if blank in file_conversion.py

###
# File I/O options
###

SESSION_OUTPUT_FOLDER = get_or_create_env_var(
    "SESSION_OUTPUT_FOLDER", "False"
)  # i.e. do you want your input and output folders saved within a subfolder based on session hash value within output/input folders

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

GRADIO_TEMP_DIR = get_or_create_env_var(
    "GRADIO_TEMP_DIR", ""
)  # Default Gradio temp folder
MPLCONFIGDIR = get_or_create_env_var("MPLCONFIGDIR", "")  # Matplotlib cache folder

###
# LOGGING OPTIONS
###

# By default, logs are put into a subfolder of today's date and the host name of the instance running the app. This is to avoid at all possible the possibility of log files from one instance overwriting the logs of another instance on S3. If running the app on one system always, or just locally, it is not necessary to make the log folders so specific.
# Another way to address this issue would be to write logs to another type of storage, e.g. database such as dynamodb. I may look into this in future.

SAVE_LOGS_TO_CSV = convert_string_to_boolean(
    get_or_create_env_var("SAVE_LOGS_TO_CSV", "True")
)

USE_LOG_SUBFOLDERS = convert_string_to_boolean(
    get_or_create_env_var("USE_LOG_SUBFOLDERS", "True")
)

FEEDBACK_LOGS_FOLDER = get_or_create_env_var("FEEDBACK_LOGS_FOLDER", "feedback/")
ACCESS_LOGS_FOLDER = get_or_create_env_var("ACCESS_LOGS_FOLDER", "logs/")
USAGE_LOGS_FOLDER = get_or_create_env_var("USAGE_LOGS_FOLDER", "usage/")

if USE_LOG_SUBFOLDERS:
    day_log_subfolder = today_rev + "/"
    host_name_subfolder = HOST_NAME + "/"
    full_log_subfolder = day_log_subfolder + host_name_subfolder

    FEEDBACK_LOGS_FOLDER = FEEDBACK_LOGS_FOLDER + full_log_subfolder
    ACCESS_LOGS_FOLDER = ACCESS_LOGS_FOLDER + full_log_subfolder
    USAGE_LOGS_FOLDER = USAGE_LOGS_FOLDER + full_log_subfolder

S3_FEEDBACK_LOGS_FOLDER = get_or_create_env_var(
    "S3_FEEDBACK_LOGS_FOLDER", "feedback/" + full_log_subfolder
)
S3_ACCESS_LOGS_FOLDER = get_or_create_env_var(
    "S3_ACCESS_LOGS_FOLDER", "logs/" + full_log_subfolder
)
S3_USAGE_LOGS_FOLDER = get_or_create_env_var(
    "S3_USAGE_LOGS_FOLDER", "usage/" + full_log_subfolder
)

# Should the redacted file name be included in the logs? In some instances, the names of the files themselves could be sensitive, and should not be disclosed beyond the app. So, by default this is false.
DISPLAY_FILE_NAMES_IN_LOGS = convert_string_to_boolean(
    get_or_create_env_var("DISPLAY_FILE_NAMES_IN_LOGS", "False")
)

# Further customisation options for CSV logs
CSV_ACCESS_LOG_HEADERS = get_or_create_env_var(
    "CSV_ACCESS_LOG_HEADERS", ""
)  # If blank, uses component labels
CSV_FEEDBACK_LOG_HEADERS = get_or_create_env_var(
    "CSV_FEEDBACK_LOG_HEADERS", ""
)  # If blank, uses component labels
CSV_USAGE_LOG_HEADERS = get_or_create_env_var(
    "CSV_USAGE_LOG_HEADERS",
    '["session_hash_textbox", "doc_full_file_name_textbox", "data_full_file_name_textbox", "actual_time_taken_number",	"total_page_count",	"textract_query_number", "pii_detection_method", "comprehend_query_number",  "cost_code", "textract_handwriting_signature", "host_name_textbox", "text_extraction_method", "is_this_a_textract_api_call", "task"]',
)  # If blank, uses component labels

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table
SAVE_LOGS_TO_DYNAMODB = convert_string_to_boolean(
    get_or_create_env_var("SAVE_LOGS_TO_DYNAMODB", "False")
)

ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "ACCESS_LOG_DYNAMODB_TABLE_NAME", "redaction_access_log"
)
DYNAMODB_ACCESS_LOG_HEADERS = get_or_create_env_var("DYNAMODB_ACCESS_LOG_HEADERS", "")

FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "FEEDBACK_LOG_DYNAMODB_TABLE_NAME", "redaction_feedback"
)
DYNAMODB_FEEDBACK_LOG_HEADERS = get_or_create_env_var(
    "DYNAMODB_FEEDBACK_LOG_HEADERS", ""
)

USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var(
    "USAGE_LOG_DYNAMODB_TABLE_NAME", "redaction_usage"
)
DYNAMODB_USAGE_LOG_HEADERS = get_or_create_env_var("DYNAMODB_USAGE_LOG_HEADERS", "")

# Report logging to console?
LOGGING = convert_string_to_boolean(get_or_create_env_var("LOGGING", "False"))

if LOGGING:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

LOG_FILE_NAME = get_or_create_env_var("LOG_FILE_NAME", "log.csv")
USAGE_LOG_FILE_NAME = get_or_create_env_var("USAGE_LOG_FILE_NAME", LOG_FILE_NAME)
FEEDBACK_LOG_FILE_NAME = get_or_create_env_var("FEEDBACK_LOG_FILE_NAME", LOG_FILE_NAME)


###
# Gradio general app options
###

FAVICON_PATH = get_or_create_env_var("FAVICON_PATH", "favicon.png")

RUN_FASTAPI = convert_string_to_boolean(get_or_create_env_var("RUN_FASTAPI", "False"))

RUN_MCP_SERVER = convert_string_to_boolean(
    get_or_create_env_var("RUN_MCP_SERVER", "False")
)

MAX_QUEUE_SIZE = int(get_or_create_env_var("MAX_QUEUE_SIZE", "5"))

MAX_FILE_SIZE = get_or_create_env_var("MAX_FILE_SIZE", "250mb").lower()

GRADIO_SERVER_NAME = get_or_create_env_var("GRADIO_SERVER_NAME", "0.0.0.0")

GRADIO_SERVER_PORT = int(get_or_create_env_var("GRADIO_SERVER_PORT", "7860"))

ALLOWED_ORIGINS = get_or_create_env_var(
    "ALLOWED_ORIGINS", ""
)  # should be a list of allowed origins e.g. ['https://example.com', 'https://www.example.com']

ALLOWED_HOSTS = get_or_create_env_var("ALLOWED_HOSTS", "")

ROOT_PATH = get_or_create_env_var("ROOT_PATH", "")
FASTAPI_ROOT_PATH = get_or_create_env_var("FASTAPI_ROOT_PATH", "/")

DEFAULT_CONCURRENCY_LIMIT = int(get_or_create_env_var("DEFAULT_CONCURRENCY_LIMIT", "3"))

# Number of pages to loop through before breaking the function and restarting from the last finished page (not currently activated).
PAGE_BREAK_VALUE = int(get_or_create_env_var("PAGE_BREAK_VALUE", "99999"))

MAX_TIME_VALUE = int(get_or_create_env_var("MAX_TIME_VALUE", "999999"))
MAX_SIMULTANEOUS_FILES = int(get_or_create_env_var("MAX_SIMULTANEOUS_FILES", "10"))
MAX_DOC_PAGES = int(get_or_create_env_var("MAX_DOC_PAGES", "3000"))
MAX_TABLE_ROWS = int(get_or_create_env_var("MAX_TABLE_ROWS", "250000"))
MAX_TABLE_COLUMNS = int(get_or_create_env_var("MAX_TABLE_COLUMNS", "100"))
MAX_OPEN_TEXT_CHARACTERS = int(
    get_or_create_env_var("MAX_OPEN_TEXT_CHARACTERS", "50000")
)

# When loading for review, should PDFs have existing redaction annotations loaded in?
LOAD_REDACTION_ANNOTATIONS_FROM_PDF = convert_string_to_boolean(
    get_or_create_env_var("LOAD_REDACTION_ANNOTATIONS_FROM_PDF", "True")
)


# Create Tesseract and Poppler folders if you have installed them locally
TESSERACT_FOLDER = get_or_create_env_var(
    "TESSERACT_FOLDER", ""
)  #  # If installing for Windows, install Tesseract 5.5.0 from here: https://github.com/UB-Mannheim/tesseract/wiki. Then this environment variable should point to the Tesseract folder e.g. tesseract/
TESSERACT_DATA_FOLDER = get_or_create_env_var(
    "TESSERACT_DATA_FOLDER", "/usr/share/tessdata"
)
POPPLER_FOLDER = get_or_create_env_var(
    "POPPLER_FOLDER", ""
)  # If installing on Windows,install Poppler from here https://github.com/oschwartz10612/poppler-windows. This variable needs to point to the poppler bin folder e.g. poppler/poppler-24.02.0/Library/bin/

if TESSERACT_FOLDER:
    add_folder_to_path(TESSERACT_FOLDER)
if POPPLER_FOLDER:
    add_folder_to_path(POPPLER_FOLDER)

# Extraction and PII options open by default:
EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT = convert_string_to_boolean(
    get_or_create_env_var("EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT", "True")
)

# List of models to use for text extraction and PII detection
# Text extraction models
SELECTABLE_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "SELECTABLE_TEXT_EXTRACT_OPTION", "Local model - selectable text"
)
TESSERACT_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "TESSERACT_TEXT_EXTRACT_OPTION", "Local OCR model - PDFs without selectable text"
)
TEXTRACT_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "TEXTRACT_TEXT_EXTRACT_OPTION", "AWS Textract service - all PDF types"
)

# PII detection models
NO_REDACTION_PII_OPTION = get_or_create_env_var(
    "NO_REDACTION_PII_OPTION", "Only extract text (no redaction)"
)
LOCAL_PII_OPTION = get_or_create_env_var("LOCAL_PII_OPTION", "Local")
AWS_PII_OPTION = get_or_create_env_var("AWS_PII_OPTION", "AWS Comprehend")

SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS", "True")
)
SHOW_AWS_TEXT_EXTRACTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_TEXT_EXTRACTION_OPTIONS", "True")
)

# Show at least local options if everything mistakenly removed
if not SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS and not SHOW_AWS_TEXT_EXTRACTION_OPTIONS:
    SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS = True

local_model_options = list()
aws_model_options = list()
text_extraction_models = list()

if SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS:
    local_model_options.append(SELECTABLE_TEXT_EXTRACT_OPTION)
    local_model_options.append(TESSERACT_TEXT_EXTRACT_OPTION)

if SHOW_AWS_TEXT_EXTRACTION_OPTIONS:
    aws_model_options.append(TEXTRACT_TEXT_EXTRACT_OPTION)

TEXT_EXTRACTION_MODELS = local_model_options + aws_model_options
DO_INITIAL_TABULAR_DATA_CLEAN = convert_string_to_boolean(
    get_or_create_env_var("DO_INITIAL_TABULAR_DATA_CLEAN", "True")
)

SHOW_LOCAL_PII_DETECTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_PII_DETECTION_OPTIONS", "True")
)
SHOW_AWS_PII_DETECTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_PII_DETECTION_OPTIONS", "True")
)

if not SHOW_LOCAL_PII_DETECTION_OPTIONS and not SHOW_AWS_PII_DETECTION_OPTIONS:
    SHOW_LOCAL_PII_DETECTION_OPTIONS = True

local_model_options = [NO_REDACTION_PII_OPTION]
aws_model_options = list()
pii_detection_models = list()

if SHOW_LOCAL_PII_DETECTION_OPTIONS:
    local_model_options.append(LOCAL_PII_OPTION)

if SHOW_AWS_PII_DETECTION_OPTIONS:
    aws_model_options.append(AWS_PII_OPTION)

PII_DETECTION_MODELS = local_model_options + aws_model_options

if SHOW_AWS_TEXT_EXTRACTION_OPTIONS:
    DEFAULT_TEXT_EXTRACTION_MODEL = get_or_create_env_var(
        "DEFAULT_TEXT_EXTRACTION_MODEL", TEXTRACT_TEXT_EXTRACT_OPTION
    )
else:
    DEFAULT_TEXT_EXTRACTION_MODEL = get_or_create_env_var(
        "DEFAULT_TEXT_EXTRACTION_MODEL", SELECTABLE_TEXT_EXTRACT_OPTION
    )

if SHOW_AWS_PII_DETECTION_OPTIONS:
    DEFAULT_PII_DETECTION_MODEL = get_or_create_env_var(
        "DEFAULT_PII_DETECTION_MODEL", AWS_PII_OPTION
    )
else:
    DEFAULT_PII_DETECTION_MODEL = get_or_create_env_var(
        "DEFAULT_PII_DETECTION_MODEL", LOCAL_PII_OPTION
    )

# Create list of PII detection models for tabular redaction
TABULAR_PII_DETECTION_MODELS = PII_DETECTION_MODELS.copy()
if NO_REDACTION_PII_OPTION in TABULAR_PII_DETECTION_MODELS:
    TABULAR_PII_DETECTION_MODELS.remove(NO_REDACTION_PII_OPTION)

DEFAULT_TEXT_COLUMNS = get_or_create_env_var("DEFAULT_TEXT_COLUMNS", "[]")
DEFAULT_EXCEL_SHEETS = get_or_create_env_var("DEFAULT_EXCEL_SHEETS", "[]")

DEFAULT_TABULAR_ANONYMISATION_STRATEGY = get_or_create_env_var(
    "DEFAULT_TABULAR_ANONYMISATION_STRATEGY", "redact completely"
)

###
# LOCAL OCR MODEL OPTIONS
###


### VLM OPTIONS

SHOW_VLM_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_VLM_MODEL_OPTIONS", "False")
)  # Whether to show the VLM model options in the UI

SELECTED_MODEL = get_or_create_env_var(
    "SELECTED_MODEL", "Dots.OCR"
)  # Selected vision model. Choose from:  "Nanonets-OCR2-3B",  "Dots.OCR", "Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "PaddleOCR-VL"

if SHOW_VLM_MODEL_OPTIONS:
    VLM_MODEL_OPTIONS = [
        SELECTED_MODEL,
    ]

MAX_SPACES_GPU_RUN_TIME = int(
    get_or_create_env_var("MAX_SPACES_GPU_RUN_TIME", "60")
)  # Maximum number of seconds to run the GPU on Spaces

MAX_NEW_TOKENS = int(
    get_or_create_env_var("MAX_NEW_TOKENS", "30")
)  # Maximum number of tokens to generate

DEFAULT_MAX_NEW_TOKENS = int(
    get_or_create_env_var("DEFAULT_MAX_NEW_TOKENS", "30")
)  # Default maximum number of tokens to generate

MAX_INPUT_TOKEN_LENGTH = int(
    get_or_create_env_var("MAX_INPUT_TOKEN_LENGTH", "4096")
)  # Maximum number of tokens to input to the VLM

VLM_MAX_IMAGE_SIZE = int(
    get_or_create_env_var("VLM_MAX_IMAGE_SIZE", "1000000")
)  # Maximum total pixels (width * height) for images passed to VLM. Images with more pixels will be resized while maintaining aspect ratio. Default is 1000000 (1000x1000).

VLM_MAX_DPI = float(
    get_or_create_env_var("VLM_MAX_DPI", "300.0")
)  # Maximum DPI for images passed to VLM. Images with higher DPI will be resized accordingly.

USE_FLASH_ATTENTION = convert_string_to_boolean(
    get_or_create_env_var("USE_FLASH_ATTENTION", "False")
)  # Whether to use flash attention for the VLM

OVERWRITE_EXISTING_OCR_RESULTS = convert_string_to_boolean(
    get_or_create_env_var("OVERWRITE_EXISTING_OCR_RESULTS", "False")
)  # If True, always create new OCR results instead of loading from existing JSON files

### Local OCR model - Tesseract vs PaddleOCR
CHOSEN_LOCAL_OCR_MODEL = get_or_create_env_var(
    "CHOSEN_LOCAL_OCR_MODEL", "tesseract"
)  # "tesseract" is the default and will work for documents with clear typed text. "paddle" is more accurate for text extraction where the text is not clear or well-formatted, but word-level extract is not natively supported, and so word bounding boxes will be inaccurate. The hybrid models will do a first pass with one model, and a second pass on words/phrases with low confidence with a more powerful model. "hybrid-paddle" will do the first pass with Tesseract, and the second with PaddleOCR. "hybrid-vlm" is a combination of Tesseract for OCR, and a second pass with the chosen vision model (VLM). "hybrid-paddle-vlm" is a combination of PaddleOCR with the chosen VLM.

SHOW_LOCAL_OCR_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_OCR_MODEL_OPTIONS", "False")
)

SHOW_PADDLE_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_PADDLE_MODEL_OPTIONS", "False")
)

LOCAL_OCR_MODEL_OPTIONS = ["tesseract"]

paddle_options = ["paddle", "hybrid-paddle"]
if SHOW_PADDLE_MODEL_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.extend(paddle_options)

vlm_options = ["hybrid-vlm"]
if SHOW_VLM_MODEL_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.extend(vlm_options)

if SHOW_PADDLE_MODEL_OPTIONS and SHOW_VLM_MODEL_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.append("hybrid-paddle-vlm")

MODEL_CACHE_PATH = get_or_create_env_var("MODEL_CACHE_PATH", "./model_cache")


HYBRID_OCR_CONFIDENCE_THRESHOLD = int(
    get_or_create_env_var("HYBRID_OCR_CONFIDENCE_THRESHOLD", "80")
)  # The tesseract confidence threshold under which the text will be passed to PaddleOCR for re-extraction using the hybrid OCR method.
HYBRID_OCR_PADDING = int(
    get_or_create_env_var("HYBRID_OCR_PADDING", "1")
)  # The padding to add to the text when passing it to PaddleOCR for re-extraction using the hybrid OCR method.

TESSERACT_WORD_LEVEL_OCR = convert_string_to_boolean(
    get_or_create_env_var("TESSERACT_WORD_LEVEL_OCR", "True")
)  # Whether to use Tesseract word-level OCR.

TESSERACT_SEGMENTATION_LEVEL = int(
    get_or_create_env_var("TESSERACT_SEGMENTATION_LEVEL", "11")
)  # Tesseract segmentation level: PSM level to use for Tesseract OCR

CONVERT_LINE_TO_WORD_LEVEL = convert_string_to_boolean(
    get_or_create_env_var("CONVERT_LINE_TO_WORD_LEVEL", "False")
)  # Whether to convert paddle line-level OCR results to word-level for better precision

LOAD_PADDLE_AT_STARTUP = convert_string_to_boolean(
    get_or_create_env_var("LOAD_PADDLE_AT_STARTUP", "False")
)  # Whether to load the PaddleOCR model at startup.

PADDLE_USE_TEXTLINE_ORIENTATION = convert_string_to_boolean(
    get_or_create_env_var("PADDLE_USE_TEXTLINE_ORIENTATION", "False")
)

PADDLE_DET_DB_UNCLIP_RATIO = float(
    get_or_create_env_var("PADDLE_DET_DB_UNCLIP_RATIO", "1.2")
)

SAVE_EXAMPLE_HYBRID_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_EXAMPLE_HYBRID_IMAGES", "False")
)  # Whether to save example images of Tesseract vs PaddleOCR re-extraction in hybrid OCR mode.

SAVE_PAGE_OCR_VISUALISATIONS = convert_string_to_boolean(
    get_or_create_env_var("SAVE_PAGE_OCR_VISUALISATIONS", "False")
)  # Whether to save visualisations of Tesseract, PaddleOCR, and Textract bounding boxes.

SAVE_WORD_SEGMENTER_OUTPUT_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_WORD_SEGMENTER_OUTPUT_IMAGES", "False")
)  # Whether to save output images from the word segmenter.

# Model storage paths for Lambda compatibility
PADDLE_MODEL_PATH = get_or_create_env_var(
    "PADDLE_MODEL_PATH", ""
)  # Directory for PaddleOCR model storage. Uses default location if not set.

SPACY_MODEL_PATH = get_or_create_env_var(
    "SPACY_MODEL_PATH", ""
)  # Directory for spaCy model storage. Uses default location if not set.

PREPROCESS_LOCAL_OCR_IMAGES = get_or_create_env_var(
    "PREPROCESS_LOCAL_OCR_IMAGES", "True"
)  # Whether to try and preprocess images before extracting text. NOTE: I have found in testing that this doesn't necessarily imporove results, and greatly slows down extraction.

SAVE_PREPROCESS_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_PREPROCESS_IMAGES", "False")
)  # Whether to save the pre-processed images.

SAVE_VLM_INPUT_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_VLM_INPUT_IMAGES", "False")
)  # Whether to save input images sent to VLM OCR for debugging.

# Entities for redaction
CHOSEN_COMPREHEND_ENTITIES = get_or_create_env_var(
    "CHOSEN_COMPREHEND_ENTITIES",
    "['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE', 'PASSPORT_NUMBER','DRIVER_ID', 'USERNAME','PASSWORD', 'IP_ADDRESS','MAC_ADDRESS', 'LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER', 'INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER']",
)

FULL_COMPREHEND_ENTITY_LIST = get_or_create_env_var(
    "FULL_COMPREHEND_ENTITY_LIST",
    "['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE','SSN','DATE_TIME','PASSPORT_NUMBER','DRIVER_ID','URL','AGE','USERNAME','PASSWORD','AWS_ACCESS_KEY','AWS_SECRET_KEY','IP_ADDRESS','MAC_ADDRESS','ALL','LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER','CA_SOCIAL_INSURANCE_NUMBER','US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER','UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER','IN_PERMANENT_ACCOUNT_NUMBER','IN_NREGA','INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER','CA_HEALTH_NUMBER','IN_AADHAAR','IN_VOTER_NUMBER', 'CUSTOM_FUZZY']",
)

# Entities for local PII redaction option
CHOSEN_REDACT_ENTITIES = get_or_create_env_var(
    "CHOSEN_REDACT_ENTITIES",
    "['TITLES', 'PERSON', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'STREETNAME', 'UKPOSTCODE', 'CUSTOM']",
)

FULL_ENTITY_LIST = get_or_create_env_var(
    "FULL_ENTITY_LIST",
    "['TITLES', 'PERSON', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'STREETNAME', 'UKPOSTCODE', 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS', 'CUSTOM', 'CUSTOM_FUZZY']",
)

CUSTOM_ENTITIES = get_or_create_env_var(
    "CUSTOM_ENTITIES", "['TITLES', 'UKPOSTCODE', 'STREETNAME', 'CUSTOM']"
)


DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX = get_or_create_env_var(
    "DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX", "['Extract handwriting']"
)

HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS = get_or_create_env_var(
    "HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS",
    "['Extract handwriting', 'Extract signatures']",
)

if HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS:
    HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS = _get_env_list(
        HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS
    )

INCLUDE_FORM_EXTRACTION_TEXTRACT_OPTION = get_or_create_env_var(
    "INCLUDE_FORM_EXTRACTION_TEXTRACT_OPTION", "False"
)
INCLUDE_LAYOUT_EXTRACTION_TEXTRACT_OPTION = get_or_create_env_var(
    "INCLUDE_LAYOUT_EXTRACTION_TEXTRACT_OPTION", "False"
)
INCLUDE_TABLE_EXTRACTION_TEXTRACT_OPTION = get_or_create_env_var(
    "INCLUDE_TABLE_EXTRACTION_TEXTRACT_OPTION", "False"
)

if INCLUDE_FORM_EXTRACTION_TEXTRACT_OPTION == "True":
    HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS.append("Extract forms")
if INCLUDE_LAYOUT_EXTRACTION_TEXTRACT_OPTION == "True":
    HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS.append("Extract layout")
if INCLUDE_TABLE_EXTRACTION_TEXTRACT_OPTION == "True":
    HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS.append("Extract tables")


DEFAULT_SEARCH_QUERY = get_or_create_env_var("DEFAULT_SEARCH_QUERY", "")
DEFAULT_FUZZY_SPELLING_MISTAKES_NUM = int(
    get_or_create_env_var("DEFAULT_FUZZY_SPELLING_MISTAKES_NUM", "1")
)

DEFAULT_PAGE_MIN = int(get_or_create_env_var("DEFAULT_PAGE_MIN", "0"))

DEFAULT_PAGE_MAX = int(get_or_create_env_var("DEFAULT_PAGE_MAX", "0"))


### Language selection options

SHOW_LANGUAGE_SELECTION = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LANGUAGE_SELECTION", "False")
)

DEFAULT_LANGUAGE_FULL_NAME = get_or_create_env_var(
    "DEFAULT_LANGUAGE_FULL_NAME", "english"
)
DEFAULT_LANGUAGE = get_or_create_env_var(
    "DEFAULT_LANGUAGE", "en"
)  # For tesseract, ensure the Tesseract language data (e.g., fra.traineddata) is installed on your system. You can find the relevant language packs here: https://github.com/tesseract-ocr/tessdata.
# For paddle, ensure the paddle language data (e.g., fra.traineddata) is installed on your system. You can find information on supported languages here: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
# For AWS Comprehend, only English and Spanish are supported https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html ['en', 'es']
# AWS Textract automatically detects the language of the document and supports the following languages: https://aws.amazon.com/textract/faqs/#topic-0. 'English, Spanish, Italian, Portuguese, French, German. Handwriting, Invoices and Receipts, Identity documents and Queries processing are in English only'

textract_language_choices = get_or_create_env_var(
    "textract_language_choices", "['en', 'es', 'fr', 'de', 'it', 'pt']"
)
aws_comprehend_language_choices = get_or_create_env_var(
    "aws_comprehend_language_choices", "['en', 'es']"
)

# The choices that the user sees
MAPPED_LANGUAGE_CHOICES = get_or_create_env_var(
    "MAPPED_LANGUAGE_CHOICES",
    "['english', 'french', 'german', 'spanish', 'italian', 'dutch', 'portuguese', 'chinese', 'japanese', 'korean', 'lithuanian', 'macedonian', 'norwegian_bokmaal', 'polish', 'romanian', 'russian', 'slovenian', 'swedish', 'catalan', 'ukrainian']",
)
LANGUAGE_CHOICES = get_or_create_env_var(
    "LANGUAGE_CHOICES",
    "['en', 'fr', 'de', 'es', 'it', 'nl', 'pt', 'zh', 'ja', 'ko', 'lt', 'mk', 'nb', 'pl', 'ro', 'ru', 'sl', 'sv', 'ca', 'uk']",
)

###
# Duplicate detection settings
###
DEFAULT_DUPLICATE_DETECTION_THRESHOLD = float(
    get_or_create_env_var("DEFAULT_DUPLICATE_DETECTION_THRESHOLD", "0.95")
)
DEFAULT_MIN_CONSECUTIVE_PAGES = int(
    get_or_create_env_var("DEFAULT_MIN_CONSECUTIVE_PAGES", "1")
)
USE_GREEDY_DUPLICATE_DETECTION = convert_string_to_boolean(
    get_or_create_env_var("USE_GREEDY_DUPLICATE_DETECTION", "True")
)
DEFAULT_COMBINE_PAGES = convert_string_to_boolean(
    get_or_create_env_var("DEFAULT_COMBINE_PAGES", "True")
)  # Combine text from the same page number within a file. Alternative will enable line-level duplicate detection.
DEFAULT_MIN_WORD_COUNT = int(get_or_create_env_var("DEFAULT_MIN_WORD_COUNT", "10"))
REMOVE_DUPLICATE_ROWS = convert_string_to_boolean(
    get_or_create_env_var("REMOVE_DUPLICATE_ROWS", "False")
)


###
# File output options
###
# Should the output pdf redaction boxes be drawn using the custom box colour?
USE_GUI_BOX_COLOURS_FOR_OUTPUTS = convert_string_to_boolean(
    get_or_create_env_var("USE_GUI_BOX_COLOURS_FOR_OUTPUTS", "False")
)

# This is the colour of the output pdf redaction boxes. Should be a tuple of three integers between 0 and 255
CUSTOM_BOX_COLOUR = get_or_create_env_var("CUSTOM_BOX_COLOUR", "(0, 0, 0)")

if CUSTOM_BOX_COLOUR == "grey":
    # only "grey" is currently supported as a custom box colour by name, or a tuple of three integers between 0 and 255
    CUSTOM_BOX_COLOUR = (128, 128, 128)
else:
    try:
        components_str = CUSTOM_BOX_COLOUR.strip("()").split(",")
        CUSTOM_BOX_COLOUR = tuple(
            int(c.strip()) for c in components_str
        )  # Always gives a tuple of three integers between 0 and 255
    except Exception as e:
        print(f"Error initialising CUSTOM_BOX_COLOUR: {e}, returning default black")
        CUSTOM_BOX_COLOUR = (
            0,
            0,
            0,
        )  # Default to black if the custom box colour is not a valid tuple of three integers between 0 and 255

# Apply redactions defaults to images, graphics, and text, from: https://pymupdf.readthedocs.io/en/latest/page.html#Page.apply_redactions
# For images, the default is set to 0, to ignore. Text presented in images is effectively removed by the overlapping rectangle shape that becomes an embedded part of the document (see the redact_single_box function in file_redaction.py).
APPLY_REDACTIONS_IMAGES = int(
    get_or_create_env_var("APPLY_REDACTIONS_IMAGES", "0")
)  # The default (2) blanks out overlapping pixels. PDF_REDACT_IMAGE_NONE | 0 ignores, and PDF_REDACT_IMAGE_REMOVE | 1 completely removes images overlapping any redaction annotation. Option PDF_REDACT_IMAGE_REMOVE_UNLESS_INVISIBLE | 3 only removes images that are actually visible.
APPLY_REDACTIONS_GRAPHICS = int(
    get_or_create_env_var("APPLY_REDACTIONS_GRAPHICS", "0")
)  # How to redact overlapping vector graphics (also called “line-art” or “drawings”). The default (2) removes any overlapping vector graphics. PDF_REDACT_LINE_ART_NONE | 0 ignores, and PDF_REDACT_LINE_ART_REMOVE_IF_COVERED | 1 removes graphics fully contained in a redaction annotation.
APPLY_REDACTIONS_TEXT = int(
    get_or_create_env_var("APPLY_REDACTIONS_TEXT", "0")
)  # The default PDF_REDACT_TEXT_REMOVE | 0 removes all characters whose boundary box overlaps any redaction rectangle. This complies with the original legal / data protection intentions of redaction annotations. Other use cases however may require to keep text while redacting vector graphics or images. This can be achieved by setting text=True|PDF_REDACT_TEXT_NONE | 1. This does not comply with the data protection intentions of redaction annotations. Do so at your own risk.

# If you don't want to redact the text, but instead just draw a box over it, set this to True
RETURN_PDF_FOR_REVIEW = convert_string_to_boolean(
    get_or_create_env_var("RETURN_PDF_FOR_REVIEW", "True")
)

RETURN_REDACTED_PDF = convert_string_to_boolean(
    get_or_create_env_var("RETURN_REDACTED_PDF", "True")
)  # Return a redacted PDF at the end of the redaction task. Could be useful to set this to "False" if you want to ensure that the user always goes to the 'Review Redactions' tab before getting the final redacted PDF product.

COMPRESS_REDACTED_PDF = convert_string_to_boolean(
    get_or_create_env_var("COMPRESS_REDACTED_PDF", "False")
)  # On low memory systems, the compression options in pymupdf can cause the app to crash if the PDF is longer than 500 pages or so. Setting this to False will save the PDF only with a basic cleaning option enabled

###
# APP RUN / GUI OPTIONS
###

TLDEXTRACT_CACHE = get_or_create_env_var("TLDEXTRACT_CACHE", "tmp/tld/")
try:
    extract = TLDExtract(cache_dir=TLDEXTRACT_CACHE)
except Exception as e:
    print(f"Error initialising TLDExtract: {e}")
    extract = TLDExtract(cache_dir=None)

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = convert_string_to_boolean(get_or_create_env_var("COGNITO_AUTH", "False"))


# Link to user guide - ensure it is a valid URL
def validate_safe_url(url_candidate: str, allowed_domains: list = None) -> str:
    """
    Validate and return a safe URL with enhanced security checks.
    """
    if allowed_domains is None:
        allowed_domains = [
            "seanpedrick-case.github.io",
            "github.io",
            "github.com",
            "sharepoint.com",
        ]

    try:
        parsed = urllib.parse.urlparse(url_candidate)

        # Basic structure validation
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL structure")

        # Security checks
        if parsed.scheme not in ["https"]:  # Only allow HTTPS
            raise ValueError("Only HTTPS URLs are allowed for security")

        # Domain validation
        domain = parsed.netloc.lower()
        if not any(domain.endswith(allowed) for allowed in allowed_domains):
            raise ValueError(f"Domain not in allowed list: {domain}")

        # Additional security checks
        if any(
            suspicious in domain for suspicious in ["..", "//", "javascript:", "data:"]
        ):
            raise ValueError("Suspicious URL patterns detected")

        # Path validation (prevent path traversal)
        if ".." in parsed.path or "//" in parsed.path:
            raise ValueError("Path traversal attempts detected")

        return url_candidate

    except Exception as e:
        print(f"URL validation failed: {e}")
        return "https://seanpedrick-case.github.io/doc_redaction"  # Safe fallback


USER_GUIDE_URL = validate_safe_url(
    get_or_create_env_var(
        "USER_GUIDE_URL", "https://seanpedrick-case.github.io/doc_redaction"
    )
)

SHOW_EXAMPLES = convert_string_to_boolean(
    get_or_create_env_var("SHOW_EXAMPLES", "True")
)
SHOW_AWS_EXAMPLES = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_EXAMPLES", "False")
)

FILE_INPUT_HEIGHT = int(get_or_create_env_var("FILE_INPUT_HEIGHT", "200"))

RUN_DIRECT_MODE = convert_string_to_boolean(
    get_or_create_env_var("RUN_DIRECT_MODE", "False")
)

# Direct mode configuration options
DIRECT_MODE_DEFAULT_USER = get_or_create_env_var(
    "DIRECT_MODE_DEFAULT_USER", ""
)  # Default username for cli/direct mode requests
DIRECT_MODE_TASK = get_or_create_env_var(
    "DIRECT_MODE_TASK", "redact"
)  # 'redact' or 'deduplicate'
DIRECT_MODE_INPUT_FILE = get_or_create_env_var(
    "DIRECT_MODE_INPUT_FILE", ""
)  # Path to input file
DIRECT_MODE_OUTPUT_DIR = get_or_create_env_var(
    "DIRECT_MODE_OUTPUT_DIR", OUTPUT_FOLDER
)  # Output directory
DIRECT_MODE_DUPLICATE_TYPE = get_or_create_env_var(
    "DIRECT_MODE_DUPLICATE_TYPE", "pages"
)  # 'pages' or 'tabular'

# Additional direct mode configuration options for user customization
DIRECT_MODE_LANGUAGE = get_or_create_env_var(
    "DIRECT_MODE_LANGUAGE", DEFAULT_LANGUAGE
)  # Language for document processing
DIRECT_MODE_PII_DETECTOR = get_or_create_env_var(
    "DIRECT_MODE_PII_DETECTOR", LOCAL_PII_OPTION
)  # PII detection method
DIRECT_MODE_OCR_METHOD = get_or_create_env_var(
    "DIRECT_MODE_OCR_METHOD", "Local OCR"
)  # OCR method for PDF/image processing
DIRECT_MODE_PAGE_MIN = int(
    get_or_create_env_var("DIRECT_MODE_PAGE_MIN", str(DEFAULT_PAGE_MIN))
)  # First page to process
DIRECT_MODE_PAGE_MAX = int(
    get_or_create_env_var("DIRECT_MODE_PAGE_MAX", str(DEFAULT_PAGE_MAX))
)  # Last page to process
DIRECT_MODE_IMAGES_DPI = float(
    get_or_create_env_var("DIRECT_MODE_IMAGES_DPI", str(IMAGES_DPI))
)  # DPI for image processing
DIRECT_MODE_CHOSEN_LOCAL_OCR_MODEL = get_or_create_env_var(
    "DIRECT_MODE_CHOSEN_LOCAL_OCR_MODEL", CHOSEN_LOCAL_OCR_MODEL
)  # Local OCR model choice
DIRECT_MODE_PREPROCESS_LOCAL_OCR_IMAGES = convert_string_to_boolean(
    get_or_create_env_var(
        "DIRECT_MODE_PREPROCESS_LOCAL_OCR_IMAGES", str(PREPROCESS_LOCAL_OCR_IMAGES)
    )
)  # Preprocess images before OCR
DIRECT_MODE_COMPRESS_REDACTED_PDF = convert_string_to_boolean(
    get_or_create_env_var(
        "DIRECT_MODE_COMPRESS_REDACTED_PDF", str(COMPRESS_REDACTED_PDF)
    )
)  # Compress redacted PDF
DIRECT_MODE_RETURN_PDF_END_OF_REDACTION = convert_string_to_boolean(
    get_or_create_env_var(
        "DIRECT_MODE_RETURN_PDF_END_OF_REDACTION", str(RETURN_REDACTED_PDF)
    )
)  # Return PDF at end of redaction
DIRECT_MODE_EXTRACT_FORMS = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_EXTRACT_FORMS", "False")
)  # Extract forms during Textract analysis
DIRECT_MODE_EXTRACT_TABLES = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_EXTRACT_TABLES", "False")
)  # Extract tables during Textract analysis
DIRECT_MODE_EXTRACT_LAYOUT = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_EXTRACT_LAYOUT", "False")
)  # Extract layout during Textract analysis
DIRECT_MODE_EXTRACT_SIGNATURES = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_EXTRACT_SIGNATURES", "False")
)  # Extract signatures during Textract analysis
DIRECT_MODE_MATCH_FUZZY_WHOLE_PHRASE_BOOL = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_MATCH_FUZZY_WHOLE_PHRASE_BOOL", "True")
)  # Match fuzzy whole phrase boolean
DIRECT_MODE_ANON_STRATEGY = get_or_create_env_var(
    "DIRECT_MODE_ANON_STRATEGY", DEFAULT_TABULAR_ANONYMISATION_STRATEGY
)  # Anonymisation strategy for tabular data
DIRECT_MODE_FUZZY_MISTAKES = int(
    get_or_create_env_var(
        "DIRECT_MODE_FUZZY_MISTAKES", str(DEFAULT_FUZZY_SPELLING_MISTAKES_NUM)
    )
)  # Number of fuzzy spelling mistakes allowed
DIRECT_MODE_SIMILARITY_THRESHOLD = float(
    get_or_create_env_var(
        "DIRECT_MODE_SIMILARITY_THRESHOLD", str(DEFAULT_DUPLICATE_DETECTION_THRESHOLD)
    )
)  # Similarity threshold for duplicate detection
DIRECT_MODE_MIN_WORD_COUNT = int(
    get_or_create_env_var("DIRECT_MODE_MIN_WORD_COUNT", str(DEFAULT_MIN_WORD_COUNT))
)  # Minimum word count for duplicate detection
DIRECT_MODE_MIN_CONSECUTIVE_PAGES = int(
    get_or_create_env_var(
        "DIRECT_MODE_MIN_CONSECUTIVE_PAGES", str(DEFAULT_MIN_CONSECUTIVE_PAGES)
    )
)  # Minimum consecutive pages for duplicate detection
DIRECT_MODE_GREEDY_MATCH = convert_string_to_boolean(
    get_or_create_env_var(
        "DIRECT_MODE_GREEDY_MATCH", str(USE_GREEDY_DUPLICATE_DETECTION)
    )
)  # Use greedy matching for duplicate detection
DIRECT_MODE_COMBINE_PAGES = convert_string_to_boolean(
    get_or_create_env_var("DIRECT_MODE_COMBINE_PAGES", str(DEFAULT_COMBINE_PAGES))
)  # Combine pages for duplicate detection
DIRECT_MODE_REMOVE_DUPLICATE_ROWS = convert_string_to_boolean(
    get_or_create_env_var(
        "DIRECT_MODE_REMOVE_DUPLICATE_ROWS", str(REMOVE_DUPLICATE_ROWS)
    )
)  # Remove duplicate rows in tabular data

# Textract Batch Operations Options
DIRECT_MODE_TEXTRACT_ACTION = get_or_create_env_var(
    "DIRECT_MODE_TEXTRACT_ACTION", ""
)  # Textract action for batch operations
DIRECT_MODE_JOB_ID = get_or_create_env_var(
    "DIRECT_MODE_JOB_ID", ""
)  # Job ID for Textract operations

# Lambda-specific configuration options
LAMBDA_POLL_INTERVAL = int(
    get_or_create_env_var("LAMBDA_POLL_INTERVAL", "30")
)  # Polling interval in seconds for Textract job status
LAMBDA_MAX_POLL_ATTEMPTS = int(
    get_or_create_env_var("LAMBDA_MAX_POLL_ATTEMPTS", "120")
)  # Maximum number of polling attempts for Textract job completion
LAMBDA_PREPARE_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("LAMBDA_PREPARE_IMAGES", "True")
)  # Prepare images for OCR processing
LAMBDA_EXTRACT_SIGNATURES = convert_string_to_boolean(
    get_or_create_env_var("LAMBDA_EXTRACT_SIGNATURES", "False")
)  # Extract signatures during Textract analysis
LAMBDA_DEFAULT_USERNAME = get_or_create_env_var(
    "LAMBDA_DEFAULT_USERNAME", "lambda_user"
)  # Default username for Lambda operations


### ALLOW LIST

GET_DEFAULT_ALLOW_LIST = convert_string_to_boolean(
    get_or_create_env_var("GET_DEFAULT_ALLOW_LIST", "False")
)

ALLOW_LIST_PATH = get_or_create_env_var(
    "ALLOW_LIST_PATH", ""
)  # config/default_allow_list.csv

S3_ALLOW_LIST_PATH = get_or_create_env_var(
    "S3_ALLOW_LIST_PATH", ""
)  # default_allow_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

if ALLOW_LIST_PATH:
    OUTPUT_ALLOW_LIST_PATH = ALLOW_LIST_PATH
else:
    OUTPUT_ALLOW_LIST_PATH = "config/default_allow_list.csv"

### DENY LIST

GET_DEFAULT_DENY_LIST = convert_string_to_boolean(
    get_or_create_env_var("GET_DEFAULT_DENY_LIST", "False")
)

S3_DENY_LIST_PATH = get_or_create_env_var(
    "S3_DENY_LIST_PATH", ""
)  # default_deny_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

DENY_LIST_PATH = get_or_create_env_var(
    "DENY_LIST_PATH", ""
)  # config/default_deny_list.csv

if DENY_LIST_PATH:
    OUTPUT_DENY_LIST_PATH = DENY_LIST_PATH
else:
    OUTPUT_DENY_LIST_PATH = "config/default_deny_list.csv"

### WHOLE PAGE REDACTION LIST

GET_DEFAULT_WHOLE_PAGE_REDACTION_LIST = get_or_create_env_var(
    "GET_DEFAULT_WHOLE_PAGE_REDACTION_LIST", "False"
)

S3_WHOLE_PAGE_REDACTION_LIST_PATH = get_or_create_env_var(
    "S3_WHOLE_PAGE_REDACTION_LIST_PATH", ""
)  # default_whole_page_redaction_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

WHOLE_PAGE_REDACTION_LIST_PATH = get_or_create_env_var(
    "WHOLE_PAGE_REDACTION_LIST_PATH", ""
)  # config/default_whole_page_redaction_list.csv

if WHOLE_PAGE_REDACTION_LIST_PATH:
    OUTPUT_WHOLE_PAGE_REDACTION_LIST_PATH = WHOLE_PAGE_REDACTION_LIST_PATH
else:
    OUTPUT_WHOLE_PAGE_REDACTION_LIST_PATH = (
        "config/default_whole_page_redaction_list.csv"
    )

###
# COST CODE OPTIONS
###

SHOW_COSTS = convert_string_to_boolean(get_or_create_env_var("SHOW_COSTS", "False"))

GET_COST_CODES = convert_string_to_boolean(
    get_or_create_env_var("GET_COST_CODES", "False")
)

DEFAULT_COST_CODE = get_or_create_env_var("DEFAULT_COST_CODE", "")

COST_CODES_PATH = get_or_create_env_var(
    "COST_CODES_PATH", ""
)  # 'config/COST_CENTRES.csv' # file should be a csv file with a single table in it that has two columns with a header. First column should contain cost codes, second column should contain a name or description for the cost code

S3_COST_CODES_PATH = get_or_create_env_var(
    "S3_COST_CODES_PATH", ""
)  # COST_CENTRES.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

# A default path in case s3 cost code location is provided but no local cost code location given
if COST_CODES_PATH:
    OUTPUT_COST_CODES_PATH = COST_CODES_PATH
else:
    OUTPUT_COST_CODES_PATH = "config/cost_codes.csv"

ENFORCE_COST_CODES = convert_string_to_boolean(
    get_or_create_env_var("ENFORCE_COST_CODES", "False")
)
# If you have cost codes listed, is it compulsory to choose one before redacting?

if ENFORCE_COST_CODES:
    GET_COST_CODES = True


###
# WHOLE DOCUMENT API OPTIONS
###

SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS", "False")
)  # This feature not currently implemented

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET = get_or_create_env_var(
    "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET", ""
)

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER = get_or_create_env_var(
    "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER", "input"
)

TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER = get_or_create_env_var(
    "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER", "output"
)

LOAD_PREVIOUS_TEXTRACT_JOBS_S3 = convert_string_to_boolean(
    get_or_create_env_var("LOAD_PREVIOUS_TEXTRACT_JOBS_S3", "False")
)
# Whether or not to load previous Textract jobs from S3

TEXTRACT_JOBS_S3_LOC = get_or_create_env_var(
    "TEXTRACT_JOBS_S3_LOC", "output"
)  # Subfolder in the DOCUMENT_REDACTION_BUCKET where the Textract jobs are stored

TEXTRACT_JOBS_S3_INPUT_LOC = get_or_create_env_var(
    "TEXTRACT_JOBS_S3_INPUT_LOC", "input"
)  # Subfolder in the DOCUMENT_REDACTION_BUCKET where the Textract jobs are stored

TEXTRACT_JOBS_LOCAL_LOC = get_or_create_env_var(
    "TEXTRACT_JOBS_LOCAL_LOC", "output"
)  # Local subfolder where the Textract jobs are stored

DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS = int(
    get_or_create_env_var("DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS", "7")
)  # How many days into the past should whole document Textract jobs be displayed? After that, the data is not deleted from the Textract jobs csv, but it is just filtered out. Included to align with S3 buckets where the file outputs will be automatically deleted after X days.


###
# Config vars output format
###

# Convert string environment variables to string or list
CSV_ACCESS_LOG_HEADERS = _get_env_list(CSV_ACCESS_LOG_HEADERS)
CSV_FEEDBACK_LOG_HEADERS = _get_env_list(CSV_FEEDBACK_LOG_HEADERS)
CSV_USAGE_LOG_HEADERS = _get_env_list(CSV_USAGE_LOG_HEADERS)

DYNAMODB_ACCESS_LOG_HEADERS = _get_env_list(DYNAMODB_ACCESS_LOG_HEADERS)
DYNAMODB_FEEDBACK_LOG_HEADERS = _get_env_list(DYNAMODB_FEEDBACK_LOG_HEADERS)
DYNAMODB_USAGE_LOG_HEADERS = _get_env_list(DYNAMODB_USAGE_LOG_HEADERS)
if CHOSEN_COMPREHEND_ENTITIES:
    CHOSEN_COMPREHEND_ENTITIES = _get_env_list(CHOSEN_COMPREHEND_ENTITIES)
if FULL_COMPREHEND_ENTITY_LIST:
    FULL_COMPREHEND_ENTITY_LIST = _get_env_list(FULL_COMPREHEND_ENTITY_LIST)
if CHOSEN_REDACT_ENTITIES:
    CHOSEN_REDACT_ENTITIES = _get_env_list(CHOSEN_REDACT_ENTITIES)
if FULL_ENTITY_LIST:
    FULL_ENTITY_LIST = _get_env_list(FULL_ENTITY_LIST)

if DEFAULT_TEXT_COLUMNS:
    DEFAULT_TEXT_COLUMNS = _get_env_list(DEFAULT_TEXT_COLUMNS)
if DEFAULT_EXCEL_SHEETS:
    DEFAULT_EXCEL_SHEETS = _get_env_list(DEFAULT_EXCEL_SHEETS)

if CUSTOM_ENTITIES:
    CUSTOM_ENTITIES = _get_env_list(CUSTOM_ENTITIES)

if DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX:
    DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX = _get_env_list(
        DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX
    )

if ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = _get_env_list(ALLOWED_ORIGINS)

if ALLOWED_HOSTS:
    ALLOWED_HOSTS = _get_env_list(ALLOWED_HOSTS)
