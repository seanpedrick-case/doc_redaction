import logging
import os
import re
import socket
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List

import bleach
from dotenv import load_dotenv
from tldextract import TLDExtract

from tools.secure_path_utils import (
    secure_file_read,
    secure_path_join,
    validate_path_safety,
)

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


def ensure_folder_within_app_directory(
    folder_path: str, app_base_dir: str = None
) -> str:
    """
    Ensure that a folder path is within the app directory for security.

    This function validates that user-defined folder paths are contained within
    the app directory to prevent path traversal attacks and ensure data isolation.

    Args:
        folder_path: The folder path to validate and normalize
        app_base_dir: The base directory of the app (defaults to current working directory)

    Returns:
        A normalized folder path that is guaranteed to be within the app directory

    Raises:
        ValueError: If the path cannot be safely contained within the app directory
    """
    if not folder_path or not folder_path.strip():
        return folder_path

    # Get the app base directory (where the app is run from)
    if app_base_dir is None:
        app_base_dir = os.getcwd()

    app_base_dir = Path(app_base_dir).resolve()
    folder_path = folder_path.strip()

    # Preserve trailing separator preference
    has_trailing_sep = folder_path.endswith(("/", "\\"))

    # Handle special case for "TEMP" - this is handled separately in the code
    if folder_path == "TEMP":
        return folder_path

    # Handle absolute paths
    if os.path.isabs(folder_path):
        folder_path_resolved = Path(folder_path).resolve()
        # Check if the absolute path is within the app directory
        try:
            folder_path_resolved.relative_to(app_base_dir)
            # Path is already within app directory, return it normalized
            result = str(folder_path_resolved)
            if has_trailing_sep and not result.endswith(os.sep):
                result = result + os.sep
            return result
        except ValueError:
            # Path is outside app directory - this is a security issue
            # For system paths like /usr/share/tessdata, we'll allow them but log a warning
            # For other absolute paths outside app directory, we'll raise an error
            normalized_path = os.path.normpath(folder_path).lower()
            system_path_prefixes = [
                "/usr",
                "/opt",
                "/var",
                "/etc",
                "/tmp",
            ]
            if any(
                normalized_path.startswith(prefix) for prefix in system_path_prefixes
            ):
                # System paths are allowed but we log a warning
                print(
                    f"Warning: Using system path outside app directory: {folder_path}"
                )
                return folder_path
            else:
                raise ValueError(
                    f"Folder path '{folder_path}' is outside the app directory '{app_base_dir}'. "
                    f"For security, all user-defined folder paths must be within the app directory."
                )

    # Handle relative paths - ensure they're within app directory
    try:
        # Use secure_path_join to safely join and validate
        # This will prevent path traversal attacks (e.g., "../../etc/passwd")
        safe_path = secure_path_join(app_base_dir, folder_path)
        result = str(safe_path)
        if has_trailing_sep and not result.endswith(os.sep):
            result = result + os.sep
        return result
    except (PermissionError, ValueError) as e:
        # If path contains dangerous patterns, sanitize and try again
        # Extract just the folder name from the path to prevent traversal
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        if folder_name:
            safe_path = secure_path_join(app_base_dir, folder_name)
            result = str(safe_path)
            if has_trailing_sep and not result.endswith(os.sep):
                result = result + os.sep
            print(
                f"Warning: Sanitized folder path '{folder_path}' to '{result}' for security"
            )
            return result
        else:
            raise ValueError(
                f"Cannot safely normalize folder path: {folder_path}"
            ) from e


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


def sanitize_markdown_text(text: str) -> str:
    """
    Sanitize markdown text by removing dangerous HTML/scripts while preserving
    safe markdown syntax.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove dangerous HTML tags and scripts using bleach
    # Define allowed tags for markdown (customize as needed)
    allowed_tags = [
        "a",
        "b",
        "strong",
        "em",
        "i",
        "u",
        "code",
        "pre",
        "blockquote",
        "ul",
        "ol",
        "li",
        "p",
        "br",
        "hr",
    ]
    allowed_attributes = {"a": ["href", "title", "rel"]}
    # Clean the text to strip (remove) any tags not in allowed_tags, and remove all script/iframe/etc.
    text = bleach.clean(
        text, tags=allowed_tags, attributes=allowed_attributes, strip=True
    )

    # Remove iframe, object, embed tags (should already be stripped, but keep for redundancy)
    text = re.sub(
        r"<(iframe|object|embed)[^>]*>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove event handlers (onclick, onerror, etc.)
    text = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', "", text, flags=re.IGNORECASE)

    # Remove javascript: and data: URLs from markdown links
    text = re.sub(
        r"\[([^\]]+)\]\(javascript:[^\)]+\)", r"[\1]", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\[([^\]]+)\]\(data:[^\)]+\)", r"[\1]", text, flags=re.IGNORECASE)

    # Remove dangerous HTML attributes
    text = re.sub(
        r'\s*(style|onerror|onload|onclick)\s*=\s*["\'][^"\']*["\']',
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


###
# LOAD CONFIG FROM ENV FILE
###

CONFIG_FOLDER = get_or_create_env_var("CONFIG_FOLDER", "config/")
CONFIG_FOLDER = ensure_folder_within_app_directory(CONFIG_FOLDER)

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

MAX_SPACES_GPU_RUN_TIME = int(
    get_or_create_env_var("MAX_SPACES_GPU_RUN_TIME", "60")
)  # Maximum number of seconds to run the GPU on Spaces

###
# File I/O options
###

SESSION_OUTPUT_FOLDER = convert_string_to_boolean(
    get_or_create_env_var("SESSION_OUTPUT_FOLDER", "False")
)  # i.e. do you want your input and output folders saved within a subfolder based on session hash value within output/input folders

OUTPUT_FOLDER = get_or_create_env_var("GRADIO_OUTPUT_FOLDER", "output/")  # 'output/'
INPUT_FOLDER = get_or_create_env_var("GRADIO_INPUT_FOLDER", "input/")  # 'input/'

# Whether to automatically upload redaction outputs to S3
SAVE_OUTPUTS_TO_S3 = convert_string_to_boolean(
    get_or_create_env_var("SAVE_OUTPUTS_TO_S3", "False")
)

# Base S3 folder (key prefix) for saving redaction outputs within the DOCUMENT_REDACTION_BUCKET.
# If left blank, S3 uploads for outputs will be skipped even if SAVE_OUTPUTS_TO_S3 is True.
S3_OUTPUTS_FOLDER = get_or_create_env_var("S3_OUTPUTS_FOLDER", "")

S3_OUTPUTS_BUCKET = get_or_create_env_var(
    "S3_OUTPUTS_BUCKET", DOCUMENT_REDACTION_BUCKET
)

# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP":
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")

        if OUTPUT_FOLDER == "TEMP":
            OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP":
            INPUT_FOLDER = temp_dir + "/"
else:
    # Ensure folders are within app directory (skip validation for TEMP as it's handled above)
    OUTPUT_FOLDER = ensure_folder_within_app_directory(OUTPUT_FOLDER)
    INPUT_FOLDER = ensure_folder_within_app_directory(INPUT_FOLDER)

GRADIO_TEMP_DIR = get_or_create_env_var(
    "GRADIO_TEMP_DIR", ""
)  # Default Gradio temp folder
if GRADIO_TEMP_DIR:
    GRADIO_TEMP_DIR = ensure_folder_within_app_directory(GRADIO_TEMP_DIR)
MPLCONFIGDIR = get_or_create_env_var("MPLCONFIGDIR", "")  # Matplotlib cache folder
if MPLCONFIGDIR:
    MPLCONFIGDIR = ensure_folder_within_app_directory(MPLCONFIGDIR)

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

# Ensure log folders are within app directory before adding subfolders
FEEDBACK_LOGS_FOLDER = ensure_folder_within_app_directory(FEEDBACK_LOGS_FOLDER)
ACCESS_LOGS_FOLDER = ensure_folder_within_app_directory(ACCESS_LOGS_FOLDER)
USAGE_LOGS_FOLDER = ensure_folder_within_app_directory(USAGE_LOGS_FOLDER)

if USE_LOG_SUBFOLDERS:
    day_log_subfolder = today_rev + "/"
    host_name_subfolder = HOST_NAME + "/"
    full_log_subfolder = day_log_subfolder + host_name_subfolder

    FEEDBACK_LOGS_FOLDER = FEEDBACK_LOGS_FOLDER + full_log_subfolder
    ACCESS_LOGS_FOLDER = ACCESS_LOGS_FOLDER + full_log_subfolder
    USAGE_LOGS_FOLDER = USAGE_LOGS_FOLDER + full_log_subfolder

    # Re-validate after adding subfolders to ensure still within app directory
    FEEDBACK_LOGS_FOLDER = ensure_folder_within_app_directory(FEEDBACK_LOGS_FOLDER)
    ACCESS_LOGS_FOLDER = ensure_folder_within_app_directory(ACCESS_LOGS_FOLDER)
    USAGE_LOGS_FOLDER = ensure_folder_within_app_directory(USAGE_LOGS_FOLDER)

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
    '["session_hash_textbox", "doc_full_file_name_textbox", "data_full_file_name_textbox", "actual_time_taken_number",	"total_page_count",	"textract_query_number", "pii_detection_method", "comprehend_query_number",  "cost_code", "textract_handwriting_signature", "host_name_textbox", "text_extraction_method", "is_this_a_textract_api_call", "task", "vlm_model_name", "vlm_total_input_tokens", "vlm_total_output_tokens", "llm_model_name", "llm_total_input_tokens", "llm_total_output_tokens"]',
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

MAX_QUEUE_SIZE = int(get_or_create_env_var("MAX_QUEUE_SIZE", "20"))

MAX_FILE_SIZE = get_or_create_env_var("MAX_FILE_SIZE", "250mb").lower()

GRADIO_SERVER_NAME = get_or_create_env_var(
    "GRADIO_SERVER_NAME", "127.0.0.1"
)  # Use "0.0.0.0" for external access

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
if TESSERACT_FOLDER:
    TESSERACT_FOLDER = ensure_folder_within_app_directory(TESSERACT_FOLDER)
    add_folder_to_path(TESSERACT_FOLDER)

TESSERACT_DATA_FOLDER = get_or_create_env_var(
    "TESSERACT_DATA_FOLDER", "/usr/share/tessdata"
)
# Only validate if it's a relative path (system paths like /usr/share/tessdata are allowed)
if TESSERACT_DATA_FOLDER and not os.path.isabs(TESSERACT_DATA_FOLDER):
    TESSERACT_DATA_FOLDER = ensure_folder_within_app_directory(TESSERACT_DATA_FOLDER)

POPPLER_FOLDER = get_or_create_env_var(
    "POPPLER_FOLDER", ""
)  # If installing on Windows,install Poppler from here https://github.com/oschwartz10612/poppler-windows. This variable needs to point to the poppler bin folder e.g. poppler/poppler-24.02.0/Library/bin/
if POPPLER_FOLDER:
    POPPLER_FOLDER = ensure_folder_within_app_directory(POPPLER_FOLDER)
    add_folder_to_path(POPPLER_FOLDER)

SHOW_QUICKSTART = convert_string_to_boolean(
    get_or_create_env_var("SHOW_QUICKSTART", "False")
)

SHOW_SUMMARISATION = convert_string_to_boolean(
    get_or_create_env_var("SHOW_SUMMARISATION", "False")
)

# Extraction and PII options open by default:
EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT = convert_string_to_boolean(
    get_or_create_env_var("EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT", "True")
)

### VLM model options and display

# List of models to use for text extraction
SELECTABLE_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "SELECTABLE_TEXT_EXTRACT_OPTION", "Local model - selectable text"
)
TESSERACT_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "TESSERACT_TEXT_EXTRACT_OPTION", "Local OCR model - PDFs without selectable text"
)
TEXTRACT_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "TEXTRACT_TEXT_EXTRACT_OPTION", "AWS Textract service - all PDF types"
)
BEDROCK_VLM_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "BEDROCK_VLM_TEXT_EXTRACT_OPTION", "AWS Bedrock VLM OCR - all PDF types"
)
GEMINI_VLM_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "GEMINI_VLM_TEXT_EXTRACT_OPTION", "Google Gemini VLM OCR - all PDF types"
)
AZURE_OPENAI_VLM_TEXT_EXTRACT_OPTION = get_or_create_env_var(
    "AZURE_OPENAI_VLM_TEXT_EXTRACT_OPTION", "Azure/OpenAI VLM OCR - all PDF types"
)

SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS", "True")
)
SHOW_AWS_TEXT_EXTRACTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_TEXT_EXTRACTION_OPTIONS", "True")
)
SHOW_BEDROCK_VLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_BEDROCK_VLM_MODELS", "False")
)
SHOW_GEMINI_VLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_GEMINI_VLM_MODELS", "False")
)
SHOW_AZURE_OPENAI_VLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AZURE_OPENAI_VLM_MODELS", "False")
)

# Show at least local options if everything mistakenly removed
if (
    not SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS
    and not SHOW_AWS_TEXT_EXTRACTION_OPTIONS
    and not SHOW_BEDROCK_VLM_MODELS
    and not SHOW_GEMINI_VLM_MODELS
    and not SHOW_AZURE_OPENAI_VLM_MODELS
):
    SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS = True

local_text_extraction_model_options = list()
aws_text_extraction_model_options = list()
cloud_vlm_model_options = list()

if SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS:
    local_text_extraction_model_options.append(SELECTABLE_TEXT_EXTRACT_OPTION)
    local_text_extraction_model_options.append(TESSERACT_TEXT_EXTRACT_OPTION)

if SHOW_AWS_TEXT_EXTRACTION_OPTIONS:
    aws_text_extraction_model_options.append(TEXTRACT_TEXT_EXTRACT_OPTION)

if SHOW_BEDROCK_VLM_MODELS:
    cloud_vlm_model_options.append(BEDROCK_VLM_TEXT_EXTRACT_OPTION)

if SHOW_GEMINI_VLM_MODELS:
    cloud_vlm_model_options.append(GEMINI_VLM_TEXT_EXTRACT_OPTION)

if SHOW_AZURE_OPENAI_VLM_MODELS:
    cloud_vlm_model_options.append(AZURE_OPENAI_VLM_TEXT_EXTRACT_OPTION)

TEXT_EXTRACTION_MODELS = (
    local_text_extraction_model_options
    + aws_text_extraction_model_options
    + cloud_vlm_model_options
)
DO_INITIAL_TABULAR_DATA_CLEAN = convert_string_to_boolean(
    get_or_create_env_var("DO_INITIAL_TABULAR_DATA_CLEAN", "True")
)

### PII model options and display

# PII detection models
NO_REDACTION_PII_OPTION = get_or_create_env_var(
    "NO_REDACTION_PII_OPTION", "Only extract text (no redaction)"
)
LOCAL_PII_OPTION = get_or_create_env_var("LOCAL_PII_OPTION", "Local")
AWS_PII_OPTION = get_or_create_env_var("AWS_PII_OPTION", "AWS Comprehend")
AWS_LLM_PII_OPTION = get_or_create_env_var("AWS_LLM_PII_OPTION", "LLM (AWS Bedrock)")
INFERENCE_SERVER_PII_OPTION = get_or_create_env_var(
    "INFERENCE_SERVER_PII_OPTION", "Local inference server"
)
LOCAL_TRANSFORMERS_LLM_PII_OPTION = get_or_create_env_var(
    "LOCAL_TRANSFORMERS_LLM_PII_OPTION", "Local transformers LLM"
)

SHOW_LOCAL_PII_DETECTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_PII_DETECTION_OPTIONS", "True")
)
SHOW_AWS_PII_DETECTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_PII_DETECTION_OPTIONS", "True")
)
SHOW_INFERENCE_SERVER_PII_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_INFERENCE_SERVER_PII_OPTIONS", "False")
)
SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS", "False")
)
SHOW_AWS_BEDROCK_LLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_BEDROCK_LLM_MODELS", "False")
)


if (
    not SHOW_LOCAL_PII_DETECTION_OPTIONS
    and not SHOW_AWS_PII_DETECTION_OPTIONS
    and not SHOW_AWS_BEDROCK_LLM_MODELS
    and not SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS
    and not SHOW_INFERENCE_SERVER_PII_OPTIONS
    and not SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS
):
    SHOW_LOCAL_PII_DETECTION_OPTIONS = True

local_pii_model_options = [NO_REDACTION_PII_OPTION]
aws_pii_model_options = list()

if SHOW_LOCAL_PII_DETECTION_OPTIONS:
    local_pii_model_options.append(LOCAL_PII_OPTION)

if SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS:
    local_pii_model_options.append(LOCAL_TRANSFORMERS_LLM_PII_OPTION)

if SHOW_INFERENCE_SERVER_PII_OPTIONS:
    local_pii_model_options.append(INFERENCE_SERVER_PII_OPTION)

if SHOW_AWS_PII_DETECTION_OPTIONS:
    aws_pii_model_options.append(AWS_PII_OPTION)

if SHOW_AWS_BEDROCK_LLM_MODELS:
    aws_pii_model_options.append(AWS_LLM_PII_OPTION)

PII_DETECTION_MODELS = local_pii_model_options + aws_pii_model_options

if SHOW_AWS_TEXT_EXTRACTION_OPTIONS:
    DEFAULT_TEXT_EXTRACTION_MODEL = get_or_create_env_var(
        "DEFAULT_TEXT_EXTRACTION_MODEL", TEXTRACT_TEXT_EXTRACT_OPTION
    )
else:
    DEFAULT_TEXT_EXTRACTION_MODEL = get_or_create_env_var(
        "DEFAULT_TEXT_EXTRACTION_MODEL", SELECTABLE_TEXT_EXTRACT_OPTION
    )

# Validate that DEFAULT_TEXT_EXTRACTION_MODEL is in the available choices
# If not, fall back to the first available option
if DEFAULT_TEXT_EXTRACTION_MODEL not in TEXT_EXTRACTION_MODELS:
    if TEXT_EXTRACTION_MODELS:
        DEFAULT_TEXT_EXTRACTION_MODEL = TEXT_EXTRACTION_MODELS[0]
        print(
            f"Warning: DEFAULT_TEXT_EXTRACTION_MODEL was not in available choices. "
            f"Using '{DEFAULT_TEXT_EXTRACTION_MODEL}' instead."
        )
    else:
        # This should never happen, but provide a fallback
        DEFAULT_TEXT_EXTRACTION_MODEL = SELECTABLE_TEXT_EXTRACT_OPTION
        print("Warning: No text extraction models available. Using default option.")

if SHOW_AWS_PII_DETECTION_OPTIONS:
    DEFAULT_PII_DETECTION_MODEL = get_or_create_env_var(
        "DEFAULT_PII_DETECTION_MODEL", AWS_PII_OPTION
    )
else:
    DEFAULT_PII_DETECTION_MODEL = get_or_create_env_var(
        "DEFAULT_PII_DETECTION_MODEL", LOCAL_PII_OPTION
    )

# Validate that DEFAULT_PII_DETECTION_MODEL is in the available choices
# If not, fall back to the first available option
if DEFAULT_PII_DETECTION_MODEL not in PII_DETECTION_MODELS:
    if PII_DETECTION_MODELS:
        DEFAULT_PII_DETECTION_MODEL = PII_DETECTION_MODELS[0]
        print(
            f"Warning: DEFAULT_PII_DETECTION_MODEL was not in available choices. "
            f"Using '{DEFAULT_PII_DETECTION_MODEL}' instead."
        )
    else:
        # This should never happen, but provide a fallback
        DEFAULT_PII_DETECTION_MODEL = LOCAL_PII_OPTION
        print("Warning: No PII detection models available. Using default option.")

SHOW_PII_IDENTIFICATION_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_PII_IDENTIFICATION_OPTIONS", "True")
)

# LLM inference method for PII detection (similar to VLM options)
# Options: "aws-bedrock", "local", "inference-server", "azure-openai", "gemini"
CHOSEN_LLM_PII_INFERENCE_METHOD = get_or_create_env_var(
    "CHOSEN_LLM_PII_INFERENCE_METHOD", "aws-bedrock"
)  # Default to AWS Bedrock for backward compatibility

SHOW_LOCAL_LLM_PII_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_LLM_PII_OPTIONS", "False")
)  # Whether to show local LLM options for PII detection

SHOW_INFERENCE_SERVER_LLM_PII_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_INFERENCE_SERVER_LLM_PII_OPTIONS", "False")
)  # Whether to show inference-server options for PII detection

SHOW_AZURE_LLM_PII_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AZURE_LLM_PII_OPTIONS", "False")
)  # Whether to show Azure/OpenAI options for PII detection

SHOW_GEMINI_LLM_PII_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_GEMINI_LLM_PII_OPTIONS", "False")
)  # Whether to show Gemini options for PII detection

# Build list of available LLM inference methods for PII detection
LLM_PII_INFERENCE_METHODS = []  # Always available

if SHOW_LOCAL_LLM_PII_OPTIONS:
    LLM_PII_INFERENCE_METHODS.append("local")

if SHOW_INFERENCE_SERVER_LLM_PII_OPTIONS:
    LLM_PII_INFERENCE_METHODS.append("inference-server")

if SHOW_AZURE_LLM_PII_OPTIONS:
    LLM_PII_INFERENCE_METHODS.append("azure-openai")

if SHOW_GEMINI_LLM_PII_OPTIONS:
    LLM_PII_INFERENCE_METHODS.append("gemini")

if SHOW_AWS_PII_DETECTION_OPTIONS:
    LLM_PII_INFERENCE_METHODS.append("aws-bedrock")

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

SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL = get_or_create_env_var(
    "SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL", "Qwen3-VL-8B-Instruct"
)  # Selected vision model. Choose from:  "Nanonets-OCR2-3B",  "Dots.OCR", "Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-235B-A22B-Instruct", "PaddleOCR-VL"

if SHOW_VLM_MODEL_OPTIONS:
    VLM_MODEL_OPTIONS = [
        SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL,
    ]


MAX_NEW_TOKENS = int(
    get_or_create_env_var("MAX_NEW_TOKENS", "4096")
)  # Maximum number of tokens to generate

DEFAULT_MAX_NEW_TOKENS = int(
    get_or_create_env_var("DEFAULT_MAX_NEW_TOKENS", "4096")
)  # Default maximum number of tokens to generate

HYBRID_OCR_MAX_NEW_TOKENS = int(
    get_or_create_env_var("HYBRID_OCR_MAX_NEW_TOKENS", "30")
)  # Maximum number of tokens to generate for hybrid OCR

MAX_INPUT_TOKEN_LENGTH = int(
    get_or_create_env_var("MAX_INPUT_TOKEN_LENGTH", "8192")
)  # Maximum number of tokens to input to the VLM

VLM_MAX_IMAGE_SIZE = int(
    get_or_create_env_var("VLM_MAX_IMAGE_SIZE", "819200")
)  # Maximum total pixels (width * height) for images passed to VLM, as a multiple of 32*32 for Qwen3-VL. Images with more pixels will be resized while maintaining aspect ratio. Default is 819200 (800*32*32).

VLM_MIN_IMAGE_SIZE = int(
    get_or_create_env_var("VLM_MIN_IMAGE_SIZE", "614400")
)  # Minimum total pixels (width * height) for images passed to VLM, as a multiple of 32*32 for Qwen3-VL. Images with less pixels will be resized while maintaining aspect ratio. Default is 614400 (600*32*32).

VLM_MAX_DPI = float(
    get_or_create_env_var("VLM_MAX_DPI", "300.0")
)  # Maximum DPI for images passed to VLM. Images with higher DPI will be resized accordingly.

USE_FLASH_ATTENTION = convert_string_to_boolean(
    get_or_create_env_var("USE_FLASH_ATTENTION", "False")
)  # Whether to use flash attention for the VLM

QUANTISE_VLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("QUANTISE_VLM_MODELS", "False")
)  # Whether to use 4-bit quantisation (bitsandbytes) for VLM models. Only applies when SHOW_VLM_MODEL_OPTIONS is True.

REPORT_VLM_OUTPUTS_TO_GUI = convert_string_to_boolean(
    get_or_create_env_var("REPORT_VLM_OUTPUTS_TO_GUI", "False")
)  # Whether to report VLM outputs to the GUI with info boxes as they are processed..

OVERWRITE_EXISTING_OCR_RESULTS = convert_string_to_boolean(
    get_or_create_env_var("OVERWRITE_EXISTING_OCR_RESULTS", "False")
)  # If True, always create new OCR results instead of loading from existing JSON files

# VLM generation parameter defaults
# If empty, these will be None and model defaults will be used instead
VLM_SEED = get_or_create_env_var(
    "VLM_SEED", ""
)  # Random seed for VLM generation. If empty, no seed is set (non-deterministic). If set to an integer, generation will be deterministic.
if VLM_SEED and VLM_SEED.strip():
    VLM_SEED = int(VLM_SEED)
else:
    VLM_SEED = None

VLM_DEFAULT_TEMPERATURE = get_or_create_env_var(
    "VLM_DEFAULT_TEMPERATURE", ""
)  # Default temperature for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_TEMPERATURE and VLM_DEFAULT_TEMPERATURE.strip():
    VLM_DEFAULT_TEMPERATURE = float(VLM_DEFAULT_TEMPERATURE)
else:
    VLM_DEFAULT_TEMPERATURE = None

VLM_DEFAULT_TOP_P = get_or_create_env_var(
    "VLM_DEFAULT_TOP_P", ""
)  # Default top_p (nucleus sampling) for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_TOP_P and VLM_DEFAULT_TOP_P.strip():
    VLM_DEFAULT_TOP_P = float(VLM_DEFAULT_TOP_P)
else:
    VLM_DEFAULT_TOP_P = None

VLM_DEFAULT_MIN_P = get_or_create_env_var(
    "VLM_DEFAULT_MIN_P", ""
)  # Default min_p (minimum probability threshold) for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_MIN_P and VLM_DEFAULT_MIN_P.strip():
    VLM_DEFAULT_MIN_P = float(VLM_DEFAULT_MIN_P)
else:
    VLM_DEFAULT_MIN_P = None

VLM_DEFAULT_TOP_K = get_or_create_env_var(
    "VLM_DEFAULT_TOP_K", ""
)  # Default top_k for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_TOP_K and VLM_DEFAULT_TOP_K.strip():
    VLM_DEFAULT_TOP_K = int(VLM_DEFAULT_TOP_K)
else:
    VLM_DEFAULT_TOP_K = None

VLM_DEFAULT_REPETITION_PENALTY = get_or_create_env_var(
    "VLM_DEFAULT_REPETITION_PENALTY", ""
)  # Default repetition penalty for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_REPETITION_PENALTY and VLM_DEFAULT_REPETITION_PENALTY.strip():
    VLM_DEFAULT_REPETITION_PENALTY = float(VLM_DEFAULT_REPETITION_PENALTY)
else:
    VLM_DEFAULT_REPETITION_PENALTY = None

VLM_DEFAULT_DO_SAMPLE = get_or_create_env_var(
    "VLM_DEFAULT_DO_SAMPLE", ""
)  # Default do_sample setting for VLM generation. If empty, model-specific defaults will be used. True means use sampling, False means use greedy decoding (do_sample=False).
if VLM_DEFAULT_DO_SAMPLE and VLM_DEFAULT_DO_SAMPLE.strip():
    VLM_DEFAULT_DO_SAMPLE = convert_string_to_boolean(VLM_DEFAULT_DO_SAMPLE)
else:
    VLM_DEFAULT_DO_SAMPLE = None

VLM_DEFAULT_PRESENCE_PENALTY = get_or_create_env_var(
    "VLM_DEFAULT_PRESENCE_PENALTY", ""
)  # Default presence penalty for VLM generation. If empty, model-specific defaults will be used.
if VLM_DEFAULT_PRESENCE_PENALTY and VLM_DEFAULT_PRESENCE_PENALTY.strip():
    VLM_DEFAULT_PRESENCE_PENALTY = float(VLM_DEFAULT_PRESENCE_PENALTY)
else:
    VLM_DEFAULT_PRESENCE_PENALTY = None


### Local OCR model - Tesseract vs PaddleOCR
CHOSEN_LOCAL_OCR_MODEL = get_or_create_env_var(
    "CHOSEN_LOCAL_OCR_MODEL", "tesseract"
)  # Choose the engine for local OCR: "tesseract", "paddle", "hybrid-paddle", "hybrid-vlm", "hybrid-paddle-vlm", "hybrid-paddle-inference-server", "vlm", "inference-server"

SHOW_OCR_GUI_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_OCR_GUI_OPTIONS", "True")
)

SHOW_LOCAL_OCR_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_LOCAL_OCR_MODEL_OPTIONS", "False")
)

SHOW_PADDLE_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_PADDLE_MODEL_OPTIONS", "False")
)

SHOW_INFERENCE_SERVER_VLM_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_INFERENCE_SERVER_VLM_OPTIONS", "False")
)

SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS", "False")
)

SHOW_HYBRID_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_HYBRID_MODELS", "False")
)

LOCAL_OCR_MODEL_OPTIONS = ["tesseract"]

CHOSEN_LOCAL_MODEL_INTRO_TEXT = get_or_create_env_var(
    "CHOSEN_LOCAL_MODEL_INTRO_TEXT",
    """Choose a local OCR model. "tesseract" is the default and will work for documents with clear typed text. """,
)

PADDLE_OCR_INTRO_TEXT = get_or_create_env_var(
    "PADDLE_OCR_INTRO_TEXT",
    """"paddle" is more accurate for text extraction where the text is not clear or well-formatted, but word-level extract is not natively supported, and so word bounding boxes will be inaccurate. """,
)

PADDLE_OCR_HYBRID_INTRO_TEXT = get_or_create_env_var(
    "PADDLE_OCR_HYBRID_INTRO_TEXT",
    """"hybrid-paddle" will do the first pass with Tesseract, and the second with PaddleOCR. """,
)

VLM_OCR_INTRO_TEXT = get_or_create_env_var(
    "VLM_OCR_INTRO_TEXT",
    """"vlm" will call the chosen vision model (VLM) to return a structured json output that is then parsed into word-level bounding boxes. """,
)

VLM_OCR_HYBRID_INTRO_TEXT = get_or_create_env_var(
    "VLM_OCR_HYBRID_INTRO_TEXT",
    """"hybrid-vlm" is a combination of Tesseract for OCR, and a second pass with the chosen vision model (VLM). """,
)

INFERENCE_SERVER_OCR_INTRO_TEXT = get_or_create_env_var(
    "INFERENCE_SERVER_OCR_INTRO_TEXT",
    """"inference-server" will call an external inference-server API to perform OCR using a vision model hosted remotely. """,
)

HYBRID_PADDLE_VLM_INTRO_TEXT = get_or_create_env_var(
    "HYBRID_PADDLE_VLM_INTRO_TEXT",
    """"hybrid-paddle-vlm" is a combination of PaddleOCR with the chosen VLM.""",
)

HYBRID_PADDLE_INFERENCE_SERVER_INTRO_TEXT = get_or_create_env_var(
    "HYBRID_PADDLE_INFERENCE_SERVER_INTRO_TEXT",
    """"hybrid-paddle-inference-server" is a combination of PaddleOCR with an external inference-server API.""",
)

paddle_options = ["paddle"]
# if SHOW_HYBRID_MODELS:
#     paddle_options.append("hybrid-paddle")
if SHOW_PADDLE_MODEL_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.extend(paddle_options)
    CHOSEN_LOCAL_MODEL_INTRO_TEXT += PADDLE_OCR_INTRO_TEXT
    # if SHOW_HYBRID_MODELS:
    #     CHOSEN_LOCAL_MODEL_INTRO_TEXT += PADDLE_OCR_HYBRID_INTRO_TEXT

vlm_options = ["vlm"]
# if SHOW_HYBRID_MODELS:
#     vlm_options.append("hybrid-vlm")
if SHOW_VLM_MODEL_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.extend(vlm_options)
    CHOSEN_LOCAL_MODEL_INTRO_TEXT += VLM_OCR_INTRO_TEXT
    # if SHOW_HYBRID_MODELS:
    #     CHOSEN_LOCAL_MODEL_INTRO_TEXT += VLM_OCR_HYBRID_INTRO_TEXT

if SHOW_PADDLE_MODEL_OPTIONS and SHOW_VLM_MODEL_OPTIONS and SHOW_HYBRID_MODELS:
    LOCAL_OCR_MODEL_OPTIONS.append("hybrid-paddle-vlm")
    CHOSEN_LOCAL_MODEL_INTRO_TEXT += HYBRID_PADDLE_VLM_INTRO_TEXT

if (
    SHOW_PADDLE_MODEL_OPTIONS
    and SHOW_INFERENCE_SERVER_VLM_OPTIONS
    and SHOW_HYBRID_MODELS
):
    LOCAL_OCR_MODEL_OPTIONS.append("hybrid-paddle-inference-server")
    CHOSEN_LOCAL_MODEL_INTRO_TEXT += HYBRID_PADDLE_INFERENCE_SERVER_INTRO_TEXT

inference_server_options = ["inference-server"]
if SHOW_INFERENCE_SERVER_VLM_OPTIONS:
    LOCAL_OCR_MODEL_OPTIONS.extend(inference_server_options)
    CHOSEN_LOCAL_MODEL_INTRO_TEXT += INFERENCE_SERVER_OCR_INTRO_TEXT

# Cloud VLM options
if SHOW_BEDROCK_VLM_MODELS:
    LOCAL_OCR_MODEL_OPTIONS.append("bedrock-vlm")

if SHOW_GEMINI_VLM_MODELS:
    LOCAL_OCR_MODEL_OPTIONS.append("gemini-vlm")

if SHOW_AZURE_OPENAI_VLM_MODELS:
    LOCAL_OCR_MODEL_OPTIONS.append("azure-openai-vlm")

# Inference-server API configuration
INFERENCE_SERVER_API_URL = get_or_create_env_var(
    "INFERENCE_SERVER_API_URL", "http://localhost:8080"
)  # Base URL of the inference-server API

INFERENCE_SERVER_MODEL_NAME = get_or_create_env_var(
    "INFERENCE_SERVER_MODEL_NAME", ""
)  # Optional model name to use. If empty, uses the default model on the server

INFERENCE_SERVER_TIMEOUT = int(
    get_or_create_env_var("INFERENCE_SERVER_TIMEOUT", "300")
)  # Timeout in seconds for API requests

DEFAULT_INFERENCE_SERVER_VLM_MODEL = get_or_create_env_var(
    "DEFAULT_INFERENCE_SERVER_VLM_MODEL", "qwen_3_vl_30b_a3b_it"
)  # Default model name for inference-server VLM API calls. If empty, uses INFERENCE_SERVER_MODEL_NAME or server default

DEFAULT_INFERENCE_SERVER_PII_MODEL = get_or_create_env_var(
    "DEFAULT_INFERENCE_SERVER_PII_MODEL", "gemma_3_12b"
)  # Default model name for inference-server PII detection API calls. If empty, uses INFERENCE_SERVER_MODEL_NAME, CHOSEN_INFERENCE_SERVER_PII_MODEL, or server default

MODEL_CACHE_PATH = get_or_create_env_var("MODEL_CACHE_PATH", "./model_cache")
MODEL_CACHE_PATH = ensure_folder_within_app_directory(MODEL_CACHE_PATH)


HYBRID_OCR_CONFIDENCE_THRESHOLD = int(
    get_or_create_env_var("HYBRID_OCR_CONFIDENCE_THRESHOLD", "95")
)  # The tesseract confidence threshold under which the text will be passed to PaddleOCR for re-extraction using the hybrid OCR method.

HYBRID_OCR_PADDING = int(
    get_or_create_env_var("HYBRID_OCR_PADDING", "1")
)  # The padding (in pixels) to add to the text when passing it to PaddleOCR for re-extraction using the hybrid OCR method.

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

INCLUDE_OCR_VISUALISATION_IN_OUTPUT_FILES = convert_string_to_boolean(
    get_or_create_env_var("INCLUDE_OCR_VISUALISATION_IN_OUTPUT_FILES", "False")
)  # Whether to include OCR visualisation outputs in the final output file list returned by choose_and_run_redactor.

SAVE_WORD_SEGMENTER_OUTPUT_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_WORD_SEGMENTER_OUTPUT_IMAGES", "False")
)  # Whether to save output images from the word segmenter.

# Model storage paths for Lambda compatibility
PADDLE_MODEL_PATH = get_or_create_env_var(
    "PADDLE_MODEL_PATH", ""
)  # Directory for PaddleOCR model storage. Uses default location if not set.
if PADDLE_MODEL_PATH:
    PADDLE_MODEL_PATH = ensure_folder_within_app_directory(PADDLE_MODEL_PATH)

PADDLE_FONT_PATH = get_or_create_env_var(
    "PADDLE_FONT_PATH", ""
)  # Custom font path for PaddleOCR. If empty, will attempt to use system fonts to avoid downloading simfang.ttf/PingFang-SC-Regular.ttf.
if PADDLE_FONT_PATH:
    PADDLE_FONT_PATH = ensure_folder_within_app_directory(PADDLE_FONT_PATH)

SPACY_MODEL_PATH = get_or_create_env_var(
    "SPACY_MODEL_PATH", ""
)  # Directory for spaCy model storage. Uses default location if not set.
if SPACY_MODEL_PATH:
    SPACY_MODEL_PATH = ensure_folder_within_app_directory(SPACY_MODEL_PATH)

PREPROCESS_LOCAL_OCR_IMAGES = get_or_create_env_var(
    "PREPROCESS_LOCAL_OCR_IMAGES", "True"
)  # Whether to try and preprocess images before extracting text. NOTE: I have found in testing that this doesn't necessarily imporove results, and greatly slows down extraction.

SAVE_PREPROCESS_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_PREPROCESS_IMAGES", "False")
)  # Whether to save the pre-processed images.

SAVE_VLM_INPUT_IMAGES = convert_string_to_boolean(
    get_or_create_env_var("SAVE_VLM_INPUT_IMAGES", "False")
)  # Whether to save input images sent to VLM OCR for debugging.

### LLM options

# Gemini settings
SHOW_GEMINI_LLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_GEMINI_LLM_MODELS", "False")
)
GEMINI_API_KEY = get_or_create_env_var("GEMINI_API_KEY", "")
# Azure/OpenAI AI Inference settings
SHOW_AZURE_LLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AZURE_LLM_MODELS", "False")
)
AZURE_OPENAI_API_KEY = get_or_create_env_var("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_INFERENCE_ENDPOINT = get_or_create_env_var(
    "AZURE_OPENAI_INFERENCE_ENDPOINT", ""
)

SHOW_INFERENCE_SERVER_LLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_INFERENCE_SERVER_LLM_MODELS", "False")
)
API_URL = get_or_create_env_var("API_URL", "http://localhost:8080")

# Build up options for models
model_full_names = list()
model_short_names = list()
model_source = list()

# Local Transformers LLM PII Detection Model Configuration
# This is a simple identifier for the model (e.g., "gemma-3-4b", "qwen-3-4b")
# The actual model loading uses LOCAL_TRANSFORMERS_LLM_PII_REPO_ID, LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE, and LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER
LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = get_or_create_env_var(
    "LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE", "gemma-3-4b"
)  # Model identifier for local transformers PII detection. This is used for display/logging purposes.
# These variables are the primary configuration for local model loading
# Define these early so they're available for use below
LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = get_or_create_env_var(
    "LOCAL_TRANSFORMERS_LLM_PII_REPO_ID", "unsloth/gemma-3-4b-it-bnb-4bit"
)  # Hugging Face repository ID for PII detection model (e.g., "unsloth/gemma-3-4b-it-bnb-4bit")
LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = get_or_create_env_var(
    "LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE", "gemma-3-4b-it-qat-UD-Q4_K_XL.gguf"
)  # Optional: Specific model filename if needed. If empty, uses the default from the repo.
LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = get_or_create_env_var(
    "LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER", "model/gemma3_4b"
)  # Optional: Local folder for PII model. If empty, uses MODEL_CACHE_PATH


USE_LLAMA_SWAP = get_or_create_env_var("USE_LLAMA_SWAP", "False")
if USE_LLAMA_SWAP == "True":
    USE_LLAMA_SWAP = True
else:
    USE_LLAMA_SWAP = False

if (
    SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS
    and LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
):
    # Use LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE for display if available, otherwise use LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
    display_name = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
    model_full_names.append(display_name)
    model_short_names.append(display_name)
    model_source.append("Local")

amazon_models = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "deepseek.v3-v1:0",
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
    "google.gemma-3-12b-it",
    "mistral.ministral-3-14b-instruct",
]

if SHOW_AWS_BEDROCK_LLM_MODELS:
    model_full_names.extend(amazon_models)
    model_short_names.extend(
        [
            "haiku",
            "sonnet_3_7",
            "sonnet_4_5",
            "nova_micro",
            "nova_lite",
            "nova_pro",
            "deepseek_v3",
            "gpt_oss_20b_aws",
            "gpt_oss_120b_aws",
            "gemma_3_12b_it",
            "ministral_3_14b_instruct",
        ]
    )
    model_source.extend(["AWS"] * len(amazon_models))

gemini_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

if SHOW_GEMINI_LLM_MODELS:
    model_full_names.extend(gemini_models)
    model_short_names.extend(
        ["gemini_flash_lite_2.5", "gemini_flash_2.5", "gemini_pro"]
    )
    model_source.extend(["Gemini"] * len(gemini_models))

azure_models = ["gpt-5-mini", "gpt-4o-mini"]

# Register Azure/OpenAI AI models (model names must match your Azure/OpenAI deployments)
if SHOW_AZURE_LLM_MODELS:
    # Example deployments; adjust to the deployments you actually create in Azure/OpenAI
    model_full_names.extend(azure_models)
    model_short_names.extend(["gpt-5-mini", "gpt-4o-mini"])
    model_source.extend(["Azure/OpenAI"] * len(azure_models))

# Register inference-server models
CHOSEN_INFERENCE_SERVER_PII_MODEL = ""
inference_server_models = [
    "unnamed-inference-server-model",
    "qwen_3_4b_it",
    "qwen_3_4b_think",
    "gpt_oss_20b",
    "gemma_3_12b",
    "ministral_3_14b_it",
]

if SHOW_INFERENCE_SERVER_LLM_MODELS:
    # Example inference-server models; adjust to the models you have available on your server
    model_full_names.extend(inference_server_models)
    model_short_names.extend(inference_server_models)
    model_source.extend(["inference-server"] * len(inference_server_models))

    CHOSEN_INFERENCE_SERVER_PII_MODEL = get_or_create_env_var(
        "CHOSEN_INFERENCE_SERVER_PII_MODEL", inference_server_models[0]
    )

    # If the chosen inference server model is not in the list of inference server models, add it to the list
    if CHOSEN_INFERENCE_SERVER_PII_MODEL not in inference_server_models:
        model_full_names.append(CHOSEN_INFERENCE_SERVER_PII_MODEL)
        model_short_names.append(CHOSEN_INFERENCE_SERVER_PII_MODEL)
        model_source.append("inference-server")

# Inference Server LLM Model Choice for PII Detection
# This is the primary config variable for choosing inference server models for PII detection
# Note: This must be defined after CHOSEN_INFERENCE_SERVER_PII_MODEL
INFERENCE_SERVER_LLM_PII_MODEL_CHOICE = get_or_create_env_var(
    "INFERENCE_SERVER_LLM_PII_MODEL_CHOICE",
    (
        DEFAULT_INFERENCE_SERVER_PII_MODEL
        if DEFAULT_INFERENCE_SERVER_PII_MODEL
        else (
            CHOSEN_INFERENCE_SERVER_PII_MODEL
            if CHOSEN_INFERENCE_SERVER_PII_MODEL
            else ""
        )
    ),
)  # Model choice for inference-server PII detection. Defaults to DEFAULT_INFERENCE_SERVER_PII_MODEL, then CHOSEN_INFERENCE_SERVER_PII_MODEL

model_name_map = {
    full: {"short_name": short, "source": source}
    for full, short, source in zip(model_full_names, model_short_names, model_source)
}

if SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS:
    default_model_choice = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
elif SHOW_INFERENCE_SERVER_LLM_MODELS:
    default_model_choice = CHOSEN_INFERENCE_SERVER_PII_MODEL
elif SHOW_AWS_BEDROCK_LLM_MODELS:
    default_model_choice = amazon_models[0]
elif SHOW_GEMINI_LLM_MODELS:
    default_model_choice = gemini_models[0]
elif SHOW_AZURE_LLM_MODELS:
    default_model_choice = azure_models[0]
else:
    default_model_choice = ""

if default_model_choice:
    default_model_source = model_name_map[default_model_choice]["source"]
    model_sources = list(
        set([model_name_map[model]["source"] for model in model_full_names])
    )
else:
    default_model_source = ""
    model_sources = []


def update_model_choice_config(default_model_source, model_name_map):
    # Filter models by source and return the first matching model name
    matching_models = [
        model_name
        for model_name, model_info in model_name_map.items()
        if model_info["source"] == default_model_source
    ]

    output_model = matching_models[0] if matching_models else model_full_names[0]

    return output_model, matching_models


if default_model_source:
    default_model_choice, default_source_models = update_model_choice_config(
        default_model_source, model_name_map
    )
else:
    default_model_choice = ""
    default_source_models = []

DIRECT_MODE_INFERENCE_SERVER_MODEL = get_or_create_env_var(
    "DIRECT_MODE_INFERENCE_SERVER_MODEL",
    CHOSEN_INFERENCE_SERVER_PII_MODEL if CHOSEN_INFERENCE_SERVER_PII_MODEL else "",
)

# Cloud LLM Model Choice for PII Detection (AWS Bedrock)
# Note: This should be set after amazon_models is defined
CLOUD_LLM_PII_MODEL_CHOICE = get_or_create_env_var(
    "CLOUD_LLM_PII_MODEL_CHOICE",
    "amazon.nova-pro-v1:0",  # "anthropic.claude-3-7-sonnet-20250219-v1:0",  # Default AWS Bedrock model for PII detection
)

# VLM Model Choice for cloud VLM OCR (defaults to first available cloud model)
# Note: This should be set after model lists are defined
CLOUD_VLM_MODEL_CHOICE = get_or_create_env_var(
    "CLOUD_VLM_MODEL_CHOICE",
    "qwen.qwen3-vl-235b-a22b",  # Will be set to default below if empty
)  # Default model choice for cloud VLM OCR (Bedrock, Gemini, or Azure/OpenAI)

# Set default CLOUD_VLM_MODEL_CHOICE if not provided
if not CLOUD_VLM_MODEL_CHOICE or not CLOUD_VLM_MODEL_CHOICE.strip():
    # Set default based on available models (priority: AWS Bedrock > Gemini > Azure/OpenAI)
    if SHOW_AWS_BEDROCK_LLM_MODELS and amazon_models:
        CLOUD_VLM_MODEL_CHOICE = amazon_models[0]  # Default to first AWS Bedrock model
    elif SHOW_GEMINI_LLM_MODELS and gemini_models:
        CLOUD_VLM_MODEL_CHOICE = gemini_models[0]  # Default to first Gemini model
    elif SHOW_AZURE_LLM_MODELS and azure_models:
        CLOUD_VLM_MODEL_CHOICE = azure_models[0]  # Default to first Azure/OpenAI model
    else:
        CLOUD_VLM_MODEL_CHOICE = ""  # No default available
else:
    # Use the value from environment variable
    CLOUD_VLM_MODEL_CHOICE = CLOUD_VLM_MODEL_CHOICE.strip()

# print("model_name_map:", model_name_map)

# HF token may or may not be needed for downloading models from Hugging Face
HF_TOKEN = get_or_create_env_var("HF_TOKEN", "")

LOAD_TRANSFORMERS_LLM_PII_MODEL_AT_START = convert_string_to_boolean(
    get_or_create_env_var("LOAD_TRANSFORMERS_LLM_PII_MODEL_AT_START", "False")
)

MULTIMODAL_PROMPT_FORMAT = convert_string_to_boolean(
    get_or_create_env_var("MULTIMODAL_PROMPT_FORMAT", "False")
)

# Following is not currently supported
# If you are using a system with low VRAM, you can set this to True to reduce the memory requirements
LOW_VRAM_SYSTEM = convert_string_to_boolean(
    get_or_create_env_var("LOW_VRAM_SYSTEM", "False")
)

if LOW_VRAM_SYSTEM:
    print("Using settings for low VRAM system")
    USE_LLAMA_CPP = get_or_create_env_var("USE_LLAMA_CPP", "True")
    LLM_MAX_NEW_TOKENS = int(get_or_create_env_var("LLM_MAX_NEW_TOKENS", "4096"))
    LLM_CONTEXT_LENGTH = int(get_or_create_env_var("LLM_CONTEXT_LENGTH", "16384"))
    LLM_BATCH_SIZE = int(get_or_create_env_var("LLM_BATCH_SIZE", "512"))
    K_QUANT_LEVEL = int(
        get_or_create_env_var("K_QUANT_LEVEL", "2")
    )  # 2 = q4_0, 8 = q8_0, 4 = fp16
    V_QUANT_LEVEL = int(
        get_or_create_env_var("V_QUANT_LEVEL", "2")
    )  # 2 = q4_0, 8 = q8_0, 4 = fp16

USE_LLAMA_CPP = get_or_create_env_var(
    "USE_LLAMA_CPP", "False"
)  # Not currently supported

GEMMA2_REPO_ID = get_or_create_env_var("GEMMA2_2B_REPO_ID", "unsloth/gemma-2-it-GGUF")
GEMMA2_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GEMMA2_2B_REPO_TRANSFORMERS_ID", "unsloth/gemma-2-2b-it-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    GEMMA2_REPO_ID = GEMMA2_REPO_TRANSFORMERS_ID
GEMMA2_MODEL_FILE = get_or_create_env_var(
    "GEMMA2_2B_MODEL_FILE", "gemma-2-2b-it.q8_0.gguf"
)
GEMMA2_MODEL_FOLDER = get_or_create_env_var("GEMMA2_2B_MODEL_FOLDER", "model/gemma")

GEMMA3_4B_REPO_ID = get_or_create_env_var(
    "GEMMA3_4B_REPO_ID", "unsloth/gemma-3-4b-it-qat-GGUF"
)
GEMMA3_4B_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GEMMA3_4B_REPO_TRANSFORMERS_ID", "unsloth/gemma-3-4b-it-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    GEMMA3_4B_REPO_ID = GEMMA3_4B_REPO_TRANSFORMERS_ID
GEMMA3_4B_MODEL_FILE = get_or_create_env_var(
    "GEMMA3_4B_MODEL_FILE", "gemma-3-4b-it-qat-UD-Q4_K_XL.gguf"
)
GEMMA3_4B_MODEL_FOLDER = get_or_create_env_var(
    "GEMMA3_4B_MODEL_FOLDER", "model/gemma3_4b"
)

GEMMA3_12B_REPO_ID = get_or_create_env_var(
    "GEMMA3_12B_REPO_ID", "unsloth/gemma-3-12b-it-GGUF"
)
GEMMA3_12B_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GEMMA3_12B_REPO_TRANSFORMERS_ID", "unsloth/gemma-3-12b-it-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    GEMMA3_12B_REPO_ID = GEMMA3_12B_REPO_TRANSFORMERS_ID
GEMMA3_12B_MODEL_FILE = get_or_create_env_var(
    "GEMMA3_12B_MODEL_FILE", "gemma-3-12b-it-UD-Q4_K_XL.gguf"
)
GEMMA3_12B_MODEL_FOLDER = get_or_create_env_var(
    "GEMMA3_12B_MODEL_FOLDER", "model/gemma3_12b"
)

GPT_OSS_REPO_ID = get_or_create_env_var("GPT_OSS_REPO_ID", "unsloth/gpt-oss-20b-GGUF")
GPT_OSS_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GPT_OSS_REPO_TRANSFORMERS_ID", "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    GPT_OSS_REPO_ID = GPT_OSS_REPO_TRANSFORMERS_ID
GPT_OSS_MODEL_FILE = get_or_create_env_var("GPT_OSS_MODEL_FILE", "gpt-oss-20b-F16.gguf")
GPT_OSS_MODEL_FOLDER = get_or_create_env_var("GPT_OSS_MODEL_FOLDER", "model/gpt_oss")

QWEN3_4B_REPO_ID = get_or_create_env_var(
    "QWEN3_4B_REPO_ID", "unsloth/Qwen3-4B-Instruct-2507-GGUF"
)
QWEN3_4B_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "QWEN3_4B_REPO_TRANSFORMERS_ID", "unsloth/Qwen3-4B-unsloth-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    QWEN3_4B_REPO_ID = QWEN3_4B_REPO_TRANSFORMERS_ID

QWEN3_4B_MODEL_FILE = get_or_create_env_var(
    "QWEN3_4B_MODEL_FILE", "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"
)
QWEN3_4B_MODEL_FOLDER = get_or_create_env_var("QWEN3_4B_MODEL_FOLDER", "model/qwen")

GRANITE_4_TINY_REPO_ID = get_or_create_env_var(
    "GRANITE_4_TINY_REPO_ID", "unsloth/granite-4.0-h-tiny-GGUF"
)
GRANITE_4_TINY_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GRANITE_4_TINY_REPO_TRANSFORMERS_ID", "unsloth/granite-4.0-h-tiny-FP8-Dynamic"
)
if USE_LLAMA_CPP == "False":
    GRANITE_4_TINY_REPO_ID = GRANITE_4_TINY_REPO_TRANSFORMERS_ID
GRANITE_4_TINY_MODEL_FILE = get_or_create_env_var(
    "GRANITE_4_TINY_MODEL_FILE", "granite-4.0-h-tiny-UD-Q4_K_XL.gguf"
)
GRANITE_4_TINY_MODEL_FOLDER = get_or_create_env_var(
    "GRANITE_4_TINY_MODEL_FOLDER", "model/granite"
)

GRANITE_4_3B_REPO_ID = get_or_create_env_var(
    "GRANITE_4_3B_REPO_ID", "unsloth/granite-4.0-h-micro-GGUF"
)
GRANITE_4_3B_REPO_TRANSFORMERS_ID = get_or_create_env_var(
    "GRANITE_4_3B_REPO_TRANSFORMERS_ID", "unsloth/granite-4.0-micro-unsloth-bnb-4bit"
)
if USE_LLAMA_CPP == "False":
    GRANITE_4_3B_REPO_ID = GRANITE_4_3B_REPO_TRANSFORMERS_ID
GRANITE_4_3B_MODEL_FILE = get_or_create_env_var(
    "GRANITE_4_3B_MODEL_FILE", "granite-4.0-h-micro-UD-Q4_K_XL.gguf"
)
GRANITE_4_3B_MODEL_FOLDER = get_or_create_env_var(
    "GRANITE_4_3B_MODEL_FOLDER", "model/granite"
)

# Override LOCAL_TRANSFORMERS_LLM_PII_* variables based on LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
# This allows users to set just the model choice and have the correct repo/file/folder automatically selected
if LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE:
    model_choice_lower = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE.lower()

    if "gemma-3-4b" in model_choice_lower or "gemma3-4b" in model_choice_lower:
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GEMMA3_4B_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GEMMA3_4B_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GEMMA3_4B_MODEL_FOLDER
    elif "gemma-3-12b" in model_choice_lower or "gemma3-12b" in model_choice_lower:
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GEMMA3_12B_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GEMMA3_12B_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GEMMA3_12B_MODEL_FOLDER
    elif "gemma-2" in model_choice_lower or "gemma2" in model_choice_lower:
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GEMMA2_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GEMMA2_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GEMMA2_MODEL_FOLDER
    elif "qwen-3-4b" in model_choice_lower or "qwen3-4b" in model_choice_lower:
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = QWEN3_4B_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = QWEN3_4B_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = QWEN3_4B_MODEL_FOLDER
    elif "gpt-oss" in model_choice_lower:
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GPT_OSS_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GPT_OSS_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GPT_OSS_MODEL_FOLDER
    elif (
        "granite-4-tiny" in model_choice_lower or "granite4-tiny" in model_choice_lower
    ):
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GRANITE_4_TINY_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GRANITE_4_TINY_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GRANITE_4_TINY_MODEL_FOLDER
    elif (
        "granite-4-micro" in model_choice_lower
        or "granite4-micro" in model_choice_lower
    ):
        LOCAL_TRANSFORMERS_LLM_PII_REPO_ID = GRANITE_4_3B_REPO_ID
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE = GRANITE_4_3B_MODEL_FILE
        LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER = GRANITE_4_3B_MODEL_FOLDER
    # If model choice doesn't match any known model, keep the existing values from environment variables

# Map LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE to LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE format
model_choice_lower = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE.lower()

if "gemma-3-4b" in model_choice_lower or "gemma3-4b" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Gemma 3 4B"
elif "gemma-3-12b" in model_choice_lower or "gemma3-12b" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Gemma 3 12B"
elif "gemma-2" in model_choice_lower or "gemma2" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Gemma 2b"
elif "qwen-3-4b" in model_choice_lower or "qwen3-4b" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Qwen 3 4B"
elif "gpt-oss" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "gpt-oss-20b"
elif "granite-4-tiny" in model_choice_lower or "granite4-tiny" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Granite 4 Tiny"
elif "granite-4-micro" in model_choice_lower or "granite4-micro" in model_choice_lower:
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE = "Granite 4 Micro"

# Set MULTIMODAL_PROMPT_FORMAT based on model choice
if LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE in ["Gemma 3 4B", "Gemma 3 12B"]:
    MULTIMODAL_PROMPT_FORMAT = True

LLM_MAX_GPU_LAYERS = int(
    get_or_create_env_var("LLM_MAX_GPU_LAYERS", "-1")
)  # Maximum possible
LLM_TEMPERATURE = float(get_or_create_env_var("LLM_TEMPERATURE", "0.6"))
LLM_TOP_K = int(
    get_or_create_env_var("LLM_TOP_K", "64")
)  # https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune
LLM_MIN_P = float(get_or_create_env_var("LLM_MIN_P", "0"))
LLM_TOP_P = float(get_or_create_env_var("LLM_TOP_P", "0.95"))
LLM_REPETITION_PENALTY = float(get_or_create_env_var("LLM_REPETITION_PENALTY", "1.0"))
LLM_LAST_N_TOKENS = int(get_or_create_env_var("LLM_LAST_N_TOKENS", "512"))
LLM_MAX_NEW_TOKENS = int(get_or_create_env_var("LLM_MAX_NEW_TOKENS", "4096"))
LLM_SEED = int(get_or_create_env_var("LLM_SEED", "42"))
LLM_RESET = convert_string_to_boolean(get_or_create_env_var("LLM_RESET", "False"))
LLM_STREAM = convert_string_to_boolean(get_or_create_env_var("LLM_STREAM", "True"))
LLM_THREADS = int(get_or_create_env_var("LLM_THREADS", "-1"))
LLM_BATCH_SIZE = int(get_or_create_env_var("LLM_BATCH_SIZE", "2048"))
LLM_CONTEXT_LENGTH = int(get_or_create_env_var("LLM_CONTEXT_LENGTH", "16384"))  # 24576
LLM_SAMPLE = convert_string_to_boolean(get_or_create_env_var("LLM_SAMPLE", "True"))
LLM_STOP_STRINGS = _get_env_list(
    get_or_create_env_var("LLM_STOP_STRINGS", r"['\n\n\n\n\n\n']")
)

SPECULATIVE_DECODING = convert_string_to_boolean(
    get_or_create_env_var("SPECULATIVE_DECODING", "False")
)
NUM_PRED_TOKENS = int(get_or_create_env_var("NUM_PRED_TOKENS", "2"))


# LLM-specific configs for PII detection
# These can be overridden via environment variables, otherwise use general LLM configs
LLM_PII_TEMPERATURE = float(
    get_or_create_env_var("LLM_PII_TEMPERATURE", str(LLM_TEMPERATURE))
)
LLM_PII_MAX_TOKENS = int(
    get_or_create_env_var("LLM_PII_MAX_TOKENS", str(LLM_MAX_NEW_TOKENS))
)
LLM_PII_NUMBER_OF_RETRY_ATTEMPTS = int(
    get_or_create_env_var("LLM_PII_NUMBER_OF_RETRY_ATTEMPTS", "3")
)
LLM_PII_TIMEOUT_WAIT = int(get_or_create_env_var("LLM_PII_TIMEOUT_WAIT", "5"))

# Additional LLM configuration options
ASSISTANT_MODEL = get_or_create_env_var("ASSISTANT_MODEL", "")
BATCH_SIZE_DEFAULT = int(get_or_create_env_var("BATCH_SIZE_DEFAULT", "512"))
COMPILE_MODE = get_or_create_env_var("COMPILE_MODE", "reduce-overhead")
COMPILE_TRANSFORMERS = convert_string_to_boolean(
    get_or_create_env_var("COMPILE_TRANSFORMERS", "False")
)
DEDUPLICATION_THRESHOLD = float(get_or_create_env_var("DEDUPLICATION_THRESHOLD", "0.9"))
INT8_WITH_OFFLOAD_TO_CPU = convert_string_to_boolean(
    get_or_create_env_var("INT8_WITH_OFFLOAD_TO_CPU", "False")
)
MAX_COMMENT_CHARS = int(get_or_create_env_var("MAX_COMMENT_CHARS", "1000"))
MAX_TIME_FOR_LOOP = int(get_or_create_env_var("MAX_TIME_FOR_LOOP", "3600"))
MODEL_DTYPE = get_or_create_env_var("MODEL_DTYPE", "bfloat16")
NUMBER_OF_RETRY_ATTEMPTS = int(get_or_create_env_var("NUMBER_OF_RETRY_ATTEMPTS", "3"))
TIMEOUT_WAIT = int(get_or_create_env_var("TIMEOUT_WAIT", "30"))
QUANTISE_TRANSFORMERS_LLM_MODELS = convert_string_to_boolean(
    get_or_create_env_var("QUANTISE_TRANSFORMERS_LLM_MODELS", "False")
)
PRINT_TRANSFORMERS_USER_PROMPT = convert_string_to_boolean(
    get_or_create_env_var("PRINT_TRANSFORMERS_USER_PROMPT", "False")
)


# If you are using e.g. gpt-oss, you can add a reasoning suffix to set reasoning level, or turn it off in the case of Qwen 3 4B
# Use LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE if available, otherwise check LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
model_type_for_reasoning = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE

if LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE == "gpt-oss-20b":
    REASONING_SUFFIX = get_or_create_env_var("REASONING_SUFFIX", "Reasoning: low")
    # print("Using REASONING_SUFFIX: Reasoning: low")
elif LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE == "Qwen 3 4B":
    # print("Using REASONING_SUFFIX: /nothink")
    REASONING_SUFFIX = get_or_create_env_var("REASONING_SUFFIX", "/nothink")
else:
    # print("No reasoning suffix applied")
    REASONING_SUFFIX = get_or_create_env_var("REASONING_SUFFIX", "")


# Entities for redaction
CHOSEN_COMPREHEND_ENTITIES = get_or_create_env_var(
    "CHOSEN_COMPREHEND_ENTITIES",
    "['EMAIL','ADDRESS','NAME','PHONE', 'PASSPORT_NUMBER', 'UK_NATIONAL_INSURANCE_NUMBER', 'UK_NATIONAL_HEALTH_SERVICE_NUMBER', 'CUSTOM']",
)

FULL_COMPREHEND_ENTITY_LIST = get_or_create_env_var(
    "FULL_COMPREHEND_ENTITY_LIST",
    "['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE','SSN','DATE_TIME','PASSPORT_NUMBER','DRIVER_ID','URL','AGE','USERNAME','PASSWORD','AWS_ACCESS_KEY','AWS_SECRET_KEY','IP_ADDRESS','MAC_ADDRESS','ALL','LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER','INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER','CUSTOM', 'CUSTOM_FUZZY']",
)

FULL_LLM_ENTITY_LIST = get_or_create_env_var(
    "FULL_LLM_ENTITY_LIST",
    "['EMAIL_ADDRESS','ADDRESS','NAME','PHONE_NUMBER', 'DATE_TIME', 'URL', 'IP_ADDRESS', 'MAC_ADDRESS', 'AGE', 'BANK_ACCOUNT_NUMBER', 'PASSPORT_NUMBER', 'CA_HEALTH_NUMBER', 'CUSTOM', 'CUSTOM_FUZZY']",
)

# Entities for LLM-based PII redaction option
CHOSEN_LLM_ENTITIES = get_or_create_env_var(
    "CHOSEN_LLM_ENTITIES",
    "['EMAIL_ADDRESS','ADDRESS','NAME','PHONE_NUMBER', 'CUSTOM']",
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
    "CUSTOM_ENTITIES",
    "['TITLES', 'UKPOSTCODE', 'STREETNAME']",
)


DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX = get_or_create_env_var(
    "DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX", "[]"
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

# Whether to split punctuation from words in Textract output
# If True, punctuation marks (full stops, commas, quotes, brackets, etc.) will be separated
# from alphanumeric characters and returned as separate words with separate bounding boxes.
# If False, words will be returned as-is from Textract (original behavior).
SPLIT_PUNCTUATION_FROM_WORDS = convert_string_to_boolean(
    get_or_create_env_var("SPLIT_PUNCTUATION_FROM_WORDS", "False")
)

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
)  # How to redact overlapping vector graphics (also called "line-art" or "drawings"). (2) removes any overlapping vector graphics. PDF_REDACT_LINE_ART_NONE | 0 ignores, and PDF_REDACT_LINE_ART_REMOVE_IF_COVERED | 1 removes graphics fully contained in a redaction annotation.
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
# Link to user guide - ensure it is a valid URL
USER_GUIDE_URL = validate_safe_url(
    get_or_create_env_var(
        "USER_GUIDE_URL", "https://seanpedrick-case.github.io/doc_redaction"
    )
)

DEFAULT_INTRO_TEXT = f"""# Document redaction

    Redact personally identifiable information (PII) from documents (pdf, png, jpg), Word files (docx), or tabular data (xlsx/csv/parquet). Please see the [User Guide]({USER_GUIDE_URL}) for a full walkthrough of all the features in the app.
    
    To extract text from documents, the 'Local' options are PikePDF for PDFs with selectable text, and OCR with Tesseract. Use AWS Textract to extract more complex elements e.g. handwriting, signatures, or unclear text. For PII identification, 'Local' (based on spaCy) gives good results if you are looking for common names or terms, or a custom list of terms to redact (see Redaction settings).  AWS Comprehend gives better results at a small cost.

    Additional options on the 'Redaction settings' include, the type of information to redact (e.g. people, places), custom terms to include/ exclude from redaction, fuzzy matching, language settings, and whole page redaction. After redaction is complete, you can view and modify suggested redactions on the 'Review redactions' tab to quickly create a final redacted document.

    NOTE: The app is not 100% accurate, and it will miss some personal information. It is essential that all outputs are reviewed **by a human** before using the final outputs."""

INTRO_TEXT = get_or_create_env_var("INTRO_TEXT", DEFAULT_INTRO_TEXT)

# Read in intro text from a text file if it is a path to a text file
if INTRO_TEXT.endswith(".txt"):
    # Validate the path is safe (with base path for relative paths)
    if validate_path_safety(INTRO_TEXT, base_path="."):
        try:
            # Use secure file read with explicit encoding
            INTRO_TEXT = secure_file_read(".", INTRO_TEXT, encoding="utf-8")
            # Format the text to replace {USER_GUIDE_URL} with the actual value
            INTRO_TEXT = INTRO_TEXT.format(USER_GUIDE_URL=USER_GUIDE_URL)
        except FileNotFoundError:
            print(f"Warning: Intro text file not found: {INTRO_TEXT}")
            INTRO_TEXT = DEFAULT_INTRO_TEXT
        except Exception as e:
            print(f"Error reading intro text file: {e}")
            # Fallback to default
            INTRO_TEXT = DEFAULT_INTRO_TEXT
    else:
        print(f"Warning: Unsafe file path detected for INTRO_TEXT: {INTRO_TEXT}")
        INTRO_TEXT = DEFAULT_INTRO_TEXT

# Sanitize the text
INTRO_TEXT = sanitize_markdown_text(INTRO_TEXT.strip('"').strip("'"))

# Ensure we have valid content after sanitization
if not INTRO_TEXT or not INTRO_TEXT.strip():
    print("Warning: Intro text is empty after sanitization, using default intro text")
    INTRO_TEXT = sanitize_markdown_text(DEFAULT_INTRO_TEXT)

TLDEXTRACT_CACHE = get_or_create_env_var("TLDEXTRACT_CACHE", "tmp/tld/")
TLDEXTRACT_CACHE = ensure_folder_within_app_directory(TLDEXTRACT_CACHE)
try:
    extract = TLDExtract(cache_dir=TLDEXTRACT_CACHE)
except Exception as e:
    print(f"Error initialising TLDExtract: {e}")
    extract = TLDExtract(cache_dir=None)

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = convert_string_to_boolean(get_or_create_env_var("COGNITO_AUTH", "False"))

SHOW_FEEDBACK_BUTTONS = convert_string_to_boolean(
    get_or_create_env_var("SHOW_FEEDBACK_BUTTONS", "False")
)

SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER = convert_string_to_boolean(
    get_or_create_env_var("SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER", "False")
)

APPLY_DUPLICATES_TO_FILE_AUTOMATICALLY = convert_string_to_boolean(
    get_or_create_env_var("APPLY_DUPLICATES_TO_FILE_AUTOMATICALLY", "False")
)

SHOW_EXAMPLES = convert_string_to_boolean(
    get_or_create_env_var("SHOW_EXAMPLES", "True")
)
SHOW_AWS_EXAMPLES = convert_string_to_boolean(
    get_or_create_env_var("SHOW_AWS_EXAMPLES", "False")
)
SHOW_DIFFICULT_OCR_EXAMPLES = convert_string_to_boolean(
    get_or_create_env_var("SHOW_DIFFICULT_OCR_EXAMPLES", "False")
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
DIRECT_MODE_OUTPUT_DIR = ensure_folder_within_app_directory(DIRECT_MODE_OUTPUT_DIR)
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
if FULL_LLM_ENTITY_LIST:
    FULL_LLM_ENTITY_LIST = _get_env_list(FULL_LLM_ENTITY_LIST)
if CHOSEN_LLM_ENTITIES:
    CHOSEN_LLM_ENTITIES = _get_env_list(CHOSEN_LLM_ENTITIES)
if CHOSEN_REDACT_ENTITIES:
    CHOSEN_REDACT_ENTITIES = _get_env_list(CHOSEN_REDACT_ENTITIES)
if FULL_ENTITY_LIST:
    FULL_ENTITY_LIST = _get_env_list(FULL_ENTITY_LIST)

if (
    SHOW_VLM_MODEL_OPTIONS
    or SHOW_INFERENCE_SERVER_VLM_OPTIONS
    or SHOW_BEDROCK_VLM_MODELS
):
    FULL_ENTITY_LIST.extend(["CUSTOM_VLM_PERSON", "CUSTOM_VLM_SIGNATURE"])
    FULL_COMPREHEND_ENTITY_LIST.extend(["CUSTOM_VLM_PERSON", "CUSTOM_VLM_SIGNATURE"])
    FULL_LLM_ENTITY_LIST.extend(["CUSTOM_VLM_PERSON", "CUSTOM_VLM_SIGNATURE"])

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

if textract_language_choices:
    textract_language_choices = _get_env_list(textract_language_choices)
if aws_comprehend_language_choices:
    aws_comprehend_language_choices = _get_env_list(aws_comprehend_language_choices)

if MAPPED_LANGUAGE_CHOICES:
    MAPPED_LANGUAGE_CHOICES = _get_env_list(MAPPED_LANGUAGE_CHOICES)
if LANGUAGE_CHOICES:
    LANGUAGE_CHOICES = _get_env_list(LANGUAGE_CHOICES)

LANGUAGE_MAP = dict(zip(MAPPED_LANGUAGE_CHOICES, LANGUAGE_CHOICES))
