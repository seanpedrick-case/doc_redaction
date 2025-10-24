import json
import os

import boto3
from dotenv import load_dotenv

# Import the main function from your CLI script
from cli_redact import main as cli_main
from tools.config import AWS_REGION


def _get_env_list(env_var_name: str | list[str] | None) -> list[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    if isinstance(env_var_name, list):
        return env_var_name
    if env_var_name is None:
        return []

    # Handle string input
    value = str(env_var_name).strip()
    if not value or value == "[]":
        return []

    # Remove brackets if present (e.g., "[item1, item2]" -> "item1, item2")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    # Remove quotes and split by comma
    value = value.replace('"', "").replace("'", "")
    if not value:
        return []

    # Split by comma and filter out any empty strings
    return [s.strip() for s in value.split(",") if s.strip()]


print("Lambda entrypoint loading...")

# Initialize S3 client outside the handler for connection reuse
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", AWS_REGION))
print("S3 client initialised")

# Lambda's only writable directory is /tmp. Ensure that all temporary files are stored in this directory.
TMP_DIR = "/tmp"
INPUT_DIR = os.path.join(TMP_DIR, "input")
OUTPUT_DIR = os.path.join(TMP_DIR, "output")
os.environ["TESSERACT_DATA_FOLDER"] = os.path.join(TMP_DIR, "share/tessdata")
os.environ["TLDEXTRACT_CACHE"] = os.path.join(TMP_DIR, "tld")
os.environ["MPLCONFIGDIR"] = os.path.join(TMP_DIR, "matplotlib_cache")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(TMP_DIR, "gradio_tmp")
os.environ["FEEDBACK_LOGS_FOLDER"] = os.path.join(TMP_DIR, "feedback")
os.environ["ACCESS_LOGS_FOLDER"] = os.path.join(TMP_DIR, "logs")
os.environ["USAGE_LOGS_FOLDER"] = os.path.join(TMP_DIR, "usage")

# Define compatible file types for processing
COMPATIBLE_FILE_TYPES = {
    ".pdf",
    ".xlsx",
    ".xls",
    ".png",
    ".jpeg",
    ".csv",
    ".parquet",
    ".txt",
    ".jpg",
}


def download_file_from_s3(bucket_name, key, download_path):
    """Download a file from S3 to the local filesystem."""
    try:
        s3_client.download_file(bucket_name, key, download_path)
        print(f"Successfully downloaded s3://{bucket_name}/{key} to {download_path}")
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


def upload_directory_to_s3(local_directory, bucket_name, s3_prefix):
    """Upload all files from a local directory to an S3 prefix."""
    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            # Create a relative path to maintain directory structure if needed
            relative_path = os.path.relpath(local_file_path, local_directory)
            output_key = os.path.join(s3_prefix, relative_path)

            try:
                s3_client.upload_file(local_file_path, bucket_name, output_key)
                print(
                    f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{output_key}"
                )
            except Exception as e:
                print(f"Error uploading to S3: {e}")
                raise


def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    # 1. Setup temporary directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Extract information from the event
    # Assumes the event is triggered by S3 and may contain an 'arguments' payload
    try:
        record = event["Records"][0]
        bucket_name = record["s3"]["bucket"]["name"]
        input_key = record["s3"]["object"]["key"]

        # The user metadata can be used to pass arguments
        # This is more robust than embedding them in the main event body
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=input_key)
            metadata = response.get("Metadata", dict())
            print(f"S3 object metadata: {metadata}")

            # Arguments can be passed as a JSON string in metadata
            arguments_str = metadata.get("arguments", "{}")
            print(f"Arguments string from metadata: '{arguments_str}'")

            if arguments_str and arguments_str != "{}":
                arguments = json.loads(arguments_str)
                print(f"Successfully parsed arguments from metadata: {arguments}")
            else:
                arguments = dict()
                print("No arguments found in metadata, using empty dictionary")
        except Exception as e:
            print(f"Warning: Could not parse metadata arguments: {e}")
            print("Using empty arguments dictionary")
            arguments = dict()

    except (KeyError, IndexError) as e:
        print(
            f"Could not parse S3 event record: {e}. Checking for direct invocation payload."
        )
        # Fallback for direct invocation (e.g., from Step Functions or manual test)
        bucket_name = event.get("bucket_name")
        input_key = event.get("input_key")
        arguments = event.get("arguments", dict())
        if not all([bucket_name, input_key]):
            raise ValueError(
                "Missing 'bucket_name' or 'input_key' in direct invocation event."
            )

    # print(f"Processing s3://{bucket_name}/{input_key}")
    # print(f"With arguments: {arguments}")
    # print(f"Arguments type: {type(arguments)}")

    # Log file type information
    file_extension = os.path.splitext(input_key)[1].lower()
    print(f"Detected file extension: '{file_extension}'")

    # 3. Download the main input file
    input_file_path = os.path.join(INPUT_DIR, os.path.basename(input_key))
    download_file_from_s3(bucket_name, input_key, input_file_path)

    # 3.1. Validate file type compatibility
    is_env_file = input_key.lower().endswith(".env")

    if not is_env_file and file_extension not in COMPATIBLE_FILE_TYPES:
        error_message = f"File type '{file_extension}' is not supported for processing. Compatible file types are: {', '.join(sorted(COMPATIBLE_FILE_TYPES))}"
        print(f"ERROR: {error_message}")
        print(f"File was not processed due to unsupported file type: {file_extension}")
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "Unsupported file type",
                    "message": error_message,
                    "supported_types": list(COMPATIBLE_FILE_TYPES),
                    "received_type": file_extension,
                    "file_processed": False,
                }
            ),
        }

    print(f"File type '{file_extension}' is compatible for processing")
    if is_env_file:
        print("Processing .env file for configuration")
    else:
        print(f"Processing {file_extension} file for redaction/anonymization")

    # 3.5. Check if the downloaded file is a .env file and handle accordingly
    actual_input_file_path = input_file_path
    if input_key.lower().endswith(".env"):
        print("Detected .env file, loading environment variables...")

        # Load environment variables from the .env file
        load_dotenv(input_file_path)
        print("Environment variables loaded from .env file")

        # Extract the actual input file path from environment variables
        # Look for common environment variable names that might contain the input file path
        env_input_file = os.getenv(
            "INPUT_FILE"
        )  # This needs to be the full S3 path to the input file, e.g.INPUT_FILE=s3://my-processing-bucket/documents/sensitive-data.pdf

        if env_input_file:
            print(f"Found input file path in environment: {env_input_file}")

            # If the path is an S3 path, download it
            if env_input_file.startswith("s3://"):
                # Parse S3 path: s3://bucket/key
                s3_path_parts = env_input_file[5:].split("/", 1)
                if len(s3_path_parts) == 2:
                    env_bucket = s3_path_parts[0]
                    env_key = s3_path_parts[1]
                    actual_input_file_path = os.path.join(
                        INPUT_DIR, os.path.basename(env_key)
                    )
                    print(
                        f"Downloading actual input file from s3://{env_bucket}/{env_key}"
                    )
                    download_file_from_s3(env_bucket, env_key, actual_input_file_path)
                else:
                    print("Warning: Invalid S3 path format in environment variable")
                    actual_input_file_path = input_file_path
            else:
                # Assume it's a local path or relative path
                actual_input_file_path = env_input_file
                print(
                    f"Using input file path from environment: {actual_input_file_path}"
                )
        else:
            print("Warning: No input file path found in environment variables")
            print(
                "Available environment variables:",
                [
                    k
                    for k in os.environ.keys()
                    if k.startswith(("INPUT", "FILE", "DOCUMENT", "DIRECT"))
                ],
            )
            # Fall back to using the .env file itself (though this might not be what we want)
            actual_input_file_path = input_file_path
    else:
        print("File is not a .env file, proceeding with normal processing")

    # 4. Prepare arguments for the CLI function
    # This dictionary should mirror the one in your app.py's "direct mode"
    # If we loaded a .env file, use environment variables as defaults
    cli_args = {
        # Task Selection
        "task": arguments.get("task", os.getenv("DIRECT_MODE_TASK", "redact")),
        # General Arguments (apply to all file types)
        "input_file": actual_input_file_path,
        "output_dir": OUTPUT_DIR,
        "input_dir": INPUT_DIR,
        "language": arguments.get("language", os.getenv("DEFAULT_LANGUAGE", "en")),
        "allow_list": arguments.get("allow_list", os.getenv("ALLOW_LIST_PATH", "")),
        "pii_detector": arguments.get(
            "pii_detector", os.getenv("LOCAL_PII_OPTION", "Local")
        ),
        "username": arguments.get(
            "username", os.getenv("DIRECT_MODE_DEFAULT_USER", "lambda_user")
        ),
        "save_to_user_folders": arguments.get(
            "save_to_user_folders", os.getenv("SESSION_OUTPUT_FOLDER", "False")
        ),
        "local_redact_entities": _get_env_list(
            arguments.get(
                "local_redact_entities", os.getenv("CHOSEN_REDACT_ENTITIES", list())
            )
        ),
        "aws_redact_entities": _get_env_list(
            arguments.get(
                "aws_redact_entities", os.getenv("CHOSEN_COMPREHEND_ENTITIES", list())
            )
        ),
        "aws_access_key": None,  # Use IAM Role instead of keys
        "aws_secret_key": None,  # Use IAM Role instead of keys
        "cost_code": arguments.get("cost_code", os.getenv("DEFAULT_COST_CODE", "")),
        "aws_region": os.getenv("AWS_REGION", ""),
        "s3_bucket": bucket_name,
        "do_initial_clean": arguments.get(
            "do_initial_clean", os.getenv("DO_INITIAL_TABULAR_DATA_CLEAN", "False")
        ),
        "save_logs_to_csv": arguments.get(
            "save_logs_to_csv", os.getenv("SAVE_LOGS_TO_CSV", "False")
        ),
        "save_logs_to_dynamodb": arguments.get(
            "save_logs_to_dynamodb", os.getenv("SAVE_LOGS_TO_DYNAMODB", "False")
        ),
        "display_file_names_in_logs": arguments.get(
            "display_file_names_in_logs",
            os.getenv("DISPLAY_FILE_NAMES_IN_LOGS", "True"),
        ),
        "upload_logs_to_s3": arguments.get(
            "upload_logs_to_s3", os.getenv("RUN_AWS_FUNCTIONS", "False")
        ),
        "s3_logs_prefix": arguments.get(
            "s3_logs_prefix", os.getenv("S3_USAGE_LOGS_FOLDER", "")
        ),
        "feedback_logs_folder": arguments.get(
            "feedback_logs_folder", os.getenv("FEEDBACK_LOGS_FOLDER", "")
        ),
        "access_logs_folder": arguments.get(
            "access_logs_folder", os.getenv("ACCESS_LOGS_FOLDER", "")
        ),
        "usage_logs_folder": arguments.get(
            "usage_logs_folder", os.getenv("USAGE_LOGS_FOLDER", "")
        ),
        # PDF/Image Redaction Arguments
        "ocr_method": arguments.get(
            "ocr_method", os.getenv("TESSERACT_TEXT_EXTRACT_OPTION", "Local OCR")
        ),
        "page_min": int(arguments.get("page_min", os.getenv("DEFAULT_PAGE_MIN", 0))),
        "page_max": int(arguments.get("page_max", os.getenv("DEFAULT_PAGE_MAX", 0))),
        "images_dpi": float(
            arguments.get("images_dpi", os.getenv("IMAGES_DPI", 300.0))
        ),
        "chosen_local_ocr_model": arguments.get(
            "chosen_local_ocr_model", os.getenv("CHOSEN_LOCAL_OCR_MODEL", "tesseract")
        ),
        "preprocess_local_ocr_images": arguments.get(
            "preprocess_local_ocr_images",
            os.getenv("PREPROCESS_LOCAL_OCR_IMAGES", "False"),
        ),
        "compress_redacted_pdf": arguments.get(
            "compress_redacted_pdf", os.getenv("COMPRESS_REDACTED_PDF", "False")
        ),
        "return_pdf_end_of_redaction": arguments.get(
            "return_pdf_end_of_redaction", os.getenv("RETURN_REDACTED_PDF", "True")
        ),
        "deny_list_file": arguments.get(
            "deny_list_file", os.getenv("DENY_LIST_PATH", "")
        ),
        "allow_list_file": arguments.get(
            "allow_list_file", os.getenv("ALLOW_LIST_PATH", "")
        ),
        "redact_whole_page_file": arguments.get(
            "redact_whole_page_file", os.getenv("WHOLE_PAGE_REDACTION_LIST_PATH", "")
        ),
        "handwrite_signature_extraction": _get_env_list(
            arguments.get(
                "handwrite_signature_extraction",
                os.getenv(
                    "DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX",
                    ["Extract handwriting", "Extract signatures"],
                ),
            )
        ),
        "extract_forms": arguments.get(
            "extract_forms",
            os.getenv("INCLUDE_FORM_EXTRACTION_TEXTRACT_OPTION", "False") == "True",
        ),
        "extract_tables": arguments.get(
            "extract_tables",
            os.getenv("INCLUDE_TABLE_EXTRACTION_TEXTRACT_OPTION", "False") == "True",
        ),
        "extract_layout": arguments.get(
            "extract_layout",
            os.getenv("INCLUDE_LAYOUT_EXTRACTION_TEXTRACT_OPTION", "False") == "True",
        ),
        # Word/Tabular Anonymisation Arguments
        "anon_strategy": arguments.get(
            "anon_strategy",
            os.getenv("DEFAULT_TABULAR_ANONYMISATION_STRATEGY", "redact completely"),
        ),
        "text_columns": arguments.get(
            "text_columns", os.getenv("DEFAULT_TEXT_COLUMNS", list())
        ),
        "excel_sheets": arguments.get(
            "excel_sheets", os.getenv("DEFAULT_EXCEL_SHEETS", list())
        ),
        "fuzzy_mistakes": int(
            arguments.get(
                "fuzzy_mistakes", os.getenv("DEFAULT_FUZZY_SPELLING_MISTAKES_NUM", 1)
            )
        ),
        "match_fuzzy_whole_phrase_bool": arguments.get(
            "match_fuzzy_whole_phrase_bool",
            os.getenv("MATCH_FUZZY_WHOLE_PHRASE_BOOL", "True"),
        ),
        # Duplicate Detection Arguments
        "duplicate_type": arguments.get(
            "duplicate_type", os.getenv("DIRECT_MODE_DUPLICATE_TYPE", "pages")
        ),
        "similarity_threshold": float(
            arguments.get(
                "similarity_threshold",
                os.getenv("DEFAULT_DUPLICATE_DETECTION_THRESHOLD", 0.95),
            )
        ),
        "min_word_count": int(
            arguments.get("min_word_count", os.getenv("DEFAULT_MIN_WORD_COUNT", 10))
        ),
        "min_consecutive_pages": int(
            arguments.get(
                "min_consecutive_pages", os.getenv("DEFAULT_MIN_CONSECUTIVE_PAGES", 1)
            )
        ),
        "greedy_match": arguments.get(
            "greedy_match", os.getenv("USE_GREEDY_DUPLICATE_DETECTION", "False")
        ),
        "combine_pages": arguments.get(
            "combine_pages", os.getenv("DEFAULT_COMBINE_PAGES", "True")
        ),
        "remove_duplicate_rows": arguments.get(
            "remove_duplicate_rows", os.getenv("REMOVE_DUPLICATE_ROWS", "False")
        ),
        # Textract Batch Operations Arguments
        "textract_action": arguments.get("textract_action", ""),
        "job_id": arguments.get("job_id", ""),
        "extract_signatures": arguments.get("extract_signatures", False),
        "textract_bucket": arguments.get(
            "textract_bucket", os.getenv("TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET", "")
        ),
        "textract_input_prefix": arguments.get(
            "textract_input_prefix",
            os.getenv("TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER", ""),
        ),
        "textract_output_prefix": arguments.get(
            "textract_output_prefix",
            os.getenv("TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER", ""),
        ),
        "s3_textract_document_logs_subfolder": arguments.get(
            "s3_textract_document_logs_subfolder", os.getenv("TEXTRACT_JOBS_S3_LOC", "")
        ),
        "local_textract_document_logs_subfolder": arguments.get(
            "local_textract_document_logs_subfolder",
            os.getenv("TEXTRACT_JOBS_LOCAL_LOC", ""),
        ),
        "poll_interval": int(arguments.get("poll_interval", 30)),
        "max_poll_attempts": int(arguments.get("max_poll_attempts", 120)),
        # Additional arguments that were missing
        "search_query": arguments.get(
            "search_query", os.getenv("DEFAULT_SEARCH_QUERY", "")
        ),
        "prepare_images": arguments.get("prepare_images", True),
    }

    # Combine extraction options
    extraction_options = (
        _get_env_list(cli_args["handwrite_signature_extraction"])
        if cli_args["handwrite_signature_extraction"]
        else list()
    )
    if cli_args["extract_forms"]:
        extraction_options.append("Extract forms")
    if cli_args["extract_tables"]:
        extraction_options.append("Extract tables")
    if cli_args["extract_layout"]:
        extraction_options.append("Extract layout")
    cli_args["handwrite_signature_extraction"] = extraction_options

    # Download optional files if they are specified
    allow_list_key = arguments.get("allow_list_file")
    if allow_list_key:
        allow_list_path = os.path.join(INPUT_DIR, "allow_list.csv")
        download_file_from_s3(bucket_name, allow_list_key, allow_list_path)
        cli_args["allow_list_file"] = allow_list_path

    deny_list_key = arguments.get("deny_list_file")
    if deny_list_key:
        deny_list_path = os.path.join(INPUT_DIR, "deny_list.csv")
        download_file_from_s3(bucket_name, deny_list_key, deny_list_path)
        cli_args["deny_list_file"] = deny_list_path

    # 5. Execute the main application logic
    try:
        print("--- Starting CLI Redact Main Function ---")
        print(f"Arguments passed to cli_main: {cli_args}")
        cli_main(direct_mode_args=cli_args)
        print("--- CLI Redact Main Function Finished ---")
    except Exception as e:
        print(f"An error occurred during CLI execution: {e}")
        # Optionally, re-raise the exception to make the Lambda fail
        raise

    # 6. Upload results back to S3
    output_s3_prefix = f"output/{os.path.splitext(os.path.basename(input_key))[0]}"
    print(
        f"Uploading contents of {OUTPUT_DIR} to s3://{bucket_name}/{output_s3_prefix}/"
    )
    upload_directory_to_s3(OUTPUT_DIR, bucket_name, output_s3_prefix)

    return {
        "statusCode": 200,
        "body": json.dumps(
            f"Processing complete for {input_key}. Output saved to s3://{bucket_name}/{output_s3_prefix}/"
        ),
    }
