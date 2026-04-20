import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime

import pandas as pd

from tools.aws_functions import download_file_from_s3, export_outputs_to_s3
from tools.config import (
    ACCESS_LOGS_FOLDER,
    ALLOW_LIST_PATH,
    AWS_ACCESS_KEY,
    AWS_LLM_PII_OPTION,
    AWS_PII_OPTION,
    AWS_REGION,
    AWS_SECRET_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    CHOSEN_COMPREHEND_ENTITIES,
    CHOSEN_LLM_ENTITIES,
    CHOSEN_LLM_PII_INFERENCE_METHOD,
    CHOSEN_REDACT_ENTITIES,
    CLOUD_LLM_PII_MODEL_CHOICE,
    CLOUD_VLM_MODEL_CHOICE,
    COMPRESS_REDACTED_PDF,
    CUSTOM_ENTITIES,
    DEFAULT_COMBINE_PAGES,
    DEFAULT_COST_CODE,
    DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
    DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
    DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
    DEFAULT_INFERENCE_SERVER_PII_MODEL,
    DEFAULT_INFERENCE_SERVER_VLM_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_LOCAL_OCR_MODEL,
    DEFAULT_MIN_CONSECUTIVE_PAGES,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
    DENY_LIST_PATH,
    DIRECT_MODE_DEFAULT_USER,
    DISPLAY_FILE_NAMES_IN_LOGS,
    DO_INITIAL_TABULAR_DATA_CLEAN,
    DOCUMENT_REDACTION_BUCKET,
    EFFICIENT_OCR,
    EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX,
    EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION,
    EFFICIENT_OCR_MIN_WORDS,
    FEEDBACK_LOGS_FOLDER,
    FULL_COMPREHEND_ENTITY_LIST,
    FULL_ENTITY_LIST,
    FULL_LLM_ENTITY_LIST,
    GEMINI_API_KEY,
    GRADIO_TEMP_DIR,
    HYBRID_TEXTRACT_BEDROCK_VLM,
    IMAGES_DPI,
    INFERENCE_SERVER_API_URL,
    INFERENCE_SERVER_PII_OPTION,
    INPUT_FOLDER,
    LLM_MAX_NEW_TOKENS,
    LLM_PII_INFERENCE_METHODS,
    LLM_TEMPERATURE,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    OCR_FIRST_PASS_MAX_WORKERS,
    OUTPUT_FOLDER,
    OVERWRITE_EXISTING_OCR_RESULTS,
    PADDLE_MODEL_PATH,
    PREPROCESS_LOCAL_OCR_IMAGES,
    REMOVE_DUPLICATE_ROWS,
    RETURN_REDACTED_PDF,
    RUN_AWS_FUNCTIONS,
    S3_OUTPUTS_BUCKET,
    S3_OUTPUTS_FOLDER,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SAVE_OUTPUTS_TO_S3,
    SAVE_PAGE_OCR_VISUALISATIONS,
    SESSION_OUTPUT_FOLDER,
    SPACY_MODEL_PATH,
    SUMMARY_PAGE_GROUP_MAX_WORKERS,
    TEXTRACT_JOBS_LOCAL_LOC,
    TEXTRACT_JOBS_S3_LOC,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    USAGE_LOGS_FOLDER,
    USE_GREEDY_DUPLICATE_DETECTION,
    WHOLE_PAGE_REDACTION_LIST_PATH,
    convert_string_to_boolean,
)


def _generate_session_hash() -> str:
    """Generate a unique session hash for logging purposes."""
    return str(uuid.uuid4())[:8]


def _sanitize_folder_name(folder_name: str, max_length: int = 50) -> str:
    """
    Sanitize folder name for S3 compatibility.

    Replaces 'strange' characters (anything that's not alphanumeric, dash, underscore, or full stop)
    with underscores, and limits the length to max_length characters.

    Args:
        folder_name: Original folder name to sanitize
        max_length: Maximum length for the folder name (default: 50)

    Returns:
        Sanitized folder name
    """
    if not folder_name:
        return folder_name

    # Replace any character that's not alphanumeric, dash, underscore, or full stop with underscore
    # This handles @, commas, exclamation marks, spaces, etc.
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", folder_name)

    # Limit length to max_length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def get_username_and_folders(
    username: str = "",
    output_folder_textbox: str = OUTPUT_FOLDER,
    input_folder_textbox: str = INPUT_FOLDER,
    session_output_folder: bool = SESSION_OUTPUT_FOLDER,
    textract_document_upload_input_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    textract_document_upload_output_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    s3_textract_document_logs_subfolder: str = TEXTRACT_JOBS_S3_LOC,
    local_textract_document_logs_subfolder: str = TEXTRACT_JOBS_LOCAL_LOC,
):

    # Generate session hash for logging. Either from input user name or generated
    if username:
        out_session_hash = username
    else:
        out_session_hash = _generate_session_hash()

    # Sanitize session hash for S3 compatibility (especially important for S3 folder paths)
    sanitized_session_hash = _sanitize_folder_name(out_session_hash)

    if session_output_folder:
        output_folder = output_folder_textbox + sanitized_session_hash + "/"
        input_folder = input_folder_textbox + sanitized_session_hash + "/"

        textract_document_upload_input_folder = (
            textract_document_upload_input_folder + "/" + sanitized_session_hash
        )
        textract_document_upload_output_folder = (
            textract_document_upload_output_folder + "/" + sanitized_session_hash
        )

        s3_textract_document_logs_subfolder = (
            s3_textract_document_logs_subfolder + "/" + sanitized_session_hash
        )
        local_textract_document_logs_subfolder = (
            local_textract_document_logs_subfolder + "/" + sanitized_session_hash + "/"
        )

    else:
        output_folder = output_folder_textbox
        input_folder = input_folder_textbox

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(input_folder):
        os.mkdir(input_folder)

    return (
        out_session_hash,
        output_folder,
        out_session_hash,
        input_folder,
        textract_document_upload_input_folder,
        textract_document_upload_output_folder,
        s3_textract_document_logs_subfolder,
        local_textract_document_logs_subfolder,
    )


def _get_env_list(env_var_name: str) -> list[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('"', "").replace("'", "")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(",") if s.strip()]


def _download_s3_file_if_needed(
    file_path: str, default_filename: str = "downloaded_file"
) -> str:
    """
    Download a file from S3 if the path starts with 's3://' or 'S3://', otherwise return the path as-is.

    Args:
        file_path: File path (either local or S3 URL)
        default_filename: Default filename to use if S3 key doesn't have a filename

    Returns:
        Local file path (downloaded from S3 or original path)
    """
    if not file_path:
        return file_path

    # Check for S3 URL (case-insensitive)
    file_path_stripped = file_path.strip()
    file_path_upper = file_path_stripped.upper()
    if not file_path_upper.startswith("S3://"):
        return file_path

    # Use GRADIO_TEMP_DIR if available, otherwise use INPUT_FOLDER as fallback
    temp_dir = GRADIO_TEMP_DIR if GRADIO_TEMP_DIR else INPUT_FOLDER
    os.makedirs(temp_dir, exist_ok=True)

    # Parse S3 URL: s3://bucket/key (preserve original case for bucket/key)
    # Remove 's3://' prefix (case-insensitive)
    s3_path = (
        file_path_stripped.split("://", 1)[1]
        if "://" in file_path_stripped
        else file_path_stripped
    )
    # Split bucket and key (first '/' separates bucket from key)
    if "/" in s3_path:
        bucket_name_s3, s3_key = s3_path.split("/", 1)
    else:
        # If no key provided, use bucket name as key (unlikely but handle it)
        bucket_name_s3 = s3_path
        s3_key = ""

    # Get the filename from the S3 key
    filename = os.path.basename(s3_key) if s3_key else bucket_name_s3
    if not filename:
        filename = default_filename

    # Create local file path in temp directory
    local_file_path = os.path.join(temp_dir, filename)

    # Download file from S3
    try:
        download_file_from_s3(
            bucket_name=bucket_name_s3,
            key=s3_key,
            local_file_path_and_name=local_file_path,
        )
        print(f"S3 file downloaded successfully: {file_path} -> {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading file from S3 ({file_path}): {e}")
        raise Exception(f"Failed to download file from S3: {e}")


def _build_s3_output_folder(
    s3_outputs_folder: str,
    session_hash: str,
    save_to_user_folders: bool,
) -> str:
    """
    Build the S3 output folder path with session hash and date suffix if needed.

    Args:
        s3_outputs_folder: Base S3 folder path
        session_hash: Session hash/username
        save_to_user_folders: Whether to append session hash to folder path

    Returns:
        Final S3 folder path with session hash and date suffix
    """
    if not s3_outputs_folder:
        return ""

    # Append session hash if save_to_user_folders is enabled
    if save_to_user_folders and session_hash:
        sanitized_session_hash = _sanitize_folder_name(session_hash)
        s3_outputs_folder = (
            s3_outputs_folder.rstrip("/") + "/" + sanitized_session_hash + "/"
        )
    else:
        # Ensure trailing slash
        if not s3_outputs_folder.endswith("/"):
            s3_outputs_folder = s3_outputs_folder + "/"

    # Append today's date (YYYYMMDD/)
    today_suffix = datetime.now().strftime("%Y%m%d") + "/"
    s3_outputs_folder = s3_outputs_folder.rstrip("/") + "/" + today_suffix

    return s3_outputs_folder


# Add custom spacy recognisers to the Comprehend list, so that local Spacy model can be used to pick up e.g. titles, streetnames, UK postcodes that are sometimes missed by comprehend
CHOSEN_COMPREHEND_ENTITIES.extend(CUSTOM_ENTITIES)
FULL_COMPREHEND_ENTITY_LIST.extend(CUSTOM_ENTITIES)

chosen_redact_entities = CHOSEN_REDACT_ENTITIES
full_entity_list = FULL_ENTITY_LIST
chosen_comprehend_entities = CHOSEN_COMPREHEND_ENTITIES
full_comprehend_entity_list = FULL_COMPREHEND_ENTITY_LIST
chosen_llm_entities = CHOSEN_LLM_ENTITIES
full_llm_entity_list = FULL_LLM_ENTITY_LIST
default_handwrite_signature_checkbox = DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX


# --- CLI parser and main ---


def build_cli_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI ArgumentParser (shared by main(), Agent API, and tests)."""
    parser = argparse.ArgumentParser(
        description="A versatile CLI for redacting PII from PDF/image files and anonymising Word/tabular data.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:

To run these, you need to do the following:

- Open a terminal window

- CD to the app folder that contains this file (cli_redact.py)

- Load the virtual environment using either conda or venv depending on your setup

- Run one of the example commands below

- Look in the output/ folder to see output files:

# Redaction

## Redact a PDF with default settings (local OCR):
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf

## Extract text from a PDF only (i.e. no redaction), using local OCR:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --redact_whole_page_file example_data/partnership_toolkit_redact_some_pages.csv --pii_detector None

## Extract text from a PDF only (i.e. no redaction), using local OCR, with a whole page redaction list:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --redact_whole_page_file example_data/partnership_toolkit_redact_some_pages.csv --pii_detector Local --local_redact_entities CUSTOM

## Redact a PDF with allow list (local OCR) and custom list of redaction entities:
python cli_redact.py --input_file example_data/graduate-job-example-cover-letter.pdf --allow_list_file example_data/test_allow_list_graduate.csv --local_redact_entities TITLES PERSON DATE_TIME

## Redact a PDF with limited pages and text extraction method (local text) with custom fuzzy matching:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --deny_list_file example_data/Partnership-Agreement-Toolkit_test_deny_list_para_single_spell.csv --local_redact_entities CUSTOM_FUZZY --page_min 1 --page_max 3 --ocr_method "Local text" --fuzzy_mistakes 3

## Redaction with custom deny list, allow list, and whole page redaction list:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --deny_list_file example_data/partnership_toolkit_redact_custom_deny_list.csv --redact_whole_page_file example_data/partnership_toolkit_redact_some_pages.csv --allow_list_file example_data/test_allow_list_partnership.csv

## Redact an image:
python cli_redact.py --input_file example_data/example_complaint_letter.jpg

## Anonymise csv file with specific columns:
python cli_redact.py --input_file example_data/combined_case_notes.csv --text_columns "Case Note" "Client" --anon_strategy replace_redacted

## Anonymise csv file with a different strategy (remove text completely):
python cli_redact.py --input_file example_data/combined_case_notes.csv --text_columns "Case Note" "Client" --anon_strategy redact

## Anonymise Excel file, remove text completely:
python cli_redact.py --input_file example_data/combined_case_notes.xlsx --text_columns "Case Note" "Client" --excel_sheets combined_case_notes --anon_strategy redact

## Anonymise a word document:
python cli_redact.py --input_file "example_data/Bold minimalist professional cover letter.docx" --anon_strategy replace_redacted

# Redaction with AWS services:

## Use Textract and Comprehend:
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf --ocr_method "AWS Textract" --pii_detector "AWS Comprehend"

# LLM PII identification (entity subset and custom instructions)

## Redact with LLM PII entity subset (NAME, EMAIL_ADDRESS, etc.) and custom instructions:
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf --llm_redact_entities NAME EMAIL_ADDRESS PHONE_NUMBER ADDRESS CUSTOM --custom_llm_instructions "Do not redact the name of the university."

## Redact with custom LLM instructions only (use default LLM entities from config):
python cli_redact.py --input_file example_data/graduate-job-example-cover-letter.pdf --custom_llm_instructions "Redact all company names with the label COMPANY_NAME."

## Redact specific pages with AWS OCR and signature extraction:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --page_min 6 --page_max 7 --ocr_method "AWS Textract" --handwrite_signature_extraction "Extract handwriting" "Extract signatures"

## Redact with AWS OCR and additional layout extraction options:
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --ocr_method "AWS Textract" --extract_layout

# Duplicate page detection

## Find duplicate pages in OCR files:
python cli_redact.py --task deduplicate --input_file example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv --duplicate_type pages --similarity_threshold 0.95

## Find duplicate in OCR files at the line level:
python cli_redact.py --task deduplicate --input_file example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv --duplicate_type pages --similarity_threshold 0.95 --combine_pages False --min_word_count 3

## Find duplicate rows in tabular data:
python cli_redact.py --task deduplicate --input_file example_data/Lambeth_2030-Our_Future_Our_Lambeth.pdf.csv --duplicate_type tabular --text_columns "text" --similarity_threshold 0.95

# AWS Textract whole document analysis

## Submit document to Textract for basic text analysis:
python cli_redact.py --task textract --textract_action submit --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf

## Submit document to Textract for analysis with signature extraction (Job ID will be printed to the console, you need this to retrieve the results):
python cli_redact.py --task textract --textract_action submit --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --extract_signatures 

## Retrieve Textract results by job ID (returns a .json file output):
python cli_redact.py --task textract --textract_action retrieve --job_id 12345678-1234-1234-1234-123456789012

## List recent Textract jobs:
python cli_redact.py --task textract --textract_action list

# Document summarisation

# Summarise from a PDF with AWS Bedrock
python cli_redact.py --task summarise --input_file example_data/example_data/Partnership-Agreement-Toolkit_0_0.pdf --summarisation_inference_method "LLM (AWS Bedrock)"

## Summarise document(s) from OCR output CSV(s) using AWS Bedrock:
python cli_redact.py --task summarise --input_file example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_ocr_output.csv --summarisation_inference_method "LLM (AWS Bedrock)"

## Summarise with local LLM and detailed format:
python cli_redact.py --task summarise --input_file example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_ocr_output.csv --summarisation_inference_method "Local transformers LLM" --summarisation_format detailed

## Summarise with additional context and instructions (concise format):
python cli_redact.py --task summarise --input_file example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_ocr_output.csv --summarisation_context "This is a partnership agreement" --summarisation_additional_instructions "Focus on key obligations and termination clauses" --summarisation_format concise

## Summarise multiple OCR CSV files:
python cli_redact.py --task summarise --input_file example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_ocr_output.csv example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_ocr_output_textract.csv --summarisation_inference_method "LLM (AWS Bedrock)"

# Combine review PDFs

## Merge redaction comments from multiple '_redactions_for_review' PDFs into one file:
python cli_redact.py --task combine_review_pdfs --input_file path/to/review1.pdf path/to/review2.pdf --output_dir output/

""",
    )

    # --- Task Selection ---
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task",
        choices=[
            "redact",
            "deduplicate",
            "textract",
            "summarise",
            "combine_review_pdfs",
            "export_review_redaction_overlay",
            "export_review_page_ocr_visualisation",
        ],
        default="redact",
        help="Task to perform: redact (PII redaction/anonymisation), deduplicate (find duplicate content), textract (AWS Textract batch operations), summarise (LLM-based document summarisation from OCR CSV files), combine_review_pdfs (merge redaction comments from multiple '_redactions_for_review' PDFs into one file), export_review_redaction_overlay (write a redaction overlay JPEG for a page image + boxes JSON), or export_review_page_ocr_visualisation (write an OCR visualisation PNG for a page image + OCR JSON).",
    )

    # --- General Arguments (apply to all file types) ---
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--input_file",
        nargs="+",
        help="Path to the input file(s) to process. Separate multiple files with a space, and use quotes if there are spaces in the file name.",
    )
    general_group.add_argument(
        "--output_dir", default=OUTPUT_FOLDER, help="Directory for all output files."
    )
    general_group.add_argument(
        "--input_dir", default=INPUT_FOLDER, help="Directory for all input files."
    )

    export_group = parser.add_argument_group(
        "Review export (page image visualisations)"
    )
    export_group.add_argument(
        "--page_image_path",
        default="",
        help="Path to a single page raster image (PNG/JPG) used as underlay for export tasks.",
    )
    export_group.add_argument(
        "--page_number",
        type=int,
        default=1,
        help="1-based page number (used for naming).",
    )
    export_group.add_argument(
        "--doc_base_name",
        default="review",
        help="Basename for output file naming (e.g. document name without extension).",
    )
    export_group.add_argument(
        "--boxes_json_path",
        default="",
        help="Path to JSON file containing a list of annotator-style boxes for overlay export.",
    )
    export_group.add_argument(
        "--review_df_json_path",
        default="",
        help="Optional path to JSON file containing review dataframe records (list[dict]) for stable label ordering/pattern mapping.",
    )
    export_group.add_argument(
        "--label_abbrev_chars",
        type=int,
        default=-1,
        help="Optional override: draw N leading label characters on overlay image (use -1 to use config default).",
    )
    export_group.add_argument(
        "--ocr_results_json_path",
        default="",
        help="Path to JSON file containing OCR-with-words results dict for OCR visualisation export.",
    )
    general_group.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="Language of the document content."
    )
    general_group.add_argument(
        "--allow_list",
        default=ALLOW_LIST_PATH,
        help="Path to a CSV file with words to exclude from redaction.",
    )
    general_group.add_argument(
        "--pii_detector",
        choices=[LOCAL_PII_OPTION, AWS_PII_OPTION, "None"],
        default=LOCAL_PII_OPTION,
        help="Core PII detection method (Local or AWS Comprehend, or None).",
    )
    general_group.add_argument(
        "--username", default=DIRECT_MODE_DEFAULT_USER, help="Username for the session."
    )
    general_group.add_argument(
        "--save_to_user_folders",
        default=SESSION_OUTPUT_FOLDER,
        help="Whether to save to user folders or not.",
    )

    general_group.add_argument(
        "--local_redact_entities",
        nargs="+",
        choices=full_entity_list,
        default=chosen_redact_entities,
        help=f"Local redaction entities to use. Default: {chosen_redact_entities}. Full list: {full_entity_list}.",
    )

    general_group.add_argument(
        "--aws_redact_entities",
        nargs="+",
        choices=full_comprehend_entity_list,
        default=chosen_comprehend_entities,
        help=f"AWS redaction entities to use. Default: {chosen_comprehend_entities}. Full list: {full_comprehend_entity_list}.",
    )

    general_group.add_argument(
        "--aws_access_key", default=AWS_ACCESS_KEY, help="Your AWS Access Key ID."
    )
    general_group.add_argument(
        "--aws_secret_key", default=AWS_SECRET_KEY, help="Your AWS Secret Access Key."
    )
    general_group.add_argument(
        "--cost_code", default=DEFAULT_COST_CODE, help="Cost code for tracking usage."
    )
    general_group.add_argument(
        "--aws_region", default=AWS_REGION, help="AWS region for cloud services."
    )
    general_group.add_argument(
        "--s3_bucket",
        default=DOCUMENT_REDACTION_BUCKET,
        help="S3 bucket name for cloud operations.",
    )
    general_group.add_argument(
        "--save_outputs_to_s3",
        default=SAVE_OUTPUTS_TO_S3,
        help="Upload output files (redacted PDFs, anonymized documents, etc.) to S3 after processing.",
    )
    general_group.add_argument(
        "--s3_outputs_folder",
        default=S3_OUTPUTS_FOLDER,
        help="S3 folder (key prefix) for saving output files. If left blank, outputs will not be uploaded even if --save_outputs_to_s3 is enabled.",
    )
    general_group.add_argument(
        "--s3_outputs_bucket",
        default=S3_OUTPUTS_BUCKET,
        help="S3 bucket name for output files (defaults to --s3_bucket if not specified).",
    )
    general_group.add_argument(
        "--do_initial_clean",
        default=DO_INITIAL_TABULAR_DATA_CLEAN,
        help="Perform initial text cleaning for tabular data.",
    )
    general_group.add_argument(
        "--save_logs_to_csv",
        default=SAVE_LOGS_TO_CSV,
        help="Save processing logs to CSV files.",
    )
    general_group.add_argument(
        "--save_logs_to_dynamodb",
        default=SAVE_LOGS_TO_DYNAMODB,
        help="Save processing logs to DynamoDB.",
    )
    general_group.add_argument(
        "--display_file_names_in_logs",
        default=DISPLAY_FILE_NAMES_IN_LOGS,
        help="Include file names in log outputs.",
    )
    general_group.add_argument(
        "--upload_logs_to_s3",
        default=RUN_AWS_FUNCTIONS,
        help="Upload log files to S3 after processing.",
    )
    general_group.add_argument(
        "--s3_logs_prefix",
        default=S3_USAGE_LOGS_FOLDER,
        help="S3 prefix for usage log files.",
    )
    general_group.add_argument(
        "--feedback_logs_folder",
        default=FEEDBACK_LOGS_FOLDER,
        help="Directory for feedback log files.",
    )
    general_group.add_argument(
        "--access_logs_folder",
        default=ACCESS_LOGS_FOLDER,
        help="Directory for access log files.",
    )
    general_group.add_argument(
        "--usage_logs_folder",
        default=USAGE_LOGS_FOLDER,
        help="Directory for usage log files.",
    )
    general_group.add_argument(
        "--paddle_model_path",
        default=PADDLE_MODEL_PATH,
        help="Directory for PaddleOCR model storage.",
    )
    general_group.add_argument(
        "--spacy_model_path",
        default=SPACY_MODEL_PATH,
        help="Directory for spaCy model storage.",
    )

    # --- PDF/Image Redaction Arguments ---
    pdf_group = parser.add_argument_group(
        "PDF/Image Redaction Options (.pdf, .png, .jpg)"
    )
    pdf_group.add_argument(
        "--ocr_method",
        choices=["AWS Textract", "Local OCR", "Local text"],
        default="Local OCR",
        help="OCR method for text extraction from images.",
    )
    pdf_group.add_argument(
        "--page_min", type=int, default=0, help="First page to redact."
    )
    pdf_group.add_argument(
        "--page_max", type=int, default=0, help="Last page to redact."
    )
    pdf_group.add_argument(
        "--images_dpi",
        type=float,
        default=float(IMAGES_DPI),
        help="DPI for image processing.",
    )
    pdf_group.add_argument(
        "--chosen_local_ocr_model",
        choices=LOCAL_OCR_MODEL_OPTIONS,
        default=DEFAULT_LOCAL_OCR_MODEL,
        help="Local OCR model to use.",
    )
    pdf_group.add_argument(
        "--preprocess_local_ocr_images",
        default=PREPROCESS_LOCAL_OCR_IMAGES,
        help="Preprocess images before OCR.",
    )
    pdf_group.add_argument(
        "--compress_redacted_pdf",
        default=COMPRESS_REDACTED_PDF,
        help="Compress the final redacted PDF.",
    )
    pdf_group.add_argument(
        "--return_pdf_end_of_redaction",
        default=RETURN_REDACTED_PDF,
        help="Return PDF at end of redaction process.",
    )
    pdf_group.add_argument(
        "--deny_list_file",
        default=DENY_LIST_PATH,
        help="Custom words file to recognize for redaction.",
    )
    pdf_group.add_argument(
        "--allow_list_file",
        default=ALLOW_LIST_PATH,
        help="Custom words file to recognize for redaction.",
    )
    pdf_group.add_argument(
        "--redact_whole_page_file",
        default=WHOLE_PAGE_REDACTION_LIST_PATH,
        help="File for pages to redact completely.",
    )
    pdf_group.add_argument(
        "--handwrite_signature_extraction",
        nargs="+",
        default=default_handwrite_signature_checkbox,
        help='Handwriting and signature extraction options. Choose from "Extract handwriting", "Extract signatures".',
    )
    pdf_group.add_argument(
        "--extract_forms",
        action="store_true",
        help="Extract forms during Textract analysis.",
    )
    pdf_group.add_argument(
        "--extract_tables",
        action="store_true",
        help="Extract tables during Textract analysis.",
    )
    pdf_group.add_argument(
        "--extract_layout",
        action="store_true",
        help="Extract layout during Textract analysis.",
    )
    pdf_group.add_argument(
        "--vlm_model_choice",
        default=CLOUD_VLM_MODEL_CHOICE,
        help="VLM model choice for OCR (e.g., 'qwen.qwen3-vl-235b-a22b' for Bedrock, or model name for other providers).",
    )
    pdf_group.add_argument(
        "--inference_server_vlm_model",
        default=DEFAULT_INFERENCE_SERVER_VLM_MODEL,
        help="Inference server VLM model name for OCR.",
    )
    pdf_group.add_argument(
        "--inference_server_api_url",
        default=INFERENCE_SERVER_API_URL,
        help="Inference server API URL.",
    )
    pdf_group.add_argument(
        "--gemini_api_key",
        default=GEMINI_API_KEY,
        help="Google Gemini API key for VLM OCR.",
    )
    pdf_group.add_argument(
        "--azure_openai_api_key",
        default=AZURE_OPENAI_API_KEY,
        help="Azure OpenAI API key for VLM OCR.",
    )
    pdf_group.add_argument(
        "--azure_openai_endpoint",
        default=AZURE_OPENAI_INFERENCE_ENDPOINT,
        help="Azure OpenAI endpoint URL for VLM OCR.",
    )
    pdf_group.add_argument(
        "--efficient_ocr",
        action="store_true",
        default=None,
        help="Use efficient OCR: try selectable text first per page, run OCR only when needed (saves time/cost). Defaults to EFFICIENT_OCR config.",
    )
    pdf_group.add_argument(
        "--no_efficient_ocr",
        action="store_false",
        dest="efficient_ocr",
        help="Disable efficient OCR (use selected OCR method for all pages).",
    )
    pdf_group.add_argument(
        "--efficient_ocr_min_words",
        type=int,
        default=None,
        metavar="N",
        help="Minimum words on a page to use text-only route; below this use OCR. Defaults to EFFICIENT_OCR_MIN_WORDS config (e.g. 20).",
    )
    pdf_group.add_argument(
        "--efficient_ocr_min_image_coverage_fraction",
        type=float,
        default=None,
        metavar="F",
        help="Efficient OCR: min fraction of page area (0-1) for an embedded image to force OCR; 0 disables. Defaults to EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION config (e.g. 0.03).",
    )
    pdf_group.add_argument(
        "--efficient_ocr_min_embedded_image_px",
        type=int,
        default=None,
        metavar="N",
        help="Efficient OCR: min width and height (PDF points, ~px at 72 dpi) for an embedded image placement to count toward image-based OCR routing; 0 disables. Defaults to EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX config (e.g. 10).",
    )
    pdf_group.add_argument(
        "--ocr_first_pass_max_workers",
        type=int,
        default=None,
        metavar="N",
        help="Max threads for OCR first pass (1 = sequential). Defaults to OCR_FIRST_PASS_MAX_WORKERS config (e.g. 3).",
    )
    pdf_group.add_argument(
        "--hybrid_textract_bedrock_vlm",
        action="store_true",
        default=None,
        help="When using AWS Textract, re-run low-confidence lines with Bedrock VLM for higher quality. Defaults to HYBRID_TEXTRACT_BEDROCK_VLM config.",
    )
    pdf_group.add_argument(
        "--no_hybrid_textract_bedrock_vlm",
        action="store_false",
        dest="hybrid_textract_bedrock_vlm",
        help="Disable hybrid Textract + Bedrock VLM (use Textract only).",
    )
    pdf_group.add_argument(
        "--overwrite_existing_ocr_results",
        action="store_true",
        default=None,
        help="Ignore cached OCR JSON files and re-run OCR. Defaults to OVERWRITE_EXISTING_OCR_RESULTS config (e.g. False).",
    )
    pdf_group.add_argument(
        "--no_overwrite_existing_ocr_results",
        action="store_false",
        dest="overwrite_existing_ocr_results",
        help="Use existing OCR results when available (do not overwrite cached JSON).",
    )
    pdf_group.add_argument(
        "--save_page_ocr_visualisations",
        action="store_true",
        default=None,
        help="Save page OCR visualisations (debug bounding boxes). Defaults to SAVE_PAGE_OCR_VISUALISATIONS config.",
    )
    pdf_group.add_argument(
        "--no_save_page_ocr_visualisations",
        action="store_false",
        dest="save_page_ocr_visualisations",
        help="Do not save page OCR visualisations (debug bounding boxes).",
    )

    # --- LLM PII Detection Arguments ---
    llm_group = parser.add_argument_group("LLM PII Detection Options")
    llm_group.add_argument(
        "--llm_model_choice",
        default=CLOUD_LLM_PII_MODEL_CHOICE,
        help="LLM model choice for PII detection. Defaults to CLOUD_LLM_PII_MODEL_CHOICE for Bedrock. "
        "Note: The actual model used is determined by pii_identification_method - "
        "CLOUD_LLM_PII_MODEL_CHOICE for Bedrock, INFERENCE_SERVER_LLM_PII_MODEL_CHOICE for inference server, "
        "LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE for local transformers.",
    )
    llm_group.add_argument(
        "--llm_inference_method",
        choices=LLM_PII_INFERENCE_METHODS,
        default=CHOSEN_LLM_PII_INFERENCE_METHOD,
        help="LLM inference method for PII detection: aws-bedrock, local, inference-server, azure-openai, or gemini.",
    )
    llm_group.add_argument(
        "--inference_server_pii_model",
        default=DEFAULT_INFERENCE_SERVER_PII_MODEL,
        help="Inference server PII detection model name.",
    )
    llm_group.add_argument(
        "--llm_temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help="Temperature for LLM PII detection (lower = more deterministic).",
    )
    llm_group.add_argument(
        "--llm_max_tokens",
        type=int,
        default=LLM_MAX_NEW_TOKENS,
        help="Maximum tokens in LLM response for PII detection.",
    )
    llm_group.add_argument(
        "--llm_redact_entities",
        nargs="+",
        choices=full_llm_entity_list,
        default=chosen_llm_entities,
        help=f"Subset of entities for LLM PII detection (when pii_detector uses an LLM). Default: {chosen_llm_entities}. Full list: {full_llm_entity_list}.",
    )
    llm_group.add_argument(
        "--custom_llm_instructions",
        default="",
        help="Custom instructions for LLM-based entity detection (e.g. 'don't redact anything related to Mark Wilson' or 'redact all company names with the label COMPANY_NAME').",
    )

    # --- Word/Tabular Anonymisation Arguments ---
    tabular_group = parser.add_argument_group(
        "Word/Tabular Anonymisation Options (.docx, .csv, .xlsx)"
    )
    tabular_group.add_argument(
        "--anon_strategy",
        choices=[
            "redact",
            "redact completely",
            "replace_redacted",
            "entity_type",
            "encrypt",
            "hash",
            "replace with 'REDACTED'",
            "replace with <ENTITY_NAME>",
            "mask",
            "fake_first_name",
        ],
        default=DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
        help="The anonymisation strategy to apply.",
    )
    tabular_group.add_argument(
        "--text_columns",
        nargs="+",
        default=list(),
        help="A list of column names to anonymise or deduplicate in tabular data.",
    )
    tabular_group.add_argument(
        "--excel_sheets",
        nargs="+",
        default=list(),
        help="Specific Excel sheet names to process.",
    )
    tabular_group.add_argument(
        "--fuzzy_mistakes",
        type=int,
        default=DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
        help="Number of allowed spelling mistakes for fuzzy matching.",
    )
    tabular_group.add_argument(
        "--match_fuzzy_whole_phrase_bool",
        default=True,
        help="Match fuzzy whole phrase boolean.",
    )
    # --- Duplicate Detection Arguments ---
    duplicate_group = parser.add_argument_group("Duplicate Detection Options")
    duplicate_group.add_argument(
        "--duplicate_type",
        choices=["pages", "tabular"],
        default="pages",
        help="Type of duplicate detection: pages (for OCR files) or tabular (for CSV/Excel files).",
    )
    duplicate_group.add_argument(
        "--similarity_threshold",
        type=float,
        default=DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
        help="Similarity threshold (0-1) to consider content as duplicates.",
    )
    duplicate_group.add_argument(
        "--min_word_count",
        type=int,
        default=DEFAULT_MIN_WORD_COUNT,
        help="Minimum word count for text to be considered in duplicate analysis.",
    )
    duplicate_group.add_argument(
        "--min_consecutive_pages",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE_PAGES,
        help="Minimum number of consecutive pages to consider as a match.",
    )
    duplicate_group.add_argument(
        "--greedy_match",
        default=USE_GREEDY_DUPLICATE_DETECTION,
        help="Use greedy matching strategy for consecutive pages.",
    )
    duplicate_group.add_argument(
        "--combine_pages",
        default=DEFAULT_COMBINE_PAGES,
        help="Combine text from the same page number within a file. Alternative will enable line-level duplicate detection.",
    )
    duplicate_group.add_argument(
        "--remove_duplicate_rows",
        default=REMOVE_DUPLICATE_ROWS,
        help="Remove duplicate rows from the output.",
    )

    # --- Document Summarisation Arguments ---
    summarisation_group = parser.add_argument_group("Document Summarisation Options")
    summarisation_group.add_argument(
        "--summarisation_inference_method",
        choices=[
            AWS_LLM_PII_OPTION,
            LOCAL_TRANSFORMERS_LLM_PII_OPTION,
            INFERENCE_SERVER_PII_OPTION,
        ],
        default=AWS_LLM_PII_OPTION,
        help="LLM inference method for summarisation (same options as GUI).",
    )
    summarisation_group.add_argument(
        "--summarisation_temperature",
        type=float,
        default=0.6,
        help="Temperature for summarisation (0.0-2.0). Lower is more deterministic.",
    )
    summarisation_group.add_argument(
        "--summarisation_max_pages_per_group",
        type=int,
        default=30,
        help="Maximum pages per page-group summary (in addition to context-length limits).",
    )
    summarisation_group.add_argument(
        "--summary_page_group_max_workers",
        type=int,
        default=SUMMARY_PAGE_GROUP_MAX_WORKERS,
        metavar="N",
        help="Max threads for page-group summarisation (1 = sequential). Defaults to SUMMARY_PAGE_GROUP_MAX_WORKERS config (e.g. 1).",
    )
    summarisation_group.add_argument(
        "--summarisation_api_key",
        default="",
        help="API key for summarisation (if required by the chosen LLM).",
    )
    summarisation_group.add_argument(
        "--summarisation_context",
        default="",
        help="Additional context for summarisation (e.g. 'This is a consultation response document').",
    )
    summarisation_group.add_argument(
        "--summarisation_format",
        choices=["concise", "detailed"],
        default="detailed",
        help="Summary format: concise (key themes only) or detailed (as much detail as possible).",
    )
    summarisation_group.add_argument(
        "--summarisation_additional_instructions",
        default="",
        help="Additional summary instructions (e.g. 'Focus on key decisions and recommendations').",
    )

    # --- Textract Batch Operations Arguments ---
    textract_group = parser.add_argument_group("Textract Batch Operations Options")
    textract_group.add_argument(
        "--textract_action",
        choices=["submit", "retrieve", "list"],
        help="Textract action to perform: submit (submit document for analysis), retrieve (get results by job ID), or list (show recent jobs).",
    )
    textract_group.add_argument("--job_id", help="Textract job ID for retrieve action.")
    textract_group.add_argument(
        "--extract_signatures",
        action="store_true",
        help="Extract signatures during Textract analysis (for submit action).",
    )
    textract_group.add_argument(
        "--textract_bucket",
        default=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
        help="S3 bucket name for Textract operations (overrides default).",
    )
    textract_group.add_argument(
        "--textract_input_prefix",
        default=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
        help="S3 prefix for input files in Textract operations.",
    )
    textract_group.add_argument(
        "--textract_output_prefix",
        default=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
        help="S3 prefix for output files in Textract operations.",
    )
    textract_group.add_argument(
        "--s3_textract_document_logs_subfolder",
        default=TEXTRACT_JOBS_S3_LOC,
        help="S3 prefix for logs in Textract operations.",
    )
    textract_group.add_argument(
        "--local_textract_document_logs_subfolder",
        default=TEXTRACT_JOBS_LOCAL_LOC,
        help="Local prefix for logs in Textract operations.",
    )
    textract_group.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Polling interval in seconds for Textract job status.",
    )
    textract_group.add_argument(
        "--max_poll_attempts",
        type=int,
        default=120,
        help="Maximum number of polling attempts for Textract job completion.",
    )
    return parser


def get_cli_default_args_dict() -> dict:
    """All CLI flag defaults as a dict; merge agent overrides then call main(direct_mode_args=...)."""
    return vars(build_cli_argument_parser().parse_args([]))


def main(direct_mode_args={}):
    """
    A unified command-line interface to prepare, redact, and anonymise various document types.

    Args:
        direct_mode_args (dict, optional): Dictionary of arguments for direct mode execution.
                                          If provided, uses these instead of parsing command line arguments.
    """
    parser = build_cli_argument_parser()
    # Parse arguments - either from command line or direct mode
    if direct_mode_args:
        # Use direct mode arguments
        args = argparse.Namespace(**direct_mode_args)
    else:
        # Parse command line arguments
        args = parser.parse_args()

    # --- Handle S3 file downloads ---
    # Download input files from S3 if needed
    # Note: args.input_file is typically a list (from CLI nargs="+" or from direct mode)
    # but we also handle pipe-separated strings for compatibility
    if args.input_file:
        if isinstance(args.input_file, list):
            # Handle list of files (may include S3 paths)
            downloaded_files = []
            for file_path in args.input_file:
                downloaded_path = _download_s3_file_if_needed(file_path)
                downloaded_files.append(downloaded_path)
            args.input_file = downloaded_files
        elif isinstance(args.input_file, str):
            # Handle pipe-separated string (for direct mode compatibility)
            if "|" in args.input_file:
                file_list = [f.strip() for f in args.input_file.split("|") if f.strip()]
                downloaded_files = []
                for file_path in file_list:
                    downloaded_path = _download_s3_file_if_needed(file_path)
                    downloaded_files.append(downloaded_path)
                args.input_file = downloaded_files
            else:
                # Single file path
                args.input_file = [_download_s3_file_if_needed(args.input_file)]

    # Download other file arguments from S3 if needed
    if args.deny_list_file:
        args.deny_list_file = _download_s3_file_if_needed(
            args.deny_list_file, default_filename="downloaded_deny_list.csv"
        )
    if args.allow_list_file:
        args.allow_list_file = _download_s3_file_if_needed(
            args.allow_list_file, default_filename="downloaded_allow_list.csv"
        )
    if args.redact_whole_page_file:
        args.redact_whole_page_file = _download_s3_file_if_needed(
            args.redact_whole_page_file,
            default_filename="downloaded_redact_whole_page.csv",
        )

    # --- Initial Setup ---
    # Convert string boolean variables to boolean
    if args.preprocess_local_ocr_images == "True":
        args.preprocess_local_ocr_images = True
    else:
        args.preprocess_local_ocr_images = False
    if args.greedy_match == "True":
        args.greedy_match = True
    else:
        args.greedy_match = False
    if args.combine_pages == "True":
        args.combine_pages = True
    else:
        args.combine_pages = False
    if args.remove_duplicate_rows == "True":
        args.remove_duplicate_rows = True
    else:
        args.remove_duplicate_rows = False
    if args.return_pdf_end_of_redaction == "True":
        args.return_pdf_end_of_redaction = True
    else:
        args.return_pdf_end_of_redaction = False
    if args.compress_redacted_pdf == "True":
        args.compress_redacted_pdf = True
    else:
        args.compress_redacted_pdf = False
    if args.do_initial_clean == "True":
        args.do_initial_clean = True
    else:
        args.do_initial_clean = False
    if args.save_logs_to_csv == "True":
        args.save_logs_to_csv = True
    else:
        args.save_logs_to_csv = False
    if args.save_logs_to_dynamodb == "True":
        args.save_logs_to_dynamodb = True
    else:
        args.save_logs_to_dynamodb = False
    if args.display_file_names_in_logs == "True":
        args.display_file_names_in_logs = True
    else:
        args.display_file_names_in_logs = False
    if args.match_fuzzy_whole_phrase_bool == "True":
        args.match_fuzzy_whole_phrase_bool = True
    else:
        args.match_fuzzy_whole_phrase_bool = False
    # Convert save_to_user_folders to boolean (handles both string and boolean values)
    args.save_to_user_folders = convert_string_to_boolean(args.save_to_user_folders)
    # Convert save_outputs_to_s3 to boolean (handles both string and boolean values)
    args.save_outputs_to_s3 = convert_string_to_boolean(args.save_outputs_to_s3)

    # Combine extraction options
    extraction_options = (
        list(args.handwrite_signature_extraction)
        if args.handwrite_signature_extraction
        else []
    )
    if args.extract_forms:
        extraction_options.append("Extract forms")
    if args.extract_tables:
        extraction_options.append("Extract tables")
    if args.extract_layout:
        extraction_options.append("Extract layout")
    args.handwrite_signature_extraction = extraction_options

    if args.task in [
        "redact",
        "deduplicate",
        "summarise",
        "combine_review_pdfs",
    ]:
        if args.input_file:
            if isinstance(args.input_file, str):
                args.input_file = [args.input_file]

            _, file_extension = os.path.splitext(args.input_file[0])
            file_extension = file_extension.lower()
        else:
            raise ValueError(f"Error: --input_file is required for '{args.task}' task.")
    else:
        file_extension = ""

    # Initialise usage logger if logging is enabled
    usage_logger = None
    if args.save_logs_to_csv or args.save_logs_to_dynamodb:
        from tools.cli_usage_logger import create_cli_usage_logger

        try:
            usage_logger = create_cli_usage_logger(logs_folder=args.usage_logs_folder)
        except Exception as e:
            print(f"Warning: Could not initialise usage logger: {e}")

    # Get username and folders
    (
        session_hash,
        args.output_dir,
        _,
        args.input_dir,
        args.textract_input_prefix,
        args.textract_output_prefix,
        args.s3_textract_document_logs_subfolder,
        args.local_textract_document_logs_subfolder,
    ) = get_username_and_folders(
        username=args.username,
        output_folder_textbox=args.output_dir,
        input_folder_textbox=args.input_dir,
        session_output_folder=args.save_to_user_folders,
        textract_document_upload_input_folder=args.textract_input_prefix,
        textract_document_upload_output_folder=args.textract_output_prefix,
        s3_textract_document_logs_subfolder=args.s3_textract_document_logs_subfolder,
        local_textract_document_logs_subfolder=args.local_textract_document_logs_subfolder,
    )

    print(
        f"Conducting analyses with user {args.username}. Outputs will be saved to {args.output_dir}."
    )

    # Build S3 output folder path if S3 uploads are enabled
    s3_output_folder = ""
    if args.save_outputs_to_s3 and args.s3_outputs_folder:
        s3_output_folder = _build_s3_output_folder(
            s3_outputs_folder=args.s3_outputs_folder,
            session_hash=session_hash,
            save_to_user_folders=args.save_to_user_folders,
        )
        if s3_output_folder:
            print(f"S3 output folder: s3://{args.s3_outputs_bucket}/{s3_output_folder}")
    elif args.save_outputs_to_s3 and not args.s3_outputs_folder:
        print(
            "Warning: --save_outputs_to_s3 is enabled but --s3_outputs_folder is not set. Outputs will not be uploaded to S3."
        )

    # --- Route to the Correct Workflow Based on Task and File Type ---

    # Validate input_file requirement for tasks that need it
    if (
        args.task in ["redact", "deduplicate", "summarise", "combine_review_pdfs"]
        and not args.input_file
    ):
        print(f"Error: --input_file is required for '{args.task}' task.")
        return

    if args.ocr_method in ["Local OCR", "AWS Textract"]:
        args.prepare_images = True
    else:
        args.prepare_images = False

    from tools.cli_usage_logger import create_cli_usage_logger, log_redaction_usage

    # Task 1: Redaction/Anonymisation
    if args.task == "redact":

        # Workflow 1: PDF/Image Redaction
        if file_extension in [".pdf", ".png", ".jpg", ".jpeg"]:
            print("--- Detected PDF/Image file. Starting Redaction Workflow... ---")
            start_time = time.time()
            try:
                from tools.file_conversion import prepare_image_or_pdf
                from tools.file_redaction import run_redaction
                from tools.redaction_types import RedactionContext, RedactionOptions

                # Step 1: Prepare the document
                print("\nStep 1: Preparing document...")
                (
                    prep_summary,
                    prepared_pdf_paths,
                    image_file_paths,
                    _,
                    _,
                    pdf_doc,
                    image_annotations,
                    _,
                    original_cropboxes,
                    page_sizes,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = prepare_image_or_pdf(
                    file_paths=args.input_file,
                    text_extract_method=args.ocr_method,
                    all_line_level_ocr_results_df=pd.DataFrame(),
                    all_page_line_level_ocr_results_with_words_df=pd.DataFrame(),
                    first_loop_state=True,
                    prepare_for_review=False,
                    output_folder=args.output_dir,
                    input_folder=args.input_dir,
                    prepare_images=args.prepare_images,
                    page_min=args.page_min,
                    page_max=args.page_max,
                )
                print(f"Preparation complete. {prep_summary}")

                # Note: VLM and LLM clients are initialized inside run_redaction
                # based on text_extraction_method and pii_identification_method.
                # Model choices (vlm_model_choice, llm_model_choice) can be overridden via
                # environment variables (CLOUD_VLM_MODEL_CHOICE, CLOUD_LLM_PII_MODEL_CHOICE) before running the CLI.
                # For CLI, we pass inference_server_vlm_model and custom_llm_instructions.
                # Other LLM parameters (temperature, max_tokens, inference_method) are set via
                # environment variables or config defaults.

                # Step 2: Redact the prepared document
                print("\nStep 2: Running redaction...")
                (
                    output_summary,
                    output_files,
                    _,
                    _,
                    log_files,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    comprehend_query_number,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    page_sizes,
                    _,
                    _,
                    _,
                    _,
                    total_textract_query_number,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    vlm_model_name,
                    vlm_total_input_tokens,
                    vlm_total_output_tokens,
                    llm_model_name,
                    llm_total_input_tokens,
                    llm_total_output_tokens,
                    _,
                ) = run_redaction(
                    args.input_file,
                    RedactionOptions(
                        chosen_redact_entities=args.local_redact_entities,
                        chosen_redact_comprehend_entities=args.aws_redact_entities,
                        chosen_llm_entities=args.llm_redact_entities,
                        text_extraction_method=args.ocr_method,
                        in_allow_list=args.allow_list_file,
                        in_deny_list=args.deny_list_file,
                        redact_whole_page_list=args.redact_whole_page_file,
                        page_min=args.page_min,
                        page_max=args.page_max,
                        handwrite_signature_checkbox=args.handwrite_signature_extraction,
                        max_fuzzy_spelling_mistakes_num=args.fuzzy_mistakes,
                        match_fuzzy_whole_phrase_bool=args.match_fuzzy_whole_phrase_bool,
                        pii_identification_method=args.pii_detector,
                        aws_access_key_textbox=args.aws_access_key,
                        aws_secret_key_textbox=args.aws_secret_key,
                        language=args.language,
                        output_folder=args.output_dir,
                        input_folder=args.input_dir,
                        custom_llm_instructions=args.custom_llm_instructions,
                        inference_server_vlm_model=(
                            args.inference_server_vlm_model
                            if args.inference_server_vlm_model
                            else DEFAULT_INFERENCE_SERVER_VLM_MODEL
                        ),
                        efficient_ocr=getattr(args, "efficient_ocr", EFFICIENT_OCR),
                        efficient_ocr_min_words=(
                            args.efficient_ocr_min_words
                            if getattr(args, "efficient_ocr_min_words", None)
                            is not None
                            else EFFICIENT_OCR_MIN_WORDS
                        ),
                        efficient_ocr_min_image_coverage_fraction=(
                            args.efficient_ocr_min_image_coverage_fraction
                            if getattr(
                                args,
                                "efficient_ocr_min_image_coverage_fraction",
                                None,
                            )
                            is not None
                            else EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION
                        ),
                        efficient_ocr_min_embedded_image_px=(
                            args.efficient_ocr_min_embedded_image_px
                            if getattr(
                                args, "efficient_ocr_min_embedded_image_px", None
                            )
                            is not None
                            else EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX
                        ),
                        ocr_first_pass_max_workers=(
                            args.ocr_first_pass_max_workers
                            if getattr(args, "ocr_first_pass_max_workers", None)
                            is not None
                            else OCR_FIRST_PASS_MAX_WORKERS
                        ),
                        hybrid_textract_bedrock_vlm=getattr(
                            args,
                            "hybrid_textract_bedrock_vlm",
                            HYBRID_TEXTRACT_BEDROCK_VLM,
                        ),
                        overwrite_existing_ocr_results=getattr(
                            args,
                            "overwrite_existing_ocr_results",
                            OVERWRITE_EXISTING_OCR_RESULTS,
                        ),
                        save_page_ocr_visualisations=(
                            getattr(args, "save_page_ocr_visualisations", None)
                            if getattr(args, "save_page_ocr_visualisations", None)
                            is not None
                            else SAVE_PAGE_OCR_VISUALISATIONS
                        ),
                    ),
                    RedactionContext(
                        prepared_pdf_file_paths=prepared_pdf_paths,
                        pdf_image_file_paths=image_file_paths,
                        pymupdf_doc=pdf_doc,
                        annotations_all_pages=image_annotations,
                        page_sizes=page_sizes,
                        document_cropboxes=original_cropboxes,
                    ),
                    # Note: bedrock_runtime, gemini_client, gemini_config, azure_openai_client
                    # are initialized inside run_redaction based on text_extraction_method
                    # but we can pass vlm_model_choice through custom_llm_instructions or other means
                    # The clients will be initialized in run_redaction based on the method
                )

                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Log usage data if logger is available
                if usage_logger:
                    try:
                        # Extract file name for logging
                        print("Saving logs to CSV")
                        doc_file_name = (
                            os.path.basename(args.input_file[0])
                            if args.display_file_names_in_logs
                            else "document"
                        )
                        data_file_name = ""  # Not applicable for PDF/image redaction

                        # Determine if this was a Textract API call
                        is_textract_call = args.ocr_method == "AWS Textract"

                        # Count pages (approximate from page_sizes if available)
                        total_pages = len(page_sizes) if page_sizes else 1

                        # Count API calls (approximate - would need to be tracked in the redaction function)
                        textract_queries = (
                            int(total_textract_query_number) if is_textract_call else 0
                        )
                        comprehend_queries = (
                            int(comprehend_query_number)
                            if args.pii_detector == "AWS Comprehend"
                            else 0
                        )

                        # Format handwriting/signature options
                        handwriting_signature = (
                            ", ".join(args.handwrite_signature_extraction)
                            if args.handwrite_signature_extraction
                            else ""
                        )

                        log_redaction_usage(
                            logger=usage_logger,
                            session_hash=session_hash,
                            doc_file_name=doc_file_name,
                            data_file_name=data_file_name,
                            time_taken=processing_time,
                            total_pages=total_pages,
                            textract_queries=textract_queries,
                            pii_method=args.pii_detector,
                            comprehend_queries=comprehend_queries,
                            cost_code=args.cost_code,
                            handwriting_signature=handwriting_signature,
                            text_extraction_method=args.ocr_method,
                            is_textract_call=is_textract_call,
                            task=args.task,
                            save_to_dynamodb=args.save_logs_to_dynamodb,
                            save_to_s3=args.upload_logs_to_s3,
                            s3_bucket=args.s3_bucket,
                            s3_key_prefix=args.s3_logs_prefix,
                            vlm_model_name=vlm_model_name,
                            vlm_total_input_tokens=vlm_total_input_tokens,
                            vlm_total_output_tokens=vlm_total_output_tokens,
                            llm_model_name=llm_model_name,
                            llm_total_input_tokens=llm_total_input_tokens,
                            llm_total_output_tokens=llm_total_output_tokens,
                        )
                    except Exception as e:
                        print(f"Warning: Could not log usage data: {e}")

                print("\n--- Redaction Process Complete ---")
                print(f"Summary: {output_summary}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"\nOutput files saved to: {args.output_dir}")
                print("Generated Files:", sorted(output_files))
                if log_files:
                    print("Log Files:", sorted(log_files))

                # Upload output files to S3 if enabled
                if args.save_outputs_to_s3 and s3_output_folder and output_files:
                    print("\n--- Uploading output files to S3 ---")
                    try:
                        # Get base file name for organizing outputs
                        (
                            os.path.splitext(os.path.basename(args.input_file[0]))[0]
                            if args.input_file
                            else None
                        )
                        export_outputs_to_s3(
                            file_list_state=output_files,
                            s3_output_folder_state_value=s3_output_folder,
                            save_outputs_to_s3_flag=args.save_outputs_to_s3,
                            base_file_state=(
                                args.input_file[0] if args.input_file else None
                            ),
                            s3_bucket=args.s3_outputs_bucket,
                        )
                    except Exception as e:
                        print(f"Warning: Could not upload output files to S3: {e}")

            except Exception as e:
                print(
                    f"\nAn error occurred during the PDF/Image redaction workflow: {e}"
                )

        # Workflow 2: Word/Tabular Data Anonymisation
        elif file_extension in [".docx", ".xlsx", ".xls", ".csv", ".parquet"]:
            print(
                "--- Detected Word/Tabular file. Starting Anonymisation Workflow... ---"
            )
            start_time = time.time()
            try:
                from tools.data_anonymise import anonymise_files_with_open_text

                # Note: anonymise_files_with_open_text initializes LLM clients internally
                # based on pii_identification_method. LLM model choices and parameters
                # can be set via environment variables (CLOUD_LLM_PII_MODEL_CHOICE, LLM_TEMPERATURE, etc.)
                # before running the CLI.

                # Run the anonymisation function directly
                (
                    output_summary,
                    output_files,
                    _,
                    _,
                    log_files,
                    _,
                    processing_time,
                    comprehend_query_number,
                    _,
                    _,
                    _,
                ) = anonymise_files_with_open_text(
                    file_paths=args.input_file,
                    in_text="",  # Not used for file-based operations
                    anon_strategy=args.anon_strategy,
                    chosen_cols=args.text_columns,
                    chosen_redact_entities=args.local_redact_entities,
                    in_allow_list=args.allow_list_file,
                    in_excel_sheets=args.excel_sheets,
                    first_loop_state=True,
                    output_folder=args.output_dir,
                    in_deny_list=args.deny_list_file,
                    max_fuzzy_spelling_mistakes_num=args.fuzzy_mistakes,
                    pii_identification_method=args.pii_detector,
                    chosen_redact_comprehend_entities=args.aws_redact_entities,
                    aws_access_key_textbox=args.aws_access_key,
                    aws_secret_key_textbox=args.aws_secret_key,
                    language=args.language,
                    do_initial_clean=args.do_initial_clean,
                )

                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Log usage data if logger is available
                if usage_logger:
                    try:
                        print("Saving logs to CSV")
                        # Extract file name for logging
                        doc_file_name = ""  # Not applicable for tabular data
                        data_file_name = (
                            os.path.basename(args.input_file[0])
                            if args.display_file_names_in_logs
                            else "data_file"
                        )

                        # Determine if this was a Textract API call (not applicable for tabular)
                        is_textract_call = False

                        # Count pages (not applicable for tabular data)
                        total_pages = 0

                        # Count API calls (approximate - would need to be tracked in the anonymisation function)
                        textract_queries = 0  # Not applicable for tabular data
                        comprehend_queries = (
                            comprehend_query_number
                            if args.pii_detector == "AWS Comprehend"
                            else 0
                        )

                        # Format handwriting/signature options (not applicable for tabular)
                        handwriting_signature = ""

                        log_redaction_usage(
                            logger=usage_logger,
                            session_hash=session_hash,
                            doc_file_name=doc_file_name,
                            data_file_name=data_file_name,
                            time_taken=processing_time,
                            total_pages=total_pages,
                            textract_queries=textract_queries,
                            pii_method=args.pii_detector,
                            comprehend_queries=comprehend_queries,
                            cost_code=args.cost_code,
                            handwriting_signature=handwriting_signature,
                            text_extraction_method="tabular",  # Indicate this is tabular processing
                            is_textract_call=is_textract_call,
                            task=args.task,
                            save_to_dynamodb=args.save_logs_to_dynamodb,
                            save_to_s3=args.upload_logs_to_s3,
                            s3_bucket=args.s3_bucket,
                            s3_key_prefix=args.s3_logs_prefix,
                            vlm_model_name="",  # TODO: Track from perform_ocr
                            vlm_total_input_tokens=0,  # TODO: Track from perform_ocr
                            vlm_total_output_tokens=0,  # TODO: Track from perform_ocr
                            llm_model_name="",  # TODO: Track from anonymise_script
                            llm_total_input_tokens=0,  # TODO: Track from anonymise_script
                            llm_total_output_tokens=0,  # TODO: Track from anonymise_script
                        )
                    except Exception as e:
                        print(f"Warning: Could not log usage data: {e}")

                print("\n--- Anonymisation Process Complete ---")
                print(f"Summary: {output_summary}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"\nOutput files saved to: {args.output_dir}")
                print("Generated Files:", sorted(output_files))
                if log_files:
                    print("Log Files:", sorted(log_files))

                # Upload output files to S3 if enabled
                if args.save_outputs_to_s3 and s3_output_folder and output_files:
                    print("\n--- Uploading output files to S3 ---")
                    try:
                        export_outputs_to_s3(
                            file_list_state=output_files,
                            s3_output_folder_state_value=s3_output_folder,
                            save_outputs_to_s3_flag=args.save_outputs_to_s3,
                            base_file_state=(
                                args.input_file[0] if args.input_file else None
                            ),
                            s3_bucket=args.s3_outputs_bucket,
                        )
                    except Exception as e:
                        print(f"Warning: Could not upload output files to S3: {e}")

            except Exception as e:
                print(
                    f"\nAn error occurred during the Word/Tabular anonymisation workflow: {e}"
                )

        else:
            print(f"Error: Unsupported file type '{file_extension}' for redaction.")
            print("Supported types for redaction: .pdf, .png, .jpg, .jpeg")
            print(
                "Supported types for anonymisation: .docx, .xlsx, .xls, .csv, .parquet"
            )

    # Task 2: Duplicate Detection
    elif args.task == "deduplicate":
        print("--- Starting Duplicate Detection Workflow... ---")
        try:
            from tools.find_duplicate_pages import run_duplicate_analysis

            if args.duplicate_type == "pages":
                # Page duplicate detection
                if file_extension == ".csv":
                    print(
                        "--- Detected OCR CSV file. Starting Page Duplicate Detection... ---"
                    )

                    start_time = time.time()

                    if args.combine_pages is True:
                        print("Combining pages...")
                    else:
                        print("Using line-level duplicate detection...")

                    # Load the CSV file as a list for the duplicate analysis function
                    (
                        results_df,
                        output_paths,
                        full_data_by_file,
                        processing_time,
                        task_textbox,
                        _,
                        _,
                        _,
                    ) = run_duplicate_analysis(
                        files=args.input_file,
                        threshold=args.similarity_threshold,
                        min_words=args.min_word_count,
                        min_consecutive=args.min_consecutive_pages,
                        greedy_match=args.greedy_match,
                        combine_pages=args.combine_pages,
                        output_folder=args.output_dir,
                        all_page_line_level_ocr_results_df_base=pd.DataFrame(),
                        ocr_df_paths_list=[],
                    )

                    end_time = time.time()
                    processing_time = end_time - start_time

                    print("\n--- Page Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths:
                        print("Generated Files:", sorted(output_paths))

                    # Upload output files to S3 if enabled
                    if args.save_outputs_to_s3 and s3_output_folder and output_paths:
                        print("\n--- Uploading output files to S3 ---")
                        try:
                            export_outputs_to_s3(
                                file_list_state=output_paths,
                                s3_output_folder_state_value=s3_output_folder,
                                save_outputs_to_s3_flag=args.save_outputs_to_s3,
                                base_file_state=(
                                    args.input_file[0] if args.input_file else None
                                ),
                                s3_bucket=args.s3_outputs_bucket,
                            )
                        except Exception as e:
                            print(f"Warning: Could not upload output files to S3: {e}")

                    # Log usage for page deduplication (match app: doc name or "document", data blank)
                    if usage_logger:
                        try:
                            print("Saving logs to CSV")
                            doc_file_name = (
                                os.path.basename(args.input_file[0])
                                if args.display_file_names_in_logs and args.input_file
                                else "document"
                            )
                            data_file_name = ""  # Not applicable for page dedup
                            log_redaction_usage(
                                logger=usage_logger,
                                session_hash=session_hash,
                                doc_file_name=doc_file_name,
                                data_file_name=data_file_name,
                                time_taken=processing_time,
                                total_pages=0,
                                textract_queries=0,
                                comprehend_queries=0,
                                pii_method=args.pii_detector,
                                cost_code=args.cost_code,
                                handwriting_signature="",
                                text_extraction_method=args.ocr_method,
                                is_textract_call=False,
                                task=args.task,
                                save_to_dynamodb=args.save_logs_to_dynamodb,
                                save_to_s3=args.upload_logs_to_s3,
                                s3_bucket=args.s3_bucket,
                                s3_key_prefix=args.s3_logs_prefix,
                                vlm_model_name="",
                                vlm_total_input_tokens=0,
                                vlm_total_output_tokens=0,
                                llm_model_name="",
                                llm_total_input_tokens=0,
                                llm_total_output_tokens=0,
                            )
                        except Exception as e:
                            print(f"Warning: Could not log usage data: {e}")

                else:
                    print(
                        "Error: Page duplicate detection requires CSV files with OCR data."
                    )
                    print("Please provide a CSV file containing OCR output data.")

                    # Log usage data if logger is available
                    if usage_logger:
                        try:
                            # Extract file name for logging
                            print("Saving logs to CSV")
                            doc_file_name = (
                                os.path.basename(args.input_file[0])
                                if args.display_file_names_in_logs
                                else "document"
                            )
                            data_file_name = (
                                ""  # Not applicable for PDF/image redaction
                            )

                            # Determine if this was a Textract API call
                            is_textract_call = False

                            # Count pages (approximate from page_sizes if available)
                            total_pages = len(page_sizes) if page_sizes else 1

                            # Count API calls (approximate - would need to be tracked in the redaction function)
                            textract_queries = 0
                            comprehend_queries = 0

                            # Format handwriting/signature options
                            handwriting_signature = ""

                            log_redaction_usage(
                                logger=usage_logger,
                                session_hash=session_hash,
                                doc_file_name=doc_file_name,
                                data_file_name=data_file_name,
                                time_taken=processing_time,
                                total_pages=total_pages,
                                textract_queries=textract_queries,
                                pii_method=args.pii_detector,
                                comprehend_queries=comprehend_queries,
                                cost_code=args.cost_code,
                                handwriting_signature=handwriting_signature,
                                text_extraction_method=args.ocr_method,
                                is_textract_call=is_textract_call,
                                task=args.task,
                                save_to_dynamodb=args.save_logs_to_dynamodb,
                                save_to_s3=args.upload_logs_to_s3,
                                s3_bucket=args.s3_bucket,
                                s3_key_prefix=args.s3_logs_prefix,
                                vlm_model_name="",  # Not applicable for duplicate detection
                                vlm_total_input_tokens=0,
                                vlm_total_output_tokens=0,
                                llm_model_name="",  # Not applicable for duplicate detection
                                llm_total_input_tokens=0,
                                llm_total_output_tokens=0,
                            )
                        except Exception as e:
                            print(f"Warning: Could not log usage data: {e}")

            elif args.duplicate_type == "tabular":
                # Tabular duplicate detection
                from tools.find_duplicate_tabular import run_tabular_duplicate_detection

                if file_extension in [".csv", ".xlsx", ".xls", ".parquet"]:
                    print(
                        "--- Detected tabular file. Starting Tabular Duplicate Detection... ---"
                    )

                    start_time = time.time()

                    (
                        results_df,
                        output_paths,
                        full_data_by_file,
                        processing_time,
                        task_textbox,
                    ) = run_tabular_duplicate_detection(
                        files=args.input_file,
                        threshold=args.similarity_threshold,
                        min_words=args.min_word_count,
                        text_columns=args.text_columns,
                        output_folder=args.output_dir,
                        do_initial_clean_dup=args.do_initial_clean,
                        in_excel_tabular_sheets=args.excel_sheets,
                        remove_duplicate_rows=args.remove_duplicate_rows,
                    )

                    end_time = time.time()
                    processing_time = end_time - start_time

                    # Log usage data if logger is available
                    if usage_logger:
                        try:
                            # Extract file name for logging
                            print("Saving logs to CSV")
                            doc_file_name = ""  # Tabular dedup: no doc (match app)
                            data_file_name = (
                                os.path.basename(args.input_file[0])
                                if args.display_file_names_in_logs and args.input_file
                                else "data_file"
                            )

                            is_textract_call = False
                            total_pages = 0  # Tabular dedup: no page count (match app)
                            textract_queries = 0
                            comprehend_queries = 0
                            handwriting_signature = ""

                            log_redaction_usage(
                                logger=usage_logger,
                                session_hash=session_hash,
                                doc_file_name=doc_file_name,
                                data_file_name=data_file_name,
                                time_taken=processing_time,
                                total_pages=total_pages,
                                textract_queries=textract_queries,
                                pii_method=args.pii_detector,
                                comprehend_queries=comprehend_queries,
                                cost_code=args.cost_code,
                                handwriting_signature=handwriting_signature,
                                text_extraction_method=args.ocr_method,
                                is_textract_call=is_textract_call,
                                task=args.task,
                                save_to_dynamodb=args.save_logs_to_dynamodb,
                                save_to_s3=args.upload_logs_to_s3,
                                s3_bucket=args.s3_bucket,
                                s3_key_prefix=args.s3_logs_prefix,
                                vlm_model_name="",  # Not applicable for duplicate detection
                                vlm_total_input_tokens=0,
                                vlm_total_output_tokens=0,
                                llm_model_name="",  # Not applicable for duplicate detection
                                llm_total_input_tokens=0,
                                llm_total_output_tokens=0,
                            )
                        except Exception as e:
                            print(f"Warning: Could not log usage data: {e}")

                    print("\n--- Tabular Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths:
                        print("Generated Files:", sorted(output_paths))

                    # Upload output files to S3 if enabled
                    if args.save_outputs_to_s3 and s3_output_folder and output_paths:
                        print("\n--- Uploading output files to S3 ---")
                        try:
                            export_outputs_to_s3(
                                file_list_state=output_paths,
                                s3_output_folder_state_value=s3_output_folder,
                                save_outputs_to_s3_flag=args.save_outputs_to_s3,
                                base_file_state=(
                                    args.input_file[0] if args.input_file else None
                                ),
                                s3_bucket=args.s3_outputs_bucket,
                            )
                        except Exception as e:
                            print(f"Warning: Could not upload output files to S3: {e}")

                else:
                    print(
                        "Error: Tabular duplicate detection requires CSV, Excel, or Parquet files."
                    )
                    print("Supported types: .csv, .xlsx, .xls, .parquet")
            else:
                print(f"Error: Invalid duplicate type '{args.duplicate_type}'.")
                print("Valid options: 'pages' or 'tabular'")

        except Exception as e:
            print(f"\nAn error occurred during the duplicate detection workflow: {e}")

    # Task 3: Textract Batch Operations
    elif args.task == "textract":
        print("--- Starting Textract Batch Operations Workflow... ---")

        if not args.textract_action:
            print("Error: --textract_action is required for textract task.")
            print("Valid options: 'submit', 'retrieve', or 'list'")
            return

        try:
            if args.textract_action == "submit":
                from tools.textract_batch_call import (
                    analyse_document_with_textract_api,
                    load_in_textract_job_details,
                )

                # Submit document to Textract for analysis
                if not args.input_file:
                    print("Error: --input_file is required for submit action.")
                    return

                print(f"--- Submitting document to Textract: {args.input_file} ---")

                start_time = time.time()

                # Load existing job details
                job_df = load_in_textract_job_details(
                    load_s3_jobs_loc=args.s3_textract_document_logs_subfolder,
                    load_local_jobs_loc=args.local_textract_document_logs_subfolder,
                )

                # Determine signature extraction options
                signature_options = (
                    ["Extract handwriting", "Extract signatures"]
                    if args.extract_signatures
                    else ["Extract handwriting"]
                )

                # Use configured bucket or override
                textract_bucket = args.textract_bucket if args.textract_bucket else ""

                # Submit the job
                (
                    result_message,
                    job_id,
                    job_type,
                    successful_job_number,
                    is_textract_call,
                    total_pages,
                    task_textbox,
                ) = analyse_document_with_textract_api(
                    local_pdf_path=args.input_file,
                    s3_input_prefix=args.textract_input_prefix,
                    s3_output_prefix=args.textract_output_prefix,
                    job_df=job_df,
                    s3_bucket_name=textract_bucket,
                    general_s3_bucket_name=args.s3_bucket,
                    local_output_dir=args.output_dir,
                    handwrite_signature_checkbox=signature_options,
                    aws_region=args.aws_region,
                )

                end_time = time.time()
                processing_time = end_time - start_time

                print("\n--- Textract Job Submitted Successfully ---")
                print(f"Job ID: {job_id}")
                print(f"Job Type: {job_type}")
                print(f"Message: {result_message}")
                print(f"Results will be available in: {args.output_dir}")

                # Log usage data if logger is available
                if usage_logger:
                    try:
                        # Extract file name for logging
                        print("Saving logs to CSV")
                        doc_file_name = (
                            os.path.basename(args.input_file[0])
                            if args.display_file_names_in_logs
                            else "document"
                        )
                        data_file_name = ""

                        # Determine if this was a Textract API call
                        is_textract_call = True
                        args.ocr_method == "AWS Textract"

                        # Count API calls (approximate - would need to be tracked in the redaction function)
                        textract_queries = total_pages
                        comprehend_queries = 0

                        # Format handwriting/signature options
                        handwriting_signature = ""

                        log_redaction_usage(
                            logger=usage_logger,
                            session_hash=session_hash,
                            doc_file_name=doc_file_name,
                            data_file_name=data_file_name,
                            time_taken=processing_time,
                            total_pages=total_pages,
                            textract_queries=textract_queries,
                            pii_method=args.pii_detector,
                            comprehend_queries=comprehend_queries,
                            cost_code=args.cost_code,
                            handwriting_signature=handwriting_signature,
                            text_extraction_method=args.ocr_method,
                            is_textract_call=is_textract_call,
                            task=args.task,
                            save_to_dynamodb=args.save_logs_to_dynamodb,
                            save_to_s3=args.upload_logs_to_s3,
                            s3_bucket=args.s3_bucket,
                            s3_key_prefix=args.s3_logs_prefix,
                            vlm_model_name="",  # Not applicable for Textract submit
                            vlm_total_input_tokens=0,
                            vlm_total_output_tokens=0,
                            llm_model_name="",  # Not applicable for Textract submit
                            llm_total_input_tokens=0,
                            llm_total_output_tokens=0,
                        )
                    except Exception as e:
                        print(f"Warning: Could not log usage data: {e}")

            elif args.textract_action == "retrieve":
                print(f"--- Retrieving Textract results for Job ID: {args.job_id} ---")

                from tools.textract_batch_call import (
                    load_in_textract_job_details,
                    poll_whole_document_textract_analysis_progress_and_download,
                )

                # Retrieve results by job ID
                if not args.job_id:
                    print("Error: --job_id is required for retrieve action.")
                    return

                # Load existing job details to get job type
                print("Loading existing job details...")
                job_df = load_in_textract_job_details(
                    load_s3_jobs_loc=args.s3_textract_document_logs_subfolder,
                    load_local_jobs_loc=args.local_textract_document_logs_subfolder,
                )

                # Find job type from the dataframe
                job_type = "document_text_detection"  # default
                if not job_df.empty and "job_id" in job_df.columns:
                    matching_jobs = job_df.loc[job_df["job_id"] == args.job_id]
                    if not matching_jobs.empty and "job_type" in matching_jobs.columns:
                        job_type = matching_jobs.iloc[0]["job_type"]

                # Use configured bucket or override
                textract_bucket = args.textract_bucket if args.textract_bucket else ""

                # Poll for completion and download results
                print("Polling for completion and downloading results...")
                downloaded_file_path, job_status, updated_job_df, output_filename = (
                    poll_whole_document_textract_analysis_progress_and_download(
                        job_id=args.job_id,
                        job_type_dropdown=job_type,
                        s3_output_prefix=args.textract_output_prefix,
                        pdf_filename="",  # Will be determined from job details
                        job_df=job_df,
                        s3_bucket_name=textract_bucket,
                        load_s3_jobs_loc=args.s3_textract_document_logs_subfolder,
                        load_local_jobs_loc=args.local_textract_document_logs_subfolder,
                        local_output_dir=args.output_dir,
                        poll_interval_seconds=args.poll_interval,
                        max_polling_attempts=args.max_poll_attempts,
                    )
                )

                print("\n--- Textract Results Retrieved Successfully ---")
                print(f"Job Status: {job_status}")
                print(f"Downloaded File: {downloaded_file_path}")
                # print(f"Output Filename: {output_filename}")

            elif args.textract_action == "list":
                from tools.textract_batch_call import load_in_textract_job_details

                # List recent Textract jobs
                print("--- Listing Recent Textract Jobs ---")

                job_df = load_in_textract_job_details(
                    load_s3_jobs_loc=args.s3_textract_document_logs_subfolder,
                    load_local_jobs_loc=args.local_textract_document_logs_subfolder,
                )

                if job_df.empty:
                    print("No recent Textract jobs found.")
                else:
                    print(f"\nFound {len(job_df)} recent Textract jobs:")
                    print("-" * 80)
                    for _, job in job_df.iterrows():
                        print(f"Job ID: {job.get('job_id', 'N/A')}")
                        print(f"File: {job.get('file_name', 'N/A')}")
                        print(f"Type: {job.get('job_type', 'N/A')}")
                        print(f"Signatures: {job.get('signature_extraction', 'N/A')}")
                        print(f"Date: {job.get('job_date_time', 'N/A')}")
                        print("-" * 80)

            else:
                print(f"Error: Invalid textract_action '{args.textract_action}'.")
                print("Valid options: 'submit', 'retrieve', or 'list'")

        except Exception as e:
            print(f"\nAn error occurred during the Textract workflow: {e}")

    elif args.task == "summarise":
        print("--- Document Summarisation ---")
        try:
            from tools.cli_usage_logger import log_redaction_usage
            from tools.file_conversion import is_pdf
            from tools.summaries import (
                concise_summary_format_prompt,
                detailed_summary_format_prompt,
                load_csv_files_to_dataframe,
                summarise_document_wrapper,
            )

            # Map format choice to prompt string (same as GUI)
            format_map = {
                "concise": concise_summary_format_prompt,
                "detailed": detailed_summary_format_prompt,
            }
            summarise_format_radio = format_map.get(
                args.summarisation_format, detailed_summary_format_prompt
            )

            # Normalise input to list of paths
            input_paths = (
                [args.input_file]
                if isinstance(args.input_file, str)
                else list(args.input_file or [])
            )
            input_paths = [p for p in input_paths if p and str(p).strip()]

            # If any input is a PDF, extract text first then summarise (same as app.py)
            summarise_from_pdf = any(is_pdf(p) for p in input_paths)
            if summarise_from_pdf:
                pdf_path = next((p for p in input_paths if is_pdf(p)), None)
                if not pdf_path:
                    print("Error: No PDF path found in input files.")
                    return
                print(
                    f"Detected PDF input. Extracting text with '{args.ocr_method}' then summarising..."
                )
                from tools.file_conversion import prepare_image_or_pdf
                from tools.file_redaction import run_redaction
                from tools.redaction_types import RedactionContext, RedactionOptions

                prepare_images = args.ocr_method in ["Local OCR", "AWS Textract"]
                (
                    _prep_summary,
                    prepared_pdf_paths,
                    image_file_paths,
                    _,
                    _,
                    pdf_doc,
                    image_annotations,
                    _,
                    original_cropboxes,
                    page_sizes,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = prepare_image_or_pdf(
                    file_paths=[pdf_path],
                    text_extract_method=args.ocr_method,
                    all_line_level_ocr_results_df=pd.DataFrame(),
                    all_page_line_level_ocr_results_with_words_df=pd.DataFrame(),
                    first_loop_state=True,
                    prepare_for_review=False,
                    output_folder=args.output_dir,
                    input_folder=args.input_dir,
                    prepare_images=prepare_images,
                    page_min=args.page_min,
                    page_max=args.page_max,
                )
                print(f"  {_prep_summary}")

                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    ocr_df,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = run_redaction(
                    [pdf_path],
                    RedactionOptions(
                        chosen_redact_entities=args.local_redact_entities or [],
                        chosen_redact_comprehend_entities=args.aws_redact_entities
                        or [],
                        chosen_llm_entities=args.llm_redact_entities or [],
                        text_extraction_method=args.ocr_method,
                        in_allow_list=args.allow_list_file,
                        in_deny_list=args.deny_list_file,
                        redact_whole_page_list=args.redact_whole_page_file,
                        page_min=args.page_min,
                        page_max=args.page_max,
                        handwrite_signature_checkbox=args.handwrite_signature_extraction
                        or [],
                        max_fuzzy_spelling_mistakes_num=getattr(
                            args, "fuzzy_mistakes", DEFAULT_FUZZY_SPELLING_MISTAKES_NUM
                        ),
                        match_fuzzy_whole_phrase_bool=getattr(
                            args, "match_fuzzy_whole_phrase_bool", True
                        ),
                        pii_identification_method=args.pii_detector or "Local",
                        aws_access_key_textbox=args.aws_access_key or "",
                        aws_secret_key_textbox=args.aws_secret_key or "",
                        language=args.language,
                        output_folder=args.output_dir,
                        input_folder=args.input_dir,
                        custom_llm_instructions=args.custom_llm_instructions or "",
                        inference_server_vlm_model=(
                            getattr(args, "inference_server_vlm_model", None)
                            or DEFAULT_INFERENCE_SERVER_VLM_MODEL
                        ),
                        efficient_ocr=getattr(args, "efficient_ocr", EFFICIENT_OCR),
                        efficient_ocr_min_words=(
                            getattr(args, "efficient_ocr_min_words", None)
                            or EFFICIENT_OCR_MIN_WORDS
                        ),
                        efficient_ocr_min_image_coverage_fraction=(
                            getattr(
                                args, "efficient_ocr_min_image_coverage_fraction", None
                            )
                            if getattr(
                                args, "efficient_ocr_min_image_coverage_fraction", None
                            )
                            is not None
                            else EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION
                        ),
                        efficient_ocr_min_embedded_image_px=(
                            getattr(args, "efficient_ocr_min_embedded_image_px", None)
                            if getattr(
                                args, "efficient_ocr_min_embedded_image_px", None
                            )
                            is not None
                            else EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX
                        ),
                        ocr_first_pass_max_workers=(
                            getattr(args, "ocr_first_pass_max_workers", None)
                            or OCR_FIRST_PASS_MAX_WORKERS
                        ),
                        hybrid_textract_bedrock_vlm=getattr(
                            args,
                            "hybrid_textract_bedrock_vlm",
                            HYBRID_TEXTRACT_BEDROCK_VLM,
                        ),
                        overwrite_existing_ocr_results=getattr(
                            args,
                            "overwrite_existing_ocr_results",
                            OVERWRITE_EXISTING_OCR_RESULTS,
                        ),
                        save_page_ocr_visualisations=(
                            getattr(args, "save_page_ocr_visualisations", None)
                            if getattr(args, "save_page_ocr_visualisations", None)
                            is not None
                            else SAVE_PAGE_OCR_VISUALISATIONS
                        ),
                        text_extraction_only=True,
                    ),
                    RedactionContext(
                        prepared_pdf_file_paths=prepared_pdf_paths,
                        pdf_image_file_paths=image_file_paths,
                        pymupdf_doc=pdf_doc,
                        annotations_all_pages=image_annotations,
                        page_sizes=page_sizes,
                        document_cropboxes=original_cropboxes,
                    ),
                )

                if ocr_df is None or (
                    isinstance(ocr_df, pd.DataFrame) and ocr_df.empty
                ):
                    print("Error: No OCR text extracted from PDF. Cannot summarise.")
                    return

                # Derive file_name from PDF path (same as app.py _file_name_from_pdf_path)
                basename = os.path.basename(pdf_path)
                file_name = os.path.splitext(basename)[0][:20]
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    file_name = file_name.replace(char, "_")
                file_name = file_name if file_name else "document"
            else:
                # CSV path: load OCR CSV file(s)
                ocr_df = load_csv_files_to_dataframe(input_paths)
                if ocr_df is None or ocr_df.empty:
                    print(
                        "Error: No valid OCR data (page, line, text columns) in input file(s)."
                    )
                    return

                first_path = input_paths[0] if input_paths else ""
                if first_path:
                    basename = os.path.basename(first_path)
                    file_name = os.path.splitext(basename)[0][:20]
                    invalid_chars = '<>:"/\\|?*'
                    for char in invalid_chars:
                        file_name = file_name.replace(char, "_")
                    file_name = file_name if file_name else "document"
                else:
                    file_name = "document"

            (
                output_files,
                status_message,
                llm_model_name,
                llm_total_input_tokens,
                llm_total_output_tokens,
                summary_display_text,
                elapsed_seconds,
            ) = summarise_document_wrapper(
                ocr_df,
                args.output_dir,
                args.summarisation_inference_method,
                args.summarisation_api_key or "",
                args.summarisation_temperature,
                file_name,
                args.summarisation_context or "",
                args.aws_access_key or "",
                args.aws_secret_key or "",
                "",
                AZURE_OPENAI_INFERENCE_ENDPOINT or "",
                summarise_format_radio,
                args.summarisation_additional_instructions or "",
                args.summarisation_max_pages_per_group,
                None,
            )

            processing_time = elapsed_seconds

            print(f"\n{status_message}")
            if output_files:
                print("Output files:")
                for p in output_files:
                    print(f"  {p}")
            if summary_display_text:
                print("\n--- Summary ---")
                print(
                    summary_display_text[:2000]
                    + ("..." if len(summary_display_text) > 2000 else "")
                )

            # Usage logging (same fields as GUI summarisation success callback)
            if usage_logger:
                try:
                    first_input = input_paths[0] if input_paths else ""
                    doc_file_name = (
                        os.path.basename(first_input)
                        if args.display_file_names_in_logs and first_input
                        else "document"
                    )
                    data_file_name = ""
                    total_pages = (
                        int(ocr_df["page"].max())
                        if "page" in ocr_df.columns and not ocr_df.empty
                        else 0
                    )

                    log_redaction_usage(
                        logger=usage_logger,
                        session_hash=session_hash,
                        doc_file_name=doc_file_name,
                        data_file_name=data_file_name,
                        time_taken=processing_time,
                        total_pages=total_pages,
                        textract_queries=0,
                        pii_method=args.summarisation_inference_method,
                        comprehend_queries=0,
                        cost_code=args.cost_code,
                        handwriting_signature="",
                        text_extraction_method="",
                        is_textract_call=False,
                        task="summarisation",
                        save_to_dynamodb=args.save_logs_to_dynamodb,
                        save_to_s3=args.upload_logs_to_s3,
                        s3_bucket=args.s3_bucket,
                        s3_key_prefix=args.s3_logs_prefix,
                        vlm_model_name="",
                        vlm_total_input_tokens=0,
                        vlm_total_output_tokens=0,
                        llm_model_name=llm_model_name or "",
                        llm_total_input_tokens=llm_total_input_tokens or 0,
                        llm_total_output_tokens=llm_total_output_tokens or 0,
                    )
                except Exception as e:
                    print(f"Warning: Could not log usage data: {e}")

        except Exception as e:
            print(f"\nAn error occurred during summarisation: {e}")
            import traceback

            traceback.print_exc()

    elif args.task == "combine_review_pdfs":
        print("--- Combine review PDFs ---")
        try:
            from tools.file_conversion import combine_review_pdf_files

            paths = (
                [args.input_file]
                if isinstance(args.input_file, str)
                else list(args.input_file)
            )
            if len(paths) < 2:
                print("Error: combine_review_pdfs requires at least 2 input PDF files.")
                return
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            result = combine_review_pdf_files(paths, output_folder=out_dir)
            if result:
                print(f"Combined PDF saved to: {result[0]}")
            else:
                print("No output produced (empty file list or no valid paths).")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"\nAn error occurred while combining review PDFs: {e}")
            import traceback

            traceback.print_exc()

    elif args.task == "export_review_redaction_overlay":
        print("--- Export review redaction overlay image ---")
        try:
            from tools.redaction_review import visualise_review_redaction_boxes

            if not args.page_image_path:
                print(
                    "Error: --page_image_path is required for export_review_redaction_overlay."
                )
                return
            if not args.boxes_json_path:
                print(
                    "Error: --boxes_json_path is required for export_review_redaction_overlay."
                )
                return

            with open(args.boxes_json_path, "r", encoding="utf-8") as f:
                boxes = json.load(f)
            if not isinstance(boxes, list) or not boxes:
                print("Error: boxes JSON must be a non-empty list of box dicts.")
                return

            review_df = pd.DataFrame()
            if args.review_df_json_path:
                with open(args.review_df_json_path, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                if isinstance(recs, list) and recs:
                    review_df = pd.DataFrame(recs)

            annotator = {"image": args.page_image_path, "boxes": boxes}
            out = visualise_review_redaction_boxes(
                annotator,
                review_df=review_df,
                output_folder=args.output_dir,
                page_number=int(args.page_number or 1),
                doc_base_name=str(args.doc_base_name or "review"),
                label_abbrev_chars=(
                    None
                    if int(args.label_abbrev_chars) < 0
                    else int(args.label_abbrev_chars)
                ),
            )
            if out:
                print(f"Overlay image written to: {out}")
            else:
                print("No output produced (invalid image/boxes or write failed).")
        except Exception as e:
            print(f"\nAn error occurred while exporting overlay image: {e}")
            import traceback

            traceback.print_exc()

    elif args.task == "export_review_page_ocr_visualisation":
        print("--- Export review page OCR visualisation image ---")
        try:
            from PIL import Image

            from tools.file_redaction import visualise_ocr_words_bounding_boxes
            from tools.helper_functions import get_file_name_without_type
            from tools.secure_path_utils import sanitize_filename

            if not args.page_image_path:
                print(
                    "Error: --page_image_path is required for export_review_page_ocr_visualisation."
                )
                return
            if not args.ocr_results_json_path:
                print(
                    "Error: --ocr_results_json_path is required for export_review_page_ocr_visualisation."
                )
                return

            with open(args.ocr_results_json_path, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
            if not isinstance(ocr_results, dict) or not ocr_results:
                print("Error: ocr_results JSON must be a non-empty dict.")
                return

            base = get_file_name_without_type(os.path.basename(str(args.doc_base_name)))
            if not base or not str(base).strip():
                base = "review"
            safe_base = sanitize_filename(str(base))
            image_name = f"{safe_base}_page{int(args.page_number or 1)}.png"

            log_paths: list[str] = []
            log_paths = visualise_ocr_words_bounding_boxes(
                Image.open(args.page_image_path).convert("RGB"),
                ocr_results,
                image_name=image_name,
                output_folder=args.output_dir,
                visualisation_folder="review_ocr_visualisations",
                add_legend=True,
                log_files_output_paths=log_paths,
            )
            if log_paths:
                print(f"OCR visualisation written to: {log_paths[-1]}")
            else:
                print("No output produced (invalid image/ocr_results or write failed).")
        except Exception as e:
            print(f"\nAn error occurred while exporting OCR visualisation image: {e}")
            import traceback

            traceback.print_exc()

    else:
        print(f"Error: Invalid task '{args.task}'.")
        print(
            "Valid options: 'redact', 'deduplicate', 'textract', 'summarise', 'combine_review_pdfs', 'export_review_redaction_overlay', or 'export_review_page_ocr_visualisation'"
        )


if __name__ == "__main__":
    main()
