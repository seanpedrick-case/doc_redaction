import argparse
import os
import time
import uuid

import pandas as pd

from tools.config import (
    ACCESS_LOGS_FOLDER,
    ALLOW_LIST_PATH,
    AWS_ACCESS_KEY,
    AWS_PII_OPTION,
    AWS_REGION,
    AWS_SECRET_KEY,
    CHOSEN_COMPREHEND_ENTITIES,
    CHOSEN_LOCAL_OCR_MODEL,
    CHOSEN_REDACT_ENTITIES,
    COMPRESS_REDACTED_PDF,
    CUSTOM_ENTITIES,
    DEFAULT_COMBINE_PAGES,
    DEFAULT_COST_CODE,
    DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
    DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
    DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONSECUTIVE_PAGES,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
    DENY_LIST_PATH,
    DIRECT_MODE_DEFAULT_USER,
    DISPLAY_FILE_NAMES_IN_LOGS,
    DO_INITIAL_TABULAR_DATA_CLEAN,
    DOCUMENT_REDACTION_BUCKET,
    FEEDBACK_LOGS_FOLDER,
    FULL_COMPREHEND_ENTITY_LIST,
    FULL_ENTITY_LIST,
    IMAGES_DPI,
    INPUT_FOLDER,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    OUTPUT_FOLDER,
    PADDLE_MODEL_PATH,
    PREPROCESS_LOCAL_OCR_IMAGES,
    REMOVE_DUPLICATE_ROWS,
    RETURN_REDACTED_PDF,
    RUN_AWS_FUNCTIONS,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SESSION_OUTPUT_FOLDER,
    SPACY_MODEL_PATH,
    TEXTRACT_JOBS_LOCAL_LOC,
    TEXTRACT_JOBS_S3_LOC,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    USAGE_LOGS_FOLDER,
    USE_GREEDY_DUPLICATE_DETECTION,
    WHOLE_PAGE_REDACTION_LIST_PATH,
)


def _generate_session_hash() -> str:
    """Generate a unique session hash for logging purposes."""
    return str(uuid.uuid4())[:8]


def get_username_and_folders(
    username: str = "",
    output_folder_textbox: str = OUTPUT_FOLDER,
    input_folder_textbox: str = INPUT_FOLDER,
    session_output_folder: str = SESSION_OUTPUT_FOLDER,
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

    if session_output_folder == "True" or session_output_folder is True:
        output_folder = output_folder_textbox + out_session_hash + "/"
        input_folder = input_folder_textbox + out_session_hash + "/"

        textract_document_upload_input_folder = (
            textract_document_upload_input_folder + "/" + out_session_hash
        )
        textract_document_upload_output_folder = (
            textract_document_upload_output_folder + "/" + out_session_hash
        )

        s3_textract_document_logs_subfolder = (
            s3_textract_document_logs_subfolder + "/" + out_session_hash
        )
        local_textract_document_logs_subfolder = (
            local_textract_document_logs_subfolder + "/" + out_session_hash + "/"
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


# Add custom spacy recognisers to the Comprehend list, so that local Spacy model can be used to pick up e.g. titles, streetnames, UK postcodes that are sometimes missed by comprehend
CHOSEN_COMPREHEND_ENTITIES.extend(CUSTOM_ENTITIES)
FULL_COMPREHEND_ENTITY_LIST.extend(CUSTOM_ENTITIES)

chosen_redact_entities = CHOSEN_REDACT_ENTITIES
full_entity_list = FULL_ENTITY_LIST
chosen_comprehend_entities = CHOSEN_COMPREHEND_ENTITIES
full_comprehend_entity_list = FULL_COMPREHEND_ENTITY_LIST
default_handwrite_signature_checkbox = DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX


# --- Main CLI Function ---
def main(direct_mode_args={}):
    """
    A unified command-line interface to prepare, redact, and anonymise various document types.

    Args:
        direct_mode_args (dict, optional): Dictionary of arguments for direct mode execution.
                                          If provided, uses these instead of parsing command line arguments.
    """
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

## Use Textract and Comprehend::
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf --ocr_method "AWS Textract" --pii_detector "AWS Comprehend"

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

""",
    )

    # --- Task Selection ---
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task",
        choices=["redact", "deduplicate", "textract"],
        default="redact",
        help="Task to perform: redact (PII redaction/anonymisation), deduplicate (find duplicate content), or textract (AWS Textract batch operations).",
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
        default=CHOSEN_LOCAL_OCR_MODEL,
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
    # Parse arguments - either from command line or direct mode
    if direct_mode_args:
        # Use direct mode arguments
        args = argparse.Namespace(**direct_mode_args)
    else:
        # Parse command line arguments
        args = parser.parse_args()

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
    if args.save_to_user_folders == "True":
        args.save_to_user_folders = True
    else:
        args.save_to_user_folders = False

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

    if args.task in ["redact", "deduplicate"]:
        if args.input_file:
            if isinstance(args.input_file, str):
                args.input_file = [args.input_file]

            _, file_extension = os.path.splitext(args.input_file[0])
            file_extension = file_extension.lower()
        else:
            raise ValueError("Error: --input_file is required for 'redact' task.")

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

    # --- Route to the Correct Workflow Based on Task and File Type ---

    # Validate input_file requirement for tasks that need it
    if args.task in ["redact", "deduplicate"] and not args.input_file:
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
                from tools.file_redaction import choose_and_run_redactor

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
                    total_textract_query_number,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = choose_and_run_redactor(
                    file_paths=args.input_file,
                    prepared_pdf_file_paths=prepared_pdf_paths,
                    pdf_image_file_paths=image_file_paths,
                    chosen_redact_entities=args.local_redact_entities,
                    chosen_redact_comprehend_entities=args.aws_redact_entities,
                    text_extraction_method=args.ocr_method,
                    in_allow_list=args.allow_list_file,
                    in_deny_list=args.deny_list_file,
                    redact_whole_page_list=args.redact_whole_page_file,
                    first_loop_state=True,
                    page_min=args.page_min,
                    page_max=args.page_max,
                    handwrite_signature_checkbox=args.handwrite_signature_extraction,
                    max_fuzzy_spelling_mistakes_num=args.fuzzy_mistakes,
                    match_fuzzy_whole_phrase_bool=args.match_fuzzy_whole_phrase_bool,
                    pymupdf_doc=pdf_doc,
                    annotations_all_pages=image_annotations,
                    page_sizes=page_sizes,
                    document_cropboxes=original_cropboxes,
                    pii_identification_method=args.pii_detector,
                    aws_access_key_textbox=args.aws_access_key,
                    aws_secret_key_textbox=args.aws_secret_key,
                    language=args.language,
                    output_folder=args.output_dir,
                    input_folder=args.input_dir,
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
                    ) = run_duplicate_analysis(
                        files=args.input_file,
                        threshold=args.similarity_threshold,
                        min_words=args.min_word_count,
                        min_consecutive=args.min_consecutive_pages,
                        greedy_match=args.greedy_match,
                        combine_pages=args.combine_pages,
                        output_folder=args.output_dir,
                    )

                    end_time = time.time()
                    processing_time = end_time - start_time

                    print("\n--- Page Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths:
                        print("Generated Files:", sorted(output_paths))

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
                            doc_file_name = ""
                            data_file_name = (
                                os.path.basename(args.input_file[0])
                                if args.display_file_names_in_logs
                                else "data_file"
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
                            )
                        except Exception as e:
                            print(f"Warning: Could not log usage data: {e}")

                    print("\n--- Tabular Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths:
                        print("Generated Files:", sorted(output_paths))

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

    else:
        print(f"Error: Invalid task '{args.task}'.")
        print("Valid options: 'redact', 'deduplicate', or 'textract'")


if __name__ == "__main__":
    main()
