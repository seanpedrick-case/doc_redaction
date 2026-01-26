import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import pandas as pd
import spaces
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from gradio_image_annotation import image_annotator

from tools.auth import authenticate_user
from tools.aws_functions import (
    download_file_from_s3,
    export_outputs_to_s3,
    upload_log_file_to_s3,
)
from tools.config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    ACCESS_LOGS_FOLDER,
    ALLOW_LIST_PATH,
    ALLOWED_HOSTS,
    ALLOWED_ORIGINS,
    AWS_ACCESS_KEY,
    AWS_LLM_PII_OPTION,
    AWS_PII_OPTION,
    AWS_REGION,
    AWS_SECRET_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    BEDROCK_VLM_TEXT_EXTRACT_OPTION,
    CHOSEN_COMPREHEND_ENTITIES,
    CHOSEN_LLM_ENTITIES,
    CHOSEN_LLM_PII_INFERENCE_METHOD,
    CHOSEN_LOCAL_MODEL_INTRO_TEXT,
    CHOSEN_LOCAL_OCR_MODEL,
    CHOSEN_REDACT_ENTITIES,
    CLOUD_LLM_PII_MODEL_CHOICE,
    CLOUD_VLM_MODEL_CHOICE,
    COGNITO_AUTH,
    CONFIG_FOLDER,
    COST_CODES_PATH,
    CSV_ACCESS_LOG_HEADERS,
    CSV_FEEDBACK_LOG_HEADERS,
    CSV_USAGE_LOG_HEADERS,
    CUSTOM_BOX_COLOUR,
    DEFAULT_CONCURRENCY_LIMIT,
    DEFAULT_COST_CODE,
    DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
    DEFAULT_EXCEL_SHEETS,
    DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
    DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
    DEFAULT_INFERENCE_SERVER_PII_MODEL,
    DEFAULT_INFERENCE_SERVER_VLM_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_LANGUAGE_FULL_NAME,
    DEFAULT_MIN_CONSECUTIVE_PAGES,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_PAGE_MAX,
    DEFAULT_PAGE_MIN,
    DEFAULT_PII_DETECTION_MODEL,
    DEFAULT_SEARCH_QUERY,
    DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
    DEFAULT_TEXT_COLUMNS,
    DEFAULT_TEXT_EXTRACTION_MODEL,
    DENY_LIST_PATH,
    DIRECT_MODE_ANON_STRATEGY,
    DIRECT_MODE_CHOSEN_LOCAL_OCR_MODEL,
    DIRECT_MODE_COMBINE_PAGES,
    DIRECT_MODE_COMPRESS_REDACTED_PDF,
    DIRECT_MODE_DEFAULT_USER,
    DIRECT_MODE_DUPLICATE_TYPE,
    DIRECT_MODE_EXTRACT_FORMS,
    DIRECT_MODE_EXTRACT_LAYOUT,
    DIRECT_MODE_EXTRACT_SIGNATURES,
    DIRECT_MODE_EXTRACT_TABLES,
    DIRECT_MODE_FUZZY_MISTAKES,
    DIRECT_MODE_GREEDY_MATCH,
    DIRECT_MODE_IMAGES_DPI,
    DIRECT_MODE_INPUT_FILE,
    DIRECT_MODE_JOB_ID,
    # Additional direct mode configuration options
    DIRECT_MODE_LANGUAGE,
    DIRECT_MODE_MATCH_FUZZY_WHOLE_PHRASE_BOOL,
    DIRECT_MODE_MIN_CONSECUTIVE_PAGES,
    DIRECT_MODE_MIN_WORD_COUNT,
    DIRECT_MODE_OCR_METHOD,
    DIRECT_MODE_OUTPUT_DIR,
    DIRECT_MODE_PAGE_MAX,
    DIRECT_MODE_PAGE_MIN,
    DIRECT_MODE_PII_DETECTOR,
    DIRECT_MODE_PREPROCESS_LOCAL_OCR_IMAGES,
    DIRECT_MODE_REMOVE_DUPLICATE_ROWS,
    DIRECT_MODE_RETURN_PDF_END_OF_REDACTION,
    DIRECT_MODE_SIMILARITY_THRESHOLD,
    DIRECT_MODE_TASK,
    DIRECT_MODE_TEXTRACT_ACTION,
    DISPLAY_FILE_NAMES_IN_LOGS,
    DO_INITIAL_TABULAR_DATA_CLEAN,
    DOCUMENT_REDACTION_BUCKET,
    DYNAMODB_ACCESS_LOG_HEADERS,
    DYNAMODB_FEEDBACK_LOG_HEADERS,
    DYNAMODB_USAGE_LOG_HEADERS,
    ENFORCE_COST_CODES,
    EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT,
    FASTAPI_ROOT_PATH,
    FAVICON_PATH,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    FEEDBACK_LOG_FILE_NAME,
    FEEDBACK_LOGS_FOLDER,
    FILE_INPUT_HEIGHT,
    FULL_COMPREHEND_ENTITY_LIST,
    FULL_ENTITY_LIST,
    FULL_LLM_ENTITY_LIST,
    GEMINI_API_KEY,
    GET_COST_CODES,
    GET_DEFAULT_ALLOW_LIST,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_TEMP_DIR,
    HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS,
    HOST_NAME,
    INFERENCE_SERVER_API_URL,
    INFERENCE_SERVER_PII_OPTION,
    INPUT_FOLDER,
    INTRO_TEXT,
    LANGUAGE_CHOICES,
    LLM_PII_MAX_TOKENS,
    LLM_PII_TEMPERATURE,
    LOAD_PREVIOUS_TEXTRACT_JOBS_S3,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    LOG_FILE_NAME,
    MAPPED_LANGUAGE_CHOICES,
    MAX_FILE_SIZE,
    MAX_OPEN_TEXT_CHARACTERS,
    MAX_QUEUE_SIZE,
    MPLCONFIGDIR,
    NO_REDACTION_PII_OPTION,
    OUTPUT_COST_CODES_PATH,
    OUTPUT_FOLDER,
    PADDLE_MODEL_PATH,
    PII_DETECTION_MODELS,
    REMOVE_DUPLICATE_ROWS,
    ROOT_PATH,
    RUN_AWS_FUNCTIONS,
    RUN_DIRECT_MODE,
    RUN_FASTAPI,
    RUN_MCP_SERVER,
    S3_ACCESS_LOGS_FOLDER,
    S3_ALLOW_LIST_PATH,
    S3_COST_CODES_PATH,
    S3_FEEDBACK_LOGS_FOLDER,
    S3_OUTPUTS_FOLDER,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SAVE_OUTPUTS_TO_S3,
    SESSION_OUTPUT_FOLDER,
    SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER,
    SHOW_AWS_EXAMPLES,
    SHOW_AWS_TEXT_EXTRACTION_OPTIONS,
    SHOW_COSTS,
    SHOW_DIFFICULT_OCR_EXAMPLES,
    SHOW_EXAMPLES,
    SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS,
    SHOW_LANGUAGE_SELECTION,
    SHOW_LOCAL_OCR_MODEL_OPTIONS,
    SHOW_OCR_GUI_OPTIONS,
    SHOW_PII_IDENTIFICATION_OPTIONS,
    SHOW_QUICKSTART,
    SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS,
    SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS,
    SPACY_MODEL_PATH,
    TABULAR_PII_DETECTION_MODELS,
    TESSERACT_TEXT_EXTRACT_OPTION,
    TEXT_EXTRACTION_MODELS,
    TEXTRACT_JOBS_LOCAL_LOC,
    TEXTRACT_JOBS_S3_INPUT_LOC,
    TEXTRACT_JOBS_S3_LOC,
    TEXTRACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_FILE_NAME,
    USAGE_LOGS_FOLDER,
    USE_GREEDY_DUPLICATE_DETECTION,
    WHOLE_PAGE_REDACTION_LIST_PATH,
)
from tools.custom_csvlogger import CSVLogger_custom
from tools.data_anonymise import anonymise_files_with_open_text
from tools.file_conversion import get_input_file_names, prepare_image_or_pdf
from tools.file_redaction import choose_and_run_redactor
from tools.find_duplicate_pages import (
    apply_whole_page_redactions_from_list,
    create_annotation_objects_from_duplicates,
    exclude_match,
    handle_selection_and_preview,
    run_duplicate_analysis,
    run_full_search_and_analysis,
)
from tools.find_duplicate_tabular import (
    clean_tabular_duplicates,
    handle_tabular_row_selection,
    run_tabular_duplicate_detection,
)
from tools.helper_functions import (
    all_outputs_file_download_fn,
    calculate_aws_costs,
    calculate_time_taken,
    check_for_existing_textract_file,
    check_for_relevant_ocr_output_with_words,
    custom_regex_load,
    enforce_cost_codes,
    ensure_folder_exists,
    get_connection_params,
    load_all_output_files,
    load_in_default_allow_list,
    load_in_default_cost_codes,
    merge_csv_files,
    put_columns_in_df,
    reset_aws_call_vars,
    reset_base_dataframe,
    reset_data_vars,
    reset_ocr_base_dataframe,
    reset_ocr_with_words_base_dataframe,
    reset_review_vars,
    reset_state_vars,
    reveal_feedback_buttons,
    update_cost_code_dataframe_from_dropdown_select,
    update_language_dropdown,
)
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.quickstart import (
    handle_main_pii_method_selection,
    handle_main_text_extract_method_selection,
    handle_pii_method_selection,
    handle_redaction_method_selection,
    handle_step_2_next,
    handle_step_3_next,
    handle_text_extract_method_selection,
    route_walkthrough_files,
    update_step_2_on_data_file_upload,
    update_step_3_tabular_visibility,
    update_step_4_visibility,
)
from tools.redaction_review import (
    apply_redactions_to_review_df_and_files,
    convert_df_to_xfdf,
    convert_xfdf_to_dataframe,
    create_annotation_objects_from_filtered_ocr_results_with_words,
    decrease_page,
    df_select_callback_cost,
    df_select_callback_dataframe_row,
    df_select_callback_dataframe_row_ocr_with_words,
    df_select_callback_ocr,
    df_select_callback_textract_api,
    exclude_selected_items_from_redaction,
    get_all_rows_with_same_text,
    get_all_rows_with_same_text_redact,
    get_and_merge_current_page_annotations,
    increase_bottom_page_count_based_on_top,
    increase_page,
    reset_dropdowns,
    undo_last_removal,
    update_all_entity_df_dropdowns,
    update_all_page_annotation_object_based_on_previous_page,
    update_annotator_object_and_filter_df,
    update_annotator_page_from_review_df,
    update_entities_df_page,
    update_entities_df_recogniser_entities,
    update_entities_df_text,
    update_other_annotator_number_from_current,
    update_redact_choice_df_from_page_dropdown,
    update_selected_review_df_row_colour,
)
from tools.textract_batch_call import (
    analyse_document_with_textract_api,
    check_for_provided_job_id,
    check_textract_outputs_exist,
    load_in_textract_job_details,
    poll_whole_document_textract_analysis_progress_and_download,
    replace_existing_pdf_input_for_whole_document_outputs,
)

# Ensure that output folders exist
ensure_folder_exists(CONFIG_FOLDER)
ensure_folder_exists(OUTPUT_FOLDER)
ensure_folder_exists(INPUT_FOLDER)
if GRADIO_TEMP_DIR:
    ensure_folder_exists(GRADIO_TEMP_DIR)
if MPLCONFIGDIR:
    ensure_folder_exists(MPLCONFIGDIR)

ensure_folder_exists(FEEDBACK_LOGS_FOLDER)
ensure_folder_exists(ACCESS_LOGS_FOLDER)
ensure_folder_exists(USAGE_LOGS_FOLDER)

# Add custom spacy recognisers to the Comprehend list, so that local Spacy model can be used to pick up e.g. titles, streetnames, UK postcodes that are sometimes missed by comprehend
CHOSEN_COMPREHEND_ENTITIES.extend(custom_entities)
FULL_COMPREHEND_ENTITY_LIST.extend(custom_entities)
# CHOSEN_LLM_ENTITIES.extend(custom_entities)

###
# Load in FastAPI app
###


# Custom logging filter to remove logs from healthiness/readiness endpoints so they don't fill up application log flow
class EndpointFilter(logging.Filter):
    def __init__(self, path: str, *args, **kwargs):
        self._path = path
        super().__init__(*args, **kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


# 2. Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    # Filter out /health logging to declutter ECS logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(EndpointFilter(path="/health"))

    # Yield control back to the application
    yield

    # --- SHUTDOWN LOGIC ---
    # (Any cleanup code would go here, e.g., closing DB connections)
    pass


def change_tab_to_tabular_or_document_redactions(is_data_file):
    if is_data_file:
        return gr.Tabs(selected=3)
    else:
        return gr.Tabs(selected=1)


def change_tab_to_review_redactions():
    return gr.Tabs(selected=2)


# 3. Initialize the App with the lifespan parameter
# Clean the ROOT_PATH for FastAPI
# Ensure it starts with / and has no trailing /
CLEAN_ROOT = f"/{FASTAPI_ROOT_PATH.strip('/')}" if FASTAPI_ROOT_PATH.strip("/") else ""
app = FastAPI(lifespan=lifespan, root_path=CLEAN_ROOT)

# Added to pass lint check, no effect
spaces.annotations

###
# Load in Gradio app components
###

# Load some components outside of blocks context that are used for examples

# Components for "Redact all PII" option (conditionally visible)
# Set initial visibility based on default redaction method ("Redact all PII")
initial_show_pii_method = SHOW_PII_IDENTIFICATION_OPTIONS  # Default is "Redact all PII"
default_pii_method = DEFAULT_PII_DETECTION_MODEL
initial_show_local_entities = initial_show_pii_method and (
    default_pii_method == LOCAL_PII_OPTION
)
initial_show_comprehend_entities = initial_show_pii_method and (
    default_pii_method == AWS_PII_OPTION
)
initial_is_llm_method = initial_show_pii_method and (
    default_pii_method == LOCAL_TRANSFORMERS_LLM_PII_OPTION
    or default_pii_method == INFERENCE_SERVER_PII_OPTION
    or default_pii_method == AWS_LLM_PII_OPTION
)

## Walkthrough / quickstart components
walkthrough_file_input = gr.File(
    label="Choose a PDF document, image file (PDF, JPG, PNG), tabular data file (Excel, CSV, Parquet), or Word document (DOCX)",
    file_count="multiple",
    file_types=[
        ".pdf",
        ".jpg",
        ".png",
        ".json",
        ".zip",
        ".xlsx",
        ".xls",
        ".csv",
        ".parquet",
        ".docx",
    ],
    height=FILE_INPUT_HEIGHT,
)

walkthrough_in_redact_entities = gr.Dropdown(
    value=CHOSEN_REDACT_ENTITIES,
    choices=FULL_ENTITY_LIST,
    multiselect=True,
    label="Local PII identification model (click empty space in box for full list)",
    visible=initial_show_local_entities,
)

walkthrough_in_redact_comprehend_entities = gr.Dropdown(
    value=CHOSEN_COMPREHEND_ENTITIES,
    choices=FULL_COMPREHEND_ENTITY_LIST,
    multiselect=True,
    label="AWS Comprehend PII identification model (click empty space in box for full list)",
    visible=initial_show_comprehend_entities,
)

# Set initial visibility for local OCR and AWS Textract based on default text extraction method
initial_local_ocr_visible = (
    DEFAULT_TEXT_EXTRACTION_MODEL == TESSERACT_TEXT_EXTRACT_OPTION
)
initial_aws_textract_visible = (
    DEFAULT_TEXT_EXTRACTION_MODEL == TEXTRACT_TEXT_EXTRACT_OPTION
)

walkthrough_text_extract_method_radio = gr.Radio(
    label="""Choose text extraction method. Local options are lower quality but cost nothing - they may be worth a try if you are willing to spend some time reviewing outputs. If shown,AWS Textract has a cost per page - £1.14 ($1.50) without signature detection (default), £2.66 ($3.50) per 1,000 pages with signature detection. Change this in the tab below (AWS Textract signature detection).""",
    value=DEFAULT_TEXT_EXTRACTION_MODEL,
    choices=TEXT_EXTRACTION_MODELS,
    visible=True,
)

walkthrough_local_ocr_method_radio = gr.Radio(
    label=CHOSEN_LOCAL_MODEL_INTRO_TEXT,
    value=CHOSEN_LOCAL_OCR_MODEL,
    choices=LOCAL_OCR_MODEL_OPTIONS,
    interactive=True,
    visible=initial_local_ocr_visible,
)

walkthrough_handwrite_signature_checkbox = gr.CheckboxGroup(
    label="AWS Textract extraction settings",
    choices=HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS,
    value=DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
    visible=initial_aws_textract_visible,
)

walkthrough_pii_identification_method_drop = gr.Radio(
    label="""Choose personal information detection method. The local model is lower quality but costs nothing - it may be worth a try if you are willing to spend some time reviewing outputs, or if you are only interested in searching for custom search terms (see Redaction settings - custom deny list). If shown, AWS Comprehend has a cost of around £0.0075 ($0.01) per 10,000 characters.""",
    value=DEFAULT_PII_DETECTION_MODEL,
    choices=PII_DETECTION_MODELS,
    visible=initial_show_pii_method,
)

walkthrough_deny_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Deny list (always redact these words)",
    interactive=True,
    multiselect=True,
)

walkthrough_allow_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Allow list (always exclude these words from redaction)",
    interactive=True,
    multiselect=True,
)

walkthrough_fully_redacted_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Fully redacted pages (fully redact these page numbers)",
    interactive=True,
    multiselect=True,
)

# Column container for the accordion (conditionally visible based on redaction method)
# Initially hidden since default is "Redact all PII"
initial_show_selected_terms_lists = False
walkthrough_selected_terms_accordion_container = gr.Column(
    visible=initial_show_selected_terms_lists,
)


## Redaction examples
in_doc_files = gr.File(
    label="Choose a PDF document or image file (PDF, JPG, PNG)",
    file_count="multiple",
    file_types=[".pdf", ".jpg", ".png", ".json", ".zip"],
    height=FILE_INPUT_HEIGHT,
)

total_pdf_page_count = gr.Number(
    label="Total page count",
    value=0,
    visible=SHOW_COSTS,
    interactive=False,
)

# Override options if OCR GUI is not shown
if not SHOW_OCR_GUI_OPTIONS:
    SHOW_AWS_TEXT_EXTRACTION_OPTIONS = False
    SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS = False
    SHOW_LOCAL_OCR_MODEL_OPTIONS = False

text_extract_method_radio = gr.Radio(
    label="""Choose text extraction method. Local options are lower quality but cost nothing - they may be worth a try if you are willing to spend some time reviewing outputs. If shown,AWS Textract has a cost per page - £1.14 ($1.50) without signature detection (default), £2.66 ($3.50) per 1,000 pages with signature detection. Change this in the tab below (AWS Textract signature detection).""",
    value=DEFAULT_TEXT_EXTRACTION_MODEL,
    choices=TEXT_EXTRACTION_MODELS,
    visible=SHOW_OCR_GUI_OPTIONS,
)

local_ocr_method_radio = gr.Radio(
    label=CHOSEN_LOCAL_MODEL_INTRO_TEXT,
    value=CHOSEN_LOCAL_OCR_MODEL,
    choices=LOCAL_OCR_MODEL_OPTIONS,
    interactive=True,
    visible=SHOW_LOCAL_OCR_MODEL_OPTIONS,
)

handwrite_signature_checkbox = gr.CheckboxGroup(
    label="AWS Textract extraction settings",
    choices=HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS,
    value=DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
    visible=SHOW_AWS_TEXT_EXTRACTION_OPTIONS,
)

inference_server_vlm_model_textbox = gr.Textbox(
    label="Inference Server VLM Model Name",
    placeholder="e.g., 'qwen2-vl-7b-instruct' or leave empty to use default",
    value=(
        DEFAULT_INFERENCE_SERVER_VLM_MODEL if DEFAULT_INFERENCE_SERVER_VLM_MODEL else ""
    ),
    lines=1,
    visible=SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS,
)

# PII identification components

# Override options if PII identification is not shown
if not SHOW_PII_IDENTIFICATION_OPTIONS:
    SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS = False

pii_identification_method_drop = gr.Radio(
    label="""Choose personal information detection method. The local model is lower quality but costs nothing - it may be worth a try if you are willing to spend some time reviewing outputs, or if you are only interested in searching for custom search terms (see Redaction settings - custom deny list). If shown, AWS Comprehend has a cost of around £0.0075 ($0.01) per 10,000 characters.""",
    value=DEFAULT_PII_DETECTION_MODEL,
    choices=PII_DETECTION_MODELS,
    visible=SHOW_PII_IDENTIFICATION_OPTIONS,
)

in_redact_entities = gr.Dropdown(
    value=CHOSEN_REDACT_ENTITIES,
    choices=FULL_ENTITY_LIST,
    multiselect=True,
    label="Local PII identification model (click empty space in box for full list)",
    visible=initial_show_local_entities,
)
in_redact_comprehend_entities = gr.Dropdown(
    value=CHOSEN_COMPREHEND_ENTITIES,
    choices=FULL_COMPREHEND_ENTITY_LIST,
    multiselect=True,
    label="AWS Comprehend PII identification model (click empty space in box for full list)",
    visible=initial_show_comprehend_entities,
)

in_redact_llm_entities = gr.Dropdown(
    value=CHOSEN_LLM_ENTITIES,
    choices=FULL_LLM_ENTITY_LIST,
    multiselect=True,
    label="LLM PII identification model - subset of entities for LLM detection (click empty space in box for full list)",
    visible=initial_is_llm_method,
)

custom_llm_instructions_textbox = gr.Textbox(
    label="Custom instructions for LLM-based entity detection",
    placeholder="e.g., 'don't redact anything related to Mark Wilson' or 'redact all company names with the label COMPANY_NAME'",
    value="",
    lines=3,
    visible=initial_is_llm_method,
)

# Allow / deny / fully redacted lists

in_deny_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Deny list (always redact these words)",
    interactive=True,
    multiselect=True,
    visible=SHOW_PII_IDENTIFICATION_OPTIONS,
)

in_allow_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Allow list (always exclude these words from redaction)",
    interactive=True,
    multiselect=True,
    visible=SHOW_PII_IDENTIFICATION_OPTIONS,
)

in_fully_redacted_list_state = gr.Dropdown(
    allow_custom_value=True,
    label="Fully redacted pages (fully redact these page numbers)",
    interactive=True,
    multiselect=True,
    visible=SHOW_PII_IDENTIFICATION_OPTIONS,
)

in_deny_list = gr.File(
    label="Import custom deny list - csv table with one column of a different word/phrase on each row (case insensitive). Terms in this file will always be redacted.",
    file_count="multiple",
    height=FILE_INPUT_HEIGHT,
)

in_fully_redacted_list = gr.File(
    label="Import fully redacted pages list - csv table with one column of page numbers on each row. Page numbers in this file will be fully redacted.",
    file_count="multiple",
    height=FILE_INPUT_HEIGHT,
)

## Page options

page_min = gr.Number(
    value=DEFAULT_PAGE_MIN,
    precision=0,
    minimum=0,
    maximum=9999,
    label="Lowest page to redact (set to 0 to redact from the first page)",
)

page_max = gr.Number(
    value=DEFAULT_PAGE_MAX,
    precision=0,
    minimum=0,
    maximum=9999,
    label="Highest page to redact (set to 0 to redact to the last page)",
)

## Deduplication examples
in_duplicate_pages = gr.File(
    label="Upload one or multiple 'ocr_output.csv' files to find duplicate pages and subdocuments",
    file_count="multiple",
    height=FILE_INPUT_HEIGHT,
    file_types=[".csv"],
)

duplicate_threshold_input = gr.Number(
    value=DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
    label="Similarity threshold",
    info="Score (0-1) to consider pages a match.",
)

min_word_count_input = gr.Number(
    value=DEFAULT_MIN_WORD_COUNT,
    label="Minimum word count",
    info="Pages with fewer words than this value are ignored.",
)

combine_page_text_for_duplicates_bool = gr.Checkbox(
    value=True,
    label="Analyse duplicate text by page (off for by line)",
)

## Tabular examples
in_data_files = gr.File(
    label="Choose Excel or csv files",
    file_count="multiple",
    file_types=[".xlsx", ".xls", ".csv", ".parquet", ".docx"],
    height=FILE_INPUT_HEIGHT,
)

in_colnames = gr.Dropdown(
    choices=["Choose columns to anonymise"],
    multiselect=True,
    allow_custom_value=True,
    label="Select columns that you want to anonymise (showing columns present across all files).",
)

in_excel_sheets = gr.Dropdown(
    choices=["Choose Excel sheets to anonymise"],
    multiselect=True,
    label="Select Excel sheets that you want to anonymise (showing sheets present across all Excel files).",
    visible=False,
    allow_custom_value=True,
)

pii_identification_method_drop_tabular = gr.Radio(
    label="Choose PII detection method. AWS Comprehend has a cost of approximately $0.01 per 10,000 characters.",
    value=DEFAULT_PII_DETECTION_MODEL,
    choices=TABULAR_PII_DETECTION_MODELS,
)

anon_strategy = gr.Radio(
    choices=[
        "replace with 'REDACTED'",
        "replace with <ENTITY_NAME>",
        "redact completely",
        "hash",
        "mask",
    ],
    label="Select an anonymisation method.",
    value=DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
)  # , "encrypt", "fake_first_name" are also available, but are not currently included as not that useful in current form

do_initial_clean = gr.Checkbox(
    label="Do initial clean of text (remove URLs, HTML tags, and non-ASCII characters)",
    value=DO_INITIAL_TABULAR_DATA_CLEAN,
)

in_tabular_duplicate_files = gr.File(
    label="Upload CSV, Excel, or Parquet files to find duplicate cells/rows. Note that the app will remove duplicates from later cells/files that are found in earlier cells/files and not vice versa.",
    file_count="multiple",
    file_types=[".csv", ".xlsx", ".xls", ".parquet"],
    height=FILE_INPUT_HEIGHT,
)

tabular_text_columns = gr.Dropdown(
    label="Choose columns to deduplicate",
    multiselect=True,
    allow_custom_value=True,
)

tabular_min_word_count = gr.Number(
    value=DEFAULT_MIN_WORD_COUNT,
    label="Minimum word count",
    info="Cells with fewer words than this are ignored.",
)

### All output file components
all_output_files_btn = gr.Button("Refresh files in output folder", variant="secondary")
all_output_files = gr.FileExplorer(
    root_dir=OUTPUT_FOLDER,
    label="Choose output files for download",
    file_count="multiple",
    visible=SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER,
    interactive=True,
    max_height=400,
)

all_outputs_file_download = gr.File(
    label="Download output files",
    file_count="multiple",
    file_types=[
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".csv",
        ".xlsx",
        ".xls",
        ".txt",
        ".doc",
        ".docx",
        ".json",
    ],
    interactive=False,
    visible=SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER,
    height=200,
)

clean_path = f"/{ROOT_PATH.strip('/')}"
base_href = f"{clean_path}/" if clean_path != "/" else "/"

if ROOT_PATH:
    print(f"✅ Setting HTML base href for Gradio to: '{base_href}'")

head_html = f"""<base href='{base_href}'>

<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js" integrity="sha256-62pj+jS8t+leByFOFwjiY0T92YlWwowYgHnFRklgv0M=" crossorigin="anonymous"></script>"""

css = """
/* Target tab navigation buttons only - not buttons inside tab content */
/* Gradio renders tab buttons with role="tab" in the navigation area */
button[role="tab"] {
    font-size: 1.3em !important;
    padding: 0.75em 1.5em !important;
}

/* Alternative selectors for different Gradio versions */
.tab-nav button,
nav button[role="tab"],
div[class*="tab-nav"] button {
    font-size: 1.2em !important;
    padding: 0.75em 1.5em !important;
}
"""

# Create the gradio interface.
if RUN_FASTAPI:
    blocks = gr.Blocks(
        theme=gr.themes.Default(primary_hue="blue"),
        head=head_html,
        css=css,
        analytics_enabled=False,
        title="Document Redaction App",
        delete_cache=(43200, 43200),  # Temporary file cache deleted every 12 hours
        fill_width=True,
    )
else:
    blocks = gr.Blocks(
        theme=gr.themes.Default(primary_hue="blue"),
        head=head_html,
        css=css,
        analytics_enabled=False,
        title="Document Redaction App",
        delete_cache=(43200, 43200),  # Temporary file cache deleted every 12 hours
        fill_width=True,
    )

with blocks:

    ###
    # STATE VARIABLES
    ###

    # Pymupdf doc needs to be stored as State objects as they do not have a standard Gradio component equivalent
    pdf_doc_state = gr.State(list())
    all_image_annotations_state = gr.Dropdown(
        "",
        label="all_image_annotations_state",
        allow_custom_value=True,
        visible=False,
    )

    all_decision_process_table_state = gr.Dataframe(
        value=pd.DataFrame(),
        headers=None,
        col_count=0,
        row_count=(0, "dynamic"),
        label="all_decision_process_table",
        visible=False,
        type="pandas",
        wrap=True,
    )

    all_page_line_level_ocr_results = gr.Dropdown(
        "",
        label="all_page_line_level_ocr_results",
        allow_custom_value=True,
        visible=False,
    )
    all_page_line_level_ocr_results_with_words = gr.Dropdown(
        "",
        label="all_page_line_level_ocr_results_with_words",
        allow_custom_value=True,
        visible=False,
    )

    session_hash_state = gr.Textbox(label="session_hash_state", value="", visible=False)
    host_name_textbox = gr.Textbox(
        label="host_name_textbox", value=HOST_NAME, visible=False
    )
    s3_output_folder_state = gr.Textbox(
        label="s3_output_folder_state", value=S3_OUTPUTS_FOLDER, visible=False
    )
    session_output_folder_textbox = gr.Textbox(
        value=str(SESSION_OUTPUT_FOLDER),
        label="session_output_folder_textbox",
        visible=False,
    )
    output_folder_textbox = gr.Textbox(
        value=OUTPUT_FOLDER, label="output_folder_textbox", visible=False
    )
    input_folder_textbox = gr.Textbox(
        value=INPUT_FOLDER, label="input_folder_textbox", visible=False
    )

    first_loop_state = gr.Checkbox(label="first_loop_state", value=True, visible=False)
    second_loop_state = gr.Checkbox(
        label="second_loop_state", value=False, visible=False
    )
    do_not_save_pdf_state = gr.Checkbox(
        label="do_not_save_pdf_state", value=False, visible=False
    )
    save_pdf_state = gr.Checkbox(label="save_pdf_state", value=True, visible=False)

    prepared_pdf_state = gr.Dropdown(
        label="prepared_pdf_list", value="", allow_custom_value=True, visible=False
    )
    document_cropboxes = gr.Dropdown(
        label="document_cropboxes", value="", allow_custom_value=True, visible=False
    )
    page_sizes = gr.Dropdown(
        label="page_sizes", value="", allow_custom_value=True, visible=False
    )
    images_pdf_state = gr.Dropdown(
        label="images_pdf_list", value="", allow_custom_value=True, visible=False
    )
    all_img_details_state = gr.Dropdown(
        label="all_img_details_state",
        value="",
        allow_custom_value=True,
        visible=False,
    )

    output_image_files_state = gr.Dropdown(
        label="output_image_files_list",
        value="",
        allow_custom_value=True,
        visible=False,
    )
    output_file_list_state = gr.Dropdown(
        label="output_file_list", value="", allow_custom_value=True, visible=False
    )
    text_output_file_list_state = gr.Dropdown(
        label="text_output_file_list",
        value="",
        allow_custom_value=True,
        visible=False,
    )
    log_files_output_list_state = gr.Dropdown(
        label="log_files_output_list",
        value="",
        allow_custom_value=True,
        visible=False,
    )
    duplication_file_path_outputs_list_state = gr.Dropdown(
        label="duplication_file_path_outputs_list",
        value=list(),
        multiselect=True,
        allow_custom_value=True,
        visible=False,
    )

    # Backup versions of these objects in case you make a mistake
    backup_review_state = gr.State(pd.DataFrame())
    backup_image_annotations_state = gr.State(list())
    backup_recogniser_entity_dataframe_base = gr.State(pd.DataFrame())
    backup_all_page_line_level_ocr_results_with_words_df_base = gr.State(pd.DataFrame())

    # Logging variables
    access_logs_state = gr.Textbox(
        label="access_logs_state",
        value=ACCESS_LOGS_FOLDER + LOG_FILE_NAME,
        visible=False,
    )
    access_s3_logs_loc_state = gr.Textbox(
        label="access_s3_logs_loc_state", value=S3_ACCESS_LOGS_FOLDER, visible=False
    )
    feedback_logs_state = gr.Textbox(
        label="feedback_logs_state",
        value=FEEDBACK_LOGS_FOLDER + FEEDBACK_LOG_FILE_NAME,
        visible=False,
    )
    feedback_s3_logs_loc_state = gr.Textbox(
        label="feedback_s3_logs_loc_state",
        value=S3_FEEDBACK_LOGS_FOLDER,
        visible=False,
    )
    usage_logs_state = gr.Textbox(
        label="usage_logs_state",
        value=USAGE_LOGS_FOLDER + USAGE_LOG_FILE_NAME,
        visible=False,
    )
    usage_s3_logs_loc_state = gr.Textbox(
        label="usage_s3_logs_loc_state", value=S3_USAGE_LOGS_FOLDER, visible=False
    )

    session_hash_textbox = gr.Textbox(
        label="session_hash_textbox", value="", visible=False
    )
    textract_metadata_textbox = gr.Textbox(
        label="textract_metadata_textbox", value="", visible=False
    )
    comprehend_query_number = gr.Number(
        label="comprehend_query_number", value=0, visible=False
    )
    textract_query_number = gr.Number(
        label="textract_query_number", value=0, visible=False
    )

    # VLM and LLM tracking components for usage logs
    vlm_model_name_textbox = gr.Textbox(label="vlm_model_name", value="", visible=False)
    vlm_total_input_tokens_number = gr.Number(
        label="vlm_total_input_tokens", value=0, visible=False
    )
    vlm_total_output_tokens_number = gr.Number(
        label="vlm_total_output_tokens", value=0, visible=False
    )
    llm_model_name_textbox = gr.Textbox(label="llm_model_name", value="", visible=False)
    llm_total_input_tokens_number = gr.Number(
        label="llm_total_input_tokens", value=0, visible=False
    )
    llm_total_output_tokens_number = gr.Number(
        label="llm_total_output_tokens", value=0, visible=False
    )

    doc_full_file_name_textbox = gr.Textbox(
        label="doc_full_file_name_textbox", value="", visible=False
    )
    doc_file_name_no_extension_textbox = gr.Textbox(
        label="doc_full_file_name_textbox", value="", visible=False
    )
    blank_doc_file_name_no_extension_textbox_for_logs = gr.Textbox(
        label="doc_full_file_name_textbox", value="", visible=False
    )
    blank_data_file_name_no_extension_textbox_for_logs = gr.Textbox(
        label="data_full_file_name_textbox", value="", visible=False
    )
    placeholder_doc_file_name_no_extension_textbox_for_logs = gr.Textbox(
        label="doc_full_file_name_textbox", value="document", visible=False
    )
    placeholder_data_file_name_no_extension_textbox_for_logs = gr.Textbox(
        label="data_full_file_name_textbox", value="data_file", visible=False
    )

    # Left blank for when user does not want to report file names
    doc_file_name_with_extension_textbox = gr.Textbox(
        label="doc_file_name_with_extension_textbox", value="", visible=False
    )
    doc_file_name_textbox_list = gr.Dropdown(
        label="doc_file_name_textbox_list",
        value="",
        allow_custom_value=True,
        visible=False,
    )
    latest_review_file_path = gr.Textbox(
        label="latest_review_file_path", value="", visible=False
    )  # Latest review file path output from redaction
    latest_ocr_file_path = gr.Textbox(
        label="latest_ocr_file_path", value="", visible=False
    )  # Latest ocr file path output from text extraction

    data_full_file_name_textbox = gr.Textbox(
        label="data_full_file_name_textbox", value="", visible=False
    )
    data_file_name_no_extension_textbox = gr.Textbox(
        label="data_full_file_name_textbox", value="", visible=False
    )
    data_file_name_with_extension_textbox = gr.Textbox(
        label="data_file_name_with_extension_textbox", value="", visible=False
    )
    data_file_name_textbox_list = gr.Dropdown(
        label="data_file_name_textbox_list",
        value="",
        allow_custom_value=True,
        visible=False,
    )

    # Constants just to use with the review dropdowns for filtering by various columns
    label_name_const = gr.Textbox(
        label="label_name_const", value="label", visible=False
    )
    text_name_const = gr.Textbox(label="text_name_const", value="text", visible=False)
    page_name_const = gr.Textbox(label="page_name_const", value="page", visible=False)

    actual_time_taken_number = gr.Number(
        label="actual_time_taken_number", value=0.0, precision=1, visible=False
    )  # This keeps track of the time taken to redact files for logging purposes.
    annotate_previous_page = gr.Number(
        value=0, label="Previous page", precision=0, visible=False
    )  # Keeps track of the last page that the annotator was on
    s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)

    ## Annotator zoom value
    annotator_zoom_number = gr.Number(
        label="Current annotator zoom level", value=100, precision=0, visible=False
    )
    zoom_true_bool = gr.Checkbox(label="zoom_true_bool", value=True, visible=False)
    zoom_false_bool = gr.Checkbox(label="zoom_false_bool", value=False, visible=False)

    clear_all_page_redactions = gr.Checkbox(
        label="clear_all_page_redactions", value=True, visible=False
    )
    prepare_for_review_bool = gr.Checkbox(
        label="prepare_for_review_bool", value=True, visible=False
    )
    prepare_for_review_bool_false = gr.Checkbox(
        label="prepare_for_review_bool_false", value=False, visible=False
    )
    prepare_images_bool_false = gr.Checkbox(
        label="prepare_images_bool_false", value=False, visible=False
    )

    ## Settings page variables
    default_deny_list_file_name = "default_deny_list.csv"
    default_deny_list_loc = OUTPUT_FOLDER + "/" + default_deny_list_file_name
    in_deny_list_text_in = gr.Textbox(value="deny_list", visible=False)

    fully_redacted_list_file_name = "default_fully_redacted_list.csv"
    fully_redacted_list_loc = OUTPUT_FOLDER + "/" + fully_redacted_list_file_name
    in_fully_redacted_text_in = gr.Textbox(
        value="fully_redacted_pages_list", visible=False
    )

    # S3 settings for default allow list load
    s3_default_bucket = gr.Textbox(
        label="Default S3 bucket", value=DOCUMENT_REDACTION_BUCKET, visible=False
    )
    s3_default_allow_list_file = gr.Textbox(
        label="Default allow list file", value=S3_ALLOW_LIST_PATH, visible=False
    )
    default_allow_list_output_folder_location = gr.Textbox(
        label="Output default allow list location",
        value=ALLOW_LIST_PATH,
        visible=False,
    )

    s3_whole_document_textract_default_bucket = gr.Textbox(
        label="Default Textract whole_document S3 bucket",
        value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
        visible=False,
    )
    s3_whole_document_textract_input_subfolder = gr.Textbox(
        label="Default Textract whole_document S3 input folder",
        value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
        visible=False,
    )
    s3_whole_document_textract_output_subfolder = gr.Textbox(
        label="Default Textract whole_document S3 output folder",
        value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
        visible=False,
    )
    successful_textract_api_call_number = gr.Number(precision=0, value=0, visible=False)
    no_redaction_method_drop = gr.Radio(
        label="""Placeholder for no redaction method after downloading Textract outputs""",
        value=NO_REDACTION_PII_OPTION,
        choices=[NO_REDACTION_PII_OPTION],
        visible=False,
    )
    textract_only_method_drop = gr.Radio(
        label="""Placeholder for Textract method after downloading Textract outputs""",
        value=TEXTRACT_TEXT_EXTRACT_OPTION,
        choices=[TEXTRACT_TEXT_EXTRACT_OPTION],
        visible=False,
    )

    load_s3_whole_document_textract_logs_bool = gr.Textbox(
        label="Load Textract logs or not",
        value=LOAD_PREVIOUS_TEXTRACT_JOBS_S3,
        visible=False,
    )
    s3_whole_document_textract_logs_subfolder = gr.Textbox(
        label="Default Textract whole_document S3 input folder",
        value=TEXTRACT_JOBS_S3_LOC,
        visible=False,
    )
    local_whole_document_textract_logs_subfolder = gr.Textbox(
        label="Default Textract whole_document S3 output folder",
        value=TEXTRACT_JOBS_LOCAL_LOC,
        visible=False,
    )

    s3_default_cost_codes_file = gr.Textbox(
        label="Default cost centre file", value=S3_COST_CODES_PATH, visible=False
    )
    default_cost_codes_output_folder_location = gr.Textbox(
        label="Output default cost centre location",
        value=OUTPUT_COST_CODES_PATH,
        visible=False,
    )
    enforce_cost_code_textbox = gr.Textbox(
        label="Enforce cost code textbox", value=ENFORCE_COST_CODES, visible=False
    )
    default_cost_code_textbox = gr.Textbox(
        label="Default cost code textbox", value=DEFAULT_COST_CODE, visible=False
    )

    # Base tables that are not modified subsequent to load
    recogniser_entity_dataframe_base = gr.State(
        pd.DataFrame(columns=["page", "label", "text", "id"])
    )
    all_page_line_level_ocr_results_df_base = gr.State(
        pd.DataFrame(
            columns=[
                "page",
                "text",
                "left",
                "top",
                "width",
                "height",
                "line",
                "conf",
            ]
        )
    )
    all_line_level_ocr_results_df_placeholder = gr.State(
        pd.DataFrame(
            columns=[
                "page",
                "text",
                "left",
                "top",
                "width",
                "height",
                "line",
                "conf",
            ]
        )
    )

    # Placeholder for selected entity dataframe row
    selected_entity_id = gr.Textbox(value="", label="selected_entity_id", visible=False)
    selected_entity_colour = gr.Textbox(
        value="", label="selected_entity_colour", visible=False
    )
    selected_entity_dataframe_row_text = gr.Textbox(
        value="", label="selected_entity_dataframe_row_text", visible=False
    )
    selected_entity_dataframe_row_text_redact = gr.Textbox(
        value="", label="selected_entity_dataframe_row_text_redact", visible=False
    )

    # This is an invisible dataframe that holds all items from the redaction outputs that have the same text as the selected row
    recogniser_entity_dataframe_same_text = gr.Dataframe(
        pd.DataFrame(
            data={"page": list(), "label": list(), "text": list(), "id": list()}
        ),
        col_count=(4, "fixed"),
        type="pandas",
        label="Table rows with same text",
        headers=["page", "label", "text", "id"],
        wrap=True,
        max_height=400,
        static_columns=[0, 1, 2, 3],
        visible=False,
    )

    to_redact_dataframe_same_text = gr.Dataframe(
        pd.DataFrame(
            data={
                "page": list(),
                "line": list(),
                "word_text": list(),
                "word_x0": list(),
                "word_y0": list(),
                "word_x1": list(),
                "word_y1": list(),
                "index": list(),
            }
        ),
        type="pandas",
        headers=[
            "page",
            "line",
            "word_text",
            "word_x0",
            "word_y0",
            "word_x1",
            "word_y1",
            "index",
        ],
        wrap=False,
        visible=False,
    )

    # Duplicate page detection
    in_duplicate_pages_text = gr.Textbox(label="in_duplicate_pages_text", visible=False)
    duplicate_pages_df = gr.Dataframe(
        value=pd.DataFrame(),
        headers=None,
        col_count=0,
        row_count=(0, "dynamic"),
        label="duplicate_pages_df",
        visible=False,
        type="pandas",
        wrap=True,
    )
    full_duplicated_data_df = gr.Dataframe(
        value=pd.DataFrame(),
        headers=None,
        col_count=0,
        row_count=(0, "dynamic"),
        label="full_duplicated_data_df",
        visible=False,
        type="pandas",
        wrap=True,
    )
    selected_duplicate_data_row_index = gr.Number(
        value=None, label="selected_duplicate_data_row_index", visible=False
    )
    full_duplicate_data_by_file = (
        gr.State()
    )  # A dictionary of the full duplicate data indexed by file

    # Tracking variables for current page (not visible)
    current_loop_page_number = gr.Number(
        value=0,
        precision=0,
        interactive=False,
        label="Last redacted page in document",
        visible=False,
    )
    page_break_return = gr.Checkbox(
        value=False, label="Page break reached", visible=False
    )

    latest_file_completed_num = gr.Number(
        value=0,
        label="Number of documents redacted",
        interactive=False,
        visible=False,
    )

    # Placeholders for elements that may be made visible later below depending on environment variables
    cost_code_dataframe_base = gr.Dataframe(
        value=pd.DataFrame(),
        row_count=(0, "dynamic"),
        label="Cost codes",
        type="pandas",
        interactive=True,
        show_search="filter",
        wrap=True,
        max_height=200,
        visible=False,
    )
    cost_code_dataframe = gr.Dataframe(
        value=pd.DataFrame(), type="pandas", visible=False, wrap=True
    )
    cost_code_choice_drop = gr.Dropdown(
        value=DEFAULT_COST_CODE,
        label="Choose cost code for analysis. Please contact Finance if you can't find your cost code in the given list.",
        choices=[DEFAULT_COST_CODE],
        allow_custom_value=False,
        visible=False,
    )

    textract_output_found_checkbox = gr.Checkbox(
        value=False,
        label="Existing Textract output file found",
        interactive=False,
        visible=False,
    )
    relevant_ocr_output_with_words_found_checkbox = gr.Checkbox(
        value=False,
        label="Existing local OCR output file found",
        interactive=False,
        visible=False,
    )

    estimated_aws_costs_number = gr.Number(
        label="Approximate AWS Textract and/or Comprehend cost ($)",
        value=0,
        visible=False,
        precision=2,
    )
    estimated_time_taken_number = gr.Number(
        label="Approximate time taken to extract text/redact (minutes)",
        value=0,
        visible=False,
        precision=2,
    )

    only_extract_text_radio = gr.Checkbox(
        value=False, label="Only extract text (no redaction)", visible=False
    )

    # Textract API call placeholders in case option not selected in config

    job_name_textbox = gr.Textbox(
        value="", label="whole_document Textract call", visible=False
    )
    send_document_to_textract_api_btn = gr.Button(
        "Analyse document with AWS Textract", variant="primary", visible=False
    )

    job_id_textbox = gr.Textbox(
        label="Latest job ID for whole_document document analysis",
        value="",
        visible=False,
    )
    check_state_of_textract_api_call_btn = gr.Button(
        "Check state of Textract document job and download",
        variant="secondary",
        visible=False,
    )
    job_current_status = gr.Textbox(
        value="", label="Analysis job current status", visible=False
    )
    job_type_dropdown = gr.Dropdown(
        value="document_text_detection",
        choices=["document_text_detection", "document_analysis"],
        label="Job type of Textract analysis job",
        allow_custom_value=False,
        visible=False,
    )
    textract_job_detail_df = gr.Dataframe(
        pd.DataFrame(
            columns=[
                "job_id",
                "file_name",
                "job_type",
                "signature_extraction",
                "job_date_time",
            ]
        ),
        label="Previous job details",
        visible=False,
        type="pandas",
        wrap=True,
    )
    selected_job_id_row = gr.Dataframe(
        pd.DataFrame(
            columns=[
                "job_id",
                "file_name",
                "job_type",
                "signature_extraction",
                "job_date_time",
            ]
        ),
        label="Selected job id row",
        visible=False,
        type="pandas",
        wrap=True,
    )
    is_a_textract_api_call = gr.Checkbox(
        value=False, label="is_this_a_textract_api_call", visible=False
    )
    task_textbox = gr.Textbox(
        value="redact", label="task", visible=False
    )  # Track the task being performed
    job_output_textbox = gr.Textbox(
        value="", label="Textract call outputs", visible=False
    )
    job_input_textbox = gr.Textbox(
        value=TEXTRACT_JOBS_S3_INPUT_LOC,
        label="Textract call outputs",
        visible=False,
    )

    textract_job_output_file = gr.File(
        label="Textract job output files", height=FILE_INPUT_HEIGHT, visible=False
    )
    convert_textract_outputs_to_ocr_results = gr.Button(
        "Placeholder - Convert Textract job outputs to OCR results (needs relevant document file uploaded above)",
        variant="secondary",
        visible=False,
    )

    ## Duplicate search object
    new_duplicate_search_annotation_object = gr.Dropdown(
        value=None,
        label="new_duplicate_search_annotation_object",
        allow_custom_value=True,
        visible=False,
    )

    # Spacy analyser state
    updated_nlp_analyser_state = gr.State(list())
    tesseract_lang_data_file_path = gr.Textbox("", visible=False)

    flag_value_placeholder = gr.Textbox(
        value="", visible=False
    )  # Placeholder for flag value

    ###
    # UI DESIGN
    ###

    gr.Markdown(INTRO_TEXT)

    # Examples for PDF/image redaction
    if SHOW_EXAMPLES:
        gr.Markdown(
            "### Try out general redaction tasks - click on an example below and then the 'Extract text and redact document' button:"
        )

        # Check which example files exist and create examples only for available files
        example_files = [
            "example_data/example_of_emails_sent_to_a_professor_before_applying.pdf",
            "example_data/example_complaint_letter.jpg",
            "example_data/graduate-job-example-cover-letter.pdf",
            "example_data/Partnership-Agreement-Toolkit_0_0.pdf",
            "example_data/partnership_toolkit_redact_custom_deny_list.csv",
            "example_data/partnership_toolkit_redact_some_pages.csv",
        ]

        available_examples = list()
        example_labels = list()

        # Check each example file and add to examples if it exists
        if os.path.exists(example_files[0]):
            available_examples.append(
                [
                    [example_files[0]],
                    "Local model - selectable text",
                    "Local",
                    [],
                    CHOSEN_REDACT_ENTITIES,
                    CHOSEN_COMPREHEND_ENTITIES,
                    [example_files[0]],
                    example_files[0],
                    [],
                    [],
                    [],
                    [],
                    2,
                ]
            )
            example_labels.append("PDF with selectable text redaction")

        if os.path.exists(example_files[1]):
            available_examples.append(
                [
                    [example_files[1]],
                    "Local OCR model - PDFs without selectable text",
                    "Local",
                    [],
                    CHOSEN_REDACT_ENTITIES,
                    CHOSEN_COMPREHEND_ENTITIES,
                    [example_files[1]],
                    example_files[1],
                    [],
                    [],
                    [],
                    [],
                    1,
                ]
            )
            example_labels.append("Image redaction with local OCR")

        if os.path.exists(example_files[2]):
            available_examples.append(
                [
                    [example_files[2]],
                    "Local OCR model - PDFs without selectable text",
                    "Local",
                    [],
                    ["TITLES", "PERSON", "DATE_TIME"],
                    CHOSEN_COMPREHEND_ENTITIES,
                    [example_files[2]],
                    example_files[2],
                    [],
                    [],
                    [],
                    [],
                    1,
                ]
            )
            example_labels.append(
                "PDF redaction with custom entities (Titles, Person, Dates)"
            )

        if os.path.exists(example_files[3]):
            if SHOW_AWS_EXAMPLES:
                available_examples.append(
                    [
                        [example_files[3]],
                        "AWS Textract service - all PDF types",
                        "AWS Comprehend",
                        ["Extract handwriting", "Extract signatures"],
                        CHOSEN_REDACT_ENTITIES,
                        CHOSEN_COMPREHEND_ENTITIES,
                        [example_files[3]],
                        example_files[3],
                        [],
                        [],
                        [],
                        [],
                        7,
                    ]
                )
                example_labels.append(
                    "PDF redaction with AWS services and signature detection"
                )

        # Add new example for custom deny list and whole page redaction
        if (
            os.path.exists(example_files[3])
            and os.path.exists(example_files[4])
            and os.path.exists(example_files[5])
        ):
            available_examples.append(
                [
                    [example_files[3]],
                    "Local OCR model - PDFs without selectable text",
                    "Local",
                    [],
                    ["CUSTOM"],  # Use CUSTOM entity to enable deny list functionality
                    CHOSEN_COMPREHEND_ENTITIES,
                    [example_files[3]],
                    example_files[3],
                    [example_files[4]],
                    [
                        "Sister",
                        "Sister City",
                        "Sister Cities",
                        "Friendship City",
                    ],
                    [example_files[5]],
                    [
                        2,
                        5,
                    ],  # pd.DataFrame(data={"fully_redacted_pages_list": [2, 5]}),
                    7,
                ],
            )
            example_labels.append(
                "PDF redaction with custom deny list and whole page redaction"
            )

        # Only create examples if we have available files
        if available_examples:

            def show_info_box_on_click(
                in_doc_files,
                text_extract_method_radio,
                pii_identification_method_drop,
                handwrite_signature_checkbox,
                in_redact_entities,
                in_redact_comprehend_entities,
                prepared_pdf_state,
                doc_full_file_name_textbox,
                in_deny_list,
                in_deny_list_state,
                in_fully_redacted_list,
                in_fully_redacted_list_state,
                total_pdf_page_count,
            ):
                gr.Info(
                    "Example data loaded. Now click on 'Extract text and redact document' below to run the example redaction."
                )

                # Convert deny_list_state, allow_list_state, and fully_redacted_list_state to lists if they are DataFrames
                # Handle deny_list_state
                deny_list_walkthrough = []
                if isinstance(in_deny_list_state, pd.DataFrame):
                    # Explicitly convert empty DataFrame to empty list
                    if in_deny_list_state.empty:
                        deny_list_walkthrough = []
                    else:
                        deny_list_walkthrough = (
                            in_deny_list_state.iloc[:, 0].dropna().astype(str).tolist()
                        )
                elif isinstance(in_deny_list_state, list):
                    deny_list_walkthrough = (
                        [str(item) for item in in_deny_list_state if item]
                        if in_deny_list_state
                        else []
                    )
                else:
                    # Default to empty list for any other type
                    deny_list_walkthrough = []

                # Handle fully_redacted_list_state
                fully_redacted_list_walkthrough = []
                if isinstance(in_fully_redacted_list_state, pd.DataFrame):
                    # Explicitly convert empty DataFrame to empty list
                    if in_fully_redacted_list_state.empty:
                        fully_redacted_list_walkthrough = []
                    else:
                        fully_redacted_list_walkthrough = (
                            in_fully_redacted_list_state.iloc[:, 0]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                elif isinstance(in_fully_redacted_list_state, list):
                    fully_redacted_list_walkthrough = (
                        [str(item) for item in in_fully_redacted_list_state if item]
                        if in_fully_redacted_list_state
                        else []
                    )
                else:
                    # Default to empty list for any other type
                    fully_redacted_list_walkthrough = []

                # Allow list is not in examples, so always set to empty list
                allow_list_walkthrough = []

                # Use default local OCR method - examples don't set this directly
                local_ocr_method = CHOSEN_LOCAL_OCR_MODEL

                return (
                    gr.File(value=in_doc_files),  # walkthrough_file_input
                    gr.Dropdown(
                        value=in_redact_entities
                    ),  # walkthrough_in_redact_entities
                    gr.Dropdown(
                        value=in_redact_comprehend_entities
                    ),  # walkthrough_in_redact_comprehend_entities
                    gr.Radio(
                        value=text_extract_method_radio
                    ),  # walkthrough_text_extract_method_radio
                    gr.Radio(
                        value=local_ocr_method
                    ),  # walkthrough_local_ocr_method_radio
                    gr.CheckboxGroup(
                        value=handwrite_signature_checkbox
                    ),  # walkthrough_handwrite_signature_checkbox
                    gr.Radio(
                        value=pii_identification_method_drop
                    ),  # walkthrough_pii_identification_method_drop
                    gr.Dropdown(
                        value=allow_list_walkthrough
                    ),  # walkthrough_allow_list_state
                    gr.Dropdown(
                        value=deny_list_walkthrough
                    ),  # walkthrough_deny_list_state
                    gr.Dropdown(
                        value=fully_redacted_list_walkthrough
                    ),  # walkthrough_fully_redacted_list_state
                )

            redaction_examples = gr.Examples(
                examples=available_examples,
                inputs=[
                    in_doc_files,
                    text_extract_method_radio,
                    pii_identification_method_drop,
                    handwrite_signature_checkbox,
                    in_redact_entities,
                    in_redact_comprehend_entities,
                    prepared_pdf_state,
                    doc_full_file_name_textbox,
                    in_deny_list,
                    in_deny_list_state,
                    in_fully_redacted_list,
                    in_fully_redacted_list_state,
                    total_pdf_page_count,
                ],
                outputs=[
                    walkthrough_file_input,
                    walkthrough_in_redact_entities,
                    walkthrough_in_redact_comprehend_entities,
                    walkthrough_text_extract_method_radio,
                    walkthrough_local_ocr_method_radio,
                    walkthrough_handwrite_signature_checkbox,
                    walkthrough_pii_identification_method_drop,
                    walkthrough_allow_list_state,
                    walkthrough_deny_list_state,
                    walkthrough_fully_redacted_list_state,
                ],
                example_labels=example_labels,
                fn=show_info_box_on_click,
                run_on_click=True,
            )
    if SHOW_DIFFICULT_OCR_EXAMPLES:
        gr.Markdown(
            "### Test out the different OCR methods available. Click on an example below and then the 'Extract text and redact document' button:"
        )
        ocr_example_files = [
            "example_data/Partnership-Agreement-Toolkit_0_0.pdf",
            "example_data/Difficult handwritten note.jpg",
            "example_data/Example-cv-university-graduaty-hr-role-with-photo-2.pdf",
        ]
        available_ocr_examples = list()
        ocr_example_labels = list()
        if os.path.exists(ocr_example_files[0]):
            available_ocr_examples.append(
                [
                    [ocr_example_files[0]],
                    "Local OCR model - PDFs without selectable text",
                    "Only extract text (no redaction)",
                    ["Extract handwriting", "Extract signatures"],
                    [ocr_example_files[0]],
                    ocr_example_files[0],
                    7,
                    1,
                    1,
                    "paddle",
                    CHOSEN_REDACT_ENTITIES,
                ],
            )
            ocr_example_labels.append("Baseline 'easy' document page")

            available_ocr_examples.append(
                [
                    [ocr_example_files[0]],
                    "Local OCR model - PDFs without selectable text",
                    "Local",
                    ["Extract handwriting", "Extract signatures"],
                    [ocr_example_files[0]],
                    ocr_example_files[0],
                    7,
                    6,
                    6,
                    "hybrid-paddle-vlm",
                    CHOSEN_REDACT_ENTITIES + ["CUSTOM_VLM_SIGNATURE"],
                ],
            )
            ocr_example_labels.append("Scanned document page with signatures")

        if os.path.exists(ocr_example_files[1]):
            available_ocr_examples.append(
                [
                    [ocr_example_files[1]],
                    "Local OCR model - PDFs without selectable text",
                    "Only extract text (no redaction)",
                    ["Extract handwriting", "Extract signatures"],
                    [ocr_example_files[1]],
                    ocr_example_files[1],
                    1,
                    0,
                    0,
                    "vlm",
                    CHOSEN_REDACT_ENTITIES,
                ],
            )
            ocr_example_labels.append("Unclear text on handwritten note")

        if os.path.exists(ocr_example_files[2]):
            available_ocr_examples.append(
                [
                    [ocr_example_files[2]],
                    "Local OCR model - PDFs without selectable text",
                    "Local",
                    ["Extract handwriting", "Extract signatures"],
                    [ocr_example_files[2]],
                    ocr_example_files[2],
                    1,
                    0,
                    0,
                    "hybrid-paddle-vlm",
                    CHOSEN_REDACT_ENTITIES + ["CUSTOM_VLM_PERSON"],
                ],
            )
            ocr_example_labels.append("CV with photo")

        # Only create examples if we have available files
        if available_ocr_examples:

            def show_info_box_on_click(
                in_doc_files,
                text_extract_method_radio,
                pii_identification_method_drop,
                handwrite_signature_checkbox,
                prepared_pdf_state,
                doc_full_file_name_textbox,
                total_pdf_page_count,
                page_min,
                page_max,
                local_ocr_method_radio,
                in_redact_entities,
            ):
                gr.Info(
                    "Example OCR data loaded. Now click on 'Extract text and redact document' below to run the OCR analysis."
                )

                return (
                    gr.File(value=in_doc_files),  # walkthrough_file_input
                    gr.Dropdown(
                        value=in_redact_entities
                    ),  # walkthrough_in_redact_entities
                    gr.Radio(
                        value=text_extract_method_radio
                    ),  # walkthrough_text_extract_method_radio
                    gr.Radio(
                        value=local_ocr_method_radio
                    ),  # walkthrough_local_ocr_method_radio
                    gr.CheckboxGroup(
                        value=handwrite_signature_checkbox
                    ),  # walkthrough_handwrite_signature_checkbox
                    gr.Radio(
                        value=pii_identification_method_drop
                    ),  # walkthrough_pii_identification_method_drop
                )

            ocr_examples = gr.Examples(
                examples=available_ocr_examples,
                inputs=[
                    in_doc_files,
                    text_extract_method_radio,
                    pii_identification_method_drop,
                    handwrite_signature_checkbox,
                    prepared_pdf_state,
                    doc_full_file_name_textbox,
                    total_pdf_page_count,
                    page_min,
                    page_max,
                    local_ocr_method_radio,
                    in_redact_entities,
                ],
                outputs=[
                    walkthrough_file_input,
                    walkthrough_in_redact_entities,
                    walkthrough_text_extract_method_radio,
                    walkthrough_local_ocr_method_radio,
                    walkthrough_handwrite_signature_checkbox,
                    walkthrough_pii_identification_method_drop,
                ],
                example_labels=ocr_example_labels,
                fn=show_info_box_on_click,
                run_on_click=True,
            )

    with gr.Tabs() as tabs:
        ###
        # QUICKSTART TAB
        ###
        if SHOW_QUICKSTART:
            with gr.Tab("Quickstart", id=0):
                # State to track if we're dealing with data files
                walkthrough_is_data_file = gr.State(value=False)

                with gr.Walkthrough(selected=1) as walkthrough:
                    with gr.Step("Load document/data", id=1):

                        walkthrough_file_input.render()
                        with gr.Row():
                            step_1_back_btn = gr.Button("Back", variant="secondary")
                            step_1_back_btn.click(
                                lambda: gr.Walkthrough(selected=0), outputs=walkthrough
                            )
                            step_1_next_btn = gr.Button("Next", variant="primary")
                    with gr.Step("Choose text extraction (OCR) method", id=2):
                        # Components for data files (conditionally visible)
                        walkthrough_excel_sheets = gr.Dropdown(
                            choices=["Choose Excel sheets to anonymise"],
                            multiselect=True,
                            label="Select Excel sheets that you want to anonymise (showing sheets present across all Excel files).",
                            visible=False,
                            allow_custom_value=True,
                        )
                        walkthrough_colnames = gr.Dropdown(
                            choices=["Choose columns to anonymise"],
                            multiselect=True,
                            allow_custom_value=True,
                            label="Select columns that you want to anonymise (showing columns present across all files).",
                            visible=False,
                        )
                        # Text extraction method radio (conditionally visible)
                        walkthrough_text_extract_method_radio.render()
                        # Local OCR method radio (shown only if Local OCR model is selected)
                        walkthrough_local_ocr_method_radio.render()
                        # AWS Textract extraction settings (shown only if AWS Textract is selected)
                        walkthrough_handwrite_signature_checkbox.render()
                        with gr.Row():
                            step_2_back_btn = gr.Button("Back", variant="secondary")
                            step_2_back_btn.click(
                                lambda: gr.Walkthrough(selected=1), outputs=walkthrough
                            )
                            step_2_next_btn = gr.Button("Next", variant="primary")
                    with gr.Step("Choose PII detection method", id=3):
                        # Redaction method selection (at the top of Step 3)
                        walkthrough_redaction_method_dropdown = gr.Radio(
                            label="Choose redaction method",
                            choices=[
                                "Extract text only",
                                "Redact all PII",
                                "Redact selected terms",
                            ],
                            value="Redact all PII",
                            interactive=True,
                        )

                        walkthrough_pii_identification_method_drop.render()

                        walkthrough_in_redact_entities.render()

                        walkthrough_in_redact_comprehend_entities.render()
                        walkthrough_in_redact_llm_entities = gr.Dropdown(
                            value=CHOSEN_LLM_ENTITIES,
                            choices=FULL_LLM_ENTITY_LIST,
                            multiselect=True,
                            label="LLM PII identification model - subset of entities for LLM detection (click empty space in box for full list)",
                            visible=initial_is_llm_method,
                        )
                        walkthrough_custom_llm_instructions_textbox = gr.Textbox(
                            label="Custom instructions for LLM-based entity detection",
                            placeholder="e.g., 'don't redact anything related to Mark Wilson' or 'redact all company names with the label COMPANY_NAME'",
                            value="",
                            lines=3,
                            visible=initial_is_llm_method,
                        )

                        with walkthrough_selected_terms_accordion_container:
                            with gr.Accordion(
                                "Terms to always include or exclude in redactions, and whole page redaction. To add many terms at once, you can load in a file on the Redaction Settings tab.",
                                open=True,
                            ):
                                # Components for "Redact selected terms" option (conditionally visible)
                                with gr.Row():
                                    walkthrough_deny_list_state.render()
                                    walkthrough_allow_list_state.render()
                                    walkthrough_fully_redacted_list_state.render()

                        # Tabular data redaction options (conditionally visible for data files)
                        walkthrough_pii_identification_method_drop_tabular = gr.Radio(
                            label="Choose PII detection method. AWS Comprehend has a cost of approximately $0.01 per 10,000 characters.",
                            value=DEFAULT_PII_DETECTION_MODEL,
                            choices=TABULAR_PII_DETECTION_MODELS,
                            visible=False,
                        )
                        walkthrough_anon_strategy = gr.Radio(
                            choices=[
                                "replace with 'REDACTED'",
                                "replace with <ENTITY_NAME>",
                                "redact completely",
                                "hash",
                                "mask",
                            ],
                            label="Select an anonymisation method.",
                            value=DEFAULT_TABULAR_ANONYMISATION_STRATEGY,
                            visible=False,
                        )
                        walkthrough_do_initial_clean = gr.Checkbox(
                            label="Do initial clean of text (remove URLs, HTML tags, and non-ASCII characters)",
                            value=DO_INITIAL_TABULAR_DATA_CLEAN,
                            visible=False,
                        )

                        with gr.Row():
                            step_3_back_btn = gr.Button("Back", variant="secondary")
                            step_3_back_btn.click(
                                lambda: gr.Walkthrough(selected=2), outputs=walkthrough
                            )
                            step_3_next_btn = gr.Button("Next", variant="primary")
                    with gr.Step("Redact", id=4):
                        # Page selection (always visible)
                        with gr.Accordion(
                            "Redact only selected pages (default is all pages)",
                            open=False,
                        ):
                            with gr.Row():
                                walkthrough_page_min = gr.Number(
                                    value=DEFAULT_PAGE_MIN,
                                    precision=0,
                                    minimum=0,
                                    maximum=9999,
                                    label="Lowest page to redact (set to 0 to redact from the first page)",
                                )
                                walkthrough_page_max = gr.Number(
                                    value=DEFAULT_PAGE_MAX,
                                    precision=0,
                                    minimum=0,
                                    maximum=9999,
                                    label="Highest page to redact (set to 0 to redact to the last page)",
                                )
                        # Currently not visible as not updating correctly
                        with gr.Accordion(
                            "Costs and time taken estimates", open=True, visible=False
                        ):
                            with gr.Row():
                                # Cost-related components (conditionally visible)
                                walkthrough_textract_output_found_checkbox = (
                                    gr.Checkbox(
                                        value=False,
                                        label="Existing Textract output file found",
                                        interactive=False,
                                        visible=SHOW_COSTS,
                                    )
                                )
                                walkthrough_relevant_ocr_output_with_words_found_checkbox = gr.Checkbox(
                                    value=False,
                                    label="Existing local OCR output file found",
                                    interactive=False,
                                    visible=SHOW_COSTS,
                                )
                                walkthrough_total_pdf_page_count = gr.Number(
                                    label="Total page count",
                                    value=0,
                                    visible=SHOW_COSTS,
                                    interactive=False,
                                )
                                walkthrough_estimated_aws_costs_number = gr.Number(
                                    label="Approximate AWS Textract and/or Comprehend cost (£)",
                                    value=0.00,
                                    precision=2,
                                    visible=SHOW_COSTS,
                                    interactive=False,
                                )
                                walkthrough_estimated_time_taken_number = gr.Number(
                                    label="Approximate time taken to extract text/redact (minutes)",
                                    value=0,
                                    visible=SHOW_COSTS,
                                    precision=2,
                                    interactive=False,
                                )
                        show_cost_codes = GET_COST_CODES or ENFORCE_COST_CODES
                        with gr.Accordion(
                            "Cost code selection", open=True, visible=show_cost_codes
                        ):
                            with gr.Row():
                                # Cost code components (conditionally visible)

                                with gr.Column():
                                    walkthrough_cost_code_dataframe = gr.Dataframe(
                                        value=pd.DataFrame(
                                            columns=["Cost code", "Description"]
                                        ),
                                        row_count=(0, "dynamic"),
                                        label="Existing cost codes",
                                        type="pandas",
                                        interactive=True,
                                        show_search="filter",
                                        visible=show_cost_codes,
                                        wrap=True,
                                        max_height=200,
                                    )
                                    walkthrough_reset_cost_code_dataframe_button = (
                                        gr.Button(
                                            value="Reset code code table filter",
                                            visible=show_cost_codes,
                                        )
                                    )
                                with gr.Column():
                                    walkthrough_cost_code_choice_drop = gr.Dropdown(
                                        value=DEFAULT_COST_CODE,
                                        label="Choose cost code for analysis",
                                        choices=[DEFAULT_COST_CODE],
                                        allow_custom_value=False,
                                        visible=show_cost_codes,
                                    )

                        TRIGGER_DOCUMENT_REDACT_BUTTON = """
                        function triggerChatButtonClick() {

                        // Find the div with id "document-redact-btn"
                        const documentRedactButton = document.getElementById("document-redact-btn");

                        if (!documentRedactButton) {
                            console.error("Error: Could not find element with id 'document-redact-btn'");
                            return;
                        }

                        // Trigger the click event
                        documentRedactButton.click();

                        }"""

                        TRIGGER_TABULAR_REDACT_BUTTON = """
                        function triggerTabularRedactButtonClick() {
                            // Find the div with id "tabular-redact-btn"
                            const tabularRedactButton = document.getElementById("tabular-redact-btn");
                            if (!tabularRedactButton) {
                                console.error("Error: Could not find element with id 'tabular-redact-btn'");
                                return;
                            }
                            // Trigger the click event
                            tabularRedactButton.click();
                        }"""

                        with gr.Row():
                            step_4_back_btn = gr.Button("Back", variant="secondary")
                            step_4_back_btn.click(
                                lambda: gr.Walkthrough(selected=3), outputs=walkthrough
                            )
                            step_4_next_document_redact_btn = gr.Button(
                                "Redact document", variant="primary", visible=True
                            )
                            step_4_next_tabular_redact_btn = gr.Button(
                                "Redact data files", variant="primary", visible=False
                            )
                            step_4_next_document_redact_btn.click(
                                fn=lambda: None, js=TRIGGER_DOCUMENT_REDACT_BUTTON
                            ).then(
                                change_tab_to_tabular_or_document_redactions,
                                inputs=walkthrough_is_data_file,
                                outputs=tabs,
                            )
                            step_4_next_tabular_redact_btn.click(
                                fn=lambda: None, js=TRIGGER_TABULAR_REDACT_BUTTON
                            ).then(
                                change_tab_to_tabular_or_document_redactions,
                                inputs=walkthrough_is_data_file,
                                outputs=tabs,
                            )

            ###
            # QUICKSTART WALKTHROUGH EVENT HANDLERS
            ###
            # Step 1: Route files to appropriate component when Next is clicked
            step_1_next_btn.click(
                fn=route_walkthrough_files,
                inputs=[walkthrough_file_input],
                outputs=[
                    in_doc_files,
                    in_data_files,
                    walkthrough_is_data_file,
                    walkthrough,
                ],
            )

            # Step 2: For data files, populate dropdowns when Next is clicked

            # Note: in_excel_sheets is defined in the "Word or Excel/csv files" tab (id=5)
            # Both tabs are in the same gr.Tabs() context, so components are accessible at runtime
            step_2_next_btn.click(
                fn=handle_step_2_next,
                inputs=[
                    in_data_files,
                    walkthrough_is_data_file,
                    walkthrough_colnames,
                    walkthrough_excel_sheets,
                    walkthrough_text_extract_method_radio,
                ],
                outputs=[
                    walkthrough_colnames,
                    walkthrough_excel_sheets,
                    in_colnames,
                    in_excel_sheets,
                    walkthrough_text_extract_method_radio,
                    walkthrough,
                ],  # type: ignore
            )

            # Update local OCR method radio and AWS Textract settings visibility when text extraction method is selected
            walkthrough_text_extract_method_radio.change(
                fn=handle_text_extract_method_selection,
                inputs=[walkthrough_text_extract_method_radio],
                outputs=[
                    walkthrough_local_ocr_method_radio,
                    walkthrough_handwrite_signature_checkbox,
                ],
            )

            # When data files are uploaded in walkthrough, automatically populate dropdowns

            # Update dropdowns when files are routed to in_data_files
            in_data_files.change(
                fn=update_step_2_on_data_file_upload,
                inputs=[in_data_files, walkthrough_is_data_file],
                outputs=[walkthrough_colnames, walkthrough_excel_sheets],
            )

            # Update Step 3 components visibility when redaction method is selected
            walkthrough_redaction_method_dropdown.change(
                fn=handle_redaction_method_selection,
                inputs=[walkthrough_redaction_method_dropdown],
                outputs=[
                    walkthrough_pii_identification_method_drop,
                    walkthrough_in_redact_entities,
                    walkthrough_in_redact_comprehend_entities,
                    walkthrough_in_redact_llm_entities,
                    walkthrough_custom_llm_instructions_textbox,
                    walkthrough_deny_list_state,
                    walkthrough_allow_list_state,
                    walkthrough_fully_redacted_list_state,
                    walkthrough_selected_terms_accordion_container,
                ],
            )

            # Update entity dropdowns when PII method is selected
            walkthrough_pii_identification_method_drop.change(
                fn=handle_pii_method_selection,
                inputs=[walkthrough_pii_identification_method_drop],
                outputs=[
                    walkthrough_in_redact_entities,
                    walkthrough_in_redact_comprehend_entities,
                    walkthrough_in_redact_llm_entities,
                    walkthrough_custom_llm_instructions_textbox,
                ],
            )

            # Update Step 3 tabular component visibility based on file type
            walkthrough_is_data_file.change(
                fn=update_step_3_tabular_visibility,
                inputs=[walkthrough_is_data_file],
                outputs=[
                    walkthrough_pii_identification_method_drop_tabular,
                    walkthrough_anon_strategy,
                    walkthrough_do_initial_clean,
                ],
            )

            # Step 3: Write values to main components when Next is clicked
            step_3_next_btn.click(
                fn=handle_step_3_next,
                inputs=[
                    walkthrough_text_extract_method_radio,
                    walkthrough_local_ocr_method_radio,
                    walkthrough_handwrite_signature_checkbox,
                    walkthrough_pii_identification_method_drop,
                    walkthrough_in_redact_entities,
                    walkthrough_in_redact_comprehend_entities,
                    walkthrough_in_redact_llm_entities,
                    walkthrough_custom_llm_instructions_textbox,
                    walkthrough_deny_list_state,
                    walkthrough_allow_list_state,
                    walkthrough_fully_redacted_list_state,
                    walkthrough_pii_identification_method_drop_tabular,
                    walkthrough_anon_strategy,
                    walkthrough_do_initial_clean,
                ],
                outputs=[
                    text_extract_method_radio,
                    local_ocr_method_radio,
                    handwrite_signature_checkbox,
                    pii_identification_method_drop,
                    in_redact_entities,
                    in_redact_comprehend_entities,
                    in_redact_llm_entities,
                    custom_llm_instructions_textbox,
                    in_deny_list_state,
                    in_allow_list_state,
                    in_fully_redacted_list_state,
                    pii_identification_method_drop_tabular,
                    anon_strategy,
                    do_initial_clean,
                    walkthrough,
                ],
            )

            # Step 4: Write values to main components when Next is clicked
            # step_4_next_btn.click(
            #     fn=handle_step_4_next,
            #     inputs=[
            #         walkthrough_page_min,
            #         walkthrough_page_max,
            #         walkthrough_textract_output_found_checkbox,
            #         walkthrough_relevant_ocr_output_with_words_found_checkbox,
            #         walkthrough_total_pdf_page_count,
            #         walkthrough_estimated_aws_costs_number,
            #         walkthrough_estimated_time_taken_number,
            #         walkthrough_cost_code_dataframe,
            #         walkthrough_cost_code_choice_drop,
            #     ],
            #     outputs=[
            #         page_min,
            #         page_max,
            #         textract_output_found_checkbox,
            #         relevant_ocr_output_with_words_found_checkbox,
            #         total_pdf_page_count,
            #         estimated_aws_costs_number,
            #         estimated_time_taken_number,
            #         cost_code_dataframe,
            #         cost_code_choice_drop,
            #         walkthrough,
            #     ],
            # )

            # if walkthrough_is_data_file.value:
            # step_4_next_btn.click(
            #     fn=change_tab_to_tabular_or_document_redactions,
            #     inputs=walkthrough_is_data_file,
            #     outputs=tabs,
            # )

            # Reset cost code dataframe filter in walkthrough
            if GET_COST_CODES or ENFORCE_COST_CODES:
                from tools.helper_functions import reset_base_dataframe

                walkthrough_reset_cost_code_dataframe_button.click(
                    reset_base_dataframe,
                    inputs=[cost_code_dataframe_base],
                    outputs=[walkthrough_cost_code_dataframe],
                )

            # Update Step 4 component visibility based on file type
            walkthrough_is_data_file.change(
                fn=update_step_4_visibility,
                inputs=[walkthrough_is_data_file],
                outputs=[
                    step_4_next_document_redact_btn,
                    step_4_next_tabular_redact_btn,
                ],
            )

            # Walkthrough extract/redact button - uses same handlers as document_redact_btn
            # but also updates walkthrough output components and syncs to original components
            # walkthrough_document_redact_btn.click().success(
            #     fn=sync_walkthrough_outputs_to_original,
            #     inputs=[
            #         redaction_output_summary_textbox,
            #         output_file,
            #     ],
            #     outputs=[
            #         walkthrough_redaction_output_summary_textbox,
            #         walkthrough_output_file,
            #         redaction_output_summary_textbox,
            #         output_file,
            #     ],
            # )

            # Walkthrough tabular data redact button - uses same handlers as tabular_data_redact_btn
            # but also updates walkthrough output components and syncs to original components
            # walkthrough_tabular_data_redact_btn.click().success(
            #     fn=sync_walkthrough_tabular_outputs_to_original,
            #     inputs=[
            #         text_output_summary,
            #         text_output_file,
            #     ],
            #     outputs=[
            #         walkthrough_text_output_summary,
            #         walkthrough_text_output_file,
            #         text_output_summary,
            #         text_output_file,
            #     ],
            # )
        ###
        # REDACTION PDF/IMAGES TABLE
        ###
        with gr.Tab("Redact PDFs/images", id=1):

            if SHOW_QUICKSTART:
                show_main_redaction_accordion = False
            else:
                show_main_redaction_accordion = True

            with gr.Accordion("Redaction settings", open=show_main_redaction_accordion):
                in_doc_files.render()
                open_tab_text = ""
                default_text = ""
                textract_text = ""
                comprehend_text = ""
                if DEFAULT_TEXT_EXTRACTION_MODEL == TEXTRACT_TEXT_EXTRACT_OPTION:
                    textract_text = " AWS Textract has a cost per page."
                else:
                    textract_text = ""
                if DEFAULT_PII_DETECTION_MODEL == AWS_PII_OPTION:
                    comprehend_text = (
                        " AWS Comprehend has a cost per character processed."
                    )
                else:
                    comprehend_text = ""
                if textract_text or comprehend_text:
                    open_tab_text = " Open tab to see more details."
                if textract_text and comprehend_text:
                    default_text = ""
                else:
                    default_text = f" The default text extraction method is {DEFAULT_TEXT_EXTRACTION_MODEL}, and the default personal information detection method is {DEFAULT_PII_DETECTION_MODEL}. "

                with gr.Accordion(
                    label=f"Change default redaction settings.{default_text}{textract_text}{comprehend_text}{open_tab_text}".strip(),
                    open=EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT,
                ):

                    if SHOW_OCR_GUI_OPTIONS:
                        with gr.Accordion(
                            "Change default text extraction OCR method",
                            open=True,
                            visible=SHOW_OCR_GUI_OPTIONS,
                        ):
                            text_extract_method_radio.render()
                            # Store accordion references for dynamic visibility control
                            # Initialize visibility based on default text extraction method
                            local_ocr_accordion = gr.Accordion(
                                label="Change default local OCR model",
                                open=EXTRACTION_AND_PII_OPTIONS_OPEN_BY_DEFAULT,
                                visible=(
                                    DEFAULT_TEXT_EXTRACTION_MODEL
                                    == TESSERACT_TEXT_EXTRACT_OPTION
                                ),
                            )
                            with local_ocr_accordion:
                                local_ocr_method_radio.render()

                            inference_server_vlm_accordion = gr.Accordion(
                                "Inference Server VLM Model (for inference-server OCR only)",
                                open=False,
                                visible=(
                                    SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS
                                    and DEFAULT_TEXT_EXTRACTION_MODEL
                                    == TESSERACT_TEXT_EXTRACT_OPTION
                                ),
                            )
                            with inference_server_vlm_accordion:
                                inference_server_vlm_model_textbox.render()

                            aws_textract_signature_accordion = gr.Accordion(
                                "Enable AWS Textract signature detection (default is off)",
                                open=False,
                                visible=(
                                    SHOW_AWS_TEXT_EXTRACTION_OPTIONS
                                    and DEFAULT_TEXT_EXTRACTION_MODEL
                                    == TEXTRACT_TEXT_EXTRACT_OPTION
                                ),
                            )
                            with aws_textract_signature_accordion:
                                handwrite_signature_checkbox.render()
                    else:
                        text_extract_method_radio.render()
                        local_ocr_method_radio.render()
                        inference_server_vlm_model_textbox.render()
                        handwrite_signature_checkbox.render()
                        # Create hidden accordions for consistency (so event handlers can reference them)
                        local_ocr_accordion = gr.Accordion(visible=False)
                        inference_server_vlm_accordion = gr.Accordion(visible=False)
                        aws_textract_signature_accordion = gr.Accordion(visible=False)

                    if SHOW_PII_IDENTIFICATION_OPTIONS:
                        with gr.Accordion(
                            "Change PII identification method",
                            open=True,
                            visible=SHOW_PII_IDENTIFICATION_OPTIONS,
                        ):
                            with gr.Row(equal_height=True):
                                pii_identification_method_drop.render()

                                with gr.Accordion(
                                    "Select entity types to redact", open=True
                                ):
                                    # Store accordion references for dynamic visibility control
                                    # Determine initial visibility based on default PII method
                                    default_pii_method = DEFAULT_PII_DETECTION_MODEL
                                    is_no_redaction_init = (
                                        default_pii_method == NO_REDACTION_PII_OPTION
                                    )
                                    show_local_entities_init = (
                                        not is_no_redaction_init
                                        and (default_pii_method == LOCAL_PII_OPTION)
                                    )
                                    show_comprehend_entities_init = (
                                        not is_no_redaction_init
                                        and (default_pii_method == AWS_PII_OPTION)
                                    )
                                    is_llm_method_init = not is_no_redaction_init and (
                                        default_pii_method
                                        == LOCAL_TRANSFORMERS_LLM_PII_OPTION
                                        or default_pii_method
                                        == INFERENCE_SERVER_PII_OPTION
                                        or default_pii_method == AWS_LLM_PII_OPTION
                                    )

                                    # local_entities_accordion = gr.Accordion(
                                    #     "Local model PII identification model entities",
                                    #     open=False,
                                    #     visible=show_local_entities_init,
                                    # )
                                    # with local_entities_accordion:
                                    in_redact_entities.render()

                                    # comprehend_entities_accordion = gr.Accordion(
                                    #     "AWS Comprehend PII identification model entities",
                                    #     open=False,
                                    #     visible=show_comprehend_entities_init,
                                    # )
                                    # with comprehend_entities_accordion:
                                    in_redact_comprehend_entities.render()

                                    # llm_entities_accordion = gr.Accordion(
                                    #     "LLM PII identification model entities",
                                    #     open=False,
                                    #     visible=is_llm_method_init,
                                    # )
                                    # with llm_entities_accordion:
                                    in_redact_llm_entities.render()

                                # llm_custom_instructions_accordion = gr.Accordion(
                                #     "LLM Custom Instructions (for LLM-based PII detection only)",
                                #     open=True,
                                #     visible=is_llm_method_init,
                                # )
                                # with llm_custom_instructions_accordion:
                                custom_llm_instructions_textbox.render()
                            with gr.Row(equal_height=True):
                                with gr.Accordion(
                                    "Terms to always include or exclude in redactions, and whole page redaction. To add many terms at once, you can load in a file on the Redaction Settings tab.",
                                    open=True,
                                ):
                                    with gr.Row():
                                        in_allow_list_state.render()
                                        in_deny_list_state.render()
                                        in_fully_redacted_list_state.render()

                    else:
                        pii_identification_method_drop.render()
                        in_redact_entities.render()
                        in_redact_comprehend_entities.render()
                        in_redact_llm_entities.render()
                        custom_llm_instructions_textbox.render()
                        in_allow_list_state.render()
                        in_deny_list_state.render()
                        in_fully_redacted_list_state.render()
                        # Create hidden accordions for consistency (so event handlers can reference them)
                        # local_entities_accordion = gr.Accordion(visible=False)
                        # comprehend_entities_accordion = gr.Accordion(visible=False)
                        # llm_entities_accordion = gr.Accordion(visible=False)
                        # llm_custom_instructions_accordion = gr.Accordion(visible=False)

                if SHOW_COSTS:
                    with gr.Accordion(
                        "Estimated costs and time taken. Note that costs shown only include direct usage of AWS services and do not include other running costs (e.g. storage, run-time costs)",
                        open=True,
                        visible=True,
                    ):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                textract_output_found_checkbox = gr.Checkbox(
                                    value=False,
                                    label="Existing Textract output file found",
                                    interactive=False,
                                    visible=True,
                                )
                                relevant_ocr_output_with_words_found_checkbox = (
                                    gr.Checkbox(
                                        value=False,
                                        label="Existing local OCR output file found",
                                        interactive=False,
                                        visible=True,
                                    )
                                )
                            with gr.Column(scale=4):
                                with gr.Row(equal_height=True):
                                    total_pdf_page_count.render()
                                    estimated_aws_costs_number = gr.Number(
                                        label="Approximate AWS Textract and/or Comprehend cost (£)",
                                        value=0.00,
                                        precision=2,
                                        visible=True,
                                        interactive=False,
                                    )
                                    estimated_time_taken_number = gr.Number(
                                        label="Approximate time taken to extract text/redact (minutes)",
                                        value=0,
                                        visible=True,
                                        precision=2,
                                        interactive=False,
                                    )
                else:
                    total_pdf_page_count.render()  # Need to render in both cases, as included in examples

                if GET_COST_CODES or ENFORCE_COST_CODES:
                    with gr.Accordion(
                        "Assign task to cost code", open=True, visible=True
                    ):
                        gr.Markdown(
                            "Please ensure that you have approval from your budget holder before using this app for redaction tasks that incur a cost."
                        )
                        with gr.Row():
                            with gr.Column():
                                with gr.Accordion(
                                    "View and filter cost code table",
                                    open=False,
                                    visible=True,
                                ):
                                    cost_code_dataframe = gr.Dataframe(
                                        value=pd.DataFrame(
                                            columns=["Cost code", "Description"]
                                        ),
                                        row_count=(0, "dynamic"),
                                        label="Existing cost codes",
                                        type="pandas",
                                        interactive=True,
                                        show_search="filter",
                                        visible=True,
                                        wrap=True,
                                        max_height=200,
                                    )
                                    reset_cost_code_dataframe_button = gr.Button(
                                        value="Reset code code table filter"
                                    )
                            with gr.Column():
                                cost_code_choice_drop = gr.Dropdown(
                                    value=DEFAULT_COST_CODE,
                                    label="Choose cost code for analysis",
                                    choices=[DEFAULT_COST_CODE],
                                    allow_custom_value=False,
                                    visible=True,
                                )

                if SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS:
                    with gr.Accordion(
                        "Submit whole document to AWS Textract API (quickest text extraction for large documents)",
                        open=False,
                        visible=True,
                    ):
                        with gr.Row(equal_height=True):
                            gr.Markdown(
                                """Document will be submitted to AWS Textract API service to extract all text in the document. Processing will take place on (secure) AWS servers, and outputs will be stored on S3 for up to 7 days. To download the results, click 'Check status' below and they will be downloaded if ready."""
                            )
                        with gr.Row(equal_height=True):
                            send_document_to_textract_api_btn = gr.Button(
                                "Analyse document with AWS Textract API call",
                                variant="primary",
                                visible=True,
                            )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                textract_job_detail_df = gr.Dataframe(
                                    pd.DataFrame(
                                        columns=[
                                            "job_id",
                                            "file_name",
                                            "job_type",
                                            "signature_extraction",
                                            "job_date_time",
                                        ]
                                    ),
                                    label="Previous job details",
                                    visible=True,
                                    type="pandas",
                                    wrap=True,
                                )
                            with gr.Column(scale=1):
                                job_id_textbox = gr.Textbox(
                                    label="Job ID to check status",
                                    value="",
                                    visible=True,
                                    lines=2,
                                )
                                check_state_of_textract_api_call_btn = gr.Button(
                                    "Check status of Textract job and download",
                                    variant="secondary",
                                    visible=True,
                                )
                        with gr.Row():
                            with gr.Column():
                                textract_job_output_file = gr.File(
                                    label="Textract job output files",
                                    height=100,
                                    visible=True,
                                )
                            with gr.Column():
                                job_current_status = gr.Textbox(
                                    value="",
                                    label="Analysis job current status",
                                    visible=True,
                                )
                                convert_textract_outputs_to_ocr_results = gr.Button(
                                    "Convert Textract job outputs to OCR results",
                                    variant="secondary",
                                    visible=True,
                                )

            with gr.Accordion(label="Extract text and redact document", open=True):
                # gr.Markdown(
                #     """If you only want to redact certain pages, or certain entities (e.g. just email addresses, or a custom list of terms), please go to the Redaction Settings tab."""
                # )
                document_redact_btn = gr.Button(
                    "Extract text and redact document",
                    variant="secondary",
                    scale=4,
                    elem_id="document-redact-btn",
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        redaction_output_summary_textbox = gr.Textbox(
                            label="Output summary", scale=1, lines=4
                        )
                    with gr.Column(scale=2):
                        output_file = gr.File(
                            label="Output files", scale=2
                        )  # , height=FILE_INPUT_HEIGHT)

                go_to_review_redactions_tab_btn = gr.Button(
                    "Review and modify redactions", variant="primary", scale=1
                )

            # Feedback elements are invisible until revealed by redaction action
            pdf_feedback_title = gr.Markdown(
                value="## Please give feedback", visible=False
            )
            pdf_feedback_radio = gr.Radio(
                label="Quality of results",
                choices=["The results were good", "The results were not good"],
                visible=False,
            )
            pdf_further_details_text = gr.Textbox(
                label="Please give more detailed feedback about the results:",
                visible=False,
            )
            pdf_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

            # Feedback elements are invisible until revealed by redaction action
            # all_outputs_in_output_folder_title = gr.Markdown(value="## All outputs in output folder", visible=False)
            # all_outputs_in_output_folder_dataframe = gr.FileExplorer(
            #     root_dir=OUTPUT_FOLDER,
            #     label="All outputs in output folder",
            #     file_count="multiple",
            #     visible=SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER,
            #     interactive=True,
            # )

        ###
        # REVIEW REDACTIONS TAB
        ###
        with gr.Tab("Review redactions", id=2):

            all_page_line_level_ocr_results_with_words_df_base = gr.Dataframe(
                type="pandas",
                label="all_page_line_level_ocr_results_with_words_df_base",
                wrap=False,
                show_search="filter",
                visible=False,
            )

            with gr.Accordion(
                label="Upload PDFs/images and OCR results for review", open=False
            ):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        input_pdf_for_review = gr.File(
                            label="Upload original or '..._for_review.pdf' PDF to begin review process.",
                            file_count="multiple",
                            height=FILE_INPUT_HEIGHT,
                        )
                        upload_pdf_for_review_btn = gr.Button(
                            "1. Load in original PDF or review PDF with redactions",
                            variant="secondary",
                        )
                    with gr.Column(scale=1):
                        input_review_files = gr.File(
                            label="Upload review files here to review suggested redactions. 'review_file' csv The 'ocr_results with words' file can also be provided for searching text and making new redactions.",
                            file_count="multiple",
                            height=FILE_INPUT_HEIGHT,
                        )
                        upload_review_files_btn = gr.Button(
                            "2. Upload review or OCR csv files", variant="secondary"
                        )
            with gr.Row():
                annotate_zoom_in = gr.Button("Zoom in", visible=False)
                annotate_zoom_out = gr.Button("Zoom out", visible=False)
            with gr.Row():
                clear_all_redactions_on_page_btn = gr.Button(
                    "Clear all redactions on page", visible=False
                )

            with gr.Accordion(label="View and edit review file data", open=False):
                review_file_df = gr.Dataframe(
                    value=pd.DataFrame(),
                    headers=[
                        "image",
                        "page",
                        "label",
                        "color",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "text",
                        "id",
                    ],
                    row_count=(0, "dynamic"),
                    label="Review file data",
                    visible=True,
                    type="pandas",
                    wrap=True,
                    show_search=True,
                )

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(equal_height=True):
                        annotation_last_page_button = gr.Button(
                            "Previous page", scale=4
                        )
                        annotate_current_page = gr.Number(
                            value=1,
                            label="Current page",
                            precision=0,
                            scale=2,
                            min_width=50,
                            minimum=1,
                        )
                        annotate_max_pages = gr.Number(
                            value=1,
                            label="Total pages",
                            precision=0,
                            interactive=False,
                            scale=2,
                            min_width=50,
                            minimum=1,
                        )
                        annotation_next_page_button = gr.Button("Next page", scale=4)

                    zoom_str = str(annotator_zoom_number) + "%"

                    annotator = image_annotator(
                        label="Modify redaction boxes",
                        label_list=["Redaction"],
                        label_colors=[(0, 0, 0)],
                        show_label=False,
                        height=zoom_str,
                        width=zoom_str,
                        box_min_size=1,
                        box_selected_thickness=2,
                        handle_size=4,
                        sources=None,  # ["upload"],
                        show_clear_button=False,
                        show_share_button=False,
                        show_remove_button=False,
                        handles_cursor=True,
                        interactive=False,
                    )

                    with gr.Row(equal_height=True):
                        annotation_last_page_button_bottom = gr.Button(
                            "Previous page", scale=4
                        )
                        annotate_current_page_bottom = gr.Number(
                            value=1,
                            label="Current page",
                            precision=0,
                            interactive=True,
                            scale=2,
                            min_width=50,
                            minimum=1,
                        )
                        annotate_max_pages_bottom = gr.Number(
                            value=1,
                            label="Total pages",
                            precision=0,
                            interactive=False,
                            scale=2,
                            min_width=50,
                            minimum=1,
                        )
                        annotation_next_page_button_bottom = gr.Button(
                            "Next page", scale=4
                        )

                with gr.Column(scale=1):
                    annotation_button_apply = gr.Button(
                        "Apply revised redactions to PDF", variant="primary"
                    )
                    update_current_page_redactions_btn = gr.Button(
                        value="Save changes on current page to file",
                        variant="secondary",
                    )

                    with gr.Tab("Modify existing redactions", id=3):
                        with gr.Accordion("Search suggested redactions", open=True):
                            with gr.Row(equal_height=True):
                                recogniser_entity_dropdown = gr.Dropdown(
                                    label="Redaction category",
                                    value="ALL",
                                    allow_custom_value=True,
                                )
                                page_entity_dropdown = gr.Dropdown(
                                    label="Page", value="ALL", allow_custom_value=True
                                )
                            text_entity_dropdown = gr.Dropdown(
                                label="Text", value="ALL", allow_custom_value=True
                            )
                            reset_dropdowns_btn = gr.Button(value="Reset filters")
                            recogniser_entity_dataframe = gr.Dataframe(
                                pd.DataFrame(
                                    data={
                                        "page": list(),
                                        "label": list(),
                                        "text": list(),
                                        "id": list(),
                                    }
                                ),
                                row_count=(0, "dynamic"),
                                type="pandas",
                                label="Click table row to select and go to page",
                                headers=["page", "label", "text", "id"],
                                wrap=True,
                                max_height=400,
                            )

                            with gr.Row(equal_height=True):
                                exclude_selected_btn = gr.Button(
                                    value="Exclude all redactions in table"
                                )

                            with gr.Accordion("Selected redaction row", open=True):
                                selected_entity_dataframe_row = gr.Dataframe(
                                    pd.DataFrame(
                                        data={
                                            "page": list(),
                                            "label": list(),
                                            "text": list(),
                                            "id": list(),
                                        }
                                    ),
                                    row_count=(0, "dynamic"),
                                    type="pandas",
                                    visible=True,
                                    headers=["page", "label", "text", "id"],
                                    wrap=True,
                                )
                                exclude_selected_row_btn = gr.Button(
                                    value="Exclude specific redaction row"
                                )
                                exclude_text_with_same_as_selected_row_btn = gr.Button(
                                    value="Exclude all redactions with same text as selected row"
                                )

                            undo_last_removal_btn = gr.Button(
                                value="Undo last element removal", variant="primary"
                            )

                    with gr.Tab("Search text and redact", id=7):
                        with gr.Accordion("Search text", open=True):
                            with gr.Row(equal_height=True):
                                page_entity_dropdown_redaction = gr.Dropdown(
                                    label="Page",
                                    value="1",
                                    allow_custom_value=True,
                                    scale=4,
                                )
                                reset_dropdowns_btn_new = gr.Button(
                                    value="Reset page filter", scale=1
                                )

                            with gr.Row(equal_height=True):
                                multi_word_search_text = gr.Textbox(
                                    label="Multi-word text search (regex enabled below)",
                                    value="",
                                    scale=4,
                                )
                                multi_word_search_text_btn = gr.Button(
                                    value="Search", scale=1
                                )

                            with gr.Accordion("Search options", open=False):
                                similarity_search_score_minimum = gr.Number(
                                    value=1.0,
                                    minimum=0.4,
                                    maximum=1.0,
                                    label="Minimum similarity score for match (max=1)",
                                    visible=False,
                                )  # Not used anymore for this exact search

                                with gr.Row():
                                    with gr.Column():
                                        new_redaction_text_label = gr.Textbox(
                                            label="Label for new redactions",
                                            value="Redaction",
                                        )
                                        colour_label = gr.Textbox(
                                            label="Colour for labels (three number RGB format, max 255 with brackets)",
                                            value=CUSTOM_BOX_COLOUR,
                                        )
                                    with gr.Column():
                                        use_regex_search = gr.Checkbox(
                                            label="Enable regex pattern matching",
                                            value=False,
                                            info="When enabled, the search text will be treated as a regular expression pattern instead of literal text",
                                        )

                            all_page_line_level_ocr_results_with_words_df = (
                                gr.Dataframe(
                                    pd.DataFrame(
                                        data={
                                            "page": list(),
                                            "line": list(),
                                            "word_text": list(),
                                            "word_x0": list(),
                                            "word_y0": list(),
                                            "word_x1": list(),
                                            "word_y1": list(),
                                        }
                                    ),
                                    row_count=(0, "dynamic"),
                                    type="pandas",
                                    label="Click table row to select and go to page",
                                    headers=[
                                        "page",
                                        "line",
                                        "word_text",
                                        "word_x0",
                                        "word_y0",
                                        "word_x1",
                                        "word_y1",
                                    ],
                                    wrap=False,
                                    max_height=400,
                                    show_search="filter",
                                )
                            )

                            redact_selected_btn = gr.Button(
                                value="Redact all text in table"
                            )
                            reset_ocr_with_words_df_btn = gr.Button(
                                value="Reset table to original state"
                            )

                            with gr.Accordion("Selected row", open=True):
                                selected_entity_dataframe_row_redact = gr.Dataframe(
                                    pd.DataFrame(
                                        data={
                                            "page": list(),
                                            "line": list(),
                                            "word_text": list(),
                                            "word_x0": list(),
                                            "word_y0": list(),
                                            "word_x1": list(),
                                            "word_y1": list(),
                                        }
                                    ),
                                    row_count=(0, "dynamic"),
                                    type="pandas",
                                    headers=[
                                        "page",
                                        "line",
                                        "word_text",
                                        "word_x0",
                                        "word_y0",
                                        "word_x1",
                                        "word_y1",
                                    ],
                                    wrap=False,
                                )
                                redact_selected_row_btn = gr.Button(
                                    value="Redact specific text row"
                                )
                                redact_text_with_same_as_selected_row_btn = gr.Button(
                                    value="Redact all words with same text as selected row"
                                )

                            undo_last_redact_btn = gr.Button(
                                value="Undo latest redaction", variant="primary"
                            )

                    with gr.Accordion("Search extracted text", open=True):
                        all_page_line_level_ocr_results_df = gr.Dataframe(
                            value=pd.DataFrame(columns=["page", "line", "text"]),
                            headers=["page", "line", "text"],
                            row_count=(0, "dynamic"),
                            label="All OCR results",
                            visible=True,
                            type="pandas",
                            wrap=True,
                            show_search="filter",
                            show_label=False,
                            column_widths=["15%", "15%", "70%"],
                            max_height=400,
                        )
                        reset_all_ocr_results_btn = gr.Button(
                            value="Reset OCR output table filter"
                        )
                        selected_ocr_dataframe_row = gr.Dataframe(
                            pd.DataFrame(
                                data={"page": list(), "line": list(), "text": list()}
                            ),
                            col_count=3,
                            type="pandas",
                            visible=False,
                            headers=["page", "line", "text"],
                            wrap=True,
                        )

            with gr.Accordion(
                "Convert review files loaded above to Adobe format, or convert from Adobe format to review file",
                open=False,
            ):
                convert_review_file_to_adobe_btn = gr.Button(
                    "Convert review file to Adobe comment format", variant="primary"
                )
                adobe_review_files_out = gr.File(
                    label="Output Adobe comment files will appear here. If converting from .xfdf file to review_file.csv, upload the original pdf with the xfdf file here then click Convert below.",
                    file_count="multiple",
                    file_types=[".csv", ".xfdf", ".pdf"],
                )
                convert_adobe_to_review_file_btn = gr.Button(
                    "Convert Adobe .xfdf comment file to review_file.csv",
                    variant="secondary",
                )

        ###
        # IDENTIFY DUPLICATE PAGES TAB
        ###
        with gr.Tab(label="Identify duplicate pages", id=4):
            gr.Markdown(
                "Search for duplicate pages/subdocuments in your ocr_output files. By default, this function will search for duplicate text across multiple pages, and then join consecutive matching pages together into matched 'subdocuments'. The results can be reviewed below, false positives removed, and then the verified results applied to a document you have loaded in on the 'Review redactions' tab."
            )

            # Examples for duplicate page detection
            if SHOW_EXAMPLES:
                gr.Markdown(
                    "### Try an example - Click on an example below and then the 'Identify duplicate pages/subdocuments' button:"
                )

                # Check if duplicate example file exists
                duplicate_example_file = "example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv"

                if os.path.exists(duplicate_example_file):

                    def show_duplicate_info_box_on_click(
                        in_duplicate_pages,
                        duplicate_threshold_input,
                        min_word_count_input,
                        combine_page_text_for_duplicates_bool,
                    ):
                        gr.Info(
                            "Example data loaded. Now click on 'Identify duplicate pages/subdocuments' below to run the example duplicate detection."
                        )

                    duplicate_examples = gr.Examples(
                        examples=[
                            [
                                [duplicate_example_file],
                                0.95,
                                10,
                                True,
                            ],
                            [
                                [duplicate_example_file],
                                0.95,
                                3,
                                False,
                            ],
                        ],
                        inputs=[
                            in_duplicate_pages,
                            duplicate_threshold_input,
                            min_word_count_input,
                            combine_page_text_for_duplicates_bool,
                        ],
                        example_labels=[
                            "Find duplicate pages of text in document OCR outputs",
                            "Find duplicate text lines in document OCR outputs",
                        ],
                        fn=show_duplicate_info_box_on_click,
                        run_on_click=True,
                    )

            with gr.Accordion("Step 1: Configure and run analysis", open=True):
                in_duplicate_pages.render()

                with gr.Accordion("Duplicate matching parameters", open=False):
                    with gr.Row():
                        duplicate_threshold_input.render()

                        min_word_count_input.render()

                        combine_page_text_for_duplicates_bool.render()

                    gr.Markdown("#### Matching Strategy")
                    greedy_match_input = gr.Checkbox(
                        label="Enable 'subdocument' matching",
                        value=USE_GREEDY_DUPLICATE_DETECTION,
                        info="If checked, finds the longest possible sequence of matching pages (subdocuments), minimum length one page. Overrides the slider below.",
                    )
                    min_consecutive_pages_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=DEFAULT_MIN_CONSECUTIVE_PAGES,
                        step=1,
                        label="Minimum consecutive pages (modified subdocument match)",
                        info="If greedy matching option above is unticked, use this to find only subdocuments of a minimum number of consecutive pages.",
                    )

                find_duplicate_pages_btn = gr.Button(
                    value="Identify duplicate pages/subdocuments", variant="primary"
                )

            with gr.Accordion("Step 2: Review and refine results", open=True):
                gr.Markdown(
                    "### Analysis summary\nClick on a row to select it for preview or exclusion."
                )

                with gr.Row():
                    results_df_preview = gr.Dataframe(
                        label="Similarity Results",
                        headers=[
                            "Page1_File",
                            "Page1_Start_Page",
                            "Page1_End_Page",
                            "Page2_File",
                            "Page2_Start_Page",
                            "Page2_End_Page",
                            "Match_Length",
                            "Avg_Similarity",
                            "Page1_Text",
                            "Page2_Text",
                        ],
                        wrap=True,
                        show_search=True,
                    )
                with gr.Row():
                    exclude_match_btn = gr.Button(
                        value="❌ Exclude Selected Match", variant="stop"
                    )
                    gr.Markdown(
                        "Click a row in the table, then click this button to remove it from the results and update the downloadable files."
                    )

                gr.Markdown("### Full Text Preview of Selected Match")
                with gr.Row():
                    page1_text_preview = gr.Dataframe(
                        label="Match Source (Document 1)",
                        wrap=True,
                        headers=["page", "text"],
                        show_search=True,
                    )
                    page2_text_preview = gr.Dataframe(
                        label="Match Duplicate (Document 2)",
                        wrap=True,
                        headers=["page", "text"],
                        show_search=True,
                    )

                gr.Markdown("### Downloadable Files")
                duplicate_files_out = gr.File(
                    label="Download analysis summary and redaction lists (.csv)",
                    file_count="multiple",
                    height=FILE_INPUT_HEIGHT,
                )

                with gr.Row():
                    apply_match_btn = gr.Button(
                        value="Apply relevant duplicate page output to document currently under review",
                        variant="secondary",
                    )

        ###
        # WORD / TABULAR DATA TAB
        ###
        with gr.Tab(label="Word or Excel/csv files", id=5):

            gr.Markdown(
                """Choose a Word or tabular data file (xlsx or csv) to redact. Note that when redacting complex Word files with e.g. images, some content/formatting will be removed, and it may not attempt to redact headers. You may prefer to convert the doc file to PDF in Word, and then run it through the first tab of this app (Print to PDF in print settings). Alternatively, an xlsx file output is provided when redacting docx files directly to allow for copying and pasting outputs back into the original document if preferred."""
            )

            # Examples for Word/Excel/csv redaction and tabular duplicate detection
            if SHOW_EXAMPLES:
                gr.Markdown(
                    "### Try an example - Click on an example below and then the 'Redact text/data files' button for redaction, or the 'Find duplicate cells/rows' button for duplicate detection:"
                )

                # Check which tabular example files exist
                tabular_example_files = [
                    "example_data/combined_case_notes.csv",
                    "example_data/Bold minimalist professional cover letter.docx",
                    "example_data/Lambeth_2030-Our_Future_Our_Lambeth.pdf.csv",
                ]

                available_tabular_examples = list()
                tabular_example_labels = list()

                # Check each tabular example file and add to examples if it exists
                if os.path.exists(tabular_example_files[0]):
                    available_tabular_examples.append(
                        [
                            [tabular_example_files[0]],
                            ["Case Note", "Client"],
                            "Local",
                            "replace with 'REDACTED'",
                            [tabular_example_files[0]],
                            ["Case Note"],
                            3,
                        ]
                    )
                    tabular_example_labels.append(
                        "CSV file redaction with specific columns - remove text"
                    )

                if os.path.exists(tabular_example_files[1]):
                    available_tabular_examples.append(
                        [
                            [tabular_example_files[1]],
                            [],
                            "Local",
                            "replace with 'REDACTED'",
                            [],
                            [],
                            3,
                        ]
                    )
                    tabular_example_labels.append(
                        "Word document redaction - replace with REDACTED"
                    )

                if os.path.exists(tabular_example_files[2]):
                    available_tabular_examples.append(
                        [
                            [tabular_example_files[2]],
                            ["text"],
                            "Local",
                            "replace with 'REDACTED'",
                            [tabular_example_files[2]],
                            ["text"],
                            3,
                        ]
                    )
                    tabular_example_labels.append(
                        "Tabular duplicate detection in CSV files"
                    )

                # Only create examples if we have available files
                if available_tabular_examples:

                    def show_tabular_info_box_on_click(
                        in_data_files,
                        in_colnames,
                        pii_identification_method_drop_tabular,
                        anon_strategy,
                        in_tabular_duplicate_files,
                        tabular_text_columns,
                        tabular_min_word_count,
                    ):
                        gr.Info(
                            "Example data loaded. Now click on 'Redact text/data files' or 'Find duplicate cells/rows' below to run the example."
                        )

                        return (
                            gr.File(value=in_data_files),  # walkthrough_file_input
                            gr.Radio(
                                value=pii_identification_method_drop_tabular
                            ),  # walkthrough_pii_identification_method_drop_tabular
                            gr.Radio(value=anon_strategy),  # walkthrough_anon_strategy
                        )

                    tabular_examples = gr.Examples(
                        examples=available_tabular_examples,
                        inputs=[
                            in_data_files,
                            in_colnames,
                            pii_identification_method_drop_tabular,
                            anon_strategy,
                            in_tabular_duplicate_files,
                            tabular_text_columns,
                            tabular_min_word_count,
                        ],
                        outputs=[
                            walkthrough_file_input,
                            walkthrough_pii_identification_method_drop_tabular,
                            walkthrough_anon_strategy,
                        ],
                        example_labels=tabular_example_labels,
                        fn=show_tabular_info_box_on_click,
                        run_on_click=True,
                    )

            with gr.Accordion(
                "Redact Word or Excel/csv files options",
                open=show_main_redaction_accordion,
            ):
                with gr.Accordion("Upload docx, xlsx, or csv files", open=True):
                    in_data_files.render()
                with gr.Accordion("Redact open text", open=False):
                    in_text = gr.Textbox(
                        label="Enter open text",
                        lines=10,
                        max_length=MAX_OPEN_TEXT_CHARACTERS,
                    )

                in_excel_sheets.render()

                in_colnames.render()

                pii_identification_method_drop_tabular.render()

                with gr.Accordion(
                    "Anonymisation output format - by default will replace PII with a blank space",
                    open=False,
                ):
                    with gr.Row():
                        anon_strategy.render()

                        do_initial_clean.render()

                tabular_data_redact_btn = gr.Button(
                    "Redact text/data files",
                    variant="primary",
                    elem_id="tabular-redact-btn",
                )

            with gr.Accordion(label="Redact Word/data files", open=True):
                with gr.Row():
                    text_output_summary = gr.Textbox(label="Output result", lines=4)
                    text_output_file = gr.File(label="Output files")
                    text_tabular_files_done = gr.Number(
                        value=0,
                        label="Number of tabular files redacted",
                        interactive=False,
                        visible=False,
                    )

            ###
            # TABULAR DUPLICATE DETECTION
            ###
            with gr.Accordion(label="Find duplicate cells in tabular data", open=False):
                gr.Markdown(
                    """Find duplicate cells or rows in CSV, Excel, or Parquet files. This tool analyses text content across all columns to identify similar or identical entries that may be duplicates. You can review the results and choose to remove duplicate rows from your files."""
                )

                with gr.Accordion(
                    "Step 1: Upload files and configure analysis", open=True
                ):
                    in_tabular_duplicate_files.render()

                    with gr.Row(equal_height=True):
                        tabular_duplicate_threshold = gr.Number(
                            value=DEFAULT_DUPLICATE_DETECTION_THRESHOLD,
                            label="Similarity threshold",
                            info="Score (0-1) to consider cells a match. 1 = perfect match.",
                        )

                        tabular_min_word_count.render()

                        do_initial_clean_dup = gr.Checkbox(
                            label="Do initial clean of text (remove URLs, HTML tags, and non-ASCII characters)",
                            value=DO_INITIAL_TABULAR_DATA_CLEAN,
                        )
                        remove_duplicate_rows = gr.Checkbox(
                            label="Remove duplicate rows from deduplicated files",
                            value=REMOVE_DUPLICATE_ROWS,
                        )

                    with gr.Row():
                        in_excel_tabular_sheets = gr.Dropdown(
                            choices=list(),
                            multiselect=True,
                            label="Select Excel sheet names that you want to deduplicate (showing sheets present across all Excel files).",
                            visible=True,
                            allow_custom_value=True,
                        )

                        tabular_text_columns.render()

                    find_tabular_duplicates_btn = gr.Button(
                        value="Find duplicate cells/rows", variant="primary"
                    )

                with gr.Accordion("Step 2: Review results", open=True):
                    gr.Markdown(
                        "### Duplicate Analysis Results\nClick on a row to see more details about the duplicate match."
                    )

                    tabular_results_df = gr.Dataframe(
                        label="Duplicate Cell Matches",
                        headers=[
                            "File1",
                            "Row1",
                            "File2",
                            "Row2",
                            "Similarity_Score",
                            "Text1",
                            "Text2",
                        ],
                        wrap=True,
                        show_search=True,
                    )

                    with gr.Row(equal_height=True):
                        tabular_selected_row_index = gr.Number(
                            value=None, visible=False
                        )
                        tabular_text1_preview = gr.Textbox(
                            label="Text from File 1", lines=3, interactive=False
                        )
                        tabular_text2_preview = gr.Textbox(
                            label="Text from File 2", lines=3, interactive=False
                        )

                with gr.Accordion("Step 3: Remove duplicates", open=True):
                    gr.Markdown(
                        "### Remove Duplicate Rows\nSelect a file and click to remove duplicate rows based on the analysis above."
                    )

                    with gr.Row():
                        tabular_file_to_clean = gr.Dropdown(
                            choices=list(),
                            label="Select file to clean",
                            info="Choose which file to remove duplicates from",
                            visible=False,
                        )
                        clean_duplicates_btn = gr.Button(
                            value="Remove duplicate rows from selected file",
                            variant="secondary",
                            visible=False,
                        )

                    tabular_cleaned_file = gr.File(
                        label="Download cleaned file (duplicates removed)",
                        visible=True,
                        interactive=False,
                    )

            # Feedback elements are invisible until revealed by redaction action
            data_feedback_title = gr.Markdown(
                value="## Please give feedback", visible=False
            )
            data_feedback_radio = gr.Radio(
                label="Please give some feedback about the results of the redaction.",
                choices=["The results were good", "The results were not good"],
                visible=False,
                show_label=True,
            )
            data_further_details_text = gr.Textbox(
                label="Please give more detailed feedback about the results:",
                visible=False,
            )
            data_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

        ###
        # SETTINGS TAB
        ###
        with gr.Tab(label="Redaction settings", id=6):
            with gr.Accordion(
                "Custom allow, deny, and full page redaction lists", open=True
            ):
                with gr.Row():
                    with gr.Column():
                        in_allow_list = gr.File(
                            label="Import allow list file - csv table with one column of a different word/phrase on each row (case insensitive). Terms in this file will not be redacted.",
                            file_count="multiple",
                            height=FILE_INPUT_HEIGHT,
                        )
                        in_allow_list_text = gr.Textbox(
                            label="Custom allow list load status"
                        )
                    with gr.Column():
                        in_deny_list.render()  # Defined at beginning of file
                        in_deny_list_text = gr.Textbox(
                            label="Custom deny list load status"
                        )
                    with gr.Column():
                        in_fully_redacted_list.render()  # Defined at beginning of file
                        in_fully_redacted_list_text = gr.Textbox(
                            label="Fully redacted page list load status"
                        )

                    with gr.Row():
                        with gr.Column(scale=2):
                            markdown_placeholder = gr.Markdown("")
                        with gr.Column(scale=1):
                            apply_fully_redacted_list_btn = gr.Button(
                                value="Apply whole page redaction list to document currently under review",
                                variant="secondary",
                            )

            with gr.Accordion("Select entity types to redact", open=True):

                with gr.Row():
                    max_fuzzy_spelling_mistakes_num = gr.Number(
                        label="Maximum number of spelling mistakes allowed for fuzzy matching (CUSTOM_FUZZY entity).",
                        value=DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
                        minimum=0,
                        maximum=9,
                        precision=0,
                    )
                    match_fuzzy_whole_phrase_bool = gr.Checkbox(
                        label="Should fuzzy search match on entire phrases in deny list (as opposed to each word individually)?",
                        value=True,
                    )

            with gr.Accordion("Redact only selected pages", open=False):
                with gr.Row():
                    page_min.render()
                    page_max.render()

            if SHOW_LANGUAGE_SELECTION:
                with gr.Accordion("Language selection", open=False):
                    gr.Markdown(
                        """Note that AWS Textract is compatible with English, Spanish, Italian, Portuguese, French, and German, and handwriting detection is only available in English. AWS Comprehend for detecting PII is only compatible with English and Spanish.
                    The local models (Tesseract and SpaCy) are compatible with the other languages in the list below. However, the language packs for these models need to be installed on your system. When you first run a document through the app, the language packs will be downloaded automatically, but please expect a delay as the models are large."""
                    )
                    with gr.Row():
                        chosen_language_full_name_drop = gr.Dropdown(
                            value=DEFAULT_LANGUAGE_FULL_NAME,
                            choices=MAPPED_LANGUAGE_CHOICES,
                            label="Chosen language",
                            multiselect=False,
                            visible=True,
                        )
                        chosen_language_drop = gr.Dropdown(
                            value=DEFAULT_LANGUAGE,
                            choices=LANGUAGE_CHOICES,
                            label="Chosen language short code",
                            multiselect=False,
                            visible=True,
                            interactive=False,
                        )
            else:
                chosen_language_full_name_drop = gr.Dropdown(
                    value=DEFAULT_LANGUAGE_FULL_NAME,
                    choices=MAPPED_LANGUAGE_CHOICES,
                    label="Chosen language",
                    multiselect=False,
                    visible=False,
                )
                chosen_language_drop = gr.Dropdown(
                    value=DEFAULT_LANGUAGE,
                    choices=LANGUAGE_CHOICES,
                    label="Chosen language short code",
                    multiselect=False,
                    visible=False,
                )

            with gr.Accordion("Use API keys for AWS services", open=False):
                with gr.Row():
                    aws_access_key_textbox = gr.Textbox(
                        value="",
                        label="AWS access key for account with permissions for AWS Textract and Comprehend",
                        visible=True,
                        type="password",
                    )
                    aws_secret_key_textbox = gr.Textbox(
                        value="",
                        label="AWS secret key for account with permissions for AWS Textract and Comprehend",
                        visible=True,
                        type="password",
                    )

            with gr.Accordion("Log file outputs", open=False):
                log_files_output = gr.File(label="Log file output", interactive=False)

            with gr.Accordion("S3 output settings", open=False):
                save_outputs_to_s3_checkbox = gr.Checkbox(
                    label="Save redaction outputs to S3 (requires RUN_AWS_FUNCTIONS=True and S3_OUTPUTS_FOLDER set)",
                    value=SAVE_OUTPUTS_TO_S3,
                )
                s3_output_folder_display = gr.Textbox(
                    label="Resolved S3 outputs folder",
                    value="",
                    interactive=False,
                )

            with gr.Accordion("Combine multiple review files", open=False):
                multiple_review_files_in_out = gr.File(
                    label="Combine multiple review_file.csv files together here.",
                    file_count="multiple",
                    file_types=[".csv"],
                )
                merge_multiple_review_files_btn = gr.Button(
                    "Merge multiple review files into one", variant="primary"
                )

        if SHOW_ALL_OUTPUTS_IN_OUTPUT_FOLDER:
            with gr.Accordion(
                "View all and download all output files from this session",
                open=False,
            ):
                all_output_files_btn.render()
                all_output_files.render()
                all_outputs_file_download.render()
        else:
            all_output_files_btn.render()
            all_output_files.render()
            all_outputs_file_download.render()

    ###
    # UI INTERACTION
    ###

    ###
    # PDF/IMAGE REDACTION
    ###
    # Recalculate estimated costs based on changes to inputs
    if SHOW_COSTS:
        # Calculate costs
        total_pdf_page_count.change(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        text_extract_method_radio.change(
            fn=check_for_relevant_ocr_output_with_words,
            inputs=[
                doc_file_name_no_extension_textbox,
                text_extract_method_radio,
                output_folder_textbox,
            ],
            outputs=[relevant_ocr_output_with_words_found_checkbox],
        ).success(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        pii_identification_method_drop.change(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        handwrite_signature_checkbox.change(
            fn=check_for_existing_textract_file,
            inputs=[
                doc_file_name_no_extension_textbox,
                output_folder_textbox,
                handwrite_signature_checkbox,
            ],
            outputs=[textract_output_found_checkbox],
        ).then(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        textract_output_found_checkbox.change(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        only_extract_text_radio.change(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )
        textract_output_found_checkbox.change(
            calculate_aws_costs,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
            ],
            outputs=[estimated_aws_costs_number],
        )

        # Calculate time taken
        total_pdf_page_count.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        text_extract_method_radio.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        pii_identification_method_drop.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        handwrite_signature_checkbox.change(
            fn=check_for_existing_textract_file,
            inputs=[
                doc_file_name_no_extension_textbox,
                output_folder_textbox,
                handwrite_signature_checkbox,
            ],
            outputs=[textract_output_found_checkbox],
        ).then(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        textract_output_found_checkbox.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                handwrite_signature_checkbox,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        only_extract_text_radio.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        textract_output_found_checkbox.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )
        relevant_ocr_output_with_words_found_checkbox.change(
            calculate_time_taken,
            inputs=[
                total_pdf_page_count,
                text_extract_method_radio,
                pii_identification_method_drop,
                textract_output_found_checkbox,
                only_extract_text_radio,
                relevant_ocr_output_with_words_found_checkbox,
            ],
            outputs=[estimated_time_taken_number],
        )

        # Automatically set local_ocr_method_radio to "bedrock-vlm" when AWS Bedrock VLM is selected
        def auto_set_local_ocr_for_bedrock_vlm(text_extract_method):
            """Automatically set local OCR method to bedrock-vlm when AWS Bedrock VLM is selected."""
            if text_extract_method == BEDROCK_VLM_TEXT_EXTRACT_OPTION:
                # Only set if "bedrock-vlm" is a valid option
                if "bedrock-vlm" in LOCAL_OCR_MODEL_OPTIONS:
                    return gr.update(value="bedrock-vlm")
            return gr.update()

        text_extract_method_radio.change(
            fn=auto_set_local_ocr_for_bedrock_vlm,
            inputs=[text_extract_method_radio],
            outputs=[local_ocr_method_radio],
        )

        # Dynamic visibility handlers for main redaction tab
        # Update visibility of OCR-related accordions based on text extraction method selection
        text_extract_method_radio.change(
            fn=handle_main_text_extract_method_selection,
            inputs=[text_extract_method_radio],
            outputs=[
                local_ocr_accordion,
                inference_server_vlm_accordion,
                aws_textract_signature_accordion,
            ],
        )

        # Update visibility of PII-related accordions based on PII method selection
        pii_identification_method_drop.change(
            fn=handle_main_pii_method_selection,
            inputs=[pii_identification_method_drop],
            outputs=[
                pii_identification_method_drop,  # Keep visible so user can change
                in_redact_entities,
                in_redact_comprehend_entities,
                in_redact_llm_entities,
                custom_llm_instructions_textbox,
            ],
        )

    # Allow user to select items from cost code dataframe for cost code
    if SHOW_COSTS and (GET_COST_CODES or ENFORCE_COST_CODES):
        cost_code_dataframe.select(
            df_select_callback_cost,
            inputs=[cost_code_dataframe],
            outputs=[cost_code_choice_drop],
        )
        reset_cost_code_dataframe_button.click(
            reset_base_dataframe,
            inputs=[cost_code_dataframe_base],
            outputs=[cost_code_dataframe],
        )

        cost_code_choice_drop.select(
            update_cost_code_dataframe_from_dropdown_select,
            inputs=[cost_code_choice_drop, cost_code_dataframe_base],
            outputs=[cost_code_dataframe],
        )

    in_doc_files.upload(
        fn=get_input_file_names,
        inputs=[in_doc_files],
        outputs=[
            doc_file_name_no_extension_textbox,
            doc_file_name_with_extension_textbox,
            doc_full_file_name_textbox,
            doc_file_name_textbox_list,
            total_pdf_page_count,
        ],
    ).success(
        fn=prepare_image_or_pdf,
        inputs=[
            in_doc_files,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            first_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool_false,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_page_line_level_ocr_results_df_base,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=check_for_existing_textract_file,
        inputs=[
            doc_file_name_no_extension_textbox,
            output_folder_textbox,
            handwrite_signature_checkbox,
        ],
        outputs=[textract_output_found_checkbox],
    ).success(
        fn=check_for_relevant_ocr_output_with_words,
        inputs=[
            doc_file_name_no_extension_textbox,
            text_extract_method_radio,
            output_folder_textbox,
        ],
        outputs=[relevant_ocr_output_with_words_found_checkbox],
    )

    # Run redaction function
    document_redact_btn.click(
        fn=reset_state_vars,
        outputs=[
            all_image_annotations_state,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            comprehend_query_number,
            textract_metadata_textbox,
            annotator,
            output_file_list_state,
            log_files_output_list_state,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            pdf_doc_state,
            duplication_file_path_outputs_list_state,
            redaction_output_summary_textbox,
            is_a_textract_api_call,
            textract_query_number,
            all_page_line_level_ocr_results_with_words,
            input_review_files,
        ],
    ).success(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        fn=choose_and_run_redactor,
        inputs=[
            in_doc_files,
            prepared_pdf_state,
            images_pdf_state,
            in_redact_entities,
            in_redact_comprehend_entities,
            in_redact_llm_entities,
            text_extract_method_radio,
            in_allow_list_state,
            in_deny_list_state,
            in_fully_redacted_list_state,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            output_file_list_state,
            log_files_output_list_state,
            first_loop_state,
            page_min,
            page_max,
            actual_time_taken_number,
            handwrite_signature_checkbox,
            textract_metadata_textbox,
            all_image_annotations_state,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            pdf_doc_state,
            current_loop_page_number,
            page_break_return,
            pii_identification_method_drop,
            comprehend_query_number,
            max_fuzzy_spelling_mistakes_num,
            match_fuzzy_whole_phrase_bool,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            annotate_max_pages,
            review_file_df,
            output_folder_textbox,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            only_extract_text_radio,
            duplication_file_path_outputs_list_state,
            latest_review_file_path,
            input_folder_textbox,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            local_ocr_method_radio,
            chosen_language_drop,
            input_review_files,
            custom_llm_instructions_textbox,
            inference_server_vlm_model_textbox,
        ],
        outputs=[
            redaction_output_summary_textbox,
            output_file,
            output_file_list_state,
            latest_file_completed_num,
            log_files_output,
            log_files_output_list_state,
            actual_time_taken_number,
            textract_metadata_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            current_loop_page_number,
            page_break_return,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            comprehend_query_number,
            input_pdf_for_review,
            annotate_max_pages,
            annotate_max_pages_bottom,
            prepared_pdf_state,
            images_pdf_state,
            review_file_df,
            page_sizes,
            duplication_file_path_outputs_list_state,
            in_duplicate_pages,
            latest_review_file_path,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            backup_review_state,
            task_textbox,
            input_review_files,
            vlm_model_name_textbox,
            vlm_total_input_tokens_number,
            vlm_total_output_tokens_number,
            llm_model_name_textbox,
            llm_total_input_tokens_number,
            llm_total_output_tokens_number,
        ],
        api_name="redact_doc",
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            output_file_list_state,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_doc_files,
        ],
        outputs=None,
    ).success(
        fn=update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            page_min,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # If a file has been completed, the function will continue onto the next document
    latest_file_completed_num.change(
        fn=choose_and_run_redactor,
        inputs=[
            in_doc_files,
            prepared_pdf_state,
            images_pdf_state,
            in_redact_entities,
            in_redact_comprehend_entities,
            in_redact_llm_entities,
            text_extract_method_radio,
            in_allow_list_state,
            in_deny_list_state,
            in_fully_redacted_list_state,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            output_file_list_state,
            log_files_output_list_state,
            second_loop_state,
            page_min,
            page_max,
            actual_time_taken_number,
            handwrite_signature_checkbox,
            textract_metadata_textbox,
            all_image_annotations_state,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            pdf_doc_state,
            current_loop_page_number,
            page_break_return,
            pii_identification_method_drop,
            comprehend_query_number,
            max_fuzzy_spelling_mistakes_num,
            match_fuzzy_whole_phrase_bool,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            annotate_max_pages,
            review_file_df,
            output_folder_textbox,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            only_extract_text_radio,
            duplication_file_path_outputs_list_state,
            latest_review_file_path,
            input_folder_textbox,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            local_ocr_method_radio,
            chosen_language_drop,
            input_review_files,
        ],
        outputs=[
            redaction_output_summary_textbox,
            output_file,
            output_file_list_state,
            latest_file_completed_num,
            log_files_output,
            log_files_output_list_state,
            actual_time_taken_number,
            textract_metadata_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            current_loop_page_number,
            page_break_return,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            comprehend_query_number,
            input_pdf_for_review,
            annotate_max_pages,
            annotate_max_pages_bottom,
            prepared_pdf_state,
            images_pdf_state,
            review_file_df,
            page_sizes,
            duplication_file_path_outputs_list_state,
            in_duplicate_pages,
            latest_review_file_path,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            backup_review_state,
            task_textbox,
            input_review_files,
            vlm_model_name_textbox,
            vlm_total_input_tokens_number,
            vlm_total_output_tokens_number,
            llm_model_name_textbox,
            llm_total_input_tokens_number,
            llm_total_output_tokens_number,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            output_file_list_state,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_doc_files,
        ],
        outputs=None,
    ).success(
        fn=update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            page_min,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        fn=check_for_existing_textract_file,
        inputs=[
            doc_file_name_no_extension_textbox,
            output_folder_textbox,
            handwrite_signature_checkbox,
        ],
        outputs=[textract_output_found_checkbox],
    ).success(
        fn=check_for_relevant_ocr_output_with_words,
        inputs=[
            doc_file_name_no_extension_textbox,
            text_extract_method_radio,
            output_folder_textbox,
        ],
        outputs=[relevant_ocr_output_with_words_found_checkbox],
    ).success(
        fn=reveal_feedback_buttons,
        outputs=[
            pdf_feedback_radio,
            pdf_further_details_text,
            pdf_submit_feedback_btn,
            pdf_feedback_title,
        ],
    ).success(
        fn=reset_aws_call_vars,
        outputs=[comprehend_query_number, textract_query_number],
    )

    # If the line level ocr results are changed by load in by user or by a new redaction task, replace the ocr results displayed in the table
    all_page_line_level_ocr_results_df_base.change(
        reset_ocr_base_dataframe,
        inputs=[all_page_line_level_ocr_results_df_base],
        outputs=[all_page_line_level_ocr_results_df],
    )
    all_page_line_level_ocr_results_with_words_df_base.change(
        reset_ocr_with_words_base_dataframe,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            page_entity_dropdown_redaction,
        ],
        outputs=[
            all_page_line_level_ocr_results_with_words_df,
            backup_all_page_line_level_ocr_results_with_words_df_base,
        ],
    )

    # Send whole document to Textract for text extraction
    send_document_to_textract_api_btn.click(
        analyse_document_with_textract_api,
        inputs=[
            prepared_pdf_state,
            s3_whole_document_textract_input_subfolder,
            s3_whole_document_textract_output_subfolder,
            textract_job_detail_df,
            s3_whole_document_textract_default_bucket,
            output_folder_textbox,
            handwrite_signature_checkbox,
            successful_textract_api_call_number,
            total_pdf_page_count,
        ],
        outputs=[
            job_output_textbox,
            job_id_textbox,
            job_type_dropdown,
            successful_textract_api_call_number,
            is_a_textract_api_call,
            textract_query_number,
            task_textbox,
        ],
        show_progress_on=[job_current_status],
    ).success(check_for_provided_job_id, inputs=[job_id_textbox]).success(
        poll_whole_document_textract_analysis_progress_and_download,
        inputs=[
            job_id_textbox,
            job_type_dropdown,
            s3_whole_document_textract_output_subfolder,
            doc_file_name_no_extension_textbox,
            textract_job_detail_df,
            s3_whole_document_textract_default_bucket,
            output_folder_textbox,
            s3_whole_document_textract_logs_subfolder,
            local_whole_document_textract_logs_subfolder,
        ],
        outputs=[
            textract_job_output_file,
            job_current_status,
            textract_job_detail_df,
            doc_file_name_no_extension_textbox,
        ],
        show_progress_on=[job_current_status],
    ).success(
        fn=check_for_existing_textract_file,
        inputs=[doc_file_name_no_extension_textbox, output_folder_textbox],
        outputs=[textract_output_found_checkbox],
        show_progress_on=[job_current_status],
    )

    check_state_of_textract_api_call_btn.click(
        check_for_provided_job_id,
        inputs=[job_id_textbox],
        show_progress_on=[job_current_status],
    ).success(
        poll_whole_document_textract_analysis_progress_and_download,
        inputs=[
            job_id_textbox,
            job_type_dropdown,
            s3_whole_document_textract_output_subfolder,
            doc_file_name_no_extension_textbox,
            textract_job_detail_df,
            s3_whole_document_textract_default_bucket,
            output_folder_textbox,
            s3_whole_document_textract_logs_subfolder,
            local_whole_document_textract_logs_subfolder,
        ],
        outputs=[
            textract_job_output_file,
            job_current_status,
            textract_job_detail_df,
            doc_file_name_no_extension_textbox,
        ],
        show_progress_on=[job_current_status],
    ).success(
        fn=check_for_existing_textract_file,
        inputs=[doc_file_name_no_extension_textbox, output_folder_textbox],
        outputs=[textract_output_found_checkbox],
        show_progress_on=[job_current_status],
    )

    textract_job_detail_df.select(
        df_select_callback_textract_api,
        inputs=[textract_output_found_checkbox],
        outputs=[job_id_textbox, job_type_dropdown, selected_job_id_row],
    )

    convert_textract_outputs_to_ocr_results.click(
        replace_existing_pdf_input_for_whole_document_outputs,
        inputs=[
            s3_whole_document_textract_input_subfolder,
            doc_file_name_no_extension_textbox,
            output_folder_textbox,
            s3_whole_document_textract_default_bucket,
            in_doc_files,
            input_folder_textbox,
        ],
        outputs=[
            in_doc_files,
            doc_file_name_no_extension_textbox,
            doc_file_name_with_extension_textbox,
            doc_full_file_name_textbox,
            doc_file_name_textbox_list,
            total_pdf_page_count,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=prepare_image_or_pdf,
        inputs=[
            in_doc_files,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            first_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool_false,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_page_line_level_ocr_results_df_base,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=check_for_existing_textract_file,
        inputs=[
            doc_file_name_no_extension_textbox,
            output_folder_textbox,
            handwrite_signature_checkbox,
        ],
        outputs=[textract_output_found_checkbox],
    ).success(
        fn=check_for_relevant_ocr_output_with_words,
        inputs=[
            doc_file_name_no_extension_textbox,
            text_extract_method_radio,
            output_folder_textbox,
        ],
        outputs=[relevant_ocr_output_with_words_found_checkbox],
    ).success(
        fn=check_textract_outputs_exist, inputs=[textract_output_found_checkbox]
    ).success(
        fn=reset_state_vars,
        outputs=[
            all_image_annotations_state,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            comprehend_query_number,
            textract_metadata_textbox,
            annotator,
            output_file_list_state,
            log_files_output_list_state,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            pdf_doc_state,
            duplication_file_path_outputs_list_state,
            redaction_output_summary_textbox,
            is_a_textract_api_call,
            textract_query_number,
            all_page_line_level_ocr_results_with_words,
            input_review_files,
        ],
    ).success(
        fn=choose_and_run_redactor,
        inputs=[
            in_doc_files,
            prepared_pdf_state,
            images_pdf_state,
            in_redact_entities,
            in_redact_comprehend_entities,
            in_redact_llm_entities,
            textract_only_method_drop,
            in_allow_list_state,
            in_deny_list_state,
            in_fully_redacted_list_state,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            output_file_list_state,
            log_files_output_list_state,
            first_loop_state,
            page_min,
            page_max,
            actual_time_taken_number,
            handwrite_signature_checkbox,
            textract_metadata_textbox,
            all_image_annotations_state,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            pdf_doc_state,
            current_loop_page_number,
            page_break_return,
            no_redaction_method_drop,
            comprehend_query_number,
            max_fuzzy_spelling_mistakes_num,
            match_fuzzy_whole_phrase_bool,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            annotate_max_pages,
            review_file_df,
            output_folder_textbox,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            only_extract_text_radio,
            duplication_file_path_outputs_list_state,
            latest_review_file_path,
            input_folder_textbox,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            local_ocr_method_radio,
            chosen_language_drop,
            input_review_files,
        ],
        outputs=[
            redaction_output_summary_textbox,
            output_file,
            output_file_list_state,
            latest_file_completed_num,
            log_files_output,
            log_files_output_list_state,
            actual_time_taken_number,
            textract_metadata_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            current_loop_page_number,
            page_break_return,
            all_page_line_level_ocr_results_df_base,
            all_decision_process_table_state,
            comprehend_query_number,
            input_pdf_for_review,
            annotate_max_pages,
            annotate_max_pages_bottom,
            prepared_pdf_state,
            images_pdf_state,
            review_file_df,
            page_sizes,
            duplication_file_path_outputs_list_state,
            in_duplicate_pages,
            latest_review_file_path,
            textract_query_number,
            latest_ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df_base,
            backup_review_state,
            task_textbox,
            input_review_files,
            vlm_model_name_textbox,
            vlm_total_input_tokens_number,
            vlm_total_output_tokens_number,
            llm_model_name_textbox,
            llm_total_input_tokens_number,
            llm_total_output_tokens_number,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            output_file_list_state,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_doc_files,
        ],
        outputs=None,
    ).success(
        fn=update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            page_min,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    go_to_review_redactions_tab_btn.click(
        fn=change_tab_to_review_redactions,
        inputs=None,
        outputs=tabs,
    )

    ###
    # REVIEW PDF REDACTIONS
    ###

    # Upload previous PDF for modifying redactions
    upload_pdf_for_review_btn.click(
        fn=reset_review_vars,
        inputs=None,
        outputs=[recogniser_entity_dataframe, recogniser_entity_dataframe_base],
    ).success(
        fn=get_input_file_names,
        inputs=[input_pdf_for_review],
        outputs=[
            doc_file_name_no_extension_textbox,
            doc_file_name_with_extension_textbox,
            doc_full_file_name_textbox,
            doc_file_name_textbox_list,
            total_pdf_page_count,
        ],
    ).success(
        fn=prepare_image_or_pdf,
        inputs=[
            input_pdf_for_review,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            second_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_page_line_level_ocr_results_df_base,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        api_name="prepare_doc",
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # Upload previous review CSV files for modifying redactions
    upload_review_files_btn.click(
        fn=prepare_image_or_pdf,
        inputs=[
            input_review_files,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            second_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_page_line_level_ocr_results_df_base,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        show_progress_on=[redaction_output_summary_textbox],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # Manual updates to review df
    review_file_df.input(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_entity_dataframe_row,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # Page number controls
    annotate_current_page.submit(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    annotation_last_page_button.click(
        fn=decrease_page,
        inputs=[annotate_current_page, all_image_annotations_state],
        outputs=[annotate_current_page, annotate_current_page_bottom],
        show_progress_on=[all_image_annotations_state],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    annotation_next_page_button.click(
        fn=increase_page,
        inputs=[annotate_current_page, all_image_annotations_state],
        outputs=[annotate_current_page, annotate_current_page_bottom],
        show_progress_on=[all_image_annotations_state],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    annotation_last_page_button_bottom.click(
        fn=decrease_page,
        inputs=[annotate_current_page, all_image_annotations_state],
        outputs=[annotate_current_page, annotate_current_page_bottom],
        show_progress_on=[all_image_annotations_state],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    annotation_next_page_button_bottom.click(
        fn=increase_page,
        inputs=[annotate_current_page, all_image_annotations_state],
        outputs=[annotate_current_page, annotate_current_page_bottom],
        show_progress_on=[all_image_annotations_state],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    annotate_current_page_bottom.submit(
        update_other_annotator_number_from_current,
        inputs=[annotate_current_page_bottom],
        outputs=[annotate_current_page],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_previous_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    # Apply page redactions
    annotation_button_apply.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        scroll_to_output=True,
        show_progress_on=[input_pdf_for_review],
    )

    # Save current page manual redactions
    update_current_page_redactions_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    ###
    # Review and exclude suggested redactions
    ###

    # Review table controls
    recogniser_entity_dropdown.select(
        update_entities_df_recogniser_entities,
        inputs=[
            recogniser_entity_dropdown,
            recogniser_entity_dataframe_base,
            page_entity_dropdown,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dataframe,
            text_entity_dropdown,
            page_entity_dropdown,
        ],
    )
    page_entity_dropdown.select(
        update_entities_df_page,
        inputs=[
            page_entity_dropdown,
            recogniser_entity_dataframe_base,
            recogniser_entity_dropdown,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dataframe,
            recogniser_entity_dropdown,
            text_entity_dropdown,
        ],
    )
    text_entity_dropdown.select(
        update_entities_df_text,
        inputs=[
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            recogniser_entity_dropdown,
            page_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dataframe,
            recogniser_entity_dropdown,
            page_entity_dropdown,
        ],
    )

    # Clicking on a cell in the recogniser entity dataframe will take you to that page, and also highlight the target redaction box in blue
    recogniser_entity_dataframe.select(
        df_select_callback_dataframe_row,
        inputs=[recogniser_entity_dataframe],
        outputs=[selected_entity_dataframe_row, selected_entity_dataframe_row_text],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_and_merge_current_page_annotations,
        inputs=[
            page_sizes,
            annotate_current_page,
            all_image_annotations_state,
            review_file_df,
        ],
        outputs=[review_file_df],
    ).success(
        update_selected_review_df_row_colour,
        inputs=[
            selected_entity_dataframe_row,
            review_file_df,
            selected_entity_id,
            selected_entity_colour,
        ],
        outputs=[review_file_df, selected_entity_id, selected_entity_colour],
    ).success(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_entity_dataframe_row,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        increase_bottom_page_count_based_on_top,
        inputs=[annotate_current_page],
        outputs=[annotate_current_page_bottom],
    )

    reset_dropdowns_btn.click(
        reset_dropdowns,
        inputs=[recogniser_entity_dataframe_base],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    ### Exclude current selection from annotator and outputs
    # Exclude only selected row
    exclude_selected_row_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_and_merge_current_page_annotations,
        inputs=[
            page_sizes,
            annotate_current_page,
            all_image_annotations_state,
            review_file_df,
        ],
        outputs=[review_file_df],
    ).success(
        exclude_selected_items_from_redaction,
        inputs=[
            review_file_df,
            selected_entity_dataframe_row,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
        ],
        outputs=[
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            recogniser_entity_dataframe_base,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown,
        ],
    )

    # Exclude all items with same text as selected row
    exclude_text_with_same_as_selected_row_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_and_merge_current_page_annotations,
        inputs=[
            page_sizes,
            annotate_current_page,
            all_image_annotations_state,
            review_file_df,
        ],
        outputs=[review_file_df],
    ).success(
        get_all_rows_with_same_text,
        inputs=[
            recogniser_entity_dataframe_base,
            selected_entity_dataframe_row_text,
        ],
        outputs=[recogniser_entity_dataframe_same_text],
    ).success(
        exclude_selected_items_from_redaction,
        inputs=[
            review_file_df,
            recogniser_entity_dataframe_same_text,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
        ],
        outputs=[
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            recogniser_entity_dataframe_base,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown,
        ],
    )

    # Exclude everything visible in table
    exclude_selected_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_and_merge_current_page_annotations,
        inputs=[
            page_sizes,
            annotate_current_page,
            all_image_annotations_state,
            review_file_df,
        ],
        outputs=[review_file_df],
    ).success(
        exclude_selected_items_from_redaction,
        inputs=[
            review_file_df,
            recogniser_entity_dataframe,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
        ],
        outputs=[
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            recogniser_entity_dataframe_base,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown,
        ],
    )

    # Undo last redaction exclusion action
    undo_last_removal_btn.click(
        undo_last_removal,
        inputs=[
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        ],
        outputs=[
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    ###
    # Add new redactions with table selection
    ###
    page_entity_dropdown_redaction.select(
        update_redact_choice_df_from_page_dropdown,
        inputs=[
            page_entity_dropdown_redaction,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        outputs=[all_page_line_level_ocr_results_with_words_df],
    )

    def run_search_with_regex_option(
        search_text, word_df, similarity_threshold, use_regex_flag
    ):
        """Wrapper function to call run_full_search_and_analysis with regex option"""
        return run_full_search_and_analysis(
            search_query_text=search_text,
            word_level_df_orig=word_df,
            similarity_threshold=similarity_threshold,
            combine_pages=False,
            min_word_count=1,
            min_consecutive_pages=1,
            greedy_match=True,
            remake_index=False,
            use_regex=use_regex_flag,
        )

    multi_word_search_text.submit(
        fn=run_search_with_regex_option,
        inputs=[
            multi_word_search_text,
            all_page_line_level_ocr_results_with_words_df_base,
            similarity_search_score_minimum,
            use_regex_search,
        ],
        outputs=[
            all_page_line_level_ocr_results_with_words_df,
            duplicate_files_out,
            full_duplicate_data_by_file,
        ],
    )

    multi_word_search_text_btn.click(
        fn=run_search_with_regex_option,
        inputs=[
            multi_word_search_text,
            all_page_line_level_ocr_results_with_words_df_base,
            similarity_search_score_minimum,
            use_regex_search,
        ],
        outputs=[
            all_page_line_level_ocr_results_with_words_df,
            duplicate_files_out,
            full_duplicate_data_by_file,
        ],
        api_name="word_level_ocr_text_search",
    )

    # Clicking on a cell in the redact items table will take you to that page
    all_page_line_level_ocr_results_with_words_df.select(
        df_select_callback_dataframe_row_ocr_with_words,
        inputs=[all_page_line_level_ocr_results_with_words_df],
        outputs=[
            selected_entity_dataframe_row_redact,
            selected_entity_dataframe_row_text_redact,
        ],
    ).success(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_and_merge_current_page_annotations,
        inputs=[
            page_sizes,
            annotate_current_page,
            all_image_annotations_state,
            review_file_df,
        ],
        outputs=[review_file_df],
    ).success(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_entity_dataframe_row_redact,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        increase_bottom_page_count_based_on_top,
        inputs=[annotate_current_page],
        outputs=[annotate_current_page_bottom],
    )

    # Reset dropdowns
    reset_dropdowns_btn_new.click(
        reset_dropdowns,
        inputs=[all_page_line_level_ocr_results_with_words_df_base],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown_redaction,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # Redact everything visible in table
    redact_selected_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        create_annotation_objects_from_filtered_ocr_results_with_words,
        inputs=[
            all_page_line_level_ocr_results_with_words_df,
            all_page_line_level_ocr_results_with_words_df_base,
            page_sizes,
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            new_redaction_text_label,
            colour_label,
            annotate_current_page,
        ],
        outputs=[
            all_image_annotations_state,
            backup_image_annotations_state,
            review_file_df,
            backup_review_state,
            recogniser_entity_dataframe,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            recogniser_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown_redaction,
        ],
    )

    # Reset redaction table following filtering
    reset_ocr_with_words_df_btn.click(
        reset_ocr_with_words_base_dataframe,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            page_entity_dropdown_redaction,
        ],
        outputs=[
            all_page_line_level_ocr_results_with_words_df,
            backup_all_page_line_level_ocr_results_with_words_df_base,
        ],
    )

    # Redact current selection
    redact_selected_row_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        create_annotation_objects_from_filtered_ocr_results_with_words,
        inputs=[
            selected_entity_dataframe_row_redact,
            all_page_line_level_ocr_results_with_words_df_base,
            page_sizes,
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            new_redaction_text_label,
            colour_label,
            annotate_current_page,
        ],
        outputs=[
            all_image_annotations_state,
            backup_image_annotations_state,
            review_file_df,
            backup_review_state,
            recogniser_entity_dataframe,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            recogniser_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown_redaction,
        ],
    )

    # Redact all items with same text as selected row
    redact_text_with_same_as_selected_row_btn.click(
        update_all_page_annotation_object_based_on_previous_page,
        inputs=[
            annotator,
            annotate_current_page,
            annotate_current_page,
            all_image_annotations_state,
            page_sizes,
        ],
        outputs=[
            all_image_annotations_state,
            annotate_previous_page,
            annotate_current_page_bottom,
        ],
    ).success(
        get_all_rows_with_same_text_redact,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            selected_entity_dataframe_row_text_redact,
        ],
        outputs=[to_redact_dataframe_same_text],
    ).success(
        create_annotation_objects_from_filtered_ocr_results_with_words,
        inputs=[
            to_redact_dataframe_same_text,
            all_page_line_level_ocr_results_with_words_df_base,
            page_sizes,
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
            new_redaction_text_label,
            colour_label,
            annotate_current_page,
        ],
        outputs=[
            all_image_annotations_state,
            backup_image_annotations_state,
            review_file_df,
            backup_review_state,
            recogniser_entity_dataframe,
            backup_recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    ).success(
        update_all_entity_df_dropdowns,
        inputs=[
            all_page_line_level_ocr_results_with_words_df_base,
            recogniser_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
        ],
        outputs=[
            recogniser_entity_dropdown,
            text_entity_dropdown,
            page_entity_dropdown_redaction,
        ],
    )

    # Undo last redaction action
    undo_last_redact_btn.click(
        undo_last_removal,
        inputs=[
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        ],
        outputs=[
            review_file_df,
            all_image_annotations_state,
            recogniser_entity_dataframe_base,
        ],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    ).success(
        apply_redactions_to_review_df_and_files,
        inputs=[
            annotator,
            doc_full_file_name_textbox,
            pdf_doc_state,
            all_image_annotations_state,
            annotate_current_page,
            review_file_df,
            output_folder_textbox,
            do_not_save_pdf_state,
            page_sizes,
        ],
        outputs=[
            pdf_doc_state,
            all_image_annotations_state,
            input_pdf_for_review,
            log_files_output,
            review_file_df,
        ],
        show_progress_on=[input_pdf_for_review],
    )

    ###
    # Review OCR text
    ###
    all_page_line_level_ocr_results_df.select(
        df_select_callback_ocr,
        inputs=[all_page_line_level_ocr_results_df],
        outputs=[annotate_current_page, selected_ocr_dataframe_row],
    ).success(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_ocr_dataframe_row,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        increase_bottom_page_count_based_on_top,
        inputs=[annotate_current_page],
        outputs=[annotate_current_page_bottom],
    )

    # Reset the OCR results filter
    reset_all_ocr_results_btn.click(
        reset_ocr_base_dataframe,
        inputs=[all_page_line_level_ocr_results_df_base],
        outputs=[all_page_line_level_ocr_results_df],
    )

    # Convert review file to xfdf Adobe format
    convert_review_file_to_adobe_btn.click(
        fn=get_input_file_names,
        inputs=[input_pdf_for_review],
        outputs=[
            doc_file_name_no_extension_textbox,
            doc_file_name_with_extension_textbox,
            doc_full_file_name_textbox,
            doc_file_name_textbox_list,
            total_pdf_page_count,
        ],
    ).success(
        fn=prepare_image_or_pdf,
        inputs=[
            input_pdf_for_review,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            second_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_line_level_ocr_results_df_placeholder,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        show_progress_on=[adobe_review_files_out],
    ).success(
        convert_df_to_xfdf,
        inputs=[
            input_pdf_for_review,
            pdf_doc_state,
            images_pdf_state,
            output_folder_textbox,
            document_cropboxes,
            page_sizes,
        ],
        outputs=[adobe_review_files_out],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            adobe_review_files_out,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            input_pdf_for_review,
        ],
        outputs=None,
    )

    # Convert xfdf Adobe file back to review_file.csv
    convert_adobe_to_review_file_btn.click(
        fn=get_input_file_names,
        inputs=[adobe_review_files_out],
        outputs=[
            doc_file_name_no_extension_textbox,
            doc_file_name_with_extension_textbox,
            doc_full_file_name_textbox,
            doc_file_name_textbox_list,
            total_pdf_page_count,
        ],
    ).success(
        fn=prepare_image_or_pdf,
        inputs=[
            adobe_review_files_out,
            text_extract_method_radio,
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
            latest_file_completed_num,
            redaction_output_summary_textbox,
            second_loop_state,
            annotate_max_pages,
            all_image_annotations_state,
            prepare_for_review_bool,
            in_fully_redacted_list_state,
            output_folder_textbox,
            input_folder_textbox,
            prepare_images_bool_false,
            page_sizes,
            pdf_doc_state,
            page_min,
            page_max,
        ],
        outputs=[
            redaction_output_summary_textbox,
            prepared_pdf_state,
            images_pdf_state,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pdf_doc_state,
            all_image_annotations_state,
            review_file_df,
            document_cropboxes,
            page_sizes,
            textract_output_found_checkbox,
            all_img_details_state,
            all_line_level_ocr_results_df_placeholder,
            relevant_ocr_output_with_words_found_checkbox,
            all_page_line_level_ocr_results_with_words_df_base,
        ],
        show_progress_on=[adobe_review_files_out],
    ).success(
        fn=convert_xfdf_to_dataframe,
        inputs=[
            adobe_review_files_out,
            pdf_doc_state,
            images_pdf_state,
            output_folder_textbox,
            input_folder_textbox,
        ],
        outputs=[input_pdf_for_review],
        scroll_to_output=True,
    )

    ###
    # WORD/TABULAR DATA REDACTION
    ###
    in_data_files.upload(
        fn=put_columns_in_df,
        inputs=[in_data_files],
        outputs=[in_colnames, in_excel_sheets],
    ).success(
        fn=get_input_file_names,
        inputs=[in_data_files],
        outputs=[
            data_file_name_no_extension_textbox,
            data_file_name_with_extension_textbox,
            data_full_file_name_textbox,
            data_file_name_textbox_list,
            total_pdf_page_count,
        ],
    )

    tabular_data_redact_btn.click(
        reset_data_vars,
        outputs=[
            actual_time_taken_number,
            log_files_output_list_state,
            comprehend_query_number,
        ],
    ).success(
        fn=anonymise_files_with_open_text,
        inputs=[
            in_data_files,
            in_text,
            anon_strategy,
            in_colnames,
            in_redact_entities,
            in_allow_list_state,
            text_tabular_files_done,
            text_output_summary,
            text_output_file_list_state,
            log_files_output_list_state,
            in_excel_sheets,
            first_loop_state,
            output_folder_textbox,
            in_deny_list_state,
            max_fuzzy_spelling_mistakes_num,
            pii_identification_method_drop_tabular,
            in_redact_comprehend_entities,
            comprehend_query_number,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            actual_time_taken_number,
            do_initial_clean,
            chosen_language_drop,
        ],
        outputs=[
            text_output_summary,
            text_output_file,
            text_output_file_list_state,
            text_tabular_files_done,
            log_files_output,
            log_files_output_list_state,
            actual_time_taken_number,
            comprehend_query_number,
        ],
        api_name="redact_data",
        show_progress_on=[text_output_summary],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            text_output_file_list_state,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )

    # If the output file count text box changes, keep going with redacting each data file until done
    text_tabular_files_done.change(
        fn=anonymise_files_with_open_text,
        inputs=[
            in_data_files,
            in_text,
            anon_strategy,
            in_colnames,
            in_redact_entities,
            in_allow_list_state,
            text_tabular_files_done,
            text_output_summary,
            text_output_file_list_state,
            log_files_output_list_state,
            in_excel_sheets,
            second_loop_state,
            output_folder_textbox,
            in_deny_list_state,
            max_fuzzy_spelling_mistakes_num,
            pii_identification_method_drop_tabular,
            in_redact_comprehend_entities,
            comprehend_query_number,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            actual_time_taken_number,
            do_initial_clean,
            chosen_language_drop,
        ],
        outputs=[
            text_output_summary,
            text_output_file,
            text_output_file_list_state,
            text_tabular_files_done,
            log_files_output,
            log_files_output_list_state,
            actual_time_taken_number,
            comprehend_query_number,
        ],
        show_progress_on=[text_output_summary],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            text_output_file_list_state,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    ).success(
        fn=reveal_feedback_buttons,
        outputs=[
            data_feedback_radio,
            data_further_details_text,
            data_submit_feedback_btn,
            data_feedback_title,
        ],
    )

    ###
    # IDENTIFY DUPLICATE PAGES
    ###

    find_duplicate_pages_btn.click(
        fn=run_duplicate_analysis,
        inputs=[
            in_duplicate_pages,
            duplicate_threshold_input,
            min_word_count_input,
            min_consecutive_pages_input,
            greedy_match_input,
            combine_page_text_for_duplicates_bool,
            output_folder_textbox,
        ],
        outputs=[
            results_df_preview,
            duplicate_files_out,
            full_duplicate_data_by_file,
            actual_time_taken_number,
            task_textbox,
        ],
        show_progress_on=[results_df_preview],
    ).success(
        fn=export_outputs_to_s3,
        # duplicate_files_out returns a single file path; export helper will normalise it
        inputs=[
            duplicate_files_out,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_duplicate_pages,
        ],
        outputs=None,
    )

    # full_duplicated_data_df,
    results_df_preview.select(
        fn=handle_selection_and_preview,
        inputs=[results_df_preview, full_duplicate_data_by_file],
        outputs=[
            selected_duplicate_data_row_index,
            page1_text_preview,
            page2_text_preview,
        ],
    )

    # When the user clicks the "Exclude" button
    exclude_match_btn.click(
        fn=exclude_match,
        inputs=[results_df_preview, selected_duplicate_data_row_index],
        outputs=[
            results_df_preview,
            duplicate_files_out,
            page1_text_preview,
            page2_text_preview,
        ],
    )

    apply_match_btn.click(
        fn=create_annotation_objects_from_duplicates,
        inputs=[
            results_df_preview,
            all_page_line_level_ocr_results_df_base,
            page_sizes,
            combine_page_text_for_duplicates_bool,
        ],
        outputs=[new_duplicate_search_annotation_object],
    ).success(
        fn=apply_whole_page_redactions_from_list,
        inputs=[
            in_fully_redacted_list_state,
            doc_file_name_with_extension_textbox,
            review_file_df,
            duplicate_files_out,
            pdf_doc_state,
            page_sizes,
            all_image_annotations_state,
            combine_page_text_for_duplicates_bool,
            new_duplicate_search_annotation_object,
        ],
        outputs=[review_file_df, all_image_annotations_state],
    ).success(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_entity_dataframe_row,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    ###
    # TABULAR DUPLICATE DETECTION
    ###

    # Event handlers
    in_tabular_duplicate_files.upload(
        fn=put_columns_in_df,
        inputs=[in_tabular_duplicate_files],
        outputs=[tabular_text_columns, in_excel_tabular_sheets],
    )

    find_tabular_duplicates_btn.click(
        fn=run_tabular_duplicate_detection,
        inputs=[
            in_tabular_duplicate_files,
            tabular_duplicate_threshold,
            tabular_min_word_count,
            tabular_text_columns,
            output_folder_textbox,
            do_initial_clean_dup,
            in_excel_tabular_sheets,
            remove_duplicate_rows,
        ],
        outputs=[
            tabular_results_df,
            tabular_cleaned_file,
            tabular_file_to_clean,
            actual_time_taken_number,
            task_textbox,
        ],
        api_name="tabular_clean_duplicates",
        show_progress_on=[tabular_results_df],
    )

    tabular_results_df.select(
        fn=handle_tabular_row_selection,
        inputs=[tabular_results_df],
        outputs=[
            tabular_selected_row_index,
            tabular_text1_preview,
            tabular_text2_preview,
        ],
    )

    clean_duplicates_btn.click(
        fn=clean_tabular_duplicates,
        inputs=[
            tabular_file_to_clean,
            tabular_results_df,
            output_folder_textbox,
            in_excel_tabular_sheets,
        ],
        outputs=[tabular_cleaned_file],
    )

    ###
    # SETTINGS PAGE INPUT / OUTPUT
    ###
    # If a custom allow/deny/duplicate page list is uploaded
    in_allow_list.change(
        fn=custom_regex_load,
        inputs=[in_allow_list],
        outputs=[in_allow_list_text, in_allow_list_state],
    )
    in_deny_list.change(
        fn=custom_regex_load,
        inputs=[in_deny_list, in_deny_list_text_in],
        outputs=[in_deny_list_text, in_deny_list_state],
    )
    in_fully_redacted_list.change(
        fn=custom_regex_load,
        inputs=[in_fully_redacted_list, in_fully_redacted_text_in],
        outputs=[in_fully_redacted_list_text, in_fully_redacted_list_state],
    )

    # Apply whole page redactions from the provided whole page redaction csv file upload/list of specific page numbers given by user
    apply_fully_redacted_list_btn.click(
        fn=apply_whole_page_redactions_from_list,
        inputs=[
            in_fully_redacted_list_state,
            doc_file_name_with_extension_textbox,
            review_file_df,
            duplicate_files_out,
            pdf_doc_state,
            page_sizes,
            all_image_annotations_state,
        ],
        outputs=[review_file_df, all_image_annotations_state],
    ).success(
        update_annotator_page_from_review_df,
        inputs=[
            review_file_df,
            images_pdf_state,
            page_sizes,
            all_image_annotations_state,
            annotator,
            selected_entity_dataframe_row,
            input_folder_textbox,
            doc_full_file_name_textbox,
        ],
        outputs=[
            annotator,
            all_image_annotations_state,
            annotate_current_page,
            page_sizes,
            review_file_df,
            annotate_previous_page,
        ],
        show_progress_on=[annotator],
    ).success(
        update_annotator_object_and_filter_df,
        inputs=[
            all_image_annotations_state,
            annotate_current_page,
            recogniser_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            text_entity_dropdown,
            recogniser_entity_dataframe_base,
            annotator_zoom_number,
            review_file_df,
            page_sizes,
            doc_full_file_name_textbox,
            input_folder_textbox,
        ],
        outputs=[
            annotator,
            annotate_current_page,
            annotate_current_page_bottom,
            annotate_previous_page,
            recogniser_entity_dropdown,
            recogniser_entity_dataframe,
            recogniser_entity_dataframe_base,
            text_entity_dropdown,
            page_entity_dropdown,
            page_entity_dropdown_redaction,
            page_sizes,
            all_image_annotations_state,
        ],
        show_progress_on=[annotator],
    )

    # Merge multiple review csv files together
    merge_multiple_review_files_btn.click(
        fn=merge_csv_files,
        inputs=multiple_review_files_in_out,
        outputs=multiple_review_files_in_out,
    )

    # Need to momentarilly change the root directory of the file explorer to another non-sensitive folder when the button is clicked to get it to update (workaround))
    all_output_files_btn.click(
        fn=lambda: gr.FileExplorer(root_dir=FEEDBACK_LOGS_FOLDER),
        inputs=None,
        outputs=all_output_files,
    ).success(
        fn=load_all_output_files,
        inputs=output_folder_textbox,
        outputs=all_output_files,
    )

    all_output_files.change(
        fn=all_outputs_file_download_fn,
        inputs=all_output_files,
        outputs=all_outputs_file_download,
    )

    # Language selection dropdown
    chosen_language_full_name_drop.select(
        update_language_dropdown,
        inputs=[chosen_language_full_name_drop],
        outputs=[chosen_language_drop],
    )

    ###
    # APP LOAD AND LOGGING
    ###

    # Get connection details on app load

    if SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS:
        blocks.load(
            get_connection_params,
            inputs=[
                output_folder_textbox,
                input_folder_textbox,
                session_output_folder_textbox,
                s3_output_folder_state,
                s3_whole_document_textract_input_subfolder,
                s3_whole_document_textract_output_subfolder,
                s3_whole_document_textract_logs_subfolder,
                local_whole_document_textract_logs_subfolder,
            ],
            outputs=[
                session_hash_state,
                output_folder_textbox,
                session_hash_textbox,
                input_folder_textbox,
                s3_whole_document_textract_input_subfolder,
                s3_whole_document_textract_output_subfolder,
                s3_whole_document_textract_logs_subfolder,
                local_whole_document_textract_logs_subfolder,
                s3_output_folder_state,
            ],
        ).success(
            load_in_textract_job_details,
            inputs=[
                load_s3_whole_document_textract_logs_bool,
                s3_whole_document_textract_logs_subfolder,
                local_whole_document_textract_logs_subfolder,
            ],
            outputs=[textract_job_detail_df],
        ).success(
            fn=load_all_output_files,
            inputs=output_folder_textbox,
            outputs=all_output_files,
        )

    else:
        blocks.load(
            get_connection_params,
            inputs=[
                output_folder_textbox,
                input_folder_textbox,
                session_output_folder_textbox,
                s3_output_folder_state,
                s3_whole_document_textract_input_subfolder,
                s3_whole_document_textract_output_subfolder,
                s3_whole_document_textract_logs_subfolder,
                local_whole_document_textract_logs_subfolder,
            ],
            outputs=[
                session_hash_state,
                output_folder_textbox,
                session_hash_textbox,
                input_folder_textbox,
                s3_whole_document_textract_input_subfolder,
                s3_whole_document_textract_output_subfolder,
                s3_whole_document_textract_logs_subfolder,
                local_whole_document_textract_logs_subfolder,
                s3_output_folder_state,
            ],
        ).success(
            fn=load_all_output_files,
            inputs=output_folder_textbox,
            outputs=all_output_files,
        )

    # If relevant environment variable is set, load in the default allow list file from S3 or locally. Even when setting S3 path, need to local path to give a download location
    if GET_DEFAULT_ALLOW_LIST and (ALLOW_LIST_PATH or S3_ALLOW_LIST_PATH):
        if (
            not os.path.exists(ALLOW_LIST_PATH)
            and S3_ALLOW_LIST_PATH
            and RUN_AWS_FUNCTIONS
        ):
            print("Downloading allow list from S3")
            blocks.load(
                download_file_from_s3,
                inputs=[
                    s3_default_bucket,
                    s3_default_allow_list_file,
                    default_allow_list_output_folder_location,
                ],
            ).success(
                load_in_default_allow_list,
                inputs=[default_allow_list_output_folder_location],
                outputs=[in_allow_list],
            )
            print("Successfully loaded allow list from S3")
        elif os.path.exists(ALLOW_LIST_PATH):
            print(
                "Loading allow list from default allow list output path location:",
                ALLOW_LIST_PATH,
            )
            blocks.load(
                load_in_default_allow_list,
                inputs=[default_allow_list_output_folder_location],
                outputs=[in_allow_list],
            )
        else:
            print("Could not load in default allow list")

    # If relevant environment variable is set, load in the default cost code file from S3 or locally
    if GET_COST_CODES and (COST_CODES_PATH or S3_COST_CODES_PATH):
        if (
            not os.path.exists(COST_CODES_PATH)
            and S3_COST_CODES_PATH
            and RUN_AWS_FUNCTIONS
        ):
            print("Downloading cost codes from S3")
            blocks.load(
                download_file_from_s3,
                inputs=[
                    s3_default_bucket,
                    s3_default_cost_codes_file,
                    default_cost_codes_output_folder_location,
                ],
            ).success(
                load_in_default_cost_codes,
                inputs=[
                    default_cost_codes_output_folder_location,
                    default_cost_code_textbox,
                ],
                outputs=[
                    cost_code_dataframe,
                    cost_code_dataframe_base,
                    cost_code_choice_drop,
                ],
            )
            print("Successfully loaded cost codes from S3")
        elif os.path.exists(COST_CODES_PATH):
            print(
                "Loading cost codes from default cost codes path location:",
                COST_CODES_PATH,
            )
            blocks.load(
                load_in_default_cost_codes,
                inputs=[
                    default_cost_codes_output_folder_location,
                    default_cost_code_textbox,
                ],
                outputs=[
                    cost_code_dataframe,
                    cost_code_dataframe_base,
                    cost_code_choice_drop,
                ],
            )
        else:
            print("Could not load in cost code data")

    ###
    # LOGGING
    ###

    ### ACCESS LOGS
    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)

    access_callback.setup([session_hash_textbox, host_name_textbox], ACCESS_LOGS_FOLDER)
    session_hash_textbox.change(
        lambda *args: access_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS,
            replacement_headers=CSV_ACCESS_LOG_HEADERS,
        ),
        [session_hash_textbox, host_name_textbox],
        outputs=[flag_value_placeholder],
        preprocess=False,
    ).success(
        fn=upload_log_file_to_s3,
        inputs=[access_logs_state, access_s3_logs_loc_state],
        outputs=[s3_logs_output_textbox],
    )

    ### FEEDBACK LOGS
    pdf_callback = CSVLogger_custom(dataset_file_name=FEEDBACK_LOG_FILE_NAME)
    data_callback = CSVLogger_custom(dataset_file_name=FEEDBACK_LOG_FILE_NAME)

    if DISPLAY_FILE_NAMES_IN_LOGS:
        # User submitted feedback for pdf redactions
        pdf_callback.setup(
            [
                pdf_feedback_radio,
                pdf_further_details_text,
                doc_file_name_no_extension_textbox,
            ],
            FEEDBACK_LOGS_FOLDER,
        )
        pdf_submit_feedback_btn.click(
            lambda *args: pdf_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
                replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
            ),
            [
                pdf_feedback_radio,
                pdf_further_details_text,
                doc_file_name_no_extension_textbox,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[feedback_logs_state, feedback_s3_logs_loc_state],
            outputs=[pdf_further_details_text],
        )

        # User submitted feedback for data redactions
        data_callback.setup(
            [
                data_feedback_radio,
                data_further_details_text,
                data_file_name_with_extension_textbox,
            ],
            FEEDBACK_LOGS_FOLDER,
        )
        data_submit_feedback_btn.click(
            lambda *args: data_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
                replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
            ),
            [
                data_feedback_radio,
                data_further_details_text,
                data_file_name_with_extension_textbox,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[feedback_logs_state, feedback_s3_logs_loc_state],
            outputs=[data_further_details_text],
        )
    else:
        # User submitted feedback for pdf redactions
        pdf_callback.setup(
            [
                pdf_feedback_radio,
                pdf_further_details_text,
                doc_file_name_no_extension_textbox,
            ],
            FEEDBACK_LOGS_FOLDER,
        )
        pdf_submit_feedback_btn.click(
            lambda *args: pdf_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
                replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
            ),
            [
                pdf_feedback_radio,
                pdf_further_details_text,
                placeholder_doc_file_name_no_extension_textbox_for_logs,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[feedback_logs_state, feedback_s3_logs_loc_state],
            outputs=[pdf_further_details_text],
        )

        # User submitted feedback for data redactions
        data_callback.setup(
            [
                data_feedback_radio,
                data_further_details_text,
                data_file_name_with_extension_textbox,
            ],
            FEEDBACK_LOGS_FOLDER,
        )
        data_submit_feedback_btn.click(
            lambda *args: data_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
                replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
            ),
            [
                data_feedback_radio,
                data_further_details_text,
                placeholder_data_file_name_no_extension_textbox_for_logs,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[feedback_logs_state, feedback_s3_logs_loc_state],
            outputs=[data_further_details_text],
        )

    ### USAGE LOGS
    # Log processing usage - time taken for redaction queries, and also logs for queries to Textract/Comprehend
    usage_callback = CSVLogger_custom(dataset_file_name=USAGE_LOG_FILE_NAME)

    if DISPLAY_FILE_NAMES_IN_LOGS:
        usage_callback.setup(
            [
                session_hash_textbox,
                doc_file_name_no_extension_textbox,
                data_file_name_with_extension_textbox,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            USAGE_LOGS_FOLDER,
        )

        latest_file_completed_num.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                doc_file_name_no_extension_textbox,
                data_file_name_with_extension_textbox,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
            api_name="usage_logs",
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        text_tabular_files_done.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                doc_file_name_no_extension_textbox,
                data_file_name_with_extension_textbox,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        successful_textract_api_call_number.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                doc_file_name_no_extension_textbox,
                data_file_name_with_extension_textbox,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        # Deduplication usage logging
        duplicate_files_out.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                blank_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        tabular_results_df.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                blank_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )
    else:
        usage_callback.setup(
            [
                session_hash_textbox,
                blank_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                total_pdf_page_count,
                actual_time_taken_number,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            USAGE_LOGS_FOLDER,
        )

        latest_file_completed_num.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                placeholder_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                actual_time_taken_number,
                total_pdf_page_count,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        text_tabular_files_done.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                blank_doc_file_name_no_extension_textbox_for_logs,
                placeholder_data_file_name_no_extension_textbox_for_logs,
                actual_time_taken_number,
                total_pdf_page_count,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        successful_textract_api_call_number.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                placeholder_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                actual_time_taken_number,
                total_pdf_page_count,
                textract_query_number,
                pii_identification_method_drop,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        # Deduplication usage logging (when file names not displayed)
        duplicate_files_out.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                placeholder_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                total_pdf_page_count,
                actual_time_taken_number,
                total_pdf_page_count,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

        tabular_results_df.change(
            lambda *args: usage_callback.flag(
                list(args),
                save_to_csv=SAVE_LOGS_TO_CSV,
                save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
                dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
                replacement_headers=CSV_USAGE_LOG_HEADERS,
            ),
            [
                session_hash_textbox,
                placeholder_doc_file_name_no_extension_textbox_for_logs,
                blank_data_file_name_no_extension_textbox_for_logs,
                total_pdf_page_count,
                actual_time_taken_number,
                total_pdf_page_count,
                textract_query_number,
                pii_identification_method_drop_tabular,
                comprehend_query_number,
                cost_code_choice_drop,
                handwrite_signature_checkbox,
                host_name_textbox,
                text_extract_method_radio,
                is_a_textract_api_call,
                task_textbox,
                vlm_model_name_textbox,
                vlm_total_input_tokens_number,
                vlm_total_output_tokens_number,
                llm_model_name_textbox,
                llm_total_input_tokens_number,
                llm_total_output_tokens_number,
            ],
            outputs=[flag_value_placeholder],
            preprocess=False,
        ).success(
            fn=upload_log_file_to_s3,
            inputs=[usage_logs_state, usage_s3_logs_loc_state],
            outputs=[s3_logs_output_textbox],
        )

    blocks.queue(
        max_size=int(MAX_QUEUE_SIZE),
        default_concurrency_limit=int(DEFAULT_CONCURRENCY_LIMIT),
    )

    if not RUN_DIRECT_MODE:
        # If running through command line with uvicorn
        if RUN_FASTAPI:
            if ALLOWED_ORIGINS:
                print(f"CORS enabled. Allowing origins: {ALLOWED_ORIGINS}")
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=ALLOWED_ORIGINS,  # The list of allowed origins
                    allow_credentials=True,  # Allow cookies to be included in cross-origin requests
                    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
                    allow_headers=["*"],  # Allow all headers
                )

            if ALLOWED_HOSTS:
                app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

            @app.get("/health", status_code=status.HTTP_200_OK)
            def health_check():
                """Simple health check endpoint."""
                return {"status": "ok"}

            app = gr.mount_gradio_app(
                app,
                blocks,
                # theme=gr.themes.Default(primary_hue="blue"),
                # head=head_html,
                # css=css,
                show_error=True,
                auth=authenticate_user if COGNITO_AUTH else None,
                max_file_size=MAX_FILE_SIZE,
                path="",
                favicon_path=Path(FAVICON_PATH),
                mcp_server=RUN_MCP_SERVER,
            )

            # Example command to run in uvicorn (in python): uvicorn.run("app:app", host=GRADIO_SERVER_NAME, port=GRADIO_SERVER_PORT)
            # In command line something like: uvicorn app:app --host=0.0.0.0 --port=7860

        else:
            if __name__ == "__main__":
                if COGNITO_AUTH:
                    blocks.launch(
                        # theme=gr.themes.Default(primary_hue="blue"),
                        # head=head_html,
                        # css=css,
                        show_error=True,
                        inbrowser=True,
                        auth=authenticate_user,
                        max_file_size=MAX_FILE_SIZE,
                        server_name=GRADIO_SERVER_NAME,
                        server_port=GRADIO_SERVER_PORT,
                        root_path=ROOT_PATH,
                        favicon_path=Path(FAVICON_PATH),
                        mcp_server=RUN_MCP_SERVER,
                    )
                else:
                    blocks.launch(
                        # theme=gr.themes.Default(primary_hue="blue"),
                        # head=head_html,
                        # css=css,
                        show_error=True,
                        inbrowser=True,
                        max_file_size=MAX_FILE_SIZE,
                        server_name=GRADIO_SERVER_NAME,
                        server_port=GRADIO_SERVER_PORT,
                        root_path=ROOT_PATH,
                        favicon_path=Path(FAVICON_PATH),
                        mcp_server=RUN_MCP_SERVER,
                    )

    else:
        if __name__ == "__main__":
            from cli_redact import main

            # Validate required direct mode configuration
            if not DIRECT_MODE_INPUT_FILE:
                print(
                    "Error: DIRECT_MODE_INPUT_FILE environment variable must be set for direct mode."
                )
                print(
                    "Please set DIRECT_MODE_INPUT_FILE to the path of your input file."
                )
                exit(1)

            # Prepare direct mode arguments based on environment variables
            direct_mode_args = {
                # Task Selection
                "task": DIRECT_MODE_TASK,
                # General Arguments (apply to all file types)
                "input_file": DIRECT_MODE_INPUT_FILE,
                "output_dir": DIRECT_MODE_OUTPUT_DIR,
                "input_dir": INPUT_FOLDER,
                "language": DIRECT_MODE_LANGUAGE,
                "allow_list": ALLOW_LIST_PATH,
                "pii_detector": DIRECT_MODE_PII_DETECTOR,
                "username": DIRECT_MODE_DEFAULT_USER,
                "save_to_user_folders": SESSION_OUTPUT_FOLDER,
                "local_redact_entities": CHOSEN_REDACT_ENTITIES,
                "aws_redact_entities": CHOSEN_COMPREHEND_ENTITIES,
                "aws_access_key": AWS_ACCESS_KEY,
                "aws_secret_key": AWS_SECRET_KEY,
                "cost_code": DEFAULT_COST_CODE,
                "aws_region": AWS_REGION,
                "s3_bucket": DOCUMENT_REDACTION_BUCKET,
                "do_initial_clean": DO_INITIAL_TABULAR_DATA_CLEAN,
                "save_logs_to_csv": SAVE_LOGS_TO_CSV,
                "save_logs_to_dynamodb": SAVE_LOGS_TO_DYNAMODB,
                "display_file_names_in_logs": DISPLAY_FILE_NAMES_IN_LOGS,
                "upload_logs_to_s3": RUN_AWS_FUNCTIONS,
                "s3_logs_prefix": S3_USAGE_LOGS_FOLDER,
                "feedback_logs_folder": FEEDBACK_LOGS_FOLDER,
                "access_logs_folder": ACCESS_LOGS_FOLDER,
                "usage_logs_folder": USAGE_LOGS_FOLDER,
                "paddle_model_path": PADDLE_MODEL_PATH,
                "spacy_model_path": SPACY_MODEL_PATH,
                # PDF/Image Redaction Arguments
                "ocr_method": DIRECT_MODE_OCR_METHOD,
                "page_min": DIRECT_MODE_PAGE_MIN,
                "page_max": DIRECT_MODE_PAGE_MAX,
                "images_dpi": DIRECT_MODE_IMAGES_DPI,
                "chosen_local_ocr_model": DIRECT_MODE_CHOSEN_LOCAL_OCR_MODEL,
                "preprocess_local_ocr_images": DIRECT_MODE_PREPROCESS_LOCAL_OCR_IMAGES,
                "compress_redacted_pdf": DIRECT_MODE_COMPRESS_REDACTED_PDF,
                "return_pdf_end_of_redaction": DIRECT_MODE_RETURN_PDF_END_OF_REDACTION,
                "deny_list_file": DENY_LIST_PATH,
                "allow_list_file": ALLOW_LIST_PATH,
                "redact_whole_page_file": WHOLE_PAGE_REDACTION_LIST_PATH,
                "handwrite_signature_extraction": DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX,
                "extract_forms": DIRECT_MODE_EXTRACT_FORMS,
                "extract_tables": DIRECT_MODE_EXTRACT_TABLES,
                "extract_layout": DIRECT_MODE_EXTRACT_LAYOUT,
                "extract_signatures": DIRECT_MODE_EXTRACT_SIGNATURES,
                "match_fuzzy_whole_phrase_bool": DIRECT_MODE_MATCH_FUZZY_WHOLE_PHRASE_BOOL,
                # VLM OCR Arguments
                "vlm_model_choice": CLOUD_VLM_MODEL_CHOICE,
                "inference_server_vlm_model": DEFAULT_INFERENCE_SERVER_VLM_MODEL,
                "inference_server_api_url": INFERENCE_SERVER_API_URL,
                "gemini_api_key": GEMINI_API_KEY,
                "azure_openai_api_key": AZURE_OPENAI_API_KEY,
                "azure_openai_endpoint": AZURE_OPENAI_INFERENCE_ENDPOINT,
                # LLM PII Detection Arguments
                # Note: The actual model used is determined by pii_identification_method in the downstream code
                # This is just the default - it will be overridden based on the selected PII method
                "llm_model_choice": CLOUD_LLM_PII_MODEL_CHOICE,
                "llm_inference_method": CHOSEN_LLM_PII_INFERENCE_METHOD,
                "inference_server_pii_model": DEFAULT_INFERENCE_SERVER_PII_MODEL,
                "llm_temperature": LLM_PII_TEMPERATURE,
                "llm_max_tokens": LLM_PII_MAX_TOKENS,
                "custom_llm_instructions": "",  # Can be set via environment variable if needed
                # Word/Tabular Anonymisation Arguments
                "anon_strategy": DIRECT_MODE_ANON_STRATEGY,
                "text_columns": DEFAULT_TEXT_COLUMNS,
                "excel_sheets": DEFAULT_EXCEL_SHEETS,
                "fuzzy_mistakes": DIRECT_MODE_FUZZY_MISTAKES,
                # Duplicate Detection Arguments
                "duplicate_type": DIRECT_MODE_DUPLICATE_TYPE,
                "similarity_threshold": DIRECT_MODE_SIMILARITY_THRESHOLD,
                "min_word_count": DIRECT_MODE_MIN_WORD_COUNT,
                "min_consecutive_pages": DIRECT_MODE_MIN_CONSECUTIVE_PAGES,
                "greedy_match": DIRECT_MODE_GREEDY_MATCH,
                "combine_pages": DIRECT_MODE_COMBINE_PAGES,
                "remove_duplicate_rows": DIRECT_MODE_REMOVE_DUPLICATE_ROWS,
                # Textract Batch Operations Arguments
                "textract_action": DIRECT_MODE_TEXTRACT_ACTION,
                "job_id": DIRECT_MODE_JOB_ID,
                "textract_bucket": TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
                "textract_input_prefix": TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
                "textract_output_prefix": TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
                "s3_textract_document_logs_subfolder": TEXTRACT_JOBS_S3_LOC,
                "local_textract_document_logs_subfolder": TEXTRACT_JOBS_LOCAL_LOC,
                "poll_interval": 30,
                "max_poll_attempts": 120,
                # Additional arguments
                "search_query": DEFAULT_SEARCH_QUERY,
            }

            print(f"Running in direct mode with task: {DIRECT_MODE_TASK}")
            print(f"Input file: {DIRECT_MODE_INPUT_FILE}")
            print(f"Output directory: {DIRECT_MODE_OUTPUT_DIR}")

            if DIRECT_MODE_TASK == "deduplicate":
                print(f"Duplicate type: {DIRECT_MODE_DUPLICATE_TYPE}")
                print(f"Similarity threshold: {DEFAULT_DUPLICATE_DETECTION_THRESHOLD}")
                print(f"Min word count: {DEFAULT_MIN_WORD_COUNT}")
                if DEFAULT_SEARCH_QUERY:
                    print(f"Search query: {DEFAULT_SEARCH_QUERY}")
                if DEFAULT_TEXT_COLUMNS:
                    print(f"Text columns: {DEFAULT_TEXT_COLUMNS}")
                print(f"Remove duplicate rows: {REMOVE_DUPLICATE_ROWS}")

            # Combine extraction options
            extraction_options = (
                list(direct_mode_args["handwrite_signature_extraction"])
                if direct_mode_args["handwrite_signature_extraction"]
                else list()
            )
            if direct_mode_args["extract_forms"]:
                extraction_options.append("Extract forms")
            if direct_mode_args["extract_tables"]:
                extraction_options.append("Extract tables")
            if direct_mode_args["extract_layout"]:
                extraction_options.append("Extract layout")
            direct_mode_args["handwrite_signature_extraction"] = extraction_options

            # Run the CLI main function with direct mode arguments
            main(direct_mode_args=direct_mode_args)
