import os
import logging
import pandas as pd
import gradio as gr
from gradio_image_annotation import image_annotator

from tools.config import OUTPUT_FOLDER, INPUT_FOLDER, RUN_DIRECT_MODE, MAX_QUEUE_SIZE, DEFAULT_CONCURRENCY_LIMIT, MAX_FILE_SIZE, GRADIO_SERVER_PORT, ROOT_PATH, GET_DEFAULT_ALLOW_LIST, ALLOW_LIST_PATH, S3_ALLOW_LIST_PATH, FEEDBACK_LOGS_FOLDER, ACCESS_LOGS_FOLDER, USAGE_LOGS_FOLDER, TESSERACT_FOLDER, POPPLER_FOLDER, REDACTION_LANGUAGE, GET_COST_CODES, COST_CODES_PATH, S3_COST_CODES_PATH, ENFORCE_COST_CODES, DISPLAY_FILE_NAMES_IN_LOGS, SHOW_COSTS, RUN_AWS_FUNCTIONS, DOCUMENT_REDACTION_BUCKET, SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS, TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET, TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER, TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER, SESSION_OUTPUT_FOLDER, LOAD_PREVIOUS_TEXTRACT_JOBS_S3, TEXTRACT_JOBS_S3_LOC, TEXTRACT_JOBS_LOCAL_LOC, HOST_NAME, DEFAULT_COST_CODE, OUTPUT_COST_CODES_PATH, OUTPUT_ALLOW_LIST_PATH, COGNITO_AUTH, SAVE_LOGS_TO_CSV, SAVE_LOGS_TO_DYNAMODB, ACCESS_LOG_DYNAMODB_TABLE_NAME, DYNAMODB_ACCESS_LOG_HEADERS, CSV_ACCESS_LOG_HEADERS, FEEDBACK_LOG_DYNAMODB_TABLE_NAME, DYNAMODB_FEEDBACK_LOG_HEADERS, CSV_FEEDBACK_LOG_HEADERS, USAGE_LOG_DYNAMODB_TABLE_NAME, DYNAMODB_USAGE_LOG_HEADERS, CSV_USAGE_LOG_HEADERS, TEXTRACT_JOBS_S3_INPUT_LOC, AWS_REGION
from tools.helper_functions import put_columns_in_df, get_connection_params, reveal_feedback_buttons, custom_regex_load, reset_state_vars, load_in_default_allow_list, tesseract_ocr_option, text_ocr_option, textract_option, local_pii_detector, aws_pii_detector, no_redaction_option, reset_review_vars, merge_csv_files, load_all_output_files, update_dataframe, check_for_existing_textract_file, load_in_default_cost_codes, enforce_cost_codes, calculate_aws_costs, calculate_time_taken, reset_base_dataframe, reset_ocr_base_dataframe, update_cost_code_dataframe_from_dropdown_select, check_for_existing_local_ocr_file, reset_data_vars, reset_aws_call_vars
from tools.aws_functions import upload_file_to_s3, download_file_from_s3, upload_log_file_to_s3
from tools.file_redaction import choose_and_run_redactor
from tools.file_conversion import prepare_image_or_pdf, get_input_file_names
from tools.redaction_review import apply_redactions_to_review_df_and_files, update_all_page_annotation_object_based_on_previous_page, decrease_page, increase_page, update_annotator_object_and_filter_df, update_entities_df_recogniser_entities, update_entities_df_page, update_entities_df_text, df_select_callback, convert_df_to_xfdf, convert_xfdf_to_dataframe, reset_dropdowns, exclude_selected_items_from_redaction, undo_last_removal, update_selected_review_df_row_colour, update_all_entity_df_dropdowns, df_select_callback_cost, update_other_annotator_number_from_current, update_annotator_page_from_review_df, df_select_callback_ocr, df_select_callback_textract_api
from tools.data_anonymise import anonymise_data_files
from tools.auth import authenticate_user
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.custom_csvlogger import CSVLogger_custom
from tools.find_duplicate_pages import identify_similar_pages
from tools.textract_batch_call import analyse_document_with_textract_api, poll_whole_document_textract_analysis_progress_and_download, load_in_textract_job_details, check_for_provided_job_id, check_textract_outputs_exist, replace_existing_pdf_input_for_whole_document_outputs

# Suppress downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

chosen_comprehend_entities = ['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE', 'PASSPORT_NUMBER','DRIVER_ID', 'USERNAME','PASSWORD', 'IP_ADDRESS','MAC_ADDRESS', 'LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER', 'INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER']

full_comprehend_entity_list = ['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE','SSN','DATE_TIME','PASSPORT_NUMBER','DRIVER_ID','URL','AGE','USERNAME','PASSWORD','AWS_ACCESS_KEY','AWS_SECRET_KEY','IP_ADDRESS','MAC_ADDRESS','ALL','LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER','CA_SOCIAL_INSURANCE_NUMBER','US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER','UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER','IN_PERMANENT_ACCOUNT_NUMBER','IN_NREGA','INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER','CA_HEALTH_NUMBER','IN_AADHAAR','IN_VOTER_NUMBER', "CUSTOM_FUZZY"]

# Add custom spacy recognisers to the Comprehend list, so that local Spacy model can be used to pick up e.g. titles, streetnames, UK postcodes that are sometimes missed by comprehend
chosen_comprehend_entities.extend(custom_entities)
full_comprehend_entity_list.extend(custom_entities)

# Entities for local PII redaction option
chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", "CUSTOM"]

full_entity_list = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS', 'CUSTOM', 'CUSTOM_FUZZY']

log_file_name = 'log.csv'

file_input_height = 200

if RUN_AWS_FUNCTIONS == "1":
    default_ocr_val = textract_option
    default_pii_detector = local_pii_detector
else:
    default_ocr_val = text_ocr_option
    default_pii_detector = local_pii_detector

SAVE_LOGS_TO_CSV = eval(SAVE_LOGS_TO_CSV)
SAVE_LOGS_TO_DYNAMODB = eval(SAVE_LOGS_TO_DYNAMODB)

if CSV_ACCESS_LOG_HEADERS: CSV_ACCESS_LOG_HEADERS = eval(CSV_ACCESS_LOG_HEADERS)
if CSV_FEEDBACK_LOG_HEADERS: CSV_FEEDBACK_LOG_HEADERS = eval(CSV_FEEDBACK_LOG_HEADERS)
if CSV_USAGE_LOG_HEADERS: CSV_USAGE_LOG_HEADERS = eval(CSV_USAGE_LOG_HEADERS)

if DYNAMODB_ACCESS_LOG_HEADERS: DYNAMODB_ACCESS_LOG_HEADERS = eval(DYNAMODB_ACCESS_LOG_HEADERS)
if DYNAMODB_FEEDBACK_LOG_HEADERS: DYNAMODB_FEEDBACK_LOG_HEADERS = eval(DYNAMODB_FEEDBACK_LOG_HEADERS)
if DYNAMODB_USAGE_LOG_HEADERS: DYNAMODB_USAGE_LOG_HEADERS = eval(DYNAMODB_USAGE_LOG_HEADERS)

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Base(), fill_width=True)

with app:

    ###
    # STATE VARIABLES
    ###
    
    # Pymupdf doc and all image annotations objects need to be stored as State objects as they do not have a standard Gradio component equivalent
    pdf_doc_state = gr.State([])    
    all_image_annotations_state = gr.State([])   

    
    all_decision_process_table_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"),  label="all_decision_process_table", visible=False, type="pandas", wrap=True)
    review_file_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="review_file_df", visible=False, type="pandas", wrap=True)

    all_page_line_level_ocr_results = gr.State([])    
    all_page_line_level_ocr_results_with_children = gr.State([])

    session_hash_state = gr.Textbox(label= "session_hash_state", value="", visible=False)
    host_name_textbox = gr.Textbox(label= "host_name_textbox", value=HOST_NAME, visible=False)
    s3_output_folder_state = gr.Textbox(label= "s3_output_folder_state", value="", visible=False)
    session_output_folder_textbox = gr.Textbox(value = SESSION_OUTPUT_FOLDER, label="session_output_folder_textbox", visible=False)
    output_folder_textbox = gr.Textbox(value = OUTPUT_FOLDER, label="output_folder_textbox", visible=False)
    input_folder_textbox = gr.Textbox(value = INPUT_FOLDER, label="input_folder_textbox", visible=False)

    first_loop_state = gr.Checkbox(label="first_loop_state", value=True, visible=False)
    second_loop_state = gr.Checkbox(label="second_loop_state", value=False, visible=False)
    do_not_save_pdf_state = gr.Checkbox(label="do_not_save_pdf_state", value=False, visible=False)
    save_pdf_state = gr.Checkbox(label="save_pdf_state", value=True, visible=False)

    prepared_pdf_state = gr.Dropdown(label = "prepared_pdf_list", value="", allow_custom_value=True,visible=False)
    document_cropboxes = gr.Dropdown(label = "document_cropboxes", value="", allow_custom_value=True,visible=False)
    page_sizes = gr.Dropdown(label = "page_sizes", value="", allow_custom_value=True, visible=False)
    images_pdf_state = gr.Dropdown(label = "images_pdf_list", value="", allow_custom_value=True,visible=False)
    all_img_details_state = gr.State([])
    
    output_image_files_state = gr.Dropdown(label = "output_image_files_list", value="", allow_custom_value=True,visible=False)
    output_file_list_state = gr.Dropdown(label = "output_file_list", value="", allow_custom_value=True,visible=False)
    text_output_file_list_state = gr.Dropdown(label = "text_output_file_list", value="", allow_custom_value=True,visible=False)
    log_files_output_list_state = gr.Dropdown(label = "log_files_output_list", value="", allow_custom_value=True,visible=False)
    duplication_file_path_outputs_list_state = gr.Dropdown(label = "duplication_file_path_outputs_list", value=[], multiselect=True, allow_custom_value=True,visible=False)

    # Backup versions of these objects in case you make a mistake
    backup_review_state = gr.Dataframe(visible=False)
    backup_image_annotations_state = gr.State([])
    backup_recogniser_entity_dataframe_base = gr.Dataframe(visible=False)    
    
    # Logging state
    feedback_logs_state = gr.Textbox(label= "feedback_logs_state", value=FEEDBACK_LOGS_FOLDER + log_file_name, visible=False)
    feedback_s3_logs_loc_state = gr.Textbox(label= "feedback_s3_logs_loc_state", value=FEEDBACK_LOGS_FOLDER, visible=False)
    access_logs_state = gr.Textbox(label= "access_logs_state", value=ACCESS_LOGS_FOLDER + log_file_name, visible=False)
    access_s3_logs_loc_state = gr.Textbox(label= "access_s3_logs_loc_state", value=ACCESS_LOGS_FOLDER, visible=False)
    usage_logs_state = gr.Textbox(label= "usage_logs_state", value=USAGE_LOGS_FOLDER + log_file_name, visible=False)
    usage_s3_logs_loc_state = gr.Textbox(label= "usage_s3_logs_loc_state", value=USAGE_LOGS_FOLDER, visible=False)

    session_hash_textbox = gr.Textbox(label= "session_hash_textbox", value="", visible=False)
    textract_metadata_textbox = gr.Textbox(label = "textract_metadata_textbox", value="", visible=False)
    comprehend_query_number = gr.Number(label = "comprehend_query_number", value=0, visible=False)
    textract_query_number = gr.Number(label = "textract_query_number", value=0, visible=False)
    
    doc_full_file_name_textbox = gr.Textbox(label = "doc_full_file_name_textbox", value="", visible=False)
    doc_file_name_no_extension_textbox = gr.Textbox(label = "doc_full_file_name_textbox", value="", visible=False)
    blank_doc_file_name_no_extension_textbox_for_logs = gr.Textbox(label = "doc_full_file_name_textbox", value="", visible=False)
    blank_data_file_name_no_extension_textbox_for_logs = gr.Textbox(label = "data_full_file_name_textbox", value="", visible=False)
    placeholder_doc_file_name_no_extension_textbox_for_logs = gr.Textbox(label = "doc_full_file_name_textbox", value="document", visible=False)
    placeholder_data_file_name_no_extension_textbox_for_logs = gr.Textbox(label = "data_full_file_name_textbox", value="data_file", visible=False)

    # Left blank for when user does not want to report file names
    doc_file_name_with_extension_textbox = gr.Textbox(label = "doc_file_name_with_extension_textbox", value="", visible=False)
    doc_file_name_textbox_list = gr.Dropdown(label = "doc_file_name_textbox_list", value="", allow_custom_value=True,visible=False)
    latest_review_file_path = gr.Textbox(label = "latest_review_file_path", value="", visible=False) # Latest review file path output from redaction
    latest_ocr_file_path = gr.Textbox(label = "latest_ocr_file_path", value="", visible=False) # Latest ocr file path output from text extraction

    data_full_file_name_textbox = gr.Textbox(label = "data_full_file_name_textbox", value="", visible=False)
    data_file_name_no_extension_textbox = gr.Textbox(label = "data_full_file_name_textbox", value="", visible=False)
    data_file_name_with_extension_textbox = gr.Textbox(label = "data_file_name_with_extension_textbox", value="", visible=False)
    data_file_name_textbox_list = gr.Dropdown(label = "data_file_name_textbox_list", value="", allow_custom_value=True,visible=False)

    # Constants just to use with the review dropdowns for filtering by various columns
    label_name_const = gr.Textbox(label="label_name_const", value="label", visible=False)
    text_name_const = gr.Textbox(label="text_name_const", value="text", visible=False)
    page_name_const = gr.Textbox(label="page_name_const", value="page", visible=False)
    
    actual_time_taken_number = gr.Number(label = "actual_time_taken_number", value=0.0, precision=1, visible=False) # This keeps track of the time taken to redact files for logging purposes.
    annotate_previous_page = gr.Number(value=0, label="Previous page", precision=0, visible=False) # Keeps track of the last page that the annotator was on
    s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)

    ## Annotator zoom value
    annotator_zoom_number = gr.Number(label = "Current annotator zoom level", value=100, precision=0, visible=False)
    zoom_true_bool = gr.Checkbox(label="zoom_true_bool", value=True, visible=False)
    zoom_false_bool = gr.Checkbox(label="zoom_false_bool", value=False, visible=False)

    clear_all_page_redactions = gr.Checkbox(label="clear_all_page_redactions", value=True, visible=False)
    prepare_for_review_bool = gr.Checkbox(label="prepare_for_review_bool", value=True, visible=False)
    prepare_for_review_bool_false = gr.Checkbox(label="prepare_for_review_bool_false", value=False, visible=False)
    prepare_images_bool_false = gr.Checkbox(label="prepare_images_bool_false", value=False, visible=False)

    ## Settings page variables
    default_deny_list_file_name = "default_deny_list.csv"
    default_deny_list_loc = OUTPUT_FOLDER + "/" + default_deny_list_file_name    
    in_deny_list_text_in = gr.Textbox(value="deny_list", visible=False)

    fully_redacted_list_file_name = "default_fully_redacted_list.csv"
    fully_redacted_list_loc = OUTPUT_FOLDER + "/" + fully_redacted_list_file_name    
    in_fully_redacted_text_in = gr.Textbox(value="fully_redacted_pages_list", visible=False)

    # S3 settings for default allow list load
    s3_default_bucket = gr.Textbox(label = "Default S3 bucket", value=DOCUMENT_REDACTION_BUCKET, visible=False)
    s3_default_allow_list_file = gr.Textbox(label = "Default allow list file", value=S3_ALLOW_LIST_PATH, visible=False)
    default_allow_list_output_folder_location = gr.Textbox(label = "Output default allow list location", value=OUTPUT_ALLOW_LIST_PATH, visible=False)

    s3_whole_document_textract_default_bucket = gr.Textbox(label = "Default Textract whole_document S3 bucket", value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET, visible=False)
    s3_whole_document_textract_input_subfolder = gr.Textbox(label = "Default Textract whole_document S3 input folder", value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER, visible=False)
    s3_whole_document_textract_output_subfolder = gr.Textbox(label = "Default Textract whole_document S3 output folder", value=TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER, visible=False)
    successful_textract_api_call_number = gr.Number(precision=0, value=0, visible=False)
    no_redaction_method_drop = gr.Radio(label = """Placeholder for no redaction method after downloading Textract outputs""", value = no_redaction_option, choices=[no_redaction_option], visible=False)
    textract_only_method_drop = gr.Radio(label="""Placeholder for Textract method after downloading Textract outputs""", value = textract_option, choices=[textract_option], visible=False)

    load_s3_whole_document_textract_logs_bool = gr.Textbox(label = "Load Textract logs or not", value=LOAD_PREVIOUS_TEXTRACT_JOBS_S3, visible=False)    
    s3_whole_document_textract_logs_subfolder = gr.Textbox(label = "Default Textract whole_document S3 input folder", value=TEXTRACT_JOBS_S3_LOC, visible=False)
    local_whole_document_textract_logs_subfolder = gr.Textbox(label = "Default Textract whole_document S3 output folder", value=TEXTRACT_JOBS_LOCAL_LOC, visible=False)       

    s3_default_cost_codes_file = gr.Textbox(label = "Default cost centre file", value=S3_COST_CODES_PATH, visible=False)
    default_cost_codes_output_folder_location = gr.Textbox(label = "Output default cost centre location", value=OUTPUT_COST_CODES_PATH, visible=False)
    enforce_cost_code_textbox = gr.Textbox(label = "Enforce cost code textbox", value=ENFORCE_COST_CODES, visible=False)
    default_cost_code_textbox = gr.Textbox(label = "Default cost code textbox", value=DEFAULT_COST_CODE, visible=False)

    # Base tables that are not modified subsequent to load
    recogniser_entity_dataframe_base = gr.Dataframe(pd.DataFrame(data={"page":[], "label":[], "text":[], "id":[]}), col_count=4, type="pandas", visible=False, label="recogniser_entity_dataframe_base", show_search="filter", headers=["page", "label", "text", "id"], show_fullscreen_button=True, wrap=True, static_columns=[0,1,2,3])
    all_line_level_ocr_results_df_base = gr.Dataframe(value=pd.DataFrame(), headers=["page", "text"], col_count=(2, 'fixed'), row_count = (0, "dynamic"),  label="All OCR results", type="pandas", wrap=True, show_fullscreen_button=True, show_search='filter', show_label=False, show_copy_button=True, visible=False)
    all_line_level_ocr_results_df_placeholder = gr.Dataframe(visible=False)
    cost_code_dataframe_base = gr.Dataframe(value=pd.DataFrame(), row_count = (0, "dynamic"), label="Cost codes", type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, show_search='filter', wrap=True, max_height=200, visible=False)

    # Duplicate page detection
    in_duplicate_pages_text = gr.Textbox(label="in_duplicate_pages_text", visible=False)
    duplicate_pages_df = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="duplicate_pages_df", visible=False, type="pandas", wrap=True)

    # Tracking variables for current page (not visible)
    current_loop_page_number = gr.Number(value=0,precision=0, interactive=False, label = "Last redacted page in document", visible=False)
    page_break_return = gr.Checkbox(value = False, label="Page break reached", visible=False)

    # Placeholders for elements that may be made visible later below depending on environment variables
    cost_code_dataframe = gr.Dataframe(value=pd.DataFrame(), type="pandas", visible=False, wrap=True)
    cost_code_choice_drop = gr.Dropdown(value=DEFAULT_COST_CODE, label="Choose cost code for analysis. Please contact Finance if you can't find your cost code in the given list.", choices=[DEFAULT_COST_CODE], allow_custom_value=False, visible=False)

    textract_output_found_checkbox = gr.Checkbox(value= False, label="Existing Textract output file found", interactive=False, visible=False)
    local_ocr_output_found_checkbox = gr.Checkbox(value= False, label="Existing local OCR output file found", interactive=False, visible=False)
    total_pdf_page_count = gr.Number(label = "Total page count", value=0, visible=False)
    estimated_aws_costs_number = gr.Number(label = "Approximate AWS Textract and/or Comprehend cost ($)", value=0, visible=False, precision=2)
    estimated_time_taken_number = gr.Number(label = "Approximate time taken to extract text/redact (minutes)", value=0, visible=False, precision=2)

    only_extract_text_radio = gr.Checkbox(value=False, label="Only extract text (no redaction)", visible=False)

    # Textract API call placeholders in case option not selected in config
                
    job_name_textbox = gr.Textbox(value="", label="whole_document Textract call", visible=False)
    send_document_to_textract_api_btn = gr.Button("Analyse document with AWS Textract", variant="primary", visible=False)

    job_id_textbox = gr.Textbox(label = "Latest job ID for whole_document document analysis", value='', visible=False)              
    check_state_of_textract_api_call_btn = gr.Button("Check state of Textract document job and download", variant="secondary", visible=False)
    job_current_status = gr.Textbox(value="", label="Analysis job current status", visible=False)
    job_type_dropdown = gr.Dropdown(value="document_text_detection", choices=["document_text_detection", "document_analysis"], label="Job type of Textract analysis job", allow_custom_value=False, visible=False)
    textract_job_detail_df = gr.Dataframe(pd.DataFrame(columns=['job_id','file_name','job_type','signature_extraction','s3_location','job_date_time']), label="Previous job details", visible=False, type="pandas", wrap=True)
    selected_job_id_row = gr.Dataframe(pd.DataFrame(columns=['job_id','file_name','job_type','signature_extraction','s3_location','job_date_time']), label="Selected job id row", visible=False, type="pandas", wrap=True)
    is_a_textract_api_call = gr.Checkbox(value=False, label="is_this_a_textract_api_call", visible=False)
    job_output_textbox = gr.Textbox(value="", label="Textract call outputs", visible=False)
    job_input_textbox = gr.Textbox(value=TEXTRACT_JOBS_S3_INPUT_LOC, label="Textract call outputs", visible=False)

    textract_job_output_file = gr.File(label="Textract job output files", height=file_input_height, visible=False)
    convert_textract_outputs_to_ocr_results = gr.Button("Placeholder - Convert Textract job outputs to OCR results (needs relevant document file uploaded above)", variant="secondary", visible=False)

    ###
    # UI DESIGN
    ###

    gr.Markdown(
    """# Document redaction

    Redact personally identifiable information (PII) from documents (PDF, images), open text, or tabular data (XLSX/CSV/Parquet). Please see the [User Guide](https://github.com/seanpedrick-case/doc_redaction/blob/main/README.md) for a walkthrough on how to use the app. Below is a very brief overview.
    
    To identify text in documents, the 'Local' text/OCR image analysis uses spacy/tesseract, and works ok for documents with typed text. If available, choose 'AWS Textract' to redact more complex elements e.g. signatures or handwriting. Then, choose a method for PII identification. 'Local' is quick and gives good results if you are primarily looking for a custom list of terms to redact (see Redaction settings). If available, AWS Comprehend gives better results at a small cost.
    
    After redaction, review suggested redactions on the 'Review redactions' tab. The original pdf can be uploaded here alongside a '...review_file.csv' to continue a previous redaction/review task. See the 'Redaction settings' tab to choose which pages to redact, the type of information to redact (e.g. people, places), or custom terms to always include/ exclude from redaction.

    NOTE: The app is not 100% accurate, and it will miss some personal information. It is essential that all outputs are reviewed **by a human** before using the final outputs.""")

    ###
    # REDACTION PDF/IMAGES TABLE
    ###
    with gr.Tab("Redact PDFs/images"):
        with gr.Accordion("Redact document", open = True):
            in_doc_files = gr.File(label="Choose a document or image file (PDF, JPG, PNG)", file_count= "multiple", file_types=['.pdf', '.jpg', '.png', '.json', '.zip'], height=file_input_height)

            text_extract_method_radio = gr.Radio(label="""Choose text extraction method. Local options are lower quality but cost nothing - they may be worth a try if you are willing to spend some time reviewing outputs. AWS Textract has a cost per page - £2.66 ($3.50) per 1,000 pages with signature detection (default), £1.14 ($1.50) without. Change the settings in the tab below (AWS Textract signature detection) to change this.""", value = default_ocr_val, choices=[text_ocr_option, tesseract_ocr_option, textract_option])

            with gr.Accordion("AWS Textract signature detection (default is on)", open = False):
                handwrite_signature_checkbox = gr.CheckboxGroup(label="AWS Textract extraction settings", choices=["Extract handwriting", "Extract signatures"], value=["Extract handwriting", "Extract signatures"])

            with gr.Row(equal_height=True):
                pii_identification_method_drop = gr.Radio(label = """Choose personal information detection method. The local model is lower quality but costs nothing - it may be worth a try if you are willing to spend some time reviewing outputs, or if you are only interested in searching for custom search terms (see Redaction settings - custom deny list). AWS Comprehend has a cost of around £0.0075 ($0.01) per 10,000 characters.""", value = default_pii_detector, choices=[no_redaction_option, local_pii_detector, aws_pii_detector])
            
            if SHOW_COSTS == "True":
                with gr.Accordion("Estimated costs and time taken. Note that costs shown only include direct usage of AWS services and do not include other running costs (e.g. storage, run-time costs)", open = True, visible=True):                        
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            textract_output_found_checkbox = gr.Checkbox(value= False, label="Existing Textract output file found", interactive=False, visible=True)
                            local_ocr_output_found_checkbox = gr.Checkbox(value= False, label="Existing local OCR output file found", interactive=False, visible=True)
                        with gr.Column(scale=4):
                            with gr.Row(equal_height=True):
                                total_pdf_page_count = gr.Number(label = "Total page count", value=0, visible=True)
                                estimated_aws_costs_number = gr.Number(label = "Approximate AWS Textract and/or Comprehend cost (£)", value=0.00, precision=2, visible=True)
                                estimated_time_taken_number = gr.Number(label = "Approximate time taken to extract text/redact (minutes)", value=0, visible=True, precision=2)       
                
            if GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True":
                with gr.Accordion("Assign task to cost code", open = True, visible=True):
                    gr.Markdown("Please ensure that you have approval from your budget holder before using this app for redaction tasks that incur a cost.")
                    with gr.Row():
                        cost_code_dataframe = gr.Dataframe(value=pd.DataFrame(), row_count = (0, "dynamic"), label="Existing cost codes", type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, show_search='filter', visible=True, wrap=True, max_height=200)
                        with gr.Column():
                            reset_cost_code_dataframe_button = gr.Button(value="Reset code code table filter")
                            cost_code_choice_drop = gr.Dropdown(value=DEFAULT_COST_CODE, label="Choose cost code for analysis", choices=[DEFAULT_COST_CODE], allow_custom_value=False, visible=True)

            if SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS == "True":
                with gr.Accordion("Submit whole document to AWS Textract API (quicker, max 3,000 pages per document)", open = False, visible=True):
                    with gr.Row(equal_height=True):
                        gr.Markdown("""Document will be submitted to AWS Textract API service to extract all text in the document. Processing will take place on (secure) AWS servers, and outputs will be stored on S3 for up to 7 days. To download the results, click 'Check status' below and they will be downloaded if ready.""")
                    with gr.Row(equal_height=True):
                        send_document_to_textract_api_btn = gr.Button("Analyse document with AWS Textract API call", variant="primary", visible=True)                        
                    with gr.Row(equal_height=False):       
                        with gr.Column(scale=2):                 
                            textract_job_detail_df = gr.Dataframe(label="Previous job details", visible=True, type="pandas", wrap=True, interactive=True, row_count=(0, 'fixed'), col_count=(6,'fixed'), static_columns=[0,1,2,3,4,5])
                        with gr.Column(scale=1):
                            job_id_textbox = gr.Textbox(label = "Job ID to check status", value='', visible=True)     
                            check_state_of_textract_api_call_btn = gr.Button("Check status of Textract job and download", variant="secondary", visible=True)
                    with gr.Row():
                        job_current_status = gr.Textbox(value="", label="Analysis job current status", visible=True)                            
                        textract_job_output_file = gr.File(label="Textract job output files", height=100, visible=True)

                    convert_textract_outputs_to_ocr_results = gr.Button("Convert Textract job outputs to OCR results (needs relevant document file uploaded above)", variant="secondary", visible=True)           

            gr.Markdown("""If you only want to redact certain pages, or certain entities (e.g. just email addresses, or a custom list of terms), please go to the Redaction Settings tab.""")      
            document_redact_btn = gr.Button("Extract text and redact document", variant="primary", scale = 4)
        
        with gr.Row():
            redaction_output_summary_textbox = gr.Textbox(label="Output summary", scale=1)
            output_file = gr.File(label="Output files", scale = 2)#, height=file_input_height)
            latest_file_completed_text = gr.Number(value=0, label="Number of documents redacted", interactive=False, visible=False)

        # Feedback elements are invisible until revealed by redaction action
        pdf_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        pdf_feedback_radio = gr.Radio(label = "Quality of results", choices=["The results were good", "The results were not good"], visible=False)
        pdf_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        pdf_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)
        
    ###
    # REVIEW REDACTIONS TAB
    ###
    with gr.Tab("Review redactions", id="tab_object_annotation"):

        with gr.Accordion(label = "Review PDF redactions", open=True):
            output_review_files = gr.File(label="Upload original PDF and 'review_file' csv here to review suggested redactions. The 'ocr_output' file can also be optionally provided for text search.", file_count='multiple', height=file_input_height)
            upload_previous_review_file_btn = gr.Button("Review PDF and 'review file' csv provided above", variant="secondary")                   
        with gr.Row():
            annotate_zoom_in = gr.Button("Zoom in", visible=False)
            annotate_zoom_out = gr.Button("Zoom out", visible=False)        
        with gr.Row():
            clear_all_redactions_on_page_btn = gr.Button("Clear all redactions on page", visible=False) 

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):                       
                    annotation_last_page_button = gr.Button("Previous page", scale = 4)
                    annotate_current_page = gr.Number(value=1, label="Current page", precision=0, scale = 2, min_width=50)
                    annotate_max_pages = gr.Number(value=1, label="Total pages", precision=0, interactive=False, scale = 2, min_width=50)
                    annotation_next_page_button = gr.Button("Next page", scale = 4)

                zoom_str = str(annotator_zoom_number) + '%'

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
                    sources=None,#["upload"],
                    show_clear_button=False,
                    show_share_button=False,
                    show_remove_button=False,
                    handles_cursor=True,
                    interactive=False
                )

                with gr.Row(equal_height=True):
                    annotation_last_page_button_bottom = gr.Button("Previous page", scale = 4)
                    annotate_current_page_bottom = gr.Number(value=0, label="Current page", precision=0, interactive=True, scale = 2, min_width=50)
                    annotate_max_pages_bottom = gr.Number(value=0, label="Total pages", precision=0, interactive=False, scale = 2, min_width=50)
                    annotation_next_page_button_bottom = gr.Button("Next page", scale = 4)

            with gr.Column(scale=1):
                annotation_button_apply = gr.Button("Apply revised redactions to PDF", variant="primary")
                update_current_page_redactions_btn = gr.Button(value="Save changes on current page to file", variant="primary")
                with gr.Accordion("Search suggested redactions", open=True):
                    with gr.Row(equal_height=True):
                        recogniser_entity_dropdown = gr.Dropdown(label="Redaction category", value="ALL", allow_custom_value=True)
                        page_entity_dropdown = gr.Dropdown(label="Page", value="ALL", allow_custom_value=True)                    
                    text_entity_dropdown = gr.Dropdown(label="Text", value="ALL", allow_custom_value=True)
                    recogniser_entity_dataframe = gr.Dataframe(pd.DataFrame(data={"page":[], "label":[], "text":[], "id":[]}), col_count=(4,"fixed"), type="pandas", label="Search results. Click to go to page", headers=["page", "label", "text", "id"], show_fullscreen_button=True, wrap=True, max_height=400, static_columns=[0,1,2,3])

                    with gr.Row(equal_height=True):
                        exclude_selected_row_btn = gr.Button(value="Exclude specific row from redactions")
                        exclude_selected_btn = gr.Button(value="Exclude all items in table from redactions")
                    with gr.Row(equal_height=True):
                        reset_dropdowns_btn = gr.Button(value="Reset filters")
                        
                    undo_last_removal_btn = gr.Button(value="Undo last element removal")
                    
                    selected_entity_dataframe_row = gr.Dataframe(pd.DataFrame(data={"page":[], "label":[], "text":[], "id":[]}), col_count=4, type="pandas", visible=False, label="selected_entity_dataframe_row", headers=["page", "label", "text", "id"], show_fullscreen_button=True, wrap=True)
                    selected_entity_id = gr.Textbox(value="", label="selected_entity_id", visible=False)
                    selected_entity_colour = gr.Textbox(value="", label="selected_entity_colour", visible=False)

                with gr.Accordion("Search all extracted text", open=True):                    
                    all_line_level_ocr_results_df = gr.Dataframe(value=pd.DataFrame(), headers=["page", "text"], col_count=(2, 'fixed'), row_count = (0, "dynamic"),  label="All OCR results", visible=True, type="pandas", wrap=True, show_fullscreen_button=True, show_search='filter', show_label=False, show_copy_button=True, max_height=400)
                    reset_all_ocr_results_btn = gr.Button(value="Reset OCR output table filter")      
        
        with gr.Accordion("Convert review files loaded above to Adobe format, or convert from Adobe format to review file", open = False):
            convert_review_file_to_adobe_btn = gr.Button("Convert review file to Adobe comment format", variant="primary")
            adobe_review_files_out = gr.File(label="Output Adobe comment files will appear here. If converting from .xfdf file to review_file.csv, upload the original pdf with the xfdf file here then click Convert below.", file_count='multiple', file_types=['.csv', '.xfdf', '.pdf']) 
            convert_adobe_to_review_file_btn = gr.Button("Convert Adobe .xfdf comment file to review_file.csv", variant="secondary")

    ###
    # IDENTIFY DUPLICATE PAGES TAB
    ###
    with gr.Tab(label="Identify duplicate pages"):
        with gr.Accordion("Identify duplicate pages to redact", open = True):            
            in_duplicate_pages = gr.File(label="Upload multiple 'ocr_output.csv' data files from redaction jobs here to compare", file_count="multiple", height=file_input_height, file_types=['.csv'])
            with gr.Row():
                duplicate_threshold_value = gr.Number(value=0.9, label="Minimum similarity to be considered a duplicate (maximum = 1)", scale =1)
                find_duplicate_pages_btn = gr.Button(value="Identify duplicate pages", variant="primary", scale = 4)                

            duplicate_pages_out = gr.File(label="Duplicate pages analysis output", file_count="multiple", height=file_input_height, file_types=['.csv'])

    ###
    # TEXT / TABULAR DATA TAB
    ###
    with gr.Tab(label="Open text or Excel/csv files"):
        gr.Markdown("""### Choose open text or a tabular data file (xlsx or csv) to redact.""")    
        with gr.Accordion("Redact open text", open = False):
            in_text = gr.Textbox(label="Enter open text", lines=10)
        with gr.Accordion("Upload xlsx or csv files", open = True):
            in_data_files = gr.File(label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'], height=file_input_height)
        
        in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheets to anonymise"], multiselect = True, label="Select Excel sheets that you want to anonymise (showing sheets present across all Excel files).", visible=False, allow_custom_value=True)

        in_colnames = gr.Dropdown(choices=["Choose columns to anonymise"], multiselect = True, label="Select columns that you want to anonymise (showing columns present across all files).")

        pii_identification_method_drop_tabular = gr.Radio(label = "Choose PII detection method. AWS Comprehend has a cost of approximately $0.01 per 10,000 characters.", value = default_pii_detector, choices=[local_pii_detector, aws_pii_detector])

        with gr.Accordion("Anonymisation output format", open = False):
            anon_strat = gr.Radio(choices=["replace with 'REDACTED'", "replace with <ENTITY_NAME>", "redact completely", "hash", "mask"], label="Select an anonymisation method.", value = "replace with 'REDACTED'") # , "encrypt", "fake_first_name" are also available, but are not currently included as not that useful in current form
        
        tabular_data_redact_btn = gr.Button("Redact text/data files", variant="primary")
        
        with gr.Row():
            text_output_summary = gr.Textbox(label="Output result")
            text_output_file = gr.File(label="Output files")
            text_tabular_files_done = gr.Number(value=0, label="Number of tabular files redacted", interactive=False, visible=False)

        # Feedback elements are invisible until revealed by redaction action
        data_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        data_feedback_radio = gr.Radio(label="Please give some feedback about the results of the redaction. A reminder that the app is only expected to identify about 60% of personally identifiable information in a given (typed) document.",
                choices=["The results were good", "The results were not good"], visible=False, show_label=True)
        data_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        data_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

    ###
    # SETTINGS TAB
    ###
    with gr.Tab(label="Redaction settings"):       
        with gr.Accordion("Custom allow, deny, and full page redaction lists", open = True):
            with gr.Row():
                with gr.Column():
                    in_allow_list = gr.File(label="Import allow list file - csv table with one column of a different word/phrase on each row (case insensitive). Terms in this file will not be redacted.", file_count="multiple", height=file_input_height)
                    in_allow_list_text = gr.Textbox(label="Custom allow list load status")
                with gr.Column():
                    in_deny_list = gr.File(label="Import custom deny list - csv table with one column of a different word/phrase on each row (case insensitive). Terms in this file will always be redacted.", file_count="multiple", height=file_input_height)
                    in_deny_list_text = gr.Textbox(label="Custom deny list load status")
                with gr.Column():
                    in_fully_redacted_list = gr.File(label="Import fully redacted pages list - csv table with one column of page numbers on each row. Page numbers in this file will be fully redacted.", file_count="multiple", height=file_input_height)
                    in_fully_redacted_list_text = gr.Textbox(label="Fully redacted page list load status")
            with gr.Accordion("Manually modify custom allow, deny, and full page redaction lists (NOTE: you need to press Enter after modifying/adding an entry to the lists to apply them)", open = False):
                with gr.Row():
                    in_allow_list_state = gr.Dataframe(value=pd.DataFrame(), headers=["allow_list"], col_count=(1, "fixed"), row_count = (0, "dynamic"), label="Allow list", visible=True, type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, wrap=True)
                    in_deny_list_state = gr.Dataframe(value=pd.DataFrame(), headers=["deny_list"], col_count=(1, "fixed"), row_count = (0, "dynamic"), label="Deny list", visible=True, type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, wrap=True)
                    in_fully_redacted_list_state = gr.Dataframe(value=pd.DataFrame(), headers=["fully_redacted_pages_list"], col_count=(1, "fixed"), row_count = (0, "dynamic"), label="Fully redacted pages", visible=True, type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, datatype='number', wrap=True)
            
        with gr.Accordion("Select entity types to redact", open = True):
                in_redact_entities = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Local PII identification model (click empty space in box for full list)")
                in_redact_comprehend_entities = gr.Dropdown(value=chosen_comprehend_entities, choices=full_comprehend_entity_list, multiselect=True, label="AWS Comprehend PII identification model (click empty space in box for full list)")

                with gr.Row():
                    max_fuzzy_spelling_mistakes_num = gr.Number(label="Maximum number of spelling mistakes allowed for fuzzy matching (CUSTOM_FUZZY entity).", value=1, minimum=0, maximum=9, precision=0)
                    match_fuzzy_whole_phrase_bool = gr.Checkbox(label="Should fuzzy search match on entire phrases in deny list (as opposed to each word individually)?", value=True)

        with gr.Accordion("Redact only selected pages", open = False):
            with gr.Row():
                page_min = gr.Number(precision=0,minimum=0,maximum=9999, label="Lowest page to redact")
                page_max = gr.Number(precision=0,minimum=0,maximum=9999, label="Highest page to redact")

        with gr.Accordion("AWS options", open = False):
            #with gr.Row():
            in_redact_language = gr.Dropdown(value = REDACTION_LANGUAGE, choices = [REDACTION_LANGUAGE], label="Redaction language", multiselect=False, visible=False)

            with gr.Row():
                aws_access_key_textbox = gr.Textbox(value='', label="AWS access key for account with permissions for AWS Textract and Comprehend", visible=True, type="password")
                aws_secret_key_textbox = gr.Textbox(value='', label="AWS secret key for account with permissions for AWS Textract and Comprehend", visible=True, type="password")


            
        with gr.Accordion("Log file outputs", open = False):
            log_files_output = gr.File(label="Log file output", interactive=False)

        with gr.Accordion("Combine multiple review files", open = False):
            multiple_review_files_in_out = gr.File(label="Combine multiple review_file.csv files together here.", file_count='multiple', file_types=['.csv']) 
            merge_multiple_review_files_btn = gr.Button("Merge multiple review files into one", variant="primary")

        with gr.Accordion("View all output files from this session", open = False):
            all_output_files_btn = gr.Button("Click here to view all output files", variant="secondary")
            all_output_files = gr.File(label="All files in output folder", file_count='multiple', file_types=['.csv'], interactive=False)

    ###
    ### UI INTERACTION ###
    ###

    ###
    # PDF/IMAGE REDACTION
    ###
    # Recalculate estimated costs based on changes to inputs
    if SHOW_COSTS == 'True':
        # Calculate costs
        total_pdf_page_count.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        text_extract_method_radio.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        pii_identification_method_drop.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        handwrite_signature_checkbox.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        textract_output_found_checkbox.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        only_extract_text_radio.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])
        textract_output_found_checkbox.change(calculate_aws_costs, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio], outputs=[estimated_aws_costs_number])

        # Calculate time taken
        total_pdf_page_count.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio,          pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        text_extract_method_radio.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        pii_identification_method_drop.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        handwrite_signature_checkbox.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        textract_output_found_checkbox.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, handwrite_signature_checkbox,  pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        only_extract_text_radio.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        textract_output_found_checkbox.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])
        local_ocr_output_found_checkbox.change(calculate_time_taken, inputs=[total_pdf_page_count, text_extract_method_radio, pii_identification_method_drop, textract_output_found_checkbox, only_extract_text_radio, local_ocr_output_found_checkbox], outputs=[estimated_time_taken_number])

    # Allow user to select items from cost code dataframe for cost code
    if SHOW_COSTS=="True" and (GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True"):
        cost_code_dataframe.select(df_select_callback_cost, inputs=[cost_code_dataframe], outputs=[cost_code_choice_drop])
        reset_cost_code_dataframe_button.click(reset_base_dataframe, inputs=[cost_code_dataframe_base], outputs=[cost_code_dataframe])

        cost_code_choice_drop.select(update_cost_code_dataframe_from_dropdown_select, inputs=[cost_code_choice_drop, cost_code_dataframe_base], outputs=[cost_code_dataframe])

    in_doc_files.upload(fn=get_input_file_names, inputs=[in_doc_files], outputs=[doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count]).\
    success(fn = prepare_image_or_pdf, inputs=[in_doc_files, text_extract_method_radio, latest_file_completed_text, redaction_output_summary_textbox, first_loop_state, annotate_max_pages, all_image_annotations_state, prepare_for_review_bool_false, in_fully_redacted_list_state, output_folder_textbox, input_folder_textbox, prepare_images_bool_false], outputs=[redaction_output_summary_textbox, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations_state, review_file_state, document_cropboxes, page_sizes, textract_output_found_checkbox, all_img_details_state, all_line_level_ocr_results_df_base, local_ocr_output_found_checkbox]).\
    success(fn=check_for_existing_textract_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[textract_output_found_checkbox]).\
    success(fn=check_for_existing_local_ocr_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[local_ocr_output_found_checkbox])

    # Run redaction function
    document_redact_btn.click(fn = reset_state_vars, outputs=[all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, textract_metadata_textbox, annotator, output_file_list_state, log_files_output_list_state, recogniser_entity_dataframe, recogniser_entity_dataframe_base, pdf_doc_state, duplication_file_path_outputs_list_state, redaction_output_summary_textbox, is_a_textract_api_call, textract_query_number]).\
        success(fn= enforce_cost_codes, inputs=[enforce_cost_code_textbox, cost_code_choice_drop, cost_code_dataframe_base]).\
        success(fn= choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, text_extract_method_radio, in_allow_list_state, in_deny_list_state, in_fully_redacted_list_state, latest_file_completed_text, redaction_output_summary_textbox, output_file_list_state, log_files_output_list_state, first_loop_state, page_min, page_max, actual_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, pii_identification_method_drop, comprehend_query_number, max_fuzzy_spelling_mistakes_num, match_fuzzy_whole_phrase_bool, aws_access_key_textbox, aws_secret_key_textbox, annotate_max_pages, review_file_state, output_folder_textbox, document_cropboxes, page_sizes, textract_output_found_checkbox, only_extract_text_radio, duplication_file_path_outputs_list_state, latest_review_file_path, input_folder_textbox, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children],
                    outputs=[redaction_output_summary_textbox, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, actual_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, output_review_files, annotate_max_pages, annotate_max_pages_bottom, prepared_pdf_state, images_pdf_state, review_file_state, page_sizes, duplication_file_path_outputs_list_state, in_duplicate_pages, latest_review_file_path, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children], api_name="redact_doc").\
                    success(fn=update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, page_min, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs=[annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state])

    # If the app has completed a batch of pages, it will rerun the redaction process until the end of all pages in the document
    current_loop_page_number.change(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, text_extract_method_radio, in_allow_list_state, in_deny_list_state, in_fully_redacted_list_state, latest_file_completed_text, redaction_output_summary_textbox, output_file_list_state, log_files_output_list_state, second_loop_state, page_min, page_max, actual_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, pii_identification_method_drop, comprehend_query_number, max_fuzzy_spelling_mistakes_num, match_fuzzy_whole_phrase_bool, aws_access_key_textbox, aws_secret_key_textbox, annotate_max_pages, review_file_state, output_folder_textbox, document_cropboxes, page_sizes, textract_output_found_checkbox, only_extract_text_radio, duplication_file_path_outputs_list_state, latest_review_file_path, input_folder_textbox, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children],
                    outputs=[redaction_output_summary_textbox, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, actual_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, output_review_files, annotate_max_pages, annotate_max_pages_bottom, prepared_pdf_state, images_pdf_state, review_file_state, page_sizes, duplication_file_path_outputs_list_state, in_duplicate_pages, latest_review_file_path, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children]).\
                    success(fn=update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, page_min, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs=[annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state])
        
    # If a file has been completed, the function will continue onto the next document
    latest_file_completed_text.change(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, text_extract_method_radio, in_allow_list_state, in_deny_list_state, in_fully_redacted_list_state, latest_file_completed_text, redaction_output_summary_textbox, output_file_list_state, log_files_output_list_state, second_loop_state, page_min, page_max, actual_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, pii_identification_method_drop, comprehend_query_number, max_fuzzy_spelling_mistakes_num, match_fuzzy_whole_phrase_bool, aws_access_key_textbox, aws_secret_key_textbox, annotate_max_pages, review_file_state, output_folder_textbox, document_cropboxes, page_sizes, textract_output_found_checkbox, only_extract_text_radio, duplication_file_path_outputs_list_state, latest_review_file_path, input_folder_textbox, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children],
                    outputs=[redaction_output_summary_textbox, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, actual_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, output_review_files, annotate_max_pages, annotate_max_pages_bottom, prepared_pdf_state, images_pdf_state, review_file_state, page_sizes, duplication_file_path_outputs_list_state, in_duplicate_pages, latest_review_file_path, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children]).\
                    success(fn=update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, page_min, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs=[annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
                    success(fn=check_for_existing_textract_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[textract_output_found_checkbox]).\
                    success(fn=check_for_existing_local_ocr_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[local_ocr_output_found_checkbox]).\
                    success(fn=reveal_feedback_buttons, outputs=[pdf_feedback_radio, pdf_further_details_text, pdf_submit_feedback_btn, pdf_feedback_title]).\
                    success(fn = reset_aws_call_vars, outputs=[comprehend_query_number, textract_query_number])
    
    # If the line level ocr results are changed by load in by user or by a new redaction task, replace the ocr results displayed in the table    
    all_line_level_ocr_results_df_base.change(reset_ocr_base_dataframe, inputs=[all_line_level_ocr_results_df_base], outputs=[all_line_level_ocr_results_df])

    # Send whole document to Textract for text extraction
    send_document_to_textract_api_btn.click(analyse_document_with_textract_api, inputs=[prepared_pdf_state, s3_whole_document_textract_input_subfolder, s3_whole_document_textract_output_subfolder, textract_job_detail_df, s3_whole_document_textract_default_bucket, output_folder_textbox, handwrite_signature_checkbox, successful_textract_api_call_number, total_pdf_page_count], outputs=[job_output_textbox, job_id_textbox, job_type_dropdown, successful_textract_api_call_number, is_a_textract_api_call, textract_query_number]).\
        success(check_for_provided_job_id, inputs=[job_id_textbox]).\
        success(poll_whole_document_textract_analysis_progress_and_download, inputs=[job_id_textbox, job_type_dropdown, s3_whole_document_textract_output_subfolder, doc_file_name_no_extension_textbox, textract_job_detail_df, s3_whole_document_textract_default_bucket, output_folder_textbox, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder], outputs = [textract_job_output_file, job_current_status, textract_job_detail_df, doc_file_name_no_extension_textbox]).\
        success(fn=check_for_existing_textract_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[textract_output_found_checkbox])
    
    check_state_of_textract_api_call_btn.click(check_for_provided_job_id, inputs=[job_id_textbox]).\
        success(poll_whole_document_textract_analysis_progress_and_download, inputs=[job_id_textbox, job_type_dropdown, s3_whole_document_textract_output_subfolder, doc_file_name_no_extension_textbox, textract_job_detail_df, s3_whole_document_textract_default_bucket, output_folder_textbox, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder], outputs = [textract_job_output_file, job_current_status, textract_job_detail_df, doc_file_name_no_extension_textbox]).\
    success(fn=check_for_existing_textract_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[textract_output_found_checkbox])

    textract_job_detail_df.select(df_select_callback_textract_api, inputs=[textract_output_found_checkbox], outputs=[job_id_textbox, job_type_dropdown, selected_job_id_row])


    convert_textract_outputs_to_ocr_results.click(replace_existing_pdf_input_for_whole_document_outputs, inputs = [s3_whole_document_textract_input_subfolder, doc_file_name_no_extension_textbox, output_folder_textbox, s3_whole_document_textract_default_bucket, in_doc_files, input_folder_textbox], outputs = [in_doc_files, doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count]).\
        success(fn = prepare_image_or_pdf, inputs=[in_doc_files, text_extract_method_radio, latest_file_completed_text, redaction_output_summary_textbox, first_loop_state, annotate_max_pages, all_image_annotations_state, prepare_for_review_bool_false, in_fully_redacted_list_state, output_folder_textbox, input_folder_textbox, prepare_images_bool_false], outputs=[redaction_output_summary_textbox, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations_state, review_file_state, document_cropboxes, page_sizes, textract_output_found_checkbox, all_img_details_state, all_line_level_ocr_results_df_base, local_ocr_output_found_checkbox]).\
        success(fn=check_for_existing_textract_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[textract_output_found_checkbox]).\
        success(fn=check_for_existing_local_ocr_file, inputs=[doc_file_name_no_extension_textbox, output_folder_textbox], outputs=[local_ocr_output_found_checkbox]).\
        success(fn= check_textract_outputs_exist, inputs=[textract_output_found_checkbox]).\
        success(fn = reset_state_vars, outputs=[all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, textract_metadata_textbox, annotator, output_file_list_state, log_files_output_list_state, recogniser_entity_dataframe, recogniser_entity_dataframe_base, pdf_doc_state, duplication_file_path_outputs_list_state, redaction_output_summary_textbox, is_a_textract_api_call, textract_query_number]).\
        success(fn= choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, textract_only_method_drop, in_allow_list_state, in_deny_list_state, in_fully_redacted_list_state, latest_file_completed_text, redaction_output_summary_textbox, output_file_list_state, log_files_output_list_state, first_loop_state, page_min, page_max, actual_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_base, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, no_redaction_method_drop, comprehend_query_number, max_fuzzy_spelling_mistakes_num, match_fuzzy_whole_phrase_bool, aws_access_key_textbox, aws_secret_key_textbox, annotate_max_pages, review_file_state, output_folder_textbox, document_cropboxes, page_sizes, textract_output_found_checkbox, only_extract_text_radio, duplication_file_path_outputs_list_state, latest_review_file_path, input_folder_textbox, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children],
                    outputs=[redaction_output_summary_textbox, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, actual_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_base, all_decision_process_table_state, comprehend_query_number, output_review_files, annotate_max_pages, annotate_max_pages_bottom, prepared_pdf_state, images_pdf_state, review_file_state, page_sizes, duplication_file_path_outputs_list_state, in_duplicate_pages, latest_review_file_path, textract_query_number, latest_ocr_file_path, all_page_line_level_ocr_results, all_page_line_level_ocr_results_with_children])
    
    ###
    # REVIEW PDF REDACTIONS
    ###

    # Upload previous files for modifying redactions
    upload_previous_review_file_btn.click(fn=reset_review_vars, inputs=None, outputs=[recogniser_entity_dataframe, recogniser_entity_dataframe_base]).\
        success(fn=get_input_file_names, inputs=[output_review_files], outputs=[doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count]).\
        success(fn = prepare_image_or_pdf, inputs=[output_review_files, text_extract_method_radio, latest_file_completed_text, redaction_output_summary_textbox, second_loop_state, annotate_max_pages, all_image_annotations_state, prepare_for_review_bool, in_fully_redacted_list_state, output_folder_textbox, input_folder_textbox, prepare_images_bool_false], outputs=[redaction_output_summary_textbox, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations_state, review_file_state, document_cropboxes, page_sizes, textract_output_found_checkbox, all_img_details_state, all_line_level_ocr_results_df_base, local_ocr_output_found_checkbox], api_name="prepare_doc").\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state])

    # Page number controls
    annotate_current_page.change(update_all_page_annotation_object_based_on_previous_page, inputs = [annotator, annotate_current_page, annotate_previous_page, all_image_annotations_state, page_sizes], outputs = [all_image_annotations_state, annotate_previous_page, annotate_current_page_bottom]).\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
        success(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, do_not_save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state])
    
    annotation_last_page_button.click(fn=decrease_page, inputs=[annotate_current_page], outputs=[annotate_current_page, annotate_current_page_bottom])
    annotation_next_page_button.click(fn=increase_page, inputs=[annotate_current_page, all_image_annotations_state], outputs=[annotate_current_page, annotate_current_page_bottom])        

    annotation_last_page_button_bottom.click(fn=decrease_page, inputs=[annotate_current_page], outputs=[annotate_current_page, annotate_current_page_bottom])    
    annotation_next_page_button_bottom.click(fn=increase_page, inputs=[annotate_current_page, all_image_annotations_state], outputs=[annotate_current_page, annotate_current_page_bottom])

    annotate_current_page_bottom.submit(update_other_annotator_number_from_current, inputs=[annotate_current_page_bottom], outputs=[annotate_current_page])

    # Apply page redactions
    annotation_button_apply.click(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state], scroll_to_output=True)

    # Save current page redactions
    update_current_page_redactions_btn.click(update_all_page_annotation_object_based_on_previous_page, inputs = [annotator, annotate_current_page, annotate_current_page, all_image_annotations_state, page_sizes], outputs = [all_image_annotations_state, annotate_previous_page, annotate_current_page_bottom]).\
    success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
    success(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, do_not_save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state])
    
    # Review table controls
    recogniser_entity_dropdown.select(update_entities_df_recogniser_entities, inputs=[recogniser_entity_dropdown, recogniser_entity_dataframe_base, page_entity_dropdown, text_entity_dropdown], outputs=[recogniser_entity_dataframe, text_entity_dropdown, page_entity_dropdown])
    page_entity_dropdown.select(update_entities_df_page, inputs=[page_entity_dropdown, recogniser_entity_dataframe_base, recogniser_entity_dropdown, text_entity_dropdown], outputs=[recogniser_entity_dataframe, recogniser_entity_dropdown, text_entity_dropdown])
    text_entity_dropdown.select(update_entities_df_text, inputs=[text_entity_dropdown, recogniser_entity_dataframe_base, recogniser_entity_dropdown, page_entity_dropdown], outputs=[recogniser_entity_dataframe, recogniser_entity_dropdown, page_entity_dropdown])

    # Clicking on a cell in the recogniser entity dataframe will take you to that page, and also highlight the target redaction box in blue
    recogniser_entity_dataframe.select(df_select_callback, inputs=[recogniser_entity_dataframe], outputs=[selected_entity_dataframe_row]).\
        success(update_selected_review_df_row_colour, inputs=[selected_entity_dataframe_row, review_file_state, selected_entity_id, selected_entity_colour], outputs=[review_file_state, selected_entity_id, selected_entity_colour]).\
        success(update_annotator_page_from_review_df, inputs=[review_file_state, images_pdf_state, page_sizes, all_image_annotations_state, annotator, selected_entity_dataframe_row, input_folder_textbox, doc_full_file_name_textbox], outputs=[annotator, all_image_annotations_state, annotate_current_page, page_sizes, review_file_state, annotate_previous_page])
   
    reset_dropdowns_btn.click(reset_dropdowns, inputs=[recogniser_entity_dataframe_base], outputs=[recogniser_entity_dropdown, text_entity_dropdown, page_entity_dropdown]).\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state])
    
    # Exclude current selection from annotator and outputs
    # Exclude only row
    exclude_selected_row_btn.click(exclude_selected_items_from_redaction, inputs=[review_file_state, selected_entity_dataframe_row, images_pdf_state, page_sizes, all_image_annotations_state, recogniser_entity_dataframe_base], outputs=[review_file_state, all_image_annotations_state, recogniser_entity_dataframe_base, backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base]).\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
        success(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, do_not_save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state]).\
        success(update_all_entity_df_dropdowns, inputs=[recogniser_entity_dataframe_base, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown], outputs=[recogniser_entity_dropdown, text_entity_dropdown, page_entity_dropdown])
    
    # Exclude everything visible in table
    exclude_selected_btn.click(exclude_selected_items_from_redaction, inputs=[review_file_state, recogniser_entity_dataframe, images_pdf_state, page_sizes, all_image_annotations_state, recogniser_entity_dataframe_base], outputs=[review_file_state, all_image_annotations_state, recogniser_entity_dataframe_base, backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base]).\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
        success(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, do_not_save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state]).\
        success(update_all_entity_df_dropdowns, inputs=[recogniser_entity_dataframe_base, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown], outputs=[recogniser_entity_dropdown, text_entity_dropdown, page_entity_dropdown])

    undo_last_removal_btn.click(undo_last_removal, inputs=[backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base], outputs=[review_file_state, all_image_annotations_state, recogniser_entity_dataframe_base]).\
        success(update_annotator_object_and_filter_df, inputs=[all_image_annotations_state, annotate_current_page, recogniser_entity_dropdown, page_entity_dropdown, text_entity_dropdown, recogniser_entity_dataframe_base, annotator_zoom_number, review_file_state, page_sizes, doc_full_file_name_textbox, input_folder_textbox], outputs = [annotator, annotate_current_page, annotate_current_page_bottom, annotate_previous_page, recogniser_entity_dropdown, recogniser_entity_dataframe, recogniser_entity_dataframe_base, text_entity_dropdown, page_entity_dropdown, page_sizes, all_image_annotations_state]).\
        success(apply_redactions_to_review_df_and_files, inputs=[annotator, doc_full_file_name_textbox, pdf_doc_state, all_image_annotations_state, annotate_current_page, review_file_state, output_folder_textbox, do_not_save_pdf_state, page_sizes], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files, log_files_output, review_file_state])
    

    
    # Review OCR text buttom
    all_line_level_ocr_results_df.select(df_select_callback_ocr, inputs=[all_line_level_ocr_results_df], outputs=[annotate_current_page, selected_entity_dataframe_row], scroll_to_output=True)
    reset_all_ocr_results_btn.click(reset_ocr_base_dataframe, inputs=[all_line_level_ocr_results_df_base], outputs=[all_line_level_ocr_results_df])
    
    # Convert review file to xfdf Adobe format
    convert_review_file_to_adobe_btn.click(fn=get_input_file_names, inputs=[output_review_files], outputs=[doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count]).\
        success(fn = prepare_image_or_pdf, inputs=[output_review_files, text_extract_method_radio, latest_file_completed_text, redaction_output_summary_textbox, second_loop_state, annotate_max_pages, all_image_annotations_state, prepare_for_review_bool, in_fully_redacted_list_state, output_folder_textbox, input_folder_textbox, prepare_images_bool_false], outputs=[redaction_output_summary_textbox, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations_state, review_file_state, document_cropboxes, page_sizes, textract_output_found_checkbox, all_img_details_state, all_line_level_ocr_results_df_placeholder, local_ocr_output_found_checkbox]).\
        success(convert_df_to_xfdf, inputs=[output_review_files, pdf_doc_state, images_pdf_state, output_folder_textbox, document_cropboxes, page_sizes], outputs=[adobe_review_files_out])
    
    # Convert xfdf Adobe file back to review_file.csv
    convert_adobe_to_review_file_btn.click(fn=get_input_file_names, inputs=[adobe_review_files_out], outputs=[doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count]).\
        success(fn = prepare_image_or_pdf, inputs=[adobe_review_files_out, text_extract_method_radio, latest_file_completed_text, redaction_output_summary_textbox, second_loop_state, annotate_max_pages, all_image_annotations_state, prepare_for_review_bool, in_fully_redacted_list_state, output_folder_textbox, input_folder_textbox, prepare_images_bool_false], outputs=[redaction_output_summary_textbox, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations_state, review_file_state, document_cropboxes, page_sizes, textract_output_found_checkbox, all_img_details_state, all_line_level_ocr_results_df_placeholder, local_ocr_output_found_checkbox]).\
        success(fn=convert_xfdf_to_dataframe, inputs=[adobe_review_files_out, pdf_doc_state, images_pdf_state, output_folder_textbox], outputs=[output_review_files], scroll_to_output=True)
    
    ###
    # TABULAR DATA REDACTION
    ###
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets]).\
                  success(fn=get_input_file_names, inputs=[in_data_files], outputs=[data_file_name_no_extension_textbox, data_file_name_with_extension_textbox, data_full_file_name_textbox, data_file_name_textbox_list, total_pdf_page_count])

    tabular_data_redact_btn.click(reset_data_vars, outputs=[actual_time_taken_number, log_files_output_list_state, comprehend_query_number]).\
    success(fn=anonymise_data_files, inputs=[in_data_files, in_text, anon_strat, in_colnames, in_redact_language, in_redact_entities, in_allow_list_state, text_tabular_files_done, text_output_summary, text_output_file_list_state, log_files_output_list_state, in_excel_sheets, first_loop_state, output_folder_textbox, in_deny_list_state, max_fuzzy_spelling_mistakes_num, pii_identification_method_drop_tabular, in_redact_comprehend_entities, comprehend_query_number, aws_access_key_textbox, aws_secret_key_textbox, actual_time_taken_number], outputs=[text_output_summary, text_output_file, text_output_file_list_state, text_tabular_files_done, log_files_output, log_files_output_list_state, actual_time_taken_number], api_name="redact_data").\
    success(fn = reveal_feedback_buttons, outputs=[data_feedback_radio, data_further_details_text, data_submit_feedback_btn, data_feedback_title])

    # Currently only supports redacting one data file at a time
    # If the output file count text box changes, keep going with redacting each data file until done
    # text_tabular_files_done.change(fn=anonymise_data_files, inputs=[in_data_files, in_text, anon_strat, in_colnames, in_redact_language, in_redact_entities, in_allow_list_state, text_tabular_files_done, text_output_summary, text_output_file_list_state, log_files_output_list_state, in_excel_sheets, second_loop_state, output_folder_textbox, in_deny_list_state, max_fuzzy_spelling_mistakes_num, pii_identification_method_drop_tabular, in_redact_comprehend_entities, comprehend_query_number, aws_access_key_textbox, aws_secret_key_textbox, actual_time_taken_number], outputs=[text_output_summary, text_output_file, text_output_file_list_state, text_tabular_files_done, log_files_output, log_files_output_list_state, actual_time_taken_number]).\
    # success(fn = reveal_feedback_buttons, outputs=[data_feedback_radio, data_further_details_text, data_submit_feedback_btn, data_feedback_title])

    ###
    # IDENTIFY DUPLICATE PAGES
    ###
    find_duplicate_pages_btn.click(fn=identify_similar_pages, inputs=[in_duplicate_pages, duplicate_threshold_value, output_folder_textbox], outputs=[duplicate_pages_df, duplicate_pages_out])

    ###
    # SETTINGS PAGE INPUT / OUTPUT
    ###
    # If a custom allow/deny/duplicate page list is uploaded
    in_allow_list.change(fn=custom_regex_load, inputs=[in_allow_list], outputs=[in_allow_list_text, in_allow_list_state])
    in_deny_list.change(fn=custom_regex_load, inputs=[in_deny_list, in_deny_list_text_in], outputs=[in_deny_list_text, in_deny_list_state])
    in_fully_redacted_list.change(fn=custom_regex_load, inputs=[in_fully_redacted_list, in_fully_redacted_text_in], outputs=[in_fully_redacted_list_text, in_fully_redacted_list_state])

    # The following allows for more reliable updates of the data in the custom list dataframes
    in_allow_list_state.input(update_dataframe, inputs=[in_allow_list_state], outputs=[in_allow_list_state])
    in_deny_list_state.input(update_dataframe, inputs=[in_deny_list_state], outputs=[in_deny_list_state])
    in_fully_redacted_list_state.input(update_dataframe, inputs=[in_fully_redacted_list_state], outputs=[in_fully_redacted_list_state])

    # Merge multiple review csv files together
    merge_multiple_review_files_btn.click(fn=merge_csv_files, inputs=multiple_review_files_in_out, outputs=multiple_review_files_in_out)

    #
    all_output_files_btn.click(fn=load_all_output_files, inputs=output_folder_textbox, outputs=all_output_files)
    
    ###
    # APP LOAD AND LOGGING
    ###

    # Get connection details on app load

    if SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS == "True":
        app.load(get_connection_params, inputs=[output_folder_textbox, input_folder_textbox, session_output_folder_textbox, s3_whole_document_textract_input_subfolder, s3_whole_document_textract_output_subfolder, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder], outputs=[session_hash_state, output_folder_textbox, session_hash_textbox, input_folder_textbox, s3_whole_document_textract_input_subfolder, s3_whole_document_textract_output_subfolder, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder]).\
        success(load_in_textract_job_details, inputs=[load_s3_whole_document_textract_logs_bool, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder], outputs=[textract_job_detail_df])
    else:
        app.load(get_connection_params, inputs=[output_folder_textbox, input_folder_textbox, session_output_folder_textbox, s3_whole_document_textract_input_subfolder, s3_whole_document_textract_output_subfolder, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder], outputs=[session_hash_state, output_folder_textbox, session_hash_textbox, input_folder_textbox, s3_whole_document_textract_input_subfolder, s3_whole_document_textract_output_subfolder, s3_whole_document_textract_logs_subfolder, local_whole_document_textract_logs_subfolder]) 
     

    # If relevant environment variable is set, load in the Textract job details

    # If relevant environment variable is set, load in the default allow list file from S3 or locally. Even when setting S3 path, need to local path to give a download location
    if GET_DEFAULT_ALLOW_LIST == "True" and (ALLOW_LIST_PATH or S3_ALLOW_LIST_PATH):
        if not os.path.exists(ALLOW_LIST_PATH) and S3_ALLOW_LIST_PATH and RUN_AWS_FUNCTIONS == "1":
            print("Downloading allow list from S3")
            app.load(download_file_from_s3, inputs=[s3_default_bucket, s3_default_allow_list_file, default_allow_list_output_folder_location]).\
            success(load_in_default_allow_list, inputs = [default_allow_list_output_folder_location], outputs=[in_allow_list])
            print("Successfully loaded allow list from S3")
        elif os.path.exists(ALLOW_LIST_PATH):
            print("Loading allow list from default allow list output path location:", ALLOW_LIST_PATH)
            app.load(load_in_default_allow_list, inputs = [default_allow_list_output_folder_location], outputs=[in_allow_list])
        else: print("Could not load in default allow list")

    # If relevant environment variable is set, load in the default cost code file from S3 or locally
    if GET_COST_CODES == "True" and (COST_CODES_PATH or S3_COST_CODES_PATH):
        if not os.path.exists(COST_CODES_PATH) and S3_COST_CODES_PATH and RUN_AWS_FUNCTIONS == "1":
            print("Downloading cost codes from S3")
            app.load(download_file_from_s3, inputs=[s3_default_bucket, s3_default_cost_codes_file, default_cost_codes_output_folder_location]).\
            success(load_in_default_cost_codes, inputs = [default_cost_codes_output_folder_location, default_cost_code_textbox], outputs=[cost_code_dataframe, cost_code_dataframe_base, cost_code_choice_drop])
            print("Successfully loaded cost codes from S3")
        elif os.path.exists(COST_CODES_PATH):
            print("Loading cost codes from default cost codes path location:", COST_CODES_PATH)
            app.load(load_in_default_cost_codes, inputs = [default_cost_codes_output_folder_location, default_cost_code_textbox], outputs=[cost_code_dataframe, cost_code_dataframe_base, cost_code_choice_drop])
        else: print("Could not load in cost code data")

    ###
    # LOGGING
    ###

    ### ACCESS LOGS
    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = CSVLogger_custom(dataset_file_name=log_file_name)
    access_callback.setup([session_hash_textbox, host_name_textbox], ACCESS_LOGS_FOLDER)    
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS, replacement_headers=CSV_ACCESS_LOG_HEADERS), [session_hash_textbox, host_name_textbox], None, preprocess=False).\
    success(fn = upload_log_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    ### FEEDBACK LOGS
    if DISPLAY_FILE_NAMES_IN_LOGS == 'True':
        # User submitted feedback for pdf redactions
        pdf_callback = CSVLogger_custom(dataset_file_name=log_file_name)
        pdf_callback.setup([pdf_feedback_radio, pdf_further_details_text, doc_file_name_no_extension_textbox], FEEDBACK_LOGS_FOLDER)
        pdf_submit_feedback_btn.click(lambda *args: pdf_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS, replacement_headers=CSV_FEEDBACK_LOG_HEADERS), [pdf_feedback_radio, pdf_further_details_text, doc_file_name_no_extension_textbox], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[pdf_further_details_text])

        # User submitted feedback for data redactions
        data_callback = CSVLogger_custom(dataset_file_name=log_file_name)
        data_callback.setup([data_feedback_radio, data_further_details_text, data_full_file_name_textbox], FEEDBACK_LOGS_FOLDER)
        data_submit_feedback_btn.click(lambda *args: data_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS, replacement_headers=CSV_FEEDBACK_LOG_HEADERS), [data_feedback_radio, data_further_details_text, data_full_file_name_textbox], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])
    else:
        # User submitted feedback for pdf redactions
        pdf_callback = CSVLogger_custom(dataset_file_name=log_file_name)
        pdf_callback.setup([pdf_feedback_radio, pdf_further_details_text, doc_file_name_no_extension_textbox], FEEDBACK_LOGS_FOLDER)
        pdf_submit_feedback_btn.click(lambda *args: pdf_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS, replacement_headers=CSV_FEEDBACK_LOG_HEADERS), [pdf_feedback_radio, pdf_further_details_text, placeholder_doc_file_name_no_extension_textbox_for_logs], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[pdf_further_details_text])

        # User submitted feedback for data redactions
        data_callback = CSVLogger_custom(dataset_file_name=log_file_name)
        data_callback.setup([data_feedback_radio, data_further_details_text, data_full_file_name_textbox], FEEDBACK_LOGS_FOLDER)
        data_submit_feedback_btn.click(lambda *args: data_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS, replacement_headers=CSV_FEEDBACK_LOG_HEADERS), [data_feedback_radio, data_further_details_text, placeholder_data_file_name_no_extension_textbox_for_logs], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])

    ### USAGE LOGS
    # Log processing usage - time taken for redaction queries, and also logs for queries to Textract/Comprehend

    usage_callback = CSVLogger_custom(dataset_file_name=log_file_name) 

    if DISPLAY_FILE_NAMES_IN_LOGS == 'True':
        usage_callback.setup([session_hash_textbox, doc_file_name_no_extension_textbox, data_full_file_name_textbox, total_pdf_page_count, actual_time_taken_number, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], USAGE_LOGS_FOLDER)

        latest_file_completed_text.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, doc_file_name_no_extension_textbox, data_full_file_name_textbox, total_pdf_page_count, actual_time_taken_number, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

        text_tabular_files_done.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, doc_file_name_no_extension_textbox, data_full_file_name_textbox, total_pdf_page_count, actual_time_taken_number, textract_query_number, pii_identification_method_drop_tabular, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

        successful_textract_api_call_number.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, doc_file_name_no_extension_textbox, data_full_file_name_textbox, total_pdf_page_count, actual_time_taken_number, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])
    else:
        usage_callback.setup([session_hash_textbox, blank_doc_file_name_no_extension_textbox_for_logs, blank_data_file_name_no_extension_textbox_for_logs, actual_time_taken_number, total_pdf_page_count, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], USAGE_LOGS_FOLDER)

        latest_file_completed_text.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, placeholder_doc_file_name_no_extension_textbox_for_logs, blank_data_file_name_no_extension_textbox_for_logs, actual_time_taken_number, total_pdf_page_count, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

        text_tabular_files_done.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, blank_doc_file_name_no_extension_textbox_for_logs, placeholder_data_file_name_no_extension_textbox_for_logs,  actual_time_taken_number, total_pdf_page_count, textract_query_number, pii_identification_method_drop_tabular, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

        successful_textract_api_call_number.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB, dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, placeholder_doc_file_name_no_extension_textbox_for_logs, blank_data_file_name_no_extension_textbox_for_logs, actual_time_taken_number, total_pdf_page_count, textract_query_number, pii_identification_method_drop, comprehend_query_number, cost_code_choice_drop, handwrite_signature_checkbox, host_name_textbox, text_extract_method_radio, is_a_textract_api_call], None, preprocess=False).\
        success(fn = upload_log_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

if __name__ == "__main__":
    if RUN_DIRECT_MODE == "0":
        
        if COGNITO_AUTH == "1":
            app.queue(max_size=int(MAX_QUEUE_SIZE), default_concurrency_limit=int(DEFAULT_CONCURRENCY_LIMIT)).launch(show_error=True, inbrowser=True, auth=authenticate_user, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
        else:
            app.queue(max_size=int(MAX_QUEUE_SIZE), default_concurrency_limit=int(DEFAULT_CONCURRENCY_LIMIT)).launch(show_error=True, inbrowser=True, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
    
    else:
        from tools.cli_redact import main

        main(first_loop_state, latest_file_completed=0, redaction_output_summary_textbox="", output_file_list=None, 
         log_files_list=None, estimated_time=0, textract_metadata="", comprehend_query_num=0, 
         current_loop_page=0, page_break=False, pdf_doc_state = [], all_image_annotations = [], all_line_level_ocr_results_df = pd.DataFrame(), all_decision_process_table = pd.DataFrame(),chosen_comprehend_entities = chosen_comprehend_entities, chosen_redact_entities = chosen_redact_entities, handwrite_signature_checkbox = ["Extract handwriting", "Extract signatures"])

# AWS options - placeholder for possibility of storing data on s3 and retrieving it in app
# with gr.Tab(label="Advanced options"):
#     with gr.Accordion(label = "AWS data access", open = True):
#         aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
#         with gr.Row():
#             in_aws_file = gr.Dropdown(label="Choose file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth borough plan"])
#             load_aws_data_button = gr.Button(value="Load data from AWS", variant="secondary")
            
#         aws_log_box = gr.Textbox(label="AWS data load status")

# ### Loading AWS data ###
# load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_doc_files, aws_log_box])  