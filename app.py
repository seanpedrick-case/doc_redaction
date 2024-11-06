import os
import socket

# By default TLDExtract will try to pull files from the internet. I have instead downloaded this file locally to avoid the requirement for an internet connection.
os.environ['TLDEXTRACT_CACHE'] = 'tld/.tld_set_snapshot'

import gradio as gr
import pandas as pd
from datetime import datetime
from gradio_image_annotation import image_annotator

from tools.helper_functions import ensure_output_folder_exists, add_folder_to_path, put_columns_in_df, get_connection_params, output_folder, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, custom_regex_load, reset_state_vars, load_in_default_allow_list
from tools.aws_functions import upload_file_to_s3, download_file_from_s3, RUN_AWS_FUNCTIONS, bucket_name
from tools.file_redaction import choose_and_run_redactor
from tools.file_conversion import prepare_image_or_pdf, get_input_file_names
from tools.redaction_review import apply_redactions, crop, get_boxes_json, modify_existing_page_redactions, decrease_page, increase_page, update_annotator
from tools.data_anonymise import anonymise_data_files
from tools.auth import authenticate_user


today_rev = datetime.now().strftime("%Y%m%d")

add_folder_to_path("tesseract/")
add_folder_to_path("poppler/poppler-24.02.0/Library/bin/")

ensure_output_folder_exists()

chosen_comprehend_entities = ['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE', 'PASSPORT_NUMBER','DRIVER_ID', 'USERNAME','PASSWORD', 'IP_ADDRESS','MAC_ADDRESS', 'LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER', 'INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER']

full_comprehend_entity_list = ['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER','CREDIT_DEBIT_CVV','CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS','NAME','PHONE','SSN','DATE_TIME','PASSPORT_NUMBER','DRIVER_ID','URL','AGE','USERNAME','PASSWORD','AWS_ACCESS_KEY','AWS_SECRET_KEY','IP_ADDRESS','MAC_ADDRESS','ALL','LICENSE_PLATE','VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER','CA_SOCIAL_INSURANCE_NUMBER','US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER','UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER','IN_PERMANENT_ACCOUNT_NUMBER','IN_NREGA','INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE','UK_NATIONAL_HEALTH_SERVICE_NUMBER','CA_HEALTH_NUMBER','IN_AADHAAR','IN_VOTER_NUMBER']

chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE"]

full_entity_list = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS']

language = 'en'

host_name = socket.gethostname()
feedback_logs_folder = 'feedback/' + today_rev + '/' + host_name + '/'
access_logs_folder = 'logs/' + today_rev + '/' + host_name + '/'
usage_logs_folder = 'usage/' + today_rev + '/' + host_name + '/'

text_ocr_option = "Simple text analysis - PDFs with selectable text"
tesseract_ocr_option = "Quick image analysis - typed text"
textract_option = "Complex image analysis - docs with handwriting/signatures (AWS Textract)"

local_pii_detector = "Local"
aws_pii_detector  = "AWS Comprehend"

if RUN_AWS_FUNCTIONS == "1":
    default_ocr_val = textract_option
    default_pii_detector = local_pii_detector
else:
    default_ocr_val = text_ocr_option
    default_pii_detector = local_pii_detector

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Base())

with app:

    ###
    # STATE VARIABLES
    ###

    pdf_doc_state = gr.State([])    
    all_image_annotations_state = gr.State([])
    all_line_level_ocr_results_df_state = gr.State(pd.DataFrame())
    all_decision_process_table_state = gr.State(pd.DataFrame())

    in_allow_list_state = gr.State(pd.DataFrame())

    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    first_loop_state = gr.State(True)
    second_loop_state = gr.State(False)

    prepared_pdf_state = gr.State([])
    images_pdf_state = gr.State([]) # List of pdf pages converted to PIL images
    
    output_image_files_state = gr.State([])
    output_file_list_state = gr.State([])
    text_output_file_list_state = gr.State([])
    log_files_output_list_state = gr.State([])
    
    # Logging state
    log_file_name = 'log.csv'

    feedback_logs_state = gr.State(feedback_logs_folder + log_file_name)
    feedback_s3_logs_loc_state = gr.State(feedback_logs_folder)
    access_logs_state = gr.State(access_logs_folder + log_file_name)
    access_s3_logs_loc_state = gr.State(access_logs_folder)
    usage_logs_state = gr.State(usage_logs_folder + log_file_name)
    usage_s3_logs_loc_state = gr.State(usage_logs_folder)    
    
    # Invisible text boxes to hold the session hash/username, Textract request metadata, data file names just for logging purposes.
    session_hash_textbox = gr.Textbox(label= "session_hash_textbox", value="", visible=False)
    textract_metadata_textbox = gr.Textbox(label = "textract_metadata_textbox", value="", visible=False)
    comprehend_query_number = gr.Number(label = "comprehend_query_number", value=0, visible=False)

    doc_file_name_textbox = gr.Textbox(label = "doc_file_name_textbox", value="", visible=False)
    doc_file_name_with_extension_textbox = gr.Textbox(label = "doc_file_name_with_extension_textbox", value="", visible=False)
    data_file_name_textbox = gr.Textbox(label = "data_file_name_textbox", value="", visible=False)
    
    estimated_time_taken_number = gr.Number(label = "estimated_time_taken_number", value=0.0, precision=1, visible=False) # This keeps track of the time taken to redact files for logging purposes.
    annotate_previous_page = gr.Number(value=0, label="Previous page", precision=0, visible=False) # Keeps track of the last page that the annotator was on

    s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)

    ## S3 default bucket and allow list file state
    default_allow_list_file_name = "default_allow_list.csv"
    default_allow_list_loc = output_folder + "/" + default_allow_list_file_name

    s3_default_bucket = gr.Textbox(label = "Default S3 bucket", value=bucket_name, visible=False)
    s3_default_allow_list_file = gr.Textbox(label = "Default allow list file", value=default_allow_list_file_name, visible=False)
    default_allow_list_output_folder_location = gr.Textbox(label = "Output default allow list location", value=default_allow_list_loc, visible=False)


    ###
    # UI DESIGN
    ###

    gr.Markdown(
    """# Document redaction

    Redact personally identifiable information (PII) from documents (pdf, images), open text, or tabular data (xlsx/csv/parquet). Documents/images can be redacted using 'Quick' image analysis that works fine for typed text, but not handwriting/signatures. On the Redaction settings tab, choose 'Complex image analysis' OCR using AWS Textract (if you are using AWS) to redact these more complex elements (this service has a cost). Addtionally you can choose the method for PII identification. 'Local' gives quick, lower quality results, AWS Comprehend gives better results but has a cost.
    
    See the 'Redaction settings' tab to choose which pages to redact, the type of information to redact (e.g. people, places), or terms to exclude from redaction.

    You can also review suggested redactions on the 'Review redactions' tab using a point and click visual interface. Please see the [User Guide](https://github.com/seanpedrick-case/doc_redaction/blob/main/README.md) for a walkthrough on how to use this and all other features in the app.

    NOTE: In testing the app seems to find about 60% of personal information on a given (typed) page of text. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.

    This app accepts a maximum file size of 100mb. Please consider giving feedback for the quality of the answers underneath the redact buttons when the option appears, this will help to improve the app.""")

    # PDF / IMAGES TAB
    with gr.Tab("PDFs/images"):
        with gr.Accordion("Redact document", open = True):
            in_doc_files = gr.File(label="Choose a document or image file (PDF, JPG, PNG)", file_count= "single", file_types=['.pdf', '.jpg', '.png', '.json'])
            in_redaction_method = gr.Radio(label="Choose document redaction method. AWS Textract has a cost per page so please only use when needed.", value = default_ocr_val, choices=[text_ocr_option, tesseract_ocr_option, textract_option])
            pii_identification_method_drop = gr.Radio(label = "Choose PII detection method", value = default_pii_detector, choices=[local_pii_detector, aws_pii_detector])

            gr.Markdown("""If you only want to redact certain pages, or certain entities (e.g. just email addresses), please go to the redaction settings tab.""")
            document_redact_btn = gr.Button("Redact document(s)", variant="primary")
            current_loop_page_number = gr.Number(value=0,precision=0, interactive=False, label = "Last redacted page in document", visible=False)
            page_break_return = gr.Checkbox(value = False, label="Page break reached", visible=False)
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary", scale=1)
            output_file = gr.File(label="Output files", scale = 2)
            latest_file_completed_text = gr.Number(value=0, label="Number of documents redacted", interactive=False, visible=False)

        with gr.Row():
            convert_text_pdf_to_img_btn = gr.Button(value="Convert pdf to image-based pdf to apply redactions", variant="secondary", visible=False)

        # Feedback elements are invisible until revealed by redaction action
        pdf_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        pdf_feedback_radio = gr.Radio(label = "Quality of results", choices=["The results were good", "The results were not good"], visible=False)
        pdf_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        pdf_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)
        
    # Object annotation
    with gr.Tab("Review redactions", id="tab_object_annotation"):

        with gr.Row():
            annotation_last_page_button = gr.Button("Previous page", scale = 3)
            annotate_current_page = gr.Number(value=1, label="Page (press enter to change)", precision=0, scale = 2)
            annotate_max_pages = gr.Number(value=1, label="Total pages", precision=0, interactive=False, scale = 1)
            annotation_next_page_button = gr.Button("Next page", scale = 3)

        annotation_button_apply = gr.Button("Apply revised redactions", variant="primary")

        annotator = image_annotator(
            label="Modify redaction boxes",
            label_list=["Redaction"],
            label_colors=[(0, 0, 0)],
            show_label=False,
            sources=None,#["upload"],
            show_clear_button=False,
            show_share_button=False,
            show_remove_button=False,
            interactive=False
        )

        with gr.Row():
            annotation_last_page_button_bottom = gr.Button("Previous page", scale = 3)
            annotate_current_page_bottom = gr.Number(value=1, label="Page (press enter to change)", precision=0, interactive=True, scale = 2)
            annotate_max_pages_bottom = gr.Number(value=1, label="Total pages", precision=0, interactive=False, scale = 1)
            annotation_next_page_button_bottom = gr.Button("Next page", scale = 3)

        output_review_files = gr.File(label="Review output files")

    # TEXT / TABULAR DATA TAB
    with gr.Tab(label="Open text or Excel/csv files"):
        gr.Markdown(
    """
    ### Choose open text or a tabular data file (xlsx or csv) to redact.
    """
        )    
        with gr.Accordion("Paste open text", open = False):
            in_text = gr.Textbox(label="Enter open text", lines=10)
        with gr.Accordion("Upload xlsx or csv files", open = True):
            in_data_files = gr.File(label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
        
        in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheets to anonymise"], multiselect = True, label="Select Excel sheets that you want to anonymise (showing sheets present across all Excel files).", visible=False, allow_custom_value=True)

        in_colnames = gr.Dropdown(choices=["Choose columns to anonymise"], multiselect = True, label="Select columns that you want to anonymise (showing columns present across all files).")
        
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

    # SETTINGS TAB
    with gr.Tab(label="Redaction settings"):
        gr.Markdown(
    """
    Define redaction settings that affect both document and open text redaction.
    """)
        with gr.Accordion("Settings for documents", open = True):
            
            with gr.Row():
                page_min = gr.Number(precision=0,minimum=0,maximum=9999, label="Lowest page to redact")
                page_max = gr.Number(precision=0,minimum=0,maximum=9999, label="Highest page to redact")
            
            
        with gr.Accordion("Settings for documents and open text/xlsx/csv files", open = True):
            with gr.Row():
                in_allow_list = gr.File(label="Import allow list file", file_count="multiple")
                with gr.Column():   
                    gr.Markdown("""Import allow list file - csv table with one column of a different word/phrase on each row (case sensitive). Terms in this file will not be redacted.""")
                    in_allow_list_text = gr.Textbox(label="Custom allow list load status")

            with gr.Accordion("Add or remove entity types to redact", open = False):
                in_redact_entities = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Entities to redact - local PII identification model (click close to down arrow for full list)")
                
                in_redact_comprehend_entities = gr.Dropdown(value=chosen_comprehend_entities, choices=full_comprehend_entity_list, multiselect=True, label="Entities to redact - AWS Comprehend PII identification model (click close to down arrow for full list)")
            
            handwrite_signature_checkbox = gr.CheckboxGroup(label="AWS Textract settings", choices=["Redact all identified handwriting", "Redact all identified signatures"], value=["Redact all identified handwriting", "Redact all identified signatures"])
            #with gr.Row():
            in_redact_language = gr.Dropdown(value = "en", choices = ["en"], label="Redaction language (only English currently supported)", multiselect=False, visible=False)
                

        with gr.Accordion("Settings for open text or xlsx/csv files", open = True):
            anon_strat = gr.Radio(choices=["replace with <REDACTED>", "replace with <ENTITY_NAME>", "redact", "hash", "mask", "encrypt", "fake_first_name"], label="Select an anonymisation method.", value = "replace with <REDACTED>")
            
        log_files_output = gr.File(label="Log file output", interactive=False)

    # If a custom allow list is uploaded
    in_allow_list.change(fn=custom_regex_load, inputs=[in_allow_list], outputs=[in_allow_list_text, in_allow_list_state])

    ###
    # PDF/IMAGE REDACTION
    ###
    in_doc_files.upload(fn=get_input_file_names, inputs=[in_doc_files], outputs=[doc_file_name_textbox, doc_file_name_with_extension_textbox])

    document_redact_btn.click(fn = reset_state_vars, outputs=[pdf_doc_state, all_image_annotations_state, all_line_level_ocr_results_df_state, all_decision_process_table_state, comprehend_query_number, textract_metadata_textbox]).\
    then(fn = prepare_image_or_pdf, inputs=[in_doc_files, in_redaction_method, in_allow_list, latest_file_completed_text, output_summary, first_loop_state, annotate_max_pages, current_loop_page_number], outputs=[output_summary, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state], api_name="prepare_doc").\
    then(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, in_redaction_method, in_allow_list_state, latest_file_completed_text, output_summary, output_file_list_state, log_files_output_list_state, first_loop_state, page_min, page_max, estimated_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_state, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, pii_identification_method_drop, comprehend_query_number],
                    outputs=[output_summary, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, estimated_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_state, all_decision_process_table_state, comprehend_query_number], api_name="redact_doc")#.\
                    #then(fn=update_annotator, inputs=[all_image_annotations_state, page_min], outputs=[annotator, annotate_current_page])
    
    # If the app has completed a batch of pages, it will run this until the end of all pages in the document
    current_loop_page_number.change(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redact_comprehend_entities, in_redaction_method, in_allow_list_state, latest_file_completed_text, output_summary, output_file_list_state, log_files_output_list_state, second_loop_state, page_min, page_max, estimated_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_state, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return, pii_identification_method_drop, comprehend_query_number],
                    outputs=[output_summary, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, estimated_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_state, all_decision_process_table_state, comprehend_query_number])
    
    # If a file has been completed, the function will continue onto the next document
    latest_file_completed_text.change(fn=update_annotator, inputs=[all_image_annotations_state, page_min], outputs=[annotator, annotate_current_page, annotate_current_page_bottom]).\
                    then(fn=reveal_feedback_buttons, outputs=[pdf_feedback_radio, pdf_further_details_text, pdf_submit_feedback_btn, pdf_feedback_title])
        # latest_file_completed_text.change(fn = prepare_image_or_pdf, inputs=[in_doc_files, in_redaction_method, in_allow_list, latest_file_completed_text, output_summary, second_loop_state, annotate_max_pages, current_loop_page_number], outputs=[output_summary, prepared_pdf_state, images_pdf_state, annotate_max_pages, annotate_max_pages_bottom, pdf_doc_state]).\
    # then(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redaction_method, in_allow_list_state, latest_file_completed_text, output_summary, output_file_list_state, log_files_output_list_state, second_loop_state, page_min, page_max, estimated_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, all_line_level_ocr_results_df_state, all_decision_process_table_state, pdf_doc_state, current_loop_page_number, page_break_return],
                    # outputs=[output_summary, output_file, output_file_list_state, latest_file_completed_text, log_files_output, log_files_output_list_state, estimated_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state, current_loop_page_number, page_break_return, all_line_level_ocr_results_df_state, all_decision_process_table_state]).\
                    #then(fn=update_annotator, inputs=[all_image_annotations_state, page_min], outputs=[annotator, annotate_current_page]).\
                    #then(fn=reveal_feedback_buttons, outputs=[pdf_feedback_radio, pdf_further_details_text, pdf_submit_feedback_btn, pdf_feedback_title])
    
    ### REVIEW REDACTIONS

    # Page controls at top
    annotate_current_page.submit(
        modify_existing_page_redactions, inputs = [annotator, annotate_current_page, annotate_previous_page, all_image_annotations_state], outputs = [all_image_annotations_state, annotate_previous_page, annotate_current_page_bottom]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])
    
    annotation_last_page_button.click(fn=decrease_page, inputs=[annotate_current_page], outputs=[annotate_current_page, annotate_current_page_bottom]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])
    annotation_next_page_button.click(fn=increase_page, inputs=[annotate_current_page, all_image_annotations_state], outputs=[annotate_current_page, annotate_current_page_bottom]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])

    #annotation_button_get.click(get_boxes_json, annotator, json_boxes)
    annotation_button_apply.click(apply_redactions, inputs=[annotator, in_doc_files, pdf_doc_state, all_image_annotations_state, annotate_current_page], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files], scroll_to_output=True)

    # Page controls at bottom
    annotate_current_page_bottom.submit(
        modify_existing_page_redactions, inputs = [annotator, annotate_current_page_bottom, annotate_previous_page, all_image_annotations_state], outputs = [all_image_annotations_state, annotate_previous_page, annotate_current_page]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])

    annotation_last_page_button_bottom.click(fn=decrease_page, inputs=[annotate_current_page], outputs=[annotate_current_page, annotate_current_page_bottom]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])
    annotation_next_page_button_bottom.click(fn=increase_page, inputs=[annotate_current_page, all_image_annotations_state], outputs=[annotate_current_page, annotate_current_page_bottom]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page, annotate_current_page_bottom])

    ###
    # TABULAR DATA REDACTION
    ###            
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets]).\
                  then(fn=get_input_file_names, inputs=[in_data_files], outputs=[data_file_name_textbox])

    tabular_data_redact_btn.click(fn=anonymise_data_files, inputs=[in_data_files, in_text, anon_strat, in_colnames, in_redact_language, in_redact_entities, in_allow_list, text_tabular_files_done, text_output_summary, text_output_file_list_state, log_files_output_list_state, in_excel_sheets, first_loop_state], outputs=[text_output_summary, text_output_file, text_output_file_list_state, text_tabular_files_done, log_files_output, log_files_output_list_state], api_name="redact_data")

    # If the output file count text box changes, keep going with redacting each data file until done
    text_tabular_files_done.change(fn=anonymise_data_files, inputs=[in_data_files, in_text, anon_strat, in_colnames, in_redact_language, in_redact_entities, in_allow_list, text_tabular_files_done, text_output_summary, text_output_file_list_state, log_files_output_list_state, in_excel_sheets, second_loop_state], outputs=[text_output_summary, text_output_file, text_output_file_list_state, text_tabular_files_done, log_files_output, log_files_output_list_state]).\
    then(fn = reveal_feedback_buttons, outputs=[data_feedback_radio, data_further_details_text, data_submit_feedback_btn, data_feedback_title])    

    ###
    # APP LOAD AND LOGGING
    ###

    # Get connection details on app load
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox])

    # If running on AWS, load in the default allow list file from S3
    if RUN_AWS_FUNCTIONS == "1":
        print("default_allow_list_output_folder_location:", default_allow_list_output_folder_location)
        if not os.path.exists(default_allow_list_loc):
            app.load(download_file_from_s3, inputs=[s3_default_bucket, s3_default_allow_list_file, default_allow_list_output_folder_location]).\
            then(load_in_default_allow_list, inputs = [default_allow_list_output_folder_location], outputs=[in_allow_list])
        else:
            app.load(load_in_default_allow_list, inputs = [default_allow_list_output_folder_location], outputs=[in_allow_list])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    access_callback.setup([session_hash_textbox], access_logs_folder)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # User submitted feedback for pdf redactions
    pdf_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    pdf_callback.setup([pdf_feedback_radio, pdf_further_details_text, doc_file_name_textbox], feedback_logs_folder)
    pdf_submit_feedback_btn.click(lambda *args: pdf_callback.flag(list(args)), [pdf_feedback_radio, pdf_further_details_text, doc_file_name_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[pdf_further_details_text])

    # User submitted feedback for data redactions
    data_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    data_callback.setup([data_feedback_radio, data_further_details_text, data_file_name_textbox], feedback_logs_folder)
    data_submit_feedback_btn.click(lambda *args: data_callback.flag(list(args)), [data_feedback_radio, data_further_details_text, data_file_name_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])

    # Log processing time/token usage when making a query
    usage_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    usage_callback.setup([session_hash_textbox, doc_file_name_textbox, data_file_name_textbox, estimated_time_taken_number, textract_metadata_textbox, pii_identification_method_drop, comprehend_query_number], usage_logs_folder)
    estimated_time_taken_number.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, doc_file_name_textbox, data_file_name_textbox, estimated_time_taken_number, textract_metadata_textbox, pii_identification_method_drop, comprehend_query_number], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

# Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')

if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        app.queue(max_size=5).launch(show_error=True, auth=authenticate_user, max_file_size='100mb')
    else:
        app.queue(max_size=5).launch(show_error=True, inbrowser=True, max_file_size='100mb')


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