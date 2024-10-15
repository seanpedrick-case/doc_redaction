import os
import socket

# By default TLDExtract will try to pull files from the internet. I have instead downloaded this file locally to avoid the requirement for an internet connection.
os.environ['TLDEXTRACT_CACHE'] = 'tld/.tld_set_snapshot'

from gradio_image_annotation import image_annotator

from tools.helper_functions import ensure_output_folder_exists, add_folder_to_path, put_columns_in_df, get_connection_params, output_folder, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, custom_regex_load
from tools.aws_functions import upload_file_to_s3
from tools.file_redaction import choose_and_run_redactor
from tools.file_conversion import prepare_image_or_pdf, get_input_file_names
from tools.redaction_review import apply_redactions, crop, get_boxes_json, modify_existing_page_redactions, decrease_page, increase_page, update_annotator
from tools.data_anonymise import anonymise_data_files
from tools.auth import authenticate_user
#from tools.aws_functions import load_data_from_aws
import gradio as gr
import pandas as pd

from datetime import datetime
today_rev = datetime.now().strftime("%Y%m%d")

add_folder_to_path("tesseract/")
add_folder_to_path("poppler/poppler-24.02.0/Library/bin/")

ensure_output_folder_exists()

chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE"] 
full_entity_list = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS']
language = 'en'

host_name = socket.gethostname()

feedback_logs_folder = 'feedback/' + today_rev + '/' + host_name + '/'
access_logs_folder = 'logs/' + today_rev + '/' + host_name + '/'
usage_logs_folder = 'usage/' + today_rev + '/' + host_name + '/'

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Base())

with app:

    ###
    # STATE VARIABLES
    ###
    prepared_pdf_state = gr.State([])
    output_image_files_state = gr.State([])
    output_file_list_state = gr.State([])
    text_output_file_list_state = gr.State([])
    log_files_output_list_state = gr.State([]) 
    first_loop_state = gr.State(True)
    second_loop_state = gr.State(False)

    in_allow_list_state = gr.State(pd.DataFrame())

    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    pdf_doc_state = gr.State([])
    images_pdf_state = gr.State([]) # List of pdf pages converted to PIL images
    all_image_annotations_state = gr.State([])

    # Logging state
    feedback_logs_state = gr.State(feedback_logs_folder + 'log.csv')
    feedback_s3_logs_loc_state = gr.State(feedback_logs_folder)
    access_logs_state = gr.State(access_logs_folder + 'log.csv')
    access_s3_logs_loc_state = gr.State(access_logs_folder)
    usage_logs_state = gr.State(usage_logs_folder + 'log.csv')
    usage_s3_logs_loc_state = gr.State(usage_logs_folder)    

    # Invisible elements effectively used as state variables
    session_hash_textbox = gr.Textbox(value="", visible=False) # Invisible text box to hold the session hash/username, Textract request metadata, data file names just for logging purposes.
    textract_metadata_textbox = gr.Textbox(value="", visible=False)
    doc_file_name_textbox = gr.Textbox(value="", visible=False)
    doc_file_name_with_extension_textbox = gr.Textbox(value="", visible=False)
    data_file_name_textbox = gr.Textbox(value="", visible=False)
    s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)
    estimated_time_taken_number = gr.Number(value=0.0, precision=1, visible=False) # This keeps track of the time taken to redact files for logging purposes.
    annotate_previous_page = gr.Number(value=0, label="Previous page", precision=0, visible=False) # Keeps track of the last page that the annotator was on


    ###
    # UI DESIGN
    ###

    gr.Markdown(
    """
    # Document redaction

    Redact personal information from documents (pdf, images), open text, or tabular data (xlsx/csv/parquet). Documents/images can be redacted using 'Quick' image analysis that works fine for typed text, but not handwriting/signatures. On the Redaction settings tab, choose 'Complex image analysis' OCR using AWS Textract (if you are using AWS) to redact these more complex elements (this service has a cost, so please only use for more complex redaction tasks). Also see the 'Redaction settings' tab to choose which pages to redact, the type of information to redact (e.g. people, places), or terms to exclude from redaction.

    NOTE: In testing the app seems to find about 60% of personal information on a given (typed) page of text. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.

    This app accepts a maximum file size of 50mb. Please consider giving feedback for the quality of the answers underneath the redact buttons when the option appears, this will help to improve the app.
    """)

    # PDF / IMAGES TAB
    with gr.Tab("PDFs/images"):
        with gr.Accordion("Redact document", open = True):
            in_doc_files = gr.File(label="Choose document/image files (PDF, JPG, PNG)", file_count= "multiple", file_types=['.pdf', '.jpg', '.png', '.json'])
            in_redaction_method = gr.Radio(label="Choose document redaction method. AWS Textract has a cost per page so please only use when needed.", value = "Simple text analysis - PDFs with selectable text", choices=["Simple text analysis - PDFs with selectable text", "Quick image analysis - typed text", "Complex image analysis - docs with handwriting/signatures (AWS Textract)"])
            gr.Markdown("""If you only want to redact certain pages, or certain entities (e.g. just email addresses), please go to the redaction settings tab.""")
            document_redact_btn = gr.Button("Redact document(s)", variant="primary")
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary")
            output_file = gr.File(label="Output files")
            text_documents_done = gr.Number(value=0, label="Number of documents redacted", interactive=False, visible=False)

        with gr.Row():
            convert_text_pdf_to_img_btn = gr.Button(value="Convert pdf to image-based pdf to apply redactions", variant="secondary", visible=False)

        # Feedback elements are invisible until revealed by redaction action
        pdf_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        pdf_feedback_radio = gr.Radio(choices=["The results were good", "The results were not good"], visible=False)
        pdf_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        pdf_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)
        
    # Object annotation
    with gr.Tab("Review redactions", id="tab_object_annotation"):

        with gr.Row():
            annotation_last_page_button = gr.Button("Previous page")
            annotate_current_page = gr.Number(value=1, label="Current page (select page number then press enter)", precision=0)
            
            annotation_next_page_button = gr.Button("Next page")

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
                choices=["The results were good", "The results were not good"], visible=False)
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
            with gr.Row():
                handwrite_signature_checkbox = gr.CheckboxGroup(label="AWS Textract settings", choices=["Redact all identified handwriting", "Redact all identified signatures"], value=["Redact all identified handwriting", "Redact all identified signatures"])
        with gr.Accordion("Settings for open text or xlsx/csv files", open = True):
            anon_strat = gr.Radio(choices=["replace with <REDACTED>", "replace with <ENTITY_NAME>", "redact", "hash", "mask", "encrypt", "fake_first_name"], label="Select an anonymisation method.", value = "replace with <REDACTED>") 

        with gr.Accordion("Settings for documents and open text/xlsx/csv files", open = True):
            in_redact_entities = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Entities to redact (click close to down arrow for full list)")
            with gr.Row():
                in_redact_language = gr.Dropdown(value = "en", choices = ["en"], label="Redaction language (only English currently supported)", multiselect=False)
                # Upload 'Allow list' for terms not to be redacted
                with gr.Row():
                    in_allow_list = gr.UploadButton(label="Import allow list file.", file_count="multiple")    
                    gr.Markdown("""Import allow list file - csv table with one column of a different word/phrase on each row (case sensitive). Terms in this file will not be redacted.""")
                    in_allow_list_text = gr.Textbox(label="Custom allow list load status")
            log_files_output = gr.File(label="Log file output", interactive=False)

    # If a custom allow list is uploaded
    in_allow_list.upload(fn=custom_regex_load, inputs=[in_allow_list], outputs=[in_allow_list_text, in_allow_list_state])
   
    ###
    # PDF/IMAGE REDACTION
    ###
    in_doc_files.upload(fn=get_input_file_names, inputs=[in_doc_files], outputs=[doc_file_name_textbox, doc_file_name_with_extension_textbox])

    document_redact_btn.click(fn = prepare_image_or_pdf, inputs=[in_doc_files, in_redaction_method, in_allow_list, text_documents_done, output_summary, first_loop_state], outputs=[output_summary, prepared_pdf_state, images_pdf_state], api_name="prepare_doc").\
    then(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redaction_method, in_allow_list_state, text_documents_done, output_summary, output_file_list_state, log_files_output_list_state, first_loop_state, page_min, page_max, estimated_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, pdf_doc_state],
                    outputs=[output_summary, output_file, output_file_list_state, text_documents_done, log_files_output, log_files_output_list_state, estimated_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state], api_name="redact_doc").\
                    then(fn=update_annotator, inputs=[all_image_annotations_state, page_min], outputs=[annotator, annotate_current_page])
    
    # If the output file count text box changes, keep going with redacting each document until done
    text_documents_done.change(fn = prepare_image_or_pdf, inputs=[in_doc_files, in_redaction_method, in_allow_list, text_documents_done, output_summary, second_loop_state], outputs=[output_summary, prepared_pdf_state, images_pdf_state]).\
    then(fn = choose_and_run_redactor, inputs=[in_doc_files, prepared_pdf_state, images_pdf_state, in_redact_language, in_redact_entities, in_redaction_method, in_allow_list_state, text_documents_done, output_summary, output_file_list_state, log_files_output_list_state, second_loop_state, page_min, page_max, estimated_time_taken_number, handwrite_signature_checkbox, textract_metadata_textbox, all_image_annotations_state, pdf_doc_state],
                    outputs=[output_summary, output_file, output_file_list_state, text_documents_done, log_files_output, log_files_output_list_state, estimated_time_taken_number, textract_metadata_textbox, pdf_doc_state, all_image_annotations_state]).\
                    then(fn=update_annotator, inputs=[all_image_annotations_state, page_min], outputs=[annotator, annotate_current_page]).\
                    then(fn = reveal_feedback_buttons, outputs=[pdf_feedback_radio, pdf_further_details_text, pdf_submit_feedback_btn, pdf_feedback_title])
    
    annotate_current_page.submit(
        modify_existing_page_redactions, inputs = [annotator, annotate_current_page, annotate_previous_page, all_image_annotations_state], outputs = [all_image_annotations_state, annotate_previous_page]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page])

    annotation_last_page_button.click(fn=decrease_page, inputs=[annotate_current_page], outputs=[annotate_current_page]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page])
    annotation_next_page_button.click(fn=increase_page, inputs=[annotate_current_page, all_image_annotations_state], outputs=[annotate_current_page]).\
        then(update_annotator, inputs=[all_image_annotations_state, annotate_current_page], outputs = [annotator, annotate_current_page])

    #annotation_button_get.click(get_boxes_json, annotator, json_boxes)
    annotation_button_apply.click(apply_redactions, inputs=[annotator, in_doc_files, pdf_doc_state, all_image_annotations_state, annotate_current_page], outputs=[pdf_doc_state, all_image_annotations_state, output_review_files], scroll_to_output=True)

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

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger()
    access_callback.setup([session_hash_textbox], access_logs_folder)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # User submitted feedback for pdf redactions
    pdf_callback = gr.CSVLogger()
    pdf_callback.setup([pdf_feedback_radio, pdf_further_details_text, in_doc_files], feedback_logs_folder)
    pdf_submit_feedback_btn.click(lambda *args: pdf_callback.flag(list(args)), [pdf_feedback_radio, pdf_further_details_text, in_doc_files], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[pdf_further_details_text])

    # User submitted feedback for data redactions
    data_callback = gr.CSVLogger()
    data_callback.setup([data_feedback_radio, data_further_details_text, in_data_files], feedback_logs_folder)
    data_submit_feedback_btn.click(lambda *args: data_callback.flag(list(args)), [data_feedback_radio, data_further_details_text, in_data_files], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])

    # Log processing time/token usage when making a query
    usage_callback = gr.CSVLogger()
    usage_callback.setup([session_hash_textbox, doc_file_name_textbox, data_file_name_textbox, estimated_time_taken_number, textract_metadata_textbox], usage_logs_folder)
    estimated_time_taken_number.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, doc_file_name_textbox, data_file_name_textbox, estimated_time_taken_number, textract_metadata_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

# Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')

if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        app.queue().launch(show_error=True, auth=authenticate_user, max_file_size='50mb')
    else:
        app.queue().launch(show_error=True, inbrowser=True, max_file_size='50mb')


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