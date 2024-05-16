from tools.file_redaction import choose_and_run_redactor
from tools.file_conversion import prepare_image_or_text_pdf, convert_text_pdf_to_img_pdf
from tools.aws_functions import load_data_from_aws
from typing import List
import pandas as pd
import gradio as gr

#file_path = "examples/Lambeth_2030-Our_Future_Our_Lambeth_foreword.pdf" #"examples/skills-based-cv-example.pdf" # "examples/graduate-job-example-cover-letter.pdf" #

chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE"] 
full_entity_list = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS']
language = 'en'


# Create the gradio interface

block = gr.Blocks(theme = gr.themes.Base())

with block:

    prepared_pdf_state = gr.State([])
    output_image_files_state = gr.State([])
    output_file_list_state = gr.State([])

    gr.Markdown(
    """
    # Document redaction
    Take an image-based PDF or image file, or text-based PDF document and redact any personal information. 'Image analysis' will convert PDF pages to image and the identify text via OCR methods before redaction, and also works with JPG or PNG files. 'Text analysis' will analyse only selectable text that exists in the original PDF before redaction. Choose 'Image analysis' if you are not sure of the type of PDF document you are working with.

    WARNING: This is a beta product. It is not 100% accurate, and it will miss some personal information. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed. Also, the output from the Text analysis ending 'as_text.pdf' is an annotated pdf, which is a layer on top of the text that can be removed. So the text has not truly been redacted. Use the '...as_img.pdf' versions instead for safer redaction.

    Other redaction entities are possible to include in this app easily, especially country-specific entities. If you want to use these, clone the repo locally and add entity names from [this link](https://microsoft.github.io/presidio/supported_entities/) to the 'full_entity_list' variable in app.py.
    """)

    with gr.Tab("Redact document"):
    
        with gr.Accordion("Input document", open = True):
            in_file = gr.File(label="Choose document file", file_count= "single")
            in_redaction_method = gr.Radio(label="Redaction method - text analysis is faster but will fail on images or image-based PDFs.", value = "Text analysis", choices=["Text analysis", "Image analysis"])
            in_redact_entities = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Entities to redact (click close to down arrow for full list)")
            in_redact_language = gr.Dropdown(value = "en", choices = ["en"], label="Redaction language", multiselect=False)
            in_allow_list = gr.Dataframe(label="Allow list - enter a new term to ignore for redaction on each row e.g. Lambeth -> add new row -> Lambeth 2030", headers=["Allow list"], row_count=1, col_count=1, value=[[""]], type="array", column_widths=["50%"])
        
        redact_btn = gr.Button("Redact document")
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary")
            output_file = gr.File(label="Output file")

        with gr.Row():
            convert_text_pdf_to_img_btn = gr.Button(value="Convert pdf to image-based pdf to apply redactions", variant="secondary")

    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label = "AWS data access", open = True):
                aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
                with gr.Row():
                    in_aws_file = gr.Dropdown(label="Choose keyword file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth borough plan"])
                    load_aws_data_button = gr.Button(value="Load keyword data from AWS", variant="secondary")
                    
                aws_log_box = gr.Textbox(label="AWS data load status")
    
    ### Loading AWS data ###
    load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_file, aws_log_box])
    
    redact_btn.click(fn = prepare_image_or_text_pdf, inputs=[in_file, in_redaction_method, in_allow_list],
                    outputs=[output_summary, prepared_pdf_state], api_name="prepare").\
    then(fn = choose_and_run_redactor, inputs=[in_file, prepared_pdf_state, in_redact_language, in_redact_entities, in_redaction_method, in_allow_list],
                    outputs=[output_summary, output_file, output_file_list_state], api_name="redact")
    
    convert_text_pdf_to_img_btn.click(fn = convert_text_pdf_to_img_pdf, inputs=[in_file, output_file_list_state],
                    outputs=[output_summary, output_file])
    
# Simple run for HF spaces or local on your computer
#block.queue().launch(debug=True) # root_path="/address-match", debug=True, server_name="0.0.0.0",

# Simple run for AWS server
block.queue().launch(ssl_verify=False) # root_path="/address-match", debug=True, server_name="0.0.0.0", server_port=7861

# Download OpenSSL from here: 
# Running on local server with https: https://discuss.huggingface.co/t/how-to-run-gradio-with-0-0-0-0-and-https/38003 or https://dev.to/rajshirolkar/fastapi-over-https-for-development-on-windows-2p7d
#block.queue().launch(ssl_verify=False, share=False, debug=False, server_name="0.0.0.0",server_port=443,
#                     ssl_certfile="cert.pem", ssl_keyfile="key.pem") # port 443 for https. Certificates currently not valid

# Running on local server without https
#block.queue().launch(server_name="0.0.0.0", server_port=7861, ssl_verify=False)
