from tools.file_redaction import redact_text_pdf, redact_image_pdf
from tools.helper_functions import get_file_path_end
from tools.file_conversion import process_file, is_pdf
from tools.aws_functions import load_data_from_aws

from typing import List
import pandas as pd
import gradio as gr
import time

file_path = "examples/Lambeth_2030-Our_Future_Our_Lambeth_foreword.pdf" #"examples/skills-based-cv-example.pdf" # "examples/graduate-job-example-cover-letter.pdf" #

chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE"] 
full_entity_list = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE", 'CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'IBAN_CODE', 'IP_ADDRESS', 'NRP', 'LOCATION', 'MEDICAL_LICENSE', 'URL', 'UK_NHS']
language = 'en'

def choose_and_run_redactor(file_path:str, language:str, chosen_redact_entities:List[str], in_redact_method:str, in_allow_list:List[List[str]]=None, progress=gr.Progress(track_tqdm=True)):

    tic = time.perf_counter()

    if is_pdf(file_path) == False:
        return "Please upload a PDF file.", None

    out_message = ''
    out_file_paths = []

    in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    if file_path:
        file_path_without_ext = get_file_path_end(file_path)
    else:
        out_message = "No file selected"
        print(out_message)
        return out_message, out_file_paths

    if in_redact_method == "Image analysis":
        # Analyse image-based pdf
        pdf_images = redact_image_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat)
        out_image_file_path = "output/" + file_path_without_ext + "_result_as_img.pdf"
        pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])

        out_file_paths.append(out_image_file_path)
        out_message = "Image-based PDF successfully redacted and saved to file."

    elif in_redact_method == "Text analysis":
        # Analyse text-based pdf
        pdf_text = redact_text_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat)
        out_text_file_path = "output/" + file_path_without_ext + "_result_as_text.pdf"
        pdf_text.save(out_text_file_path)

        out_file_paths.append(out_text_file_path)

        # Convert annotated text pdf back to image to give genuine redactions
        pdf_text_image_paths = process_file(out_text_file_path)
        out_text_image_file_path = "output/" + file_path_without_ext + "_result_as_text_back_to_img.pdf"
        pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_text_image_paths[1:])

        out_file_paths.append(out_text_image_file_path)

        out_message = "Image-based PDF successfully redacted and saved to text-based annotated file, and image-based file."

    else:
        out_message = "No redaction method selected"
        print(out_message)
        return out_message, out_file_paths
    
    toc = time.perf_counter()
    out_time = f"Time taken: {toc - tic:0.1f} seconds."
    print(out_time)

    out_message = out_message + "\n\n" + out_time

    return out_message, out_file_paths


# Create the gradio interface

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
    ref_data_state = gr.State(pd.DataFrame())
    results_data_state = gr.State(pd.DataFrame())
    ref_results_data_state =gr.State(pd.DataFrame())

    gr.Markdown(
    """
    # Document redaction
    Take an image-based or text-based PDF document and redact any personal information. 'Image analysis' will convert PDF pages to image and the identify text via OCR methods before redaction. 'Text analysis' will analyse only selectable text that exists in the original PDF before redaction. Choose 'Image analysis' if you are not sure of the type of PDF document you are working with.

    WARNING: This is a beta product. It is not 100% accurate, and it will miss some personal information. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.
    """)

    with gr.Tab("Redact document"):
    
        with gr.Accordion("Input document", open = True):
            in_file = gr.File(label="Choose document file", file_count= "single")
            in_redaction_method = gr.Radio(label="Redaction method", value = "Image analysis", choices=["Image analysis", "Text analysis"])
            in_redact_entities = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Entities to redact (click close to down arrow for full list)")
            in_redact_language = gr.Dropdown(value = "en", choices = ["en"], label="Redaction language", multiselect=False)
            in_allow_list = gr.Dataframe(label="Allow list - enter a new term to ignore for redaction on each row e.g. Lambeth -> add new row -> Lambeth 2030", headers=["Allow list"], row_count=1, col_count=1, value=[[""]], type="array", column_widths=["50%"])
        
        redact_btn = gr.Button("Redact document")
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary")
            output_file = gr.File(label="Output file")

    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label = "AWS data access", open = False):
                aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
                with gr.Row():
                    in_aws_file = gr.Dropdown(label="Choose keyword file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth borough plan"])
                    load_aws_data_button = gr.Button(value="Load keyword data from AWS", variant="secondary")
                    
                aws_log_box = gr.Textbox(label="AWS data load status")

    
    ### Loading AWS data ###
    load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_file, aws_log_box])
    

    # Updates to components
    #in_file.change(fn = initial_data_load, inputs=[in_file], outputs=[output_summary, in_redact_entities, in_existing, data_state, results_data_state])
    #in_ref.change(fn = initial_data_load, inputs=[in_ref], outputs=[output_summary, in_refcol, in_joincol, ref_data_state, ref_results_data_state])      

    redact_btn.click(fn = choose_and_run_redactor, inputs=[in_file, in_redact_language, in_redact_entities, in_redaction_method, in_allow_list],
                    outputs=[output_summary, output_file], api_name="redact")
    
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
