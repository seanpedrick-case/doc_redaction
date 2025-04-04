import os
import re
import boto3
from botocore.exceptions import ClientError
import gradio as gr
import pandas as pd
import numpy as np
import unicodedata
from typing import List
from math import ceil
from gradio_image_annotation import image_annotator
from tools.config import CUSTOM_HEADER_VALUE, CUSTOM_HEADER, OUTPUT_FOLDER, INPUT_FOLDER, SESSION_OUTPUT_FOLDER, AWS_USER_POOL_ID

# Names for options labels
text_ocr_option = "Local model - selectable text"
tesseract_ocr_option = "Local OCR model - PDFs without selectable text"
textract_option = "AWS Textract service - all PDF types"

no_redaction_option = "Only extract text (no redaction)"
local_pii_detector = "Local"
aws_pii_detector  = "AWS Comprehend"

def reset_state_vars():
    return [], pd.DataFrame(), pd.DataFrame(), 0, "", image_annotator(
            label="Modify redaction boxes",
            label_list=["Redaction"],
            label_colors=[(0, 0, 0)],
            show_label=False,
            sources=None,#["upload"],
            show_clear_button=False,
            show_share_button=False,
            show_remove_button=False,
            interactive=False
        ), [], [], pd.DataFrame(), pd.DataFrame(), [], [], ""

def reset_ocr_results_state():
    return pd.DataFrame(), pd.DataFrame(), []

def reset_review_vars():
    return pd.DataFrame(), pd.DataFrame()

def load_in_default_allow_list(allow_list_file_path):
    if isinstance(allow_list_file_path, str):
        allow_list_file_path = [allow_list_file_path]
    return allow_list_file_path

def load_in_default_cost_codes(cost_codes_path:str):
    cost_codes_df = pd.read_csv(cost_codes_path)

    dropdown_choices = cost_codes_df.iloc[:,0].to_list()
    dropdown_choices.insert(0, "")


    out_dropdown = gr.Dropdown(value="", label="Choose cost code for analysis", choices=dropdown_choices, allow_custom_value=True)
    
    return cost_codes_df, out_dropdown

def enforce_cost_codes(enforce_cost_code_textbox, cost_code_choice):
    if enforce_cost_code_textbox == "True":
        if not cost_code_choice:
            raise Exception("Please choose a cost code before continuing")
    return

def update_dataframe(df:pd.DataFrame):
    df_copy = df.copy()
    return df_copy

def get_file_name_without_type(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pdf'):
        return 'pdf'
    elif filename.endswith('.jpg'):
        return 'jpg'
    elif filename.endswith('.jpeg'):
        return 'jpeg'
    elif filename.endswith('.png'):
        return 'png'
    elif filename.endswith('.xfdf'):
        return 'xfdf'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def ensure_output_folder_exists(output_folder:str):
    """Checks if the specified folder exists, creates it if not."""   

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")

def custom_regex_load(in_file:List[str], file_type:str = "allow_list"):
    '''
    When file is loaded, update the column dropdown choices and write to relevant data states.
    '''
    custom_regex_df = pd.DataFrame()

    if in_file:
        file_list = [string.name for string in in_file]

        regex_file_names = [string for string in file_list if "csv" in string.lower()]
        if regex_file_names:
            regex_file_name = regex_file_names[0]
            custom_regex_df = pd.read_csv(regex_file_name, low_memory=False, header=None)
            
            # Select just first columns
            custom_regex_df = pd.DataFrame(custom_regex_df.iloc[:,[0]])
            custom_regex_df.rename(columns={0:file_type}, inplace=True)

            custom_regex_df.columns = custom_regex_df.columns.astype(str)

            output_text = file_type + " file loaded."
            print(output_text)
    else:
        output_text = "No file provided."
        print(output_text)
        return output_text, custom_regex_df
       
    return output_text, custom_regex_df

def put_columns_in_df(in_file:List[str]):
    new_choices = []
    concat_choices = []
    all_sheet_names = []
    number_of_excel_files = 0
    
    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        print("File type is:", file_type)

        if file_type == 'xlsx':
            number_of_excel_files += 1
            new_choices = []
            print("Running through all xlsx sheets")
            anon_xlsx = pd.ExcelFile(file_name)
            new_sheet_names = anon_xlsx.sheet_names
            # Iterate through the sheet names
            for sheet_name in new_sheet_names:
                # Read each sheet into a DataFrame
                df = pd.read_excel(file_name, sheet_name=sheet_name)

                # Process the DataFrame (e.g., print its contents)
                print(f"Sheet Name: {sheet_name}")
                print(df.head())  # Print the first few rows

                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        else:
            df = read_file(file_name)
            new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        
    # Drop duplicate columns
    concat_choices = list(set(concat_choices))

    if number_of_excel_files > 0:      
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(choices=all_sheet_names, value=all_sheet_names, visible=True)
    else:
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(visible=False)

def check_for_existing_textract_file(doc_file_name_no_extension_textbox:str, output_folder:str=OUTPUT_FOLDER):
    textract_output_path = os.path.join(output_folder, doc_file_name_no_extension_textbox + "_textract.json")

    if os.path.exists(textract_output_path):
        print("Existing Textract file found.")    
        return True
    
    else:
        return False

# Following function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
def add_folder_to_path(folder_path: str):
    '''
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist.
    '''

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ['PATH']
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ['PATH'] = full_path_extension
            #print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")

# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    return gr.Radio(visible=True, label="Please give some feedback about the results of the redaction. A reminder that the app is only expected to identify about 60% of personally identifiable information in a given (typed) document."), gr.Textbox(visible=True), gr.Button(visible=True), gr.Markdown(visible=True)

def wipe_logs(feedback_logs_loc:str, usage_logs_loc:str):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)

def merge_csv_files(file_list:List[str], output_folder:str=OUTPUT_FOLDER):

    # Initialise an empty list to hold DataFrames
    dataframes = []
    output_files = []

    # Loop through each file in the file list
    for file in file_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file.name)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)    

    for col in ['xmin', 'xmax', 'ymin', 'ymax']:
        merged_df[col] = np.floor(merged_df[col])

    merged_df = merged_df.drop_duplicates(subset=['page', 'label', 'color', 'xmin', 'ymin', 'xmax', 'ymax'])

    merged_df = merged_df.sort_values(['page', 'ymin', 'xmin', 'label'])

    file_out_name = os.path.basename(file_list[0])

    merged_csv_path = output_folder + file_out_name + "_merged.csv"

    # Save the merged DataFrame to a CSV file
    #merged_csv = StringIO()
    merged_df.to_csv(merged_csv_path, index=False)
    output_files.append(merged_csv_path)
    #merged_csv.seek(0)  # Move to the beginning of the StringIO object

    return output_files

async def get_connection_params(request: gr.Request, output_folder_textbox:str=OUTPUT_FOLDER, input_folder_textbox:str=INPUT_FOLDER, session_output_folder:str=SESSION_OUTPUT_FOLDER):

    #print("Session hash:", request.session_hash)

    if CUSTOM_HEADER and CUSTOM_HEADER_VALUE:
            if CUSTOM_HEADER in request.headers:
                supplied_custom_header_value = request.headers[CUSTOM_HEADER]
                if supplied_custom_header_value == CUSTOM_HEADER_VALUE:
                    print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
                else:
                    print("Custom header value does not match expected value.")
                    raise ValueError("Custom header value does not match expected value.")
            else:
                print("Custom header value not found.")
                raise ValueError("Custom header value not found.")   

    # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

    if request.username:
        out_session_hash = request.username
        #print("Request username found:", out_session_hash)

    elif 'x-cognito-id' in request.headers:
        out_session_hash = request.headers['x-cognito-id']
        #print("Cognito ID found:", out_session_hash)

    elif 'x-amzn-oidc-identity' in request.headers:
        out_session_hash = request.headers['x-amzn-oidc-identity']

        # Fetch email address using Cognito client
        cognito_client = boto3.client('cognito-idp')
        try:
            response = cognito_client.admin_get_user(
                UserPoolId=AWS_USER_POOL_ID,  # Replace with your User Pool ID
                Username=out_session_hash
            )
            email = next(attr['Value'] for attr in response['UserAttributes'] if attr['Name'] == 'email')
            #print("Email address found:", email)

            out_session_hash = email
        except ClientError as e:
            print("Error fetching user details:", e)
            email = None

        print("Cognito ID found:", out_session_hash)

    else:
        out_session_hash = request.session_hash

    if session_output_folder == 'True':
        output_folder = output_folder_textbox + out_session_hash + "/"
        input_folder = input_folder_textbox + out_session_hash + "/"
    else:
        output_folder = output_folder_textbox
        input_folder = input_folder_textbox

    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(input_folder): os.mkdir(input_folder)


    return out_session_hash, output_folder, out_session_hash, input_folder

def clean_unicode_text(text:str):
    # Step 1: Normalise unicode characters to decompose any special forms
    normalized_text = unicodedata.normalize('NFKC', text)

    # Step 2: Replace smart quotes and special punctuation with standard ASCII equivalents
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', 
        '–': '-', '—': '-', '…': '...', '•': '*',
    }

    # Perform replacements
    for old_char, new_char in replacements.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    # Step 3: Optionally remove non-ASCII characters if needed
    # This regex removes any remaining non-ASCII characters, if desired.
    # Comment this line if you want to keep all Unicode characters.
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', normalized_text)

    return cleaned_text
   
def load_all_output_files(folder_path:str=OUTPUT_FOLDER) -> List[str]:
    """Get the file paths of all files in the given folder."""
    file_paths = []
    
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        full_path = os.path.join(folder_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    
    return file_paths

def calculate_aws_costs(number_of_pages:str,
                        text_extract_method_radio:str,
                        handwrite_signature_checkbox:List[str],
                        pii_identification_method:str,
                        textract_output_found_checkbox:bool,
                        only_extract_text_radio:bool,
                        textract_page_cost:float=1.5/1000,
                        textract_signature_cost:float=2.0/1000,
                        comprehend_unit_cost:float=0.0001,
                        comprehend_size_unit_average:float=250,
                        average_characters_per_page:float=2000,
                        textract_option:str=textract_option,
                        no_redaction_option:str=no_redaction_option,
                        aws_pii_detector:str=aws_pii_detector):
    '''
    Calculate the approximate cost of submitting a document to AWS Textract and/or AWS Comprehend, assuming that Textract outputs do not already exist in the output folder.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - handwrite_signature_checkbox: Whether signatures are being extracted or not.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - textract_output_found_checkbox: Whether existing Textract results have been found in the output folder. Assumes that results exist for all pages and files in the output folder.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - textract_page_cost (float, optional): AWS pricing for Textract text extraction per page ($).
    - textract_signature_cost (float, optional): Additional AWS cost above standard AWS Textract extraction for extracting signatures.
    - comprehend_unit_cost (float, optional): Cost per 'unit' (300 character minimum) for identifying PII in text with AWS Comprehend.
    - comprehend_size_unit_average (float, optional): Average size of a 'unit' of text passed to AWS Comprehend by the app through the batching process
    - average_characters_per_page (float, optional): Average number of characters on an A4 page.
    - textract_option (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - no_redaction_option (str, optional): String label for pii_identification_method_drop for no redaction.
    - aws_pii_detector (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    '''
    text_extraction_cost = 0
    pii_identification_cost = 0
    calculated_aws_cost = 0
    number_of_pages = int(number_of_pages)
    
    if textract_output_found_checkbox != True:
        if text_extract_method_radio == textract_option:
            text_extraction_cost = number_of_pages * textract_page_cost

            if "Extract signatures" in handwrite_signature_checkbox:
                text_extraction_cost += (textract_signature_cost * number_of_pages)

    if pii_identification_method != no_redaction_option:
        if pii_identification_method == aws_pii_detector:
            comprehend_page_cost = ceil(average_characters_per_page / comprehend_size_unit_average) * comprehend_unit_cost
            pii_identification_cost = comprehend_page_cost * number_of_pages

    calculated_aws_cost = calculated_aws_cost + text_extraction_cost + pii_identification_cost

    return calculated_aws_cost

def calculate_time_taken(number_of_pages:str,
                        text_extract_method_radio:str,
                        pii_identification_method:str,
                        textract_output_found_checkbox:bool,
                        only_extract_text_radio:bool,
                        convert_page_time:float=0.5,
                        textract_page_time:float=1,
                        comprehend_page_time:float=1,
                        local_text_extraction_page_time:float=0.3,
                        local_pii_redaction_page_time:float=0.5,                        
                        local_ocr_extraction_page_time:float=1.5,
                        textract_option:str=textract_option,
                        text_ocr_option:str=text_ocr_option,
                        local_ocr_option:str=tesseract_ocr_option,
                        no_redaction_option:str=no_redaction_option,
                        aws_pii_detector:str=aws_pii_detector):
    '''
    Calculate the approximate time to redact a document.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - textract_page_time (float, optional): Approximate time to query AWS Textract.
    - comprehend_page_time (float, optional): Approximate time to query text on a page with AWS Comprehend.
    - local_text_redaction_page_time (float, optional): Approximate time to extract text on a page with the local text redaction option.
    - local_pii_redaction_page_time (float, optional): Approximate time to redact text on a page with the local text redaction option.
    - local_ocr_extraction_page_time (float, optional): Approximate time to extract text from a page with the local OCR redaction option.
    - textract_option (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - text_ocr_option (str, optional): String label for text_extract_method_radio for text extraction.
    - local_ocr_option (str, optional): String label for text_extract_method_radio for local OCR.
    - no_redaction_option (str, optional): String label for pii_identification_method_drop for no redaction.    
    - aws_pii_detector (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    '''
    calculated_time_taken = 0
    page_conversion_time_taken = 0
    page_extraction_time_taken = 0
    page_redaction_time_taken = 0

    number_of_pages = int(number_of_pages)

    # Page preparation/conversion to image time
    if (text_extract_method_radio != text_ocr_option) and (textract_output_found_checkbox != True):
        page_conversion_time_taken = number_of_pages * convert_page_time

    # Page text extraction time
    if text_extract_method_radio == textract_option:
        if textract_output_found_checkbox != True:
            page_extraction_time_taken = number_of_pages * textract_page_time
    elif text_extract_method_radio == local_ocr_option:
        page_extraction_time_taken = number_of_pages * local_ocr_extraction_page_time
    elif text_extract_method_radio == text_ocr_option:
        page_conversion_time_taken = number_of_pages * local_text_extraction_page_time

    # Page redaction time
    if pii_identification_method != no_redaction_option:
        if pii_identification_method == aws_pii_detector:
            page_redaction_time_taken = number_of_pages * comprehend_page_time
        else:
            page_redaction_time_taken = number_of_pages * local_pii_redaction_page_time

    calculated_time_taken = (page_conversion_time_taken + page_extraction_time_taken + page_redaction_time_taken)/60

    return calculated_time_taken
    