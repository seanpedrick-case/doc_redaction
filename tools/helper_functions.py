import os
import random
import string
import unicodedata
from math import ceil
from pathlib import Path
from typing import List, Set

import boto3
import gradio as gr
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from gradio_image_annotation import image_annotator

from tools.config import (
    AWS_PII_OPTION,
    AWS_USER_POOL_ID,
    CUSTOM_HEADER,
    CUSTOM_HEADER_VALUE,
    DEFAULT_LANGUAGE,
    INPUT_FOLDER,
    LANGUAGE_CHOICES,
    LANGUAGE_MAP,
    NO_REDACTION_PII_OPTION,
    OUTPUT_FOLDER,
    SELECTABLE_TEXT_EXTRACT_OPTION,
    SESSION_OUTPUT_FOLDER,
    SHOW_FEEDBACK_BUTTONS,
    TESSERACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_JOBS_LOCAL_LOC,
    TEXTRACT_JOBS_S3_LOC,
    TEXTRACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    aws_comprehend_language_choices,
    textract_language_choices,
)
from tools.secure_path_utils import secure_join


def reset_state_vars():
    return (
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        0,
        "",
        image_annotator(
            label="Modify redaction boxes",
            label_list=["Redaction"],
            label_colors=[(0, 0, 0)],
            show_label=False,
            sources=None,  # ["upload"],
            show_clear_button=False,
            show_share_button=False,
            show_remove_button=False,
            interactive=False,
        ),
        [],
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        [],
        "",
        False,
        0,
        [],
        [],
    )


def reset_ocr_results_state():
    return pd.DataFrame(), pd.DataFrame(), []


def reset_review_vars():
    return pd.DataFrame(), pd.DataFrame()


def reset_data_vars():
    return 0, [], 0


def reset_aws_call_vars():
    return 0, 0


def load_in_default_allow_list(allow_list_file_path):
    if isinstance(allow_list_file_path, str):
        allow_list_file_path = [allow_list_file_path]
    return allow_list_file_path


def load_in_default_cost_codes(cost_codes_path: str, default_cost_code: str = ""):
    """
    Load in the cost codes list from file.
    """
    cost_codes_df = pd.read_csv(cost_codes_path)
    dropdown_choices = cost_codes_df.iloc[:, 0].astype(str).tolist()

    # Avoid inserting duplicate or empty cost code values
    if default_cost_code and default_cost_code not in dropdown_choices:
        dropdown_choices.insert(0, default_cost_code)

    # Always have a blank option at the top
    if "" not in dropdown_choices:
        dropdown_choices.insert(0, "")

    out_dropdown = gr.Dropdown(
        value=default_cost_code if default_cost_code in dropdown_choices else "",
        label="Choose cost code for analysis",
        choices=dropdown_choices,
        allow_custom_value=False,
    )

    return cost_codes_df, cost_codes_df, out_dropdown


def enforce_cost_codes(
    enforce_cost_code_textbox: str,
    cost_code_choice: str,
    cost_code_df: pd.DataFrame,
    verify_cost_codes: bool = True,
):
    """
    Check if the enforce cost codes variable is set to true, and then check that a cost cost has been chosen. If not, raise an error. Then, check against the values in the cost code dataframe to ensure that the cost code exists.
    """

    if enforce_cost_code_textbox == "True":
        if not cost_code_choice:
            raise Exception("Please choose a cost code before continuing")

        if verify_cost_codes is True:
            if cost_code_df.empty:
                raise Exception("No cost codes present in dataframe for verification")
            else:
                valid_cost_codes_list = list(cost_code_df.iloc[:, 0].unique())

                if cost_code_choice not in valid_cost_codes_list:
                    raise Exception(
                        "Selected cost code not found in list. Please contact Finance if you cannot find the correct cost code from the given list of suggestions."
                    )
    return


def update_cost_code_dataframe_from_dropdown_select(
    cost_dropdown_selection: str, cost_code_df: pd.DataFrame
):
    cost_code_df = cost_code_df.loc[
        cost_code_df.iloc[:, 0] == cost_dropdown_selection, :
    ]
    return cost_code_df


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def update_dataframe(df: pd.DataFrame):
    df_copy = df.copy()
    return df_copy


def get_file_name_without_type(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)

    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    # print(filename_without_extension)

    return filename_without_extension


def detect_file_type(filename: str):
    """Detect the file type based on its extension."""
    if not isinstance(filename, str):
        filename = str(filename)

    if (
        (filename.endswith(".csv"))
        | (filename.endswith(".csv.gz"))
        | (filename.endswith(".zip"))
    ):
        return "csv"
    elif filename.endswith(".xlsx"):
        return "xlsx"
    elif filename.endswith(".xls"):
        return "xls"
    elif filename.endswith(".parquet"):
        return "parquet"
    elif filename.endswith(".pdf"):
        return "pdf"
    elif filename.endswith(".jpg"):
        return "jpg"
    elif filename.endswith(".jpeg"):
        return "jpeg"
    elif filename.endswith(".png"):
        return "png"
    elif filename.endswith(".xfdf"):
        return "xfdf"
    elif filename.endswith(".docx"):
        return "docx"
    else:
        raise ValueError("Unsupported file type.")


def read_file(filename: str, excel_sheet_name: str = ""):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return pd.read_csv(filename, low_memory=False)
    elif file_type == "xlsx":
        if excel_sheet_name:
            try:
                return pd.read_excel(filename, sheet_name=excel_sheet_name)
            except Exception as e:
                print(
                    f"Error reading {filename} with sheet name {excel_sheet_name}: {e}"
                )
                return pd.DataFrame()
        else:
            return pd.read_excel(filename)
    elif file_type == "parquet":
        return pd.read_parquet(filename)


def ensure_output_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def custom_regex_load(in_file: List[str], file_type: str = "allow_list"):
    """
    When file is loaded, update the column dropdown choices and write to relevant data states.
    """
    custom_regex_df = pd.DataFrame()

    if in_file:
        file_list = [string.name for string in in_file]

        regex_file_names = [string for string in file_list if "csv" in string.lower()]
        if regex_file_names:
            regex_file_name = regex_file_names[0]
            custom_regex_df = pd.read_csv(
                regex_file_name, low_memory=False, header=None
            )

            # Select just first columns
            custom_regex_df = pd.DataFrame(custom_regex_df.iloc[:, [0]])
            custom_regex_df.rename(columns={0: file_type}, inplace=True)

            custom_regex_df.columns = custom_regex_df.columns.astype(str)

            output_text = file_type + " file loaded."
            print(output_text)
    else:
        output_text = "No file provided."
        # print(output_text)
        return output_text, custom_regex_df

    return output_text, custom_regex_df


def put_columns_in_df(in_file: List[str]):
    new_choices = []
    concat_choices = []
    all_sheet_names = []
    number_of_excel_files = 0

    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        print("File type is:", file_type)

        if (file_type == "xlsx") | (file_type == "xls"):
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
                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        elif (file_type == "csv") | (file_type == "parquet"):
            df = read_file(file_name)
            new_choices = list(df.columns)

        else:
            new_choices = []

        concat_choices.extend(new_choices)

    # Drop duplicate columns
    concat_choices = list(set(concat_choices))

    if number_of_excel_files > 0:
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(
            choices=all_sheet_names, value=all_sheet_names, visible=True
        )
    else:
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(
            visible=False
        )


def get_textract_file_suffix(handwrite_signature_checkbox: List[str] = list()) -> str:
    """
    Generate a suffix for textract JSON files based on the selected feature types.

    Args:
        handwrite_signature_checkbox: List of selected Textract feature types.
            Options: "Extract signatures", "Extract forms", "Extract layout", "Extract tables"
            "Extract handwriting" is the default and doesn't add a suffix.

    Returns:
        A suffix string like "_sig", "_form", "_sig_form", etc., or empty string if only handwriting is selected.
    """
    if not handwrite_signature_checkbox:
        return ""

    # Map feature types to short suffixes
    feature_map = {
        "Extract signatures": "sig",
        "Extract forms": "form",
        "Extract layout": "layout",
        "Extract tables": "table",
    }

    # Collect suffixes for selected features (excluding handwriting which is default)
    suffixes = []
    for feature in handwrite_signature_checkbox:
        if feature in feature_map:
            suffixes.append(feature_map[feature])

    # Sort alphabetically for consistent naming
    suffixes.sort()

    # Return suffix with underscore prefix if any features selected
    if suffixes:
        return "_" + "_".join(suffixes)
    return ""


def check_for_existing_textract_file(
    doc_file_name_no_extension_textbox: str,
    output_folder: str = OUTPUT_FOLDER,
    handwrite_signature_checkbox: List[str] = list(),
):
    # Generate suffix based on checkbox options
    suffix = get_textract_file_suffix(handwrite_signature_checkbox)
    textract_output_path = secure_join(
        output_folder, doc_file_name_no_extension_textbox + suffix + "_textract.json"
    )

    if os.path.exists(textract_output_path):
        # print("Existing Textract analysis output file found.")
        return True

    else:
        return False


def check_for_relevant_ocr_output_with_words(
    doc_file_name_no_extension_textbox: str,
    text_extraction_method: str,
    output_folder: str = OUTPUT_FOLDER,
):
    if text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_local_text.json"
    elif text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_local_ocr.json"
    elif text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_textract.json"
    else:
        print("No valid text extraction method found. Returning False")
        return False

    doc_file_with_ending = doc_file_name_no_extension_textbox + file_ending

    local_ocr_output_path = secure_join(output_folder, doc_file_with_ending)

    if os.path.exists(local_ocr_output_path):
        print("Existing OCR with words analysis output file found.")
        return True
    else:
        return False


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    if SHOW_FEEDBACK_BUTTONS:
        is_visible = True
    else:
        is_visible = False
    return (
        gr.Radio(
            visible=is_visible,
            label="Please give some feedback about the results of the redaction. A reminder that the app is only expected to identify about 80% of personally identifiable information in a given (typed) document.",
        ),
        gr.Textbox(visible=is_visible),
        gr.Button(visible=is_visible),
        gr.Markdown(visible=is_visible),
    )


def wipe_logs(feedback_logs_loc: str, usage_logs_loc: str):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)


def merge_csv_files(file_list: List[str], output_folder: str = OUTPUT_FOLDER):

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

    for col in ["xmin", "xmax", "ymin", "ymax"]:
        merged_df[col] = np.floor(merged_df[col])

    merged_df = merged_df.drop_duplicates(
        subset=["page", "label", "color", "xmin", "ymin", "xmax", "ymax"]
    )

    merged_df = merged_df.sort_values(["page", "ymin", "xmin", "label"])

    file_out_name = os.path.basename(file_list[0])

    merged_csv_path = output_folder + file_out_name + "_merged.csv"

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    output_files.append(merged_csv_path)

    return output_files


async def get_connection_params(
    request: gr.Request,
    output_folder_textbox: str = OUTPUT_FOLDER,
    input_folder_textbox: str = INPUT_FOLDER,
    session_output_folder: str = SESSION_OUTPUT_FOLDER,
    textract_document_upload_input_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    textract_document_upload_output_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    s3_textract_document_logs_subfolder: str = TEXTRACT_JOBS_S3_LOC,
    local_textract_document_logs_subfolder: str = TEXTRACT_JOBS_LOCAL_LOC,
):

    # print("Session hash:", request.session_hash)

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
        # print("Request username found:", out_session_hash)

    elif "x-cognito-id" in request.headers:
        out_session_hash = request.headers["x-cognito-id"]
        # print("Cognito ID found:", out_session_hash)

    elif "x-amzn-oidc-identity" in request.headers:
        out_session_hash = request.headers["x-amzn-oidc-identity"]

        # Fetch email address using Cognito client
        cognito_client = boto3.client("cognito-idp")
        try:
            response = cognito_client.admin_get_user(
                UserPoolId=AWS_USER_POOL_ID,  # Replace with your User Pool ID
                Username=out_session_hash,
            )
            email = next(
                attr["Value"]
                for attr in response["UserAttributes"]
                if attr["Name"] == "email"
            )
            # print("Email address found:", email)

            out_session_hash = email
        except ClientError as e:
            print("Error fetching user details:", e)
            email = None

        print("Cognito ID found:", out_session_hash)

    else:
        out_session_hash = request.session_hash

    if session_output_folder == "True":
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


def clean_unicode_text(text: str):
    # Step 1: Normalise unicode characters to decompose any special forms
    normalized_text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace smart quotes and special punctuation with standard ASCII equivalents
    replacements = {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        "•": "*",
    }

    # Perform replacements
    for old_char, new_char in replacements.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    # Step 3: Optionally remove non-ASCII characters if needed
    # This regex removes any remaining non-ASCII characters, if desired.
    # Comment this line if you want to keep all Unicode characters.
    from tools.secure_regex_utils import safe_remove_non_ascii

    cleaned_text = safe_remove_non_ascii(normalized_text)

    return cleaned_text


# --- Helper Function for ID Generation ---
# This function encapsulates your ID logic in a performant, batch-oriented way.
def _generate_unique_ids(
    num_ids_to_generate: int, existing_ids_set: Set[str]
) -> List[str]:
    """
    Generates a specified number of unique, 12-character alphanumeric IDs.

    This is a batch-oriented, performant version of the original
    `fill_missing_ids_in_list` logic, designed to work efficiently
    with DataFrames.

    Args:
        num_ids_to_generate (int): The number of unique IDs to create.
        existing_ids_set (Set[str]): A set of IDs that are already in use and
                                     should be avoided.

    Returns:
        List[str]: A list of newly generated unique IDs.
    """
    id_length = 12
    character_set = string.ascii_letters + string.digits

    newly_generated_ids = set()

    # The while loop ensures we generate exactly the number of IDs required,
    # automatically handling the astronomically rare case of a collision.
    while len(newly_generated_ids) < num_ids_to_generate:
        candidate_id = "".join(random.choices(character_set, k=id_length))

        # Check against both pre-existing IDs and IDs generated in this batch
        if (
            candidate_id not in existing_ids_set
            and candidate_id not in newly_generated_ids
        ):
            newly_generated_ids.add(candidate_id)

    return list(newly_generated_ids)


def load_all_output_files(folder_path: str = OUTPUT_FOLDER) -> List[str]:
    """Get the file paths of all files in the given folder and its subfolders."""

    safe_folder_path_resolved = Path(folder_path).resolve()

    return gr.FileExplorer(
        root_dir=safe_folder_path_resolved,
    )


def update_file_explorer_object():
    return gr.FileExplorer()


def all_outputs_file_download_fn(file_explorer_object: list[str]):
    return file_explorer_object


def calculate_aws_costs(
    number_of_pages: str,
    text_extract_method_radio: str,
    handwrite_signature_checkbox: List[str],
    pii_identification_method: str,
    textract_output_found_checkbox: bool,
    only_extract_text_radio: bool,
    convert_to_gbp: bool = True,
    usd_gbp_conversion_rate: float = 0.76,
    textract_page_cost: float = 1.5 / 1000,
    textract_signature_cost: float = 2.0 / 1000,
    comprehend_unit_cost: float = 0.0001,
    comprehend_size_unit_average: float = 250,
    average_characters_per_page: float = 2000,
    TEXTRACT_TEXT_EXTRACT_OPTION: str = TEXTRACT_TEXT_EXTRACT_OPTION,
    NO_REDACTION_PII_OPTION: str = NO_REDACTION_PII_OPTION,
    AWS_PII_OPTION: str = AWS_PII_OPTION,
):
    """
    Calculate the approximate cost of submitting a document to AWS Textract and/or AWS Comprehend, assuming that Textract outputs do not already exist in the output folder.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - handwrite_signature_checkbox: Whether signatures are being extracted or not.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - textract_output_found_checkbox: Whether existing Textract results have been found in the output folder. Assumes that results exist for all pages and files in the output folder.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - convert_to_gbp (bool, optional): Should suggested costs be converted from USD to GBP.
    - usd_gbp_conversion_rate (float, optional): Conversion rate used for USD to GBP. Last changed 14th April 2025.
    - textract_page_cost (float, optional): AWS pricing for Textract text extraction per page ($).
    - textract_signature_cost (float, optional): Additional AWS cost above standard AWS Textract extraction for extracting signatures.
    - comprehend_unit_cost (float, optional): Cost per 'unit' (300 character minimum) for identifying PII in text with AWS Comprehend.
    - comprehend_size_unit_average (float, optional): Average size of a 'unit' of text passed to AWS Comprehend by the app through the batching process
    - average_characters_per_page (float, optional): Average number of characters on an A4 page.
    - TEXTRACT_TEXT_EXTRACT_OPTION (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - NO_REDACTION_PII_OPTION (str, optional): String label for pii_identification_method_drop for no redaction.
    - AWS_PII_OPTION (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    """
    text_extraction_cost = 0
    pii_identification_cost = 0
    calculated_aws_cost = 0
    number_of_pages = int(number_of_pages)

    if textract_output_found_checkbox is not True:
        if text_extract_method_radio == TEXTRACT_TEXT_EXTRACT_OPTION:
            text_extraction_cost = number_of_pages * textract_page_cost

            if "Extract signatures" in handwrite_signature_checkbox:
                text_extraction_cost += textract_signature_cost * number_of_pages

    if pii_identification_method != NO_REDACTION_PII_OPTION:
        if pii_identification_method == AWS_PII_OPTION:
            comprehend_page_cost = (
                ceil(average_characters_per_page / comprehend_size_unit_average)
                * comprehend_unit_cost
            )
            pii_identification_cost = comprehend_page_cost * number_of_pages

    calculated_aws_cost = (
        calculated_aws_cost + text_extraction_cost + pii_identification_cost
    )

    if convert_to_gbp is True:
        calculated_aws_cost *= usd_gbp_conversion_rate

    return calculated_aws_cost


def calculate_time_taken(
    number_of_pages: str,
    text_extract_method_radio: str,
    pii_identification_method: str,
    textract_output_found_checkbox: bool,
    only_extract_text_radio: bool,
    local_ocr_output_found_checkbox: bool,
    convert_page_time: float = 0.5,
    textract_page_time: float = 1.2,
    comprehend_page_time: float = 1.2,
    local_text_extraction_page_time: float = 0.3,
    local_pii_redaction_page_time: float = 0.5,
    local_ocr_extraction_page_time: float = 1.5,
    TEXTRACT_TEXT_EXTRACT_OPTION: str = TEXTRACT_TEXT_EXTRACT_OPTION,
    SELECTABLE_TEXT_EXTRACT_OPTION: str = SELECTABLE_TEXT_EXTRACT_OPTION,
    local_ocr_option: str = TESSERACT_TEXT_EXTRACT_OPTION,
    NO_REDACTION_PII_OPTION: str = NO_REDACTION_PII_OPTION,
    AWS_PII_OPTION: str = AWS_PII_OPTION,
):
    """
    Calculate the approximate time to redact a document.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - textract_output_found_checkbox (bool, optional): Boolean indicating if AWS Textract text extraction outputs have been found.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - local_ocr_output_found_checkbox (bool, optional): Boolean indicating if local OCR text extraction outputs have been found.
    - textract_page_time (float, optional): Approximate time to query AWS Textract.
    - comprehend_page_time (float, optional): Approximate time to query text on a page with AWS Comprehend.
    - local_text_redaction_page_time (float, optional): Approximate time to extract text on a page with the local text redaction option.
    - local_pii_redaction_page_time (float, optional): Approximate time to redact text on a page with the local text redaction option.
    - local_ocr_extraction_page_time (float, optional): Approximate time to extract text from a page with the local OCR redaction option.
    - TEXTRACT_TEXT_EXTRACT_OPTION (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - SELECTABLE_TEXT_EXTRACT_OPTION (str, optional): String label for text_extract_method_radio for text extraction.
    - local_ocr_option (str, optional): String label for text_extract_method_radio for local OCR.
    - NO_REDACTION_PII_OPTION (str, optional): String label for pii_identification_method_drop for no redaction.
    - AWS_PII_OPTION (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    """
    calculated_time_taken = 0
    page_conversion_time_taken = 0
    page_extraction_time_taken = 0
    page_redaction_time_taken = 0

    number_of_pages = int(number_of_pages)

    # Page preparation/conversion to image time
    if (text_extract_method_radio != SELECTABLE_TEXT_EXTRACT_OPTION) and (
        textract_output_found_checkbox is not True
    ):
        page_conversion_time_taken = number_of_pages * convert_page_time

    # Page text extraction time
    if text_extract_method_radio == TEXTRACT_TEXT_EXTRACT_OPTION:
        if textract_output_found_checkbox is not True:
            page_extraction_time_taken = number_of_pages * textract_page_time
    elif text_extract_method_radio == local_ocr_option:
        if local_ocr_output_found_checkbox is not True:
            page_extraction_time_taken = (
                number_of_pages * local_ocr_extraction_page_time
            )
    elif text_extract_method_radio == SELECTABLE_TEXT_EXTRACT_OPTION:
        page_conversion_time_taken = number_of_pages * local_text_extraction_page_time

    # Page redaction time
    if pii_identification_method != NO_REDACTION_PII_OPTION:
        if pii_identification_method == AWS_PII_OPTION:
            page_redaction_time_taken = number_of_pages * comprehend_page_time
        else:
            page_redaction_time_taken = number_of_pages * local_pii_redaction_page_time

    calculated_time_taken = (
        page_conversion_time_taken
        + page_extraction_time_taken
        + page_redaction_time_taken
    ) / 60

    return calculated_time_taken


def reset_base_dataframe(df: pd.DataFrame):
    return df


def reset_ocr_base_dataframe(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["page", "line", "text"])
    else:
        return df.loc[:, ["page", "line", "text"]]


def reset_ocr_with_words_base_dataframe(
    df: pd.DataFrame, page_entity_dropdown_redaction_value: str
):

    df["index"] = df.index
    output_df = df.copy()

    df["page"] = df["page"].astype(str)

    output_df_filtered = df.loc[
        df["page"] == str(page_entity_dropdown_redaction_value),
        [
            "page",
            "line",
            "word_text",
            "word_x0",
            "word_y0",
            "word_x1",
            "word_y1",
            "index",
        ],
    ]
    return output_df_filtered, output_df


def update_language_dropdown(
    chosen_language_full_name_drop,
    textract_language_choices=textract_language_choices,
    aws_comprehend_language_choices=aws_comprehend_language_choices,
    LANGUAGE_MAP=LANGUAGE_MAP,
):

    try:
        full_language_name = chosen_language_full_name_drop.lower()
        matched_language = LANGUAGE_MAP[full_language_name]

        chosen_language_drop = gr.Dropdown(
            value=matched_language,
            choices=LANGUAGE_CHOICES,
            label="Chosen language short code",
            multiselect=False,
            visible=True,
        )

        if (
            matched_language not in aws_comprehend_language_choices
            and matched_language not in textract_language_choices
        ):
            gr.Info(
                f"Note that {full_language_name} is not supported by AWS Comprehend or AWS Textract"
            )
        elif matched_language not in aws_comprehend_language_choices:
            gr.Info(
                f"Note that {full_language_name} is not supported by AWS Comprehend"
            )
        elif matched_language not in textract_language_choices:
            gr.Info(f"Note that {full_language_name} is not supported by AWS Textract")
    except Exception as e:
        print(e)
        gr.Info("Could not find language in list")
        chosen_language_drop = gr.Dropdown(
            value=DEFAULT_LANGUAGE,
            choices=LANGUAGE_CHOICES,
            label="Chosen language short code",
            multiselect=False,
        )

    return chosen_language_drop
