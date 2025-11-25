import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import boto3
import pandas as pd
import pikepdf

from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS,
    RUN_AWS_FUNCTIONS,
)
from tools.custom_image_analyser_engine import CustomImageRecognizerResult, OCRResult
from tools.helper_functions import _generate_unique_ids
from tools.secure_path_utils import secure_file_read


def extract_textract_metadata(response: object):
    """Extracts metadata from an AWS Textract response."""

    request_id = response["ResponseMetadata"]["RequestId"]
    pages = response["DocumentMetadata"]["Pages"]

    return str({"RequestId": request_id, "Pages": pages})


def analyse_page_with_textract(
    pdf_page_bytes: object,
    page_no: int,
    client: str = "",
    handwrite_signature_checkbox: List[str] = ["Extract handwriting"],
    textract_output_found: bool = False,
    aws_access_question_textbox: str = AWS_ACCESS_KEY,
    aws_secret_question_textbox: str = AWS_SECRET_KEY,
    RUN_AWS_FUNCTIONS: bool = RUN_AWS_FUNCTIONS,
    PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS: bool = PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS,
):
    """
    Analyzes a single page of a document using AWS Textract to extract text and other features.

    Args:
        pdf_page_bytes (object): The content of the PDF page or image as bytes.
        page_no (int): The page number being analyzed.
        client (str, optional): An optional pre-initialized AWS Textract client. If not provided,
                                the function will attempt to create one based on configuration.
                                Defaults to "".
        handwrite_signature_checkbox (List[str], optional): A list of feature types to extract
                                                            from the document. Options include
                                                            "Extract handwriting", "Extract signatures",
                                                            "Extract forms", "Extract layout", "Extract tables".
                                                            Defaults to ["Extract handwriting"].
        textract_output_found (bool, optional): A flag indicating whether existing Textract output
                                                for the document has been found. This can prevent
                                                unnecessary API calls. Defaults to False.
        aws_access_question_textbox (str, optional): AWS access question provided by the user, if not using
                                                SSO or environment variables. Defaults to AWS_ACCESS_KEY.
        aws_secret_question_textbox (str, optional): AWS secret question provided by the user, if not using
                                                SSO or environment variables. Defaults to AWS_SECRET_KEY.
        RUN_AWS_FUNCTIONS (bool, optional): Configuration flag to enable or
                                           disable AWS functions. Defaults to RUN_AWS_FUNCTIONS.
        PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS (bool, optional): Configuration flag (e.g., True or False)
                                                                 to prioritize AWS SSO credentials
                                                                 over environment variables.
                                                                 Defaults to True.

    Returns:
        Tuple[List[Dict], str]: A tuple containing:
            - A list of dictionaries, where each dictionary represents a Textract block (e.g., LINE, WORD, FORM, TABLE).
            - A string containing metadata about the Textract request.
    """

    # print("handwrite_signature_checkbox in analyse_page_with_textract:", handwrite_signature_checkbox)
    if client == "":
        try:
            # Try to connect to AWS Textract Client if using that text extraction method
            if RUN_AWS_FUNCTIONS and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS:
                print("Connecting to Textract via existing SSO connection")
                client = boto3.client("textract", region_name=AWS_REGION)
            elif aws_access_question_textbox and aws_secret_question_textbox:
                print(
                    "Connecting to Textract using AWS access question and secret questions from user input."
                )
                client = boto3.client(
                    "textract",
                    aws_access_question_id=aws_access_question_textbox,
                    aws_secret_access_question=aws_secret_question_textbox,
                    region_name=AWS_REGION,
                )
            elif RUN_AWS_FUNCTIONS is True:
                print("Connecting to Textract via existing SSO connection")
                client = boto3.client("textract", region_name=AWS_REGION)
            elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
                print("Getting Textract credentials from environment variables.")
                client = boto3.client(
                    "textract",
                    aws_access_question_id=AWS_ACCESS_KEY,
                    aws_secret_access_question=AWS_SECRET_KEY,
                    region_name=AWS_REGION,
                )
            elif textract_output_found is True:
                print(
                    "Existing Textract data found for file, no need to connect to AWS Textract"
                )
                client = boto3.client("textract", region_name=AWS_REGION)
            else:
                client = ""
                out_message = "Cannot connect to AWS Textract service."
                print(out_message)
                raise Exception(out_message)
        except Exception as e:
            out_message = "Cannot connect to AWS Textract"
            print(out_message, "due to:", e)
            raise Exception(out_message)
            return [], ""  # Return an empty list and an empty string

    # Redact signatures if specified
    feature_types = list()
    if (
        "Extract signatures" in handwrite_signature_checkbox
        or "Extract forms" in handwrite_signature_checkbox
        or "Extract layout" in handwrite_signature_checkbox
        or "Extract tables" in handwrite_signature_checkbox
    ):
        if "Extract signatures" in handwrite_signature_checkbox:
            feature_types.append("SIGNATURES")
        if "Extract forms" in handwrite_signature_checkbox:
            feature_types.append("FORMS")
        if "Extract layout" in handwrite_signature_checkbox:
            feature_types.append("LAYOUT")
        if "Extract tables" in handwrite_signature_checkbox:
            feature_types.append("TABLES")
        try:
            response = client.analyze_document(
                Document={"Bytes": pdf_page_bytes}, FeatureTypes=feature_types
            )
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 3 seconds.")
            time.sleep(3)
            response = client.analyze_document(
                Document={"Bytes": pdf_page_bytes}, FeatureTypes=feature_types
            )

    if (
        "Extract signatures" not in handwrite_signature_checkbox
        and "Extract forms" not in handwrite_signature_checkbox
        and "Extract layout" not in handwrite_signature_checkbox
        and "Extract tables" not in handwrite_signature_checkbox
    ):
        # Call detect_document_text to extract plain text
        try:
            response = client.detect_document_text(Document={"Bytes": pdf_page_bytes})
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 5 seconds.")
            time.sleep(5)
            response = client.detect_document_text(Document={"Bytes": pdf_page_bytes})

    # Add the 'Page' attribute to each block
    if "Blocks" in response:
        for block in response["Blocks"]:
            block["Page"] = page_no  # Inject the page number into each block

    # Wrap the response with the page number in the desired format
    wrapped_response = {"page_no": page_no, "data": response}

    request_metadata = extract_textract_metadata(
        response
    )  # Metadata comes out as a string

    # Return a list containing the wrapped response and the metadata
    return (
        wrapped_response,
        request_metadata,
    )  # Return as a list to match the desired structure


def convert_pike_pdf_page_to_bytes(pdf: object, page_num: int):
    # Create a new empty PDF
    new_pdf = pikepdf.Pdf.new()

    # Specify the page number you want to extract (0-based index)
    page_num = 0  # Example: first page

    # Extract the specific page and add it to the new PDF
    new_pdf.pages.append(pdf.pages[page_num])

    # Save the new PDF to a bytes buffer
    buffer = io.BytesIO()
    new_pdf.save(buffer)

    # Get the PDF bytes
    pdf_bytes = buffer.getanswer()

    # Now you can use the `pdf_bytes` to convert it to an image or further process
    buffer.close()

    return pdf_bytes


def json_to_ocrresult(
    json_data: dict, page_width: float, page_height: float, page_no: int
):
    """
    Convert Textract JSON to structured OCR, handling lines, words, signatures,
    selection elements (associating them with lines), and question-answer form data.
    The question-answer data is sorted in a top-to-bottom, left-to-right reading order.

    Args:
        json_data (dict): The raw JSON output from AWS Textract for a specific page.
        page_width (float): The width of the page in pixels or points.
        page_height (float): The height of the page in pixels or points.
        page_no (int): The 1-based page number being processed.
    """
    # --- STAGE 1: Block Mapping & Initial Data Collection ---
    # text_blocks = json_data.get("Blocks", [])
    # Find the specific page data
    page_json_data = json_data  # next((page for page in json_data["pages"] if page["page_no"] == page_no), None)

    if "Blocks" in page_json_data:
        # Access the data for the specific page
        text_blocks = page_json_data["Blocks"]  # Access the Blocks within the page data
    # This is a new page
    elif "page_no" in page_json_data:
        text_blocks = page_json_data["data"]["Blocks"]
    else:
        text_blocks = []

    block_map = {block["Id"]: block for block in text_blocks}

    lines_data = list()
    selections_data = list()
    signature_or_handwriting_recogniser_results = list()
    signature_recogniser_results = list()
    handwriting_recogniser_results = list()

    def _get_text_from_block(block, b_map):
        text_parts = list()
        if "Relationships" in block:
            for rel in block["Relationships"]:
                if rel["Type"] == "CHILD":
                    for child_id in rel["Ids"]:
                        child = b_map.get(child_id)
                        if child:
                            if child["BlockType"] == "WORD":
                                text_parts.append(child["Text"])
                            elif child["BlockType"] == "SELECTION_ELEMENT":
                                text_parts.append(f"[{child['SelectionStatus']}]")
        return " ".join(text_parts)

    # text_line_number = 1

    for block in text_blocks:
        block_type = block.get("BlockType")

        if block_type == "LINE":
            bbox = block["Geometry"]["BoundingBox"]
            line_info = {
                "id": block["Id"],
                "text": block.get("Text", ""),
                "confidence": round(block.get("Confidence", 0.0), 0),
                "words": [],
                "geometry": {
                    "left": int(bbox["Left"] * page_width),
                    "top": int(bbox["Top"] * page_height),
                    "width": int(bbox["Width"] * page_width),
                    "height": int(bbox["Height"] * page_height),
                },
            }
            if "Relationships" in block:
                for rel in block.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        for child_id in rel["Ids"]:
                            word_block = block_map.get(child_id)
                            if word_block and word_block["BlockType"] == "WORD":
                                w_bbox = word_block["Geometry"]["BoundingBox"]
                                line_info["words"].append(
                                    {
                                        "text": word_block.get("Text", ""),
                                        "confidence": round(
                                            word_block.get("Confidence", 0.0), 0
                                        ),
                                        "bounding_box": (
                                            int(w_bbox["Left"] * page_width),
                                            int(w_bbox["Top"] * page_height),
                                            int(
                                                (w_bbox["Left"] + w_bbox["Width"])
                                                * page_width
                                            ),
                                            int(
                                                (w_bbox["Top"] + w_bbox["Height"])
                                                * page_height
                                            ),
                                        ),
                                    }
                                )
                                if word_block.get("TextType") == "HANDWRITING":
                                    rec_res = CustomImageRecognizerResult(
                                        entity_type="HANDWRITING",
                                        text=word_block.get("Text", ""),
                                        score=round(
                                            word_block.get("Confidence", 0.0), 0
                                        ),
                                        start=0,
                                        end=len(word_block.get("Text", "")),
                                        left=int(w_bbox["Left"] * page_width),
                                        top=int(w_bbox["Top"] * page_height),
                                        width=int(w_bbox["Width"] * page_width),
                                        height=int(w_bbox["Height"] * page_height),
                                    )
                                    handwriting_recogniser_results.append(rec_res)
                                    signature_or_handwriting_recogniser_results.append(
                                        rec_res
                                    )
            lines_data.append(line_info)

        elif block_type == "SELECTION_ELEMENT":
            bbox = block["Geometry"]["BoundingBox"]
            selections_data.append(
                {
                    "id": block["Id"],
                    "status": block.get("SelectionStatus", "UNKNOWN"),
                    "confidence": round(block.get("Confidence", 0.0), 0),
                    "geometry": {
                        "left": int(bbox["Left"] * page_width),
                        "top": int(bbox["Top"] * page_height),
                        "width": int(bbox["Width"] * page_width),
                        "height": int(bbox["Height"] * page_height),
                    },
                }
            )

        elif block_type == "SIGNATURE":
            bbox = block["Geometry"]["BoundingBox"]
            rec_res = CustomImageRecognizerResult(
                entity_type="SIGNATURE",
                text="SIGNATURE",
                score=round(block.get("Confidence", 0.0), 0),
                start=0,
                end=9,
                left=int(bbox["Left"] * page_width),
                top=int(bbox["Top"] * page_height),
                width=int(bbox["Width"] * page_width),
                height=int(bbox["Height"] * page_height),
            )
            signature_recogniser_results.append(rec_res)
            signature_or_handwriting_recogniser_results.append(rec_res)

    # --- STAGE 2: Question-Answer Pair Extraction & Sorting ---
    def _create_question_answer_results_object(text_blocks):
        question_answer_results = list()
        key_blocks = [
            b
            for b in text_blocks
            if b.get("BlockType") == "KEY_VALUE_SET"
            and "KEY" in b.get("EntityTypes", [])
        ]
        for question_block in key_blocks:
            answer_block = next(
                (
                    block_map.get(rel["Ids"][0])
                    for rel in question_block.get("Relationships", [])
                    if rel["Type"] == "VALUE"
                ),
                None,
            )

            # The check for value_block now happens BEFORE we try to access its properties.
            if answer_block:
                question_bbox = question_block["Geometry"]["BoundingBox"]
                # We also get the answer_bbox safely inside this block.
                answer_bbox = answer_block["Geometry"]["BoundingBox"]

                question_answer_results.append(
                    {
                        # Data for final output
                        "Page": page_no,
                        "Question": _get_text_from_block(question_block, block_map),
                        "Answer": _get_text_from_block(answer_block, block_map),
                        "Confidence Score % (Question)": round(
                            question_block.get("Confidence", 0.0), 0
                        ),
                        "Confidence Score % (Answer)": round(
                            answer_block.get("Confidence", 0.0), 0
                        ),
                        "Question_left": round(question_bbox["Left"], 5),
                        "Question_top": round(question_bbox["Top"], 5),
                        "Question_width": round(question_bbox["Width"], 5),
                        "Question_height": round(question_bbox["Height"], 5),
                        "Answer_left": round(answer_bbox["Left"], 5),
                        "Answer_top": round(answer_bbox["Top"], 5),
                        "Answer_width": round(answer_bbox["Width"], 5),
                        "Answer_height": round(answer_bbox["Height"], 5),
                    }
                )

        question_answer_results.sort(
            key=lambda item: (item["Question_top"], item["Question_left"])
        )

        return question_answer_results

    question_answer_results = _create_question_answer_results_object(text_blocks)

    # --- STAGE 3: Association of Selection Elements to Lines ---
    unmatched_selections = list()
    for selection in selections_data:
        best_match_line = None
        min_dist = float("inf")
        sel_geom = selection["geometry"]
        sel_y_center = sel_geom["top"] + sel_geom["height"] / 2
        for line in lines_data:
            line_geom = line["geometry"]
            line_y_center = line_geom["top"] + line_geom["height"] / 2
            if abs(sel_y_center - line_y_center) < line_geom["height"]:
                dist = 0
                if sel_geom["left"] > (line_geom["left"] + line_geom["width"]):
                    dist = sel_geom["left"] - (line_geom["left"] + line_geom["width"])
                elif line_geom["left"] > (sel_geom["left"] + sel_geom["width"]):
                    dist = line_geom["left"] - (sel_geom["left"] + sel_geom["width"])
                if dist < min_dist:
                    min_dist = dist
                    best_match_line = line
        if best_match_line and min_dist < (best_match_line["geometry"]["height"] * 5):
            selection_as_word = {
                "text": f"[{selection['status']}]",
                "confidence": round(selection["confidence"], 0),
                "bounding_box": (
                    sel_geom["left"],
                    sel_geom["top"],
                    sel_geom["left"] + sel_geom["width"],
                    sel_geom["top"] + sel_geom["height"],
                ),
            }
            best_match_line["words"].append(selection_as_word)
            best_match_line["words"].sort(key=lambda w: w["bounding_box"][0])
        else:
            unmatched_selections.append(selection)

    # --- STAGE 4: Final Output Generation ---
    all_ocr_results = list()
    ocr_results_with_words = dict()
    selection_element_results = list()
    for i, line in enumerate(lines_data):
        line_num = i + 1
        line_geom = line["geometry"]
        reconstructed_text = " ".join(w["text"] for w in line["words"])
        all_ocr_results.append(
            OCRResult(
                reconstructed_text,
                line_geom["left"],
                line_geom["top"],
                line_geom["width"],
                line_geom["height"],
                round(line["confidence"], 0),
                line_num,
            )
        )
        ocr_results_with_words[f"text_line_{line_num}"] = {
            "line": line_num,
            "text": reconstructed_text,
            "confidence": line["confidence"],
            "bounding_box": (
                line_geom["left"],
                line_geom["top"],
                line_geom["left"] + line_geom["width"],
                line_geom["top"] + line_geom["height"],
            ),
            "words": line["words"],
            "page": page_no,
        }
    for selection in unmatched_selections:
        sel_geom = selection["geometry"]
        sel_text = f"[{selection['status']}]"
        all_ocr_results.append(
            OCRResult(
                sel_text,
                sel_geom["left"],
                sel_geom["top"],
                sel_geom["width"],
                sel_geom["height"],
                round(selection["confidence"], 0),
                -1,
            )
        )
    for selection in selections_data:
        sel_geom = selection["geometry"]
        selection_element_results.append(
            {
                "status": selection["status"],
                "confidence": round(selection["confidence"], 0),
                "bounding_box": (
                    sel_geom["left"],
                    sel_geom["top"],
                    sel_geom["left"] + sel_geom["width"],
                    sel_geom["top"] + sel_geom["height"],
                ),
                "page": page_no,
            }
        )

    all_ocr_results_with_page = {"page": page_no, "results": all_ocr_results}
    ocr_results_with_words_with_page = {
        "page": page_no,
        "results": ocr_results_with_words,
    }

    return (
        all_ocr_results_with_page,
        signature_or_handwriting_recogniser_results,
        signature_recogniser_results,
        handwriting_recogniser_results,
        ocr_results_with_words_with_page,
        selection_element_results,
        question_answer_results,
    )


def load_and_convert_textract_json(
    textract_json_file_path: str,
    log_files_output_paths: str,
    page_sizes_df: pd.DataFrame,
):
    """
    Loads Textract JSON from a file, detects if conversion is needed, and converts if necessary.

    Args:
        textract_json_file_path (str): The file path to the Textract JSON output.
        log_files_output_paths (str): A list of paths to log files, used for tracking.
        page_sizes_df (pd.DataFrame): A DataFrame containing page size information for the document.
    """

    if not os.path.exists(textract_json_file_path):
        print("No existing Textract results file found.")
        return (
            {},
            True,
            log_files_output_paths,
        )  # Return empty dict and flag indicating missing file

    print("Found existing Textract json results file.")

    # Track log files
    if textract_json_file_path not in log_files_output_paths:
        log_files_output_paths.append(textract_json_file_path)

    try:
        # Split the path into base directory and filename for security
        textract_json_file_path_obj = Path(textract_json_file_path)
        base_dir = textract_json_file_path_obj.parent
        filename = textract_json_file_path_obj.name

        json_content = secure_file_read(base_dir, filename, encoding="utf-8")
        textract_data = json.loads(json_content)
    except json.JSONDecodeError:
        print("Error: Failed to parse Textract JSON file. Returning empty data.")
        return {}, True, log_files_output_paths  # Indicate failure

    # Check if conversion is needed
    if "pages" in textract_data:
        print("JSON already in the correct format for app. No changes needed.")
        return textract_data, False, log_files_output_paths  # No conversion required

    if "Blocks" in textract_data:
        print("Need to convert Textract JSON to app format.")
        try:

            textract_data = restructure_textract_output(textract_data, page_sizes_df)
            return (
                textract_data,
                False,
                log_files_output_paths,
            )  # Successfully converted

        except Exception as e:
            print("Failed to convert JSON data to app format due to:", e)
            return {}, True, log_files_output_paths  # Conversion failed
    else:
        print("Invalid Textract JSON format: 'Blocks' missing.")
        # print("textract data:", textract_data)
        return (
            {},
            True,
            log_files_output_paths,
        )  # Return empty data if JSON is not recognized


def restructure_textract_output(textract_output: dict, page_sizes_df: pd.DataFrame):
    """
    Reorganise Textract output from the bulk Textract analysis option on AWS
    into a format that works in this redaction app, reducing size.

    Args:
        textract_output (dict): The raw JSON output from AWS Textract.
        page_sizes_df (pd.DataFrame): A Pandas DataFrame containing page size
                                      information, including cropbox and mediabox
                                      dimensions and offsets for each page.
    """
    pages_dict = dict()

    # Extract total pages from DocumentMetadata
    document_metadata = textract_output.get("DocumentMetadata", {})

    # For efficient lookup, set 'page' as index if it's not already
    if "page" in page_sizes_df.columns:
        page_sizes_df = page_sizes_df.set_index("page")

    for block in textract_output.get("Blocks", []):
        page_no = block.get("Page", 1)  # Default to 1 if missing

        # --- Geometry Conversion Logic ---
        try:
            page_info = page_sizes_df.loc[page_no]
            cb_width = page_info["cropbox_width"]
            cb_height = page_info["cropbox_height"]
            mb_width = page_info["mediabox_width"]
            mb_height = page_info["mediabox_height"]
            cb_x_offset = page_info["cropbox_x_offset"]
            cb_y_offset_top = page_info["cropbox_y_offset_from_top"]

            # Check if conversion is needed (and avoid division by zero)
            needs_conversion = (
                (abs(cb_width - mb_width) > 1e-6 or abs(cb_height - mb_height) > 1e-6)
                and mb_width > 1e-6
                and mb_height > 1e-6
            )  # Avoid division by zero

            if needs_conversion and "Geometry" in block:
                geometry = block["Geometry"]  # Work directly on the block's geometry

                # --- Convert BoundingBox ---
                if "BoundingBox" in geometry:
                    bbox = geometry["BoundingBox"]
                    old_left = bbox["Left"]
                    old_top = bbox["Top"]
                    old_width = bbox["Width"]
                    old_height = bbox["Height"]

                    # Calculate absolute coordinates within CropBox
                    abs_cb_x = old_left * cb_width
                    abs_cb_y = old_top * cb_height
                    abs_cb_width = old_width * cb_width
                    abs_cb_height = old_height * cb_height

                    # Calculate absolute coordinates relative to MediaBox top-left
                    abs_mb_x = cb_x_offset + abs_cb_x
                    abs_mb_y = cb_y_offset_top + abs_cb_y

                    # Convert back to normalized coordinates relative to MediaBox
                    bbox["Left"] = abs_mb_x / mb_width
                    bbox["Top"] = abs_mb_y / mb_height
                    bbox["Width"] = abs_cb_width / mb_width
                    bbox["Height"] = abs_cb_height / mb_height
        except KeyError:
            print(
                f"Warning: Page number {page_no} not found in page_sizes_df. Skipping coordinate conversion for this block."
            )
            # Decide how to handle missing page info: skip conversion, raise error, etc.
        except ZeroDivisionError:
            print(
                f"Warning: MediaBox width or height is zero for page {page_no}. Skipping coordinate conversion for this block."
            )

        # Initialise page structure if not already present
        if page_no not in pages_dict:
            pages_dict[page_no] = {"page_no": str(page_no), "data": {"Blocks": []}}

        # Keep only essential fields to reduce size
        filtered_block = {
            question: block[question]
            for question in [
                "BlockType",
                "Confidence",
                "Text",
                "Geometry",
                "Page",
                "Id",
                "Relationships",
            ]
            if question in block
        }

        pages_dict[page_no]["data"]["Blocks"].append(filtered_block)

    # Convert pages dictionary to a sorted list
    structured_output = {
        "DocumentMetadata": document_metadata,  # Store metadata separately
        "pages": [pages_dict[page] for page in sorted(pages_dict.questions())],
    }

    return structured_output


def convert_question_answer_to_dataframe(
    question_answer_results: List[Dict[str, Any]], page_sizes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert question-answer results to DataFrame format matching convert_annotation_data_to_dataframe.

    Each Question and Answer will be on separate lines in the resulting dataframe.
    The 'image' column will be populated with the page number as f'placeholder_image_page{i}.png'.

    Args:
        question_answer_results: List of question-answer dictionaries from _create_question_answer_results_object
        page_sizes_df: DataFrame containing page sizes

    Returns:
        pd.DataFrame: DataFrame with columns ["image", "page", "label", "color", "xmin", "xmax", "ymin", "ymax", "text", "id"]
    """

    if not question_answer_results:
        # Return empty DataFrame with expected schema
        return pd.DataFrame(
            columns=[
                "image",
                "page",
                "label",
                "color",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "text",
                "id",
            ]
        )

    # Prepare data for DataFrame
    rows = list()
    existing_ids = set()

    for i, qa_result in enumerate(question_answer_results):
        page_num = int(qa_result.get("Page", 1))
        page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
        page_sizes_df.dropna(subset=["page"], inplace=True)
        if not page_sizes_df.empty:
            page_sizes_df["page"] = page_sizes_df["page"].astype(int)
        else:
            print("Warning: Page sizes DataFrame became empty after processing.")

        image_name = page_sizes_df.loc[
            page_sizes_df["page"] == page_num, "image_path"
        ].iloc[0]
        if pd.isna(image_name):
            image_name = f"placeholder_image_{page_num}.png"

        # Create Question row
        question_bbox = {
            "Question_left": qa_result.get("Question_left", 0),
            "Question_top": qa_result.get("Question_top", 0),
            "Question_width": qa_result.get("Question_width", 0),
            "Question_height": qa_result.get("Question_height", 0),
        }

        question_row = {
            "image": image_name,
            "page": page_num,
            "label": f"Question {i+1}",
            "color": "(0,0,255)",
            "xmin": question_bbox["Question_left"],
            "xmax": question_bbox["Question_left"] + question_bbox["Question_width"],
            "ymin": question_bbox["Question_top"],
            "ymax": question_bbox["Question_top"] + question_bbox["Question_height"],
            "text": qa_result.get("Question", ""),
            "id": None,  # Will be filled after generating IDs
        }

        # Create Answer row
        answer_bbox = {
            "Answer_left": qa_result.get("Answer_left", 0),
            "Answer_top": qa_result.get("Answer_top", 0),
            "Answer_width": qa_result.get("Answer_width", 0),
            "Answer_height": qa_result.get("Answer_height", 0),
        }

        answer_row = {
            "image": image_name,
            "page": page_num,
            "label": f"Answer {i+1}",
            "color": "(0,255,0)",
            "xmin": answer_bbox["Answer_left"],
            "xmax": answer_bbox["Answer_left"] + answer_bbox["Answer_width"],
            "ymin": answer_bbox["Answer_top"],
            "ymax": answer_bbox["Answer_top"] + answer_bbox["Answer_height"],
            "text": qa_result.get("Answer", ""),
            "id": None,  # Will be filled after generating IDs
        }

        rows.extend([question_row, answer_row])

    # Generate unique IDs for all rows
    num_ids_needed = len(rows)
    unique_ids = _generate_unique_ids(num_ids_needed, existing_ids)

    # Assign IDs to rows
    for i, row in enumerate(rows):
        row["id"] = unique_ids[i]

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure all required columns are present and in correct order
    required_columns = [
        "image",
        "page",
        "label",
        "color",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "text",
        "id",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder columns to match expected format
    df = df.reindex(columns=required_columns, fill_value=pd.NA)

    return df


def convert_question_answer_to_annotation_json(
    question_answer_results: List[Dict[str, Any]], page_sizes_df: pd.DataFrame
) -> List[Dict]:
    """
    Convert question-answer results directly to Gradio Annotation JSON format.

    This function combines the functionality of convert_question_answer_to_dataframe
    and convert_review_df_to_annotation_json to directly convert question-answer
    results to the annotation JSON format without the intermediate DataFrame step.

    Args:
        question_answer_results: List of question-answer dictionaries from _create_question_answer_results_object
        page_sizes_df: DataFrame containing page sizes with columns ['page', 'image_path', 'image_width', 'image_height']

    Returns:
        List of dictionaries suitable for Gradio Annotation output, one dict per image/page.
        Each dict has structure: {"image": image_path, "boxes": [list of annotation boxes]}
    """

    if not question_answer_results:
        # Return empty structure based on page_sizes_df
        json_data = list()
        for _, row in page_sizes_df.iterrows():
            json_data.append(
                {
                    "image": row.get(
                        "image_path", f"placeholder_image_{row.get('page', 1)}.png"
                    ),
                    "boxes": [],
                }
            )
        return json_data

    # Validate required columns in page_sizes_df
    required_ps_cols = {"page", "image_path", "image_width", "image_height"}
    if not required_ps_cols.issubset(page_sizes_df.columns):
        missing = required_ps_cols - set(page_sizes_df.columns)
        raise ValueError(f"page_sizes_df is missing required columns: {missing}")

    # Convert page sizes columns to appropriate numeric types
    page_sizes_df = page_sizes_df.copy()  # Work with a copy to avoid modifying original
    page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
    page_sizes_df["image_width"] = pd.to_numeric(
        page_sizes_df["image_width"], errors="coerce"
    )
    page_sizes_df["image_height"] = pd.to_numeric(
        page_sizes_df["image_height"], errors="coerce"
    )
    page_sizes_df["page"] = page_sizes_df["page"].astype("Int64")

    # Prepare data for processing
    rows = list()
    existing_ids = set()

    for i, qa_result in enumerate(question_answer_results):
        page_num = int(qa_result.get("Page", 1))

        # Get image path for this page
        page_row = page_sizes_df[page_sizes_df["page"] == page_num]
        if not page_row.empty:
            page_row["image_path"].iloc[0]
        else:
            pass

        # Create Question box.
        question_bbox = {
            "Question_left": qa_result.get("Question_left", 0),
            "Question_top": qa_result.get("Question_top", 0),
            "Question_width": qa_result.get("Question_width", 0),
            "Question_height": qa_result.get("Question_height", 0),
        }

        question_box = {
            "label": f"Question {i+1}",
            "color": (0, 0, 255),  # Blue for questions
            "xmin": question_bbox["Question_left"],
            "xmax": question_bbox["Question_left"] + question_bbox["Question_width"],
            "ymin": question_bbox["Question_top"],
            "ymax": question_bbox["Question_top"] + question_bbox["Question_height"],
            "text": qa_result.get("Question", ""),
            "id": None,  # Will be filled after generating IDs
        }

        # Create Answer box
        answer_bbox = {
            "Answer_left": qa_result.get("Answer_left", 0),
            "Answer_top": qa_result.get("Answer_top", 0),
            "Answer_width": qa_result.get("Answer_width", 0),
            "Answer_height": qa_result.get("Answer_height", 0),
        }

        answer_box = {
            "label": f"Answer {i+1}",
            "color": (0, 255, 0),  # Green for answers
            "xmin": answer_bbox["Answer_left"],
            "xmax": answer_bbox["Answer_left"] + answer_bbox["Answer_width"],
            "ymin": answer_bbox["Answer_top"],
            "ymax": answer_bbox["Answer_top"] + answer_bbox["Answer_height"],
            "text": qa_result.get("Answer", ""),
            "id": None,  # Will be filled after generating IDs
        }

        rows.extend([(page_num, question_box), (page_num, answer_box)])

    # Generate unique IDs for all boxes
    num_ids_needed = len(rows)
    unique_ids = _generate_unique_ids(num_ids_needed, existing_ids)

    # Assign IDs to boxes
    for i, (page_num, box) in enumerate(rows):
        box["id"] = unique_ids[i]
        rows[i] = (page_num, box)

    # Group boxes by page
    boxes_by_page = {}
    for page_num, box in rows:
        if page_num not in boxes_by_page:
            boxes_by_page[page_num] = list()
        boxes_by_page[page_num].append(box)

    # Build JSON structure based on page_sizes
    json_data = list()
    for _, row in page_sizes_df.iterrows():
        page_num = row["page"]
        pdf_image_path = row["image_path"]

        # Get boxes for this page
        annotation_boxes = boxes_by_page.get(page_num, [])

        # Append the structured data for this image/page
        json_data.append({"image": pdf_image_path, "boxes": annotation_boxes})

    return json_data


def convert_page_question_answer_to_custom_image_recognizer_results(
    question_answer_results: List[Dict[str, Any]],
    page_sizes_df: pd.DataFrame,
    reported_page_number: int,
) -> List["CustomImageRecognizerResult"]:
    """
    Convert question-answer results to a list of CustomImageRecognizerResult objects.

    Args:
        question_answer_results: List of question-answer dictionaries from _create_question_answer_results_object
        page_sizes_df: DataFrame containing page sizes with columns ['page', 'image_path', 'image_width', 'image_height']
        reported_page_number: The page number reported by the user
    Returns:
        List of CustomImageRecognizerResult objects for questions and answers
    """
    from tools.custom_image_analyser_engine import CustomImageRecognizerResult

    if not question_answer_results:
        return list()

    results = list()

    # Pre-process page_sizes_df once for efficiency
    page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
    page_sizes_df.dropna(subset=["page"], inplace=True)
    if not page_sizes_df.empty:
        page_sizes_df["page"] = page_sizes_df["page"].astype(int)
    else:
        print("Warning: Page sizes DataFrame became empty after processing.")
        return list()  # Return empty list if no page sizes are available

    page_row = page_sizes_df.loc[page_sizes_df["page"] == int(reported_page_number)]

    if page_row.empty:
        print(
            f"Warning: Page {reported_page_number} not found in page_sizes_df. Skipping this entry."
        )
        return list()  # Return empty list if page not found

    for i, qa_result in enumerate(question_answer_results):
        current_page = int(qa_result.get("Page", 1))

        if current_page != int(reported_page_number):
            continue  # Skip this entry if page number does not match reported page number

        # Get image dimensions safely
        # Textract coordinates are normalized (0-1) relative to MediaBox
        # We need to convert to image coordinates, not PDF page coordinates
        # Try to get image dimensions first, fallback to mediabox if not available
        try:
            if "image_width" in page_sizes_df.columns:
                image_width_val = page_row["image_width"].iloc[0]
                if pd.notna(image_width_val) and image_width_val > 0:
                    image_width = image_width_val
                else:
                    image_width = page_row["mediabox_width"].iloc[0]
            else:
                image_width = page_row["mediabox_width"].iloc[0]
        except (KeyError, IndexError):
            image_width = page_row["mediabox_width"].iloc[0]

        try:
            if "image_height" in page_sizes_df.columns:
                image_height_val = page_row["image_height"].iloc[0]
                if pd.notna(image_height_val) and image_height_val > 0:
                    image_height = image_height_val
                else:
                    image_height = page_row["mediabox_height"].iloc[0]
            else:
                image_height = page_row["mediabox_height"].iloc[0]
        except (KeyError, IndexError):
            image_height = page_row["mediabox_height"].iloc[0]

        # Get question and answer text safely
        question_text = qa_result.get("Question", "")
        answer_text = qa_result.get("Answer", "")

        # Get scores and handle potential type issues
        question_score = float(qa_result.get("'Confidence Score % (Question)'", 0.0))
        answer_score = float(qa_result.get("'Confidence Score % (Answer)'", 0.0))

        # --- Process Question Bounding Box ---
        question_bbox = {
            "left": qa_result.get("Question_left", 0) * image_width,
            "top": qa_result.get("Question_top", 0) * image_height,
            "width": qa_result.get("Question_width", 0) * image_width,
            "height": qa_result.get("Question_height", 0) * image_height,
        }

        question_result = CustomImageRecognizerResult(
            entity_type=f"QUESTION {i+1}",
            start=0,
            end=len(question_text),
            score=question_score,
            left=float(question_bbox.get("left", 0)),
            top=float(question_bbox.get("top", 0)),
            width=float(question_bbox.get("width", 0)),
            height=float(question_bbox.get("height", 0)),
            text=question_text,
            color=(0, 0, 255),
        )
        results.append(question_result)

        # --- Process Answer Bounding Box ---
        answer_bbox = {
            "left": qa_result.get("Answer_left", 0) * image_width,
            "top": qa_result.get("Answer_top", 0) * image_height,
            "width": qa_result.get("Answer_width", 0) * image_width,
            "height": qa_result.get("Answer_height", 0) * image_height,
        }

        answer_result = CustomImageRecognizerResult(
            entity_type=f"ANSWER {i+1}",
            start=0,
            end=len(answer_text),
            score=answer_score,
            left=float(answer_bbox.get("left", 0)),
            top=float(answer_bbox.get("top", 0)),
            width=float(answer_bbox.get("width", 0)),
            height=float(answer_bbox.get("height", 0)),
            text=answer_text,
            color=(0, 255, 0),
        )
        results.append(answer_result)

    return results
