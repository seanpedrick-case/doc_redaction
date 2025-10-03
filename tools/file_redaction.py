import copy
import io
import json
import os
import time
from collections import defaultdict  # For efficient grouping
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3
import gradio as gr
import pandas as pd
import pymupdf
from gradio import Progress
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTAnno,
    LTTextContainer,
    LTTextLine,
    LTTextLineHorizontal,
)
from pikepdf import Dictionary, Name, Pdf
from PIL import Image, ImageDraw, ImageFile
from presidio_analyzer import AnalyzerEngine
from pymupdf import Document, Page, Rect
from tqdm import tqdm

from tools.aws_textract import (
    analyse_page_with_textract,
    convert_page_question_answer_to_custom_image_recognizer_results,
    convert_question_answer_to_dataframe,
    json_to_ocrresult,
    load_and_convert_textract_json,
)
from tools.config import (
    APPLY_REDACTIONS_GRAPHICS,
    APPLY_REDACTIONS_IMAGES,
    APPLY_REDACTIONS_TEXT,
    AWS_ACCESS_KEY,
    AWS_PII_OPTION,
    AWS_REGION,
    AWS_SECRET_KEY,
    CUSTOM_BOX_COLOUR,
    CUSTOM_ENTITIES,
    DEFAULT_LANGUAGE,
    IMAGES_DPI,
    INPUT_FOLDER,
    LOAD_TRUNCATED_IMAGES,
    MAX_DOC_PAGES,
    MAX_IMAGE_PIXELS,
    MAX_SIMULTANEOUS_FILES,
    MAX_TIME_VALUE,
    NO_REDACTION_PII_OPTION,
    OUTPUT_FOLDER,
    PAGE_BREAK_VALUE,
    PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS,
    RETURN_PDF_FOR_REVIEW,
    RETURN_REDACTED_PDF,
    RUN_AWS_FUNCTIONS,
    SELECTABLE_TEXT_EXTRACT_OPTION,
    TESSERACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_TEXT_EXTRACT_OPTION,
    USE_GUI_BOX_COLOURS_FOR_OUTPUTS,
    aws_comprehend_language_choices,
    textract_language_choices,
)
from tools.custom_image_analyser_engine import (
    CustomImageAnalyzerEngine,
    CustomImageRecognizerResult,
    OCRResult,
    combine_ocr_results,
    recreate_page_line_level_ocr_results_with_page,
    run_page_text_redaction,
)
from tools.file_conversion import (
    convert_annotation_data_to_dataframe,
    convert_annotation_json_to_review_df,
    create_annotation_dicts_from_annotation_df,
    divide_coordinates_by_page_sizes,
    fill_missing_box_ids,
    fill_missing_ids,
    is_pdf,
    is_pdf_or_image,
    load_and_convert_ocr_results_with_words_json,
    prepare_image_or_pdf,
    process_single_page_for_image_conversion,
    remove_duplicate_images_with_blank_boxes,
    save_pdf_with_or_without_compression,
    word_level_ocr_output_to_dataframe,
)
from tools.helper_functions import (
    clean_unicode_text,
    get_file_name_without_type,
)
from tools.load_spacy_model_custom_recognisers import (
    CustomWordFuzzyRecognizer,
    create_nlp_analyser,
    custom_word_list_recogniser,
    download_tesseract_lang_pack,
    load_spacy_model,
    nlp_analyser,
    score_threshold,
)
from tools.secure_path_utils import (
    secure_file_write,
    validate_path_containment,
)

ImageFile.LOAD_TRUNCATED_IMAGES = LOAD_TRUNCATED_IMAGES.lower() == "true"
if not MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = None
else:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
image_dpi = float(IMAGES_DPI)

custom_entities = CUSTOM_ENTITIES


def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (
        box1[0] < box2[2]
        and box2[0] < box1[2]
        and box1[1] < box2[3]
        and box2[1] < box1[3]
    )


def sum_numbers_before_seconds(string: str):
    """Extracts numbers that precede the word 'seconds' from a string and adds them up.

    Args:
        string: The input string.

    Returns:
        The sum of all numbers before 'seconds' in the string.
    """

    # Extract numbers before 'seconds' using secure regex
    from tools.secure_regex_utils import safe_extract_numbers_with_seconds

    numbers = safe_extract_numbers_with_seconds(string)

    # Sum up the extracted numbers
    sum_of_numbers = round(sum(numbers), 1)

    return sum_of_numbers


def reverse_y_coords(df: pd.DataFrame, column: str):
    df[column] = df[column]
    df[column] = 1 - df[column].astype(float)

    df[column] = df[column].round(6)

    return df[column]


def merge_page_results(data: list):
    merged = dict()

    for item in data:
        page = item["page"]

        if page not in merged:
            merged[page] = {"page": page, "results": {}}

        # Merge line-level results into the existing page
        merged[page]["results"].update(item.get("results", {}))

    return list(merged.values())


def choose_and_run_redactor(
    file_paths: List[str],
    prepared_pdf_file_paths: List[str],
    pdf_image_file_paths: List[str],
    chosen_redact_entities: List[str],
    chosen_redact_comprehend_entities: List[str],
    text_extraction_method: str,
    in_allow_list: List[str] = list(),
    in_deny_list: List[str] = list(),
    redact_whole_page_list: List[str] = list(),
    latest_file_completed: int = 0,
    combined_out_message: List = list(),
    out_file_paths: List = list(),
    log_files_output_paths: List = list(),
    first_loop_state: bool = False,
    page_min: int = 0,
    page_max: int = 999,
    estimated_time_taken_state: float = 0.0,
    handwrite_signature_checkbox: List[str] = list(["Extract handwriting"]),
    all_request_metadata_str: str = "",
    annotations_all_pages: List[dict] = list(),
    all_page_line_level_ocr_results_df: pd.DataFrame = None,
    all_pages_decision_process_table: pd.DataFrame = None,
    pymupdf_doc=list(),
    current_loop_page: int = 0,
    page_break_return: bool = False,
    pii_identification_method: str = "Local",
    comprehend_query_number: int = 0,
    max_fuzzy_spelling_mistakes_num: int = 1,
    match_fuzzy_whole_phrase_bool: bool = True,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    annotate_max_pages: int = 1,
    review_file_state: pd.DataFrame = list(),
    output_folder: str = OUTPUT_FOLDER,
    document_cropboxes: List = list(),
    page_sizes: List[dict] = list(),
    textract_output_found: bool = False,
    text_extraction_only: bool = False,
    duplication_file_path_outputs: list = list(),
    review_file_path: str = "",
    input_folder: str = INPUT_FOLDER,
    total_textract_query_number: int = 0,
    ocr_file_path: str = "",
    all_page_line_level_ocr_results: list[dict] = list(),
    all_page_line_level_ocr_results_with_words: list[dict] = list(),
    all_page_line_level_ocr_results_with_words_df: pd.DataFrame = None,
    chosen_local_model: str = "tesseract",
    language: str = DEFAULT_LANGUAGE,
    ocr_review_files: list = list(),
    prepare_images: bool = True,
    RETURN_REDACTED_PDF: bool = RETURN_REDACTED_PDF,
    RETURN_PDF_FOR_REVIEW: bool = RETURN_PDF_FOR_REVIEW,
    progress=gr.Progress(track_tqdm=True),
):
    """
    This function orchestrates the redaction process based on the specified method and parameters. It takes the following inputs:

    - file_paths (List[str]): A list of paths to the files to be redacted.
    - prepared_pdf_file_paths (List[str]): A list of paths to the PDF files prepared for redaction.
    - pdf_image_file_paths (List[str]): A list of paths to the PDF files converted to images for redaction.

    - chosen_redact_entities (List[str]): A list of entity types to redact from the files using the local model (spacy) with Microsoft Presidio.
    - chosen_redact_comprehend_entities (List[str]): A list of entity types to redact from files, chosen from the official list from AWS Comprehend service.
    - text_extraction_method (str): The method to use to extract text from documents.
    - in_allow_list (List[List[str]], optional): A list of allowed terms for redaction. Defaults to empty list. Can also be entered as a string path to a CSV file, or as a single column pandas dataframe.
    - in_deny_list (List[List[str]], optional): A list of denied terms for redaction. Defaults to empty list. Can also be entered as a string path to a CSV file, or as a single column pandas dataframe.
    - redact_whole_page_list (List[List[str]], optional): A list of whole page numbers for redaction. Defaults to empty list. Can also be entered as a string path to a CSV file, or as a single column pandas dataframe.
    - latest_file_completed (int, optional): The index of the last completed file. Defaults to 0.
    - combined_out_message (list, optional): A list to store output messages. Defaults to an empty list.
    - out_file_paths (list, optional): A list to store paths to the output files. Defaults to an empty list.
    - log_files_output_paths (list, optional): A list to store paths to the log files. Defaults to an empty list.
    - first_loop_state (bool, optional): A flag indicating if this is the first iteration. Defaults to False.
    - page_min (int, optional): The minimum page number to start redaction from. Defaults to 0.
    - page_max (int, optional): The maximum page number to end redaction at. Defaults to 999.
    - estimated_time_taken_state (float, optional): The estimated time taken for the redaction process. Defaults to 0.0.
    - handwrite_signature_checkbox (List[str], optional): A list of options for redacting handwriting and signatures. Defaults to ["Extract handwriting", "Extract signatures"].
    - all_request_metadata_str (str, optional): A string containing all request metadata. Defaults to an empty string.
    - annotations_all_pages (List[dict], optional): A list of dictionaries containing all image annotations. Defaults to an empty list.
    - all_page_line_level_ocr_results_df (pd.DataFrame, optional): A DataFrame containing all line-level OCR results. Defaults to an empty DataFrame.
    - all_pages_decision_process_table (pd.DataFrame, optional): A DataFrame containing all decision process tables. Defaults to an empty DataFrame.
    - pymupdf_doc (optional): A list containing the PDF document object. Defaults to an empty list.
    - current_loop_page (int, optional): The current page being processed in the loop. Defaults to 0.
    - page_break_return (bool, optional): A flag indicating if the function should return after a page break. Defaults to False.
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - max_fuzzy_spelling_mistakes_num (int, optional): The maximum number of spelling mistakes allowed in a searched phrase for fuzzy matching. Can range from 0-9.
    - match_fuzzy_whole_phrase_bool (bool, optional): A boolean where 'True' means that the whole phrase is fuzzy matched, and 'False' means that each word is fuzzy matched separately (excluding stop words).
    - aws_access_key_textbox (str, optional): AWS access key for account with Textract and Comprehend permissions.
    - aws_secret_key_textbox (str, optional): AWS secret key for account with Textract and Comprehend permissions.
    - annotate_max_pages (int, optional): Maximum page value for the annotation object.
    - review_file_state (pd.DataFrame, optional): Output review file dataframe.
    - output_folder (str, optional): Output folder for results.
    - document_cropboxes (List, optional): List of document cropboxes for the PDF.
    - page_sizes (List[dict], optional): List of dictionaries of PDF page sizes in PDF or image format.
    - textract_output_found (bool, optional): Boolean is true when a textract OCR output for the file has been found.
    - text_extraction_only (bool, optional): Boolean to determine if function should only extract text from the document, and not redact.
    - duplication_file_outputs (list, optional): List to allow for export to the duplication function page.
    - review_file_path (str, optional): The latest review file path created by the app
    - input_folder (str, optional): The custom input path, if provided
    - total_textract_query_number (int, optional): The number of textract queries up until this point.
    - ocr_file_path (str, optional): The latest ocr file path created by the app.
    - all_page_line_level_ocr_results (list, optional): All line level text on the page with bounding boxes.
    - all_page_line_level_ocr_results_with_words (list, optional): All word level text on the page with bounding boxes.
    - all_page_line_level_ocr_results_with_words_df (pd.Dataframe, optional): All word level text on the page with bounding boxes as a dataframe.
    - chosen_local_model (str): Which local model is being used for OCR on images - "tesseract", "paddle" for PaddleOCR, or "hybrid" to combine both.
    - language (str, optional): The language of the text in the files. Defaults to English.
    - language (str, optional): The language to do AWS Comprehend calls. Defaults to value of language if not provided.
    - ocr_review_files (list, optional): A list of OCR review files to be used for the redaction process. Defaults to an empty list.
    - prepare_images (bool, optional): Boolean to determine whether to load images for the PDF.
    - RETURN_REDACTED_PDF (bool, optional): Boolean to determine whether to return a redacted PDF at the end of the redaction process.
    - progress (gr.Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.
    - RETURN_PDF_FOR_REVIEW (bool, optional): Boolean to determine whether to return a review PDF at the end of the redaction process.
    The function returns a redacted document along with processing logs. If both RETURN_PDF_FOR_REVIEW and RETURN_REDACTED_PDF
    are True, the function will return both a review PDF (with annotation boxes for review) and a final redacted PDF (with text permanently removed).
    """
    tic = time.perf_counter()

    out_message = ""
    pdf_file_name_with_ext = ""
    pdf_file_name_without_ext = ""
    page_break_return = False
    blank_request_metadata = list()
    custom_recogniser_word_list_flat = list()
    all_textract_request_metadata = (
        all_request_metadata_str.split("\n") if all_request_metadata_str else []
    )

    task_textbox = "redact"
    selection_element_results_list_df = pd.DataFrame()
    form_key_value_results_list_df = pd.DataFrame()
    out_review_pdf_file_path = ""
    out_redacted_pdf_file_path = ""
    if not ocr_review_files:
        ocr_review_files = list()

    # CLI mode may provide options to enter method names in a different format
    if text_extraction_method == "AWS Textract":
        text_extraction_method = TEXTRACT_TEXT_EXTRACT_OPTION
    if text_extraction_method == "Local OCR":
        text_extraction_method = TESSERACT_TEXT_EXTRACT_OPTION
    if text_extraction_method == "Local text":
        text_extraction_method = SELECTABLE_TEXT_EXTRACT_OPTION
    if pii_identification_method == "None":
        pii_identification_method = NO_REDACTION_PII_OPTION

    # If output folder doesn't end with a forward slash, add one
    if not output_folder.endswith("/"):
        output_folder = output_folder + "/"

    # Use provided language or default
    language = language or DEFAULT_LANGUAGE

    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        if language not in textract_language_choices:
            out_message = f"Language '{language}' is not supported by AWS Textract. Please select a different language."
            raise Warning(out_message)
    elif pii_identification_method == AWS_PII_OPTION:
        if language not in aws_comprehend_language_choices:
            out_message = f"Language '{language}' is not supported by AWS Comprehend. Please select a different language."
            raise Warning(out_message)

    if all_page_line_level_ocr_results_with_words_df is None:
        all_page_line_level_ocr_results_with_words_df = pd.DataFrame()

    # Create copies of out_file_path objects to avoid overwriting each other on append actions
    out_file_paths = out_file_paths.copy()
    log_files_output_paths = log_files_output_paths.copy()

    # Ensure all_pages_decision_process_table is in correct format for downstream processes
    if isinstance(all_pages_decision_process_table, list):
        if not all_pages_decision_process_table:
            all_pages_decision_process_table = pd.DataFrame(
                columns=[
                    "image_path",
                    "page",
                    "label",
                    "xmin",
                    "xmax",
                    "ymin",
                    "ymax",
                    "boundingBox",
                    "text",
                    "start",
                    "end",
                    "score",
                    "id",
                ]
            )
    elif isinstance(all_pages_decision_process_table, pd.DataFrame):
        if all_pages_decision_process_table.empty:
            all_pages_decision_process_table = pd.DataFrame(
                columns=[
                    "image_path",
                    "page",
                    "label",
                    "xmin",
                    "xmax",
                    "ymin",
                    "ymax",
                    "boundingBox",
                    "text",
                    "start",
                    "end",
                    "score",
                    "id",
                ]
            )

    # If this is the first time around, set variables to 0/blank
    if first_loop_state is True:
        # print("First_loop_state is True")
        latest_file_completed = 0
        current_loop_page = 0
        out_file_paths = list()
        log_files_output_paths = list()
        estimate_total_processing_time = 0
        estimated_time_taken_state = 0
        comprehend_query_number = 0
        total_textract_query_number = 0
    elif current_loop_page == 0:
        comprehend_query_number = 0
        total_textract_query_number = 0
    # If not the first time around, and the current page loop has been set to a huge number (been through all pages), reset current page to 0
    elif (first_loop_state is False) & (current_loop_page == 999):
        current_loop_page = 0
        total_textract_query_number = 0
        comprehend_query_number = 0

    if not file_paths:
        raise Exception("No files to redact")

    if prepared_pdf_file_paths:
        review_out_file_paths = [prepared_pdf_file_paths[0]]
    else:
        review_out_file_paths = list()

    # Choose the correct file to prepare
    if isinstance(file_paths, str):
        file_paths_list = [os.path.abspath(file_paths)]
    elif isinstance(file_paths, dict):
        file_paths = file_paths["name"]
        file_paths_list = [os.path.abspath(file_paths)]
    else:
        file_paths_list = file_paths

    if len(file_paths_list) > MAX_SIMULTANEOUS_FILES:
        out_message = f"Number of files to redact is greater than {MAX_SIMULTANEOUS_FILES}. Please submit a smaller number of files."
        print(out_message)
        raise Exception(out_message)

    valid_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    # Filter only files with valid extensions. Currently only allowing one file to be redacted at a time
    # Filter the file_paths_list to include only files with valid extensions
    filtered_files = [
        file
        for file in file_paths_list
        if os.path.splitext(file)[1].lower() in valid_extensions
    ]

    # Check if any files were found and assign to file_paths_list
    file_paths_list = filtered_files if filtered_files else []

    print("Latest file completed:", latest_file_completed)

    # If latest_file_completed is used, get the specific file
    if not isinstance(file_paths, (str, dict)):
        file_paths_loop = (
            [file_paths_list[int(latest_file_completed)]]
            if len(file_paths_list) > latest_file_completed
            else []
        )
    else:
        file_paths_loop = file_paths_list

    latest_file_completed = int(latest_file_completed)

    if isinstance(file_paths, str):
        number_of_files = 1
    else:
        number_of_files = len(file_paths_list)

    # If we have already redacted the last file, return the input out_message and file list to the relevant outputs
    if latest_file_completed >= number_of_files:

        print("Completed last file")
        progress(0.95, "Completed last file, performing final checks")
        current_loop_page = 0

        if isinstance(combined_out_message, list):
            combined_out_message = "\n".join(combined_out_message)

        if isinstance(out_message, list) and out_message:
            combined_out_message = combined_out_message + "\n".join(out_message)
        elif out_message:
            combined_out_message = combined_out_message + "\n" + out_message

        from tools.secure_regex_utils import safe_remove_leading_newlines

        combined_out_message = safe_remove_leading_newlines(combined_out_message)

        end_message = "\n\nPlease review and modify the suggested redaction outputs on the 'Review redactions' tab of the app (you can find this under the introduction text at the top of the page)."

        if end_message not in combined_out_message:
            combined_out_message = combined_out_message + end_message

        # Only send across review file if redaction has been done
        if pii_identification_method != NO_REDACTION_PII_OPTION:

            if len(review_out_file_paths) == 1:
                if review_file_path:
                    review_out_file_paths.append(review_file_path)

        if not isinstance(pymupdf_doc, list):
            number_of_pages = pymupdf_doc.page_count
            if total_textract_query_number > number_of_pages:
                total_textract_query_number = number_of_pages

        estimate_total_processing_time = sum_numbers_before_seconds(
            combined_out_message
        )
        print("Estimated total processing time:", str(estimate_total_processing_time))

        gr.Info(combined_out_message)

        page_break_return = True

        return (
            combined_out_message,
            out_file_paths,
            out_file_paths,
            latest_file_completed,
            log_files_output_paths,
            log_files_output_paths,
            estimated_time_taken_state,
            all_request_metadata_str,
            pymupdf_doc,
            annotations_all_pages,
            current_loop_page,
            page_break_return,
            all_page_line_level_ocr_results_df,
            all_pages_decision_process_table,
            comprehend_query_number,
            review_out_file_paths,
            annotate_max_pages,
            annotate_max_pages,
            prepared_pdf_file_paths,
            pdf_image_file_paths,
            review_file_state,
            page_sizes,
            duplication_file_path_outputs,
            duplication_file_path_outputs,
            review_file_path,
            total_textract_query_number,
            ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df,
            review_file_state,
            task_textbox,
            ocr_review_files,
        )

    # if first_loop_state == False:
    # Prepare documents and images as required if they don't already exist
    prepare_images_flag = None  # Determines whether to call prepare_image_or_pdf

    if textract_output_found and text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        print("Existing Textract outputs found, not preparing images or documents.")
        prepare_images_flag = False
        # return  # No need to call `prepare_image_or_pdf`, exit early

    elif text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:
        print("Running text extraction analysis, not preparing images.")
        prepare_images_flag = False

    elif prepare_images and not pdf_image_file_paths:
        print("Prepared PDF images not found, loading from file")
        prepare_images_flag = True

    elif not prepare_images:
        print("Not loading images for file")
        prepare_images_flag = False

    else:
        print("Loading images for file")
        prepare_images_flag = True

    # Call prepare_image_or_pdf only if needed
    if prepare_images_flag is not None:
        (
            out_message,
            prepared_pdf_file_paths,
            pdf_image_file_paths,
            annotate_max_pages,
            annotate_max_pages_bottom,
            pymupdf_doc,
            annotations_all_pages,
            review_file_state,
            document_cropboxes,
            page_sizes,
            textract_output_found,
            all_img_details_state,
            placeholder_ocr_results_df,
            local_ocr_output_found_checkbox,
            all_page_line_level_ocr_results_with_words_df,
        ) = prepare_image_or_pdf(
            file_paths_loop,
            text_extraction_method,
            all_page_line_level_ocr_results_df,
            all_page_line_level_ocr_results_with_words_df,
            0,
            out_message,
            True,
            annotate_max_pages,
            annotations_all_pages,
            document_cropboxes,
            redact_whole_page_list,
            output_folder=output_folder,
            prepare_images=prepare_images_flag,
            page_sizes=page_sizes,
            pymupdf_doc=pymupdf_doc,
            input_folder=input_folder,
        )

    page_sizes_df = pd.DataFrame(page_sizes)

    if page_sizes_df.empty:
        page_sizes_df = pd.DataFrame(
            columns=[
                "page",
                "image_path",
                "image_width",
                "image_height",
                "mediabox_width",
                "mediabox_height",
                "cropbox_width",
                "cropbox_height",
                "original_cropbox",
            ]
        )
    page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(
        pd.to_numeric, errors="coerce"
    )

    page_sizes = page_sizes_df.to_dict(orient="records")

    number_of_pages = pymupdf_doc.page_count

    if number_of_pages > MAX_DOC_PAGES:
        out_message = f"Number of pages in document is greater than {MAX_DOC_PAGES}. Please submit a smaller document."
        print(out_message)
        raise Exception(out_message)

    # If we have reached the last page, return message and outputs
    if current_loop_page >= number_of_pages:
        print("Reached last page of document:", current_loop_page)

        if total_textract_query_number > number_of_pages:
            total_textract_query_number = number_of_pages

        # Set to a very high number so as not to mix up with subsequent file processing by the user
        current_loop_page = 999
        if out_message:
            combined_out_message = combined_out_message + "\n" + out_message

        # Only send across review file if redaction has been done
        if pii_identification_method != NO_REDACTION_PII_OPTION:
            # If only pdf currently in review outputs, add on the latest review file
            if len(review_out_file_paths) == 1:
                if review_file_path:
                    review_out_file_paths.append(review_file_path)

        page_break_return = False

        return (
            combined_out_message,
            out_file_paths,
            out_file_paths,
            latest_file_completed,
            log_files_output_paths,
            log_files_output_paths,
            estimated_time_taken_state,
            all_request_metadata_str,
            pymupdf_doc,
            annotations_all_pages,
            current_loop_page,
            page_break_return,
            all_page_line_level_ocr_results_df,
            all_pages_decision_process_table,
            comprehend_query_number,
            review_out_file_paths,
            annotate_max_pages,
            annotate_max_pages,
            prepared_pdf_file_paths,
            pdf_image_file_paths,
            review_file_state,
            page_sizes,
            duplication_file_path_outputs,
            duplication_file_path_outputs,
            review_file_path,
            total_textract_query_number,
            ocr_file_path,
            all_page_line_level_ocr_results,
            all_page_line_level_ocr_results_with_words,
            all_page_line_level_ocr_results_with_words_df,
            review_file_state,
            task_textbox,
            ocr_review_files,
        )

    ### Load/create allow list, deny list, and whole page redaction list

    ### Load/create allow list
    # If string, assume file path
    if isinstance(in_allow_list, str):
        if in_allow_list:
            in_allow_list = pd.read_csv(in_allow_list, header=None)
    # Now, should be a pandas dataframe format
    if isinstance(in_allow_list, pd.DataFrame):
        if not in_allow_list.empty:
            in_allow_list_flat = in_allow_list.iloc[:, 0].tolist()
        else:
            in_allow_list_flat = list()
    else:
        in_allow_list_flat = list()

    ### Load/create deny list
    # If string, assume file path
    if isinstance(in_deny_list, str):
        if in_deny_list:
            in_deny_list = pd.read_csv(in_deny_list, header=None)

    if isinstance(in_deny_list, pd.DataFrame):
        if not in_deny_list.empty:
            custom_recogniser_word_list_flat = in_deny_list.iloc[:, 0].tolist()
        else:
            custom_recogniser_word_list_flat = list()
        # Sort the strings in order from the longest string to the shortest
        custom_recogniser_word_list_flat = sorted(
            custom_recogniser_word_list_flat, key=len, reverse=True
        )
    else:
        custom_recogniser_word_list_flat = list()

    ### Load/create whole page redaction list
    # If string, assume file path
    if isinstance(redact_whole_page_list, str):
        if redact_whole_page_list:
            redact_whole_page_list = pd.read_csv(redact_whole_page_list, header=None)
    if isinstance(redact_whole_page_list, pd.DataFrame):
        if not redact_whole_page_list.empty:
            try:
                redact_whole_page_list_flat = (
                    redact_whole_page_list.iloc[:, 0].astype(int).tolist()
                )
            except Exception as e:
                print(
                    "Could not convert whole page redaction data to number list due to:",
                    e,
                )
                redact_whole_page_list_flat = redact_whole_page_list.iloc[:, 0].tolist()
        else:
            redact_whole_page_list_flat = list()
    else:
        redact_whole_page_list_flat = list()

    ### Load/create PII identification method

    # Try to connect to AWS services directly only if RUN_AWS_FUNCTIONS environmental variable is 1, otherwise an environment variable or direct textbox input is needed.
    if pii_identification_method == AWS_PII_OPTION:
        if RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
            print("Connecting to Comprehend via existing SSO connection")
            comprehend_client = boto3.client("comprehend", region_name=AWS_REGION)
        elif aws_access_key_textbox and aws_secret_key_textbox:
            print(
                "Connecting to Comprehend using AWS access key and secret keys from user input."
            )
            comprehend_client = boto3.client(
                "comprehend",
                aws_access_key_id=aws_access_key_textbox,
                aws_secret_access_key=aws_secret_key_textbox,
                region_name=AWS_REGION,
            )
        elif RUN_AWS_FUNCTIONS == "1":
            print("Connecting to Comprehend via existing SSO connection")
            comprehend_client = boto3.client("comprehend", region_name=AWS_REGION)
        elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
            print("Getting Comprehend credentials from environment variables")
            comprehend_client = boto3.client(
                "comprehend",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
            )
        else:
            comprehend_client = ""
            out_message = "Cannot connect to AWS Comprehend service. Please provide access keys under Textract settings on the Redaction settings tab, or choose another PII identification method."
            print(out_message)
            raise Exception(out_message)
    else:
        comprehend_client = ""

    # Try to connect to AWS Textract Client if using that text extraction method
    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        if RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
            print("Connecting to Textract via existing SSO connection")
            textract_client = boto3.client("textract", region_name=AWS_REGION)
        elif aws_access_key_textbox and aws_secret_key_textbox:
            print(
                "Connecting to Textract using AWS access key and secret keys from user input."
            )
            textract_client = boto3.client(
                "textract",
                aws_access_key_id=aws_access_key_textbox,
                aws_secret_access_key=aws_secret_key_textbox,
                region_name=AWS_REGION,
            )
        elif RUN_AWS_FUNCTIONS == "1":
            print("Connecting to Textract via existing SSO connection")
            textract_client = boto3.client("textract", region_name=AWS_REGION)
        elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
            print("Getting Textract credentials from environment variables.")
            textract_client = boto3.client(
                "textract",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
            )
        elif textract_output_found is True:
            print(
                "Existing Textract data found for file, no need to connect to AWS Textract"
            )
            textract_client = boto3.client("textract", region_name=AWS_REGION)
        else:
            textract_client = ""
            out_message = "Cannot connect to AWS Textract service."
            print(out_message)
            raise Exception(out_message)
    else:
        textract_client = ""

    ### Language check - check if selected language packs exist
    try:
        if (
            text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION
            and chosen_local_model == "tesseract"
        ):
            if language != "en":
                progress(
                    0.1, desc=f"Downloading Tesseract language pack for {language}"
                )
                download_tesseract_lang_pack(language)

        if language != "en":
            progress(0.1, desc=f"Loading SpaCy model for {language}")
        load_spacy_model(language)

    except Exception as e:
        print(f"Error downloading language packs for {language}: {e}")
        raise Exception(f"Error downloading language packs for {language}: {e}")

    # Check if output_folder exists, create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    progress(0.5, desc="Extracting text and redacting document")

    all_pages_decision_process_table = pd.DataFrame(
        columns=[
            "image_path",
            "page",
            "label",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "boundingBox",
            "text",
            "start",
            "end",
            "score",
            "id",
        ]
    )
    all_page_line_level_ocr_results_df = pd.DataFrame(
        columns=["page", "text", "left", "top", "width", "height", "line", "conf"]
    )

    # Run through file loop, redact each file at a time
    for file in file_paths_loop:

        # Get a string file path
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        if file_path:
            pdf_file_name_without_ext = get_file_name_without_type(file_path)
            pdf_file_name_with_ext = os.path.basename(file_path)

            is_a_pdf = is_pdf(file_path) is True
            if (
                is_a_pdf is False
                and text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION
            ):
                # If user has not submitted a pdf, assume it's an image
                print(
                    "File is not a PDF, assuming that image analysis needs to be used."
                )
                text_extraction_method = TESSERACT_TEXT_EXTRACT_OPTION
        else:
            out_message = "No file selected"
            print(out_message)
            raise Exception(out_message)

        # Output file paths names
        orig_pdf_file_path = output_folder + pdf_file_name_without_ext
        review_file_path = orig_pdf_file_path + "_review_file.csv"

        # Load in all_ocr_results_with_words if it exists as a file path and doesn't exist already
        # file_name = get_file_name_without_type(file_path)

        if text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:
            file_ending = "local_text"
        elif text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:
            file_ending = "local_ocr"
        elif text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
            file_ending = "textract"
        all_page_line_level_ocr_results_with_words_json_file_path = (
            output_folder
            + pdf_file_name_without_ext
            + "_ocr_results_with_words_"
            + file_ending
            + ".json"
        )

        if not all_page_line_level_ocr_results_with_words:
            if local_ocr_output_found_checkbox is True and os.path.exists(
                all_page_line_level_ocr_results_with_words_json_file_path
            ):
                (
                    all_page_line_level_ocr_results_with_words,
                    is_missing,
                    log_files_output_paths,
                ) = load_and_convert_ocr_results_with_words_json(
                    all_page_line_level_ocr_results_with_words_json_file_path,
                    log_files_output_paths,
                    page_sizes_df,
                )
                # original_all_page_line_level_ocr_results_with_words = all_page_line_level_ocr_results_with_words.copy()

        # Remove any existing review_file paths from the review file outputs
        if (
            text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION
            or text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION
        ):

            # Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) is False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                raise Exception(out_message)

            print(
                "Redacting file " + pdf_file_name_with_ext + " as an image-based file"
            )

            (
                pymupdf_doc,
                all_pages_decision_process_table,
                out_file_paths,
                new_textract_request_metadata,
                annotations_all_pages,
                current_loop_page,
                page_break_return,
                all_page_line_level_ocr_results_df,
                comprehend_query_number,
                all_page_line_level_ocr_results,
                all_page_line_level_ocr_results_with_words,
                selection_element_results_list_df,
                form_key_value_results_list_df,
            ) = redact_image_pdf(
                file_path,
                pdf_image_file_paths,
                language,
                chosen_redact_entities,
                chosen_redact_comprehend_entities,
                in_allow_list_flat,
                page_min,
                page_max,
                text_extraction_method,
                handwrite_signature_checkbox,
                blank_request_metadata,
                current_loop_page,
                page_break_return,
                annotations_all_pages,
                all_page_line_level_ocr_results_df,
                all_pages_decision_process_table,
                pymupdf_doc,
                pii_identification_method,
                comprehend_query_number,
                comprehend_client,
                textract_client,
                custom_recogniser_word_list_flat,
                redact_whole_page_list_flat,
                max_fuzzy_spelling_mistakes_num,
                match_fuzzy_whole_phrase_bool,
                page_sizes_df,
                text_extraction_only,
                all_page_line_level_ocr_results,
                all_page_line_level_ocr_results_with_words,
                chosen_local_model,
                log_files_output_paths=log_files_output_paths,
                nlp_analyser=nlp_analyser,
                output_folder=output_folder,
                input_folder=input_folder,
            )

            # This line creates a copy of out_file_paths to break potential links with log_files_output_paths
            out_file_paths = out_file_paths.copy()

            # Save Textract request metadata (if exists)
            if new_textract_request_metadata and isinstance(
                new_textract_request_metadata, list
            ):
                all_textract_request_metadata.extend(new_textract_request_metadata)

        elif text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:

            if is_pdf(file_path) is False:
                out_message = "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'."
                raise Exception(out_message)

            # Analyse text-based pdf
            print("Redacting file as text-based PDF")

            (
                pymupdf_doc,
                all_pages_decision_process_table,
                all_page_line_level_ocr_results_df,
                annotations_all_pages,
                current_loop_page,
                page_break_return,
                comprehend_query_number,
                all_page_line_level_ocr_results_with_words,
            ) = redact_text_pdf(
                file_path,
                language,
                chosen_redact_entities,
                chosen_redact_comprehend_entities,
                in_allow_list_flat,
                page_min,
                page_max,
                current_loop_page,
                page_break_return,
                annotations_all_pages,
                all_page_line_level_ocr_results_df,
                all_pages_decision_process_table,
                pymupdf_doc,
                all_page_line_level_ocr_results_with_words,
                pii_identification_method,
                comprehend_query_number,
                comprehend_client,
                custom_recogniser_word_list_flat,
                redact_whole_page_list_flat,
                max_fuzzy_spelling_mistakes_num,
                match_fuzzy_whole_phrase_bool,
                page_sizes_df,
                document_cropboxes,
                text_extraction_only,
                output_folder=output_folder,
                input_folder=input_folder,
            )
        else:
            out_message = "No redaction method selected"
            print(out_message)
            raise Exception(out_message)

        # If at last page, save to file
        if current_loop_page >= number_of_pages:

            print("Current page loop:", current_loop_page, "is the last page.")
            latest_file_completed += 1
            current_loop_page = 999

            if latest_file_completed != len(file_paths_list):
                print(
                    "Completed file number:",
                    str(latest_file_completed),
                    "there are more files to do",
                )

            # Save redacted file
            if pii_identification_method != NO_REDACTION_PII_OPTION:
                if RETURN_REDACTED_PDF is True:
                    progress(0.9, "Saving redacted file")

                    if is_pdf(file_path) is False:
                        out_redacted_pdf_file_path = (
                            output_folder + pdf_file_name_without_ext + "_redacted.png"
                        )
                        # pymupdf_doc is an image list in this case
                        if isinstance(pymupdf_doc[-1], str):
                            # Normalize and validate path safety before opening image
                            normalized_path = os.path.normpath(
                                os.path.abspath(pymupdf_doc[-1])
                            )
                            if validate_path_containment(normalized_path, INPUT_FOLDER):
                                img = Image.open(normalized_path)
                            else:
                                raise ValueError(
                                    f"Unsafe image path detected: {pymupdf_doc[-1]}"
                                )
                        # Otherwise could be an image object
                        else:
                            img = pymupdf_doc[-1]
                        img.save(
                            out_redacted_pdf_file_path, "PNG", resolution=image_dpi
                        )

                        if isinstance(out_redacted_pdf_file_path, str):
                            out_file_paths.append(out_redacted_pdf_file_path)
                        else:
                            out_file_paths.append(out_redacted_pdf_file_path[0])

                    else:
                        # Check if we have dual PDF documents to save
                        applied_redaction_pymupdf_doc = None

                        if RETURN_PDF_FOR_REVIEW and RETURN_REDACTED_PDF:
                            if (
                                hasattr(redact_image_pdf, "_applied_redaction_pages")
                                and redact_image_pdf._applied_redaction_pages
                            ):

                                # Create final document by copying the original document and replacing specific pages
                                applied_redaction_pymupdf_doc = pymupdf.open()
                                applied_redaction_pymupdf_doc.insert_pdf(pymupdf_doc)

                                # Create a mapping of original page numbers to final pages
                                applied_redaction_pages_map = {}
                                for (
                                    applied_redaction_page_data
                                ) in redact_image_pdf._applied_redaction_pages:
                                    if isinstance(applied_redaction_page_data, tuple):
                                        applied_redaction_page, original_page_number = (
                                            applied_redaction_page_data
                                        )
                                        applied_redaction_pages_map[
                                            original_page_number
                                        ] = applied_redaction_page
                                    else:
                                        applied_redaction_page = (
                                            applied_redaction_page_data
                                        )
                                        applied_redaction_pages_map[0] = (
                                            applied_redaction_page  # Default to page 0 if no original number
                                        )

                                # Replace pages in the final document with their final versions
                                for (
                                    original_page_number,
                                    applied_redaction_page,
                                ) in applied_redaction_pages_map.items():
                                    if (
                                        original_page_number
                                        < applied_redaction_pymupdf_doc.page_count
                                    ):
                                        # Remove the original page and insert the final page
                                        applied_redaction_pymupdf_doc.delete_page(
                                            original_page_number
                                        )
                                        applied_redaction_pymupdf_doc.insert_pdf(
                                            applied_redaction_page.parent,
                                            from_page=applied_redaction_page.number,
                                            to_page=applied_redaction_page.number,
                                            start_at=original_page_number,
                                        )

                                        applied_redaction_pymupdf_doc[
                                            original_page_number
                                        ].apply_redactions(
                                            images=APPLY_REDACTIONS_IMAGES,
                                            graphics=APPLY_REDACTIONS_GRAPHICS,
                                            text=APPLY_REDACTIONS_TEXT,
                                        )
                                # Clear the stored final pages
                                delattr(redact_image_pdf, "_applied_redaction_pages")
                            elif (
                                hasattr(redact_text_pdf, "_applied_redaction_pages")
                                and redact_text_pdf._applied_redaction_pages
                            ):
                                # Create final document by copying the original document and replacing specific pages
                                applied_redaction_pymupdf_doc = pymupdf.open()
                                applied_redaction_pymupdf_doc.insert_pdf(pymupdf_doc)

                                # Create a mapping of original page numbers to final pages
                                applied_redaction_pages_map = {}
                                for (
                                    applied_redaction_page_data
                                ) in redact_text_pdf._applied_redaction_pages:
                                    if isinstance(applied_redaction_page_data, tuple):
                                        applied_redaction_page, original_page_number = (
                                            applied_redaction_page_data
                                        )
                                        applied_redaction_pages_map[
                                            original_page_number
                                        ] = applied_redaction_page
                                    else:
                                        applied_redaction_page = (
                                            applied_redaction_page_data
                                        )
                                        applied_redaction_pages_map[0] = (
                                            applied_redaction_page  # Default to page 0 if no original number
                                        )

                                # Replace pages in the final document with their final versions
                                for (
                                    original_page_number,
                                    applied_redaction_page,
                                ) in applied_redaction_pages_map.items():
                                    if (
                                        original_page_number
                                        < applied_redaction_pymupdf_doc.page_count
                                    ):
                                        # Remove the original page and insert the final page
                                        applied_redaction_pymupdf_doc.delete_page(
                                            original_page_number
                                        )
                                        applied_redaction_pymupdf_doc.insert_pdf(
                                            applied_redaction_page.parent,
                                            from_page=applied_redaction_page.number,
                                            to_page=applied_redaction_page.number,
                                            start_at=original_page_number,
                                        )

                                        applied_redaction_pymupdf_doc[
                                            original_page_number
                                        ].apply_redactions(
                                            images=APPLY_REDACTIONS_IMAGES,
                                            graphics=APPLY_REDACTIONS_GRAPHICS,
                                            text=APPLY_REDACTIONS_TEXT,
                                        )
                                # Clear the stored final pages
                                delattr(redact_text_pdf, "_applied_redaction_pages")

                        # Save final redacted PDF if we have dual outputs or if RETURN_PDF_FOR_REVIEW is False
                        if (
                            RETURN_PDF_FOR_REVIEW is False
                            or applied_redaction_pymupdf_doc
                        ):
                            out_redacted_pdf_file_path = (
                                output_folder
                                + pdf_file_name_without_ext
                                + "_redacted.pdf"
                            )
                            print(
                                "Saving redacted PDF file:", out_redacted_pdf_file_path
                            )

                            # Use final document if available, otherwise use main document
                            doc_to_save = (
                                applied_redaction_pymupdf_doc
                                if applied_redaction_pymupdf_doc
                                else pymupdf_doc
                            )

                            if out_redacted_pdf_file_path:
                                save_pdf_with_or_without_compression(
                                    doc_to_save, out_redacted_pdf_file_path
                                )

                                if isinstance(out_redacted_pdf_file_path, str):
                                    out_file_paths.append(out_redacted_pdf_file_path)
                                else:
                                    out_file_paths.append(out_redacted_pdf_file_path[0])

            # Always return a file for review if a pdf is given and RETURN_PDF_FOR_REVIEW is True
            if is_pdf(file_path) is True:
                if RETURN_PDF_FOR_REVIEW is True:
                    out_review_pdf_file_path = (
                        output_folder
                        + pdf_file_name_without_ext
                        + "_redactions_for_review.pdf"
                    )
                    print("Saving PDF file for review:", out_review_pdf_file_path)

                    if out_review_pdf_file_path:
                        save_pdf_with_or_without_compression(
                            pymupdf_doc, out_review_pdf_file_path
                        )
                        if isinstance(out_review_pdf_file_path, str):
                            out_file_paths.append(out_review_pdf_file_path)
                        else:
                            out_file_paths.append(out_review_pdf_file_path[0])

            if not all_page_line_level_ocr_results_df.empty:
                all_page_line_level_ocr_results_df = all_page_line_level_ocr_results_df[
                    ["page", "text", "left", "top", "width", "height", "line", "conf"]
                ]
            else:
                all_page_line_level_ocr_results_df = pd.DataFrame(
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

            ocr_file_path = (
                output_folder
                + pdf_file_name_without_ext
                + "_ocr_output_"
                + file_ending
                + ".csv"
            )
            all_page_line_level_ocr_results_df.sort_values(
                ["page", "line"], inplace=True
            )
            all_page_line_level_ocr_results_df.to_csv(
                ocr_file_path, index=None, encoding="utf-8-sig"
            )

            if isinstance(ocr_file_path, str):
                out_file_paths.append(ocr_file_path)
            else:
                duplication_file_path_outputs.append(ocr_file_path[0])

            if all_page_line_level_ocr_results_with_words:
                all_page_line_level_ocr_results_with_words = merge_page_results(
                    all_page_line_level_ocr_results_with_words
                )

                with open(
                    all_page_line_level_ocr_results_with_words_json_file_path, "w"
                ) as json_file:
                    json.dump(
                        all_page_line_level_ocr_results_with_words,
                        json_file,
                        separators=(",", ":"),
                    )

                all_page_line_level_ocr_results_with_words_df = (
                    word_level_ocr_output_to_dataframe(
                        all_page_line_level_ocr_results_with_words
                    )
                )

                all_page_line_level_ocr_results_with_words_df = (
                    divide_coordinates_by_page_sizes(
                        all_page_line_level_ocr_results_with_words_df,
                        page_sizes_df,
                        xmin="word_x0",
                        xmax="word_x1",
                        ymin="word_y0",
                        ymax="word_y1",
                    )
                )

                if text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:
                    # Coordinates need to be reversed for ymin and ymax to match with image annotator objects downstream
                    if not all_page_line_level_ocr_results_with_words_df.empty:
                        all_page_line_level_ocr_results_with_words_df["word_y0"] = (
                            reverse_y_coords(
                                all_page_line_level_ocr_results_with_words_df, "word_y0"
                            )
                        )
                        all_page_line_level_ocr_results_with_words_df["word_y1"] = (
                            reverse_y_coords(
                                all_page_line_level_ocr_results_with_words_df, "word_y1"
                            )
                        )

                all_page_line_level_ocr_results_with_words_df["line_text"] = ""
                all_page_line_level_ocr_results_with_words_df["line_x0"] = ""
                all_page_line_level_ocr_results_with_words_df["line_x1"] = ""
                all_page_line_level_ocr_results_with_words_df["line_y0"] = ""
                all_page_line_level_ocr_results_with_words_df["line_y1"] = ""

                all_page_line_level_ocr_results_with_words_df.sort_values(
                    ["page", "line", "word_x0"], inplace=True
                )
                all_page_line_level_ocr_results_with_words_df_file_path = (
                    all_page_line_level_ocr_results_with_words_json_file_path.replace(
                        ".json", ".csv"
                    )
                )
                all_page_line_level_ocr_results_with_words_df.to_csv(
                    all_page_line_level_ocr_results_with_words_df_file_path,
                    index=None,
                    encoding="utf-8-sig",
                )

                if (
                    all_page_line_level_ocr_results_with_words_json_file_path
                    not in log_files_output_paths
                ):
                    if isinstance(
                        all_page_line_level_ocr_results_with_words_json_file_path, str
                    ):
                        log_files_output_paths.append(
                            all_page_line_level_ocr_results_with_words_json_file_path
                        )
                    else:
                        log_files_output_paths.append(
                            all_page_line_level_ocr_results_with_words_json_file_path[0]
                        )

                if (
                    all_page_line_level_ocr_results_with_words_df_file_path
                    not in log_files_output_paths
                ):
                    if isinstance(
                        all_page_line_level_ocr_results_with_words_df_file_path, str
                    ):
                        log_files_output_paths.append(
                            all_page_line_level_ocr_results_with_words_df_file_path
                        )
                    else:
                        log_files_output_paths.append(
                            all_page_line_level_ocr_results_with_words_df_file_path[0]
                        )

                if (
                    all_page_line_level_ocr_results_with_words_df_file_path
                    not in out_file_paths
                ):
                    if isinstance(
                        all_page_line_level_ocr_results_with_words_df_file_path, str
                    ):
                        out_file_paths.append(
                            all_page_line_level_ocr_results_with_words_df_file_path
                        )
                    else:
                        out_file_paths.append(
                            all_page_line_level_ocr_results_with_words_df_file_path[0]
                        )

            # Save decision process outputs
            if not all_pages_decision_process_table.empty:
                all_pages_decision_process_table_file_path = (
                    output_folder
                    + pdf_file_name_without_ext
                    + "_all_pages_decision_process_table_output_"
                    + file_ending
                    + ".csv"
                )
                all_pages_decision_process_table.to_csv(
                    all_pages_decision_process_table_file_path,
                    index=None,
                    encoding="utf-8-sig",
                )
                log_files_output_paths.append(
                    all_pages_decision_process_table_file_path
                )

            # Save outputs from form analysis if they exist
            if not selection_element_results_list_df.empty:
                selection_element_results_list_df_file_path = (
                    output_folder
                    + pdf_file_name_without_ext
                    + "_selection_element_results_output_"
                    + file_ending
                    + ".csv"
                )
                selection_element_results_list_df.to_csv(
                    selection_element_results_list_df_file_path,
                    index=None,
                    encoding="utf-8-sig",
                )
                out_file_paths.append(selection_element_results_list_df_file_path)

            if not form_key_value_results_list_df.empty:
                form_key_value_results_list_df_file_path = (
                    output_folder
                    + pdf_file_name_without_ext
                    + "_form_key_value_results_output_"
                    + file_ending
                    + ".csv"
                )
                form_key_value_results_list_df.to_csv(
                    form_key_value_results_list_df_file_path,
                    index=None,
                    encoding="utf-8-sig",
                )
                out_file_paths.append(form_key_value_results_list_df_file_path)

            # Convert the gradio annotation boxes to relative coordinates
            progress(0.93, "Creating review file output")
            page_sizes = page_sizes_df.to_dict(orient="records")
            all_image_annotations_df = convert_annotation_data_to_dataframe(
                annotations_all_pages
            )
            all_image_annotations_df = divide_coordinates_by_page_sizes(
                all_image_annotations_df,
                page_sizes_df,
                xmin="xmin",
                xmax="xmax",
                ymin="ymin",
                ymax="ymax",
            )
            annotations_all_pages_divide = create_annotation_dicts_from_annotation_df(
                all_image_annotations_df, page_sizes
            )
            annotations_all_pages_divide = remove_duplicate_images_with_blank_boxes(
                annotations_all_pages_divide
            )

            # Save the gradio_annotation_boxes to a review csv file
            review_file_state = convert_annotation_json_to_review_df(
                annotations_all_pages_divide,
                all_pages_decision_process_table,
                page_sizes=page_sizes,
            )

            # Don't need page sizes in outputs
            review_file_state.drop(
                [
                    "image_width",
                    "image_height",
                    "mediabox_width",
                    "mediabox_height",
                    "cropbox_width",
                    "cropbox_height",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )

            if (
                pii_identification_method == NO_REDACTION_PII_OPTION
                and not form_key_value_results_list_df.empty
            ):
                print(
                    "Form outputs found with no redaction method selected. Creating review file from form outputs."
                )
                review_file_state = form_key_value_results_list_df
                annotations_all_pages_divide = (
                    create_annotation_dicts_from_annotation_df(
                        review_file_state, page_sizes
                    )
                )

            if isinstance(review_file_path, str):
                review_file_state.to_csv(
                    review_file_path, index=None, encoding="utf-8-sig"
                )
            else:
                review_file_state.to_csv(
                    review_file_path[0], index=None, encoding="utf-8-sig"
                )

            if pii_identification_method != NO_REDACTION_PII_OPTION:
                if isinstance(review_file_path, str):
                    out_file_paths.append(review_file_path)
                else:
                    out_file_paths.append(review_file_path[0])

            # Make a combined message for the file
            if isinstance(combined_out_message, list):
                combined_out_message = "\n".join(combined_out_message)
            elif combined_out_message is None:
                combined_out_message = ""

            if isinstance(out_message, list) and out_message:
                combined_out_message = combined_out_message + "\n".join(out_message)
            elif isinstance(out_message, str) and out_message:
                combined_out_message = combined_out_message + "\n" + out_message

            toc = time.perf_counter()
            time_taken = toc - tic
            estimated_time_taken_state += time_taken

            out_time_message = (
                f" Redacted in {estimated_time_taken_state:0.1f} seconds."
            )
            combined_out_message = (
                combined_out_message + " " + out_time_message
            )  # Ensure this is a single string

            estimate_total_processing_time = sum_numbers_before_seconds(
                combined_out_message
            )

        else:
            toc = time.perf_counter()
            time_taken = toc - tic
            estimated_time_taken_state += time_taken

    # If textract requests made, write to logging file. Also record number of Textract requests
    if all_textract_request_metadata and isinstance(
        all_textract_request_metadata, list
    ):
        all_request_metadata_str = "\n".join(all_textract_request_metadata).strip()

        # all_textract_request_metadata_file_path is constructed by output_folder + filename
        # Split output_folder (trusted base) from pdf_file_name_without_ext + "_textract_metadata.txt" (untrusted)
        secure_file_write(
            output_folder,
            pdf_file_name_without_ext + "_textract_metadata.txt",
            all_request_metadata_str,
        )

        # Reconstruct the full path for logging purposes
        all_textract_request_metadata_file_path = (
            output_folder + pdf_file_name_without_ext + "_textract_metadata.txt"
        )

        # Add the request metadata to the log outputs if not there already
        if all_textract_request_metadata_file_path not in log_files_output_paths:
            if isinstance(all_textract_request_metadata_file_path, str):
                log_files_output_paths.append(all_textract_request_metadata_file_path)
            else:
                log_files_output_paths.append(
                    all_textract_request_metadata_file_path[0]
                )

        new_textract_query_numbers = len(all_textract_request_metadata)
        total_textract_query_number += new_textract_query_numbers

    # Ensure no duplicated output files
    log_files_output_paths = sorted(list(set(log_files_output_paths)))
    out_file_paths = sorted(list(set(out_file_paths)))

    # Create OCR review files list for input_review_files component

    if ocr_file_path:
        if isinstance(ocr_file_path, str):
            ocr_review_files.append(ocr_file_path)
        else:
            ocr_review_files.append(ocr_file_path[0])

    if all_page_line_level_ocr_results_with_words_df_file_path:
        if isinstance(all_page_line_level_ocr_results_with_words_df_file_path, str):
            ocr_review_files.append(
                all_page_line_level_ocr_results_with_words_df_file_path
            )
        else:
            ocr_review_files.append(
                all_page_line_level_ocr_results_with_words_df_file_path[0]
            )

    # Output file paths
    if not review_file_path:
        review_out_file_paths = [prepared_pdf_file_paths[-1]]
    else:
        review_out_file_paths = [prepared_pdf_file_paths[-1], review_file_path]

    if total_textract_query_number > number_of_pages:
        total_textract_query_number = number_of_pages

    page_break_return = True

    return (
        combined_out_message,
        out_file_paths,
        out_file_paths,
        latest_file_completed,
        log_files_output_paths,
        log_files_output_paths,
        estimated_time_taken_state,
        all_request_metadata_str,
        pymupdf_doc,
        annotations_all_pages_divide,
        current_loop_page,
        page_break_return,
        all_page_line_level_ocr_results_df,
        all_pages_decision_process_table,
        comprehend_query_number,
        review_out_file_paths,
        annotate_max_pages,
        annotate_max_pages,
        prepared_pdf_file_paths,
        pdf_image_file_paths,
        review_file_state,
        page_sizes,
        duplication_file_path_outputs,
        duplication_file_path_outputs,
        review_file_path,
        total_textract_query_number,
        ocr_file_path,
        all_page_line_level_ocr_results,
        all_page_line_level_ocr_results_with_words,
        all_page_line_level_ocr_results_with_words_df,
        review_file_state,
        task_textbox,
        ocr_review_files,
    )


def convert_pikepdf_coords_to_pymupdf(
    pymupdf_page: Page, pikepdf_bbox, type="pikepdf_annot"
):
    """
    Convert annotations from pikepdf to pymupdf format, handling the mediabox larger than rect.
    """
    # Use cropbox if available, otherwise use mediabox
    reference_box = pymupdf_page.rect
    mediabox = pymupdf_page.mediabox

    reference_box_height = reference_box.height
    reference_box_width = reference_box.width

    # Convert PyMuPDF coordinates back to PDF coordinates (bottom-left origin)
    media_height = mediabox.height
    media_width = mediabox.width

    media_reference_y_diff = media_height - reference_box_height
    media_reference_x_diff = media_width - reference_box_width

    y_diff_ratio = media_reference_y_diff / reference_box_height
    x_diff_ratio = media_reference_x_diff / reference_box_width

    # Extract the annotation rectangle field
    if type == "pikepdf_annot":
        rect_field = pikepdf_bbox["/Rect"]
    else:
        rect_field = pikepdf_bbox

    rect_coordinates = [float(coord) for coord in rect_field]  # Convert to floats

    # Unpack coordinates
    x1, y1, x2, y2 = rect_coordinates

    new_x1 = x1 - (media_reference_x_diff * x_diff_ratio)
    new_y1 = media_height - y2 - (media_reference_y_diff * y_diff_ratio)
    new_x2 = x2 - (media_reference_x_diff * x_diff_ratio)
    new_y2 = media_height - y1 - (media_reference_y_diff * y_diff_ratio)

    return new_x1, new_y1, new_x2, new_y2


def convert_pikepdf_to_image_coords(
    pymupdf_page, annot, image: Image, type="pikepdf_annot"
):
    """
    Convert annotations from pikepdf coordinates to image coordinates.
    """

    # Get the dimensions of the page in points with pymupdf
    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width

    # Get the dimensions of the image
    image_page_width, image_page_height = image.size

    # Calculate scaling factors between pymupdf and PIL image
    scale_width = image_page_width / rect_width
    scale_height = image_page_height / rect_height

    # Extract the /Rect field
    if type == "pikepdf_annot":
        rect_field = annot["/Rect"]
    else:
        rect_field = annot

    # Convert the extracted /Rect field to a list of floats
    rect_coordinates = [float(coord) for coord in rect_field]

    # Convert the Y-coordinates (flip using the image height)
    x1, y1, x2, y2 = rect_coordinates
    x1_image = x1 * scale_width
    new_y1_image = image_page_height - (
        y2 * scale_height
    )  # Flip Y0 (since it starts from bottom)
    x2_image = x2 * scale_width
    new_y2_image = image_page_height - (y1 * scale_height)  # Flip Y1

    return x1_image, new_y1_image, x2_image, new_y2_image


def convert_pikepdf_decision_output_to_image_coords(
    pymupdf_page: Document, pikepdf_decision_ouput_data: List[dict], image: Image
):
    if isinstance(image, str):
        # Normalize and validate path safety before opening image
        normalized_path = os.path.normpath(os.path.abspath(image))
        if validate_path_containment(normalized_path, INPUT_FOLDER):
            image_path = normalized_path
            image = Image.open(image_path)
        else:
            raise ValueError(f"Unsafe image path detected: {image}")

    # Loop through each item in the data
    for item in pikepdf_decision_ouput_data:
        # Extract the bounding box
        bounding_box = item["boundingBox"]

        # Create a pikepdf_bbox dictionary to match the expected input
        pikepdf_bbox = {"/Rect": bounding_box}

        # Call the conversion function
        new_x1, new_y1, new_x2, new_y2 = convert_pikepdf_to_image_coords(
            pymupdf_page, pikepdf_bbox, image, type="pikepdf_annot"
        )

        # Update the original object with the new bounding box values
        item["boundingBox"] = [new_x1, new_y1, new_x2, new_y2]

    return pikepdf_decision_ouput_data


def convert_image_coords_to_pymupdf(
    pymupdf_page: Document, annot: dict, image: Image, type: str = "image_recognizer"
):
    """
    Converts an image with redaction coordinates from a CustomImageRecognizerResult or pikepdf object with image coordinates to pymupdf coordinates.
    """

    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width

    image_page_width, image_page_height = image.size

    # Calculate scaling factors between PIL image and pymupdf
    scale_width = rect_width / image_page_width
    scale_height = rect_height / image_page_height

    # Calculate scaled coordinates
    if type == "image_recognizer":
        x1 = annot.left * scale_width  # + page_x_adjust
        new_y1 = (
            annot.top * scale_height
        )  # - page_y_adjust  # Flip Y0 (since it starts from bottom)
        x2 = (annot.left + annot.width) * scale_width  # + page_x_adjust  # Calculate x1
        new_y2 = (
            annot.top + annot.height
        ) * scale_height  # - page_y_adjust  # Calculate y1 correctly
    # Else assume it is a pikepdf derived object
    else:
        rect_field = annot["/Rect"]
        rect_coordinates = [float(coord) for coord in rect_field]  # Convert to floats

        # Unpack coordinates
        x1, y1, x2, y2 = rect_coordinates

        x1 = x1 * scale_width  # + page_x_adjust
        new_y1 = (
            y2 + (y1 - y2)
        ) * scale_height  # - page_y_adjust  # Calculate y1 correctly
        x2 = (x1 + (x2 - x1)) * scale_width  # + page_x_adjust  # Calculate x1
        new_y2 = (
            y2 * scale_height
        )  # - page_y_adjust  # Flip Y0 (since it starts from bottom)

    return x1, new_y1, x2, new_y2


def convert_gradio_image_annotator_object_coords_to_pymupdf(
    pymupdf_page: Page, annot: dict, image: Image, image_dimensions: dict = None
):
    """
    Converts an image with redaction coordinates from a gradio annotation component to pymupdf coordinates.
    """

    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width

    if image_dimensions:
        image_page_width = image_dimensions["image_width"]
        image_page_height = image_dimensions["image_height"]
    elif image:
        image_page_width, image_page_height = image.size

    # Calculate scaling factors between PIL image and pymupdf
    scale_width = rect_width / image_page_width
    scale_height = rect_height / image_page_height

    # Calculate scaled coordinates
    x1 = annot["xmin"] * scale_width  # + page_x_adjust
    new_y1 = (
        annot["ymin"] * scale_height
    )  # - page_y_adjust  # Flip Y0 (since it starts from bottom)
    x2 = (annot["xmax"]) * scale_width  # + page_x_adjust  # Calculate x1
    new_y2 = (annot["ymax"]) * scale_height  # - page_y_adjust  # Calculate y1 correctly

    return x1, new_y1, x2, new_y2


def move_page_info(file_path: str) -> str:
    # Split the string at '.png'
    base, extension = file_path.rsplit(".pdf", 1)

    # Extract the page info
    page_info = base.split("page ")[1].split(" of")[0]  # Get the page number
    new_base = base.replace(
        f"page {page_info} of ", ""
    )  # Remove the page info from the original position

    # Construct the new file path
    new_file_path = f"{new_base}_page_{page_info}.png"

    return new_file_path


def prepare_custom_image_recogniser_result_annotation_box(
    page: Page,
    annot: CustomImageRecognizerResult,
    image: Image,
    page_sizes_df: pd.DataFrame,
    custom_colours: bool = USE_GUI_BOX_COLOURS_FOR_OUTPUTS,
):
    """
    Prepare an image annotation box and coordinates based on a CustomImageRecogniserResult, PyMuPDF page, and PIL Image.
    """

    img_annotation_box = dict()

    # For efficient lookup, set 'page' as index if it's not already
    if "page" in page_sizes_df.columns:
        page_sizes_df = page_sizes_df.set_index("page")

    # PyMuPDF page numbers are 0-based, DataFrame index assumed 1-based
    page_num_one_based = page.number + 1

    pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = 0, 0, 0, 0  # Initialize defaults

    if image:
        pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = (
            convert_image_coords_to_pymupdf(page, annot, image)
        )

    else:
        # --- Calculate coordinates when no image is present ---
        # Assumes annot coords are normalized relative to MediaBox (top-left origin)
        try:
            # 1. Get MediaBox dimensions from the DataFrame
            page_info = page_sizes_df.loc[page_num_one_based]
            mb_width = page_info["mediabox_width"]
            mb_height = page_info["mediabox_height"]
            x_offset = page_info["cropbox_x_offset"]
            y_offset = page_info["cropbox_y_offset_from_top"]

            # Check for invalid dimensions
            if mb_width <= 0 or mb_height <= 0:
                print(
                    f"Warning: Invalid MediaBox dimensions ({mb_width}x{mb_height}) for page {page_num_one_based}. Setting coords to 0."
                )
            else:
                pymupdf_x1 = annot.left - x_offset
                pymupdf_x2 = annot.left + annot.width - x_offset
                pymupdf_y1 = annot.top - y_offset
                pymupdf_y2 = annot.top + annot.height - y_offset

        except KeyError:
            print(
                f"Warning: Page number {page_num_one_based} not found in page_sizes_df. Cannot get MediaBox dimensions. Setting coords to 0."
            )
        except AttributeError as e:
            print(
                f"Error accessing attributes ('left', 'top', etc.) on 'annot' object for page {page_num_one_based}: {e}"
            )
        except Exception as e:
            print(
                f"Error during coordinate calculation for page {page_num_one_based}: {e}"
            )

    rect = Rect(
        pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2
    )  # Create the PyMuPDF Rect

    # Now creating image annotation object
    image_x1 = annot.left
    image_x2 = annot.left + annot.width
    image_y1 = annot.top
    image_y2 = annot.top + annot.height

    # Create image annotation boxes
    img_annotation_box["xmin"] = image_x1
    img_annotation_box["ymin"] = image_y1
    img_annotation_box["xmax"] = image_x2  # annot.left + annot.width
    img_annotation_box["ymax"] = image_y2  # annot.top + annot.height
    img_annotation_box["color"] = (
        annot.color if custom_colours is True else CUSTOM_BOX_COLOUR
    )
    try:
        img_annotation_box["label"] = str(annot.entity_type)
    except Exception as e:
        print(f"Error getting entity type: {e}")
        img_annotation_box["label"] = "Redaction"

    if hasattr(annot, "text") and annot.text:
        img_annotation_box["text"] = str(annot.text)
    else:
        img_annotation_box["text"] = ""

    # Assign an id
    img_annotation_box = fill_missing_box_ids(img_annotation_box)

    return img_annotation_box, rect


def convert_pikepdf_annotations_to_result_annotation_box(
    page: Page,
    annot: dict,
    image: Image = None,
    convert_pikepdf_to_pymupdf_coords: bool = True,
    page_sizes_df: pd.DataFrame = pd.DataFrame(),
    image_dimensions: dict = dict(),
):
    """
    Convert redaction objects with pikepdf coordinates to annotation boxes for PyMuPDF that can then be redacted from the document. First 1. converts pikepdf to pymupdf coordinates, then 2. converts pymupdf coordinates to image coordinates if page is an image.
    """
    img_annotation_box = dict()
    page_no = page.number

    if convert_pikepdf_to_pymupdf_coords is True:
        pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = (
            convert_pikepdf_coords_to_pymupdf(page, annot)
        )
    else:
        pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = (
            convert_image_coords_to_pymupdf(
                page, annot, image, type="pikepdf_image_coords"
            )
        )

    rect = Rect(pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2)

    convert_df = pd.DataFrame(
        {
            "page": [page_no],
            "xmin": [pymupdf_x1],
            "ymin": [pymupdf_y1],
            "xmax": [pymupdf_x2],
            "ymax": [pymupdf_y2],
        }
    )

    converted_df = convert_df  # divide_coordinates_by_page_sizes(convert_df, page_sizes_df, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")

    img_annotation_box["xmin"] = converted_df["xmin"].max()
    img_annotation_box["ymin"] = converted_df["ymin"].max()
    img_annotation_box["xmax"] = converted_df["xmax"].max()
    img_annotation_box["ymax"] = converted_df["ymax"].max()

    img_annotation_box["color"] = (0, 0, 0)

    if isinstance(annot, Dictionary):
        img_annotation_box["label"] = str(annot["/T"])

        if hasattr(annot, "Contents"):
            img_annotation_box["text"] = str(annot.Contents)
        else:
            img_annotation_box["text"] = ""
    else:
        img_annotation_box["label"] = "REDACTION"
        img_annotation_box["text"] = ""

    return img_annotation_box, rect


def set_cropbox_safely(page: Page, original_cropbox: Optional[Rect]):
    """
    Sets the cropbox of a PyMuPDF page safely and defensively.

    If the 'original_cropbox' is valid (i.e., a pymupdf.Rect instance, not None, not empty,
    not infinite, and fully contained within the page's mediabox), it is set as the cropbox.

    Otherwise, the page's mediabox is used, and a warning is printed to explain why.

    Args:
        page: The PyMuPDF page object.
        original_cropbox: The Rect representing the desired cropbox.
    """
    mediabox = page.mediabox
    reason_for_defaulting = ""

    # Check for None
    if original_cropbox is None:
        reason_for_defaulting = "the original cropbox is None."
    # Check for incorrect type
    elif not isinstance(original_cropbox, Rect):
        reason_for_defaulting = f"the original cropbox is not a pymupdf.Rect instance (got {type(original_cropbox)})."
    else:
        # Normalise the cropbox (ensures x0 < x1 and y0 < y1)
        original_cropbox.normalize()

        # Check for empty or infinite or out-of-bounds
        if original_cropbox.is_empty:
            reason_for_defaulting = (
                f"the provided original cropbox {original_cropbox} is empty."
            )
        elif original_cropbox.is_infinite:
            reason_for_defaulting = (
                f"the provided original cropbox {original_cropbox} is infinite."
            )
        elif not mediabox.contains(original_cropbox):
            reason_for_defaulting = (
                f"the provided original cropbox {original_cropbox} is not fully contained "
                f"within the page's mediabox {mediabox}."
            )

    if reason_for_defaulting:
        print(
            f"Warning (Page {page.number}): Cannot use original cropbox because {reason_for_defaulting} "
            f"Defaulting to the page's mediabox as the cropbox."
        )
        page.set_cropbox(mediabox)
    else:
        page.set_cropbox(original_cropbox)


def convert_color_to_range_0_1(color):
    return tuple(component / 255 for component in color)


def define_box_colour(
    custom_colours: bool, img_annotation_box: dict, CUSTOM_BOX_COLOUR: tuple
):
    """
    Determines the color for a bounding box annotation.

    If `custom_colours` is True, it attempts to parse the color from `img_annotation_box['color']`.
    It supports color strings in "(R,G,B)" format (0-255 integers) or tuples/lists of (R,G,B)
    where components are either 0-1 floats or 0-255 integers.
    If parsing fails or `custom_colours` is False, it defaults to `CUSTOM_BOX_COLOUR`.
    All output colors are converted to a 0.0-1.0 float range.

    Args:
        custom_colours (bool): If True, attempts to use a custom color from `img_annotation_box`.
        img_annotation_box (dict): A dictionary that may contain a 'color' key with the custom color.
        CUSTOM_BOX_COLOUR (tuple): The default color to use if custom colors are not enabled or parsing fails.
                                   Expected to be a tuple of (R, G, B) with values in the 0.0-1.0 range.

    Returns:
        tuple: A tuple (R, G, B) representing the chosen color, with components in the 0.0-1.0 float range.
    """
    if custom_colours is True:
        color_input = img_annotation_box["color"]
        out_colour = (0, 0, 0)  # Initialize with a default black color (0.0-1.0 range)

        if isinstance(color_input, str):
            # Expected format: "(R,G,B)" where R,G,B are integers 0-255 (e.g., "(255,0,0)")
            try:
                # Remove parentheses and split by comma, then convert to integers
                components_str = color_input.strip().strip("()").split(",")
                colour_tuple_int = tuple(int(c.strip()) for c in components_str)

                # Validate the parsed integer tuple
                if len(colour_tuple_int) == 3 and not all(
                    0 <= c <= 1 for c in colour_tuple_int
                ):
                    out_colour = convert_color_to_range_0_1(colour_tuple_int)
                elif len(colour_tuple_int) == 3 and all(
                    0 <= c <= 1 for c in colour_tuple_int
                ):
                    out_colour = colour_tuple_int
                else:
                    print(
                        f"Warning: Invalid color string values or length for '{color_input}'. Expected (R,G,B) with R,G,B in 0-255. Defaulting to black."
                    )
            except (ValueError, IndexError):
                print(
                    f"Warning: Could not parse color string '{color_input}'. Expected '(R,G,B)' format. Defaulting to black."
                )
        elif isinstance(color_input, (tuple, list)) and len(color_input) == 3:
            # Expected formats: (R,G,B) where R,G,B are either 0-1 floats or 0-255 integers
            if all(isinstance(c, (int, float)) for c in color_input):
                # Case 1: Components are already 0.0-1.0 floats
                if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in color_input):
                    out_colour = tuple(color_input)
                # Case 2: Components are 0-255 integers
                elif not all(
                    isinstance(c, float) and 0.0 <= c <= 1.0 for c in color_input
                ):
                    out_colour = convert_color_to_range_0_1(color_input)
                else:
                    # Numeric values but not in expected 0-1 float or 0-255 integer ranges
                    print(
                        f"Warning: Invalid color tuple/list values {color_input}. Expected (R,G,B) with R,G,B in 0-1 floats or 0-255 integers. Defaulting to black."
                    )
            else:
                # Contains non-numeric values (e.g., (1, 'a', 3))
                print(
                    f"Warning: Color tuple/list {color_input} contains non-numeric values. Defaulting to black."
                )
        else:
            # Catch-all for any other unexpected format (e.g., None, dict, etc.)
            print(
                f"Warning: Unexpected color format for {color_input}. Expected string '(R,G,B)' or tuple/list (R,G,B). Defaulting to black."
            )

        # Final safeguard: Ensure out_colour is always a valid PyMuPDF color tuple (3 floats 0.0-1.0)
        if not (
            isinstance(out_colour, tuple)
            and len(out_colour) == 3
            and all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in out_colour)
        ):
            out_colour = (
                0,
                0,
                0,
            )  # Fallback to black if any previous logic resulted in an invalid state
            out_colour = img_annotation_box["color"]
    else:
        if CUSTOM_BOX_COLOUR:
            # Should be a tuple of three integers between 0 and 255 from config
            if (
                isinstance(CUSTOM_BOX_COLOUR, (tuple, list))
                and len(CUSTOM_BOX_COLOUR) >= 3
            ):
                # Convert from 0-255 range to 0-1 range
                out_colour = tuple(
                    float(component / 255) if component >= 1 else float(component)
                    for component in CUSTOM_BOX_COLOUR[:3]
                )
        else:
            out_colour = (
                0,
                0,
                0,
            )  # Fallback to black if no custom box colour is provided

    return out_colour


def redact_single_box(
    pymupdf_page: Page,
    pymupdf_rect: Rect,
    img_annotation_box: dict,
    custom_colours: bool = USE_GUI_BOX_COLOURS_FOR_OUTPUTS,
    retain_text: bool = RETURN_PDF_FOR_REVIEW,
    return_pdf_end_of_redaction: bool = RETURN_REDACTED_PDF,
):
    """
    Commit redaction boxes to a PyMuPDF page.

    Args:
        pymupdf_page (Page): The PyMuPDF page object to which the redaction will be applied.
        pymupdf_rect (Rect): The PyMuPDF rectangle defining the bounds of the redaction box.
        img_annotation_box (dict): A dictionary containing annotation details, such as label, text, and color.
        custom_colours (bool, optional): If True, uses custom colors for the redaction box.
                                        Defaults to USE_GUI_BOX_COLOURS_FOR_OUTPUTS.
        retain_text (bool, optional): If True, adds a redaction annotation but retains the underlying text.
                                      If False, the text within the redaction area is deleted.
                                      Defaults to RETURN_PDF_FOR_REVIEW.
        return_pdf_end_of_redaction (bool, optional): If True, returns both review and final redacted page objects.
                                                      Defaults to RETURN_REDACTED_PDF.

    Returns:
        Page or Tuple[Page, Page]: If return_pdf_end_of_redaction is True and retain_text is True,
                                  returns a tuple of (review_page, applied_redaction_page). Otherwise returns a single Page.
    """

    pymupdf_x1 = pymupdf_rect[0]
    pymupdf_y1 = pymupdf_rect[1]
    pymupdf_x2 = pymupdf_rect[2]
    pymupdf_y2 = pymupdf_rect[3]

    # Full size redaction box for covering all the text of a word
    full_size_redaction_box = Rect(
        pymupdf_x1 - 1, pymupdf_y1 - 1, pymupdf_x2 + 1, pymupdf_y2 + 1
    )

    # Calculate tiny height redaction box so that it doesn't delete text from adjacent lines
    redact_bottom_y = pymupdf_y1 + 2
    redact_top_y = pymupdf_y2 - 2

    # Calculate the middle y value and set a small height if default values are too close together
    if (redact_top_y - redact_bottom_y) < 1:
        middle_y = (pymupdf_y1 + pymupdf_y2) / 2
        redact_bottom_y = middle_y - 1
        redact_top_y = middle_y + 1

    rect_small_pixel_height = Rect(
        pymupdf_x1 + 2, redact_bottom_y, pymupdf_x2 - 2, redact_top_y
    )  # Slightly smaller than outside box

    out_colour = define_box_colour(
        custom_colours, img_annotation_box, CUSTOM_BOX_COLOUR
    )

    img_annotation_box["text"] = img_annotation_box.get("text") or ""
    img_annotation_box["label"] = img_annotation_box.get("label") or "Redaction"

    # Create a copy of the page for final redaction if needed
    applied_redaction_page = None
    if return_pdf_end_of_redaction and retain_text:
        # Create a deep copy of the page for final redaction

        applied_redaction_page = pymupdf.open()
        applied_redaction_page.insert_pdf(
            pymupdf_page.parent,
            from_page=pymupdf_page.number,
            to_page=pymupdf_page.number,
        )
        applied_redaction_page = applied_redaction_page[0]

    # Handle review page first, then deal with final redacted page (retain_text = True)
    if retain_text is True:

        annot = pymupdf_page.add_redact_annot(full_size_redaction_box)
        annot.set_colors(stroke=out_colour, fill=out_colour, colors=out_colour)
        annot.set_name(img_annotation_box["label"])
        annot.set_info(
            info=img_annotation_box["label"],
            title=img_annotation_box["label"],
            subject=img_annotation_box["label"],
            content=img_annotation_box["text"],
            creationDate=datetime.now().strftime("%Y%m%d%H%M%S"),
        )
        annot.update(opacity=0.5, cross_out=False)

        # If we need both review and final pages, and the applied redaction page has been prepared, apply final redaction to the copy
        if return_pdf_end_of_redaction and applied_redaction_page is not None:
            # Apply final redaction to the copy

            # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
            applied_redaction_page.add_redact_annot(rect_small_pixel_height)

            # Only create a box over the whole rect if we want to delete the text
            shape = applied_redaction_page.new_shape()
            shape.draw_rect(pymupdf_rect)

            # Use solid fill for normal redaction
            shape.finish(color=out_colour, fill=out_colour)
            shape.commit()

            return pymupdf_page, applied_redaction_page
        else:
            return pymupdf_page

    # If we don't need to retain the text, we only have one page which is the applied redaction page, so just apply the redaction to the page
    else:
        # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
        pymupdf_page.add_redact_annot(rect_small_pixel_height)

        # Only create a box over the whole rect if we want to delete the text
        shape = pymupdf_page.new_shape()
        shape.draw_rect(pymupdf_rect)

        # Use solid fill for normal redaction
        shape.finish(color=out_colour, fill=out_colour)
        shape.commit()

        return pymupdf_page


def redact_whole_pymupdf_page(
    rect_height: float,
    rect_width: float,
    page: Page,
    custom_colours: bool = False,
    border: float = 5,
    redact_pdf: bool = True,
):
    """
    Redacts a whole page of a PDF document.

    Args:
        rect_height (float): The height of the page in points.
        rect_width (float): The width of the page in points.
        page (Page): The PyMuPDF page object to be redacted.
        custom_colours (bool, optional): If True, uses custom colors for the redaction box.
        border (float, optional): The border width in points. Defaults to 5.
        redact_pdf (bool, optional): If True, redacts the PDF document. Defaults to True.
    """
    # Small border to page that remains white

    # Define the coordinates for the Rect (PDF coordinates for actual redaction)
    whole_page_x1, whole_page_y1 = 0 + border, 0 + border  # Bottom-left corner
    whole_page_x2, whole_page_y2 = (
        rect_width - border,
        rect_height - border,
    )  # Top-right corner

    # Create new image annotation element based on whole page coordinates
    whole_page_rect = Rect(whole_page_x1, whole_page_y1, whole_page_x2, whole_page_y2)

    # Calculate relative coordinates for the annotation box (0-1 range)
    # This ensures the coordinates are already in relative format for output files
    relative_border = border / min(
        rect_width, rect_height
    )  # Scale border proportionally
    relative_x1 = relative_border
    relative_y1 = relative_border
    relative_x2 = 1 - relative_border
    relative_y2 = 1 - relative_border

    # Write whole page annotation to annotation boxes using relative coordinates
    whole_page_img_annotation_box = dict()
    whole_page_img_annotation_box["xmin"] = relative_x1
    whole_page_img_annotation_box["ymin"] = relative_y1
    whole_page_img_annotation_box["xmax"] = relative_x2
    whole_page_img_annotation_box["ymax"] = relative_y2
    whole_page_img_annotation_box["color"] = (0, 0, 0)
    whole_page_img_annotation_box["label"] = "Whole page"

    if redact_pdf is True:
        redact_single_box(
            page, whole_page_rect, whole_page_img_annotation_box, custom_colours
        )

    return whole_page_img_annotation_box


def redact_page_with_pymupdf(
    page: Page,
    page_annotations: dict,
    image: Image = None,
    custom_colours: bool = USE_GUI_BOX_COLOURS_FOR_OUTPUTS,
    redact_whole_page: bool = False,
    convert_pikepdf_to_pymupdf_coords: bool = True,
    original_cropbox: List[Rect] = list(),
    page_sizes_df: pd.DataFrame = pd.DataFrame(),
    return_pdf_for_review: bool = RETURN_PDF_FOR_REVIEW,
    return_pdf_end_of_redaction: bool = RETURN_REDACTED_PDF,
    input_folder: str = INPUT_FOLDER,
):
    """
    Applies redactions to a single PyMuPDF page based on provided annotations.

    This function processes various types of annotations (Gradio, CustomImageRecognizerResult,
    or pikepdf-like) and applies them as redactions to the given PyMuPDF page. It can also
    redact the entire page if specified.

    Args:
        page (Page): The PyMuPDF page object to which redactions will be applied.
        page_annotations (dict): A dictionary containing annotation data for the current page.
                                 Expected to have a 'boxes' key with a list of annotation boxes.
        image (Image, optional): A PIL Image object or path to an image file associated with the page.
                                 Used for coordinate conversions if available. Defaults to None.
        custom_colours (bool, optional): If True, custom box colors will be used for redactions.
                                         Defaults to USE_GUI_BOX_COLOURS_FOR_OUTPUTS.
        redact_whole_page (bool, optional): If True, the entire page will be redacted. Defaults to False.
        convert_pikepdf_to_pymupdf_coords (bool, optional): If True, coordinates from pikepdf-like
                                                            annotations will be converted to PyMuPDF's
                                                            coordinate system. Defaults to True.
        original_cropbox (List[Rect], optional): The original cropbox of the page. This is used
                                                 to restore the cropbox after redactions. Defaults to an empty list.
        page_sizes_df (pd.DataFrame, optional): A DataFrame containing page size and image dimension
                                                information, used for coordinate scaling. Defaults to an empty DataFrame.
        return_pdf_for_review (bool, optional): If True, redactions are applied in a way suitable for
                                                review (e.g., not removing underlying text/images completely).
                                                Defaults to RETURN_PDF_FOR_REVIEW.
        return_pdf_end_of_redaction (bool, optional): If True, returns both review and final redacted page objects.
                                                      Defaults to RETURN_REDACTED_PDF.

    Returns:
        Tuple[Page, dict] or Tuple[Tuple[Page, Page], dict]: A tuple containing:
            - page (Page or Tuple[Page, Page]): The PyMuPDF page object(s) with redactions applied.
                                               If return_pdf_end_of_redaction is True and return_pdf_for_review is True,
                                               returns a tuple of (review_page, applied_redaction_page).
            - out_annotation_boxes (dict): A dictionary containing the processed annotation boxes
                                           for the page, including the image path.
    """

    rect_height = page.rect.height
    rect_width = page.rect.width

    mediabox_height = page.mediabox.height
    mediabox_width = page.mediabox.width

    page_no = page.number
    page_num_reported = page_no + 1

    page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Check if image dimensions for page exist in page_sizes_df
    image_dimensions = dict()

    if not image and "image_width" in page_sizes_df.columns:
        page_sizes_df[["image_width"]] = page_sizes_df[["image_width"]].apply(
            pd.to_numeric, errors="coerce"
        )
        page_sizes_df[["image_height"]] = page_sizes_df[["image_height"]].apply(
            pd.to_numeric, errors="coerce"
        )

        image_dimensions["image_width"] = page_sizes_df.loc[
            page_sizes_df["page"] == page_num_reported, "image_width"
        ].max()
        image_dimensions["image_height"] = page_sizes_df.loc[
            page_sizes_df["page"] == page_num_reported, "image_height"
        ].max()

        if pd.isna(image_dimensions["image_width"]):
            image_dimensions = dict()

    out_annotation_boxes = dict()
    all_image_annotation_boxes = list()

    if isinstance(image, Image.Image):
        # Create an image path using the input folder with PDF filename
        # Get the PDF filename from the page's parent document
        pdf_filename = (
            os.path.basename(page.parent.name)
            if hasattr(page.parent, "name") and page.parent.name
            else "document"
        )
        # pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
        image_path = os.path.join(input_folder, f"{pdf_filename}_{page.number}.png")
        if not os.path.exists(image_path):
            image.save(image_path)
    elif isinstance(image, str):
        # Normalize and validate path safety before checking existence
        normalized_path = os.path.normpath(os.path.abspath(image))
        if validate_path_containment(normalized_path, INPUT_FOLDER):
            image_path = normalized_path
            image = Image.open(image_path)
        elif "image_path" in page_sizes_df.columns:
            try:
                image_path = page_sizes_df.loc[
                    page_sizes_df["page"] == (page_no + 1), "image_path"
                ].iloc[0]
            except IndexError:
                image_path = ""
            image = None
        else:
            image_path = ""
            image = None
    else:
        # print("image is not an Image object or string")
        image_path = ""
        image = None

    # Check if this is an object used in the Gradio Annotation component
    if isinstance(page_annotations, dict):
        page_annotations = page_annotations["boxes"]

    for annot in page_annotations:
        # Check if an Image recogniser result, or a Gradio annotation object
        if (isinstance(annot, CustomImageRecognizerResult)) | isinstance(annot, dict):

            img_annotation_box = dict()

            # Should already be in correct format if img_annotator_box is an input
            if isinstance(annot, dict):
                annot = fill_missing_box_ids(annot)
                img_annotation_box = annot

                box_coordinates = (
                    img_annotation_box["xmin"],
                    img_annotation_box["ymin"],
                    img_annotation_box["xmax"],
                    img_annotation_box["ymax"],
                )

                # Check if all coordinates are equal to or less than 1
                are_coordinates_relative = all(coord <= 1 for coord in box_coordinates)

                if are_coordinates_relative is True:
                    # Check if coordinates are relative, if so then multiply by mediabox size
                    pymupdf_x1 = img_annotation_box["xmin"] * mediabox_width
                    pymupdf_y1 = img_annotation_box["ymin"] * mediabox_height
                    pymupdf_x2 = img_annotation_box["xmax"] * mediabox_width
                    pymupdf_y2 = img_annotation_box["ymax"] * mediabox_height

                elif image_dimensions or image:
                    pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = (
                        convert_gradio_image_annotator_object_coords_to_pymupdf(
                            page, img_annotation_box, image, image_dimensions
                        )
                    )
                else:
                    print(
                        "Could not convert image annotator coordinates in redact_page_with_pymupdf"
                    )
                    print("img_annotation_box", img_annotation_box)
                    pymupdf_x1 = img_annotation_box["xmin"]
                    pymupdf_y1 = img_annotation_box["ymin"]
                    pymupdf_x2 = img_annotation_box["xmax"]
                    pymupdf_y2 = img_annotation_box["ymax"]

                if "text" in annot and annot["text"]:
                    img_annotation_box["text"] = str(annot["text"])
                else:
                    img_annotation_box["text"] = ""

                rect = Rect(
                    pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2
                )  # Create the PyMuPDF Rect

            # Else should be CustomImageRecognizerResult
            elif isinstance(annot, CustomImageRecognizerResult):
                # print("annot is a CustomImageRecognizerResult")
                img_annotation_box, rect = (
                    prepare_custom_image_recogniser_result_annotation_box(
                        page, annot, image, page_sizes_df, custom_colours
                    )
                )

        # Else it should be a pikepdf annotation object
        else:
            if not image:
                convert_pikepdf_to_pymupdf_coords = True
            else:
                convert_pikepdf_to_pymupdf_coords = False

            img_annotation_box, rect = (
                convert_pikepdf_annotations_to_result_annotation_box(
                    page,
                    annot,
                    image,
                    convert_pikepdf_to_pymupdf_coords,
                    page_sizes_df,
                    image_dimensions=image_dimensions,
                )
            )

            img_annotation_box = fill_missing_box_ids(img_annotation_box)

        all_image_annotation_boxes.append(img_annotation_box)

        # Redact the annotations from the document
        redact_result = redact_single_box(
            page,
            rect,
            img_annotation_box,
            custom_colours,
            return_pdf_for_review,
            return_pdf_end_of_redaction,
        )

        # Handle dual page objects if returned
        if isinstance(redact_result, tuple):
            page, applied_redaction_page = redact_result
            # Store the final page for later use
            if not hasattr(redact_page_with_pymupdf, "_applied_redaction_page"):
                redact_page_with_pymupdf._applied_redaction_page = (
                    applied_redaction_page
                )
            else:
                # If we already have a final page, we need to handle multiple pages
                # For now, we'll use the last final page
                redact_page_with_pymupdf._applied_redaction_page = (
                    applied_redaction_page
                )

    # If whole page is to be redacted, do that here
    if redact_whole_page is True:

        whole_page_img_annotation_box = redact_whole_pymupdf_page(
            rect_height, rect_width, page, custom_colours, border=5
        )
        # Ensure the whole page annotation box has a unique ID
        whole_page_img_annotation_box = fill_missing_box_ids(
            whole_page_img_annotation_box
        )
        all_image_annotation_boxes.append(whole_page_img_annotation_box)

        # Handle dual page objects for whole page redaction if needed
        if return_pdf_end_of_redaction and return_pdf_for_review:
            # Create a copy of the page for final redaction using the same approach as redact_single_box

            applied_redaction_doc = pymupdf.open()
            applied_redaction_doc.insert_pdf(
                page.parent,
                from_page=page.number,
                to_page=page.number,
            )
            applied_redaction_page = applied_redaction_doc[0]

            # Apply the whole page redaction to the final page as well
            redact_whole_pymupdf_page(
                rect_height,
                rect_width,
                applied_redaction_page,
                custom_colours,
                border=5,
            )

            # Store the final page with its original page number for later use
            if not hasattr(redact_page_with_pymupdf, "_applied_redaction_page"):
                redact_page_with_pymupdf._applied_redaction_page = (
                    applied_redaction_page,
                    page.number,
                )
            else:
                # If we already have a final page, we need to handle multiple pages
                # For now, we'll use the last final page
                redact_page_with_pymupdf._applied_redaction_page = (
                    applied_redaction_page,
                    page.number,
                )

    out_annotation_boxes = {
        "image": image_path,  # Image.open(image_path), #image_path,
        "boxes": all_image_annotation_boxes,
    }

    # If we are not returning the review page, can directly remove text and all images
    if return_pdf_for_review is False:
        page.apply_redactions(
            images=APPLY_REDACTIONS_IMAGES,
            graphics=APPLY_REDACTIONS_GRAPHICS,
            text=APPLY_REDACTIONS_TEXT,
        )

    set_cropbox_safely(page, original_cropbox)
    page.clean_contents()

    # Handle dual page objects if we have a final page
    if (
        return_pdf_end_of_redaction
        and return_pdf_for_review
        and hasattr(redact_page_with_pymupdf, "_applied_redaction_page")
    ):
        applied_redaction_page_data = redact_page_with_pymupdf._applied_redaction_page
        # Handle both tuple format (new) and single page format (backward compatibility)
        if isinstance(applied_redaction_page_data, tuple):
            applied_redaction_page, original_page_number = applied_redaction_page_data
        else:
            applied_redaction_page = applied_redaction_page_data

        # Apply redactions to applied redaction page only
        applied_redaction_page.apply_redactions(
            images=APPLY_REDACTIONS_IMAGES,
            graphics=APPLY_REDACTIONS_GRAPHICS,
            text=APPLY_REDACTIONS_TEXT,
        )

        set_cropbox_safely(applied_redaction_page, original_cropbox)
        applied_redaction_page.clean_contents()
        # Clear the stored final page
        delattr(redact_page_with_pymupdf, "_applied_redaction_page")
        return (page, applied_redaction_page), out_annotation_boxes

    else:
        return page, out_annotation_boxes


###
# IMAGE-BASED OCR PDF TEXT DETECTION/REDACTION WITH TESSERACT OR AWS TEXTRACT
###


def merge_img_bboxes(
    bboxes: list,
    combined_results: Dict,
    page_signature_recogniser_results: list = list(),
    page_handwriting_recogniser_results: list = list(),
    handwrite_signature_checkbox: List[str] = [
        "Extract handwriting",
        "Extract signatures",
    ],
    horizontal_threshold: int = 50,
    vertical_threshold: int = 12,
):
    """
    Merges bounding boxes for image annotations based on the provided results from signature and handwriting recognizers.

    Args:
        bboxes (list): A list of bounding boxes to be merged.
        combined_results (Dict): A dictionary containing combined results with line text and their corresponding bounding boxes.
        page_signature_recogniser_results (list, optional): A list of results from the signature recognizer. Defaults to an empty list.
        page_handwriting_recogniser_results (list, optional): A list of results from the handwriting recognizer. Defaults to an empty list.
        handwrite_signature_checkbox (List[str], optional): A list of options indicating whether to extract handwriting and signatures. Defaults to ["Extract handwriting", "Extract signatures"].
        horizontal_threshold (int, optional): The threshold for merging bounding boxes horizontally. Defaults to 50.
        vertical_threshold (int, optional): The threshold for merging bounding boxes vertically. Defaults to 12.

    Returns:
        None: This function modifies the bounding boxes in place and does not return a value.
    """

    all_bboxes = list()
    merged_bboxes = list()
    grouped_bboxes = defaultdict(list)

    # Deep copy original bounding boxes to retain them
    original_bboxes = copy.deepcopy(bboxes)

    # Process signature and handwriting results
    if page_signature_recogniser_results or page_handwriting_recogniser_results:

        if "Extract handwriting" in handwrite_signature_checkbox:
            print("Extracting handwriting in merge_img_bboxes function")
            merged_bboxes.extend(copy.deepcopy(page_handwriting_recogniser_results))

        if "Extract signatures" in handwrite_signature_checkbox:
            print("Extracting signatures in merge_img_bboxes function")
            merged_bboxes.extend(copy.deepcopy(page_signature_recogniser_results))

    # Reconstruct bounding boxes for substrings of interest
    reconstructed_bboxes = list()
    for bbox in bboxes:
        bbox_box = (bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height)
        for line_text, line_info in combined_results.items():
            line_box = line_info["bounding_box"]
            if bounding_boxes_overlap(bbox_box, line_box):
                if bbox.text in line_text:
                    start_char = line_text.index(bbox.text)
                    end_char = start_char + len(bbox.text)

                    relevant_words = list()
                    current_char = 0
                    for word in line_info["words"]:
                        word_end = current_char + len(word["text"])
                        if (
                            current_char <= start_char < word_end
                            or current_char < end_char <= word_end
                            or (start_char <= current_char and word_end <= end_char)
                        ):
                            relevant_words.append(word)
                        if word_end >= end_char:
                            break
                        current_char = word_end
                        if not word["text"].endswith(" "):
                            current_char += 1  # +1 for space if the word doesn't already end with a space

                    if relevant_words:
                        left = min(word["bounding_box"][0] for word in relevant_words)
                        top = min(word["bounding_box"][1] for word in relevant_words)
                        right = max(word["bounding_box"][2] for word in relevant_words)
                        bottom = max(word["bounding_box"][3] for word in relevant_words)

                        combined_text = " ".join(
                            word["text"] for word in relevant_words
                        )

                        reconstructed_bbox = CustomImageRecognizerResult(
                            bbox.entity_type,
                            bbox.start,
                            bbox.end,
                            bbox.score,
                            left,
                            top,
                            right - left,  # width
                            bottom - top,  # height,
                            combined_text,
                        )
                        # reconstructed_bboxes.append(bbox)  # Add original bbox
                        reconstructed_bboxes.append(
                            reconstructed_bbox
                        )  # Add merged bbox
                        break
        else:
            reconstructed_bboxes.append(bbox)

    # Group reconstructed bboxes by approximate vertical proximity
    for box in reconstructed_bboxes:
        grouped_bboxes[round(box.top / vertical_threshold)].append(box)

    # Merge within each group
    for _, group in grouped_bboxes.items():
        group.sort(key=lambda box: box.left)

        merged_box = group[0]
        for next_box in group[1:]:
            if (
                next_box.left - (merged_box.left + merged_box.width)
                <= horizontal_threshold
            ):
                if next_box.text != merged_box.text:
                    new_text = merged_box.text + " " + next_box.text
                else:
                    new_text = merged_box.text

                if merged_box.entity_type != next_box.entity_type:
                    new_entity_type = (
                        merged_box.entity_type + " - " + next_box.entity_type
                    )
                else:
                    new_entity_type = merged_box.entity_type

                new_left = min(merged_box.left, next_box.left)
                new_top = min(merged_box.top, next_box.top)
                new_width = (
                    max(
                        merged_box.left + merged_box.width,
                        next_box.left + next_box.width,
                    )
                    - new_left
                )
                new_height = (
                    max(
                        merged_box.top + merged_box.height,
                        next_box.top + next_box.height,
                    )
                    - new_top
                )

                merged_box = CustomImageRecognizerResult(
                    new_entity_type,
                    merged_box.start,
                    merged_box.end,
                    merged_box.score,
                    new_left,
                    new_top,
                    new_width,
                    new_height,
                    new_text,
                )
            else:
                merged_bboxes.append(merged_box)
                merged_box = next_box

        merged_bboxes.append(merged_box)

    all_bboxes.extend(original_bboxes)
    all_bboxes.extend(merged_bboxes)

    # Return the unique original and merged bounding boxes
    unique_bboxes = list(
        {
            (bbox.left, bbox.top, bbox.width, bbox.height): bbox for bbox in all_bboxes
        }.values()
    )
    return unique_bboxes


def redact_image_pdf(
    file_path: str,
    pdf_image_file_paths: List[str],
    language: str,
    chosen_redact_entities: List[str],
    chosen_redact_comprehend_entities: List[str],
    allow_list: List[str] = None,
    page_min: int = 0,
    page_max: int = 999,
    text_extraction_method: str = TESSERACT_TEXT_EXTRACT_OPTION,
    handwrite_signature_checkbox: List[str] = [
        "Extract handwriting",
        "Extract signatures",
    ],
    textract_request_metadata: list = list(),
    current_loop_page: int = 0,
    page_break_return: bool = False,
    annotations_all_pages: List = list(),
    all_page_line_level_ocr_results_df: pd.DataFrame = pd.DataFrame(
        columns=["page", "text", "left", "top", "width", "height", "line", "conf"]
    ),
    all_pages_decision_process_table: pd.DataFrame = pd.DataFrame(
        columns=[
            "image_path",
            "page",
            "label",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "boundingBox",
            "text",
            "start",
            "end",
            "score",
            "id",
        ]
    ),
    pymupdf_doc: Document = list(),
    pii_identification_method: str = "Local",
    comprehend_query_number: int = 0,
    comprehend_client: str = "",
    textract_client: str = "",
    in_deny_list: List[str] = list(),
    redact_whole_page_list: List[str] = list(),
    max_fuzzy_spelling_mistakes_num: int = 1,
    match_fuzzy_whole_phrase_bool: bool = True,
    page_sizes_df: pd.DataFrame = pd.DataFrame(),
    text_extraction_only: bool = False,
    all_page_line_level_ocr_results=list(),
    all_page_line_level_ocr_results_with_words=list(),
    chosen_local_model: str = "tesseract",
    page_break_val: int = int(PAGE_BREAK_VALUE),
    log_files_output_paths: List = list(),
    max_time: int = int(MAX_TIME_VALUE),
    nlp_analyser: AnalyzerEngine = nlp_analyser,
    output_folder: str = OUTPUT_FOLDER,
    input_folder: str = INPUT_FOLDER,
    progress=Progress(track_tqdm=True),
):
    """
    This function redacts sensitive information from a PDF document. It takes the following parameters in order:

    - file_path (str): The path to the PDF file to be redacted.
    - pdf_image_file_paths (List[str]): A list of paths to the PDF file pages converted to images.
    - language (str): The language of the text in the PDF.
    - chosen_redact_entities (List[str]): A list of entity types to redact from the PDF.
    - chosen_redact_comprehend_entities (List[str]): A list of entity types to redact from the list allowed by the AWS Comprehend service.
    - allow_list (List[str], optional): A list of entity types to allow in the PDF. Defaults to None.
    - page_min (int, optional): The minimum page number to start redaction from. Defaults to 0.
    - page_max (int, optional): The maximum page number to end redaction at. Defaults to 999.
    - text_extraction_method (str, optional): The type of analysis to perform on the PDF. Defaults to TESSERACT_TEXT_EXTRACT_OPTION.
    - handwrite_signature_checkbox (List[str], optional): A list of options for redacting handwriting and signatures. Defaults to ["Extract handwriting", "Extract signatures"].
    - textract_request_metadata (list, optional): Metadata related to the redaction request. Defaults to an empty string.
    - current_loop_page (int, optional): The current page being processed. Defaults to 0.
    - page_break_return (bool, optional): Indicates if the function should return after a page break. Defaults to False.
    - annotations_all_pages (List, optional): List of annotations on all pages that is used by the gradio_image_annotation object.
    - all_page_line_level_ocr_results_df (pd.DataFrame, optional): All line level OCR results for the document as a Pandas dataframe,
    - all_pages_decision_process_table (pd.DataFrame, optional): All redaction decisions for document as a Pandas dataframe.
    - pymupdf_doc (Document, optional): The document as a PyMupdf object.
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - comprehend_client (optional): A connection to the AWS Comprehend service via the boto3 package.
    - textract_client (optional): A connection to the AWS Textract service via the boto3 package.
    - in_deny_list (optional): A list of custom words that the user has chosen specifically to redact.
    - redact_whole_page_list (optional, List[str]): A list of pages to fully redact.
    - max_fuzzy_spelling_mistakes_num (int, optional): The maximum number of spelling mistakes allowed in a searched phrase for fuzzy matching. Can range from 0-9.
    - match_fuzzy_whole_phrase_bool (bool, optional): A boolean where 'True' means that the whole phrase is fuzzy matched, and 'False' means that each word is fuzzy matched separately (excluding stop words).
    - page_sizes_df (pd.DataFrame, optional): A pandas dataframe of PDF page sizes in PDF or image format.
    - text_extraction_only (bool, optional): Should the function only extract text, or also do redaction.
    - all_page_line_level_ocr_results (optional): List of all page line level OCR results.
    - all_page_line_level_ocr_results_with_words (optional): List of all page line level OCR results with words.
    - chosen_local_model (str, optional): The local model chosen for OCR. Defaults to "tesseract", other choices are "paddle" for PaddleOCR, or "hybrid" for a combination of both.
    - page_break_val (int, optional): The value at which to trigger a page break. Defaults to PAGE_BREAK_VALUE.
    - log_files_output_paths (List, optional): List of file paths used for saving redaction process logging results.
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.
    - nlp_analyser (AnalyzerEngine, optional): The nlp_analyser object to use for entity detection. Defaults to nlp_analyser.
    - output_folder (str, optional): The folder for file outputs.
    - progress (Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.
    - input_folder (str, optional): The folder for file inputs.
    The function returns a redacted PDF document along with processing output objects.
    """

    tic = time.perf_counter()

    file_name = get_file_name_without_type(file_path)
    comprehend_query_number_new = 0
    selection_element_results_list_df = pd.DataFrame()
    form_key_value_results_list_df = pd.DataFrame()

    # Try updating the supported languages for the spacy analyser
    try:
        nlp_analyser = create_nlp_analyser(language, existing_nlp_analyser=nlp_analyser)
        # Check list of nlp_analyser recognisers and languages
        if language != "en":
            gr.Info(
                f"Language: {language} only supports the following entity detection: {str(nlp_analyser.registry.get_supported_entities(languages=[language]))}"
            )

    except Exception as e:
        print(f"Error creating nlp_analyser for {language}: {e}")
        raise Exception(f"Error creating nlp_analyser for {language}: {e}")

    # Update custom word list analyser object with any new words that have been added to the custom deny list
    if in_deny_list:
        nlp_analyser.registry.remove_recognizer("CUSTOM")
        new_custom_recogniser = custom_word_list_recogniser(in_deny_list)
        nlp_analyser.registry.add_recognizer(new_custom_recogniser)

        nlp_analyser.registry.remove_recognizer("CustomWordFuzzyRecognizer")
        new_custom_fuzzy_recogniser = CustomWordFuzzyRecognizer(
            supported_entities=["CUSTOM_FUZZY"],
            custom_list=in_deny_list,
            spelling_mistakes_max=max_fuzzy_spelling_mistakes_num,
            search_whole_phrase=match_fuzzy_whole_phrase_bool,
        )
        nlp_analyser.registry.add_recognizer(new_custom_fuzzy_recogniser)

    # Only load in PaddleOCR models if not running Textract
    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        image_analyser = CustomImageAnalyzerEngine(
            analyzer_engine=nlp_analyser, ocr_engine="tesseract", language=language
        )
    else:
        image_analyser = CustomImageAnalyzerEngine(
            analyzer_engine=nlp_analyser,
            ocr_engine=chosen_local_model,
            language=language,
        )

    if pii_identification_method == "AWS Comprehend" and comprehend_client == "":
        out_message = "Connection to AWS Comprehend service unsuccessful."
        print(out_message)
        raise Exception(out_message)

    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION and textract_client == "":
        out_message_warning = "Connection to AWS Textract service unsuccessful. Redaction will only continue if local AWS Textract results can be found."
        print(out_message_warning)
        # raise Exception(out_message)

    number_of_pages = pymupdf_doc.page_count
    print("Number of pages:", str(number_of_pages))

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages

    if page_min <= 0:
        page_min = 0
    else:
        page_min = page_min - 1

    print("Page range:", str(page_min + 1), "to", str(page_max))

    # If running Textract, check if file already exists. If it does, load in existing data
    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        textract_json_file_path = output_folder + file_name + "_textract.json"
        textract_data, is_missing, log_files_output_paths = (
            load_and_convert_textract_json(
                textract_json_file_path, log_files_output_paths, page_sizes_df
            )
        )
        original_textract_data = textract_data.copy()

        # print("Successfully loaded in Textract analysis results from file")

    # If running local OCR option, check if file already exists. If it does, load in existing data
    if text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:
        all_page_line_level_ocr_results_with_words_json_file_path = (
            output_folder + file_name + "_ocr_results_with_words_local_ocr.json"
        )
        (
            all_page_line_level_ocr_results_with_words,
            is_missing,
            log_files_output_paths,
        ) = load_and_convert_ocr_results_with_words_json(
            all_page_line_level_ocr_results_with_words_json_file_path,
            log_files_output_paths,
            page_sizes_df,
        )
        original_all_page_line_level_ocr_results_with_words = (
            all_page_line_level_ocr_results_with_words.copy()
        )

    ###
    if current_loop_page == 0:
        page_loop_start = 0
    else:
        page_loop_start = current_loop_page

    progress_bar = tqdm(
        range(page_loop_start, number_of_pages),
        unit="pages remaining",
        desc="Redacting pages",
    )

    # If there's data from a previous run (passed in via the DataFrame parameters), add it
    all_line_level_ocr_results_list = list()
    all_pages_decision_process_list = list()
    selection_element_results_list = list()
    form_key_value_results_list = list()

    if not all_page_line_level_ocr_results_df.empty:
        all_line_level_ocr_results_list.extend(
            all_page_line_level_ocr_results_df.to_dict("records")
        )
    if not all_pages_decision_process_table.empty:
        all_pages_decision_process_list.extend(
            all_pages_decision_process_table.to_dict("records")
        )

    # Go through each page
    for page_no in progress_bar:

        handwriting_or_signature_boxes = list()
        page_signature_recogniser_results = list()
        page_handwriting_recogniser_results = list()
        page_line_level_ocr_results_with_words = list()
        page_break_return = False
        reported_page_number = str(page_no + 1)

        # Try to find image location
        try:
            image_path = page_sizes_df.loc[
                page_sizes_df["page"] == (page_no + 1), "image_path"
            ].iloc[0]
        except Exception as e:
            print("Could not find image_path in page_sizes_df due to:", e)
            image_path = pdf_image_file_paths[page_no]

        page_image_annotations = {"image": image_path, "boxes": []}
        pymupdf_page = pymupdf_doc.load_page(page_no)

        if page_no >= page_min and page_no < page_max:
            # Need image size to convert OCR outputs to the correct sizes
            if isinstance(image_path, str):
                # Normalize and validate path safety before checking existence
                normalized_path = os.path.normpath(os.path.abspath(image_path))
                if validate_path_containment(normalized_path, INPUT_FOLDER):
                    image = Image.open(normalized_path)
                    page_width, page_height = image.size
                else:
                    # print("Image path does not exist, using mediabox coordinates as page sizes")
                    image = None
                    page_width = pymupdf_page.mediabox.width
                    page_height = pymupdf_page.mediabox.height
            elif not isinstance(image_path, Image.Image):
                print(
                    f"Unexpected image_path type: {type(image_path)}, using page mediabox coordinates as page sizes"
                )  # Ensure image_path is valid
                image = None
                page_width = pymupdf_page.mediabox.width
                page_height = pymupdf_page.mediabox.height

            try:
                if not page_sizes_df.empty:
                    original_cropbox = page_sizes_df.loc[
                        page_sizes_df["page"] == (page_no + 1), "original_cropbox"
                    ].iloc[0]
            except IndexError:
                print(
                    "Can't find original cropbox details for page, using current PyMuPDF page cropbox"
                )
                original_cropbox = pymupdf_page.cropbox.irect

            # Step 1: Perform OCR. Either with Tesseract, or with AWS Textract
            # If using Tesseract
            if text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:

                if all_page_line_level_ocr_results_with_words:
                    # Find the first dict where 'page' matches

                    matching_page = next(
                        (
                            item
                            for item in all_page_line_level_ocr_results_with_words
                            if int(item.get("page", -1)) == int(reported_page_number)
                        ),
                        None,
                    )

                    page_line_level_ocr_results_with_words = (
                        matching_page if matching_page else []
                    )
                else:
                    page_line_level_ocr_results_with_words = list()

                if page_line_level_ocr_results_with_words:
                    print(
                        "Found OCR results for page in existing OCR with words object"
                    )

                    page_line_level_ocr_results = (
                        recreate_page_line_level_ocr_results_with_page(
                            page_line_level_ocr_results_with_words
                        )
                    )

                else:
                    page_word_level_ocr_results = image_analyser.perform_ocr(image_path)

                    (
                        page_line_level_ocr_results,
                        page_line_level_ocr_results_with_words,
                    ) = combine_ocr_results(
                        page_word_level_ocr_results, page=reported_page_number
                    )

                    if all_page_line_level_ocr_results_with_words is None:
                        all_page_line_level_ocr_results_with_words = list()

                    all_page_line_level_ocr_results_with_words.append(
                        page_line_level_ocr_results_with_words
                    )

            # Check if page exists in existing textract data. If not, send to service to analyse
            if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
                text_blocks = list()

                if not textract_data:
                    try:
                        # Convert the image_path to bytes using an in-memory buffer
                        image_buffer = io.BytesIO()
                        image.save(
                            image_buffer, format="PNG"
                        )  # Save as PNG, or adjust format if needed
                        pdf_page_as_bytes = image_buffer.getvalue()

                        text_blocks, new_textract_request_metadata = (
                            analyse_page_with_textract(
                                pdf_page_as_bytes,
                                reported_page_number,
                                textract_client,
                                handwrite_signature_checkbox,
                            )
                        )  # Analyse page with Textract

                        if textract_json_file_path not in log_files_output_paths:
                            log_files_output_paths.append(textract_json_file_path)

                        textract_data = {"pages": [text_blocks]}
                    except Exception as e:
                        print(
                            "Textract extraction for page",
                            reported_page_number,
                            "failed due to:",
                            e,
                        )
                        textract_data = {"pages": []}
                        new_textract_request_metadata = "Failed Textract API call"

                    textract_request_metadata.append(new_textract_request_metadata)

                else:
                    # Check if the current reported_page_number exists in the loaded JSON
                    page_exists = any(
                        page["page_no"] == reported_page_number
                        for page in textract_data.get("pages", [])
                    )

                    if not page_exists:  # If the page does not exist, analyze again
                        print(
                            f"Page number {reported_page_number} not found in existing Textract data. Analysing."
                        )

                        try:
                            if not image:
                                page_num, image_path, width, height = (
                                    process_single_page_for_image_conversion(
                                        file_path, page_no
                                    )
                                )

                                # Normalize and validate path safety before opening image
                                normalized_path = os.path.normpath(
                                    os.path.abspath(image_path)
                                )
                                if validate_path_containment(
                                    normalized_path, INPUT_FOLDER
                                ):
                                    image = Image.open(normalized_path)
                                else:
                                    raise ValueError(
                                        f"Unsafe image path detected: {image_path}"
                                    )

                            # Convert the image_path to bytes using an in-memory buffer
                            image_buffer = io.BytesIO()
                            image.save(
                                image_buffer, format="PNG"
                            )  # Save as PNG, or adjust format if needed
                            pdf_page_as_bytes = image_buffer.getvalue()

                            text_blocks, new_textract_request_metadata = (
                                analyse_page_with_textract(
                                    pdf_page_as_bytes,
                                    reported_page_number,
                                    textract_client,
                                    handwrite_signature_checkbox,
                                )
                            )  # Analyse page with Textract

                            # Check if "pages" key exists, if not, initialise it as an empty list
                            if "pages" not in textract_data:
                                textract_data["pages"] = list()

                            # Append the new page data
                            textract_data["pages"].append(text_blocks)

                        except Exception as e:
                            out_message = (
                                "Textract extraction for page "
                                + reported_page_number
                                + " failed due to:"
                                + str(e)
                            )
                            print(out_message)
                            text_blocks = list()
                            new_textract_request_metadata = "Failed Textract API call"

                            # Check if "pages" key exists, if not, initialise it as an empty list
                            if "pages" not in textract_data:
                                textract_data["pages"] = list()

                            raise Exception(out_message)

                        textract_request_metadata.append(new_textract_request_metadata)

                    else:
                        # If the page exists, retrieve the data
                        text_blocks = next(
                            page["data"]
                            for page in textract_data["pages"]
                            if page["page_no"] == reported_page_number
                        )

                (
                    page_line_level_ocr_results,
                    handwriting_or_signature_boxes,
                    page_signature_recogniser_results,
                    page_handwriting_recogniser_results,
                    page_line_level_ocr_results_with_words,
                    selection_element_results,
                    form_key_value_results,
                ) = json_to_ocrresult(
                    text_blocks, page_width, page_height, reported_page_number
                )

                if all_page_line_level_ocr_results_with_words is None:
                    all_page_line_level_ocr_results_with_words = list()

                all_page_line_level_ocr_results_with_words.append(
                    page_line_level_ocr_results_with_words
                )

                if selection_element_results:
                    selection_element_results_list.extend(selection_element_results)
                if form_key_value_results:
                    form_key_value_results_list.extend(form_key_value_results)

            # Convert to DataFrame and add to ongoing logging table
            line_level_ocr_results_df = pd.DataFrame(
                [
                    {
                        "page": page_line_level_ocr_results["page"],
                        "text": result.text,
                        "left": result.left,
                        "top": result.top,
                        "width": result.width,
                        "height": result.height,
                        "line": result.line,
                        "conf": result.conf,
                    }
                    for result in page_line_level_ocr_results["results"]
                ]
            )

            if not line_level_ocr_results_df.empty:  # Ensure there are records to add
                all_line_level_ocr_results_list.extend(
                    line_level_ocr_results_df.to_dict("records")
                )

            if (
                pii_identification_method != NO_REDACTION_PII_OPTION
                or RETURN_PDF_FOR_REVIEW is True
            ):
                page_redaction_bounding_boxes = list()
                comprehend_query_number = 0
                comprehend_query_number_new = 0
                redact_whole_page = False

                if pii_identification_method != NO_REDACTION_PII_OPTION:
                    # Step 2: Analyse text and identify PII
                    if chosen_redact_entities or chosen_redact_comprehend_entities:

                        page_redaction_bounding_boxes, comprehend_query_number_new = (
                            image_analyser.analyze_text(
                                page_line_level_ocr_results["results"],
                                page_line_level_ocr_results_with_words["results"],
                                chosen_redact_comprehend_entities=chosen_redact_comprehend_entities,
                                pii_identification_method=pii_identification_method,
                                comprehend_client=comprehend_client,
                                custom_entities=chosen_redact_entities,
                                language=language,
                                allow_list=allow_list,
                                score_threshold=score_threshold,
                                nlp_analyser=nlp_analyser,
                            )
                        )

                        comprehend_query_number = (
                            comprehend_query_number + comprehend_query_number_new
                        )

                    else:
                        page_redaction_bounding_boxes = list()

                    # Merge redaction bounding boxes that are close together
                    page_merged_redaction_bboxes = merge_img_bboxes(
                        page_redaction_bounding_boxes,
                        page_line_level_ocr_results_with_words["results"],
                        page_signature_recogniser_results,
                        page_handwriting_recogniser_results,
                        handwrite_signature_checkbox,
                    )

                else:
                    page_merged_redaction_bboxes = list()

                if is_pdf(file_path) is True:
                    if redact_whole_page_list:
                        int_reported_page_number = int(reported_page_number)
                        if int_reported_page_number in redact_whole_page_list:
                            redact_whole_page = True
                        else:
                            redact_whole_page = False
                    else:
                        redact_whole_page = False

                    # Check if there are question answer boxes
                    if form_key_value_results_list:
                        page_merged_redaction_bboxes.extend(
                            convert_page_question_answer_to_custom_image_recognizer_results(
                                form_key_value_results_list,
                                page_sizes_df,
                                reported_page_number,
                            )
                        )

                    # 3. Draw the merged boxes
                    ## Apply annotations to pdf with pymupdf
                    redact_result = redact_page_with_pymupdf(
                        pymupdf_page,
                        page_merged_redaction_bboxes,
                        image_path,
                        redact_whole_page=redact_whole_page,
                        original_cropbox=original_cropbox,
                        page_sizes_df=page_sizes_df,
                        input_folder=input_folder,
                    )

                    # Handle dual page objects if returned
                    if isinstance(redact_result[0], tuple):
                        (
                            pymupdf_page,
                            pymupdf_applied_redaction_page,
                        ), page_image_annotations = redact_result
                        # Store the final page with its original page number for later use
                        if not hasattr(redact_image_pdf, "_applied_redaction_pages"):
                            redact_image_pdf._applied_redaction_pages = list()
                        redact_image_pdf._applied_redaction_pages.append(
                            (pymupdf_applied_redaction_page, page_no)
                        )
                    else:
                        pymupdf_page, page_image_annotations = redact_result

                # If an image_path file, draw onto the image_path
                elif is_pdf(file_path) is False:
                    if isinstance(image_path, str):
                        # Normalise and validate path safety before checking existence
                        normalized_path = os.path.normpath(os.path.abspath(image_path))

                        # Check if it's a Gradio temporary file
                        is_gradio_temp = (
                            "gradio" in normalized_path.lower()
                            and "temp" in normalized_path.lower()
                        )

                        if is_gradio_temp or validate_path_containment(
                            normalized_path, INPUT_FOLDER
                        ):
                            image = Image.open(normalized_path)
                        else:
                            print(f"Path validation failed for: {normalized_path}")
                            # You might want to handle this case differently
                            continue  # or raise an exception
                    elif isinstance(image_path, Image.Image):
                        image = image_path
                    else:
                        # Assume image_path is an image
                        image = image_path

                    fill = CUSTOM_BOX_COLOUR  # Fill colour for redactions
                    draw = ImageDraw.Draw(image)

                    all_image_annotations_boxes = list()

                    for box in page_merged_redaction_bboxes:

                        try:
                            x0 = box.left
                            y0 = box.top
                            x1 = x0 + box.width
                            y1 = y0 + box.height
                            label = box.entity_type  # Attempt to get the label
                            text = box.text
                        except AttributeError as e:
                            print(f"Error accessing box attributes: {e}")
                            label = "Redaction"  # Default label if there's an error

                        # Check if coordinates are valid numbers
                        if any(v is None for v in [x0, y0, x1, y1]):
                            print(f"Invalid coordinates for box: {box}")
                            continue  # Skip this box if coordinates are invalid

                        img_annotation_box = {
                            "xmin": x0,
                            "ymin": y0,
                            "xmax": x1,
                            "ymax": y1,
                            "label": label,
                            "color": CUSTOM_BOX_COLOUR,
                            "text": text,
                        }
                        img_annotation_box = fill_missing_box_ids(img_annotation_box)

                        # Directly append the dictionary with the required keys
                        all_image_annotations_boxes.append(img_annotation_box)

                        # Draw the rectangle
                        try:
                            draw.rectangle([x0, y0, x1, y1], fill=fill)
                        except Exception as e:
                            print(f"Error drawing rectangle: {e}")

                    page_image_annotations = {
                        "image": file_path,
                        "boxes": all_image_annotations_boxes,
                    }

                    redacted_image = image.copy()

            # Convert decision process to table
            decision_process_table = pd.DataFrame(
                [
                    {
                        "text": result.text,
                        "xmin": result.left,
                        "ymin": result.top,
                        "xmax": result.left + result.width,
                        "ymax": result.top + result.height,
                        "label": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "page": reported_page_number,
                    }
                    for result in page_merged_redaction_bboxes
                ]
            )

            # all_pages_decision_process_list.append(decision_process_table.to_dict('records'))

            if not decision_process_table.empty:  # Ensure there are records to add
                all_pages_decision_process_list.extend(
                    decision_process_table.to_dict("records")
                )

            decision_process_table = fill_missing_ids(decision_process_table)

            toc = time.perf_counter()

            time_taken = toc - tic

            # Break if time taken is greater than max_time seconds
            if time_taken > max_time:
                print("Processing for", max_time, "seconds, breaking loop.")
                page_break_return = True
                progress.close(_tqdm=progress_bar)
                tqdm._instances.clear()

                if is_pdf(file_path) is False:
                    pdf_image_file_paths.append(redacted_image)  # .append(image_path)
                    pymupdf_doc = pdf_image_file_paths

                # Check if the image_path already exists in annotations_all_pages
                existing_index = next(
                    (
                        index
                        for index, ann in enumerate(annotations_all_pages)
                        if ann["image"] == page_image_annotations["image"]
                    ),
                    None,
                )
                if existing_index is not None:
                    # Replace the existing annotation
                    annotations_all_pages[existing_index] = page_image_annotations
                else:
                    # Append new annotation if it doesn't exist
                    annotations_all_pages.append(page_image_annotations)

                # Save word level options
                if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
                    if original_textract_data != textract_data:
                        # Write the updated existing textract data back to the JSON file
                        secure_file_write(
                            output_folder,
                            file_name + "_textract.json",
                            json.dumps(textract_data, separators=(",", ":")),
                        )

                        if textract_json_file_path not in log_files_output_paths:
                            log_files_output_paths.append(textract_json_file_path)

                all_pages_decision_process_table = pd.DataFrame(
                    all_pages_decision_process_list
                )

                all_line_level_ocr_results_df = pd.DataFrame(
                    all_line_level_ocr_results_list
                )
                if selection_element_results_list:
                    selection_element_results_list_df = pd.DataFrame(
                        selection_element_results_list
                    )
                if form_key_value_results_list:
                    pd.DataFrame(form_key_value_results_list)
                    form_key_value_results_list_df = (
                        convert_question_answer_to_dataframe(
                            form_key_value_results_list, page_sizes_df
                        )
                    )

                current_loop_page += 1

                return (
                    pymupdf_doc,
                    all_pages_decision_process_table,
                    log_files_output_paths,
                    textract_request_metadata,
                    annotations_all_pages,
                    current_loop_page,
                    page_break_return,
                    all_line_level_ocr_results_df,
                    comprehend_query_number,
                    all_page_line_level_ocr_results,
                    all_page_line_level_ocr_results_with_words,
                    selection_element_results_list_df,
                    form_key_value_results_list_df,
                )

        # If it's an image file
        if is_pdf(file_path) is False:
            pdf_image_file_paths.append(redacted_image)  # .append(image_path)
            pymupdf_doc = pdf_image_file_paths

        # Check if the image_path already exists in annotations_all_pages
        existing_index = next(
            (
                index
                for index, ann in enumerate(annotations_all_pages)
                if ann["image"] == page_image_annotations["image"]
            ),
            None,
        )
        if existing_index is not None:
            # Replace the existing annotation
            annotations_all_pages[existing_index] = page_image_annotations
        else:
            # Append new annotation if it doesn't exist
            annotations_all_pages.append(page_image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of chosen page_break_val
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)
            tqdm._instances.clear()

            if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
                # Write the updated existing textract data back to the JSON file
                if original_textract_data != textract_data:
                    secure_file_write(
                        output_folder,
                        file_name + "_textract.json",
                        json.dumps(textract_data, separators=(",", ":")),
                    )

                if textract_json_file_path not in log_files_output_paths:
                    log_files_output_paths.append(textract_json_file_path)

            if text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:
                if (
                    original_all_page_line_level_ocr_results_with_words
                    != all_page_line_level_ocr_results_with_words
                ):
                    # Write the updated existing textract data back to the JSON file
                    with open(
                        all_page_line_level_ocr_results_with_words_json_file_path, "w"
                    ) as json_file:
                        json.dump(
                            all_page_line_level_ocr_results_with_words,
                            json_file,
                            separators=(",", ":"),
                        )  # indent=4 makes the JSON file pretty-printed

                if (
                    all_page_line_level_ocr_results_with_words_json_file_path
                    not in log_files_output_paths
                ):
                    log_files_output_paths.append(
                        all_page_line_level_ocr_results_with_words_json_file_path
                    )

            all_pages_decision_process_table = pd.DataFrame(
                all_pages_decision_process_list
            )

            all_line_level_ocr_results_df = pd.DataFrame(
                all_line_level_ocr_results_list
            )

            if selection_element_results_list:
                selection_element_results_list_df = pd.DataFrame(
                    selection_element_results_list
                )
            if form_key_value_results_list:
                pd.DataFrame(form_key_value_results_list)
                form_key_value_results_list_df = convert_question_answer_to_dataframe(
                    form_key_value_results_list, page_sizes_df
                )

            return (
                pymupdf_doc,
                all_pages_decision_process_table,
                log_files_output_paths,
                textract_request_metadata,
                annotations_all_pages,
                current_loop_page,
                page_break_return,
                all_line_level_ocr_results_df,
                comprehend_query_number,
                all_page_line_level_ocr_results,
                all_page_line_level_ocr_results_with_words,
                selection_element_results_list_df,
                form_key_value_results_list_df,
            )

    if text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        # Write the updated existing textract data back to the JSON file

        if original_textract_data != textract_data:
            secure_file_write(
                output_folder,
                file_name + "_textract.json",
                json.dumps(textract_data, separators=(",", ":")),
            )

        if textract_json_file_path not in log_files_output_paths:
            log_files_output_paths.append(textract_json_file_path)

    if text_extraction_method == TESSERACT_TEXT_EXTRACT_OPTION:
        if (
            original_all_page_line_level_ocr_results_with_words
            != all_page_line_level_ocr_results_with_words
        ):
            # Write the updated existing textract data back to the JSON file
            with open(
                all_page_line_level_ocr_results_with_words_json_file_path, "w"
            ) as json_file:
                json.dump(
                    all_page_line_level_ocr_results_with_words,
                    json_file,
                    separators=(",", ":"),
                )  # indent=4 makes the JSON file pretty-printed

        if (
            all_page_line_level_ocr_results_with_words_json_file_path
            not in log_files_output_paths
        ):
            log_files_output_paths.append(
                all_page_line_level_ocr_results_with_words_json_file_path
            )

    all_pages_decision_process_table = pd.DataFrame(all_pages_decision_process_list)

    all_line_level_ocr_results_df = pd.DataFrame(all_line_level_ocr_results_list)

    # Convert decision table and ocr results to relative coordinates
    all_pages_decision_process_table = divide_coordinates_by_page_sizes(
        all_pages_decision_process_table,
        page_sizes_df,
        xmin="xmin",
        xmax="xmax",
        ymin="ymin",
        ymax="ymax",
    )

    all_line_level_ocr_results_df = divide_coordinates_by_page_sizes(
        all_line_level_ocr_results_df,
        page_sizes_df,
        xmin="left",
        xmax="width",
        ymin="top",
        ymax="height",
    )

    if selection_element_results_list:
        selection_element_results_list_df = pd.DataFrame(selection_element_results_list)
    if form_key_value_results_list:
        pd.DataFrame(form_key_value_results_list)
        form_key_value_results_list_df = convert_question_answer_to_dataframe(
            form_key_value_results_list, page_sizes_df
        )

    return (
        pymupdf_doc,
        all_pages_decision_process_table,
        log_files_output_paths,
        textract_request_metadata,
        annotations_all_pages,
        current_loop_page,
        page_break_return,
        all_line_level_ocr_results_df,
        comprehend_query_number,
        all_page_line_level_ocr_results,
        all_page_line_level_ocr_results_with_words,
        selection_element_results_list_df,
        form_key_value_results_list_df,
    )


###
# PIKEPDF TEXT DETECTION/REDACTION
###


def get_text_container_characters(text_container: LTTextContainer):

    if isinstance(text_container, LTTextContainer):
        characters = [
            char
            for line in text_container
            if isinstance(line, LTTextLine) or isinstance(line, LTTextLineHorizontal)
            for char in line
        ]

        return characters
    return []


def create_line_level_ocr_results_from_characters(
    char_objects: List, line_number: int
) -> Tuple[List[OCRResult], List[List]]:
    """
    Create OCRResult objects based on a list of pdfminer LTChar objects.
    This version is corrected to use the specified OCRResult class definition.
    """
    line_level_results_out = list()
    line_level_characters_out = list()
    character_objects_out = list()

    full_text = ""
    # [x0, y0, x1, y1]
    overall_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for char in char_objects:
        character_objects_out.append(char)

        if isinstance(char, LTAnno):
            added_text = char.get_text()
            full_text += added_text

            if "\n" in added_text:
                if full_text.strip():
                    # Create OCRResult for line
                    line_level_results_out.append(
                        OCRResult(
                            text=full_text.strip(),
                            left=round(overall_bbox[0], 2),
                            top=round(overall_bbox[1], 2),
                            width=round(overall_bbox[2] - overall_bbox[0], 2),
                            height=round(overall_bbox[3] - overall_bbox[1], 2),
                            line=line_number,
                        )
                    )
                    line_level_characters_out.append(character_objects_out)

                # Reset for the next line
                character_objects_out = list()
                full_text = ""
                overall_bbox = [
                    float("inf"),
                    float("inf"),
                    float("-inf"),
                    float("-inf"),
                ]
                line_number += 1
            continue

        # This part handles LTChar objects
        added_text = clean_unicode_text(char.get_text())
        full_text += added_text

        x0, y0, x1, y1 = char.bbox
        overall_bbox[0] = min(overall_bbox[0], x0)
        overall_bbox[1] = min(overall_bbox[1], y0)
        overall_bbox[2] = max(overall_bbox[2], x1)
        overall_bbox[3] = max(overall_bbox[3], y1)

    # Process the last line
    if full_text.strip():
        line_number += 1
        line_ocr_result = OCRResult(
            text=full_text.strip(),
            left=round(overall_bbox[0], 2),
            top=round(overall_bbox[1], 2),
            width=round(overall_bbox[2] - overall_bbox[0], 2),
            height=round(overall_bbox[3] - overall_bbox[1], 2),
            line=line_number,
        )
        line_level_results_out.append(line_ocr_result)
        line_level_characters_out.append(character_objects_out)

    return line_level_results_out, line_level_characters_out


def generate_words_for_line(line_chars: List) -> List[Dict[str, Any]]:
    """
    Generates word-level results for a single, pre-defined line of characters.

    This robust version correctly identifies word breaks by:
    1. Treating specific punctuation characters as standalone words.
    2. Explicitly using space characters (' ') as a primary word separator.
    3. Using a geometric gap between characters as a secondary, heuristic separator.

    Args:
        line_chars: A list of pdfminer.six LTChar/LTAnno objects for one line.

    Returns:
        A list of dictionaries, where each dictionary represents an individual word.
    """
    # We only care about characters with coordinates and text for word building.
    text_chars = [c for c in line_chars if hasattr(c, "bbox") and c.get_text()]

    if not text_chars:
        return []

    # Sort characters by horizontal position for correct processing.
    text_chars.sort(key=lambda c: c.bbox[0])

    # NEW: Define punctuation that should be split into separate words.
    # The hyphen '-' is intentionally excluded to keep words like 'high-tech' together.
    PUNCTUATION_TO_SPLIT = {".", ",", "?", "!", ":", ";", "(", ")", "[", "]", "{", "}"}

    line_words = list()
    current_word_text = ""
    current_word_bbox = [float("inf"), float("inf"), -1, -1]  # [x0, y0, x1, y1]
    prev_char = None

    def finalize_word():
        nonlocal current_word_text, current_word_bbox
        # Only add the word if it contains non-space text
        if current_word_text.strip():
            # bbox from [x0, y0, x1, y1] to your required format
            final_bbox = [
                round(current_word_bbox[0], 2),
                round(current_word_bbox[3], 2),  # Note: using y1 from pdfminer bbox
                round(current_word_bbox[2], 2),
                round(current_word_bbox[1], 2),  # Note: using y0 from pdfminer bbox
            ]
            line_words.append(
                {
                    "text": current_word_text.strip(),
                    "bounding_box": final_bbox,
                    "conf": 100.0,
                }
            )
        # Reset for the next word
        current_word_text = ""
        current_word_bbox = [float("inf"), float("inf"), -1, -1]

    for char in text_chars:
        char_text = clean_unicode_text(char.get_text())

        # 1. NEW: Check for splitting punctuation first.
        if char_text in PUNCTUATION_TO_SPLIT:
            # Finalize any word that came immediately before the punctuation.
            finalize_word()

            # Treat the punctuation itself as a separate word.
            px0, py0, px1, py1 = char.bbox
            punc_bbox = [round(px0, 2), round(py1, 2), round(px1, 2), round(py0, 2)]
            line_words.append(
                {"text": char_text, "bounding_box": punc_bbox, "conf": 100.0}
            )

            prev_char = char
            continue  # Skip to the next character

        # 2. Primary Signal: Is the character a space?
        if char_text.isspace():
            finalize_word()  # End the preceding word
            prev_char = char
            continue  # Skip to the next character, do not add the space to any word

        # 3. Secondary Signal: Is there a large geometric gap?
        if prev_char:
            # A gap is considered a word break if it's larger than a fraction of the font size.
            space_threshold = prev_char.size * 0.25  # 25% of the char size
            min_gap = 1.0  # Or at least 1.0 unit
            gap = (
                char.bbox[0] - prev_char.bbox[2]
            )  # gap = current_char.x0 - prev_char.x1

            if gap > max(space_threshold, min_gap):
                finalize_word()  # Found a gap, so end the previous word.

        # Append the character's text and update the bounding box for the current word
        current_word_text += char_text

        x0, y0, x1, y1 = char.bbox
        current_word_bbox[0] = min(current_word_bbox[0], x0)
        current_word_bbox[1] = min(current_word_bbox[3], y0)  # pdfminer y0 is bottom
        current_word_bbox[2] = max(current_word_bbox[2], x1)
        current_word_bbox[3] = max(current_word_bbox[1], y1)  # pdfminer y1 is top

        prev_char = char

    # After the loop, finalize the last word that was being built.
    finalize_word()

    return line_words


def process_page_to_structured_ocr(
    all_char_objects: List,
    page_number: int,
    text_line_number: int,  # This will now be treated as the STARTING line number
) -> Tuple[Dict[str, Any], List[OCRResult], List[List]]:
    """
    Orchestrates the OCR process, correctly handling multiple lines.

    Returns:
        A tuple containing:
        1. A dictionary with detailed line/word results for the page.
        2. A list of the complete OCRResult objects for each line.
        3. A list of lists, containing the character objects for each line.
    """
    page_data = {"page": str(page_number), "results": {}}

    # Step 1: Get definitive lines and their character groups.
    # This function correctly returns all lines found in the input characters.
    line_results, lines_char_groups = create_line_level_ocr_results_from_characters(
        all_char_objects, text_line_number
    )

    if not line_results:
        return {}, [], []

    # Step 2: Iterate through each found line and generate its words.
    for i, (line_info, char_group) in enumerate(zip(line_results, lines_char_groups)):

        current_line_number = line_info.line  # text_line_number + i

        word_level_results = generate_words_for_line(char_group)

        # Create a unique, incrementing line number for each iteration.

        line_key = f"text_line_{current_line_number}"

        line_bbox = [
            line_info.left,
            line_info.top,
            line_info.left + line_info.width,
            line_info.top + line_info.height,
        ]

        # Now, each line is added to the dictionary with its own unique key.
        page_data["results"][line_key] = {
            "line": current_line_number,  # Use the unique line number
            "text": line_info.text,
            "bounding_box": line_bbox,
            "words": word_level_results,
            "conf": 100.0,
        }

    # The list of OCRResult objects is already correct.
    line_level_ocr_results_list = line_results

    # Return the structured dictionary, the list of OCRResult objects, and the character groups
    return page_data, line_level_ocr_results_list, lines_char_groups


def create_text_redaction_process_results(
    analyser_results, analysed_bounding_boxes, page_num
):
    decision_process_table = pd.DataFrame()

    if len(analyser_results) > 0:
        # Create summary df of annotations to be made
        analysed_bounding_boxes_df_new = pd.DataFrame(analysed_bounding_boxes)

        # Remove brackets and split the string into four separate columns
        # Split the boundingBox list into four separate columns
        analysed_bounding_boxes_df_new[["xmin", "ymin", "xmax", "ymax"]] = (
            analysed_bounding_boxes_df_new["boundingBox"].apply(pd.Series)
        )

        # Convert the new columns to integers (if needed)
        # analysed_bounding_boxes_df_new.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']] = (analysed_bounding_boxes_df_new[['xmin', 'ymin', 'xmax', 'ymax']].astype(float) / 5).round() * 5

        analysed_bounding_boxes_df_text = (
            analysed_bounding_boxes_df_new["result"]
            .astype(str)
            .str.split(",", expand=True)
            .replace(".*: ", "", regex=True)
        )
        analysed_bounding_boxes_df_text.columns = ["label", "start", "end", "score"]
        analysed_bounding_boxes_df_new = pd.concat(
            [analysed_bounding_boxes_df_new, analysed_bounding_boxes_df_text], axis=1
        )
        analysed_bounding_boxes_df_new["page"] = page_num + 1

        decision_process_table = pd.concat(
            [decision_process_table, analysed_bounding_boxes_df_new], axis=0
        ).drop("result", axis=1)

    return decision_process_table


def create_pikepdf_annotations_for_bounding_boxes(analysed_bounding_boxes):
    pikepdf_redaction_annotations_on_page = list()
    for analysed_bounding_box in analysed_bounding_boxes:

        bounding_box = analysed_bounding_box["boundingBox"]
        annotation = Dictionary(
            Type=Name.Annot,
            Subtype=Name.Square,  # Name.Highlight,
            QuadPoints=[
                bounding_box[0],
                bounding_box[3],
                bounding_box[2],
                bounding_box[3],
                bounding_box[0],
                bounding_box[1],
                bounding_box[2],
                bounding_box[1],
            ],
            Rect=[bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
            C=[0, 0, 0],
            IC=[0, 0, 0],
            CA=1,  # Transparency
            T=analysed_bounding_box["result"].entity_type,
            Contents=analysed_bounding_box["text"],
            BS=Dictionary(
                W=0, S=Name.S  # Border width: 1 point  # Border style: solid
            ),
        )
        pikepdf_redaction_annotations_on_page.append(annotation)
    return pikepdf_redaction_annotations_on_page


def redact_text_pdf(
    file_path: str,  # Path to the PDF file to be redacted
    language: str,  # Language of the PDF content
    chosen_redact_entities: List[str],  # List of entities to be redacted
    chosen_redact_comprehend_entities: List[str],
    allow_list: List[str] = None,  # Optional list of allowed entities
    page_min: int = 0,  # Minimum page number to start redaction
    page_max: int = 999,  # Maximum page number to end redaction
    current_loop_page: int = 0,  # Current page being processed in the loop
    page_break_return: bool = False,  # Flag to indicate if a page break should be returned
    annotations_all_pages: List[dict] = list(),  # List of annotations across all pages
    all_line_level_ocr_results_df: pd.DataFrame = pd.DataFrame(
        columns=["page", "text", "left", "top", "width", "height", "line", "conf"]
    ),  # DataFrame for OCR results
    all_pages_decision_process_table: pd.DataFrame = pd.DataFrame(
        columns=[
            "image_path",
            "page",
            "label",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "text",
            "id",
        ]
    ),  # DataFrame for decision process table
    pymupdf_doc: List = list(),  # List of PyMuPDF documents
    all_page_line_level_ocr_results_with_words: List = list(),
    pii_identification_method: str = "Local",
    comprehend_query_number: int = 0,
    comprehend_client="",
    in_deny_list: List[str] = list(),
    redact_whole_page_list: List[str] = list(),
    max_fuzzy_spelling_mistakes_num: int = 1,
    match_fuzzy_whole_phrase_bool: bool = True,
    page_sizes_df: pd.DataFrame = pd.DataFrame(),
    original_cropboxes: List[dict] = list(),
    text_extraction_only: bool = False,
    output_folder: str = OUTPUT_FOLDER,
    input_folder: str = INPUT_FOLDER,
    page_break_val: int = int(PAGE_BREAK_VALUE),  # Value for page break
    max_time: int = int(MAX_TIME_VALUE),
    nlp_analyser: AnalyzerEngine = nlp_analyser,
    progress: Progress = Progress(track_tqdm=True),  # Progress tracking object
):
    """
    Redact chosen entities from a PDF that is made up of multiple pages that are not images.

    Input Variables:
    - file_path: Path to the PDF file to be redacted
    - language: Language of the PDF content
    - chosen_redact_entities: List of entities to be redacted
    - chosen_redact_comprehend_entities: List of entities to be redacted for AWS Comprehend
    - allow_list: Optional list of allowed entities
    - page_min: Minimum page number to start redaction
    - page_max: Maximum page number to end redaction
    - text_extraction_method: Type of analysis to perform
    - current_loop_page: Current page being processed in the loop
    - page_break_return: Flag to indicate if a page break should be returned
    - annotations_all_pages: List of annotations across all pages
    - all_line_level_ocr_results_df: DataFrame for OCR results
    - all_pages_decision_process_table: DataFrame for decision process table
    - pymupdf_doc: List of PyMuPDF documents
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - comprehend_client (optional): A connection to the AWS Comprehend service via the boto3 package.
    - in_deny_list (optional, List[str]): A list of custom words that the user has chosen specifically to redact.
    - redact_whole_page_list (optional, List[str]): A list of pages to fully redact.
    -  max_fuzzy_spelling_mistakes_num (int, optional): The maximum number of spelling mistakes allowed in a searched phrase for fuzzy matching. Can range from 0-9.
    -  match_fuzzy_whole_phrase_bool (bool, optional): A boolean where 'True' means that the whole phrase is fuzzy matched, and 'False' means that each word is fuzzy matched separately (excluding stop words).
    - page_sizes_df (pd.DataFrame, optional): A pandas dataframe of PDF page sizes in PDF or image format.
    - original_cropboxes (List[dict], optional): A list of dictionaries containing pymupdf cropbox information.
    - text_extraction_only (bool, optional): Should the function only extract text, or also do redaction.
    - language (str, optional): The language to do AWS Comprehend calls. Defaults to value of language if not provided.
    - output_folder (str, optional): The output folder for the function
    - input_folder (str, optional): The folder for file inputs.
    - page_break_val: Value for page break
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.
    - nlp_analyser (AnalyzerEngine, optional): The nlp_analyser object to use for entity detection. Defaults to nlp_analyser.
    - progress: Progress tracking object
    """

    tic = time.perf_counter()

    if isinstance(all_line_level_ocr_results_df, pd.DataFrame):
        all_line_level_ocr_results_list = [all_line_level_ocr_results_df]

    if isinstance(all_pages_decision_process_table, pd.DataFrame):
        # Convert decision outputs to list of dataframes:
        all_pages_decision_process_list = [all_pages_decision_process_table]

    if pii_identification_method == "AWS Comprehend" and comprehend_client == "":
        out_message = "Connection to AWS Comprehend service not found."
        raise Exception(out_message)

    # Try updating the supported languages for the spacy analyser
    try:
        nlp_analyser = create_nlp_analyser(language, existing_nlp_analyser=nlp_analyser)
        # Check list of nlp_analyser recognisers and languages
        if language != "en":
            gr.Info(
                f"Language: {language} only supports the following entity detection: {str(nlp_analyser.registry.get_supported_entities(languages=[language]))}"
            )

    except Exception as e:
        print(f"Error creating nlp_analyser for {language}: {e}")
        raise Exception(f"Error creating nlp_analyser for {language}: {e}")

    # Update custom word list analyser object with any new words that have been added to the custom deny list
    if in_deny_list:
        nlp_analyser.registry.remove_recognizer("CUSTOM")
        new_custom_recogniser = custom_word_list_recogniser(in_deny_list)
        nlp_analyser.registry.add_recognizer(new_custom_recogniser)

        nlp_analyser.registry.remove_recognizer("CustomWordFuzzyRecognizer")
        new_custom_fuzzy_recogniser = CustomWordFuzzyRecognizer(
            supported_entities=["CUSTOM_FUZZY"],
            custom_list=in_deny_list,
            spelling_mistakes_max=max_fuzzy_spelling_mistakes_num,
            search_whole_phrase=match_fuzzy_whole_phrase_bool,
        )
        nlp_analyser.registry.add_recognizer(new_custom_fuzzy_recogniser)

    # Open with Pikepdf to get text lines
    pikepdf_pdf = Pdf.open(file_path)
    number_of_pages = len(pikepdf_pdf.pages)

    # file_name = get_file_name_without_type(file_path)

    if not all_page_line_level_ocr_results_with_words:
        all_page_line_level_ocr_results_with_words = list()

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages

    if page_min <= 0:
        page_min = 0
    else:
        page_min = page_min - 1

    print("Page range is", str(page_min + 1), "to", str(page_max))

    # Run through each page in document to 1. Extract text and then 2. Create redaction boxes
    progress_bar = tqdm(
        range(current_loop_page, number_of_pages),
        unit="pages remaining",
        desc="Redacting pages",
    )

    for page_no in progress_bar:
        reported_page_number = str(page_no + 1)
        # Create annotations for every page, even if blank.

        # Try to find image path location
        try:
            image_path = page_sizes_df.loc[
                page_sizes_df["page"] == int(reported_page_number), "image_path"
            ].iloc[0]
        except Exception as e:
            print("Image path not found:", e)
            image_path = ""

        page_image_annotations = {"image": image_path, "boxes": []}  # image

        pymupdf_page = pymupdf_doc.load_page(page_no)
        pymupdf_page.set_cropbox(pymupdf_page.mediabox)  # Set CropBox to MediaBox

        if page_min <= page_no < page_max:
            # Go page by page
            for page_layout in extract_pages(
                file_path, page_numbers=[page_no], maxpages=1
            ):

                all_page_line_text_extraction_characters = list()
                all_page_line_level_text_extraction_results_list = list()
                page_analyser_results = list()
                page_redaction_bounding_boxes = list()

                characters = list()
                pikepdf_redaction_annotations_on_page = list()
                page_decision_process_table = pd.DataFrame(
                    columns=[
                        "image_path",
                        "page",
                        "label",
                        "xmin",
                        "xmax",
                        "ymin",
                        "ymax",
                        "text",
                        "id",
                    ]
                )
                page_text_ocr_outputs = pd.DataFrame(
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
                page_text_ocr_outputs_list = list()

                text_line_no = 1
                for n, text_container in enumerate(page_layout):
                    characters = list()

                    if isinstance(text_container, LTTextContainer) or isinstance(
                        text_container, LTAnno
                    ):
                        characters = get_text_container_characters(text_container)
                        # text_line_no += 1

                    # Create dataframe for all the text on the page
                    # line_level_text_results_list, line_characters = create_line_level_ocr_results_from_characters(characters)

                    # line_level_ocr_results_with_words = generate_word_level_ocr(characters, page_number=int(reported_page_number), text_line_number=text_line_no)

                    (
                        line_level_ocr_results_with_words,
                        line_level_text_results_list,
                        line_characters,
                    ) = process_page_to_structured_ocr(
                        characters,
                        page_number=int(reported_page_number),
                        text_line_number=text_line_no,
                    )

                    text_line_no += len(line_level_text_results_list)

                    ### Create page_text_ocr_outputs (OCR format outputs)
                    if line_level_text_results_list:
                        # Convert to DataFrame and add to ongoing logging table
                        line_level_text_results_df = pd.DataFrame(
                            [
                                {
                                    "page": page_no + 1,
                                    "text": (result.text).strip(),
                                    "left": result.left,
                                    "top": result.top,
                                    "width": result.width,
                                    "height": result.height,
                                    "line": result.line,
                                    "conf": 100.0,
                                }
                                for result in line_level_text_results_list
                            ]
                        )

                        page_text_ocr_outputs_list.append(line_level_text_results_df)

                    all_page_line_level_text_extraction_results_list.extend(
                        line_level_text_results_list
                    )
                    all_page_line_text_extraction_characters.extend(line_characters)
                    all_page_line_level_ocr_results_with_words.append(
                        line_level_ocr_results_with_words
                    )

                if page_text_ocr_outputs_list:
                    page_text_ocr_outputs = pd.concat(page_text_ocr_outputs_list)
                else:
                    page_text_ocr_outputs = pd.DataFrame(
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

                ### REDACTION
                if pii_identification_method != NO_REDACTION_PII_OPTION:
                    if chosen_redact_entities or chosen_redact_comprehend_entities:
                        page_redaction_bounding_boxes = run_page_text_redaction(
                            language,
                            chosen_redact_entities,
                            chosen_redact_comprehend_entities,
                            all_page_line_level_text_extraction_results_list,
                            all_page_line_text_extraction_characters,
                            page_analyser_results,
                            page_redaction_bounding_boxes,
                            comprehend_client,
                            allow_list,
                            pii_identification_method,
                            nlp_analyser,
                            score_threshold,
                            custom_entities,
                            comprehend_query_number,
                        )

                        # Annotate redactions on page
                        pikepdf_redaction_annotations_on_page = (
                            create_pikepdf_annotations_for_bounding_boxes(
                                page_redaction_bounding_boxes
                            )
                        )

                    else:
                        pikepdf_redaction_annotations_on_page = list()

                    # Make pymupdf page redactions
                    if redact_whole_page_list:
                        int_reported_page_number = int(reported_page_number)
                        if int_reported_page_number in redact_whole_page_list:
                            redact_whole_page = True
                        else:
                            redact_whole_page = False
                    else:
                        redact_whole_page = False

                    redact_result = redact_page_with_pymupdf(
                        pymupdf_page,
                        pikepdf_redaction_annotations_on_page,
                        image_path,
                        redact_whole_page=redact_whole_page,
                        convert_pikepdf_to_pymupdf_coords=True,
                        original_cropbox=original_cropboxes[page_no],
                        page_sizes_df=page_sizes_df,
                        input_folder=input_folder,
                    )

                    # Handle dual page objects if returned
                    if isinstance(redact_result[0], tuple):
                        (
                            pymupdf_page,
                            pymupdf_applied_redaction_page,
                        ), page_image_annotations = redact_result
                        # Store the final page with its original page number for later use
                        if not hasattr(redact_text_pdf, "_applied_redaction_pages"):
                            redact_text_pdf._applied_redaction_pages = list()
                        redact_text_pdf._applied_redaction_pages.append(
                            (pymupdf_applied_redaction_page, page_no)
                        )
                    else:
                        pymupdf_page, page_image_annotations = redact_result

                    # Create decision process table
                    page_decision_process_table = create_text_redaction_process_results(
                        page_analyser_results,
                        page_redaction_bounding_boxes,
                        current_loop_page,
                    )

                    if not page_decision_process_table.empty:
                        all_pages_decision_process_list.append(
                            page_decision_process_table
                        )

                # Else, user chose not to run redaction
                else:
                    pass

                # Join extracted text outputs for all lines together
                if not page_text_ocr_outputs.empty:
                    page_text_ocr_outputs = page_text_ocr_outputs.sort_values(
                        ["line"]
                    ).reset_index(drop=True)
                    page_text_ocr_outputs = page_text_ocr_outputs.loc[
                        :,
                        [
                            "page",
                            "text",
                            "left",
                            "top",
                            "width",
                            "height",
                            "line",
                            "conf",
                        ],
                    ]
                    all_line_level_ocr_results_list.append(page_text_ocr_outputs)

                toc = time.perf_counter()

                time_taken = toc - tic

                # Break if time taken is greater than max_time seconds
                if time_taken > max_time:
                    print("Processing for", max_time, "seconds, breaking.")
                    page_break_return = True
                    progress.close(_tqdm=progress_bar)
                    tqdm._instances.clear()

                    # Check if the image already exists in annotations_all_pages
                    existing_index = next(
                        (
                            index
                            for index, ann in enumerate(annotations_all_pages)
                            if ann["image"] == page_image_annotations["image"]
                        ),
                        None,
                    )
                    if existing_index is not None:
                        # Replace the existing annotation
                        annotations_all_pages[existing_index] = page_image_annotations
                    else:
                        # Append new annotation if it doesn't exist
                        annotations_all_pages.append(page_image_annotations)

                    # Write logs
                    all_pages_decision_process_table = pd.concat(
                        all_pages_decision_process_list
                    )
                    all_line_level_ocr_results_df = pd.concat(
                        all_line_level_ocr_results_list
                    )

                    current_loop_page += 1

                    return (
                        pymupdf_doc,
                        all_pages_decision_process_table,
                        all_line_level_ocr_results_df,
                        annotations_all_pages,
                        current_loop_page,
                        page_break_return,
                        comprehend_query_number,
                        all_page_line_level_ocr_results_with_words,
                    )

        # Check if the image already exists in annotations_all_pages
        existing_index = next(
            (
                index
                for index, ann in enumerate(annotations_all_pages)
                if ann["image"] == page_image_annotations["image"]
            ),
            None,
        )
        if existing_index is not None:
            # Replace the existing annotation
            annotations_all_pages[existing_index] = page_image_annotations
        else:
            # Append new annotation if it doesn't exist
            annotations_all_pages.append(page_image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of page_break_val
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)

            # Write logs
            all_pages_decision_process_table = pd.concat(
                all_pages_decision_process_list
            )

            return (
                pymupdf_doc,
                all_pages_decision_process_table,
                all_line_level_ocr_results_df,
                annotations_all_pages,
                current_loop_page,
                page_break_return,
                comprehend_query_number,
                all_page_line_level_ocr_results_with_words,
            )

    # Write all page outputs
    all_pages_decision_process_table = pd.concat(all_pages_decision_process_list)
    all_line_level_ocr_results_df = pd.concat(all_line_level_ocr_results_list)

    # Convert decision table to relative coordinates
    all_pages_decision_process_table = divide_coordinates_by_page_sizes(
        all_pages_decision_process_table,
        page_sizes_df,
        xmin="xmin",
        xmax="xmax",
        ymin="ymin",
        ymax="ymax",
    )

    # Coordinates need to be reversed for ymin and ymax to match with image annotator objects downstream
    all_pages_decision_process_table["ymin"] = reverse_y_coords(
        all_pages_decision_process_table, "ymin"
    )
    all_pages_decision_process_table["ymax"] = reverse_y_coords(
        all_pages_decision_process_table, "ymax"
    )

    # Convert decision table to relative coordinates
    all_line_level_ocr_results_df = divide_coordinates_by_page_sizes(
        all_line_level_ocr_results_df,
        page_sizes_df,
        xmin="left",
        xmax="width",
        ymin="top",
        ymax="height",
    )

    # print("all_line_level_ocr_results_df:", all_line_level_ocr_results_df)

    # Coordinates need to be reversed for ymin and ymax to match with image annotator objects downstream
    if not all_line_level_ocr_results_df.empty:
        all_line_level_ocr_results_df["top"] = reverse_y_coords(
            all_line_level_ocr_results_df, "top"
        )

    # Remove empty dictionary items from ocr results with words
    all_page_line_level_ocr_results_with_words = [
        d for d in all_page_line_level_ocr_results_with_words if d
    ]

    return (
        pymupdf_doc,
        all_pages_decision_process_table,
        all_line_level_ocr_results_df,
        annotations_all_pages,
        current_loop_page,
        page_break_return,
        comprehend_query_number,
        all_page_line_level_ocr_results_with_words,
    )
