import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

import defusedxml
import defusedxml.ElementTree as defused_etree
import defusedxml.minidom as defused_minidom

# Defuse the standard library XML modules for security
defusedxml.defuse_stdlib()

import gradio as gr
import numpy as np
import pandas as pd
import polars as pl
import pymupdf
from gradio_image_annotation.image_annotator import AnnotatedImageData
from PIL import Image, ImageDraw
from pymupdf import Document, Rect

from tools.config import (
    COMPRESS_REDACTED_PDF,
    CUSTOM_BOX_COLOUR,
    ENABLE_PARALLEL_FILES_APPLY_REDACTIONS,
    ENABLE_REVIEW_CSV_PARALLELISM,
    INPUT_FOLDER,
    MAX_IMAGE_PIXELS,
    MAX_WORKERS,
    OUTPUT_FOLDER,
    PROFILE_REDACTION_APPLY,
    RETURN_PDF_FOR_REVIEW,
    USE_POLARS_FOR_REVIEW,
)
from tools.file_conversion import (
    convert_annotation_data_to_dataframe,
    convert_annotation_json_to_review_df,
    convert_review_df_to_annotation_json,
    divide_coordinates_by_page_sizes,
    divide_coordinates_by_page_sizes_pl,
    fill_missing_ids,
    is_pdf,
    multiply_coordinates_by_page_sizes,
    process_single_page_for_image_conversion,
    remove_duplicate_images_with_blank_boxes,
    save_pdf_with_or_without_compression,
)
from tools.file_redaction import redact_page_with_pymupdf, set_cropbox_safely
from tools.helper_functions import (
    _generate_unique_ids,
    detect_file_type,
    get_file_name_without_type,
)
from tools.secure_path_utils import (
    secure_file_write,
)

if not MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = None

# Chunked review CSV: minimum number of pages to enable parallel annotation->DF build
REVIEW_CSV_PARALLEL_MIN_PAGES = 20
# Pages per chunk when building review DF from annotations in parallel
REVIEW_CSV_PAGES_PER_CHUNK = 15


def _ensure_box_colour_string(colour):
    """Ensure colour is a string for gradio_image_annotation (JS expects .startsWith)."""
    if colour is None:
        return "(0, 0, 0)"
    if isinstance(colour, str):
        return colour
    if isinstance(colour, (tuple, list)) and len(colour) >= 3:
        return f"({int(colour[0])}, {int(colour[1])}, {int(colour[2])})"
    return "(0, 0, 0)"


def decrease_page(number: int, all_annotations: dict):
    """
    Decrease page number for review redactions page.
    """
    if not all_annotations:
        raise Warning("No annotator object loaded")

    if number > 1:
        return number - 1, number - 1
    elif number <= 1:
        # return 1, 1
        raise Warning("At first page")
    else:
        raise Warning("At first page")


def increase_page(number: int, all_annotations: dict):
    """
    Increase page number for review redactions page.
    """

    if not all_annotations:
        raise Warning("No annotator object loaded")
        # return 1, 1

    max_pages = len(all_annotations)

    if number < max_pages:
        return number + 1, number + 1
    # elif number == max_pages:
    #    return max_pages, max_pages
    else:
        raise Warning("At last page")


def update_zoom(
    current_zoom_level: int, annotate_current_page: int, decrease: bool = True
):
    if decrease is False:
        if current_zoom_level >= 70:
            current_zoom_level -= 10
    else:
        if current_zoom_level < 110:
            current_zoom_level += 10

    return current_zoom_level, annotate_current_page


def update_dropdown_list_based_on_dataframe(
    df: pd.DataFrame, column: str
) -> List["str"]:
    """
    Gather unique elements from a string pandas Series, then append 'ALL' to the start and return the list.
    """
    if isinstance(df, pd.DataFrame):
        # Check if the Series is empty or all NaN
        if column not in df.columns or df[column].empty or df[column].isna().all():
            return ["ALL"]
        elif column != "page":
            entities = df[column].astype(str).unique().tolist()
            entities_for_drop = sorted(entities)
            entities_for_drop.insert(0, "ALL")
        else:
            # Ensure the column can be converted to int - assumes it is the page column
            try:
                entities = df[column].astype(int).unique()
                entities_for_drop = sorted(entities)
                entities_for_drop = [
                    str(e) for e in entities_for_drop
                ]  # Convert back to string
                entities_for_drop.insert(0, "ALL")
            except ValueError:
                return ["ALL"]  # Handle case where conversion fails

        return entities_for_drop  # Ensure to return the list
    else:
        return ["ALL"]


def get_filtered_recogniser_dataframe_and_dropdowns(
    page_image_annotator_object: AnnotatedImageData,
    recogniser_dataframe_base: pd.DataFrame,
    recogniser_dropdown_value: str,
    text_dropdown_value: str,
    page_dropdown_value: str,
    review_df: pd.DataFrame = list(),
    page_sizes: List[str] = list(),
):
    """
    Create a filtered recogniser dataframe and associated dropdowns based on current information in the image annotator and review data frame.
    """

    recogniser_entities_list = ["Redaction"]
    recogniser_dataframe_out = recogniser_dataframe_base
    pd.DataFrame()
    review_dataframe = review_df

    try:

        review_dataframe = convert_annotation_json_to_review_df(
            page_image_annotator_object, review_df, page_sizes
        )

        recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
            review_dataframe, "label"
        )
        recogniser_entities_drop_spec = dict(
            value=recogniser_dropdown_value,
            choices=recogniser_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        )

        # This is the choice list for entities when creating a new redaction box
        recogniser_entities_list = [
            entity
            for entity in recogniser_entities_for_drop.copy()
            if entity != "Redaction" and entity != "ALL"
        ]  # Remove any existing 'Redaction'
        recogniser_entities_list.insert(
            0, "Redaction"
        )  # Add 'Redaction' to the start of the list

        text_entities_for_drop = update_dropdown_list_based_on_dataframe(
            review_dataframe, "text"
        )
        text_entities_drop_spec = dict(
            value=text_dropdown_value,
            choices=text_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        )

        page_entities_for_drop = update_dropdown_list_based_on_dataframe(
            review_dataframe, "page"
        )
        page_entities_drop_spec = dict(
            value=page_dropdown_value,
            choices=page_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        )

        # recogniser_dataframe_out_gr = gr.Dataframe(
        #     review_dataframe[["page", "label", "text", "id"]],
        #     show_search="filter",
        #     type="pandas",
        #     headers=["page", "label", "text", "id"],
        #     wrap=True,
        #     max_height=400,
        # )
        # recogniser_dataframe_out_gr = pd.DataFrame(
        #     review_dataframe[["page", "label", "text", "id"]]

        recogniser_dataframe_out = review_dataframe.loc[
            :, ["page", "label", "text", "id"]
        ]

    except Exception as e:
        print("Could not extract recogniser information:", e)
        recogniser_dataframe_out = recogniser_dataframe_base.loc[
            :, ["page", "label", "text", "id"]
        ]

        label_choices = review_dataframe["label"].astype(str).unique().tolist()
        text_choices = review_dataframe["text"].astype(str).unique().tolist()
        page_choices = review_dataframe["page"].astype(str).unique().tolist()

        recogniser_entities_drop_spec = dict(
            value=recogniser_dropdown_value,
            choices=label_choices,
            allow_custom_value=True,
            interactive=True,
        )
        recogniser_entities_list = ["Redaction"]
        text_entities_drop_spec = dict(
            value=text_dropdown_value,
            choices=text_choices,
            allow_custom_value=True,
            interactive=True,
        )
        page_entities_drop_spec = dict(
            value=page_dropdown_value,
            choices=page_choices,
            allow_custom_value=True,
            interactive=True,
        )

    return (
        recogniser_dataframe_out,
        recogniser_dataframe_out,
        recogniser_entities_drop_spec,
        recogniser_entities_list,
        text_entities_drop_spec,
        page_entities_drop_spec,
    )


def update_recogniser_dataframes(
    page_image_annotator_object: AnnotatedImageData,
    recogniser_dataframe_base: pd.DataFrame,
    recogniser_entities_dropdown_value: str = "ALL",
    text_dropdown_value: str = "ALL",
    page_dropdown_value: str = "ALL",
    review_df: pd.DataFrame = list(),
    page_sizes: list[str] = list(),
):
    """
    Update recogniser dataframe information that appears alongside the pdf pages on the review screen.
    """
    recogniser_entities_list = ["Redaction"]
    recogniser_dataframe_out = pd.DataFrame()
    recogniser_dataframe_out_gr = pd.DataFrame()

    # If base recogniser dataframe is empy, need to create it.
    if recogniser_dataframe_base.empty:
        (
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            recogniser_entities_drop_spec,
            recogniser_entities_list,
            text_entities_drop_spec,
            page_entities_drop_spec,
        ) = get_filtered_recogniser_dataframe_and_dropdowns(
            page_image_annotator_object,
            recogniser_dataframe_base,
            recogniser_entities_dropdown_value,
            text_dropdown_value,
            page_dropdown_value,
            review_df,
            page_sizes,
        )
        return (
            recogniser_entities_list,
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            gr.update(**recogniser_entities_drop_spec),
            gr.update(**text_entities_drop_spec),
            gr.update(**page_entities_drop_spec),
        )
    elif recogniser_dataframe_base.iloc[0, 0] == "":
        (
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            recogniser_entities_drop_spec,
            recogniser_entities_list,
            text_entities_drop_spec,
            page_entities_drop_spec,
        ) = get_filtered_recogniser_dataframe_and_dropdowns(
            page_image_annotator_object,
            recogniser_dataframe_base,
            recogniser_entities_dropdown_value,
            text_dropdown_value,
            page_dropdown_value,
            review_df,
            page_sizes,
        )
        return (
            recogniser_entities_list,
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            gr.update(**recogniser_entities_drop_spec),
            gr.update(**text_entities_drop_spec),
            gr.update(**page_entities_drop_spec),
        )
    else:
        (
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            _recogniser_drop_spec,
            recogniser_entities_list,
            _text_drop_spec,
            _page_drop_spec,
        ) = get_filtered_recogniser_dataframe_and_dropdowns(
            page_image_annotator_object,
            recogniser_dataframe_base,
            recogniser_entities_dropdown_value,
            text_dropdown_value,
            page_dropdown_value,
            review_df,
            page_sizes,
        )

        review_dataframe, text_entities_drop, page_entities_drop = (
            update_entities_df_recogniser_entities(
                recogniser_entities_dropdown_value,
                recogniser_dataframe_out,
                page_dropdown_value,
                text_dropdown_value,
            )
        )

        # recogniser_dataframe_out_gr = gr.Dataframe(
        #     review_dataframe[["page", "label", "text", "id"]],
        #     show_search="filter",
        #     type="pandas",
        #     headers=["page", "label", "text", "id"],
        #     wrap=True,
        #     max_height=400,
        # )

        recogniser_dataframe_out_gr = review_dataframe[["page", "label", "text", "id"]]

        recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
            recogniser_dataframe_out, "label"
        )

        recogniser_entities_list_base = (
            recogniser_dataframe_out["label"].astype(str).unique().tolist()
        )

        # Recogniser entities list is the list of choices that appear when you make a new redaction box
        recogniser_entities_list = [
            entity for entity in recogniser_entities_list_base if entity != "Redaction"
        ]
        recogniser_entities_list.insert(0, "Redaction")

        return (
            recogniser_entities_list,
            recogniser_dataframe_out_gr,
            recogniser_dataframe_out,
            gr.update(
                value=recogniser_entities_dropdown_value,
                choices=recogniser_entities_for_drop,
                allow_custom_value=True,
                interactive=True,
            ),
            text_entities_drop,
            page_entities_drop,
        )


def undo_last_removal(
    backup_review_state: pd.DataFrame,
    backup_image_annotations_state: list[dict],
    backup_recogniser_entity_dataframe_base: pd.DataFrame,
):

    if backup_image_annotations_state:
        return (
            backup_review_state,
            backup_image_annotations_state,
            backup_recogniser_entity_dataframe_base,
        )
    else:
        raise Warning("No actions have been taken to undo")


def update_annotator_page_from_review_df(
    review_df: pd.DataFrame,
    image_file_paths: List[str],
    page_sizes: List[dict],
    current_image_annotations_state: List[dict],
    current_page_annotator: object,
    selected_recogniser_entity_df_row: pd.DataFrame,
    input_folder: str,
    doc_full_file_name_textbox: str,
) -> Tuple[object, List[dict], int, List[dict], pd.DataFrame, int]:
    """
    Update the visible annotation object and related objects with the latest review file information,
    optimising by processing only the current page's data.

    Args:
        review_df (pd.DataFrame): The DataFrame containing review information for all annotations.
        image_file_paths (List[str]): List of image file paths, one per document page.
        page_sizes (List[dict]): List of dictionaries holding page size metadata (width/height etc) for each page.
        current_image_annotations_state (List[dict]): Annotation state for all pages; typically a list of dicts, one per page.
        current_page_annotator (object): The annotation object for the currently visible page, usually a dict or a custom annotation object.
        selected_recogniser_entity_df_row (pd.DataFrame): DataFrame row of the currently selected recogniser/entity, used to extract current page info.
        input_folder (str): Folder containing input source data.
        doc_full_file_name_textbox (str): The full filename of the document as displayed in the textbox/UI.

    Returns:
        Tuple[object, List[dict], int, List[dict], pd.DataFrame, int]:
            A tuple containing:
                - The updated annotation object for the current page.
                - The updated annotation state for all pages.
                - The current page number being displayed (1-based).
                - The annotation state for all pages after any updates.
                - The possibly updated recogniser/entity DataFrame row.
                - The previous page number to annotate (for navigation/state logic).
    """
    # Assume current_image_annotations_state is List[dict] and current_page_annotator is dict
    out_image_annotations_state: List[dict] = list(
        current_image_annotations_state
    )  # Make a copy to avoid modifying input in place
    out_current_page_annotator: dict = current_page_annotator

    # Get the target page number from the selected row
    # Safely access the page number, handling potential errors or empty DataFrame
    gradio_annotator_current_page_number: int = 1
    annotate_previous_page: int = (
        0  # Renaming for clarity if needed, matches original output
    )

    if (
        not selected_recogniser_entity_df_row.empty
        and "page" in selected_recogniser_entity_df_row.columns
    ):
        try:
            selected_page = selected_recogniser_entity_df_row["page"].iloc[0]
            gradio_annotator_current_page_number = int(selected_page)
            annotate_previous_page = (
                gradio_annotator_current_page_number  # Store original page number
            )
        except (IndexError, ValueError, TypeError):
            print(
                "Warning: Could not extract valid page number from selected_recogniser_entity_df_row. Defaulting to page 1."
            )
            gradio_annotator_current_page_number = (
                1  # Or 0 depending on 1-based vs 0-based indexing elsewhere
            )

    # Ensure page number is valid and 1-based for external display/logic
    if gradio_annotator_current_page_number <= 0:
        gradio_annotator_current_page_number = 1

    page_max_reported = len(page_sizes)  # len(out_image_annotations_state)
    if gradio_annotator_current_page_number > page_max_reported:
        print("current page is greater than highest page:", page_max_reported)
        gradio_annotator_current_page_number = page_max_reported  # Cap at max pages

    page_num_reported_zero_indexed = gradio_annotator_current_page_number - 1

    # Process page sizes DataFrame early, as it's needed for image path handling and potentially coordinate multiplication
    page_sizes_df = pd.DataFrame(page_sizes)
    if not page_sizes_df.empty:
        # Safely convert page column to numeric and then int
        page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
        page_sizes_df.dropna(subset=["page"], inplace=True)
        if not page_sizes_df.empty:
            page_sizes_df["page"] = page_sizes_df["page"].astype(int)
        else:
            print("Warning: Page sizes DataFrame became empty after processing.")

    if not review_df.empty:
        # Filter review_df for the current page
        # Ensure 'page' column in review_df is comparable to page_num_reported
        if "page" in review_df.columns:
            review_df["page"] = (
                pd.to_numeric(review_df["page"], errors="coerce").fillna(-1).astype(int)
            )

            current_image_path = out_image_annotations_state[
                page_num_reported_zero_indexed
            ]["image"]

            replaced_image_path, page_sizes_df = (
                replace_placeholder_image_with_real_image(
                    doc_full_file_name_textbox,
                    current_image_path,
                    page_sizes_df,
                    gradio_annotator_current_page_number,
                    input_folder,
                )
            )

            # page_sizes_df has been changed - save back to page_sizes_object
            page_sizes = page_sizes_df.to_dict(orient="records")
            review_df.loc[
                review_df["page"] == gradio_annotator_current_page_number, "image"
            ] = replaced_image_path
            images_list = list(page_sizes_df["image_path"])
            images_list[page_num_reported_zero_indexed] = replaced_image_path
            out_image_annotations_state[page_num_reported_zero_indexed][
                "image"
            ] = replaced_image_path

            current_page_review_df = review_df[
                review_df["page"] == gradio_annotator_current_page_number
            ].copy()
            current_page_review_df = multiply_coordinates_by_page_sizes(
                current_page_review_df, page_sizes_df
            )

        else:
            print(
                f"Warning: 'page' column not found in review_df. Cannot filter for page {gradio_annotator_current_page_number}. Skipping update from review_df."
            )
            current_page_review_df = pd.DataFrame()  # Empty dataframe if filter fails

        if not current_page_review_df.empty:
            # Convert the current page's review data to annotation list format for *this page*

            current_page_annotations_list = list()
            # Define expected annotation dict keys, including 'image', 'page', coords, 'label', 'text', 'color' etc.
            # Assuming review_df has compatible columns
            expected_annotation_keys = [
                "label",
                "color",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "text",
                "id",
            ]  # Add/remove as needed

            # Ensure necessary columns exist in current_page_review_df before converting rows
            for key in expected_annotation_keys:
                if key not in current_page_review_df.columns:
                    # Add missing column with default value. Use 0.0 for coords so
                    # gradio_image_annotation never receives None/NaN (causes TypeError in preprocess_boxes).
                    default_value = (
                        0.0 if key in ["xmin", "ymin", "xmax", "ymax"] else ""
                    )
                    current_page_review_df[key] = default_value

            # Ensure coord columns have no NaN/None so image_annotator preprocess_boxes doesn't raise TypeError
            for coord in ["xmin", "ymin", "xmax", "ymax"]:
                if coord in current_page_review_df.columns:
                    current_page_review_df[coord] = pd.to_numeric(
                        current_page_review_df[coord], errors="coerce"
                    ).fillna(0.0)

            # Convert filtered DataFrame rows to list of dicts
            # Using .to_dict(orient='records') is efficient for this
            current_page_annotations_list_raw = current_page_review_df[
                expected_annotation_keys
            ].to_dict(orient="records")

            current_page_annotations_list = current_page_annotations_list_raw

            # Update the annotations state for the current page
            page_state_entry_found = False
            for i, page_state_entry in enumerate(out_image_annotations_state):
                # Assuming page_state_entry has a 'page' key (1-based)

                from tools.secure_regex_utils import (
                    safe_extract_page_number_from_filename,
                )

                page_no = safe_extract_page_number_from_filename(
                    page_state_entry["image"]
                )
                if page_no is None:
                    page_no = 0

                if (
                    "image" in page_state_entry
                    and page_no == page_num_reported_zero_indexed
                ):
                    # Replace the annotations list for this page with the new list from review_df
                    out_image_annotations_state[i][
                        "boxes"
                    ] = current_page_annotations_list

                    # Update the image path as well, based on review_df if available, or keep existing
                    # Assuming review_df has an 'image' column for this page
                    if (
                        "image" in current_page_review_df.columns
                        and not current_page_review_df.empty
                    ):
                        # Use the image path from the first row of the filtered review_df
                        out_image_annotations_state[i]["image"] = (
                            current_page_review_df["image"].iloc[0]
                        )
                    page_state_entry_found = True
                    break

            if not page_state_entry_found:
                print(
                    f"Warning: Entry for page {gradio_annotator_current_page_number} not found in current_image_annotations_state. Cannot update page annotations."
                )

    # --- Image Path and Page Size Handling ---
    # Get the image path for the current page from the updated state
    current_image_path = None
    if (
        len(out_image_annotations_state) > page_num_reported_zero_indexed
        and "image" in out_image_annotations_state[page_num_reported_zero_indexed]
    ):
        current_image_path = out_image_annotations_state[
            page_num_reported_zero_indexed
        ]["image"]
    else:
        print(
            f"Warning: Could not get image path from state for page index {page_num_reported_zero_indexed}."
        )

    # Replace placeholder image with real image path if needed
    if current_image_path and not page_sizes_df.empty:
        try:
            replaced_image_path, page_sizes_df = (
                replace_placeholder_image_with_real_image(
                    doc_full_file_name_textbox,
                    current_image_path,
                    page_sizes_df,
                    gradio_annotator_current_page_number,
                    input_folder,  # Use 1-based page number
                )
            )

            # Update state and review_df with the potentially replaced image path
            if len(out_image_annotations_state) > page_num_reported_zero_indexed:
                out_image_annotations_state[page_num_reported_zero_indexed][
                    "image"
                ] = replaced_image_path

            if "page" in review_df.columns and "image" in review_df.columns:
                review_df.loc[
                    review_df["page"] == gradio_annotator_current_page_number, "image"
                ] = replaced_image_path

        except Exception as e:
            print(
                f"Error during image path replacement for page {gradio_annotator_current_page_number}: {e}"
            )
    else:
        print(
            f"Warning: Page index {page_num_reported_zero_indexed} out of bounds for all_image_annotations list."
        )

    # Save back page_sizes_df to page_sizes list format
    if not page_sizes_df.empty:
        page_sizes = page_sizes_df.to_dict(orient="records")
    else:
        page_sizes = list()  # Ensure page_sizes is a list if df is empty

    # --- Re-evaluate Coordinate Multiplication and Duplicate Removal ---
    # Let's assume remove_duplicate_images_with_blank_boxes expects the raw list of dicts state format:
    try:
        out_image_annotations_state = remove_duplicate_images_with_blank_boxes(
            out_image_annotations_state
        )
    except Exception as e:
        print(
            f"Error during duplicate removal: {e}. Proceeding without duplicate removal."
        )

    # Select the current page's annotation object from the (potentially updated) state
    if len(out_image_annotations_state) > page_num_reported_zero_indexed:
        out_current_page_annotator = out_image_annotations_state[
            page_num_reported_zero_indexed
        ]
    else:
        print(
            f"Warning: Cannot select current page annotator object for index {page_num_reported_zero_indexed}."
        )
        out_current_page_annotator = {}  # Or None, depending on expected output type

    # Return final page number
    final_page_number_returned = gradio_annotator_current_page_number

    return (
        out_current_page_annotator,
        out_image_annotations_state,
        final_page_number_returned,
        page_sizes,
        review_df,  # review_df might have its 'page' column type changed, keep it as is or revert if necessary
        annotate_previous_page,
    )  # The original page number from selected_recogniser_entity_df_row


def _merge_horizontally_adjacent_boxes(
    df: pd.DataFrame,
    x_merge_threshold: float = 0.02,
    y_merge_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Merges horizontally adjacent bounding boxes within the same visual line.

    Only merges boxes that are on the same visual line (similar y position),
    so that merged boxes do not span multiple lines and get incorrect ymax
    (e.g. 1.0 when the OCR "line" field is shared across the page).

    Args:
        df (pd.DataFrame): DataFrame containing annotation boxes with columns
                           like 'page', 'line', 'xmin', 'xmax', 'ymin', 'ymax', etc.
        x_merge_threshold (float): The maximum gap on the x-axis (normalised 0-1)
                                   to consider two boxes as adjacent.
        y_merge_threshold (float): The maximum vertical distance (normalised 0-1)
                                   to consider two boxes on the same visual line.

    Returns:
        pd.DataFrame: A new DataFrame with adjacent boxes merged.
    """
    if df.empty:
        return df

    # 1. Sort by page, then by vertical position (ymin) then horizontal (xmin)
    #    so that we compare consecutive words on the same visual line.
    df_sorted = df.sort_values(by=["page", "line", "xmin"]).copy()

    # 2. Identify groups of boxes to merge using shift() and cumsum()
    # Get properties of the 'previous' box in the sorted list
    prev_xmax = df_sorted["xmax"].shift(1)
    prev_page = df_sorted["page"].shift(1)
    prev_line = df_sorted["line"].shift(1)

    # Same text line
    same_visual_line = (df_sorted["page"] == prev_page) & (
        df_sorted["line"] == prev_line
    )

    # A box should be merged with the previous one if it's on the same page,
    # same visual line (similar y), and the horizontal gap is within threshold.
    is_adjacent = same_visual_line & (
        df_sorted["xmin"] - prev_xmax <= x_merge_threshold
    )

    # A new group starts wherever a box is NOT adjacent to the previous one.
    # cumsum() on this boolean series creates a unique ID for each group.
    df_sorted["merge_group"] = (~is_adjacent).cumsum()

    # 3. Aggregate each group into a single bounding box
    # Define how to aggregate each column
    agg_funcs = {
        "xmin": "min",
        "ymin": "min",  # To get the highest point of the combined box
        "xmax": "max",
        "ymax": "max",  # To ensure we cover all text
        "text": lambda s: " ".join(s.astype(str)),  # Join the text
        # Carry over the first value for columns that are constant within a group
        "page": "first",
        "line": "first",
        "image": "first",
        "label": "first",
        "color": "first",
    }

    merged_df = df_sorted.groupby("merge_group").agg(agg_funcs).reset_index(drop=True)

    return merged_df


def get_and_merge_current_page_annotations(
    page_sizes: List[Dict],
    annotate_current_page: int,
    existing_annotations_list: List[Dict],
    existing_annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to extract and merge annotations for the current page
    into the main existing_annotations_df.
    """
    current_page_image = page_sizes[annotate_current_page - 1]["image_path"]

    existing_annotations_current_page = [
        item
        for item in existing_annotations_list
        if item["image"] == current_page_image
    ]

    current_page_annotations_df = convert_annotation_data_to_dataframe(
        existing_annotations_current_page
    )

    # Concatenate and clean, ensuring no duplicates and sorted order.
    # Deduplicate only by non-null id: pandas treats NaN==NaN in drop_duplicates(subset=["id"]),
    # which would collapse all rows with missing id to one and drop annotations on other pages.
    dfs_to_concat = [
        df
        for df in [existing_annotations_df, current_page_annotations_df]
        if not df.empty
    ]
    if dfs_to_concat:
        if len(dfs_to_concat) == 1:
            combined = dfs_to_concat[0].copy()
        else:
            combined = pd.concat(dfs_to_concat, ignore_index=True)
        if "id" in combined.columns:
            has_id = combined["id"].notna()
            if has_id.any():
                deduped = combined.loc[has_id].drop_duplicates(
                    subset=["id"], keep="first"
                )
                no_id = combined.loc[~has_id]
                parts = [p for p in [no_id, deduped] if not p.empty]
                if len(parts) == 1:
                    updated_df = parts[0].sort_values(by=["page", "xmin", "ymin"])
                else:
                    updated_df = pd.concat(parts, ignore_index=True).sort_values(
                        by=["page", "xmin", "ymin"]
                    )
            else:
                updated_df = combined.sort_values(by=["page", "xmin", "ymin"])
        else:
            updated_df = combined.sort_values(by=["page", "xmin", "ymin"])
    else:
        # Return empty DataFrame with expected columns from convert_annotation_data_to_dataframe
        updated_df = pd.DataFrame(
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

    # Ensure no box spans to the very bottom (ymax == 1); cap ymax to just below 1
    # so that unmerged boxes (e.g. from OCR with line shared across page) don't get ymax=1.
    if (
        not updated_df.empty
        and "ymax" in updated_df.columns
        and "ymin" in updated_df.columns
    ):
        ymax_cap = 1.0 - 1e-6
        ymax_vals = pd.to_numeric(updated_df["ymax"], errors="coerce")
        need_cap = ymax_vals >= 1.0
        if need_cap.any():
            updated_df = updated_df.copy()
            updated_df.loc[need_cap, "ymax"] = ymax_vals.loc[need_cap].clip(
                upper=ymax_cap
            )
            # Keep box valid: ymax must remain > ymin
            ymin_vals = pd.to_numeric(updated_df.loc[need_cap, "ymin"], errors="coerce")
            invalid = updated_df.loc[need_cap, "ymax"].values <= ymin_vals.values
            if invalid.any():
                idx = updated_df.index[need_cap][invalid]
                updated_df.loc[idx, "ymax"] = (
                    pd.to_numeric(updated_df.loc[idx, "ymin"], errors="coerce") + 1e-6
                )

    return updated_df


def create_annotation_objects_from_filtered_ocr_results_with_words(
    filtered_ocr_results_with_words_df: pd.DataFrame,
    ocr_results_with_words_df_base: pd.DataFrame,
    page_sizes: List[Dict],
    existing_annotations_df: pd.DataFrame,
    existing_annotations_list: List[Dict],
    existing_recogniser_entity_df: pd.DataFrame,
    redaction_label: str = "Redaction",
    colour_label: str = "(0, 0, 0)",
    annotate_current_page: int = 1,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[
    List[Dict], List[Dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    This function processes filtered OCR results with words to create new annotation objects. It merges these new annotations with existing ones, ensuring that horizontally adjacent boxes are combined for cleaner redactions. The function also updates the existing recogniser entity DataFrame and returns the updated annotations in both DataFrame and list-of-dicts formats.

    Args:
        filtered_ocr_results_with_words_df (pd.DataFrame): A DataFrame containing filtered OCR results with words.
        ocr_results_with_words_df_base (pd.DataFrame): The base DataFrame of OCR results with words.
        page_sizes (List[Dict]): A list of dictionaries containing page sizes.
        existing_annotations_df (pd.DataFrame): A DataFrame of existing annotations.
        existing_annotations_list (List[Dict]): A list of dictionaries representing existing annotations.
        existing_recogniser_entity_df (pd.DataFrame): A DataFrame of existing recogniser entities.
        progress (gr.Progress, optional): A progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        Tuple[List[Dict], List[Dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the updated annotations list, updated existing annotations list, updated annotations DataFrame, updated existing annotations DataFrame, updated recogniser entity DataFrame, and the original existing recogniser entity DataFrame.
    """

    # Validate colour_label: must be a 3-number tuple with each value in [0, 255]
    # If invalid, fallback to '(0, 0, 0)' as requested
    fallback_colour = "(0, 0, 0)"

    existing_annotations_df = get_and_merge_current_page_annotations(
        page_sizes,
        annotate_current_page,
        existing_annotations_list,
        existing_annotations_df,
    )

    try:
        valid = False
        if isinstance(colour_label, str):
            label_str = colour_label.strip()
            from tools.secure_regex_utils import safe_extract_rgb_values

            rgb_values = safe_extract_rgb_values(label_str)
            if rgb_values:
                r_val, g_val, b_val = rgb_values
                if 0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255:
                    valid = True
        elif isinstance(colour_label, (tuple, list)) and len(colour_label) == 3:
            r_val, g_val, b_val = colour_label
            if all(isinstance(v, int) for v in (r_val, g_val, b_val)) and all(
                0 <= v <= 255 for v in (r_val, g_val, b_val)
            ):
                colour_label = f"({r_val}, {g_val}, {b_val})"
                valid = True
        if not valid:
            colour_label = fallback_colour
    except Exception:
        colour_label = fallback_colour

    progress(0.2, desc="Identifying new redactions to add")
    print("Identifying new redactions to add")
    if filtered_ocr_results_with_words_df.empty:
        print("No new annotations to add.")
        updated_annotations_df = existing_annotations_df.copy()
    else:
        # Assuming index relationship holds for fast lookup
        filtered_ocr_results_with_words_df.index = filtered_ocr_results_with_words_df[
            "index"
        ]
        new_annotations_df = ocr_results_with_words_df_base.loc[
            filtered_ocr_results_with_words_df.index
        ].copy()

        if new_annotations_df.empty:
            print("No new annotations to add.")
            updated_annotations_df = existing_annotations_df.copy()
        else:
            page_to_image_map = {
                item["page"]: item["image_path"] for item in page_sizes
            }

            # Prepare the initial new annotations DataFrame
            new_annotations_df = new_annotations_df.assign(
                image=lambda df: df["page"].map(page_to_image_map),
                label=redaction_label,
                color=colour_label,
            ).rename(
                columns={
                    "word_x0": "xmin",
                    "word_y0": "ymin",
                    "word_x1": "xmax",
                    "word_y1": "ymax",
                    "word_text": "text",
                }
            )

            # Clip box to line-level bounds (all four coordinates) when available
            _eps = 1e-6
            line_cols = ["line_x0", "line_x1", "line_y0", "line_y1"]
            has_line = all(c in new_annotations_df.columns for c in line_cols)
            if has_line:
                ymax_fallback = 1.0 - _eps
                lx0 = pd.to_numeric(new_annotations_df["line_x0"], errors="coerce")
                lx1 = pd.to_numeric(new_annotations_df["line_x1"], errors="coerce")
                ly0 = pd.to_numeric(new_annotations_df["line_y0"], errors="coerce")
                ly1 = pd.to_numeric(new_annotations_df["line_y1"], errors="coerce")
                valid = (
                    lx0.notna()
                    & lx1.notna()
                    & ly0.notna()
                    & ly1.notna()
                    & (lx0 >= 0)
                    & (lx1 <= 1)
                    & (ly0 >= 0)
                    & (ly1 <= 1)
                    & (lx0 < lx1)
                    & (ly0 < ly1)
                )
                if valid.any():
                    new_annotations_df = new_annotations_df.copy()
                    ly1_safe = ly1.where(ly1 < 1).fillna(ymax_fallback)
                    new_annotations_df.loc[valid, "xmin"] = pd.to_numeric(
                        new_annotations_df.loc[valid, "xmin"], errors="coerce"
                    ).clip(lower=lx0.loc[valid])
                    new_annotations_df.loc[valid, "xmax"] = pd.to_numeric(
                        new_annotations_df.loc[valid, "xmax"], errors="coerce"
                    ).clip(upper=lx1.loc[valid])
                    new_annotations_df.loc[valid, "ymin"] = pd.to_numeric(
                        new_annotations_df.loc[valid, "ymin"], errors="coerce"
                    ).clip(lower=ly0.loc[valid])
                    new_annotations_df.loc[valid, "ymax"] = pd.to_numeric(
                        new_annotations_df.loc[valid, "ymax"], errors="coerce"
                    ).clip(upper=ly1_safe.loc[valid])
                    # Ensure valid box
                    xinv = (
                        new_annotations_df.loc[valid, "xmin"]
                        >= new_annotations_df.loc[valid, "xmax"]
                    )
                    yinv = (
                        new_annotations_df.loc[valid, "ymin"]
                        >= new_annotations_df.loc[valid, "ymax"]
                    )
                    if xinv.any():
                        idx = new_annotations_df.index[valid][xinv]
                        mid = (
                            pd.to_numeric(
                                new_annotations_df.loc[idx, "xmin"], errors="coerce"
                            )
                            + pd.to_numeric(
                                new_annotations_df.loc[idx, "xmax"], errors="coerce"
                            )
                        ) / 2
                        new_annotations_df.loc[idx, "xmin"] = (mid - _eps).clip(0, 1)
                        new_annotations_df.loc[idx, "xmax"] = (mid + _eps).clip(0, 1)
                    if yinv.any():
                        idx = new_annotations_df.index[valid][yinv]
                        mid = (
                            pd.to_numeric(
                                new_annotations_df.loc[idx, "ymin"], errors="coerce"
                            )
                            + pd.to_numeric(
                                new_annotations_df.loc[idx, "ymax"], errors="coerce"
                            )
                        ) / 2
                        new_annotations_df.loc[idx, "ymin"] = (mid - _eps).clip(0, 1)
                        new_annotations_df.loc[idx, "ymax"] = (mid + _eps).clip(0, 1)
            else:
                # No line bounds: cap ymax only so no box spans to bottom
                ymax_vals = pd.to_numeric(new_annotations_df["ymax"], errors="coerce")
                need_cap = ymax_vals >= 1.0
                if need_cap.any():
                    new_annotations_df = new_annotations_df.copy()
                    new_annotations_df.loc[need_cap, "ymax"] = ymax_vals.loc[
                        need_cap
                    ].clip(upper=1.0 - _eps)
                    ymin_vals = pd.to_numeric(
                        new_annotations_df.loc[need_cap, "ymin"], errors="coerce"
                    )
                    invalid = (
                        new_annotations_df.loc[need_cap, "ymax"].values
                        <= ymin_vals.values
                    )
                    if invalid.any():
                        idx = new_annotations_df.index[need_cap][invalid]
                        new_annotations_df.loc[idx, "ymax"] = (
                            pd.to_numeric(
                                new_annotations_df.loc[idx, "ymin"], errors="coerce"
                            )
                            + _eps
                        )

            progress(0.3, desc="Checking for adjacent annotations to merge...")
            new_annotations_df = _merge_horizontally_adjacent_boxes(new_annotations_df)

            progress(0.4, desc="Creating new redaction IDs...")
            existing_ids = (
                set(existing_annotations_df["id"].dropna())
                if "id" in existing_annotations_df.columns
                else set()
            )
            num_new_ids = len(new_annotations_df)
            new_id_list = _generate_unique_ids(num_new_ids, existing_ids)
            new_annotations_df["id"] = new_id_list

            annotation_cols = [
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
            ]
            new_annotations_df = new_annotations_df[annotation_cols]

            key_cols = ["page", "label", "xmin", "ymin", "xmax", "ymax", "text"]

            progress(0.5, desc="Checking for duplicate redactions")

            if existing_annotations_df.empty or not all(
                col in existing_annotations_df.columns for col in key_cols
            ):
                unique_new_df = new_annotations_df
            else:
                # Ensure that columns of both sides have the same type
                new_annotations_df.loc[:, key_cols] = new_annotations_df.loc[
                    :, key_cols
                ].astype(existing_annotations_df.loc[:, key_cols].dtypes)

                # Do not add duplicate redactions
                merged = pd.merge(
                    new_annotations_df,
                    existing_annotations_df[key_cols].drop_duplicates(),
                    on=key_cols,
                    how="left",
                    indicator=True,
                )
                unique_new_df = merged[merged["_merge"] == "left_only"].drop(
                    columns=["_merge"]
                )

            print(f"Found {len(unique_new_df)} new unique annotations to add.")
            gr.Info(f"Found {len(unique_new_df)} new unique annotations to add.")
            # Filter out empty DataFrames before concatenation to avoid FutureWarning
            dfs_to_concat = [
                df for df in [existing_annotations_df, unique_new_df] if not df.empty
            ]
            if dfs_to_concat:
                updated_annotations_df = pd.concat(dfs_to_concat, ignore_index=True)
            else:
                # Return empty DataFrame with expected columns matching existing_annotations_df structure
                updated_annotations_df = pd.DataFrame(
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

    # --- Part 4: Convert final DataFrame to list-of-dicts ---
    updated_recogniser_entity_df = pd.DataFrame()
    if not updated_annotations_df.empty:
        updated_recogniser_entity_df = updated_annotations_df[
            ["page", "label", "text", "id"]
        ]

    if not page_sizes:
        print("Warning: page_sizes is empty. No pages to process.")
        return (
            [],
            existing_annotations_list,
            pd.DataFrame(),
            existing_annotations_df,
            pd.DataFrame(),
            existing_recogniser_entity_df,
        )

    # Always derive image paths from page using current page_sizes, so that
    # updated_annotations_df never has None/missing image when page is valid
    # (e.g. after copy from existing_annotations_df or concat with unique_new_df).

    all_pages_df = pd.DataFrame(page_sizes).rename(columns={"image_path": "image"})

    # Join image paths to updated_annotations_df based on page number
    # Drop image column from updated_annotations_df
    updated_annotations_df = updated_annotations_df.drop(columns=["image"])

    # set page to number
    updated_annotations_df["page"] = updated_annotations_df["page"].astype(int)
    all_pages_df["page"] = all_pages_df["page"].astype(int)

    updated_annotations_df = pd.merge(
        updated_annotations_df, all_pages_df[["page", "image"]], on="page", how="left"
    )

    if not updated_annotations_df.empty and "page" in updated_annotations_df.columns:
        missing_image = updated_annotations_df["image"].isna()
        if missing_image.any():
            n_missing = missing_image.sum()
            print(
                f"Warning: {n_missing} annotation(s) have page not in page_sizes; "
                "they will not appear in output. Dropping them from updated_annotations_df."
            )
            updated_annotations_df = updated_annotations_df.loc[~missing_image].copy()
        # Keep recogniser entity in sync with possibly trimmed annotations
        if not updated_annotations_df.empty:
            updated_recogniser_entity_df = updated_annotations_df[
                ["page", "label", "text", "id"]
            ]
        else:
            updated_recogniser_entity_df = pd.DataFrame()

    if not updated_annotations_df.empty:
        merged_df = pd.merge(
            all_pages_df[["image"]], updated_annotations_df, on="image", how="left"
        )
    else:
        merged_df = all_pages_df[["image"]]

    # 1. Get the list of image paths in the exact order they appear in page_sizes.
    #    all_pages_df was created from page_sizes, so it preserves this order.
    image_order = all_pages_df["image"].tolist()

    # 2. Convert the 'image' column to a special 'Categorical' type.
    #    This tells pandas that this column has a custom, non-alphabetical order.
    merged_df["image"] = pd.Categorical(
        merged_df["image"], categories=image_order, ordered=True
    )

    # 3. Sort the DataFrame based on this new custom order.
    merged_df = merged_df.sort_values("image")

    final_annotations_list = list()
    box_cols = ["label", "color", "xmin", "ymin", "xmax", "ymax", "text", "id"]

    # Process each (image_path, group) in parallel; preserve order via index.
    group_items = [
        (i, image_path, group)
        for i, (image_path, group) in enumerate(
            merged_df.groupby("image", sort=False, observed=False)
        )
    ]

    def _process_one_group(item):
        _i, _image_path, _group = item
        if pd.isna(_group.iloc[0].get("id")):
            _boxes = list()
        else:
            _valid_box_cols = [col for col in box_cols if col in _group.columns]
            _sorted_group = _group.sort_values(by=["ymin", "xmin"]).copy()
            # Ensure coord columns have no NaN so image_annotator preprocess_boxes doesn't raise TypeError
            for coord in ["xmin", "ymin", "xmax", "ymax"]:
                if coord in _sorted_group.columns:
                    _sorted_group[coord] = pd.to_numeric(
                        _sorted_group[coord], errors="coerce"
                    ).fillna(0.0)
            _boxes = _sorted_group[_valid_box_cols].to_dict("records")
        return (_i, {"image": _image_path, "boxes": _boxes})

    if group_items:
        n_groups = len(group_items)
        max_workers = min(MAX_WORKERS, n_groups)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ordered_results = sorted(
                executor.map(_process_one_group, group_items), key=lambda x: x[0]
            )
        final_annotations_list = [r[1] for r in ordered_results]

    progress(1.0, desc="Completed annotation processing")

    return (
        final_annotations_list,
        existing_annotations_list,
        updated_annotations_df,
        existing_annotations_df,
        updated_recogniser_entity_df,
        existing_recogniser_entity_df,
    )


def exclude_selected_items_from_redaction(
    review_df: pd.DataFrame,
    selected_rows_df: pd.DataFrame,
    image_file_paths: List[str],
    page_sizes: List[dict],
    image_annotations_state: dict,
    recogniser_entity_dataframe_base: pd.DataFrame,
):
    """
    Remove selected items from the review dataframe from the annotation object and review dataframe.
    """

    backup_review_state = review_df
    backup_image_annotations_state = image_annotations_state
    backup_recogniser_entity_dataframe_base = recogniser_entity_dataframe_base

    if not selected_rows_df.empty and not review_df.empty:
        use_id = (
            "id" in selected_rows_df.columns
            and "id" in review_df.columns
            and not selected_rows_df["id"].isnull().all()
            and not review_df["id"].isnull().all()
        )

        selected_merge_cols = ["id"] if use_id else ["label", "page", "text"]

        # Subset and drop duplicates from selected_rows_df
        selected_subset = selected_rows_df[selected_merge_cols].drop_duplicates(
            subset=selected_merge_cols
        )

        # Perform anti-join using merge with indicator
        merged_df = review_df.merge(
            selected_subset, on=selected_merge_cols, how="left", indicator=True
        )
        out_review_df = merged_df[merged_df["_merge"] == "left_only"].drop(
            columns=["_merge"]
        )

        out_image_annotations_state = convert_review_df_to_annotation_json(
            out_review_df, image_file_paths, page_sizes
        )

        out_recogniser_entity_dataframe_base = out_review_df[
            ["page", "label", "text", "id"]
        ]

    # Either there is nothing left in the selection dataframe, or the review dataframe
    else:
        out_review_df = review_df
        out_recogniser_entity_dataframe_base = recogniser_entity_dataframe_base
        out_image_annotations_state = image_annotations_state

    return (
        out_review_df,
        out_image_annotations_state,
        out_recogniser_entity_dataframe_base,
        backup_review_state,
        backup_image_annotations_state,
        backup_recogniser_entity_dataframe_base,
    )


def replace_annotator_object_img_np_array_with_page_sizes_image_path(
    all_image_annotations: List[dict],
    page_image_annotator_object: AnnotatedImageData,
    page_sizes: List[dict],
    page: int,
    page_sizes_df: pd.DataFrame = None,
):
    """
    Check if the image value in an AnnotatedImageData dict is a placeholder or np.array. If either of these, replace the value with the file path of the image that is hopefully already loaded into the app related to this page.
    """
    page_zero_index = page - 1

    if (
        isinstance(all_image_annotations[page_zero_index]["image"], np.ndarray)
        or "placeholder_image"
        in str(all_image_annotations[page_zero_index].get("image", ""))
        or isinstance(page_image_annotator_object.get("image"), np.ndarray)
    ):
        if page_sizes_df is None or page_sizes_df.empty:
            page_sizes_df = pd.DataFrame(page_sizes)
            page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(
                pd.to_numeric, errors="coerce"
            )

        # Check for matching pages (single .loc)
        matching_paths = page_sizes_df.loc[
            page_sizes_df["page"] == page, "image_path"
        ].unique()

        if matching_paths.size > 0:
            image_path = matching_paths[0]
            page_image_annotator_object["image"] = image_path
            all_image_annotations[page_zero_index]["image"] = image_path
        else:
            print(f"No image path found for page {page}.")

    return page_image_annotator_object, all_image_annotations


def replace_placeholder_image_with_real_image(
    doc_full_file_name_textbox: str,
    current_image_path: str,
    page_sizes_df: pd.DataFrame,
    page_num_reported: int,
    input_folder: str,
):
    """If image path is still not valid, load in a new image an overwrite it. Then replace all items in the image annotation object for all pages based on the updated information."""
    if page_num_reported <= 0:
        page_num_reported = 1

    page_num_reported_zero_indexed = page_num_reported - 1

    # Compute mask once to avoid repeated boolean indexing over the full DataFrame
    if "page" not in page_sizes_df.columns:
        page_mask = pd.Series(False, index=page_sizes_df.index)
    else:
        page_col = pd.to_numeric(page_sizes_df["page"], errors="coerce")
        page_mask = page_col == page_num_reported

    if not os.path.exists(current_image_path):

        page_num, replaced_image_path, width, height = (
            process_single_page_for_image_conversion(
                doc_full_file_name_textbox,
                page_num_reported_zero_indexed,
                input_folder=input_folder,
            )
        )

        page_sizes_df.loc[page_mask, "image_width"] = width
        page_sizes_df.loc[page_mask, "image_height"] = height
        page_sizes_df.loc[page_mask, "image_path"] = replaced_image_path

    else:
        if page_mask.any():
            width_vals = page_sizes_df.loc[page_mask, "image_width"]
            if not width_vals.isnull().all():
                width = width_vals.max()
                height = page_sizes_df.loc[page_mask, "image_height"].max()
            else:
                image = Image.open(current_image_path)
                width = image.width
                height = image.height
                page_sizes_df.loc[page_mask, "image_width"] = width
                page_sizes_df.loc[page_mask, "image_height"] = height
        else:
            width = height = None

        page_sizes_df.loc[page_mask, "image_path"] = current_image_path
        replaced_image_path = current_image_path

    return replaced_image_path, page_sizes_df


def update_annotator_object_and_filter_df(
    all_image_annotations: List[AnnotatedImageData],
    gradio_annotator_current_page_number: int,
    recogniser_entities_dropdown_value: str = "ALL",
    page_dropdown_value: str = "ALL",
    page_dropdown_redaction_value: str = "1",
    text_dropdown_value: str = "ALL",
    recogniser_dataframe_base: pd.DataFrame = None,  # Simplified default
    zoom: int = 100,
    review_df: pd.DataFrame = None,  # Use None for default empty DataFrame
    page_sizes: List[dict] = list(),
    doc_full_file_name_textbox: str = "",
    input_folder: str = INPUT_FOLDER,
) -> Tuple[
    AnnotatedImageData,
    int,
    int,
    int,
    str,
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    List[str],
    List[dict],
    List[AnnotatedImageData],
]:
    """
    Update a gradio_image_annotation object with new annotation data for the current page
    and update filter dataframes, optimizing by processing only the current page's data for display.

    Args:
        all_image_annotations (List[AnnotatedImageData]): All image annotation objects to process.
        gradio_annotator_current_page_number (int): The current page number as selected in the annotator.
        recogniser_entities_dropdown_value (str, optional): Value for the recogniser dropdown filter. Defaults to "ALL".
        page_dropdown_value (str, optional): Value for the page dropdown filter. Defaults to "ALL".
        page_dropdown_redaction_value (str, optional): Value for the redaction page dropdown filter. Defaults to "1".
        text_dropdown_value (str, optional): Value for the text dropdown filter. Defaults to "ALL".
        recogniser_dataframe_base (pd.DataFrame, optional): The base recogniser dataframe. Defaults to None.
        zoom (int, optional): Zoom level for display in the annotator. Defaults to 100.
        review_df (pd.DataFrame, optional): Review DataFrame containing annotation boxes. Defaults to None.
        page_sizes (List[dict], optional): List of dictionaries containing page size information. Defaults to empty list.
        doc_full_file_name_textbox (str, optional): Full file name shown in the textbox. Defaults to empty string.
        input_folder (str, optional): Path to the input folder. Defaults to INPUT_FOLDER.

    Returns:
        Tuple[
            image_annotator,
            int,
            int,
            int,
            str,
            pd.DataFrame,
            pd.DataFrame,
            List[str],
            List[str],
            List[dict],
            List[AnnotatedImageData],
        ]: Updated Gradio components and relevant page annotations.
    """

    str(zoom) + "%"

    # Handle default empty review_df and recogniser_dataframe_base
    if review_df is None or not isinstance(review_df, pd.DataFrame):
        review_df = pd.DataFrame(
            columns=[
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
            ]
        )
    if recogniser_dataframe_base is None:  # Create a simple default if None
        recogniser_dataframe_base = pd.DataFrame(
            pd.DataFrame(columns=["page", "label", "text", "id"])
        )

    # Handle empty all_image_annotations state early
    if not all_image_annotations:
        print("No all_image_annotation object found")
        # Return blank/default outputs

        blank_annotator = None
        blank_df_out_gr = pd.DataFrame(columns=["page", "label", "text", "id"])
        blank_df_modified = pd.DataFrame(columns=["page", "label", "text", "id"])

        return (
            blank_annotator,
            1,
            1,
            1,
            recogniser_entities_dropdown_value,
            blank_df_out_gr,
            blank_df_modified,
            [],
            [],
            [],
            [],
            [],
        )  # Return empty lists/defaults for other outputs

    # Validate and bound the current page number (1-based logic)
    page_num_reported = max(
        1, gradio_annotator_current_page_number
    )  # Minimum page is 1
    page_max_reported = len(all_image_annotations)
    if page_num_reported > page_max_reported:
        page_num_reported = page_max_reported

    page_num_reported_zero_indexed = page_num_reported - 1

    # --- Process page sizes DataFrame ---
    page_sizes_df = pd.DataFrame(page_sizes)
    if not page_sizes_df.empty:
        page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
        page_sizes_df.dropna(subset=["page"], inplace=True)
        if not page_sizes_df.empty:
            page_sizes_df["page"] = page_sizes_df["page"].astype(int)
        else:
            print("Warning: Page sizes DataFrame became empty after processing.")

    # --- Handle Image Path Replacement for the Current Page ---
    if len(all_image_annotations) > page_num_reported_zero_indexed:

        page_object_to_update = all_image_annotations[page_num_reported_zero_indexed]

        # Use the helper function to replace the image path within the page object
        updated_page_object, all_image_annotations_after_img_replace = (
            replace_annotator_object_img_np_array_with_page_sizes_image_path(
                all_image_annotations,
                page_object_to_update,
                page_sizes,
                page_num_reported,
                page_sizes_df=page_sizes_df,
            )
        )

        all_image_annotations = all_image_annotations_after_img_replace

        # Now handle the actual image file path replacement using replace_placeholder_image_with_real_image
        current_image_path = updated_page_object.get(
            "image"
        )  # Get potentially updated image path

        if current_image_path and not page_sizes_df.empty:
            try:
                replaced_image_path, page_sizes_df = (
                    replace_placeholder_image_with_real_image(
                        doc_full_file_name_textbox,
                        current_image_path,
                        page_sizes_df,
                        page_num_reported,
                        input_folder=input_folder,  # Use 1-based page num
                    )
                )

                # Update the image path in the state and review_df for the current page
                # Find the correct entry in all_image_annotations list again by index
                if len(all_image_annotations) > page_num_reported_zero_indexed:
                    all_image_annotations[page_num_reported_zero_indexed][
                        "image"
                    ] = replaced_image_path

                # Update review_df's image path for this page
                if "page" in review_df.columns and "image" in review_df.columns:
                    if not pd.api.types.is_numeric_dtype(review_df["page"]):
                        review_df["page"] = (
                            pd.to_numeric(review_df["page"], errors="coerce")
                            .fillna(-1)
                            .astype(int)
                        )
                    review_df.loc[review_df["page"] == page_num_reported, "image"] = (
                        replaced_image_path
                    )

            except Exception as e:
                print(
                    f"Error during image path replacement for page {page_num_reported}: {e}"
                )
    else:
        print(
            f"Warning: Page index {page_num_reported_zero_indexed} out of bounds for all_image_annotations list."
        )

    # Save back page_sizes_df to page_sizes list format
    if not page_sizes_df.empty:
        page_sizes = page_sizes_df.to_dict(orient="records")
    else:
        page_sizes = list()  # Ensure page_sizes is a list if df is empty

    # --- Prepare data *only* for the current page for display ---
    current_page_image_annotator_object = None
    if len(all_image_annotations) > page_num_reported_zero_indexed:
        page_data_for_display = all_image_annotations[page_num_reported_zero_indexed]

        # Convert current page annotations list to DataFrame for coordinate multiplication IF needed
        # Assuming coordinate multiplication IS needed for display if state stores relative coords
        current_page_annotations_df = convert_annotation_data_to_dataframe(
            [page_data_for_display]
        )

        if not current_page_annotations_df.empty and not page_sizes_df.empty:
            # Multiply coordinates *only* for this page's DataFrame (reuse single filter)
            try:
                page_size_row = page_sizes_df[
                    page_sizes_df["page"] == page_num_reported
                ]
                if not page_size_row.empty:
                    current_page_annotations_df = multiply_coordinates_by_page_sizes(
                        current_page_annotations_df,
                        page_size_row,
                        xmin="xmin",
                        xmax="xmax",
                        ymin="ymin",
                        ymax="ymax",
                    )
            except Exception as e:
                print(
                    f"Warning: Error during coordinate multiplication for page {page_num_reported}: {e}. Using original coordinates."
                )
                # If error, proceed with original coordinates or handle as needed

        if "color" not in current_page_annotations_df.columns:
            current_page_annotations_df["color"] = CUSTOM_BOX_COLOUR
        # gradio_image_annotation JS expects colour as string (e.g. .startsWith("rgba"))
        current_page_annotations_df["color"] = current_page_annotations_df[
            "color"
        ].apply(_ensure_box_colour_string)

        # Ensure coord columns have no NaN/None so image_annotator preprocess_boxes doesn't raise TypeError
        coord_cols = ["xmin", "xmax", "ymin", "ymax"]
        for col in coord_cols:
            if col in current_page_annotations_df.columns:
                current_page_annotations_df[col] = pd.to_numeric(
                    current_page_annotations_df[col], errors="coerce"
                ).fillna(0.0)

        # Convert the processed DataFrame back to the list of dicts format for the annotator
        processed_current_page_annotations_list = current_page_annotations_df[
            ["xmin", "xmax", "ymin", "ymax", "label", "color", "text", "id"]
        ].to_dict(orient="records")

        # Construct the final object expected by the Gradio ImageAnnotator value parameter
        current_page_image_annotator_object: AnnotatedImageData = {
            "image": page_data_for_display.get(
                "image"
            ),  # Use the (potentially updated) image path
            "boxes": processed_current_page_annotations_list,
        }

    # --- Update Dropdowns and Review DataFrame ---
    try:
        (
            recogniser_entities_list,
            recogniser_dataframe_out_gr,
            recogniser_dataframe_modified,
            recogniser_entities_dropdown_value,
            text_entities_drop,
            page_entities_drop,
        ) = update_recogniser_dataframes(
            all_image_annotations,  # Pass the updated full state
            recogniser_dataframe_base,
            recogniser_entities_dropdown_value,
            text_dropdown_value,
            page_dropdown_value,
            review_df.copy(),  # Keep the copy as per original function call
            page_sizes,  # Pass updated page sizes
        )
        # Generate default colors for labels (library expects hex string or RGB tuple; tuples are converted to hex)
        [
            CUSTOM_BOX_COLOUR for _ in range(len(recogniser_entities_list))
        ]

    except Exception as e:
        print(
            f"Error calling update_recogniser_dataframes: {e}. Returning empty/default filter data."
        )
        recogniser_entities_list = list()
        recogniser_dataframe_out_gr = pd.DataFrame(
            columns=["page", "label", "text", "id"]
        )
        recogniser_dataframe_modified = pd.DataFrame(
            columns=["page", "label", "text", "id"]
        )
        text_entities_drop = list()
        page_entities_drop = list()

    # --- Final Output Components ---
    page_number_update = (
        gr.update(value=page_num_reported, maximum=len(page_sizes)) if page_sizes else 0
    )

    ### Present image_annotator outputs
    # Handle the case where current_page_image_annotator_object couldn't be prepared
    if current_page_image_annotator_object is None:
        # This should ideally be covered by the initial empty check for all_image_annotations,
        # but as a safeguard:
        print("Warning: Could not prepare annotator object for the current page.")
        out_image_annotator = None
    else:
        if current_page_image_annotator_object["image"].startswith("placeholder_image"):
            current_page_image_annotator_object["image"], page_sizes_df = (
                replace_placeholder_image_with_real_image(
                    doc_full_file_name_textbox,
                    current_page_image_annotator_object["image"],
                    page_sizes_df,
                    gradio_annotator_current_page_number,
                    input_folder,
                )
            )

        out_image_annotator = current_page_image_annotator_object

    page_entities_drop_redaction_list = ["ALL"]
    all_pages_in_doc_list = [str(i) for i in range(1, len(page_sizes) + 1)]
    page_entities_drop_redaction_list.extend(all_pages_in_doc_list)

    return (
        out_image_annotator,
        page_number_update,
        page_number_update,  # Redundant, but matches original return signature
        page_num_reported,  # Plain integer value
        recogniser_entities_dropdown_value,
        recogniser_dataframe_out_gr,
        recogniser_dataframe_modified,
        text_entities_drop,  # List of text entities for dropdown
        page_entities_drop,  # List of page numbers for dropdown
        gr.update(
            value=page_dropdown_redaction_value,
            choices=page_entities_drop_redaction_list,
            allow_custom_value=True,
            interactive=True,
        ),
        page_sizes,  # Updated page_sizes list
        all_image_annotations,
    )  # Return the updated full state


def update_all_page_annotation_object_based_on_previous_page(
    page_image_annotator_object: AnnotatedImageData,
    current_page: int,
    previous_page: int,
    all_image_annotations: List[AnnotatedImageData],
    page_sizes: List[dict] = list(),
    clear_all: bool = False,
):
    """
    Overwrite image annotations on the page we are moving from with modifications.

    Converts annotator output coordinates to relative (0-1) before storing, so that
    manually added boxes (which the annotator returns in display/canvas pixel space)
    are stored consistently with existing boxes. Without this, new boxes would be
    misplaced on the next display (shifted and scaled incorrectly).
    """

    if current_page > len(page_sizes):
        raise Warning("Selected page is higher than last page number")
    elif current_page <= 0:
        raise Warning("Selected page is lower than first page")

    previous_page_zero_index = previous_page - 1

    if not current_page:
        current_page = 1

    # This replaces the numpy array image object with the image file path
    page_image_annotator_object, all_image_annotations = (
        replace_annotator_object_img_np_array_with_page_sizes_image_path(
            all_image_annotations,
            page_image_annotator_object,
            page_sizes,
            previous_page,
        )
    )

    if clear_all is False:
        all_image_annotations[previous_page_zero_index] = page_image_annotator_object
    else:
        all_image_annotations[previous_page_zero_index]["boxes"] = list()

    return all_image_annotations, current_page, current_page


def _load_one_page_image_for_redact(
    i: int,
    all_image_annotations: List[AnnotatedImageData],
    page_to_image_path: Dict[int, str],
    input_folder: str,
    file_name_with_ext: str,
) -> Tuple[int, object, bool]:
    """
    Load (and optionally save) the image for page i. Safe to run in a thread.
    Returns (page_index, image, should_close). Caller must close image if should_close.
    """
    image_loc = all_image_annotations[i]["image"]
    should_close = False
    image = None
    if isinstance(image_loc, np.ndarray):
        image = Image.fromarray(image_loc.astype("uint8"))
        should_close = True
    elif isinstance(image_loc, Image.Image):
        image = image_loc
    elif isinstance(image_loc, str):
        path = image_loc
        if not os.path.exists(path):
            path = page_to_image_path.get(i + 1, path)
        try:
            image = Image.open(path)
            should_close = True
        except Exception:
            image = None
    if image is not None and hasattr(image, "save"):
        expected_path = os.path.join(input_folder, f"{file_name_with_ext}_{i}.png")
        if not os.path.exists(expected_path):
            try:
                image.save(expected_path)
            except Exception:
                pass
    return (i, image, should_close)


def apply_redactions_to_review_df_and_files(
    page_image_annotator_object: AnnotatedImageData,
    file_paths: List[str],
    doc: Document,
    all_image_annotations: List[AnnotatedImageData],
    current_page: int,
    review_file_state: pd.DataFrame,
    output_folder: str = OUTPUT_FOLDER,
    save_pdf: bool = True,
    page_sizes: List[dict] = list(),
    COMPRESS_REDACTED_PDF: bool = COMPRESS_REDACTED_PDF,
    input_folder: str = INPUT_FOLDER,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Applies the modified redaction annotations from the UI to the PyMuPDF document
    and exports the updated review files, including the redacted PDF and associated logs.

    Args:
        page_image_annotator_object (AnnotatedImageData): The annotation data for the current page,
                                                          potentially including user modifications.
        file_paths (List[str]): A list of file paths associated with the document, typically
                                including the original PDF and any generated image paths.
        doc (Document): The PyMuPDF Document object representing the PDF file.
        all_image_annotations (List[AnnotatedImageData]): A list containing annotation data
                                                          for all pages of the document.
        current_page (int): The 1-based index of the page currently being processed or viewed.
        review_file_state (pd.DataFrame): A Pandas DataFrame holding the current state of
                                          redaction reviews, reflecting user selections.
        output_folder (str, optional): The directory where output files (redacted PDFs,
                                       log files) will be saved. Defaults to OUTPUT_FOLDER.
        save_pdf (bool, optional): If True, the redacted PDF will be saved. Defaults to True.
        page_sizes (List[dict], optional): A list of dictionaries, each containing size
                                           information (e.g., width, height) for a page.
                                           Defaults to an empty list.
        COMPRESS_REDACTED_PDF (bool, optional): If True, the output PDF will be compressed.
                                                Defaults to COMPRESS_REDACTED_PDF.
        input_folder (str, optional): The directory where input files are located and where
                                     page images should be saved. Defaults to INPUT_FOLDER.
        progress (gr.Progress, optional): Gradio progress object for tracking task progress.
                                          Defaults to gr.Progress(track_tqdm=True).

    Returns:
        Tuple[Document, List[AnnotatedImageData], List[str], List[str], pd.DataFrame]:
            - doc: The updated PyMuPDF Document object (potentially redacted).
            - all_image_annotations: The updated list of all image annotations.
            - output_files: A list of paths to the generated output files (e.g., redacted PDF).
            - output_log_files: A list of paths to any generated log files.
            - review_df: The final Pandas DataFrame representing the review state.
    """

    output_files = list()
    output_log_files = list()
    review_df = review_file_state

    # Always use the provided input_folder parameter
    # This ensures images are created in the specified input folder, not in example_data

    page_image_annotator_object = all_image_annotations[current_page - 1]

    # This replaces the numpy array image object with the image file path
    page_image_annotator_object, all_image_annotations = (
        replace_annotator_object_img_np_array_with_page_sizes_image_path(
            all_image_annotations, page_image_annotator_object, page_sizes, current_page
        )
    )
    page_image_annotator_object["image"] = all_image_annotations[current_page - 1][
        "image"
    ]

    if not page_image_annotator_object:
        print("No image annotations object found for page")
        return doc, all_image_annotations, output_files, output_log_files, review_df

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Remove empty/blank entries that give meaningless file_extension = ""
    file_paths = [fp for fp in file_paths if fp and fp.strip()]

    # If file_paths is still empty, try to recover the source path from the
    # PyMuPDF Document that was passed in (pdf_doc_state). This handles the
    # common case where doc_full_file_name_textbox is blank because the Review
    # tab was populated programmatically (not via a user upload).
    if not file_paths and hasattr(doc, "name") and doc.name:
        recovered = doc.name
        if os.path.isfile(recovered):
            print(
                f"file_paths was empty; recovering source path from doc.name: {recovered}"
            )
            file_paths = [recovered]

    if not file_paths:
        print("No valid file paths found. Cannot apply redactions.")
        return doc, all_image_annotations, output_files, output_log_files, review_df

    def _run_apply_redactions_loop(file_paths_to_process):
        _out_files = []
        _out_log_files = []
        _review_df = review_file_state
        for file_path in file_paths_to_process:
            pdf_doc = None
            review_pdf_doc = None
            number_of_pages = 0
            _tmp_pdf_path = None
            _profile_page_times = []
            _profile_image_times = []
            file_name_without_ext = get_file_name_without_type(file_path)
        file_name_with_ext = os.path.basename(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()

        # If the UI passed only a review CSV (e.g. after duplicate-pages flow),
        # resolve the corresponding PDF so we can save the redacted output.
        if (
            save_pdf is True
            and file_extension == ".csv"
            and "_review_file" in (file_name_without_ext or "")
        ):
            pdf_basename = file_name_with_ext.replace("_review_file.csv", "")
            review_dir = os.path.dirname(file_path)
            if not review_dir:
                review_dir = output_folder or "."
            candidates = [
                os.path.join(review_dir, pdf_basename),
            ]
            if output_folder:
                candidates.append(
                    (output_folder + pdf_basename)
                    if output_folder.endswith(("/", os.sep))
                    else os.path.join(output_folder, pdf_basename)
                )
            if input_folder:
                candidates.append(
                    (input_folder + pdf_basename)
                    if input_folder.endswith(("/", os.sep))
                    else os.path.join(input_folder, pdf_basename)
                )
            for candidate in candidates:
                if candidate and os.path.isfile(candidate):
                    file_path = candidate
                    file_name_without_ext = get_file_name_without_type(file_path)
                    file_name_with_ext = os.path.basename(file_path)
                    file_extension = os.path.splitext(file_path)[1].lower()
                    break

        # Build page_sizes_df and lookups once per file (reused for PDF redaction and review CSV)
        _t0_page_sizes = time.perf_counter() if PROFILE_REDACTION_APPLY else None
        page_sizes_df = pd.DataFrame(page_sizes) if page_sizes else pd.DataFrame()
        page_to_image_path = {}
        page_to_image_dimensions = {}
        if not page_sizes_df.empty:
            if "page" in page_sizes_df.columns:
                page_sizes_df = page_sizes_df.copy()
                page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(
                    pd.to_numeric, errors="coerce"
                )
            if "image_width" in page_sizes_df.columns:
                page_sizes_df[["image_width"]] = page_sizes_df[["image_width"]].apply(
                    pd.to_numeric, errors="coerce"
                )
            if "image_height" in page_sizes_df.columns:
                page_sizes_df[["image_height"]] = page_sizes_df[["image_height"]].apply(
                    pd.to_numeric, errors="coerce"
                )
            if (
                "image_path" in page_sizes_df.columns
                and "page" in page_sizes_df.columns
            ):
                sub = page_sizes_df[["page", "image_path"]].drop_duplicates("page")
                for p, path in zip(sub["page"], sub["image_path"]):
                    if pd.notna(p):
                        page_to_image_path[int(p)] = path
            if (
                "page" in page_sizes_df.columns
                and "image_width" in page_sizes_df.columns
                and "image_height" in page_sizes_df.columns
            ):
                sub = page_sizes_df[
                    ["page", "image_width", "image_height"]
                ].drop_duplicates("page")
                for _, row in sub.iterrows():
                    p = row["page"]
                    if pd.notna(p):
                        w, h = row["image_width"], row["image_height"]
                        if pd.notna(w) and pd.notna(h):
                            page_to_image_dimensions[int(p)] = {
                                "image_width": float(w),
                                "image_height": float(h),
                            }
        if PROFILE_REDACTION_APPLY:
            _t_page_sizes = time.perf_counter() - _t0_page_sizes
        else:
            _t_page_sizes = 0.0

        if save_pdf is True:
            # If working with image docs
            if (is_pdf(file_path) is False) & (file_extension != ".csv"):
                image = Image.open(file_path)

                draw = ImageDraw.Draw(image)

                output_image_path = (
                    output_folder + file_name_without_ext + "_redacted.png"
                )
                for img_annotation_box in page_image_annotator_object["boxes"]:
                    coords = [
                        img_annotation_box["xmin"],
                        img_annotation_box["ymin"],
                        img_annotation_box["xmax"],
                        img_annotation_box["ymax"],
                    ]

                    fill = img_annotation_box["color"]

                    # Parse color: may be (r,g,b) tuple/list or string like "(128, 128, 128)" / "[128 128 128]"
                    if not isinstance(fill, tuple):
                        if isinstance(fill, list) and len(fill) == 3:
                            fill = tuple(fill)
                        elif isinstance(fill, str):
                            from tools.secure_regex_utils import safe_extract_rgb_values

                            parsed = safe_extract_rgb_values(fill.strip())
                            if parsed is not None:
                                fill = parsed
                            else:
                                # Try bracket+space format e.g. "[128 128 128]"
                                match = re.match(
                                    r"\[\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s*\]",
                                    fill.strip(),
                                )
                                if match:
                                    r, g, b = (
                                        int(match.group(1)),
                                        int(match.group(2)),
                                        int(match.group(3)),
                                    )
                                    if (
                                        0 <= r <= 255
                                        and 0 <= g <= 255
                                        and 0 <= b <= 255
                                    ):
                                        fill = (r, g, b)
                                    else:
                                        fill = CUSTOM_BOX_COLOUR
                                else:
                                    fill = CUSTOM_BOX_COLOUR
                        else:
                            try:
                                fill = tuple(fill)
                            except Exception:
                                fill = CUSTOM_BOX_COLOUR

                    # Ensure fill is a valid RGB tuple with integer values 0-255
                    # Handle both list and tuple formats, and convert float values to proper RGB
                    if isinstance(fill, (list, tuple)) and len(fill) == 3:
                        # Convert to tuple if it's a list
                        if isinstance(fill, list):
                            fill = tuple(fill)

                        # Check if all elements are valid RGB values
                        valid_rgb = True
                        converted_fill = []

                        for c in fill:
                            if isinstance(c, (int, float)):
                                # If it's a float between 0-1, convert to 0-255 range
                                if isinstance(c, float) and 0 <= c <= 1:
                                    converted_fill.append(int(c * 255))
                                # If it's already an integer 0-255, use as is
                                elif isinstance(c, int) and 0 <= c <= 255:
                                    converted_fill.append(c)
                                # If it's a float > 1, assume it's already in 0-255 range
                                elif isinstance(c, float) and c > 1:
                                    converted_fill.append(int(c))
                                else:
                                    valid_rgb = False
                                    break
                            else:
                                valid_rgb = False
                                break

                        if valid_rgb:
                            fill = tuple(converted_fill)
                        else:
                            print(
                                f"Invalid color values: {fill}. Defaulting to CUSTOM_BOX_COLOUR."
                            )
                            fill = CUSTOM_BOX_COLOUR
                    else:
                        print(
                            f"Invalid fill format: {fill}. Defaulting to CUSTOM_BOX_COLOUR."
                        )
                        fill = CUSTOM_BOX_COLOUR

                        # Ensure the image is in RGB mode
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")

                    draw = ImageDraw.Draw(image)

                    draw.rectangle(coords, fill=fill)

                image.save(output_image_path)
                _out_files.append(output_image_path)

                # For image under review, also produce _redacted.pdf and _redactions_for_review.pdf (same as PDF route)
                if doc is not None and getattr(doc, "page_count", 0) >= 1:
                    try:
                        _tmp_pdf_path = os.path.join(
                            output_folder,
                            file_name_without_ext + "_temp_apply.pdf",
                        )
                        doc.save(_tmp_pdf_path)
                        pdf_doc = pymupdf.open(_tmp_pdf_path)
                        review_pdf_doc = (
                            pymupdf.open(_tmp_pdf_path)
                            if RETURN_PDF_FOR_REVIEW
                            else None
                        )
                        number_of_pages = pdf_doc.page_count
                    except Exception as e:
                        print(f"Failed to create PDFs from image doc: {e}")
                        pdf_doc = None
                        review_pdf_doc = None
                        _tmp_pdf_path = None
                else:
                    # Fallback: doc not available (e.g. pdf_doc_state is list() or None after initial redaction).
                    # Create one-page PDF from the image file so we still produce both PDFs.
                    try:
                        _tmp_pdf_path = os.path.join(
                            output_folder,
                            file_name_without_ext + "_temp_apply.pdf",
                        )
                        img_pdf = pymupdf.open()
                        img_page = img_pdf.new_page(
                            width=image.width, height=image.height
                        )
                        img_page.insert_image(img_page.rect, filename=file_path)
                        img_pdf.save(_tmp_pdf_path)
                        img_pdf.close()
                        pdf_doc = pymupdf.open(_tmp_pdf_path)
                        review_pdf_doc = (
                            pymupdf.open(_tmp_pdf_path)
                            if RETURN_PDF_FOR_REVIEW
                            else None
                        )
                        number_of_pages = pdf_doc.page_count
                    except Exception as e:
                        print(f"Failed to create PDFs from image file: {e}")
                        pdf_doc = None
                        review_pdf_doc = None
                        _tmp_pdf_path = None

            elif file_extension == ".csv":
                pdf_doc = list()

            # If working with pdfs
            elif is_pdf(file_path) is True:
                pdf_doc = pymupdf.open(file_path)
                orig_pdf_file_path = file_path

                _out_files.append(orig_pdf_file_path)

                number_of_pages = pdf_doc.page_count

                # Create review PDF document if RETURN_PDF_FOR_REVIEW is True
                if RETURN_PDF_FOR_REVIEW:
                    review_pdf_doc = pymupdf.open(file_path)
                else:
                    review_pdf_doc = None

            else:
                print("File type not recognised.")

            # Run page loop for both PDF and image (when doc was converted to temp PDF)
            if (
                pdf_doc is not None
                and hasattr(pdf_doc, "page_count")
                and not isinstance(pdf_doc, list)
                and number_of_pages > 0
            ):
                # page_sizes_df and page_to_image_path / page_to_image_dimensions
                # already built once per file above

                # Load images on demand per page (avoids holding all N images in memory).
                # PyMuPDF is not thread-safe for document modification, so redaction stays sequential.
                _page_iter = (
                    progress.tqdm(
                        range(0, number_of_pages),
                        desc="Saving redacted pages to file",
                        unit="pages",
                    )
                    if progress is not None
                    else range(0, number_of_pages)
                )
                for i in _page_iter:
                    page_annotations = (
                        all_image_annotations[i]
                        if i < len(all_image_annotations)
                        else {}
                    )
                    page_boxes = (
                        page_annotations.get("boxes")
                        if isinstance(page_annotations, dict)
                        else []
                    )
                    has_boxes = bool(page_boxes and len(page_boxes) > 0)

                    # Load image only when page has redaction boxes (avoids I/O for blank pages).
                    image = None
                    image_should_close = False
                    if has_boxes:
                        if PROFILE_REDACTION_APPLY:
                            _t_img0 = time.perf_counter()
                        try:
                            _, image, image_should_close = (
                                _load_one_page_image_for_redact(
                                    i,
                                    all_image_annotations,
                                    page_to_image_path,
                                    input_folder,
                                    file_name_with_ext,
                                )
                            )
                        except Exception:
                            image, image_should_close = None, False
                        if image is None:
                            image_should_close = False
                        if PROFILE_REDACTION_APPLY:
                            _profile_image_times.append(time.perf_counter() - _t_img0)
                    elif PROFILE_REDACTION_APPLY:
                        _profile_image_times.append(0.0)

                    pymupdf_page = pdf_doc.load_page(i)
                    current_cropbox = pymupdf_page.cropbox
                    pymupdf_page.set_cropbox(pymupdf_page.mediabox)

                    # Remove existing redaction annotations (collect first to avoid iterator issues)
                    annots_to_remove = [
                        a
                        for a in pymupdf_page.annots()
                        if a.type[0] == pymupdf.PDF_ANNOT_REDACT
                    ]
                    for annot in annots_to_remove:
                        pymupdf_page.delete_annot(annot)

                    # Precomputed dimensions for this page (avoids .loc in redact_page_with_pymupdf)
                    dims = page_to_image_dimensions.get(i + 1)

                    review_pymupdf_page = None
                    if RETURN_PDF_FOR_REVIEW and review_pdf_doc:
                        review_pymupdf_page = review_pdf_doc.load_page(i)
                        review_pymupdf_page.set_cropbox(review_pymupdf_page.mediabox)
                        review_annots_to_remove = [
                            a
                            for a in review_pymupdf_page.annots()
                            if a.type[0] == pymupdf.PDF_ANNOT_REDACT
                        ]
                        for annot in review_annots_to_remove:
                            review_pymupdf_page.delete_annot(annot)

                    # Single pass: apply redactions to both final and (if requested) review page.
                    if has_boxes:
                        if PROFILE_REDACTION_APPLY:
                            _t_redact0 = time.perf_counter()
                        pymupdf_page = redact_page_with_pymupdf(
                            page=pymupdf_page,
                            page_annotations=all_image_annotations[i],
                            image=image,
                            original_cropbox=current_cropbox,
                            page_sizes_df=page_sizes_df,
                            return_pdf_for_review=bool(review_pymupdf_page is None),
                            return_pdf_end_of_redaction=False,
                            input_folder=input_folder,
                            image_dimensions_override=dims,
                            review_page=review_pymupdf_page,
                        )
                        if PROFILE_REDACTION_APPLY:
                            _profile_page_times.append(time.perf_counter() - _t_redact0)
                    else:
                        set_cropbox_safely(pymupdf_page, current_cropbox)
                        pymupdf_page.clean_contents()
                        if review_pymupdf_page is not None:
                            set_cropbox_safely(review_pymupdf_page, current_cropbox)
                            review_pymupdf_page.clean_contents()
                        if PROFILE_REDACTION_APPLY:
                            _profile_page_times.append(0.0)

                    # Close image immediately to free memory before next page
                    if image_should_close and image is not None:
                        try:
                            image.close()
                        except Exception:
                            pass
                    image = None

            progress(0.9, "Saving output files")

            if pdf_doc:
                # Save final redacted PDF
                out_pdf_file_path = (
                    output_folder + file_name_without_ext + "_redacted.pdf"
                )
                save_pdf_with_or_without_compression(
                    pdf_doc, out_pdf_file_path, COMPRESS_REDACTED_PDF
                )
                _out_files.append(out_pdf_file_path)
                pdf_doc.close()
                pdf_doc = None

                # Save review PDF if RETURN_PDF_FOR_REVIEW is True

                if RETURN_PDF_FOR_REVIEW and review_pdf_doc:
                    output_file_name = (
                        file_name_without_ext + "_redactions_for_review.pdf"
                    )
                    out_review_pdf_file_path = output_folder + output_file_name
                    print("Saving PDF file for review:", output_file_name)
                    save_pdf_with_or_without_compression(
                        review_pdf_doc, out_review_pdf_file_path, COMPRESS_REDACTED_PDF
                    )
                    _out_files.append(out_review_pdf_file_path)
                    review_pdf_doc.close()
                    review_pdf_doc = None

                # Remove temp PDF used for image->PDF route
                if _tmp_pdf_path and os.path.isfile(_tmp_pdf_path):
                    try:
                        os.remove(_tmp_pdf_path)
                    except Exception:
                        pass

            else:
                print("PDF input not found. Outputs not saved to PDF.")

        # If save_pdf is not true, then add the original pdf to the output files
        else:
            if is_pdf(file_path) is True:
                orig_pdf_file_path = file_path
                _out_files.append(orig_pdf_file_path)

        _t_review_csv = 0.0
        try:
            if PROFILE_REDACTION_APPLY:
                _t_review0 = time.perf_counter()
            if (
                ENABLE_REVIEW_CSV_PARALLELISM
                and len(all_image_annotations) >= REVIEW_CSV_PARALLEL_MIN_PAGES
            ):
                chunk_size = REVIEW_CSV_PAGES_PER_CHUNK
                chunks = [
                    all_image_annotations[i : i + chunk_size]
                    for i in range(0, len(all_image_annotations), chunk_size)
                ]
                with ThreadPoolExecutor(
                    max_workers=min(MAX_WORKERS, len(chunks))
                ) as executor:
                    partial_dfs = list(
                        executor.map(convert_annotation_data_to_dataframe, chunks)
                    )
                combined = pd.concat(partial_dfs, ignore_index=True)
                _review_df = convert_annotation_json_to_review_df(
                    all_image_annotations,
                    review_file_state.copy(),
                    page_sizes=page_sizes,
                    prebuilt_df=combined,
                )
            else:
                _review_df = convert_annotation_json_to_review_df(
                    all_image_annotations,
                    review_file_state.copy(),
                    page_sizes=page_sizes,
                )

            out_review_file_file_path = (
                output_folder + file_name_with_ext + "_review_file.csv"
            )
            review_cols = [
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
            ]

            if USE_POLARS_FOR_REVIEW and not _review_df.empty:
                coord_cols = ["xmin", "xmax", "ymin", "ymax"]
                cols_to_convert = coord_cols + ["page"]
                temp_pd = _review_df.copy()
                for col in cols_to_convert:
                    if col in temp_pd.columns:
                        temp_pd[col] = pd.to_numeric(temp_pd[col], errors="coerce")
                for col in temp_pd.columns:
                    if col not in cols_to_convert and temp_pd[col].dtype == object:
                        temp_pd[col] = temp_pd[col].astype(str)
                pl_df = pl.from_pandas(temp_pd)
                pl_df = divide_coordinates_by_page_sizes_pl(pl_df, page_sizes_df)
                pl_df = pl_df.select([c for c in review_cols if c in pl_df.columns])
                pl_df.write_csv(out_review_file_file_path)
                _review_df = pl_df.to_pandas()
                if "page" in _review_df.columns and not _review_df.empty:
                    _review_df["page"] = pd.to_numeric(
                        _review_df["page"], errors="coerce"
                    )
                    _review_df["page"] = _review_df["page"].astype("Int64")
                for c in coord_cols:
                    if c in _review_df.columns:
                        _review_df[c] = _review_df[c].astype(float)
            else:
                _review_df = divide_coordinates_by_page_sizes(_review_df, page_sizes_df)
                _review_df = _review_df[review_cols]
                _review_df.to_csv(out_review_file_file_path, index=None)

            _out_files.append(out_review_file_file_path)
            if PROFILE_REDACTION_APPLY:
                _t_review_csv = time.perf_counter() - _t_review0

        except Exception as e:
            print(
                "In apply redactions function, could not save annotations to csv file:",
                e,
            )
        if PROFILE_REDACTION_APPLY:
            _total_page = sum(_profile_page_times)
            _total_img = sum(_profile_image_times)
            print(
                "[PROFILE_REDACTION_APPLY] file=%s | page_sizes=%.3fs | image_load_total=%.3fs | redact_pages_total=%.3fs | review_csv=%.3fs"
                % (
                    file_name_with_ext or file_path,
                    _t_page_sizes,
                    _total_img,
                    _total_page,
                    _t_review_csv,
                )
            )

        return (_out_files, _out_log_files, _review_df)

    if ENABLE_PARALLEL_FILES_APPLY_REDACTIONS and len(file_paths) > 1:
        with ThreadPoolExecutor(
            max_workers=min(MAX_WORKERS, len(file_paths))
        ) as executor:
            futures = [
                executor.submit(_run_apply_redactions_loop, [fp]) for fp in file_paths
            ]
            for fut in as_completed(futures):
                o_f, o_l, rev_df = fut.result()
                output_files.extend(o_f)
                output_log_files.extend(o_l)
                review_df = rev_df
    else:
        o_f, o_l, review_df = _run_apply_redactions_loop(file_paths)
        output_files.extend(o_f)
        output_log_files.extend(o_l)

    return doc, all_image_annotations, output_files, output_log_files, review_df


def get_boxes_json(annotations: AnnotatedImageData):
    return annotations["boxes"]


def update_all_entity_df_dropdowns(
    df: pd.DataFrame,
    label_dropdown_value: str,
    page_dropdown_value: str,
    text_dropdown_value: str,
):
    """
    Update all dropdowns based on rows that exist in a dataframe
    """

    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]

    # Guard against empty lists (e.g. from Gradio when nothing is selected)
    if not label_dropdown_value:
        label_dropdown_value = ["ALL"]
    if not text_dropdown_value:
        text_dropdown_value = ["ALL"]
    if not page_dropdown_value:
        page_dropdown_value = ["1"]

    filtered_df = df.copy()

    if not label_dropdown_value[0]:
        label_dropdown_value[0] = "ALL"
    if not text_dropdown_value[0]:
        text_dropdown_value[0] = "ALL"
    if not page_dropdown_value[0]:
        page_dropdown_value[0] = "1"

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "label"
    )
    text_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "text"
    )
    page_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "page"
    )

    return (
        gr.update(
            value=label_dropdown_value[0],
            choices=recogniser_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value=text_dropdown_value[0],
            choices=text_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value=page_dropdown_value[0],
            choices=page_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
    )


def update_entities_df_recogniser_entities(
    choice: str, df: pd.DataFrame, page_dropdown_value: str, text_dropdown_value: str
):
    """
    Update the rows in a dataframe depending on the user choice from a dropdown
    """

    if isinstance(choice, str):
        choice = [choice]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]

    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    if "ALL" not in page_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["page"].astype(str).isin(page_dropdown_value)
        ]

    if "ALL" not in text_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["text"].astype(str).isin(text_dropdown_value)
        ]

    if "ALL" not in choice:
        filtered_df = filtered_df[filtered_df["label"].astype(str).isin(choice)]

    if not choice[0]:
        choice[0] = "ALL"
    if not text_dropdown_value[0]:
        text_dropdown_value[0] = "ALL"
    if not page_dropdown_value[0]:
        page_dropdown_value[0] = "1"

    # recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
    #     filtered_df, "label"
    # )
    # gr.Dropdown(
    #     value=choice[0],
    #     choices=recogniser_entities_for_drop,
    #     allow_custom_value=True,
    #     interactive=True,
    # )

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "text"
    )
    page_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "page"
    )

    return (
        filtered_df,
        gr.update(
            value=text_dropdown_value[0],
            choices=text_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value=page_dropdown_value[0],
            choices=page_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
    )


def update_entities_df_page(
    choice: str, df: pd.DataFrame, label_dropdown_value: str, text_dropdown_value: str
):
    """
    Update the rows in a dataframe depending on the user choice from a dropdown
    """
    if isinstance(choice, str):
        choice = [choice]
    elif not isinstance(choice, list):
        choice = [str(choice)]
    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    elif not isinstance(label_dropdown_value, list):
        label_dropdown_value = [str(label_dropdown_value)]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]
    elif not isinstance(text_dropdown_value, list):
        text_dropdown_value = [str(text_dropdown_value)]

    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    if "ALL" not in text_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["text"].astype(str).isin(text_dropdown_value)
        ]

    if "ALL" not in label_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["label"].astype(str).isin(label_dropdown_value)
        ]

    if "ALL" not in choice:
        filtered_df = filtered_df[filtered_df["page"].astype(str).isin(choice)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "label"
    )
    text_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "text"
    )

    return (
        filtered_df,
        gr.update(
            value=label_dropdown_value[0],
            choices=recogniser_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value=text_dropdown_value[0],
            choices=text_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
    )


def update_redact_choice_df_from_page_dropdown(choice: str, df: pd.DataFrame):
    """
    Update the rows in a dataframe depending on the user choice from a dropdown
    """
    if isinstance(choice, str):
        choice = [choice]
    elif not isinstance(choice, list):
        choice = [str(choice)]

    if "index" not in df.columns:
        df["index"] = df.index

    filtered_df = df[
        [
            "page",
            "line",
            "word_text",
            "index",
        ]
    ].copy()

    # Apply filtering based on dropdown selections
    if "ALL" not in choice:
        filtered_df = filtered_df.loc[filtered_df["page"].astype(str).isin(choice)]

    # page_entities_for_drop = update_dropdown_list_based_on_dataframe(
    #     filtered_df, "page"
    # )
    # gr.Dropdown(
    #     value=choice[0],
    #     choices=page_entities_for_drop,
    #     allow_custom_value=True,
    #     interactive=True,
    # )

    return filtered_df


def update_entities_df_text(
    choice: str, df: pd.DataFrame, label_dropdown_value: str, page_dropdown_value: str
):
    """
    Update the rows in a dataframe depending on the user choice from a dropdown
    """
    if isinstance(choice, str):
        choice = [choice]
    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]

    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    if "ALL" not in page_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["page"].astype(str).isin(page_dropdown_value)
        ]

    if "ALL" not in label_dropdown_value:
        filtered_df = filtered_df[
            filtered_df["label"].astype(str).isin(label_dropdown_value)
        ]

    if "ALL" not in choice:
        filtered_df = filtered_df[filtered_df["text"].astype(str).isin(choice)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "label"
    )
    page_entities_for_drop = update_dropdown_list_based_on_dataframe(
        filtered_df, "page"
    )

    return (
        filtered_df,
        gr.update(
            value=label_dropdown_value[0],
            choices=recogniser_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value=page_dropdown_value[0],
            choices=page_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
    )


def reset_dropdowns(df: pd.DataFrame):
    """
    Return Gradio dropdown objects with value 'ALL'.
    """
    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "label")
    text_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "text")
    page_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "page")

    return (
        gr.update(
            value="ALL",
            choices=recogniser_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value="ALL",
            choices=text_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
        gr.update(
            value="ALL",
            choices=page_entities_for_drop,
            allow_custom_value=True,
            interactive=True,
        ),
    )


def increase_bottom_page_count_based_on_top(page_number: int):
    return int(page_number)


def df_select_callback_dataframe_row_ocr_with_words(
    df: pd.DataFrame, evt: gr.SelectData
):

    row_value_page = int(evt.row_value[0])  # This is the page number value
    row_value_line = int(evt.row_value[1])  # This is the label number value
    row_value_text = evt.row_value[2]  # This is the text number value

    row_value_index = evt.row_value[3]  # This is the index value

    row_value_df = pd.DataFrame(
        data={
            "page": [row_value_page],
            "line": [row_value_line],
            "word_text": [row_value_text],
            "index": row_value_index,
        }
    )

    return row_value_df, row_value_text


def df_select_callback_dataframe_row(df: pd.DataFrame, evt: gr.SelectData):

    row_value_page = int(evt.row_value[0])  # This is the page number value
    row_value_label = evt.row_value[1]  # This is the label number value
    row_value_text = evt.row_value[2]  # This is the text number value
    row_value_id = evt.row_value[3]  # This is the text number value

    row_value_df = pd.DataFrame(
        data={
            "page": [row_value_page],
            "label": [row_value_label],
            "text": [row_value_text],
            "id": [row_value_id],
        }
    )

    return row_value_df, row_value_text


def df_select_callback_textract_api(df: pd.DataFrame, evt: gr.SelectData):

    row_value_job_id = evt.row_value[0]  # This is the page number value
    # row_value_label = evt.row_value[1] # This is the label number value
    row_value_job_type = evt.row_value[2]  # This is the text number value

    row_value_df = pd.DataFrame(
        data={"job_id": [row_value_job_id], "label": [row_value_job_type]}
    )

    return row_value_job_id, row_value_job_type, row_value_df


def df_select_callback_cost(df: pd.DataFrame, evt: gr.SelectData):

    row_value_code = evt.row_value[0]  # This is the value for cost code
    # row_value_label = evt.row_value[1] # This is the label number value

    # row_value_df = pd.DataFrame(data={"page":[row_value_code], "label":[row_value_label]})

    return row_value_code


def df_select_callback_ocr(df: pd.DataFrame, evt: gr.SelectData):

    row_value_page = int(evt.row_value[0])  # This is the page_number value
    row_value_text = evt.row_value[1]  # This is the text contents

    row_value_df = pd.DataFrame(
        data={"page": [row_value_page], "text": [row_value_text]}
    )

    return row_value_page, row_value_df


# When a user selects a row in the duplicate results table
def store_duplicate_selection(evt: gr.SelectData):
    if not evt.empty:
        selected_index = evt.index[0]
    else:
        selected_index = None

    return selected_index


def get_all_rows_with_same_text(df: pd.DataFrame, text: str):
    """
    Get all rows with the same text as the selected row
    """
    if text:
        # Get all rows with the same text as the selected row
        return df.loc[df["text"] == text]
    else:
        return pd.DataFrame(columns=["page", "label", "text", "id"])


def get_all_rows_with_same_text_redact(df: pd.DataFrame, text: str):
    """
    Get all rows with the same text as the selected row for redaction tasks
    """
    if "index" not in df.columns:
        df["index"] = df.index

    if text and not df.empty:
        # Get all rows with the same text as the selected row
        return df.loc[df["word_text"] == text]
    else:
        return pd.DataFrame(
            columns=[
                "page",
                "line",
                "label",
                "word_text",
                "word_x0",
                "word_y0",
                "word_x1",
                "word_y1",
                "index",
            ]
        )


def update_selected_review_df_row_colour(
    redaction_row_selection: pd.DataFrame,
    review_df: pd.DataFrame,
    previous_id: str = "",
    previous_colour: str = "(0, 0, 0)",
    colour: str = "(1, 0, 255)",
) -> tuple[pd.DataFrame, str, str]:
    """
    Update the colour of a single redaction box based on the values in a selection row
    (Optimized Version)
    """

    # Ensure 'color' column exists, default to previous_colour if previous_id is provided
    if "color" not in review_df.columns:
        review_df["color"] = previous_colour if previous_id else "(0, 0, 0)"

    # Ensure 'id' column exists
    if "id" not in review_df.columns:
        # Assuming fill_missing_ids is a defined function that returns a DataFrame
        # It's more efficient if this is handled outside if possible,
        # or optimized internally.
        print("Warning: 'id' column not found. Calling fill_missing_ids.")
        review_df = fill_missing_ids(
            review_df
        )  # Keep this if necessary, but note it can be slow

    # --- Optimization 1 & 2: Reset existing highlight colours using vectorized assignment ---
    # Reset the color of the previously highlighted row
    if previous_id and previous_id in review_df["id"].values:
        review_df.loc[review_df["id"] == previous_id, "color"] = previous_colour

    # Reset the color of any row that currently has the highlight colour (handle cases where previous_id might not have been tracked correctly)
    # Convert to string for comparison only if the dtype might be mixed or not purely string
    # If 'color' is consistently string, the .astype(str) might be avoidable.
    # Assuming color is consistently string format like '(R, G, B)'
    review_df.loc[review_df["color"] == colour, "color"] = "(0, 0, 0)"

    if not redaction_row_selection.empty and not review_df.empty:
        use_id = (
            "id" in redaction_row_selection.columns
            and "id" in review_df.columns
            and not redaction_row_selection["id"].isnull().all()
            and not review_df["id"].isnull().all()
        )

        selected_merge_cols = ["id"] if use_id else ["label", "page", "text"]

        # --- Optimization 3: Use inner merge directly ---
        # Merge to find rows in review_df that match redaction_row_selection
        merged_reviews = review_df.merge(
            redaction_row_selection[selected_merge_cols],
            on=selected_merge_cols,
            how="inner",  # Use inner join as we only care about matches
        )

        if not merged_reviews.empty:
            # Assuming we only expect one match for highlighting a single row
            # If multiple matches are possible and you want to highlight all,
            # the logic for previous_id and previous_colour needs adjustment.
            new_previous_colour = str(merged_reviews["color"].iloc[0])
            new_previous_id = merged_reviews["id"].iloc[0]

            # --- Optimization 1 & 2: Update color of the matched row using vectorized assignment ---

            if use_id:
                # Faster update if using unique 'id' as merge key
                review_df.loc[review_df["id"].isin(merged_reviews["id"]), "color"] = (
                    colour
                )
            else:
                # More general case using multiple columns - might be slower
                # Create a temporary key for comparison
                def create_merge_key(df, cols):
                    return df[cols].astype(str).agg("_".join, axis=1)

                review_df_key = create_merge_key(review_df, selected_merge_cols)
                merged_reviews_key = create_merge_key(
                    merged_reviews, selected_merge_cols
                )

                review_df.loc[review_df_key.isin(merged_reviews_key), "color"] = colour

            previous_colour = new_previous_colour
            previous_id = new_previous_id
        else:
            # No rows matched the selection
            print("No reviews found matching selection criteria")

            previous_colour = (
                "(0, 0, 0)"  # Reset previous_colour as no row was highlighted
            )
            previous_id = ""  # Reset previous_id

    else:
        # If selection is empty, reset any existing highlights
        review_df.loc[review_df["color"] == colour, "color"] = "(0, 0, 0)"
        previous_colour = "(0, 0, 0)"
        previous_id = ""

    # Ensure column order is maintained if necessary, though pandas generally preserves order
    # Creating a new DataFrame here might involve copying data, consider if this is strictly needed.
    if set(
        [
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
        ]
    ).issubset(review_df.columns):
        review_df = review_df[
            [
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
            ]
        ]
    else:
        print(
            "Warning: Not all expected columns are present in review_df for reordering."
        )

    return review_df, previous_id, previous_colour


def _update_one_page_boxes_color(
    page_idx: int,
    image_obj: dict,
    selection_set: set,
    colour: tuple,
) -> Tuple[int, dict]:
    """Process one page's boxes for color update; safe to run in a thread."""
    out = {
        "image": image_obj.get("image"),
        "boxes": [
            {
                **box,
                "color": (
                    colour
                    if (page_idx, box["label"]) in selection_set
                    else box["color"]
                ),
            }
            for box in image_obj.get("boxes", [])
        ],
    }
    return (page_idx, out)


def update_boxes_color(
    images: list, redaction_row_selection: pd.DataFrame, colour: tuple = (0, 255, 0)
):
    """
    Update the color of bounding boxes in the images list based on redaction_row_selection.

    Parameters:
    - images (list): List of dictionaries containing image paths and box metadata.
    - redaction_row_selection (pd.DataFrame): DataFrame with 'page', 'label', and optionally 'text' columns.
    - colour (tuple): RGB tuple for the new color.

    Returns:
    - Updated list with modified colors.
    """
    selection_set = set(
        zip(redaction_row_selection["page"], redaction_row_selection["label"])
    )
    if not images:
        return images

    max_workers = min(MAX_WORKERS, len(images))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda i_obj: _update_one_page_boxes_color(
                    i_obj[0], i_obj[1], selection_set, colour
                ),
                [(idx, img) for idx, img in enumerate(images)],
            )
        )
    ordered = sorted(results, key=lambda x: x[0])
    return [out for _, out in ordered]


def update_other_annotator_number_from_current(page_number_first_counter: int):
    return page_number_first_counter


def convert_image_coords_to_adobe(
    pdf_page_width: float,
    pdf_page_height: float,
    image_width: float,
    image_height: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Converts coordinates from image space to Adobe PDF space.

    Parameters:
    - pdf_page_width: Width of the PDF page
    - pdf_page_height: Height of the PDF page
    - image_width: Width of the source image
    - image_height: Height of the source image
    - x1, y1, x2, y2: Coordinates in image space
    - page_sizes: List of dicts containing sizes of page as pymupdf page or PIL image

    Returns:
    - Tuple of converted coordinates (x1, y1, x2, y2) in Adobe PDF space
    """

    # Calculate scaling factors
    scale_width = pdf_page_width / image_width
    scale_height = pdf_page_height / image_height

    # Convert coordinates
    pdf_x1 = x1 * scale_width
    pdf_x2 = x2 * scale_width

    # Convert Y coordinates (flip vertical axis)
    # Adobe coordinates start from bottom-left
    pdf_y1 = pdf_page_height - (y1 * scale_height)
    pdf_y2 = pdf_page_height - (y2 * scale_height)

    # Make sure y1 is always less than y2 for Adobe's coordinate system
    if pdf_y1 > pdf_y2:
        pdf_y1, pdf_y2 = pdf_y2, pdf_y1

    return pdf_x1, pdf_y1, pdf_x2, pdf_y2


def convert_pymupdf_coords_to_adobe(
    x1: float, y1: float, x2: float, y2: float, pdf_page_height: float
):
    """
    Converts coordinates from PyMuPDF (fitz) space to Adobe PDF space.

    Parameters:
    - x1, y1, x2, y2: Coordinates in PyMuPDF space
    - pdf_page_height: Total height of the PDF page

    Returns:
    - Tuple of converted coordinates (x1, y1, x2, y2) in Adobe PDF space
    """

    # PyMuPDF uses (0,0) at the bottom-left, while Adobe uses (0,0) at the top-left
    adobe_y1 = pdf_page_height - y2  # Convert top coordinate
    adobe_y2 = pdf_page_height - y1  # Convert bottom coordinate

    return x1, adobe_y1, x2, adobe_y2


def _build_one_redact_element(
    row_dict: dict, pdf_page_height: float, date_str: str
) -> Element:
    """Build a single redact XML element from a row; safe to run in a thread."""
    redact_annot = Element("redact")
    redact_annot.set("opacity", "0.500000")
    redact_annot.set("interior-color", "#000000")
    redact_annot.set("date", date_str)
    redact_annot.set("name", str(uuid.uuid4()))
    page_python_format = int(row_dict["page"]) - 1
    redact_annot.set("page", str(page_python_format))
    redact_annot.set("mimetype", "Form")

    x1_pdf = row_dict["xmin"]
    y1_pdf = row_dict["ymin"]
    x2_pdf = row_dict["xmax"]
    y2_pdf = row_dict["ymax"]
    adobe_x1, adobe_y1, adobe_x2, adobe_y2 = convert_pymupdf_coords_to_adobe(
        x1_pdf, y1_pdf, x2_pdf, y2_pdf, pdf_page_height
    )
    redact_annot.set(
        "rect", f"{adobe_x1:.6f},{adobe_y1:.6f},{adobe_x2:.6f},{adobe_y2:.6f}"
    )
    redact_annot.set("subject", str(row_dict["label"]))
    redact_annot.set("title", str(row_dict.get("label", "Unknown")))

    contents_richtext = SubElement(redact_annot, "contents-richtext")
    body_attrs = {
        "xmlns": "http://www.w3.org/1999/xhtml",
        "{http://www.xfa.org/schema/xfa-data/1.0/}APIVersion": "Acrobat:25.1.0",
        "{http://www.xfa.org/schema/xfa-data/1.0/}spec": "2.0.2",
    }
    body = SubElement(contents_richtext, "body", attrib=body_attrs)
    p_element = SubElement(body, "p", dir="ltr")
    span_attrs = {
        "dir": "ltr",
        "style": "font-size:10.0pt;text-align:left;color:#000000;font-weight:normal;font-style:normal",
    }
    span_element = SubElement(p_element, "span", attrib=span_attrs)
    span_element.text = str(row_dict.get("text", "")).strip()

    pdf_ops_for_black_fill_and_outline = [
        "1 w",
        "0 g",
        "0 G",
        "1 0 0 1 0 0 cm",
        f"{adobe_x1:.2f} {adobe_y1:.2f} m",
        f"{adobe_x2:.2f} {adobe_y1:.2f} l",
        f"{adobe_x2:.2f} {adobe_y2:.2f} l",
        f"{adobe_x1:.2f} {adobe_y2:.2f} l",
        "h",
        "B",
    ]
    data_content_string = "\n".join(pdf_ops_for_black_fill_and_outline) + "\n"
    data_element = SubElement(redact_annot, "data")
    data_element.set("MODE", "filtered")
    data_element.set("encoding", "ascii")
    data_element.set("length", str(len(data_content_string.encode("ascii"))))
    data_element.text = data_content_string
    return redact_annot


def create_xfdf(
    review_file_df: pd.DataFrame,
    pdf_path: str,
    pymupdf_doc: object,
    image_paths: List[str] = list(),
    document_cropboxes: List = list(),
    page_sizes: List[dict] = list(),
):
    """
    Create an xfdf file from a review csv file and a pdf
    """
    xfdf_root = Element(
        "xfdf", xmlns="http://ns.adobe.com/xfdf/", **{"xml:space": "preserve"}
    )
    annots = SubElement(xfdf_root, "annots")

    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        if not page_sizes_df.empty and "mediabox_width" not in review_file_df.columns:
            review_file_df = review_file_df.merge(page_sizes_df, how="left", on="page")
        if "xmin" in review_file_df.columns and review_file_df["xmin"].max() <= 1:
            if (
                "mediabox_width" in review_file_df.columns
                and "mediabox_height" in review_file_df.columns
            ):
                review_file_df["xmin"] = (
                    review_file_df["xmin"] * review_file_df["mediabox_width"]
                )
                review_file_df["xmax"] = (
                    review_file_df["xmax"] * review_file_df["mediabox_width"]
                )
                review_file_df["ymin"] = (
                    review_file_df["ymin"] * review_file_df["mediabox_height"]
                )
                review_file_df["ymax"] = (
                    review_file_df["ymax"] * review_file_df["mediabox_height"]
                )
        elif "image_width" in review_file_df.columns and not page_sizes_df.empty:
            review_file_df = multiply_coordinates_by_page_sizes(
                review_file_df,
                page_sizes_df,
                xmin="xmin",
                xmax="xmax",
                ymin="ymin",
                ymax="ymax",
            )

    # Sequential pass: load each unique page once, set cropbox, store height (PyMuPDF is not thread-safe).
    page_heights = {}
    for page_num_reported in review_file_df["page"].astype(int).unique():
        page_python_format = int(page_num_reported) - 1  # to 0-based
        pymupdf_page = pymupdf_doc.load_page(page_python_format)
        if document_cropboxes and page_python_format < len(document_cropboxes):
            from tools.secure_regex_utils import safe_extract_numbers

            match = safe_extract_numbers(document_cropboxes[page_python_format])
            if match and len(match) == 4:
                rect_values = list(map(float, match))
                pymupdf_page.set_cropbox(Rect(*rect_values))
        page_heights[page_python_format] = pymupdf_page.mediabox.height

    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = (
        now.strftime("D:%Y%m%d%H%M%S")
        + now.strftime("%z")[:3]
        + "'"
        + now.strftime("%z")[3:]
        + "'"
    )

    # Build redact elements in parallel (no PyMuPDF in workers).
    rows_with_heights = []
    for idx, row in review_file_df.iterrows():
        page_python_format = int(row["page"]) - 1
        rows_with_heights.append(
            (idx, row.to_dict(), page_heights.get(page_python_format, 0.0))
        )

    if rows_with_heights:
        max_workers = min(MAX_WORKERS, len(rows_with_heights))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda item: (
                        item[0],
                        _build_one_redact_element(item[1], item[2], date_str),
                    ),
                    rows_with_heights,
                )
            )
        for _, elem in sorted(results, key=lambda x: x[0]):
            annots.append(elem)

    rough_string = tostring(xfdf_root, encoding="unicode", method="xml")
    reparsed = defused_minidom.parseString(rough_string)
    return reparsed.toxml()  # .toprettyxml(indent="  ")


def convert_df_to_xfdf(
    input_files: List[str],
    pdf_doc: Document,
    image_paths: List[str],
    output_folder: str = OUTPUT_FOLDER,
    document_cropboxes: List = list(),
    page_sizes: List[dict] = list(),
):
    """
    Load in files to convert a review file into an Adobe comment file format
    """
    output_paths = list()
    pdf_name = ""
    file_path_name = ""

    if isinstance(input_files, str):
        file_paths_list = [input_files]
    else:
        file_paths_list = input_files

    # Sort the file paths so that the pdfs come first
    file_paths_list = sorted(
        file_paths_list,
        key=lambda x: (
            os.path.splitext(x)[1] != ".pdf",
            os.path.splitext(x)[1] != ".json",
        ),
    )

    for file in file_paths_list:

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        file_path_name = get_file_name_without_type(file_path)
        file_path_end = detect_file_type(file_path)

        if file_path_end == "pdf":
            pdf_name = os.path.basename(file_path)

        if file_path_end == "csv" and "review_file" in file_path_name:
            # If no pdf name, just get the name of the file path
            if not pdf_name:
                pdf_name = file_path_name
            # Read CSV file
            review_file_df = pd.read_csv(file_path)

            # Replace NaN in review file with an empty string
            if "text" in review_file_df.columns:
                review_file_df["text"] = review_file_df["text"].fillna("")
            if "label" in review_file_df.columns:
                review_file_df["label"] = review_file_df["label"].fillna("")

            xfdf_content = create_xfdf(
                review_file_df,
                pdf_name,
                pdf_doc,
                image_paths,
                document_cropboxes,
                page_sizes,
            )

            # Split output_folder (trusted base) from filename (untrusted)
            secure_file_write(
                output_folder,
                file_path_name + "_adobe.xfdf",
                xfdf_content,
                encoding="utf-8",
            )

            # Reconstruct the full path for logging purposes
            output_path = output_folder + file_path_name + "_adobe.xfdf"

            output_paths.append(output_path)

    return output_paths


### Convert xfdf coordinates back to image for app


def convert_adobe_coords_to_image(
    pdf_page_width: float,
    pdf_page_height: float,
    image_width: float,
    image_height: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Converts coordinates from Adobe PDF space to image space.

    Parameters:
    - pdf_page_width: Width of the PDF page
    - pdf_page_height: Height of the PDF page
    - image_width: Width of the source image
    - image_height: Height of the source image
    - x1, y1, x2, y2: Coordinates in Adobe PDF space

    Returns:
    - Tuple of converted coordinates (x1, y1, x2, y2) in image space
    """

    # Calculate scaling factors
    scale_width = image_width / pdf_page_width
    scale_height = image_height / pdf_page_height

    # Convert coordinates
    image_x1 = x1 * scale_width
    image_x2 = x2 * scale_width

    # Convert Y coordinates (flip vertical axis)
    # Adobe coordinates start from bottom-left
    image_y1 = (pdf_page_height - y1) * scale_height
    image_y2 = (pdf_page_height - y2) * scale_height

    # Make sure y1 is always less than y2 for image's coordinate system
    if image_y1 > image_y2:
        image_y1, image_y2 = image_y2, image_y1

    return image_x1, image_y1, image_x2, image_y2


def parse_xfdf(xfdf_path: str):
    """
    Parse the XFDF file and extract redaction annotations.

    Parameters:
    - xfdf_path: Path to the XFDF file

    Returns:
    - List of dictionaries containing redaction information
    """
    # Assuming xfdf_path is a file path. If you are passing the XML string,
    # you would use defused_etree.fromstring(xfdf_string) instead of .parse()
    tree = defused_etree.parse(xfdf_path)
    root = tree.getroot()

    # Define the namespace
    namespace = {"xfdf": "http://ns.adobe.com/xfdf/"}

    redactions = list()

    # Find all redact elements using the namespace
    for redact in root.findall(".//xfdf:redact", namespaces=namespace):

        # Extract text from contents-richtext if it exists
        text_content = ""

        # *** THE FIX IS HERE ***
        # Use the namespace to find the contents-richtext element
        contents_richtext = redact.find(
            ".//xfdf:contents-richtext", namespaces=namespace
        )

        if contents_richtext is not None:
            # Get all text content from the HTML structure
            # The children of contents-richtext (body, p, span) have a different namespace
            # but itertext() cleverly handles that for us.
            text_content = "".join(contents_richtext.itertext()).strip()

        # Fallback to contents attribute if no richtext content
        if not text_content:
            text_content = redact.get("contents", "")

        redaction_info = {
            "image": "",  # Image will be filled in later
            "page": int(redact.get("page")) + 1,  # Convert to 1-based index
            "xmin": float(redact.get("rect").split(",")[0]),
            "ymin": float(redact.get("rect").split(",")[1]),
            "xmax": float(redact.get("rect").split(",")[2]),
            "ymax": float(redact.get("rect").split(",")[3]),
            "label": redact.get("title"),
            "text": text_content,  # Use the extracted text content
            "color": redact.get(
                "border-color", "(0, 0, 0)"
            ),  # Default to black if not specified
        }
        redactions.append(redaction_info)

    return redactions


def convert_xfdf_to_dataframe(
    file_paths_list: List[str],
    pymupdf_doc: Document,
    image_paths: List[str],
    output_folder: str = OUTPUT_FOLDER,
    input_folder: str = INPUT_FOLDER,
):
    """
    Convert redaction annotations from XFDF and associated images into a DataFrame.

    Parameters:
    - xfdf_path: Path to the XFDF file
    - pdf_doc: PyMuPDF document object
    - image_paths: List of PIL Image objects corresponding to PDF pages
    - output_folder: Output folder for file save
    - input_folder: Input folder for image creation

    Returns:
    - DataFrame containing redaction information
    """
    output_paths = list()
    df = pd.DataFrame()
    pdf_name = ""
    pdf_path = ""

    # Sort the file paths so that the pdfs come first
    file_paths_list = sorted(
        file_paths_list,
        key=lambda x: (
            os.path.splitext(x)[1] != ".pdf",
            os.path.splitext(x)[1] != ".json",
        ),
    )

    for file in file_paths_list:

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        file_path_name = get_file_name_without_type(file_path)
        file_path_end = detect_file_type(file_path)

        if file_path_end == "pdf":
            pdf_name = os.path.basename(file_path)
            pdf_path = file_path

            # Add pdf to outputs
            output_paths.append(file_path)

        if file_path_end == "xfdf":

            if not pdf_name:
                message = "Original PDF needed to convert from .xfdf format"
                print(message)
                raise ValueError(message)
            xfdf_path = file

            file_path_name = get_file_name_without_type(xfdf_path)

            # Parse the XFDF file
            redactions = parse_xfdf(xfdf_path)

            # Create a DataFrame from the redaction information
            df = pd.DataFrame(redactions)

            df.fillna("", inplace=True)  # Replace NaN with an empty string

            for _, row in df.iterrows():
                page_python_format = int(row["page"]) - 1

                pymupdf_page = pymupdf_doc.load_page(page_python_format)

                pdf_page_height = pymupdf_page.rect.height
                pdf_page_width = pymupdf_page.rect.width

                image_path = image_paths[page_python_format]

                if isinstance(image_path, str):
                    try:
                        image = Image.open(image_path)
                    except Exception:
                        page_num, out_path, width, height = (
                            process_single_page_for_image_conversion(
                                pdf_path, page_python_format, input_folder=input_folder
                            )
                        )

                        image = Image.open(out_path)

                image_page_width, image_page_height = image.size

                # Convert to image coordinates
                image_x1, image_y1, image_x2, image_y2 = convert_adobe_coords_to_image(
                    pdf_page_width,
                    pdf_page_height,
                    image_page_width,
                    image_page_height,
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"],
                )

                df.loc[_, ["xmin", "ymin", "xmax", "ymax"]] = [
                    image_x1,
                    image_y1,
                    image_x2,
                    image_y2,
                ]

                # Optionally, you can add the image path or other relevant information
                df.loc[_, "image"] = image_path

    out_file_path = output_folder + file_path_name + "_review_file.csv"
    df.to_csv(out_file_path, index=None)

    output_paths.append(out_file_path)

    gr.Info(
        f"Review file saved to {out_file_path}. Now click on '1. Upload original pdf' to view the pdf with the annotations."
    )

    return output_paths
