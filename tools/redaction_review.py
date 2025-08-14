import os
import re
import gradio as gr
import pandas as pd
import numpy as np
import pandas as pd
import string
import random
from xml.etree.ElementTree import Element, SubElement, tostring, parse
from xml.dom import minidom
import uuid
from typing import List, Tuple, Dict, Set
from gradio_image_annotation import image_annotator
from gradio_image_annotation.image_annotator import AnnotatedImageData
from pymupdf import Document, Rect
import pymupdf
from PIL import ImageDraw, Image
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from tools.config import OUTPUT_FOLDER, MAX_IMAGE_PIXELS, INPUT_FOLDER, COMPRESS_REDACTED_PDF
from tools.file_conversion import is_pdf, convert_annotation_json_to_review_df, convert_review_df_to_annotation_json, process_single_page_for_image_conversion, multiply_coordinates_by_page_sizes, convert_annotation_data_to_dataframe, remove_duplicate_images_with_blank_boxes, fill_missing_ids, divide_coordinates_by_page_sizes, save_pdf_with_or_without_compression, fill_missing_ids_in_list
from tools.helper_functions import get_file_name_without_type,  detect_file_type
from tools.file_redaction import redact_page_with_pymupdf

if not MAX_IMAGE_PIXELS: Image.MAX_IMAGE_PIXELS = None

def decrease_page(number:int, all_annotations:dict):
    '''
    Decrease page number for review redactions page.
    '''
    if not all_annotations:
        raise Warning("No annotator object loaded")

    if number > 1:
        return number - 1, number - 1
    elif number <= 1:
        #return 1, 1
        raise Warning("At first page")
    else:
        raise Warning("At first page")        

def increase_page(number:int, all_annotations:dict):
    '''
    Increase page number for review redactions page.
    '''

    if not all_annotations:
        raise Warning("No annotator object loaded")
        #return 1, 1

    max_pages = len(all_annotations)

    if number < max_pages:
        return number + 1, number + 1
    #elif number == max_pages:
    #    return max_pages, max_pages
    else:
        raise Warning("At last page")

def update_zoom(current_zoom_level:int, annotate_current_page:int, decrease:bool=True):
    if decrease == False:
        if current_zoom_level >= 70:
            current_zoom_level -= 10
    else:    
        if current_zoom_level < 110:
            current_zoom_level += 10
        
    return current_zoom_level, annotate_current_page

def update_dropdown_list_based_on_dataframe(df:pd.DataFrame, column:str) -> List["str"]:
    '''
    Gather unique elements from a string pandas Series, then append 'ALL' to the start and return the list.
    '''
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
                entities_for_drop = [str(e) for e in entities_for_drop]  # Convert back to string
                entities_for_drop.insert(0, "ALL")
            except ValueError:
                return ["ALL"]  # Handle case where conversion fails

        return entities_for_drop  # Ensure to return the list
    else:
        return ["ALL"]

def get_filtered_recogniser_dataframe_and_dropdowns(page_image_annotator_object:AnnotatedImageData,
                                 recogniser_dataframe_base:pd.DataFrame,
                                 recogniser_dropdown_value:str,
                                 text_dropdown_value:str,
                                 page_dropdown_value:str,
                                 review_df:pd.DataFrame=[],
                                 page_sizes:List[str]=[]):
    '''
    Create a filtered recogniser dataframe and associated dropdowns based on current information in the image annotator and review data frame.
    '''

    recogniser_entities_list = ["Redaction"]
    recogniser_dataframe_out = recogniser_dataframe_base
    recogniser_dataframe_out_gr = gr.Dataframe()
    review_dataframe = review_df

    try:
        #print("converting annotation json in get_filtered_recogniser...")

        review_dataframe = convert_annotation_json_to_review_df(page_image_annotator_object, review_df, page_sizes)

        recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(review_dataframe, "label")
        recogniser_entities_drop = gr.Dropdown(value=recogniser_dropdown_value, choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)

        # This is the choice list for entities when creating a new redaction box
        recogniser_entities_list = [entity for entity in recogniser_entities_for_drop.copy() if entity != 'Redaction' and entity != 'ALL']  # Remove any existing 'Redaction'
        recogniser_entities_list.insert(0, 'Redaction')  # Add 'Redaction' to the start of the list        

        text_entities_for_drop = update_dropdown_list_based_on_dataframe(review_dataframe, "text")
        text_entities_drop = gr.Dropdown(value=text_dropdown_value, choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

        page_entities_for_drop = update_dropdown_list_based_on_dataframe(review_dataframe, "page")
        page_entities_drop = gr.Dropdown(value=page_dropdown_value, choices=page_entities_for_drop, allow_custom_value=True, interactive=True)

        recogniser_dataframe_out_gr = gr.Dataframe(review_dataframe[["page", "label", "text", "id"]], show_search="filter", col_count=(4, "fixed"), type="pandas", headers=["page", "label", "text", "id"], show_fullscreen_button=True, wrap=True, max_height=400, static_columns=[0,1,2,3])

        recogniser_dataframe_out = review_dataframe[["page", "label", "text", "id"]]

    except Exception as e:
        print("Could not extract recogniser information:", e)
        recogniser_dataframe_out = recogniser_dataframe_base[["page", "label", "text", "id"]]

        label_choices = review_dataframe["label"].astype(str).unique().tolist()
        text_choices = review_dataframe["text"].astype(str).unique().tolist()
        page_choices = review_dataframe["page"].astype(str).unique().tolist()

        recogniser_entities_drop = gr.Dropdown(value=recogniser_dropdown_value, choices=label_choices, allow_custom_value=True, interactive=True)
        recogniser_entities_list = ["Redaction"]
        text_entities_drop = gr.Dropdown(value=text_dropdown_value, choices=text_choices, allow_custom_value=True, interactive=True)
        page_entities_drop = gr.Dropdown(value=page_dropdown_value, choices=page_choices, allow_custom_value=True, interactive=True)

    return recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_drop, recogniser_entities_list, text_entities_drop, page_entities_drop

def update_recogniser_dataframes(page_image_annotator_object:AnnotatedImageData, recogniser_dataframe_base:pd.DataFrame, recogniser_entities_dropdown_value:str="ALL", text_dropdown_value:str="ALL", page_dropdown_value:str="ALL", review_df:pd.DataFrame=[], page_sizes:list[str]=[]):
    '''
    Update recogniser dataframe information that appears alongside the pdf pages on the review screen.
    '''
    recogniser_entities_list = ["Redaction"]
    recogniser_dataframe_out = pd.DataFrame()
    recogniser_dataframe_out_gr = gr.Dataframe()

    # If base recogniser dataframe is empy, need to create it.
    if recogniser_dataframe_base.empty:
        recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_drop, recogniser_entities_list, text_entities_drop, page_entities_drop = get_filtered_recogniser_dataframe_and_dropdowns(page_image_annotator_object, recogniser_dataframe_base, recogniser_entities_dropdown_value, text_dropdown_value, page_dropdown_value, review_df, page_sizes)    
    elif recogniser_dataframe_base.iloc[0,0] == "":
        recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_dropdown_value, recogniser_entities_list, text_entities_drop, page_entities_drop = get_filtered_recogniser_dataframe_and_dropdowns(page_image_annotator_object, recogniser_dataframe_base, recogniser_entities_dropdown_value, text_dropdown_value, page_dropdown_value, review_df, page_sizes)
    else:
        recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_dropdown, recogniser_entities_list, text_dropdown, page_dropdown = get_filtered_recogniser_dataframe_and_dropdowns(page_image_annotator_object, recogniser_dataframe_base, recogniser_entities_dropdown_value, text_dropdown_value, page_dropdown_value, review_df, page_sizes)

        review_dataframe, text_entities_drop, page_entities_drop = update_entities_df_recogniser_entities(recogniser_entities_dropdown_value, recogniser_dataframe_out, page_dropdown_value, text_dropdown_value)

        recogniser_dataframe_out_gr = gr.Dataframe(review_dataframe[["page", "label", "text", "id"]], show_search="filter", col_count=(4, "fixed"), type="pandas", headers=["page", "label", "text", "id"], show_fullscreen_button=True, wrap=True, max_height=400, static_columns=[0,1,2,3])
        
        recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(recogniser_dataframe_out, "label")
        recogniser_entities_drop = gr.Dropdown(value=recogniser_entities_dropdown_value, choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)

        recogniser_entities_list_base = recogniser_dataframe_out["label"].astype(str).unique().tolist()

        # Recogniser entities list is the list of choices that appear when you make a new redaction box
        recogniser_entities_list = [entity for entity in recogniser_entities_list_base if entity != 'Redaction']
        recogniser_entities_list.insert(0, 'Redaction')

    return recogniser_entities_list, recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_drop, text_entities_drop, page_entities_drop

def undo_last_removal(backup_review_state:pd.DataFrame, backup_image_annotations_state:list[dict], backup_recogniser_entity_dataframe_base:pd.DataFrame):

    if backup_image_annotations_state:
        return backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base
    else:
        raise Warning("No actions have been taken to undo")

def update_annotator_page_from_review_df(
    review_df: pd.DataFrame,
    image_file_paths:List[str], # Note: This input doesn't seem used in the original logic flow after the first line was removed
    page_sizes:List[dict],
    current_image_annotations_state:List[str], # This should ideally be List[dict] based on its usage
    current_page_annotator:object, # Should be dict or a custom annotation object for one page
    selected_recogniser_entity_df_row:pd.DataFrame,
    input_folder:str,
    doc_full_file_name_textbox:str
) -> Tuple[object, List[dict], int, List[dict], pd.DataFrame, int]: # Correcting return types based on usage
    '''
    Update the visible annotation object and related objects with the latest review file information,
    optimising by processing only the current page's data.
    '''
    # Assume current_image_annotations_state is List[dict] and current_page_annotator is dict
    out_image_annotations_state: List[dict] = list(current_image_annotations_state) # Make a copy to avoid modifying input in place
    out_current_page_annotator: dict = current_page_annotator

    # Get the target page number from the selected row
    # Safely access the page number, handling potential errors or empty DataFrame
    gradio_annotator_current_page_number: int = 1
    annotate_previous_page: int = 0 # Renaming for clarity if needed, matches original output

    if not selected_recogniser_entity_df_row.empty and 'page' in selected_recogniser_entity_df_row.columns:
        try:
            selected_page= selected_recogniser_entity_df_row['page'].iloc[0]
            gradio_annotator_current_page_number = int(selected_page)
            annotate_previous_page = gradio_annotator_current_page_number # Store original page number
        except (IndexError, ValueError, TypeError):
            print("Warning: Could not extract valid page number from selected_recogniser_entity_df_row. Defaulting to page 1.")
            gradio_annotator_current_page_number = 1 # Or 0 depending on 1-based vs 0-based indexing elsewhere

    # Ensure page number is valid and 1-based for external display/logic
    if gradio_annotator_current_page_number <= 0: gradio_annotator_current_page_number = 1

    page_max_reported = len(page_sizes) #len(out_image_annotations_state)
    if gradio_annotator_current_page_number > page_max_reported:
        print("current page is greater than highest page:", page_max_reported)
        gradio_annotator_current_page_number = page_max_reported # Cap at max pages

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
        if 'page' in review_df.columns:
             review_df['page'] = pd.to_numeric(review_df['page'], errors='coerce').fillna(-1).astype(int)

             current_image_path = out_image_annotations_state[page_num_reported_zero_indexed]['image']

             replaced_image_path, page_sizes_df = replace_placeholder_image_with_real_image(doc_full_file_name_textbox, current_image_path, page_sizes_df, gradio_annotator_current_page_number, input_folder)

             # page_sizes_df has been changed - save back to page_sizes_object
             page_sizes = page_sizes_df.to_dict(orient='records')
             review_df.loc[review_df["page"]==gradio_annotator_current_page_number, 'image'] = replaced_image_path
             images_list = list(page_sizes_df["image_path"])
             images_list[page_num_reported_zero_indexed] = replaced_image_path
             out_image_annotations_state[page_num_reported_zero_indexed]['image'] = replaced_image_path

             current_page_review_df = review_df[review_df['page'] == gradio_annotator_current_page_number].copy()          
             current_page_review_df = multiply_coordinates_by_page_sizes(current_page_review_df, page_sizes_df)

        else:
            print(f"Warning: 'page' column not found in review_df. Cannot filter for page {gradio_annotator_current_page_number}. Skipping update from review_df.")
            current_page_review_df = pd.DataFrame() # Empty dataframe if filter fails

        if not current_page_review_df.empty:
            # Convert the current page's review data to annotation list format for *this page*

            current_page_annotations_list = []
            # Define expected annotation dict keys, including 'image', 'page', coords, 'label', 'text', 'color' etc.
            # Assuming review_df has compatible columns
            expected_annotation_keys = ['label', 'color', 'xmin', 'ymin', 'xmax', 'ymax', 'text', 'id'] # Add/remove as needed

            # Ensure necessary columns exist in current_page_review_df before converting rows
            for key in expected_annotation_keys:
                 if key not in current_page_review_df.columns:
                      # Add missing column with default value
                      # Use np.nan for numeric, '' for string/object
                      default_value = np.nan if key in ['xmin', 'ymin', 'xmax', 'ymax'] else ''
                      current_page_review_df[key] = default_value

            # Convert filtered DataFrame rows to list of dicts
            # Using .to_dict(orient='records') is efficient for this
            current_page_annotations_list_raw = current_page_review_df[expected_annotation_keys].to_dict(orient='records')

            current_page_annotations_list = current_page_annotations_list_raw

            # Update the annotations state for the current page
            page_state_entry_found = False
            for i, page_state_entry in enumerate(out_image_annotations_state):
                # Assuming page_state_entry has a 'page' key (1-based)

                match = re.search(r"(\d+)\.png$", page_state_entry['image'])
                if match: page_no = int(match.group(1))
                else: page_no = 0

                if 'image' in page_state_entry and page_no == page_num_reported_zero_indexed:
                    # Replace the annotations list for this page with the new list from review_df
                    out_image_annotations_state[i]['boxes'] = current_page_annotations_list

                    # Update the image path as well, based on review_df if available, or keep existing
                    # Assuming review_df has an 'image' column for this page
                    if 'image' in current_page_review_df.columns and not current_page_review_df.empty:
                         # Use the image path from the first row of the filtered review_df
                         out_image_annotations_state[i]['image'] = current_page_review_df['image'].iloc[0]
                    page_state_entry_found = True
                    break

            if not page_state_entry_found:
                 print(f"Warning: Entry for page {gradio_annotator_current_page_number} not found in current_image_annotations_state. Cannot update page annotations.")

    # --- Image Path and Page Size Handling ---
    # Get the image path for the current page from the updated state
    current_image_path = None
    if len(out_image_annotations_state) > page_num_reported_zero_indexed and 'image' in out_image_annotations_state[page_num_reported_zero_indexed]:
         current_image_path = out_image_annotations_state[page_num_reported_zero_indexed]['image']
    else:
         print(f"Warning: Could not get image path from state for page index {page_num_reported_zero_indexed}.")


    # Replace placeholder image with real image path if needed
    if current_image_path and not page_sizes_df.empty:
        try:
            replaced_image_path, page_sizes_df = replace_placeholder_image_with_real_image(
                doc_full_file_name_textbox, current_image_path, page_sizes_df,
                gradio_annotator_current_page_number, input_folder # Use 1-based page number
            )

            # Update state and review_df with the potentially replaced image path
            if len(out_image_annotations_state) > page_num_reported_zero_indexed:
                 out_image_annotations_state[page_num_reported_zero_indexed]['image'] = replaced_image_path

            if 'page' in review_df.columns and 'image' in review_df.columns:
                 review_df.loc[review_df["page"]==gradio_annotator_current_page_number, 'image'] = replaced_image_path

        except Exception as e:
             print(f"Error during image path replacement for page {gradio_annotator_current_page_number}: {e}")


    # Save back page_sizes_df to page_sizes list format
    if not page_sizes_df.empty:
        page_sizes = page_sizes_df.to_dict(orient='records')
    else:
        page_sizes = [] # Ensure page_sizes is a list if df is empty

    # --- Re-evaluate Coordinate Multiplication and Duplicate Removal ---
    # Let's assume remove_duplicate_images_with_blank_boxes expects the raw list of dicts state format:
    try:
         out_image_annotations_state = remove_duplicate_images_with_blank_boxes(out_image_annotations_state)
    except Exception as e:
         print(f"Error during duplicate removal: {e}. Proceeding without duplicate removal.")


    # Select the current page's annotation object from the (potentially updated) state
    if len(out_image_annotations_state) > page_num_reported_zero_indexed:
         out_current_page_annotator = out_image_annotations_state[page_num_reported_zero_indexed]
    else:
         print(f"Warning: Cannot select current page annotator object for index {page_num_reported_zero_indexed}.")
         out_current_page_annotator = {} # Or None, depending on expected output type

    # Return final page number
    final_page_number_returned = gradio_annotator_current_page_number

    return (out_current_page_annotator,
            out_image_annotations_state,
            final_page_number_returned,
            page_sizes,
            review_df, # review_df might have its 'page' column type changed, keep it as is or revert if necessary
            annotate_previous_page) # The original page number from selected_recogniser_entity_df_row

# --- Helper Function for ID Generation ---
# This function encapsulates your ID logic in a performant, batch-oriented way.
def _generate_unique_ids(
    num_ids_to_generate: int, 
    existing_ids_set: Set[str]
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
        candidate_id = ''.join(random.choices(character_set, k=id_length))
        
        # Check against both pre-existing IDs and IDs generated in this batch
        if candidate_id not in existing_ids_set and candidate_id not in newly_generated_ids:
            newly_generated_ids.add(candidate_id)
            
    return list(newly_generated_ids)

def create_annotation_objects_from_filtered_ocr_results_with_words(
    filtered_ocr_results_with_words_df: pd.DataFrame, 
    ocr_results_with_words_df_base: pd.DataFrame,
    page_sizes: List[Dict],
    existing_annotations_df: pd.DataFrame,
    existing_annotations_list: List[Dict],
    existing_recogniser_entity_df: pd.DataFrame,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[List[Dict], List[Dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates and merges new annotations, using custom ID logic.
    """

    print("Identifying new redactions to add")
    progress(0.1, "Identifying new redactions to add")
    if filtered_ocr_results_with_words_df.empty:
        print("No new annotations to add.")
        updated_annotations_df = existing_annotations_df.copy()
    else:
        # join_keys = ['page', 'line', 'word_text', 'word_x0']
        # new_annotations_df = pd.merge(
        #     ocr_results_with_words_df_base,
        #     filtered_ocr_results_with_words_df[join_keys],
        #     on=join_keys,
        #     how='inner'
        # )

        filtered_ocr_results_with_words_df.index = filtered_ocr_results_with_words_df["index"]

        new_annotations_df = ocr_results_with_words_df_base.loc[filtered_ocr_results_with_words_df.index].copy()

        if new_annotations_df.empty:
             print("No new annotations to add.")
             updated_annotations_df = existing_annotations_df.copy()
        else:
            # --- Custom ID Generation ---
            progress(0.2, "Creating new redaction IDs")
            # 1. Get all IDs that already exist to ensure we don't create duplicates.
            existing_ids = set()
            if 'id' in existing_annotations_df.columns:
                existing_ids = set(existing_annotations_df['id'].dropna())
            
            # 2. Generate the exact number of new, unique IDs required.
            num_new_ids = len(new_annotations_df)
            new_id_list = _generate_unique_ids(num_new_ids, existing_ids)
            
            # 3. Assign the new IDs and other columns in a vectorized way.
            page_to_image_map = {item['page']: item['image_path'] for item in page_sizes}
            
            progress(0.4, "Assigning new redaction details to dataframe")
            new_annotations_df = new_annotations_df.assign(
                image=lambda df: df['page'].map(page_to_image_map),
                label="Redaction",
                color='(0, 0, 0)',
                id=new_id_list  # Assign the pre-generated list of unique IDs
            ).rename(columns={
                'word_x0': 'xmin',
                'word_y0': 'ymin',
                'word_x1': 'xmax',
                'word_y1': 'ymax',
                'word_text': 'text'
            })
            
            annotation_cols = ['image', 'page', 'label', 'color', 'xmin', 'ymin', 'xmax', 'ymax', 'text', 'id']
            new_annotations_df = new_annotations_df[annotation_cols]

            key_cols = ['page', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'text']

            progress(0.5, "Checking suggested redactions against existing")
            
            if existing_annotations_df.empty or not all(col in existing_annotations_df.columns for col in key_cols):
                unique_new_df = new_annotations_df
            else:
                # I'm not doing checks on this anymore
                # merged = pd.merge(
                #     new_annotations_df,
                #     existing_annotations_df[key_cols].drop_duplicates(),
                #     on=key_cols,
                #     how='left',
                #     indicator=True
                # )
                # unique_new_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                unique_new_df = new_annotations_df

            print(f"Found {len(unique_new_df)} new unique annotations to add.")
            updated_annotations_df = pd.concat([existing_annotations_df, unique_new_df], ignore_index=True)

    # --- Part 4: Convert final DataFrame to list-of-dicts ---
    updated_recogniser_entity_df = pd.DataFrame()
    if not updated_annotations_df.empty:
         updated_recogniser_entity_df = updated_annotations_df[["page", "label", "text", "id"]]

    if not page_sizes:
        print("Warning: page_sizes is empty. No pages to process.")
        return [], existing_annotations_list, pd.DataFrame(), existing_annotations_df, pd.DataFrame(), existing_recogniser_entity_df

    all_pages_df = pd.DataFrame(page_sizes).rename(columns={'image_path': 'image'})
    
    if not updated_annotations_df.empty:
        page_to_image_map = {item['page']: item['image_path'] for item in page_sizes}
        updated_annotations_df['image'] = updated_annotations_df['page'].map(page_to_image_map)
        merged_df = pd.merge(all_pages_df[['image']], updated_annotations_df, on='image', how='left')
    else:
        merged_df = all_pages_df[['image']]
        
    final_annotations_list = []
    box_cols = ['label', 'color', 'xmin', 'ymin', 'xmax', 'ymax', 'text', 'id']
    
    for image_path, group in progress.tqdm(merged_df.groupby('image'), desc="Adding redaction boxes to annotation object"):
        if pd.isna(group.iloc[0].get('id')):
            boxes = []
        else:
            valid_box_cols = [col for col in box_cols if col in group.columns]
            boxes = group[valid_box_cols].to_dict('records')
            
        final_annotations_list.append({
            "image": image_path,
            "boxes": boxes
        })

    return final_annotations_list, existing_annotations_list, updated_annotations_df, existing_annotations_df, updated_recogniser_entity_df, existing_recogniser_entity_df

def exclude_selected_items_from_redaction(review_df: pd.DataFrame,
                                          selected_rows_df: pd.DataFrame,
                                          image_file_paths:List[str],
                                          page_sizes:List[dict],
                                          image_annotations_state:dict,
                                          recogniser_entity_dataframe_base:pd.DataFrame):
    ''' 
    Remove selected items from the review dataframe from the annotation object and review dataframe.
    '''

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
        selected_subset = selected_rows_df[selected_merge_cols].drop_duplicates(subset=selected_merge_cols)

        # Perform anti-join using merge with indicator
        merged_df = review_df.merge(selected_subset, on=selected_merge_cols, how='left', indicator=True)
        out_review_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        out_image_annotations_state = convert_review_df_to_annotation_json(out_review_df, image_file_paths, page_sizes)

        out_recogniser_entity_dataframe_base = out_review_df[["page", "label", "text", "id"]]
    
    # Either there is nothing left in the selection dataframe, or the review dataframe
    else:
        out_review_df = review_df
        out_recogniser_entity_dataframe_base = recogniser_entity_dataframe_base
        out_image_annotations_state = image_annotations_state

    return out_review_df, out_image_annotations_state, out_recogniser_entity_dataframe_base, backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base

def replace_annotator_object_img_np_array_with_page_sizes_image_path(
        all_image_annotations:List[dict],
        page_image_annotator_object:AnnotatedImageData,
        page_sizes:List[dict],
        page:int):

        '''
        Check if the image value in an AnnotatedImageData dict is a placeholder or np.array. If either of these, replace the value with the file path of the image that is hopefully already loaded into the app related to this page.
        '''

        page_zero_index = page - 1
        
        if isinstance(all_image_annotations[page_zero_index]["image"], np.ndarray) or "placeholder_image" in all_image_annotations[page_zero_index]["image"] or isinstance(page_image_annotator_object['image'], np.ndarray):
            page_sizes_df = pd.DataFrame(page_sizes)
            page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(pd.to_numeric, errors="coerce")

            # Check for matching pages
            matching_paths = page_sizes_df.loc[page_sizes_df['page'] == page, "image_path"].unique()

            if matching_paths.size > 0:
                image_path = matching_paths[0]
                page_image_annotator_object['image'] = image_path
                all_image_annotations[page_zero_index]["image"] = image_path
            else:
                print(f"No image path found for page {page}.")

        return page_image_annotator_object, all_image_annotations

def replace_placeholder_image_with_real_image(doc_full_file_name_textbox:str, current_image_path:str, page_sizes_df:pd.DataFrame, page_num_reported:int, input_folder:str):
        ''' If image path is still not valid, load in a new image an overwrite it. Then replace all items in the image annotation object for all pages based on the updated information.'''

        page_num_reported_zero_indexed = page_num_reported - 1

        if not os.path.exists(current_image_path):        

            page_num, replaced_image_path, width, height = process_single_page_for_image_conversion(doc_full_file_name_textbox, page_num_reported_zero_indexed, input_folder=input_folder)

            # Overwrite page_sizes values 
            page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_width"] = width
            page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_height"] = height
            page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_path"] = replaced_image_path
        
        else:
            if not page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_width"].isnull().all():
                width = page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_width"].max()
                height = page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_height"].max()      
            else:
                image = Image.open(current_image_path)
                width = image.width
                height = image.height

                page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_width"] = width
                page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_height"] = height

            page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_path"] = current_image_path

            replaced_image_path = current_image_path
        
        return replaced_image_path, page_sizes_df

def update_annotator_object_and_filter_df(
    all_image_annotations:List[AnnotatedImageData],
    gradio_annotator_current_page_number:int,
    recogniser_entities_dropdown_value:str="ALL",
    page_dropdown_value:str="ALL",
    page_dropdown_redaction_value:str="1",
    text_dropdown_value:str="ALL",
    recogniser_dataframe_base:pd.DataFrame=None, # Simplified default
    zoom:int=100,
    review_df:pd.DataFrame=None, # Use None for default empty DataFrame
    page_sizes:List[dict]=[],
    doc_full_file_name_textbox:str='',
    input_folder:str=INPUT_FOLDER
) -> Tuple[image_annotator, gr.Number, gr.Number, int, str, gr.Dataframe, pd.DataFrame, List[str], List[str], List[dict], List[AnnotatedImageData]]:
    '''
    Update a gradio_image_annotation object with new annotation data for the current page
    and update filter dataframes, optimizing by processing only the current page's data for display.
    '''

    zoom_str = str(zoom) + '%'

    # Handle default empty review_df and recogniser_dataframe_base
    if review_df is None or not isinstance(review_df, pd.DataFrame):
         review_df = pd.DataFrame(columns=["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text", "id"])
    if recogniser_dataframe_base is None: # Create a simple default if None
         recogniser_dataframe_base = gr.Dataframe(pd.DataFrame(data={"page":[], "label":[], "text":[], "id":[]}))


    # Handle empty all_image_annotations state early
    if not all_image_annotations:
        print("No all_image_annotation object found")
        # Return blank/default outputs
        
        blank_annotator = image_annotator(
            value = None, boxes_alpha=0.1, box_thickness=1, label_list=[], label_colors=[],
            show_label=False, height=zoom_str, width=zoom_str, box_min_size=1,
            box_selected_thickness=2, handle_size=4, sources=None,
            show_clear_button=False, show_share_button=False, show_remove_button=False,
            handles_cursor=True, interactive=True, use_default_label=True
        )
        blank_df_out_gr = gr.Dataframe(pd.DataFrame(columns=["page", "label", "text", "id"]))
        blank_df_modified = pd.DataFrame(columns=["page", "label", "text", "id"])

        return (blank_annotator, gr.Number(value=1), gr.Number(value=1), 1,
                recogniser_entities_dropdown_value, blank_df_out_gr, blank_df_modified,
                [], [], [], [], []) # Return empty lists/defaults for other outputs

    # Validate and bound the current page number (1-based logic)
    page_num_reported = max(1, gradio_annotator_current_page_number) # Minimum page is 1
    page_max_reported = len(all_image_annotations)
    if page_num_reported > page_max_reported:
        page_num_reported = page_max_reported

    page_num_reported_zero_indexed = page_num_reported - 1
    annotate_previous_page = page_num_reported # Store the determined page number

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
        updated_page_object, all_image_annotations_after_img_replace = replace_annotator_object_img_np_array_with_page_sizes_image_path(
             all_image_annotations, page_object_to_update, page_sizes, page_num_reported)

        all_image_annotations = all_image_annotations_after_img_replace

        # Now handle the actual image file path replacement using replace_placeholder_image_with_real_image
        current_image_path = updated_page_object.get('image') # Get potentially updated image path

        if current_image_path and not page_sizes_df.empty:
            try:
                replaced_image_path, page_sizes_df = replace_placeholder_image_with_real_image(
                    doc_full_file_name_textbox, current_image_path, page_sizes_df,
                    page_num_reported, input_folder=input_folder # Use 1-based page num
                )

                # Update the image path in the state and review_df for the current page
                # Find the correct entry in all_image_annotations list again by index
                if len(all_image_annotations) > page_num_reported_zero_indexed:
                     all_image_annotations[page_num_reported_zero_indexed]['image'] = replaced_image_path

                # Update review_df's image path for this page
                if 'page' in review_df.columns and 'image' in review_df.columns:
                     # Ensure review_df page column is numeric for filtering
                     review_df['page'] = pd.to_numeric(review_df['page'], errors='coerce').fillna(-1).astype(int)
                     review_df.loc[review_df["page"]==page_num_reported, 'image'] = replaced_image_path


            except Exception as e:
                 print(f"Error during image path replacement for page {page_num_reported}: {e}")
    else:
         print(f"Warning: Page index {page_num_reported_zero_indexed} out of bounds for all_image_annotations list.")


    # Save back page_sizes_df to page_sizes list format
    if not page_sizes_df.empty:
        page_sizes = page_sizes_df.to_dict(orient='records')
    else:
        page_sizes = [] # Ensure page_sizes is a list if df is empty

    # --- OPTIMIZATION: Prepare data *only* for the current page for display ---
    current_page_image_annotator_object = None
    if len(all_image_annotations) > page_num_reported_zero_indexed:
        page_data_for_display = all_image_annotations[page_num_reported_zero_indexed]

        # Convert current page annotations list to DataFrame for coordinate multiplication IF needed
        # Assuming coordinate multiplication IS needed for display if state stores relative coords
        current_page_annotations_df = convert_annotation_data_to_dataframe([page_data_for_display])

        if not current_page_annotations_df.empty and not page_sizes_df.empty:
             # Multiply coordinates *only* for this page's DataFrame
             try:
                 # Need the specific page's size for multiplication
                 page_size_row = page_sizes_df[page_sizes_df['page'] == page_num_reported]
                 if not page_size_row.empty:
                      current_page_annotations_df = multiply_coordinates_by_page_sizes(
                          current_page_annotations_df, page_size_row, # Pass only the row for the current page
                          xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"
                      )
             
             except Exception as e:
                  print(f"Warning: Error during coordinate multiplication for page {page_num_reported}: {e}. Using original coordinates.")
                  # If error, proceed with original coordinates or handle as needed

        if "color" not in current_page_annotations_df.columns:
            current_page_annotations_df['color'] = '(0, 0, 0)'

        # Convert the processed DataFrame back to the list of dicts format for the annotator
        processed_current_page_annotations_list = current_page_annotations_df[["xmin", "xmax", "ymin", "ymax", "label", "color", "text", "id"]].to_dict(orient='records')

        # Construct the final object expected by the Gradio ImageAnnotator value parameter
        current_page_image_annotator_object: AnnotatedImageData = {
            'image': page_data_for_display.get('image'), # Use the (potentially updated) image path
            'boxes': processed_current_page_annotations_list
        }

    # --- Update Dropdowns and Review DataFrame ---
    # This external function still operates on potentially large DataFrames.
    # It receives all_image_annotations and a copy of review_df.
    try:
        recogniser_entities_list, recogniser_dataframe_out_gr, recogniser_dataframe_modified, recogniser_entities_dropdown_value, text_entities_drop, page_entities_drop = update_recogniser_dataframes(
             all_image_annotations, # Pass the updated full state
             recogniser_dataframe_base,
             recogniser_entities_dropdown_value,
             text_dropdown_value,
             page_dropdown_value,
             review_df.copy(), # Keep the copy as per original function call
             page_sizes # Pass updated page sizes
        )
        # Generate default black colors for labels if needed by image_annotator
        recogniser_colour_list = [(0, 0, 0) for _ in range(len(recogniser_entities_list))]

    except Exception as e:
        print(f"Error calling update_recogniser_dataframes: {e}. Returning empty/default filter data.")
        recogniser_entities_list = []
        recogniser_colour_list = []
        recogniser_dataframe_out_gr = gr.Dataframe(pd.DataFrame(columns=["page", "label", "text", "id"]))
        recogniser_dataframe_modified = pd.DataFrame(columns=["page", "label", "text", "id"])
        text_entities_drop = []
        page_entities_drop = []


    # --- Final Output Components ---
    page_number_reported_gradio_comp = gr.Number(label = "Current page", value=page_num_reported, precision=0)

    ### Present image_annotator outputs
    # Handle the case where current_page_image_annotator_object couldn't be prepared
    if current_page_image_annotator_object is None:
        # This should ideally be covered by the initial empty check for all_image_annotations,
        # but as a safeguard:
        print("Warning: Could not prepare annotator object for the current page.")
        out_image_annotator = image_annotator(value=None, interactive=False) # Present blank/non-interactive
    else:
        out_image_annotator = image_annotator(
            value = current_page_image_annotator_object,
            boxes_alpha=0.1,
            box_thickness=1,
            label_list=recogniser_entities_list, # Use labels from update_recogniser_dataframes
            label_colors=recogniser_colour_list,
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
            interactive=True # Keep interactive if data is present
        )

    page_entities_drop_redaction_list = []
    all_pages_in_doc_list = [str(i) for i in range(1, len(page_sizes) + 1)]
    page_entities_drop_redaction_list.extend(all_pages_in_doc_list)

    page_entities_drop_redaction = gr.Dropdown(value = page_dropdown_redaction_value, choices=page_entities_drop_redaction_list, label="Page", allow_custom_value=True)
    
    return (out_image_annotator,
            page_number_reported_gradio_comp,
            page_number_reported_gradio_comp, # Redundant, but matches original return signature
            page_num_reported, # Plain integer value
            recogniser_entities_dropdown_value,
            recogniser_dataframe_out_gr,
            recogniser_dataframe_modified,
            text_entities_drop, # List of text entities for dropdown
            page_entities_drop, # List of page numbers for dropdown
            page_entities_drop_redaction,
            page_sizes, # Updated page_sizes list
            all_image_annotations) # Return the updated full state

def update_all_page_annotation_object_based_on_previous_page(
                                    page_image_annotator_object:AnnotatedImageData,
                                    current_page:int,
                                    previous_page:int,
                                    all_image_annotations:List[AnnotatedImageData],
                                    page_sizes:List[dict]=[],
                                    clear_all:bool=False
                                    ):
    '''
    Overwrite image annotations on the page we are moving from with modifications.
    '''

    if current_page > len(page_sizes):
        raise Warning("Selected page is higher than last page number")
    elif current_page <= 0:
        raise Warning("Selected page is lower than first page")
    

    previous_page_zero_index = previous_page -1
 
    if not current_page: current_page = 1
    
    # This replaces the numpy array image object with the image file path
    page_image_annotator_object, all_image_annotations = replace_annotator_object_img_np_array_with_page_sizes_image_path(all_image_annotations, page_image_annotator_object, page_sizes, previous_page)

    if clear_all == False: all_image_annotations[previous_page_zero_index] = page_image_annotator_object
    else: all_image_annotations[previous_page_zero_index]["boxes"] = []

    return all_image_annotations, current_page, current_page

def apply_redactions_to_review_df_and_files(page_image_annotator_object:AnnotatedImageData,
                     file_paths:List[str],
                     doc:Document,
                     all_image_annotations:List[AnnotatedImageData],
                     current_page:int,
                     review_file_state:pd.DataFrame,
                     output_folder:str = OUTPUT_FOLDER,
                     save_pdf:bool=True,
                     page_sizes:List[dict]=[],
                     COMPRESS_REDACTED_PDF:bool=COMPRESS_REDACTED_PDF,
                     progress=gr.Progress(track_tqdm=True)):
    '''
    Apply modified redactions to a pymupdf and export review files.
    '''

    output_files = []
    output_log_files = []
    pdf_doc = []
    review_df = review_file_state

    page_image_annotator_object = all_image_annotations[current_page - 1]   

    # This replaces the numpy array image object with the image file path
    page_image_annotator_object, all_image_annotations = replace_annotator_object_img_np_array_with_page_sizes_image_path(all_image_annotations, page_image_annotator_object, page_sizes, current_page)
    page_image_annotator_object['image'] = all_image_annotations[current_page - 1]["image"]

    if not page_image_annotator_object:
        print("No image annotations object found for page")
        return doc, all_image_annotations, output_files, output_log_files, review_df
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        file_name_without_ext = get_file_name_without_type(file_path)
        file_name_with_ext = os.path.basename(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()
        
        if save_pdf == True:
            # If working with image docs
            if (is_pdf(file_path) == False) & (file_extension not in '.csv'):
                image = Image.open(file_paths[-1])

                draw = ImageDraw.Draw(image)

                for img_annotation_box in page_image_annotator_object['boxes']:
                    coords = [img_annotation_box["xmin"],
                    img_annotation_box["ymin"],
                    img_annotation_box["xmax"],
                    img_annotation_box["ymax"]]

                    fill = img_annotation_box["color"]

                    # Ensure fill is a valid RGB tuple
                    if isinstance(fill, tuple) and len(fill) == 3:
                        # Check if all elements are integers in the range 0-255
                        if all(isinstance(c, int) and 0 <= c <= 255 for c in fill):
                            pass

                        else:
                            print(f"Invalid color values: {fill}. Defaulting to black.")
                            fill = (0, 0, 0)  # Default to black if invalid
                    else:
                        print(f"Invalid fill format: {fill}. Defaulting to black.")
                        fill = (0, 0, 0)  # Default to black if not a valid tuple

                        # Ensure the image is in RGB mode
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")

                    draw = ImageDraw.Draw(image)

                    draw.rectangle(coords, fill=fill)
                    
                    output_image_path = output_folder + file_name_without_ext + "_redacted.png"
                    image.save(output_folder + file_name_without_ext + "_redacted.png")

                output_files.append(output_image_path)

                doc = [image]

            elif file_extension in '.csv':
                pdf_doc = []

            # If working with pdfs
            elif is_pdf(file_path) == True:
                pdf_doc = pymupdf.open(file_path)
                orig_pdf_file_path = file_path

                output_files.append(orig_pdf_file_path)

                number_of_pages = pdf_doc.page_count
                original_cropboxes = []

                page_sizes_df = pd.DataFrame(page_sizes)
                page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(pd.to_numeric, errors="coerce")

                for i in progress.tqdm(range(0, number_of_pages), desc="Saving redacted pages to file", unit = "pages"):
           
                    image_loc = all_image_annotations[i]['image']

                    # Load in image object
                    if isinstance(image_loc, np.ndarray):
                        image = Image.fromarray(image_loc.astype('uint8'))
                    elif isinstance(image_loc, Image.Image):
                        image = image_loc
                    elif isinstance(image_loc, str):
                        if not os.path.exists(image_loc):
                            image=page_sizes_df.loc[page_sizes_df['page']==i, "image_path"]
                        try:
                            image = Image.open(image_loc)
                        except Exception as e:
                            image = None

                    pymupdf_page = pdf_doc.load_page(i) #doc.load_page(current_page -1)
                    original_cropboxes.append(pymupdf_page.cropbox)
                    pymupdf_page.set_cropbox(pymupdf_page.mediabox)

                    pymupdf_page = redact_page_with_pymupdf(page=pymupdf_page, page_annotations=all_image_annotations[i], image=image, original_cropbox=original_cropboxes[-1], page_sizes_df= page_sizes_df) # image=image,
            else:
                print("File type not recognised.")

            progress(0.9, "Saving output files")

            #try:
            if pdf_doc:
                out_pdf_file_path = output_folder + file_name_without_ext + "_redacted.pdf"
                save_pdf_with_or_without_compression(pdf_doc, out_pdf_file_path, COMPRESS_REDACTED_PDF)
                output_files.append(out_pdf_file_path)

            else:
                print("PDF input not found. Outputs not saved to PDF.")

        # If save_pdf is not true, then add the original pdf to the output files
        else:
            if is_pdf(file_path) == True:                
                orig_pdf_file_path = file_path
                output_files.append(orig_pdf_file_path)

        try:
            #print("Saving review file.")
            review_df = convert_annotation_json_to_review_df(all_image_annotations, review_file_state.copy(), page_sizes=page_sizes)

            page_sizes_df = pd.DataFrame(page_sizes)
            page_sizes_df .loc[:, "page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
            review_df = divide_coordinates_by_page_sizes(review_df, page_sizes_df)

            review_df = review_df[["image",	"page",	"label","color", "xmin", "ymin", "xmax", "ymax", "text", "id"]]

            out_review_file_file_path = output_folder + file_name_with_ext + '_review_file.csv'

            review_df.to_csv(out_review_file_file_path, index=None)
            output_files.append(out_review_file_file_path)

        except Exception as e:
            print("In apply redactions function, could not save annotations to csv file:", e)

    return doc, all_image_annotations, output_files, output_log_files, review_df

def get_boxes_json(annotations:AnnotatedImageData):
    return annotations["boxes"]

def update_all_entity_df_dropdowns(df:pd.DataFrame, label_dropdown_value:str, page_dropdown_value:str, text_dropdown_value:str):
    '''
    Update all dropdowns based on rows that exist in a dataframe
    '''

    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]
    
    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    # if not "ALL" in page_dropdown_value:
    #     filtered_df = filtered_df[filtered_df["page"].astype(str).isin(page_dropdown_value)]
    
    # if not "ALL" in text_dropdown_value:
    #     filtered_df = filtered_df[filtered_df["text"].astype(str).isin(text_dropdown_value)]

    # if not "ALL" in label_dropdown_value:
    #     filtered_df = filtered_df[filtered_df["label"].astype(str).isin(label_dropdown_value)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "label")
    recogniser_entities_drop = gr.Dropdown(value=label_dropdown_value[0], choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)    

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "text")
    text_entities_drop = gr.Dropdown(value=text_dropdown_value[0], choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "page")
    page_entities_drop = gr.Dropdown(value=page_dropdown_value[0], choices=page_entities_for_drop, allow_custom_value=True, interactive=True)

    return recogniser_entities_drop, text_entities_drop, page_entities_drop

def update_entities_df_recogniser_entities(choice:str, df:pd.DataFrame, page_dropdown_value:str, text_dropdown_value:str):
    '''
    Update the rows in a dataframe depending on the user choice from a dropdown
    '''

    if isinstance(choice, str):
        choice = [choice]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]
    
    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    if not "ALL" in page_dropdown_value:
        filtered_df = filtered_df[filtered_df["page"].astype(str).isin(page_dropdown_value)]
    
    if not "ALL" in text_dropdown_value:
        filtered_df = filtered_df[filtered_df["text"].astype(str).isin(text_dropdown_value)]

    if not "ALL" in choice:
        filtered_df = filtered_df[filtered_df["label"].astype(str).isin(choice)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "label")
    recogniser_entities_drop = gr.Dropdown(value=choice[0], choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)    

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "text")
    text_entities_drop = gr.Dropdown(value=text_dropdown_value[0], choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "page")
    page_entities_drop = gr.Dropdown(value=page_dropdown_value[0], choices=page_entities_for_drop, allow_custom_value=True, interactive=True)

    return filtered_df, text_entities_drop, page_entities_drop
    
def update_entities_df_page(choice:str, df:pd.DataFrame, label_dropdown_value:str, text_dropdown_value:str):
    '''
    Update the rows in a dataframe depending on the user choice from a dropdown
    '''
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
    if not "ALL" in text_dropdown_value:
        filtered_df = filtered_df[filtered_df["text"].astype(str).isin(text_dropdown_value)]
    
    if not "ALL" in label_dropdown_value:
        filtered_df = filtered_df[filtered_df["label"].astype(str).isin(label_dropdown_value)]

    if not "ALL" in choice:
        filtered_df = filtered_df[filtered_df["page"].astype(str).isin(choice)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "label")
    recogniser_entities_drop = gr.Dropdown(value=label_dropdown_value[0], choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)    

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "text")
    text_entities_drop = gr.Dropdown(value=text_dropdown_value[0], choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "page")
    page_entities_drop = gr.Dropdown(value=choice[0], choices=page_entities_for_drop, allow_custom_value=True, interactive=True)    

    return filtered_df, recogniser_entities_drop, text_entities_drop
    
def update_redact_choice_df_from_page_dropdown(choice:str, df:pd.DataFrame):
    '''
    Update the rows in a dataframe depending on the user choice from a dropdown
    '''
    if isinstance(choice, str):
        choice = [choice]
    elif not isinstance(choice, list):
        choice = [str(choice)]

    if "index" not in df.columns:
        df["index"] = df.index

    filtered_df = df[["page", "line", "word_text", "word_x0", "word_y0", "word_x1", "word_y1", "index"]].copy()

    # Apply filtering based on dropdown selections
    if not "ALL" in choice:
        filtered_df = filtered_df.loc[filtered_df["page"].astype(str).isin(choice)]

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "page")
    page_entities_drop = gr.Dropdown(value=choice[0], choices=page_entities_for_drop, allow_custom_value=True, interactive=True)    

    return filtered_df
  
def update_entities_df_text(choice:str, df:pd.DataFrame, label_dropdown_value:str, page_dropdown_value:str):
    '''
    Update the rows in a dataframe depending on the user choice from a dropdown
    '''
    if isinstance(choice, str):
        choice = [choice]
    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    if isinstance(page_dropdown_value, str):
        page_dropdown_value = [page_dropdown_value]

    filtered_df = df.copy()

    # Apply filtering based on dropdown selections
    if not "ALL" in page_dropdown_value:
        filtered_df = filtered_df[filtered_df["page"].astype(str).isin(page_dropdown_value)]
    
    if not "ALL" in label_dropdown_value:
        filtered_df = filtered_df[filtered_df["label"].astype(str).isin(label_dropdown_value)]

    if not "ALL" in choice:
        filtered_df = filtered_df[filtered_df["text"].astype(str).isin(choice)]

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "label")
    recogniser_entities_drop = gr.Dropdown(value=label_dropdown_value[0], choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)    

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "text")
    text_entities_drop = gr.Dropdown(value=choice[0], choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(filtered_df, "page")
    page_entities_drop = gr.Dropdown(value=page_dropdown_value[0], choices=page_entities_for_drop, allow_custom_value=True, interactive=True)    

    return filtered_df, recogniser_entities_drop, page_entities_drop
    
def reset_dropdowns(df:pd.DataFrame):
    '''
    Return Gradio dropdown objects with value 'ALL'.
    '''

    recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "label")
    recogniser_entities_drop = gr.Dropdown(value="ALL", choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)    

    text_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "text")
    text_entities_drop = gr.Dropdown(value="ALL", choices=text_entities_for_drop, allow_custom_value=True, interactive=True)

    page_entities_for_drop = update_dropdown_list_based_on_dataframe(df, "page")
    page_entities_drop = gr.Dropdown(value="ALL", choices=page_entities_for_drop, allow_custom_value=True, interactive=True)

    return recogniser_entities_drop, text_entities_drop, page_entities_drop
    
def increase_bottom_page_count_based_on_top(page_number:int):
    return int(page_number)

def df_select_callback_dataframe_row_ocr_with_words(df: pd.DataFrame, evt: gr.SelectData):

        row_value_page = int(evt.row_value[0]) # This is the page number value
        row_value_line = int(evt.row_value[1]) # This is the label number value
        row_value_text = evt.row_value[2] # This is the text number value

        row_value_x0 = evt.row_value[3] # This is the x0 value
        row_value_y0 = evt.row_value[4] # This is the y0 value
        row_value_x1 = evt.row_value[5] # This is the x1 value
        row_value_y1 = evt.row_value[6] # This is the y1 value
        row_value_index = evt.row_value[7] # This is the y1 value

        row_value_df = pd.DataFrame(data={"page":[row_value_page], "line":[row_value_line], "word_text":[row_value_text],
                                          "word_x0":[row_value_x0],	"word_y0":[row_value_y0],	"word_x1":[row_value_x1], "word_y1":[row_value_y1], "index":row_value_index
                                          })

        return row_value_df, row_value_text

def df_select_callback_dataframe_row(df: pd.DataFrame, evt: gr.SelectData):

        row_value_page = int(evt.row_value[0]) # This is the page number value
        row_value_label = evt.row_value[1] # This is the label number value
        row_value_text = evt.row_value[2] # This is the text number value
        row_value_id = evt.row_value[3] # This is the text number value

        row_value_df = pd.DataFrame(data={"page":[row_value_page], "label":[row_value_label], "text":[row_value_text], "id":[row_value_id]})

        return row_value_df, row_value_text

def df_select_callback_textract_api(df: pd.DataFrame, evt: gr.SelectData):

        row_value_job_id = evt.row_value[0] # This is the page number value
        # row_value_label = evt.row_value[1] # This is the label number value
        row_value_job_type = evt.row_value[2] # This is the text number value

        row_value_df = pd.DataFrame(data={"job_id":[row_value_job_id], "label":[row_value_job_type]})

        return row_value_job_id, row_value_job_type, row_value_df

def df_select_callback_cost(df: pd.DataFrame, evt: gr.SelectData):

        row_value_code = evt.row_value[0] # This is the value for cost code
        #row_value_label = evt.row_value[1] # This is the label number value

        #row_value_df = pd.DataFrame(data={"page":[row_value_code], "label":[row_value_label]})

        return row_value_code

def df_select_callback_ocr(df: pd.DataFrame, evt: gr.SelectData):

        row_value_page = int(evt.row_value[0]) # This is the page_number value
        row_value_text = evt.row_value[1] # This is the text contents

        row_value_df = pd.DataFrame(data={"page":[row_value_page], "text":[row_value_text]})

        return row_value_page, row_value_df

# When a user selects a row in the duplicate results table
def store_duplicate_selection(evt: gr.SelectData):
    if not evt.empty:
        selected_index = evt.index[0]
    else:
        selected_index = None
        
    return selected_index

def get_all_rows_with_same_text(df: pd.DataFrame, text: str):
    '''
    Get all rows with the same text as the selected row
    '''
    if text:
        # Get all rows with the same text as the selected row
        return df.loc[df["text"] == text]
    else:
        return pd.DataFrame(columns=["page", "label", "text", "id"])
    
def get_all_rows_with_same_text_redact(df: pd.DataFrame, text: str):
    '''
    Get all rows with the same text as the selected row for redaction tasks
    '''
    if "index" not in df.columns:
        df["index"] = df.index

    if text and not df.empty:
        # Get all rows with the same text as the selected row
        return df.loc[df["word_text"] == text]
    else:
        return pd.DataFrame(columns=["page", "line", "label",  "word_text", "word_x0", "word_y0", "word_x1", "word_y1", "index"])

def update_selected_review_df_row_colour(
    redaction_row_selection: pd.DataFrame,
    review_df: pd.DataFrame,
    previous_id: str = "",
    previous_colour: str = '(0, 0, 0)',
    colour: str = '(1, 0, 255)'
) -> tuple[pd.DataFrame, str, str]:
    '''
    Update the colour of a single redaction box based on the values in a selection row
    (Optimized Version)
    '''

    # Ensure 'color' column exists, default to previous_colour if previous_id is provided
    if "color" not in review_df.columns:
        review_df["color"] = previous_colour if previous_id else '(0, 0, 0)'

    # Ensure 'id' column exists
    if "id" not in review_df.columns:
         # Assuming fill_missing_ids is a defined function that returns a DataFrame
         # It's more efficient if this is handled outside if possible,
         # or optimized internally.
         print("Warning: 'id' column not found. Calling fill_missing_ids.")
         review_df = fill_missing_ids(review_df) # Keep this if necessary, but note it can be slow

    # --- Optimization 1 & 2: Reset existing highlight colours using vectorized assignment ---
    # Reset the color of the previously highlighted row
    if previous_id and previous_id in review_df["id"].values:
         review_df.loc[review_df["id"] == previous_id, "color"] = previous_colour

    # Reset the color of any row that currently has the highlight colour (handle cases where previous_id might not have been tracked correctly)
    # Convert to string for comparison only if the dtype might be mixed or not purely string
    # If 'color' is consistently string, the .astype(str) might be avoidable.
    # Assuming color is consistently string format like '(R, G, B)'
    review_df.loc[review_df["color"] == colour, "color"] = '(0, 0, 0)'


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
            how="inner" # Use inner join as we only care about matches
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
                 review_df.loc[review_df["id"].isin(merged_reviews["id"]), "color"] = colour
            else:
                 # More general case using multiple columns - might be slower
                 # Create a temporary key for comparison
                 def create_merge_key(df, cols):
                     return df[cols].astype(str).agg('_'.join, axis=1)

                 review_df_key = create_merge_key(review_df, selected_merge_cols)
                 merged_reviews_key = create_merge_key(merged_reviews, selected_merge_cols)

                 review_df.loc[review_df_key.isin(merged_reviews_key), "color"] = colour

            previous_colour = new_previous_colour
            previous_id = new_previous_id
        else:
             # No rows matched the selection
             print("No reviews found matching selection criteria")
             # The reset logic at the beginning already handles setting color to (0, 0, 0)
             # if it was the highlight colour and didn't match.
             # No specific action needed here for color reset beyond what's done initially.
             previous_colour = '(0, 0, 0)' # Reset previous_colour as no row was highlighted
             previous_id = '' # Reset previous_id

    else:
         # If selection is empty, reset any existing highlights
         review_df.loc[review_df["color"] == colour, "color"] = '(0, 0, 0)'
         previous_colour = '(0, 0, 0)'
         previous_id = ''


    # Ensure column order is maintained if necessary, though pandas generally preserves order
    # Creating a new DataFrame here might involve copying data, consider if this is strictly needed.
    if set(["image", "page", "label", "color", "xmin","ymin", "xmax", "ymax", "text", "id"]).issubset(review_df.columns):
        review_df = review_df[["image", "page", "label", "color", "xmin","ymin", "xmax", "ymax", "text", "id"]]
    else:
         print("Warning: Not all expected columns are present in review_df for reordering.")


    return review_df, previous_id, previous_colour

def update_boxes_color(images: list, redaction_row_selection: pd.DataFrame, colour: tuple = (0, 255, 0)):
    """
    Update the color of bounding boxes in the images list based on redaction_row_selection.
    
    Parameters:
    - images (list): List of dictionaries containing image paths and box metadata.
    - redaction_row_selection (pd.DataFrame): DataFrame with 'page', 'label', and optionally 'text' columns.
    - colour (tuple): RGB tuple for the new color.
    
    Returns:
    - Updated list with modified colors.
    """
    # Convert DataFrame to a set for fast lookup
    selection_set = set(zip(redaction_row_selection["page"], redaction_row_selection["label"]))

    for page_idx, image_obj in enumerate(images):
        if "boxes" in image_obj:
            for box in image_obj["boxes"]:
                if (page_idx, box["label"]) in selection_set:
                    box["color"] = colour  # Update color
    
    return images

def update_other_annotator_number_from_current(page_number_first_counter:int):
    return page_number_first_counter

def convert_image_coords_to_adobe(pdf_page_width:float, pdf_page_height:float, image_width:float, image_height:float, x1:float, y1:float, x2:float, y2:float):
    '''
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
    '''

    
    
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

def convert_pymupdf_coords_to_adobe(x1: float, y1: float, x2: float, y2: float, pdf_page_height: float):
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

def create_xfdf(review_file_df:pd.DataFrame, pdf_path:str, pymupdf_doc:object, image_paths:List[str]=[], document_cropboxes:List=[], page_sizes:List[dict]=[]):
    '''
    Create an xfdf file from a review csv file and a pdf
    '''
    xfdf_root = Element('xfdf', xmlns="http://ns.adobe.com/xfdf/", **{'xml:space':"preserve"})
    annots = SubElement(xfdf_root, 'annots')

    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        if not page_sizes_df.empty and "mediabox_width" not in review_file_df.columns:
            review_file_df = review_file_df.merge(page_sizes_df, how="left", on="page")
        if "xmin" in review_file_df.columns and review_file_df["xmin"].max() <= 1:
            if "mediabox_width" in review_file_df.columns and "mediabox_height" in review_file_df.columns:
                review_file_df["xmin"] = review_file_df["xmin"] * review_file_df["mediabox_width"]
                review_file_df["xmax"] = review_file_df["xmax"] * review_file_df["mediabox_width"]
                review_file_df["ymin"] = review_file_df["ymin"] * review_file_df["mediabox_height"]
                review_file_df["ymax"] = review_file_df["ymax"] * review_file_df["mediabox_height"]
        elif "image_width" in review_file_df.columns and not page_sizes_df.empty :
            review_file_df = multiply_coordinates_by_page_sizes(review_file_df, page_sizes_df, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")

    for _, row in review_file_df.iterrows():
        page_num_reported = int(row["page"])
        page_python_format = page_num_reported - 1
        pymupdf_page = pymupdf_doc.load_page(page_python_format)

        if document_cropboxes and page_python_format < len(document_cropboxes):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", document_cropboxes[page_python_format])
            if match and len(match) == 4:
                rect_values = list(map(float, match))
                pymupdf_page.set_cropbox(Rect(*rect_values))

        pdf_page_height = pymupdf_page.mediabox.height
        redact_annot = SubElement(annots, 'redact')
        redact_annot.set('opacity', "0.500000")
        redact_annot.set('interior-color', "#000000")

        now = datetime.now(timezone(timedelta(hours=1))) # Consider making tz configurable or UTC
        date_str = now.strftime("D:%Y%m%d%H%M%S") + now.strftime("%z")[:3] + "'" + now.strftime("%z")[3:] + "'"
        redact_annot.set('date', date_str)

        annot_id = str(uuid.uuid4())
        redact_annot.set('name', annot_id)
        redact_annot.set('page', str(page_python_format))
        redact_annot.set('mimetype', "Form")

        x1_pdf, y1_pdf, x2_pdf, y2_pdf = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        adobe_x1, adobe_y1, adobe_x2, adobe_y2 = convert_pymupdf_coords_to_adobe(
            x1_pdf, y1_pdf, x2_pdf, y2_pdf, pdf_page_height
        )
        redact_annot.set('rect', f"{adobe_x1:.6f},{adobe_y1:.6f},{adobe_x2:.6f},{adobe_y2:.6f}")

        redact_annot.set('subject', str(row['label'])) # Changed from row['text'] to row['label']
        redact_annot.set('title', str(row.get('label', 'Unknown'))) # Fallback for title

        contents_richtext = SubElement(redact_annot, 'contents-richtext')
        body_attrs = {
            'xmlns': "http://www.w3.org/1999/xhtml",
            '{http://www.xfa.org/schema/xfa-data/1.0/}APIVersion': "Acrobat:25.1.0",
            '{http://www.xfa.org/schema/xfa-data/1.0/}spec': "2.0.2"
        }
        body = SubElement(contents_richtext, 'body', attrib=body_attrs)
        p_element = SubElement(body, 'p', dir="ltr")
        span_attrs = {
            'dir': "ltr",
            'style': "font-size:10.0pt;text-align:left;color:#000000;font-weight:normal;font-style:normal"
        }
        span_element = SubElement(p_element, 'span', attrib=span_attrs)
        span_element.text = str(row['text']).strip() # Added .strip()

        pdf_ops_for_black_fill_and_outline = [
            "1 w",                             # 1. Set line width to 1 point for the stroke
            "0 g",                             # 2. Set NON-STROKING (fill) color to black
            "0 G",                             # 3. Set STROKING (outline) color to black
            "1 0 0 1 0 0 cm",                  # 4. CTM (using absolute page coordinates)
            f"{adobe_x1:.2f} {adobe_y1:.2f} m",  # 5. Path definition: move to start
            f"{adobe_x2:.2f} {adobe_y1:.2f} l",  # line
            f"{adobe_x2:.2f} {adobe_y2:.2f} l",  # line
            f"{adobe_x1:.2f} {adobe_y2:.2f} l",  # line
            "h",                               # 6. Close the path (creates the last line back to start)
            "B"                                # 7. Fill AND Stroke the path using non-zero winding rule
        ]
        data_content_string = "\n".join(pdf_ops_for_black_fill_and_outline) + "\n"
        data_element = SubElement(redact_annot, 'data')
        data_element.set('MODE', "filtered")
        data_element.set('encoding', "ascii")
        data_element.set('length', str(len(data_content_string.encode('ascii'))))
        data_element.text = data_content_string

    rough_string = tostring(xfdf_root, encoding='unicode', method='xml')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toxml() #.toprettyxml(indent="  ")

def convert_df_to_xfdf(input_files:List[str], pdf_doc:Document, image_paths:List[str], output_folder:str = OUTPUT_FOLDER, document_cropboxes:List=[], page_sizes:List[dict]=[]):
    '''
    Load in files to convert a review file into an Adobe comment file format
    '''
    output_paths = []
    pdf_name = ""
    file_path_name = ""

    if isinstance(input_files, str):
        file_paths_list = [input_files]
    else:
        file_paths_list = input_files

    # Sort the file paths so that the pdfs come first
    file_paths_list = sorted(file_paths_list, key=lambda x: (os.path.splitext(x)[1] != '.pdf', os.path.splitext(x)[1] != '.json')) 
    
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
            if 'text' in review_file_df.columns:  review_file_df['text'] = review_file_df['text'].fillna('')  
            if 'label' in review_file_df.columns: review_file_df['label'] = review_file_df['label'].fillna('')

            xfdf_content = create_xfdf(review_file_df, pdf_name, pdf_doc, image_paths, document_cropboxes, page_sizes)

            output_path = output_folder + file_path_name + "_adobe.xfdf"        
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xfdf_content)

            output_paths.append(output_path)

    return output_paths

### Convert xfdf coordinates back to image for app

def convert_adobe_coords_to_image(pdf_page_width:float, pdf_page_height:float, image_width:float, image_height:float, x1:float, y1:float, x2:float, y2:float):
    '''
    Converts coordinates from Adobe PDF space to image space.
    
    Parameters:
    - pdf_page_width: Width of the PDF page
    - pdf_page_height: Height of the PDF page
    - image_width: Width of the source image
    - image_height: Height of the source image
    - x1, y1, x2, y2: Coordinates in Adobe PDF space
    
    Returns:
    - Tuple of converted coordinates (x1, y1, x2, y2) in image space
    '''
    
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

def parse_xfdf(xfdf_path:str):
    '''
    Parse the XFDF file and extract redaction annotations.
    
    Parameters:
    - xfdf_path: Path to the XFDF file
    
    Returns:
    - List of dictionaries containing redaction information
    '''
    tree = parse(xfdf_path)
    root = tree.getroot()
    
    # Define the namespace
    namespace = {'xfdf': 'http://ns.adobe.com/xfdf/'}
    
    redactions = []
    
    # Find all redact elements using the namespace
    for redact in root.findall('.//xfdf:redact', namespaces=namespace):

        redaction_info = {
            'image': '', # Image will be filled in later
            'page': int(redact.get('page')) + 1,  # Convert to 1-based index
            'xmin': float(redact.get('rect').split(',')[0]),
            'ymin': float(redact.get('rect').split(',')[1]),
            'xmax': float(redact.get('rect').split(',')[2]),
            'ymax': float(redact.get('rect').split(',')[3]),
            'label': redact.get('title'),
            'text': redact.get('contents'),
            'color': redact.get('border-color', '(0, 0, 0)')  # Default to black if not specified
        }
        redactions.append(redaction_info)
    
    return redactions

def convert_xfdf_to_dataframe(file_paths_list:List[str], pymupdf_doc, image_paths:List[str], output_folder:str=OUTPUT_FOLDER):
    '''
    Convert redaction annotations from XFDF and associated images into a DataFrame.
    
    Parameters:
    - xfdf_path: Path to the XFDF file
    - pdf_doc: PyMuPDF document object
    - image_paths: List of PIL Image objects corresponding to PDF pages
    
    Returns:
    - DataFrame containing redaction information
    '''
    output_paths = []
    xfdf_paths = []
    df = pd.DataFrame()

    # Sort the file paths so that the pdfs come first
    file_paths_list = sorted(file_paths_list, key=lambda x: (os.path.splitext(x)[1] != '.pdf', os.path.splitext(x)[1] != '.json'))
    
    for file in file_paths_list:

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name
    
        file_path_name = get_file_name_without_type(file_path)
        file_path_end = detect_file_type(file_path)

        if file_path_end == "pdf":
            pdf_name = os.path.basename(file_path)

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

            df.fillna('', inplace=True)  # Replace NaN with an empty string

            for _, row in df.iterrows():
                page_python_format = int(row["page"])-1

                pymupdf_page = pymupdf_doc.load_page(page_python_format)

                pdf_page_height = pymupdf_page.rect.height
                pdf_page_width = pymupdf_page.rect.width 

                image_path = image_paths[page_python_format]

                if isinstance(image_path, str):
                    image = Image.open(image_path)

                image_page_width, image_page_height = image.size

                # Convert to image coordinates
                image_x1, image_y1, image_x2, image_y2 = convert_adobe_coords_to_image(pdf_page_width, pdf_page_height, image_page_width, image_page_height, row['xmin'], row['ymin'], row['xmax'], row['ymax'])

                df.loc[_, ['xmin', 'ymin', 'xmax', 'ymax']] = [image_x1, image_y1, image_x2, image_y2]
            
                # Optionally, you can add the image path or other relevant information
                df.loc[_, 'image'] = image_path

    out_file_path = output_folder + file_path_name + "_review_file.csv"
    df.to_csv(out_file_path, index=None)

    output_paths.append(out_file_path)
    
    return output_paths