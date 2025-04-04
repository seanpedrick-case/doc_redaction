import os
import re
import gradio as gr
import pandas as pd
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring, parse
from xml.dom import minidom
import uuid
from typing import List
from gradio_image_annotation import image_annotator
from gradio_image_annotation.image_annotator import AnnotatedImageData
from pymupdf import Document, Rect
import pymupdf
#from fitz 
from PIL import ImageDraw, Image

from tools.config import OUTPUT_FOLDER, CUSTOM_BOX_COLOUR, MAX_IMAGE_PIXELS, INPUT_FOLDER
from tools.file_conversion import is_pdf, convert_annotation_json_to_review_df, convert_review_df_to_annotation_json, process_single_page_for_image_conversion, multiply_coordinates_by_page_sizes, convert_annotation_data_to_dataframe, create_annotation_dicts_from_annotation_df, remove_duplicate_images_with_blank_boxes
from tools.helper_functions import get_file_name_without_type,  detect_file_type
from tools.file_redaction import redact_page_with_pymupdf

if not MAX_IMAGE_PIXELS: Image.MAX_IMAGE_PIXELS = None

def decrease_page(number:int):
    '''
    Decrease page number for review redactions page.
    '''
    if number > 1:
        return number - 1, number - 1
    else:
        return 1, 1

def increase_page(number:int, page_image_annotator_object:AnnotatedImageData):
    '''
    Increase page number for review redactions page.
    '''

    if not page_image_annotator_object:
        return 1, 1

    max_pages = len(page_image_annotator_object)

    if number < max_pages:
        return number + 1, number + 1
    else:
        return max_pages, max_pages

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

        recogniser_dataframe_out_gr = gr.Dataframe(review_dataframe[["page", "label", "text"]], show_search="filter", col_count=(3, "fixed"), type="pandas", headers=["page", "label", "text"], show_fullscreen_button=True, wrap=True)

        recogniser_dataframe_out = review_dataframe[["page", "label", "text"]]

    except Exception as e:
        print("Could not extract recogniser information:", e)
        recogniser_dataframe_out = recogniser_dataframe_base[["page", "label", "text"]]

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

        recogniser_dataframe_out_gr = gr.Dataframe(review_dataframe[["page", "label", "text"]], show_search="filter", col_count=(3, "fixed"), type="pandas", headers=["page", "label", "text"], show_fullscreen_button=True, wrap=True)
        
        recogniser_entities_for_drop = update_dropdown_list_based_on_dataframe(recogniser_dataframe_out, "label")
        recogniser_entities_drop = gr.Dropdown(value=recogniser_entities_dropdown_value, choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)

        recogniser_entities_list_base = recogniser_dataframe_out["label"].astype(str).unique().tolist()

        # Recogniser entities list is the list of choices that appear when you make a new redaction box
        recogniser_entities_list = [entity for entity in recogniser_entities_list_base if entity != 'Redaction']
        recogniser_entities_list.insert(0, 'Redaction')

    return recogniser_entities_list, recogniser_dataframe_out_gr, recogniser_dataframe_out, recogniser_entities_drop, text_entities_drop, page_entities_drop

def undo_last_removal(backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base):
    return backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base

def update_annotator_page_from_review_df(review_df: pd.DataFrame,
                                          image_file_paths:List[str],
                                          page_sizes:List[dict],
                                          current_page:int,
                                          previous_page:int,
                                          current_image_annotations_state:List[str],
                                          current_page_annotator:object):
    '''
    Update the visible annotation object with the latest review file information
    '''
    out_image_annotations_state = current_image_annotations_state
    out_current_page_annotator = current_page_annotator

    print("page_sizes:", page_sizes)

    review_df.to_csv(OUTPUT_FOLDER + "review_df_in_update_annotator.csv")

    if not review_df.empty:

        out_image_annotations_state = convert_review_df_to_annotation_json(review_df, image_file_paths, page_sizes)

        print("out_image_annotations_state[current_page-1]:", out_image_annotations_state[current_page-1])

        if previous_page == current_page:
            out_current_page_annotator = out_image_annotations_state[current_page-1]

    return out_current_page_annotator, out_image_annotations_state




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
        # Ensure selected_rows_df has the same relevant columns
        selected_subset = selected_rows_df[['label', 'page', 'text']].drop_duplicates(subset=['label', 'page', 'text'])

        # Perform anti-join using merge with an indicator column
        merged_df = review_df.merge(selected_subset, on=['label', 'page', 'text'], how='left', indicator=True)
        
        # Keep only the rows that do not have a match in selected_rows_df
        out_review_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        out_image_annotations_state = convert_review_df_to_annotation_json(out_review_df, image_file_paths, page_sizes)

        out_recogniser_entity_dataframe_base = out_review_df[["page", "label", "text"]]
    
    # Either there is nothing left in the selection dataframe, or the review dataframe
    else:
        out_review_df = review_df
        out_recogniser_entity_dataframe_base = recogniser_entity_dataframe_base

        out_image_annotations_state = image_annotations_state

    return out_review_df, out_image_annotations_state, out_recogniser_entity_dataframe_base, backup_review_state, backup_image_annotations_state, backup_recogniser_entity_dataframe_base

def update_annotator_object_and_filter_df(
                    all_image_annotations:List[AnnotatedImageData],
                    gradio_annotator_current_page_number:int,
                    recogniser_entities_dropdown_value:str="ALL",
                    page_dropdown_value:str="ALL",
                    text_dropdown_value:str="ALL",
                    recogniser_dataframe_base:gr.Dataframe=gr.Dataframe(pd.DataFrame(data={"page":[], "label":[], "text":[]}), type="pandas", headers=["page", "label", "text"], show_fullscreen_button=True, wrap=True),
                    zoom:int=100,
                    review_df:pd.DataFrame=[],
                    page_sizes:List[dict]=[],
                    doc_full_file_name_textbox:str='',
                    input_folder:str=INPUT_FOLDER):
    '''
    Update a gradio_image_annotation object with new annotation data.
    '''
    zoom_str = str(zoom) + '%'
    
    if not gradio_annotator_current_page_number: gradio_annotator_current_page_number = 0

    # Check bounding values for current page and page max
    if gradio_annotator_current_page_number > 0: page_num_reported = gradio_annotator_current_page_number
    elif gradio_annotator_current_page_number == 0: page_num_reported = 1 # minimum possible reported page is 1
    else: 
        gradio_annotator_current_page_number = 0
        page_num_reported = 1

    # Ensure page displayed can't exceed number of pages in document
    page_max_reported = len(all_image_annotations)
    if page_num_reported > page_max_reported: page_num_reported = page_max_reported

    page_num_reported_zero_indexed = page_num_reported - 1

    # First, check that the image on the current page is valid, replace with what exists in page_sizes object if not
    page_image_annotator_object, all_image_annotations = replace_images_in_image_annotation_object(all_image_annotations, all_image_annotations[page_num_reported_zero_indexed], page_sizes, page_num_reported)    

    all_image_annotations[page_num_reported_zero_indexed] = page_image_annotator_object
    
    current_image_path = all_image_annotations[page_num_reported_zero_indexed]['image']

    # If image path is still not valid, load in a new image an overwrite it. Then replace all items in the image annotation object for all pages based on the updated information.
    page_sizes_df = pd.DataFrame(page_sizes)

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

    if review_df.empty: review_df = pd.DataFrame(columns=["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text"])

    ##
    
    review_df.loc[review_df["page"]==page_num_reported, 'image'] = replaced_image_path

    # Update dropdowns and review selection dataframe with the updated annotator object
    recogniser_entities_list, recogniser_dataframe_out_gr, recogniser_dataframe_modified, recogniser_entities_dropdown_value, text_entities_drop, page_entities_drop = update_recogniser_dataframes(all_image_annotations, recogniser_dataframe_base, recogniser_entities_dropdown_value, text_dropdown_value, page_dropdown_value, review_df.copy(), page_sizes)
    
    recogniser_colour_list = [(0, 0, 0) for _ in range(len(recogniser_entities_list))]

    # page_sizes_df has been changed - save back to page_sizes_object
    page_sizes = page_sizes_df.to_dict(orient='records')

    images_list = list(page_sizes_df["image_path"])
    images_list[page_num_reported_zero_indexed] = replaced_image_path

    all_image_annotations[page_num_reported_zero_indexed]['image'] = replaced_image_path
    
    # Multiply out image_annotation coordinates from relative to absolute if necessary
    all_image_annotations_df = convert_annotation_data_to_dataframe(all_image_annotations)

    all_image_annotations_df = multiply_coordinates_by_page_sizes(all_image_annotations_df, page_sizes_df, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")   

    all_image_annotations = create_annotation_dicts_from_annotation_df(all_image_annotations_df, page_sizes)

    # Remove blank duplicate entries
    all_image_annotations = remove_duplicate_images_with_blank_boxes(all_image_annotations)

    current_page_image_annotator_object = all_image_annotations[page_num_reported_zero_indexed]

    page_number_reported_gradio = gr.Number(label = "Current page", value=page_num_reported, precision=0)
    
    ###
    # If no data, present a blank page
    if not all_image_annotations:
        print("No all_image_annotation object found")
        page_num_reported = 1

        out_image_annotator = image_annotator(
        value = None,
        boxes_alpha=0.1,
        box_thickness=1,
        label_list=recogniser_entities_list,
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
        interactive=True
        )        

        return out_image_annotator, page_number_reported_gradio, page_number_reported_gradio, page_num_reported, recogniser_entities_dropdown_value, recogniser_dataframe_out_gr, recogniser_dataframe_modified, text_entities_drop, page_entities_drop, page_sizes, all_image_annotations
    
    else:
        ### Present image_annotator outputs
        out_image_annotator = image_annotator(
            value = current_page_image_annotator_object,
            boxes_alpha=0.1,
            box_thickness=1,
            label_list=recogniser_entities_list,
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
            interactive=True
        )

    #print("all_image_annotations at end of update_annotator...:", all_image_annotations)
    #print("review_df at end of update_annotator_object:", review_df)

    return out_image_annotator, page_number_reported_gradio, page_number_reported_gradio, page_num_reported, recogniser_entities_dropdown_value, recogniser_dataframe_out_gr, recogniser_dataframe_modified, text_entities_drop, page_entities_drop, page_sizes, all_image_annotations

def replace_images_in_image_annotation_object(
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

    previous_page_zero_index = previous_page -1
 
    if not current_page: current_page = 1

    #print("page_image_annotator_object at start of update_all_page_annotation_object:", page_image_annotator_object)
      
    page_image_annotator_object, all_image_annotations = replace_images_in_image_annotation_object(all_image_annotations, page_image_annotator_object, page_sizes, previous_page)

    #print("page_image_annotator_object after replace_images in update_all_page_annotation_object:", page_image_annotator_object)

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
                     progress=gr.Progress(track_tqdm=True)):
    '''
    Apply modified redactions to a pymupdf and export review files
    '''

    output_files = []
    output_log_files = []
    pdf_doc = []
    review_df = review_file_state

    page_image_annotator_object = all_image_annotations[current_page - 1]   

    # This replaces the numpy array image object with the image file path
    page_image_annotator_object, all_image_annotations = replace_images_in_image_annotation_object(all_image_annotations, page_image_annotator_object, page_sizes, current_page)
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
                            #print("fill:", fill)
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
                #print("This is a csv")
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

                for i in progress.tqdm(range(0, number_of_pages), desc="Saving redactions to file", unit = "pages"):
           
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
                    original_cropboxes.append(pymupdf_page.cropbox.irect)
                    pymupdf_page.set_cropbox = pymupdf_page.mediabox

                    pymupdf_page = redact_page_with_pymupdf(page=pymupdf_page, page_annotations=all_image_annotations[i], image=image, original_cropbox=original_cropboxes[-1], page_sizes_df= page_sizes_df) # image=image,
            else:
                print("File type not recognised.")
                    
            #try:
            if pdf_doc:
                out_pdf_file_path = output_folder + file_name_without_ext + "_redacted.pdf"
                pdf_doc.save(out_pdf_file_path, garbage=4, deflate=True, clean=True)
                output_files.append(out_pdf_file_path)

            else:
                print("PDF input not found. Outputs not saved to PDF.")

        # If save_pdf is not true, then add the original pdf to the output files
        else:
            if is_pdf(file_path) == True:                
                orig_pdf_file_path = file_path
                output_files.append(orig_pdf_file_path)

        try:
            review_df = convert_annotation_json_to_review_df(all_image_annotations, review_file_state.copy(), page_sizes=page_sizes)[["image",	"page",	"label","color", "xmin", "ymin", "xmax", "ymax", "text"]]#.drop_duplicates(subset=["image",	"page",	"text",	"label","color", "xmin", "ymin", "xmax", "ymax"])
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
    if isinstance(label_dropdown_value, str):
        label_dropdown_value = [label_dropdown_value]
    if isinstance(text_dropdown_value, str):
        text_dropdown_value = [text_dropdown_value]

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
    
def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):

        row_value_page = evt.row_value[0] # This is the page number value
        row_value_label = evt.row_value[1] # This is the label number value
        row_value_text = evt.row_value[2] # This is the text number value

        row_value_df = pd.DataFrame(data={"page":[row_value_page], "label":[row_value_label], "text":[row_value_text]})

        return row_value_page, row_value_df

def df_select_callback_cost(df: pd.DataFrame, evt: gr.SelectData):

        row_value_code = evt.row_value[0] # This is the value for cost code
        row_value_label = evt.row_value[1] # This is the label number value

        #row_value_df = pd.DataFrame(data={"page":[row_value_code], "label":[row_value_label]})

        return row_value_code

def update_selected_review_df_row_colour(redaction_row_selection:pd.DataFrame, review_df:pd.DataFrame, colour:tuple=(0,0,255)):
    '''
    Update the colour of a single redaction box based on the values in a selection row
    '''
    colour_tuple =  str(tuple(colour))

    if "color" not in review_df.columns: review_df["color"] = None

    # Reset existing highlight colours
    review_df.loc[review_df["color"]==colour_tuple, "color"] = review_df.loc[review_df["color"]==colour_tuple, "color"].apply(lambda _: '(0, 0, 0)')

    review_df = review_df.merge(redaction_row_selection, on=["page", "label", "text"], indicator=True, how="left")
    review_df.loc[review_df["_merge"]=="both", "color"] =  review_df.loc[review_df["_merge"] == "both", "color"].apply(lambda _: '(0, 0, 255)')

    review_df.drop("_merge", axis=1, inplace=True)

    review_df.to_csv(OUTPUT_FOLDER + "review_df_in_update_selected_review.csv")

    return review_df

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

def create_xfdf(review_file_df:pd.DataFrame, pdf_path:str, pymupdf_doc:object, image_paths:List[str], document_cropboxes:List=[], page_sizes:List[dict]=[]):
    '''
    Create an xfdf file from a review csv file and a pdf
    '''
    pages_are_images = True

    # Create root element
    xfdf = Element('xfdf', xmlns="http://ns.adobe.com/xfdf/", xml_space="preserve")
    
    # Add header
    header = SubElement(xfdf, 'header')
    header.set('pdf-filepath', pdf_path)
    
    # Add annots
    annots = SubElement(xfdf, 'annots')

    # Check if page size object exists, and if current coordinates are in relative format or image coordinates format.
    if page_sizes:        
        page_sizes_df = pd.DataFrame(page_sizes)

        # If there are no image coordinates, then convert coordinates to pymupdf coordinates prior to export
        #if len(page_sizes_df.loc[page_sizes_df["image_width"].isnull(),"image_width"]) == len(page_sizes_df["image_width"]):
        print("Using pymupdf coordinates for conversion.")

        pages_are_images = False

        if "mediabox_width" not in review_file_df.columns:            
                review_file_df = review_file_df.merge(page_sizes_df, how="left", on = "page")
        
        # If all coordinates are less or equal to one, this is a relative page scaling - change back to image coordinates
        if review_file_df["xmin"].max() <= 1 and review_file_df["xmax"].max() <= 1 and review_file_df["ymin"].max() <= 1 and review_file_df["ymax"].max() <= 1:
            review_file_df["xmin"] = review_file_df["xmin"] * review_file_df["mediabox_width"]
            review_file_df["xmax"] = review_file_df["xmax"] * review_file_df["mediabox_width"]
            review_file_df["ymin"] = review_file_df["ymin"] * review_file_df["mediabox_height"]
            review_file_df["ymax"] = review_file_df["ymax"] * review_file_df["mediabox_height"]

        # If all nulls, then can do image coordinate conversion
        if len(page_sizes_df.loc[page_sizes_df["mediabox_width"].isnull(),"mediabox_width"]) == len(page_sizes_df["mediabox_width"]):

            pages_are_images = True

            review_file_df = multiply_coordinates_by_page_sizes(review_file_df, page_sizes_df, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")

            # if "image_width" not in review_file_df.columns:            
            #         review_file_df = review_file_df.merge(page_sizes_df, how="left", on = "page")
            
            # # If all coordinates are less or equal to one, this is a relative page scaling - change back to image coordinates
            # if review_file_df["xmin"].max() <= 1 and review_file_df["xmax"].max() <= 1 and review_file_df["ymin"].max() <= 1 and review_file_df["ymax"].max() <= 1:
            #     review_file_df["xmin"] = review_file_df["xmin"] * review_file_df["image_width"]
            #     review_file_df["xmax"] = review_file_df["xmax"] * review_file_df["image_width"]
            #     review_file_df["ymin"] = review_file_df["ymin"] * review_file_df["image_height"]
            #     review_file_df["ymax"] = review_file_df["ymax"] * review_file_df["image_height"]

                
    
    # Go through each row of the review_file_df, create an entry in the output Adobe xfdf file.
    for _, row in review_file_df.iterrows():
        page_num_reported = row["page"]
        page_python_format = int(row["page"])-1

        pymupdf_page = pymupdf_doc.load_page(page_python_format)

        # Load cropbox sizes. Set cropbox to the original cropbox sizes from when the document was loaded into the app.
        if document_cropboxes:

            # Extract numbers safely using regex
            match = re.findall(r"[-+]?\d*\.\d+|\d+", document_cropboxes[page_python_format])

            if match and len(match) == 4:
                rect_values = list(map(float, match))  # Convert extracted strings to floats
                pymupdf_page.set_cropbox(Rect(*rect_values))
            else:
                raise ValueError(f"Invalid cropbox format: {document_cropboxes[page_python_format]}")
        else:
            print("Document cropboxes not found.")

        
        pdf_page_height = pymupdf_page.mediabox.height
        pdf_page_width = pymupdf_page.mediabox.width

        # Check if image dimensions for page exist in page_sizes_df
        # image_dimensions = {}

        # image_dimensions['image_width'] = page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_width"].max()
        # image_dimensions['image_height'] = page_sizes_df.loc[page_sizes_df['page']==page_num_reported, "image_height"].max()

        # if pd.isna(image_dimensions['image_width']):
        #     image_dimensions = {}

        # image = image_paths[page_python_format]

        # if image_dimensions:
        #     image_page_width, image_page_height = image_dimensions["image_width"], image_dimensions["image_height"]
        # if isinstance(image, str) and 'placeholder' not in image:
        #     image = Image.open(image)
        #     image_page_width, image_page_height = image.size
        # else:
        #     try:
        #         image = Image.open(image)
        #         image_page_width, image_page_height = image.size
        #     except Exception as e:
        #         print("Could not get image sizes due to:", e)        

        # Create redaction annotation
        redact_annot = SubElement(annots, 'redact')
        
        # Generate unique ID
        annot_id = str(uuid.uuid4())
        redact_annot.set('name', annot_id)
        
        # Set page number (subtract 1 as PDF pages are 0-based)
        redact_annot.set('page', str(int(row['page']) - 1))
        
        # # Convert coordinates
        # if pages_are_images == True:
        #     x1, y1, x2, y2 = convert_image_coords_to_adobe(
        #         pdf_page_width,
        #         pdf_page_height,
        #         image_page_width,
        #         image_page_height,
        #         row['xmin'],
        #         row['ymin'],
        #         row['xmax'],
        #         row['ymax']
        #     )
        # else:
        x1, y1, x2, y2 = convert_pymupdf_coords_to_adobe(row['xmin'],
            row['ymin'],
            row['xmax'],
            row['ymax'], pdf_page_height)

        if CUSTOM_BOX_COLOUR == "grey":
            colour_str = "0.5,0.5,0.5"        
        else:
            colour_str = row['color'].strip('()').replace(' ', '')
        
        # Set coordinates
        redact_annot.set('rect', f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}")
        
        # Set redaction properties
        redact_annot.set('title', row['label'])  # The type of redaction (e.g., "PERSON")
        redact_annot.set('contents', row['text'])  # The redacted text
        redact_annot.set('subject', row['label'])  # The redacted text
        redact_annot.set('mimetype', "Form")
        
        # Set appearance properties
        redact_annot.set('border-color', colour_str)  # Black border
        redact_annot.set('repeat', 'false')
        redact_annot.set('interior-color', colour_str)
        #redact_annot.set('fill-color', colour_str)
        #redact_annot.set('outline-color', colour_str)
        #redact_annot.set('overlay-color', colour_str)
        #redact_annot.set('overlay-text', row['label'])
        redact_annot.set('opacity', "0.5")

        # Add appearance dictionary
        # appearanceDict = SubElement(redact_annot, 'appearancedict')
        
        # # Normal appearance
        # normal = SubElement(appearanceDict, 'normal')
        # #normal.set('appearance', 'redact')
                
        # # Color settings for the mark (before applying redaction)
        # markAppearance = SubElement(redact_annot, 'markappearance')
        # markAppearance.set('stroke-color', colour_str)  # Red outline
        # markAppearance.set('fill-color', colour_str)    # Light red fill
        # markAppearance.set('opacity', '0.5')          # 50% opacity
        
        # # Final redaction appearance (after applying)
        # redactAppearance = SubElement(redact_annot, 'redactAppearance')
        # redactAppearance.set('fillColor', colour_str)  # Black fill
        # redactAppearance.set('fontName', 'Helvetica')
        # redactAppearance.set('fontSize', '12')
        # redactAppearance.set('textAlignment', 'left')
        # redactAppearance.set('textColor', colour_str)  # White text
    
    # Convert to pretty XML string
    xml_str = minidom.parseString(tostring(xfdf)).toprettyxml(indent="  ")
    
    return xml_str

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

        if file_path_end == "csv":
            # If no pdf name, just get the name of the file path
            if not pdf_name:
                pdf_name = file_path_name
            # Read CSV file
            review_file_df = pd.read_csv(file_path)

            review_file_df.fillna('', inplace=True)  # Replace NaN in review file with an empty string

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

                #print('row:', row)

    out_file_path = output_folder + file_path_name + "_review_file.csv"
    df.to_csv(out_file_path, index=None)

    output_paths.append(out_file_path)
    
    return output_paths