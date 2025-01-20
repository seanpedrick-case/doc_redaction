import gradio as gr
import pandas as pd
import numpy as np
from typing import List
from gradio_image_annotation import image_annotator
from gradio_image_annotation.image_annotator import AnnotatedImageData

from tools.file_conversion import is_pdf, convert_review_json_to_pandas_df
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_redaction import redact_page_with_pymupdf
import json
import os
import pymupdf
from fitz import Document
from PIL import ImageDraw, Image
from collections import defaultdict

Image.MAX_IMAGE_PIXELS = None

def decrease_page(number:int):
    '''
    Decrease page number for review redactions page.
    '''
    #print("number:", str(number))
    if number > 1:
        return number - 1, number - 1
    else:
        return 1, 1

def increase_page(number:int, image_annotator_object:AnnotatedImageData):
    '''
    Increase page number for review redactions page.
    '''

    if not image_annotator_object:
        return 1, 1

    max_pages = len(image_annotator_object)

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

def remove_duplicate_images_with_blank_boxes(data: List[dict]) -> List[dict]:
    '''
    Remove items from the annotator object where the same page exists twice.
    '''
    # Group items by 'image'
    image_groups = defaultdict(list)
    for item in data:
        image_groups[item['image']].append(item)

    # Process each group to prioritize items with non-empty boxes
    result = []
    for image, items in image_groups.items():
        # Filter items with non-empty boxes
        non_empty_boxes = [item for item in items if item.get('boxes')]
        if non_empty_boxes:
            # Keep the first entry with non-empty boxes
            result.append(non_empty_boxes[0])
        else:
            # If all items have empty or missing boxes, keep the first item
            result.append(items[0])

    return result

def get_recogniser_dataframe_out(image_annotator_object, recogniser_dataframe_gr):
    recogniser_entities_list = ["Redaction"]
    recogniser_entities_drop = gr.Dropdown(value="", choices=[""], allow_custom_value=True, interactive=True)
    recogniser_dataframe_out = recogniser_dataframe_gr

    try:
        review_dataframe = convert_review_json_to_pandas_df(image_annotator_object)[["page", "label"]]
        recogniser_entities = review_dataframe["label"].unique().tolist()
        recogniser_entities.append("ALL")
        recogniser_entities_for_drop = sorted(recogniser_entities)


        recogniser_dataframe_out = gr.Dataframe(review_dataframe)
        recogniser_entities_drop = gr.Dropdown(value=recogniser_entities_for_drop[0], choices=recogniser_entities_for_drop, allow_custom_value=True, interactive=True)

        recogniser_entities_list = [entity for entity in recogniser_entities_for_drop if entity != 'Redaction' and entity != 'ALL']  # Remove any existing 'Redaction'
        recogniser_entities_list.insert(0, 'Redaction')  # Add 'Redaction' to the start of the list

    except Exception as e:
        print("Could not extract recogniser information:", e)
        recogniser_dataframe_out = recogniser_dataframe_gr
        recogniser_entities_drop = gr.Dropdown(value="", choices=[""], allow_custom_value=True, interactive=True)
        recogniser_entities_list = ["Redaction"]

    return recogniser_dataframe_out, recogniser_dataframe_out, recogniser_entities_drop, recogniser_entities_list

def update_annotator(image_annotator_object:AnnotatedImageData, page_num:int, recogniser_entities_drop=gr.Dropdown(value="ALL", allow_custom_value=True), recogniser_dataframe_gr=gr.Dataframe(pd.DataFrame(data={"page":[], "label":[]})), zoom:int=100):
    '''
    Update a gradio_image_annotation object with new annotation data
    '''    
    recogniser_entities_list = ["Redaction"]
    recogniser_dataframe_out = pd.DataFrame()

    if recogniser_dataframe_gr.empty:
        recogniser_dataframe_gr, recogniser_dataframe_out, recogniser_entities_drop, recogniser_entities_list = get_recogniser_dataframe_out(image_annotator_object, recogniser_dataframe_gr)    
    elif recogniser_dataframe_gr.iloc[0,0] == "":
        recogniser_dataframe_gr, recogniser_dataframe_out, recogniser_entities_drop, recogniser_entities_list = get_recogniser_dataframe_out(image_annotator_object, recogniser_dataframe_gr)
    else:        
        review_dataframe = update_entities_df(recogniser_entities_drop, recogniser_dataframe_gr)
        recogniser_dataframe_out = gr.Dataframe(review_dataframe)
        recogniser_entities_list = recogniser_dataframe_gr["label"].unique().tolist()

        print("recogniser_entities_list all options:", recogniser_entities_list)

        recogniser_entities_list = sorted(recogniser_entities_list)
        recogniser_entities_list = [entity for entity in recogniser_entities_list if entity != 'Redaction']  # Remove any existing 'Redaction'
        recogniser_entities_list.insert(0, 'Redaction')  # Add 'Redaction' to the start of the list

        print("recogniser_entities_list:", recogniser_entities_list)

    zoom_str = str(zoom) + '%'
    recogniser_colour_list = [(0, 0, 0) for _ in range(len(recogniser_entities_list))]

    if not image_annotator_object:
        page_num_reported = 1

        out_image_annotator = image_annotator(
        image_annotator_object[page_num_reported - 1],
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
        number_reported = gr.Number(label = "Page (press enter to change)", value=page_num_reported, precision=0)

        return out_image_annotator, number_reported, number_reported, page_num_reported, recogniser_entities_drop, recogniser_dataframe_out, recogniser_dataframe_gr
    
    #print("page_num at start of update_annotator function:", page_num)

    if page_num is None:
        page_num = 0

    # Check bounding values for current page and page max
    if page_num > 0:
        page_num_reported = page_num

    elif page_num == 0: page_num_reported = 1

    else: 
        page_num = 0   
        page_num_reported = 1 

    page_max_reported = len(image_annotator_object)

    if page_num_reported > page_max_reported:
        page_num_reported = page_max_reported

    image_annotator_object = remove_duplicate_images_with_blank_boxes(image_annotator_object)

    out_image_annotator = image_annotator(
        value = image_annotator_object[page_num_reported - 1],
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

    number_reported = gr.Number(label = "Page (press enter to change)", value=page_num_reported, precision=0)

    return out_image_annotator, number_reported, number_reported, page_num_reported, recogniser_entities_drop, recogniser_dataframe_out, recogniser_dataframe_gr

def modify_existing_page_redactions(image_annotated:AnnotatedImageData, current_page:int, previous_page:int, all_image_annotations:List[AnnotatedImageData], recogniser_entities_drop=gr.Dropdown(value="ALL", allow_custom_value=True),recogniser_dataframe=gr.Dataframe(pd.DataFrame(data={"page":[], "label":[]})), clear_all:bool=False):
    '''
    Overwrite current image annotations with modifications
    '''

    if not current_page:
        current_page = 1

    #If no previous page or is 0, i.e. first time run, then rewrite current page
    #if not previous_page:
    #    previous_page = current_page

    #print("image_annotated:", image_annotated)
    
    image_annotated['image'] = all_image_annotations[previous_page - 1]["image"]

    if clear_all == False:
        all_image_annotations[previous_page - 1] = image_annotated
    else:
        all_image_annotations[previous_page - 1]["boxes"] = []

    #print("all_image_annotations:", all_image_annotations)

    # Rewrite all_image_annotations search dataframe with latest updates
    try:
        review_dataframe = convert_review_json_to_pandas_df(all_image_annotations)[["page", "label"]]
        #print("review_dataframe['label']", review_dataframe["label"])
        recogniser_entities = review_dataframe["label"].unique().tolist()
        recogniser_entities.append("ALL")
        recogniser_entities = sorted(recogniser_entities)

        recogniser_dataframe_out = gr.Dataframe(review_dataframe)
        #recogniser_dataframe_gr = gr.Dataframe(review_dataframe)
        recogniser_entities_drop = gr.Dropdown(value=recogniser_entities_drop, choices=recogniser_entities, allow_custom_value=True, interactive=True)
    except Exception as e:
        print("Could not extract recogniser information:", e)
        recogniser_dataframe_out = recogniser_dataframe

    return all_image_annotations, current_page, current_page, recogniser_entities_drop, recogniser_dataframe_out

def apply_redactions(image_annotated:AnnotatedImageData, file_paths:List[str], doc:Document, all_image_annotations:List[AnnotatedImageData], current_page:int, review_file_state, save_pdf:bool=True, progress=gr.Progress(track_tqdm=True)):
    '''
    Apply modified redactions to a pymupdf and export review files
    '''
    #print("all_image_annotations:", all_image_annotations)

    output_files = []
    output_log_files = []

    #print("File paths in apply_redactions:", file_paths)

    image_annotated['image'] = all_image_annotations[current_page - 1]["image"]

    all_image_annotations[current_page - 1] = image_annotated

    if not image_annotated:
        print("No image annotations found")
        return doc, all_image_annotations
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        #print("file_path:", file_path)
        file_base = get_file_path_end(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()
        
        if save_pdf == True:
            # If working with image docs
            if (is_pdf(file_path) == False) & (file_extension not in '.csv'):
                image = Image.open(file_paths[-1])

                #image = pdf_doc

                draw = ImageDraw.Draw(image)

                for img_annotation_box in image_annotated['boxes']:
                    coords = [img_annotation_box["xmin"],
                    img_annotation_box["ymin"],
                    img_annotation_box["xmax"],
                    img_annotation_box["ymax"]]

                    fill = img_annotation_box["color"]

                    draw.rectangle(coords, fill=fill)

                    image.save(output_folder + file_base + "_redacted.png")

                doc = [image]

            elif file_extension in '.csv':
                print("This is a csv")
                pdf_doc = []

            # If working with pdfs
            elif is_pdf(file_path) == True:
                pdf_doc = pymupdf.open(file_path)

                number_of_pages = pdf_doc.page_count

                print("Saving pages to file.")

                for i in progress.tqdm(range(0, number_of_pages), desc="Saving redactions to file", unit = "pages"):

                    #print("Saving page", str(i))
                    
                    image_loc = all_image_annotations[i]['image']
                    #print("Image location:", image_loc)

                    # Load in image object
                    if isinstance(image_loc, np.ndarray):
                        image = Image.fromarray(image_loc.astype('uint8'))
                        #all_image_annotations[i]['image'] = image_loc.tolist()
                    elif isinstance(image_loc, Image.Image):
                        image = image_loc
                        #image_out_folder = output_folder + file_base + "_page_" + str(i) + ".png"
                        #image_loc.save(image_out_folder)
                        #all_image_annotations[i]['image'] = image_out_folder
                    elif isinstance(image_loc, str):
                        image = Image.open(image_loc)

                    pymupdf_page = pdf_doc.load_page(i) #doc.load_page(current_page -1)
                    pymupdf_page = redact_page_with_pymupdf(pymupdf_page, all_image_annotations[i], image)

            else:
                print("File type not recognised.")
                    
            #try:
            if pdf_doc:
                out_pdf_file_path = output_folder + file_base + "_redacted.pdf"
                pdf_doc.save(out_pdf_file_path)
                output_files.append(out_pdf_file_path)

        try:
            print("Saving annotations to JSON")

            out_annotation_file_path = output_folder + file_base + '_review_file.json'
            with open(out_annotation_file_path, 'w') as f:
                json.dump(all_image_annotations, f)
            output_log_files.append(out_annotation_file_path)

            print("Saving annotations to CSV review file")

            #print("review_file_state:", review_file_state)

            # Convert json to csv and also save this
            review_df = convert_review_json_to_pandas_df(all_image_annotations, review_file_state)
            out_review_file_file_path = output_folder + file_base + '_review_file.csv'
            review_df.to_csv(out_review_file_file_path, index=None)
            output_files.append(out_review_file_file_path)

        except Exception as e:
            print("Could not save annotations to json or csv file:", e)

    return doc, all_image_annotations, output_files, output_log_files

def get_boxes_json(annotations:AnnotatedImageData):
    return annotations["boxes"]

def update_entities_df(choice:str, df:pd.DataFrame):
    if choice=="ALL":
        return df
    else:
        return df.loc[df["label"]==choice,:]
    
def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
        #print("index", evt.index)
        #print("value", evt.value)
        #print("row_value", evt.row_value)
        row_value_page = evt.row_value[0] # This is the page number value
        return row_value_page

