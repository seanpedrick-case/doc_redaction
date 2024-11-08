import gradio as gr
import numpy as np
from typing import List
from gradio_image_annotation import image_annotator
from gradio_image_annotation.image_annotator import AnnotatedImageData

from tools.file_conversion import is_pdf, convert_pdf_to_images
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_redaction import redact_page_with_pymupdf
import json
import pymupdf
from fitz import Document
from PIL import ImageDraw, Image

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
        if current_zoom_level >= 50:
            current_zoom_level -= 10
    else:    
        if current_zoom_level < 100:
            current_zoom_level += 10
        
    return current_zoom_level, annotate_current_page


def update_annotator(image_annotator_object:AnnotatedImageData, page_num:int, zoom:int=100):
    # print("\nImage annotator object:", image_annotator_object)

    zoom_str = str(zoom) + '%'

    if not image_annotator_object:
        return image_annotator(
        label="Modify redaction boxes",
        #label_list=["Redaction"],
        #label_colors=[(0, 0, 0)],
        show_label=False,
        sources=["upload"],
        show_clear_button=False,
        show_share_button=False,
        show_remove_button=False,
        interactive=False
    ), gr.Number(label = "Page (press enter to change)", value=1, precision=0)

    if page_num is None:
        page_num = 0

    # Check bounding values for current page and page max
    if page_num > 0:
        page_num_reported = page_num
        #page_num = page_num - 1
    elif page_num == 0: page_num_reported = 1
    else: 
        page_num = 0   
        page_num_reported = 1 

    page_max_reported = len(image_annotator_object)

    if page_num_reported > page_max_reported:
        page_num_reported = page_max_reported

    out_image_annotator = image_annotator(value = image_annotator_object[page_num_reported - 1],
        boxes_alpha=0.1,
        box_thickness=1,
        #label_list=["Redaction"],
        #label_colors=[(0, 0, 0)],
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

    return out_image_annotator, number_reported, number_reported

def modify_existing_page_redactions(image_annotated:AnnotatedImageData, current_page:int, previous_page:int, all_image_annotations:List[AnnotatedImageData]):
    '''
    Overwrite current image annotations with modifications
    '''
    
    if not current_page:
        current_page = 1

    #If no previous page or is 0, i.e. first time run, then rewrite current page
    if not previous_page:
        previous_page = current_page
        #return all_image_annotations, current_page, current_page

    #print("all_image_annotations before:",all_image_annotations)
    
    image_annotated['image'] = all_image_annotations[previous_page - 1]["image"]

    #print("image_annotated:", image_annotated)

    all_image_annotations[previous_page - 1] = image_annotated

    #print("all_image_annotations after:",all_image_annotations)

    return all_image_annotations, current_page, current_page

def apply_redactions(image_annotated:AnnotatedImageData, file_paths:str, doc:Document, all_image_annotations:List[AnnotatedImageData], current_page:int, progress=gr.Progress(track_tqdm=True)):
    '''
    Apply modified redactions to a pymupdf
    '''
    #print("all_image_annotations:", all_image_annotations)

    output_files = []

    image_annotated['image'] = all_image_annotations[current_page - 1]["image"]

    all_image_annotations[current_page - 1] = image_annotated

    if not image_annotated:
        print("No image annotations found")
        return doc, all_image_annotations
    
    if isinstance(file_paths, list):
        file_path = file_paths[-1].name
    else:
        file_path = file_paths

    print("file_path:", file_path)
    file_base = get_file_path_end(file_path)
    
    # If working with image docs
    if is_pdf(file_path) == False:
        unredacted_doc = Image.open(file_paths[-1])

        image = unredacted_doc

        # try:
        #     image = Image.open(image_annotated['image'])
        # except:
        #     image = Image.fromarray(image_annotated['image'].astype('uint8'))

        draw = ImageDraw.Draw(unredacted_doc)

        for img_annotation_box in image_annotated['boxes']:
            coords = [img_annotation_box["xmin"],
            img_annotation_box["ymin"],
            img_annotation_box["xmax"],
            img_annotation_box["ymax"]]

            fill = img_annotation_box["color"]

            draw.rectangle(coords, fill=fill)

            image.save(output_folder + file_base + "_redacted_mod.png")

        doc = [image]

    # If working with pdfs
    else:
        unredacted_doc = pymupdf.open(file_path)

        number_of_pages = unredacted_doc.page_count

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

            pymupdf_page = unredacted_doc.load_page(i) #doc.load_page(current_page -1)
            pymupdf_page = redact_page_with_pymupdf(pymupdf_page, all_image_annotations[i], image)
              
    #try:
    out_pdf_file_path = output_folder + file_base + "_redacted_mod.pdf"
    unredacted_doc.save(out_pdf_file_path)
    output_files.append(out_pdf_file_path)

    # Save the gradio_annotation_boxes to a JSON file
    try:
        out_annotation_file_path = output_folder + file_base + '_modified_redactions.json'
        with open(out_annotation_file_path, 'w') as f:
            json.dump(all_image_annotations, f)
        output_files.append(out_annotation_file_path)
    except:
        print("Could not save annotations to json file.")

    return doc, all_image_annotations, output_files

def crop(annotations:AnnotatedImageData):
    if annotations["boxes"]:
        box = annotations["boxes"][0]
        return annotations["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]
        ]
    return None

def get_boxes_json(annotations:AnnotatedImageData):
    return annotations["boxes"]
