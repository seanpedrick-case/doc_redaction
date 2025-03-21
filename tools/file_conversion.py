from pdf2image import convert_from_path, pdfinfo_from_path
from tools.helper_functions import get_file_name_without_type, output_folder, tesseract_ocr_option, text_ocr_option, textract_option, read_file, get_or_create_env_var
from PIL import Image, ImageFile
import os
import re
import time
import json
import pymupdf
import pandas as pd
import numpy as np
import shutil
from pymupdf import Rect
from fitz import Page
from tqdm import tqdm
from gradio import Progress
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from PIL import Image
from scipy.spatial import cKDTree

image_dpi = 300.0
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def is_pdf_or_image(filename):
    """
    Check if a file name is a PDF or an image file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", ".jpg", or ".png", False otherwise.
    """
    if filename.lower().endswith(".pdf") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
        output = True
    else:
        output = False
    return output

def is_pdf(filename):
    """
    Check if a file name is a PDF.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", False otherwise.
    """
    return filename.lower().endswith(".pdf")

# %%
## Convert pdf to image if necessary

CUSTOM_BOX_COLOUR = get_or_create_env_var("CUSTOM_BOX_COLOUR", "")
print(f'The value of CUSTOM_BOX_COLOUR is {CUSTOM_BOX_COLOUR}')

def check_image_size_and_reduce(out_path:str, image:Image):
    '''
    Check if a given image size is above around 4.5mb, and reduce size if necessary. 5mb is the maximum possible to submit to AWS Textract.
    '''

    # Check file size and resize if necessary
    max_size = 4.5 * 1024 * 1024  # 5 MB in bytes # 5
    file_size = os.path.getsize(out_path)        

    width = image.width
    height = image.height

    # Resize images if they are too big
    if file_size > max_size:
        # Start with the original image size          

        print(f"Image size before {width}x{height}, original file_size: {file_size}")

        while file_size > max_size:
            # Reduce the size by a factor (e.g., 50% of the current size)
            new_width = int(width * 0.5)
            new_height = int(height * 0.5)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            image.save(out_path, format="PNG", optimize=True)
            
            # Update the file size
            file_size = os.path.getsize(out_path)
            print(f"Resized to {new_width}x{new_height}, new file_size: {file_size}")
    else:
        new_width = width
        new_height = height 

    return new_width, new_height

def process_single_page(pdf_path: str, page_num: int, image_dpi: float, output_dir: str = 'input') -> tuple[int, str]:
    try:
        # Construct the full output directory path
        output_dir = os.path.join(os.getcwd(), output_dir)
        out_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_{page_num}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            # Load existing image
            image = Image.open(out_path)
        else:
            # Convert PDF page to image
            image_l = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, 
                                        dpi=image_dpi, use_cropbox=False, use_pdftocairo=False)
            image = image_l[0]
            image = image.convert("L")
            image.save(out_path, format="PNG")

        width, height = image.size

        # Check if image size too large and reduce if necessary
        width, height = check_image_size_and_reduce(out_path, image)                

        return page_num, out_path, width, height

    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return page_num, "", width, height

def convert_pdf_to_images(pdf_path: str, prepare_for_review:bool=False, page_min: int = 0, image_dpi: float = image_dpi, num_threads: int = 8, output_dir: str = '/input'):

    # If preparing for review, just load the first page (not used)
    if prepare_for_review == True:
        page_count = pdfinfo_from_path(pdf_path)['Pages'] #1
    else:
        page_count = pdfinfo_from_path(pdf_path)['Pages']

    print(f"Number of pages in PDF: {page_count}")

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for page_num in range(page_min, page_count):
            futures.append(executor.submit(process_single_page, pdf_path, page_num, image_dpi))
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="pages", desc="Converting pages"):
            page_num, result, width, height = future.result()
            if result:
                results.append((page_num, result, width, height))
            else:
                print(f"Page {page_num + 1} failed to process.")
    
    # Sort results by page number
    results.sort(key=lambda x: x[0])
    images = [result[1] for result in results]
    widths = [result[2] for result in results]
    heights = [result[3] for result in results]

    print("PDF has been converted to images.")
    return images, widths, heights

# Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file(file_path:str, prepare_for_review:bool=False):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
 
    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        img_object = [file_path] #[Image.open(file_path)]

        # Load images from the file paths. Test to see if it is bigger than 4.5 mb and reduct if needed (Textract limit is 5mb)
        image = Image.open(file_path)
        img_object, image_sizes_width, image_sizes_height = check_image_size_and_reduce(file_path, image)

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        print(f"{file_path} is a PDF file. Converting to image set")
        # Run your function for processing PDF files here
        img_object, image_sizes_width, image_sizes_height = convert_pdf_to_images(file_path, prepare_for_review)

    else:
        print(f"{file_path} is not an image or PDF file.")
        img_object = []
        image_sizes_width = []
        image_sizes_height = []

    return img_object, image_sizes_width, image_sizes_height

def get_input_file_names(file_input:List[str]):
    '''
    Get list of input files to report to logs.
    '''

    all_relevant_files = []
    file_name_with_extension = ""
    full_file_name = ""

    #print("file_input in input file names:", file_input)
    if isinstance(file_input, dict):
        file_input = os.path.abspath(file_input["name"])

    if isinstance(file_input, str):
        file_input_list = [file_input]
    else:
        file_input_list = file_input

    for file in file_input_list:
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        file_path_without_ext = get_file_name_without_type(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()

        # Check if the file is an image type
        if (file_extension in ['.jpg', '.jpeg', '.png', '.pdf', '.xlsx', '.csv', '.parquet']) & ("review_file" not in file_path_without_ext):
            all_relevant_files.append(file_path_without_ext)
            file_name_with_extension = file_path_without_ext + file_extension
            full_file_name = file_path
    
    all_relevant_files_str = ", ".join(all_relevant_files)

    #print("all_relevant_files_str in input_file_names", all_relevant_files_str)
    #print("all_relevant_files in input_file_names", all_relevant_files)

    return all_relevant_files_str, file_name_with_extension, full_file_name, all_relevant_files

def convert_color_to_range_0_1(color):
    return tuple(component / 255 for component in color)

def redact_single_box(pymupdf_page:Page, pymupdf_rect:Rect, img_annotation_box:dict, custom_colours:bool=False):
    pymupdf_x1 = pymupdf_rect[0]
    pymupdf_y1 = pymupdf_rect[1]
    pymupdf_x2 = pymupdf_rect[2]
    pymupdf_y2 = pymupdf_rect[3]

    # Calculate area to actually remove text from the pdf (different from black box size)     
    redact_bottom_y = pymupdf_y1 + 2
    redact_top_y = pymupdf_y2 - 2

    # Calculate the middle y value and set a small height if default values are too close together
    if (redact_top_y - redact_bottom_y) < 1:        
        middle_y = (pymupdf_y1 + pymupdf_y2) / 2
        redact_bottom_y = middle_y - 1
        redact_top_y = middle_y + 1

    #print("Rect:", rect)

    rect_small_pixel_height = Rect(pymupdf_x1, redact_bottom_y, pymupdf_x2, redact_top_y)  # Slightly smaller than outside box

    # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
    #page.add_redact_annot(rect)#rect_small_pixel_height)
    pymupdf_page.add_redact_annot(rect_small_pixel_height)

    # Set up drawing a black box over the whole rect
    shape = pymupdf_page.new_shape()
    shape.draw_rect(pymupdf_rect)

    if custom_colours == True:
        if img_annotation_box["color"][0] > 1:
            out_colour = convert_color_to_range_0_1(img_annotation_box["color"])
        else:
            out_colour = img_annotation_box["color"]
    else:
        if CUSTOM_BOX_COLOUR == "grey":
            out_colour = (0.5, 0.5, 0.5)        
        else:
            out_colour = (0,0,0)

    shape.finish(color=out_colour, fill=out_colour)  # Black fill for the rectangle
    #shape.finish(color=(0, 0, 0))  # Black fill for the rectangle
    shape.commit()


def convert_pymupdf_to_image_coords(pymupdf_page, x1, y1, x2, y2, image: Image):
    '''
    Converts coordinates from pymupdf format to image coordinates,
    accounting for mediabox dimensions and offset.
    '''
    # Get rect dimensions
    rect = pymupdf_page.rect
    rect_width = rect.width
    rect_height = rect.height
    
    # Get mediabox dimensions and position
    mediabox = pymupdf_page.mediabox
    mediabox_width = mediabox.width
    mediabox_height = mediabox.height
    
    # Get target image dimensions
    image_page_width, image_page_height = image.size

    # Calculate scaling factors
    image_to_mediabox_x_scale = image_page_width / mediabox_width
    image_to_mediabox_y_scale = image_page_height / mediabox_height

    image_to_rect_scale_width = image_page_width / rect_width
    image_to_rect_scale_height = image_page_height / rect_height

    # Adjust for offsets (difference in position between mediabox and rect)
    x_offset = rect.x0 - mediabox.x0  # Difference in x position
    y_offset = rect.y0 - mediabox.y0  # Difference in y position

    #print("x_offset:", x_offset)
    #print("y_offset:", y_offset)

    # Adjust coordinates:
    # Apply scaling to match image dimensions
    x1_image = x1 * image_to_mediabox_x_scale    
    x2_image = x2 * image_to_mediabox_x_scale
    y1_image = y1 * image_to_mediabox_y_scale
    y2_image = y2 * image_to_mediabox_y_scale

    # Correct for difference in rect and mediabox size
    if mediabox_width != rect_width:
        
        mediabox_to_rect_x_scale = mediabox_width / rect_width
        mediabox_to_rect_y_scale = mediabox_height / rect_height

        rect_to_mediabox_x_scale = rect_width / mediabox_width
        #rect_to_mediabox_y_scale = rect_height / mediabox_height

        mediabox_rect_x_diff = (mediabox_width - rect_width) * (image_to_mediabox_x_scale / 2)
        mediabox_rect_y_diff = (mediabox_height - rect_height) * (image_to_mediabox_y_scale / 2)

        x1_image -= mediabox_rect_x_diff
        x2_image -= mediabox_rect_x_diff
        y1_image += mediabox_rect_y_diff
        y2_image += mediabox_rect_y_diff

        #
        x1_image *= mediabox_to_rect_x_scale
        x2_image *= mediabox_to_rect_x_scale
        y1_image *= mediabox_to_rect_y_scale
        y2_image *= mediabox_to_rect_y_scale

    return x1_image, y1_image, x2_image, y2_image

def redact_whole_pymupdf_page(rect_height, rect_width, image, page, custom_colours, border = 5):
    # Small border to page that remains white
    border = 5
    # Define the coordinates for the Rect
    whole_page_x1, whole_page_y1 = 0 + border, 0 + border  # Bottom-left corner
    whole_page_x2, whole_page_y2 = rect_width - border, rect_height - border  # Top-right corner

    whole_page_image_x1, whole_page_image_y1, whole_page_image_x2, whole_page_image_y2 = convert_pymupdf_to_image_coords(page, whole_page_x1, whole_page_y1, whole_page_x2, whole_page_y2, image)

    # Create new image annotation element based on whole page coordinates
    whole_page_rect = Rect(whole_page_x1, whole_page_y1, whole_page_x2, whole_page_y2)

    # Write whole page annotation to annotation boxes
    whole_page_img_annotation_box = {}
    whole_page_img_annotation_box["xmin"] = whole_page_image_x1
    whole_page_img_annotation_box["ymin"] = whole_page_image_y1
    whole_page_img_annotation_box["xmax"] = whole_page_image_x2
    whole_page_img_annotation_box["ymax"] = whole_page_image_y2
    whole_page_img_annotation_box["color"] = (0,0,0)
    whole_page_img_annotation_box["label"] = "Whole page"

    redact_single_box(page, whole_page_rect, whole_page_img_annotation_box, custom_colours)

    return whole_page_img_annotation_box

def prepare_image_or_pdf(
    file_paths: List[str],
    in_redact_method: str,
    latest_file_completed: int = 0,
    out_message: List[str] = [],
    first_loop_state: bool = False,
    number_of_pages:int = 1,
    all_annotations_object:List = [],
    prepare_for_review:bool = False,
    in_fully_redacted_list:List[int]=[],
    output_folder:str=output_folder,
    progress: Progress = Progress(track_tqdm=True)
) -> tuple[List[str], List[str]]:
    """
    Prepare and process image or text PDF files for redaction.

    This function takes a list of file paths, processes each file based on the specified redaction method,
    and returns the output messages and processed file paths.

    Args:
        file_paths (List[str]): List of file paths to process.
        in_redact_method (str): The redaction method to use.
        latest_file_completed (optional, int): Index of the last completed file.
        out_message (optional, List[str]): List to store output messages.
        first_loop_state (optional, bool): Flag indicating if this is the first iteration.
        number_of_pages (optional, int): integer indicating the number of pages in the document
        all_annotations_object(optional, List of annotation objects): All annotations for current document
        prepare_for_review(optional, bool): Is this preparation step preparing pdfs and json files to review current redactions?
        in_fully_redacted_list(optional, List of int): A list of pages to fully redact
        output_folder (optional, str): The output folder for file save
        progress (optional, Progress): Progress tracker for the operation
        

    Returns:
        tuple[List[str], List[str]]: A tuple containing the output messages and processed file paths.
    """

    tic = time.perf_counter()
    json_from_csv = False
    original_cropboxes = []  # Store original CropBox values

    if isinstance(in_fully_redacted_list, pd.DataFrame):
        if not in_fully_redacted_list.empty:
            in_fully_redacted_list = in_fully_redacted_list.iloc[:,0].tolist()

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("first_loop_state is True")
        latest_file_completed = 0
        out_message = []
        all_annotations_object = []
    else:
        print("Now attempting file:", str(latest_file_completed))
    
    # This is only run when a new page is loaded, so can reset page loop values. If end of last file (99), current loop number set to 999
    # if latest_file_completed == 99:
    #     current_loop_page_number = 999
    #     page_break_return = False
    # else:
    #     current_loop_page_number = 0
    #     page_break_return = False

    # If out message or converted_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]  

    converted_file_paths = []
    image_file_paths = []
    pymupdf_doc = []
    review_file_csv = pd.DataFrame()

    if not file_paths:
        file_paths = []

    if isinstance(file_paths, dict):
        file_paths = os.path.abspath(file_paths["name"])

    if isinstance(file_paths, str):
        file_path_number = 1
    else:
        file_path_number = len(file_paths)

    #print("Current_loop_page_number at start of prepare_image_or_pdf function is:", current_loop_page_number)
    print("Number of file paths:", file_path_number)
    print("Latest_file_completed:", latest_file_completed)
    
    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= file_path_number:
        print("Last file reached, returning files:", str(latest_file_completed))
        if isinstance(out_message, list):
            final_out_message = '\n'.join(out_message)
        else:
            final_out_message = out_message
        return final_out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes

    #in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    progress(0.1, desc='Preparing file')

    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
        file_paths_loop = file_paths_list
    else:
        if prepare_for_review == False:
            file_paths_list = file_paths
            file_paths_loop = [file_paths_list[int(latest_file_completed)]]
        else:
            file_paths_list = file_paths
            file_paths_loop = file_paths
             # Sort files to prioritise PDF files first, then JSON files. This means that the pdf can be loaded in, and pdf page path locations can be added to the json
            file_paths_loop = sorted(file_paths_loop, key=lambda x: (os.path.splitext(x)[1] != '.pdf', os.path.splitext(x)[1] != '.json'))      

    # Loop through files to load in
    for file in file_paths_loop:
        converted_file_path = []
        image_file_path = []

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name
        file_path_without_ext = get_file_name_without_type(file_path)
        file_name_with_ext = os.path.basename(file_path)

        if not file_path:
            out_message = "Please select a file."
            print(out_message)
            raise Exception(out_message)
            
        file_extension = os.path.splitext(file_path)[1].lower()

        # If a pdf, load as a pymupdf document
        if is_pdf(file_path):
            pymupdf_doc = pymupdf.open(file_path)

            # Load cropbox dimensions to use later  

            converted_file_path = file_path
            image_file_paths, image_sizes_width, image_sizes_height = process_file(file_path, prepare_for_review)
            page_sizes = []

            for i, page in enumerate(pymupdf_doc):
                page_no = i
                reported_page_no = i + 1
                
                pymupdf_page = pymupdf_doc.load_page(page_no)
                original_cropboxes.append(pymupdf_page.cropbox)  # Save original CropBox

                # Create a page_sizes_object
                out_page_image_sizes = {"page":reported_page_no, "image_width":image_sizes_width[page_no], "image_height":image_sizes_height[page_no], "mediabox_width":pymupdf_page.mediabox.width, "mediabox_height": pymupdf_page.mediabox.height, "cropbox_width":pymupdf_page.cropbox.width, "cropbox_height":pymupdf_page.cropbox.height}
                page_sizes.append(out_page_image_sizes)

            #Create base version of the annotation object that doesn't have any annotations in it
            if (not all_annotations_object) & (prepare_for_review == True):
                all_annotations_object = []

                for image_path in image_file_paths:
                    annotation = {}
                    annotation["image"] = image_path

                    all_annotations_object.append(annotation)
            
        elif is_pdf_or_image(file_path):  # Alternatively, if it's an image
            # Check if the file is an image type and the user selected text ocr option
            if file_extension in ['.jpg', '.jpeg', '.png'] and in_redact_method == text_ocr_option:
                in_redact_method = tesseract_ocr_option

            # Convert image to a pymupdf document
            pymupdf_doc = pymupdf.open()  # Create a new empty document

            img = Image.open(file_path)  # Open the image file
            rect = pymupdf.Rect(0, 0, img.width, img.height)  # Create a rectangle for the image
            pymupdf_page = pymupdf_doc.new_page(width=img.width, height=img.height)  # Add a new page
            pymupdf_page.insert_image(rect, filename=file_path)  # Insert the image into the page
            pymupdf_page = pymupdf_doc.load_page(0)

            original_cropboxes.append(pymupdf_page.cropbox)  # Save original CropBox

            file_path_str = str(file_path)

            image_file_paths, image_sizes_width, image_sizes_height = process_file(file_path_str, prepare_for_review)

            #print("image_file_paths:", image_file_paths)
            # Create a page_sizes_object
            out_page_image_sizes = {"page":1, "image_width":image_sizes_width[page_no], "image_height":image_sizes_height[page_no], "mediabox_width":pymupdf_page.mediabox.width, "mediabox_height": pymupdf_page.mediabox.height, "cropbox_width":original_cropboxes[-1].width, "cropbox_height":original_cropboxes[-1].height}
            page_sizes.append(out_page_image_sizes)

            converted_file_path = output_folder + file_name_with_ext

            pymupdf_doc.save(converted_file_path)

            print("Inserted image into PDF file")

        elif file_extension in ['.csv']:
            review_file_csv = read_file(file)
            all_annotations_object = convert_pandas_df_to_review_json(review_file_csv, image_file_paths, page_sizes)
            json_from_csv = True
            print("Converted CSV review file to json")

        # If the file name ends with redactions.json, assume it is an annoations object, overwrite the current variable
        if (file_extension in ['.json']) | (json_from_csv == True):

            if (file_extension in ['.json']) &  (prepare_for_review == True):
                print("Preparing file for review")
                if isinstance(file_path, str):
                    with open(file_path, 'r') as json_file:
                        all_annotations_object = json.load(json_file)
                else:
                    # Assuming file_path is a NamedString or similar
                    all_annotations_object = json.loads(file_path)  # Use loads for string content

            # Assume it's a textract json
            elif (file_extension == '.json') and (prepare_for_review is not True):
                # If the file ends with textract.json, assume it's a Textract response object.
                # Copy it to the output folder so it can be used later.
                out_folder = os.path.join(output_folder, file_path_without_ext + ".json")

                # Use shutil to copy the file directly
                shutil.copy2(file_path, out_folder)  # Preserves metadata
                
                continue

            # If you have an annotations object from the above code
            if all_annotations_object:
                #print("out_annotations_object before reloading images:", all_annotations_object)

                # Get list of page numbers
                image_file_paths_pages = [
                int(re.search(r'_(\d+)\.png$', os.path.basename(s)).group(1)) 
                for s in image_file_paths 
                if re.search(r'_(\d+)\.png$', os.path.basename(s))
                ]
                image_file_paths_pages = [int(i) for i in image_file_paths_pages]
                
                # If PDF pages have been converted to image files, replace the current image paths in the json to this. 
                if image_file_paths:
                    #print("Image file paths found")

                    #print("Image_file_paths:", image_file_paths)

                    #for i, annotation in enumerate(all_annotations_object):
                    for i, image_file_path in enumerate(image_file_paths):

                        if i < len(all_annotations_object): 
                            annotation = all_annotations_object[i]
                        else: 
                            annotation = {}
                            all_annotations_object.append(annotation)

                        #print("annotation:", annotation, "for page:", str(i))
                        try:
                            if not annotation:
                                annotation = {"image":"", "boxes": []}
                                annotation_page_number = int(re.search(r'_(\d+)\.png$', image_file_path).group(1))

                            else:
                                annotation_page_number = int(re.search(r'_(\d+)\.png$', annotation["image"]).group(1))
                        except Exception as e:
                            print("Extracting page number from image failed due to:", e)
                            annotation_page_number = 0
                        #print("Annotation page number:", annotation_page_number)

                        # Check if the annotation page number exists in the image file paths pages
                        if annotation_page_number in image_file_paths_pages:

                            # Set the correct image page directly since we know it's in the list
                            correct_image_page = annotation_page_number
                            annotation["image"] = image_file_paths[correct_image_page]
                        else:
                            print("Page", annotation_page_number, "image file not found.")

                        all_annotations_object[i] = annotation

                    #print("all_annotations_object at end of json/csv load part:", all_annotations_object)

                # Get list of pages that are to be fully redacted and redact them
                # if not in_fully_redacted_list.empty:
                #     print("Redacting whole pages")

                #     for i, image in enumerate(image_file_paths):
                #         page = pymupdf_doc.load_page(i)
                #         rect_height = page.rect.height
                #         rect_width = page.rect.width 
                #         whole_page_img_annotation_box = redact_whole_pymupdf_page(rect_height, rect_width, image, page, custom_colours = False, border = 5)

                #         all_annotations_object.append(whole_page_img_annotation_box)

                # Write the response to a JSON file in output folder
                out_folder = output_folder + file_path_without_ext + ".json"
                with open(out_folder, 'w') as json_file:
                    json.dump(all_annotations_object, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed
                continue

        # Must be something else, return with error message
        else:
            if in_redact_method == tesseract_ocr_option or in_redact_method == textract_option:
                if is_pdf_or_image(file_path) == False:
                    out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                    print(out_message)
                    raise Exception(out_message)

            elif in_redact_method == text_ocr_option:
                if is_pdf(file_path) == False:
                    out_message = "Please upload a PDF file for text analysis."
                    print(out_message)
                    raise Exception(out_message)  


        converted_file_paths.append(converted_file_path)
        image_file_paths.extend(image_file_path)        

        toc = time.perf_counter()
        out_time = f"File '{file_path_without_ext}' prepared in {toc - tic:0.1f} seconds."

        print(out_time)

        out_message.append(out_time)
        out_message_out = '\n'.join(out_message)

    number_of_pages = len(image_file_paths)
        
    return out_message_out, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes

def convert_text_pdf_to_img_pdf(in_file_path:str, out_text_file_path:List[str], image_dpi:float=image_dpi):
    file_path_without_ext = get_file_name_without_type(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions
    print("Creating image version of redacted PDF to embed redactions.")
    
    pdf_text_image_paths, image_sizes_width, image_sizes_height = process_file(out_text_file_path[0])
    out_text_image_file_path = output_folder + file_path_without_ext + "_text_redacted_as_img.pdf"
    pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=image_dpi, save_all=True, append_images=pdf_text_image_paths[1:])

    # out_file_paths.append(out_text_image_file_path)

    out_file_paths = [out_text_image_file_path]

    out_message = "PDF " + file_path_without_ext + " converted to image-based file."
    print(out_message)

    #print("Out file paths:", out_file_paths)

    return out_message, out_file_paths

def join_values_within_threshold(df1, df2):
    # Threshold for matching
    threshold = 5

    # Perform a cross join
    df1['key'] = 1
    df2['key'] = 1
    merged = pd.merge(df1, df2, on='key').drop(columns=['key'])

    # Apply conditions for all columns
    conditions = (
        (abs(merged['xmin_x'] - merged['xmin_y']) <= threshold) &
        (abs(merged['xmax_x'] - merged['xmax_y']) <= threshold) &
        (abs(merged['ymin_x'] - merged['ymin_y']) <= threshold) &
        (abs(merged['ymax_x'] - merged['ymax_y']) <= threshold)
    )

    # Filter rows that satisfy all conditions
    filtered = merged[conditions]

    # Drop duplicates if needed (e.g., keep only the first match for each row in df1)
    result = filtered.drop_duplicates(subset=['xmin_x', 'xmax_x', 'ymin_x', 'ymax_x'])

    # Merge back into the original DataFrame (if necessary)
    final_df = pd.merge(df1, result, left_on=['xmin', 'xmax', 'ymin', 'ymax'], right_on=['xmin_x', 'xmax_x', 'ymin_x', 'ymax_x'], how='left')

    # Clean up extra columns
    final_df = final_df.drop(columns=['key'])
    print(final_df)


def convert_review_json_to_pandas_df(all_annotations:List[dict], redaction_decision_output:pd.DataFrame=pd.DataFrame(), page_sizes:List[dict]=[]) -> pd.DataFrame:
    '''
    Convert the annotation json data to a dataframe format. Add on any text from the initial review_file dataframe by joining on pages/co-ordinates (doesn't work very well currently).
    '''
    # Flatten the data
    flattened_annotation_data = []
    page_sizes_df = pd.DataFrame()

    if not isinstance(redaction_decision_output, pd.DataFrame):
        redaction_decision_output = pd.DataFrame()

    for annotation in all_annotations:
        #print("annotation:", annotation)
        #print("flattened_data:", flattened_data)
        image_path = annotation["image"]

        # Use regex to find the number before .png
        match = re.search(r'_(\d+)\.png$', image_path)
        if match:
            number = match.group(1)  # Extract the number
            #print(number)  # Output: 0
            reported_number = int(number) + 1
        else:
            print("No number found before .png. Returning page 1.")
            reported_number = 1

        # Check if 'boxes' is in the annotation, if not, add an empty list
        if 'boxes' not in annotation:
            annotation['boxes'] = []        

        for box in annotation["boxes"]:
            if 'text' not in box:
                data_to_add = {"image": image_path, "page": reported_number,  **box} # "text": annotation['text'],
            else:
                data_to_add = {"image": image_path, "page": reported_number, "text": box['text'], **box}
            #print("data_to_add:", data_to_add)
            flattened_annotation_data.append(data_to_add)

    # Convert to a DataFrame
    review_file_df = pd.DataFrame(flattened_annotation_data)

    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        page_sizes_df["page"] = page_sizes_df["page"].astype(int)

    # Convert data to same coordinate system
    # If all coordinates all greater than one, this is a absolute image coordinates - change back to relative coordinates
    if "xmin" in review_file_df.columns:
        if review_file_df["xmin"].max() >= 1 and review_file_df["xmax"].max() >= 1 and review_file_df["ymin"].max() >= 1 and review_file_df["ymax"].max() >= 1:
            print("review file df has large coordinates")
            review_file_df["page"] = review_file_df["page"].astype(int)

            if "image_width" not in review_file_df.columns and not page_sizes_df.empty:                            
                review_file_df = review_file_df.merge(page_sizes_df, on="page", how="left")

            if "image_width" in review_file_df.columns:
                print("Dividing coordinates in review file")
                review_file_df["xmin"] = review_file_df["xmin"] / review_file_df["image_width"]
                review_file_df["xmax"] = review_file_df["xmax"] / review_file_df["image_width"]
                review_file_df["ymin"] = review_file_df["ymin"] / review_file_df["image_height"]
                review_file_df["ymax"] = review_file_df["ymax"] / review_file_df["image_height"]

                #print("review_file_df after coordinates divided:", review_file_df)

    if not redaction_decision_output.empty:
        # If all coordinates all greater than one, this is a absolute image coordinates - change back to relative coordinates
        if redaction_decision_output["xmin"].max() >= 1 and redaction_decision_output["xmax"].max() >= 1 and redaction_decision_output["ymin"].max() >= 1 and redaction_decision_output["ymax"].max() >= 1:

            redaction_decision_output["page"] = redaction_decision_output["page"].astype(int)

            if "image_width" not in redaction_decision_output.columns and not page_sizes_df.empty:                            
                redaction_decision_output = redaction_decision_output.merge(page_sizes_df, on="page", how="left")

            if "image_width" in redaction_decision_output.columns:
                redaction_decision_output["xmin"] = redaction_decision_output["xmin"] / redaction_decision_output["image_width"]
                redaction_decision_output["xmax"] = redaction_decision_output["xmax"] / redaction_decision_output["image_width"]
                redaction_decision_output["ymin"] = redaction_decision_output["ymin"] / redaction_decision_output["image_height"]
                redaction_decision_output["ymax"] = redaction_decision_output["ymax"] / redaction_decision_output["image_height"]

    #print("convert_review_json review_file_df before merges:", review_file_df[['xmin', 'ymin', 'xmax', 'ymax', 'label']])
    #print("review_file_df[xmin]", review_file_df["xmin"])

    #print("redaction_decision_output:", redaction_decision_output)
    #print("review_file_df:", review_file_df)

    # Join on additional text data from decision output results if included, if text not already there
    if not redaction_decision_output.empty: 
        if not 'text' in redaction_decision_output.columns:
            redaction_decision_output['text'] = ''

        if not 'text' in review_file_df.columns:
            review_file_df['text'] = ''

        # Load DataFrames
        df1 = review_file_df.copy()
        df2 = redaction_decision_output.copy()

        #print("review_file before tolerance merge:", review_file_df)
        #print("redaction_decision_output before tolerance merge:", redaction_decision_output)

        # Create a unique key based on coordinates and label for exact merge
        merge_keys = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'page']
        df1['key'] = df1[merge_keys].astype(str).agg('_'.join, axis=1)
        df2['key'] = df2[merge_keys].astype(str).agg('_'.join, axis=1)

        # Attempt exact merge first
        #merged_df = df1.merge(df2[['key', 'text']], on='key', how='left')

        # Attempt exact merge first, renaming df2['text'] to avoid suffixes
        merged_df = df1.merge(df2[['key', 'text']], on='key', how='left', suffixes=('', '_duplicate'))

        # If a match is found, keep that text; otherwise, keep the original df1 text
        merged_df['text'] = merged_df['text'].combine_first(merged_df.pop('text_duplicate'))

        #print("merged_df['text']:", merged_df['text'])

        # Handle missing matches using a proximity-based approach
        #if merged_df['text'].isnull().sum() > 0:
        print("Attempting tolerance-based merge for text")
        # Convert coordinates to numpy arrays for KDTree lookup
        tree = cKDTree(df2[['xmin', 'ymin', 'xmax', 'ymax']].values)
        query_coords = df1[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Find nearest neighbors within a reasonable tolerance (e.g., 1% of page)
        tolerance = 0.01
        distances, indices = tree.query(query_coords, distance_upper_bound=tolerance)

        # Assign text values where matches are found
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < tolerance and idx < len(df2):
                merged_df.at[i, 'text'] = df2.iloc[idx]['text']

        # Drop the temporary key column
        merged_df.drop(columns=['key'], inplace=True)

        review_file_df = merged_df

        review_file_df = review_file_df[["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text"]]

    # Ensure required columns exist, filling with blank if they don't
    for col in ["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text"]:
        if col not in review_file_df.columns:
            review_file_df[col] = ''

    #for col in ['xmin', 'xmax', 'ymin', 'ymax']:
    #    review_file_df[col] = np.floor(review_file_df[col])

    # If colours are saved as list, convert to tuple
    review_file_df["color"] = review_file_df["color"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # print("page_sizes:", page_sizes)

    # Convert page sizes to relative values
    # if page_sizes:
    #     print("Checking page sizes")
        
    #     page_sizes_df = pd.DataFrame(page_sizes)

    #     if "image_width" not in review_file_df.columns:
    #         review_file_df = review_file_df.merge(page_sizes_df, how="left", on = "page")
        
    #     # If all coordinates all greater than one, this is a absolute image coordinates - change back to relative coordinates
    #     if review_file_df["xmin"].max() > 1 and review_file_df["xmax"].max() > 1 and review_file_df["ymin"].max() > 1 and review_file_df["ymax"].max() > 1:
    #         print("Dividing coordinates by image width and height.")
    #         review_file_df["xmin"] = review_file_df["xmin"] / review_file_df["image_width"]
    #         review_file_df["xmax"] = review_file_df["xmax"] / review_file_df["image_width"]
    #         review_file_df["ymin"] = review_file_df["ymin"] / review_file_df["image_height"]
    #         review_file_df["ymax"] = review_file_df["ymax"] / review_file_df["image_height"]

    review_file_df = review_file_df.sort_values(['page', 'ymin', 'xmin', 'label'])

    review_file_df.to_csv(output_folder + "review_file_test.csv", index=None)

    return review_file_df

def convert_pandas_df_to_review_json(review_file_df: pd.DataFrame, image_paths: List[Image.Image], page_sizes:List[dict]=[]) -> List[dict]:
    '''
    Convert a review csv to a json file for use by the Gradio Annotation object.
    '''
    
    if page_sizes:
        
        page_sizes_df = pd.DataFrame(page_sizes)

        #print(page_sizes_df)

        if "image_width" not in review_file_df.columns:
            review_file_df = review_file_df.merge(page_sizes_df, how="left", on = "page")

        #print("review_file_df in convert pandas df to review json function:", review_file_df[["xmin", "xmax", "ymin", "ymax"]])
        
        # If all coordinates are less or equal to one, this is a relative page scaling - change back to image coordinates
        if review_file_df["xmin"].max() <= 1 and review_file_df["xmax"].max() <= 1 and review_file_df["ymin"].max() <= 1 and review_file_df["ymax"].max() <= 1:
            review_file_df["xmin"] = review_file_df["xmin"] * review_file_df["image_width"]
            review_file_df["xmax"] = review_file_df["xmax"] * review_file_df["image_width"]
            review_file_df["ymin"] = review_file_df["ymin"] * review_file_df["image_height"]
            review_file_df["ymax"] = review_file_df["ymax"] * review_file_df["image_height"]
            
    # Keep only necessary columns
    review_file_df = review_file_df[["image", "page", "xmin", "ymin", "xmax", "ymax", "color", "label"]]

    # If colours are saved as list, convert to tuple
    review_file_df.loc[:, "color"] = review_file_df.loc[:,"color"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Group the DataFrame by the 'image' column
    grouped_csv_pages = review_file_df.groupby('page')

    # Create a list to hold the JSON data
    json_data = []

    for n, pdf_image_path in enumerate(image_paths):
        reported_page_number = int(n + 1)
            

        if reported_page_number in review_file_df["page"].values:

            # Convert each relevant group to a list of box dictionaries
            selected_csv_pages = grouped_csv_pages.get_group(reported_page_number)
            annotation_boxes = selected_csv_pages.drop(columns=['image', 'page']).to_dict(orient='records')

            # If all bbox coordinates are below 1, then they are relative. Need to convert based on image size.
            
            annotation = {
                "image": pdf_image_path,
                "boxes": annotation_boxes
            }

        else:
            annotation = {}
            annotation["image"] = pdf_image_path

        # Append the structured data to the json_data list
        json_data.append(annotation)

    return json_data