from pdf2image import convert_from_path, pdfinfo_from_path
from tools.helper_functions import get_file_path_end, output_folder
from PIL import Image
import os
from gradio import Progress
from typing import List, Optional

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

def convert_pdf_to_images(pdf_path:str, page_min:int = 0, progress=Progress(track_tqdm=True)):

    # Get the number of pages in the PDF
    page_count = pdfinfo_from_path(pdf_path)['Pages']
    print("Number of pages in PDF: ", str(page_count))

    images = []

    # Open the PDF file
    #for page_num in progress.tqdm(range(0,page_count), total=page_count, unit="pages", desc="Converting pages"):
    for page_num in range(page_min,page_count): #progress.tqdm(range(0,page_count), total=page_count, unit="pages", desc="Converting pages"):
        
        # print("Current page: ", str(page_num + 1))

        # Convert one page to image
        image = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300, use_cropbox=True, use_pdftocairo=False)
        

        # If no images are returned, break the loop
        if not image:
            print("Conversion of page", str(page_num), "to file failed.")
            break

        # print("Conversion of page", str(page_num), "to file succeeded.")
        # print("image:", image)

        images.extend(image)

    print("PDF has been converted to images.")
    # print("Images:", images)

    return images


# %% Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        img_object = [Image.open(file_path)]
        # Load images from the file paths

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        print(f"{file_path} is a PDF file. Converting to image set")
        # Run your function for processing PDF files here
        img_object = convert_pdf_to_images(file_path)

    else:
        print(f"{file_path} is not an image or PDF file.")
        img_object = ['']

    return img_object

def prepare_image_or_text_pdf(
    file_paths: List[str],
    in_redact_method: str,
    in_allow_list: Optional[List[List[str]]] = None,
    latest_file_completed: int = 0,
    out_message: List[str] = [],
    first_loop_state: bool = False,
    progress: Progress = Progress(track_tqdm=True)
) -> tuple[List[str], List[str]]:
    """
    Prepare and process image or text PDF files for redaction.

    This function takes a list of file paths, processes each file based on the specified redaction method,
    and returns the output messages and processed file paths.

    Args:
        file_paths (List[str]): List of file paths to process.
        in_redact_method (str): The redaction method to use.
        in_allow_list (Optional[List[List[str]]]): List of allowed terms for redaction.
        latest_file_completed (int): Index of the last completed file.
        out_message (List[str]): List to store output messages.
        first_loop_state (bool): Flag indicating if this is the first iteration.
        progress (Progress): Progress tracker for the operation.

    Returns:
        tuple[List[str], List[str]]: A tuple containing the output messages and processed file paths.
    """

    # If out message or out_file_paths are blank, change to a list so it can be appended to
    #if isinstance(out_message, str):
    #    out_message = [out_message]    

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        out_message = []
        out_file_paths = []
    else:
        print("Now attempting file:", str(latest_file_completed))
        out_file_paths = []  

    if not file_paths:
        file_paths = []

    #out_file_paths = file_paths
    
    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= len(file_paths):
        print("Last file reached, returning files:", str(latest_file_completed))
        #final_out_message = '\n'.join(out_message)
        return out_message, out_file_paths

    #in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    file_paths_loop = [file_paths[int(latest_file_completed)]]
    #print("file_paths_loop:", str(file_paths_loop))

    #for file in progress.tqdm(file_paths, desc="Preparing files"):
    for file in file_paths_loop:
        file_path = file.name

        print("file_path:", file_path)

        #if file_path:
        #    file_path_without_ext = get_file_path_end(file_path)
        if not file_path:
            out_message = "No file selected"
            print(out_message)
            return out_message, out_file_paths

        if in_redact_method == "Image analysis":
            # Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                print(out_message)
                return out_message, out_file_paths
            
            out_file_path = process_file(file_path)
            #print("Out file path at image conversion step:", out_file_path)

        elif in_redact_method == "Text analysis":
            if is_pdf(file_path) == False:
                out_message = "Please upload a PDF file for text analysis."
                print(out_message)
                return out_message, out_file_paths
            
            out_file_path = file_path

        out_file_paths.append(out_file_path)
    
    return out_message, out_file_paths

def convert_text_pdf_to_img_pdf(in_file_path:str, out_text_file_path:List[str]):
    file_path_without_ext = get_file_path_end(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions
    print("Creating image version of redacted PDF to embed redactions.")
    
    pdf_text_image_paths = process_file(out_text_file_path[0])
    out_text_image_file_path = output_folder + file_path_without_ext + "_text_redacted_as_img.pdf"
    pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=300.0, save_all=True, append_images=pdf_text_image_paths[1:])

    # out_file_paths.append(out_text_image_file_path)

    out_file_paths = [out_text_image_file_path]

    out_message = "PDF " + file_path_without_ext + " converted to image-based file."
    print(out_message)

    #print("Out file paths:", out_file_paths)

    return out_message, out_file_paths