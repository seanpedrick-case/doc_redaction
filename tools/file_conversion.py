from pdf2image import convert_from_path
import os

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

def convert_pdf_to_images(pdf_path):

    image_paths = []

    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)

    # Save each image as a separate file
    # for i, image in enumerate(images):
    #     page_path = f"processing/page_{i+1}.png"
    #     image.save(page_path, "PNG")
    #     image_paths.append(page_path)

    print("PDF has been converted to images.")

    return images

# %%
def process_file(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        out_path = [file_path]

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        print(f"{file_path} is a PDF file. Converting to image set")
        # Run your function for processing PDF files here
        out_path = convert_pdf_to_images(file_path)

    else:
        print(f"{file_path} is not an image or PDF file.")
        out_path = ['']

    return out_path

