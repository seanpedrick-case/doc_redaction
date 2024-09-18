import boto3
from PIL import Image
import io
import json
import pikepdf
# Example: converting this single page to an image
from pdf2image import convert_from_bytes
from tools.custom_image_analyser_engine import OCRResult, CustomImageRecognizerResult

def analyse_page_with_textract(pdf_page_bytes, json_file_path):
    '''
    Analyse page with AWS Textract
    '''
    try:
        client = boto3.client('textract')
    except:
        print("Cannot connect to AWS Textract")
        return "", "", ""

    print("Analysing page with AWS Textract")
    
    # Convert the image to bytes using an in-memory buffer
    #image_buffer = io.BytesIO()
    #image.save(image_buffer, format='PNG')  # Save as PNG, or adjust format if needed
    #image_bytes = image_buffer.getvalue()

    #response = client.detect_document_text(Document={'Bytes': image_bytes})
    response = client.analyze_document(Document={'Bytes': pdf_page_bytes}, FeatureTypes=["SIGNATURES"])

    text_blocks = response['Blocks']    

    # Write the response to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(response, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

    print("Response has been written to output:", json_file_path)       
            
    return text_blocks


def convert_pike_pdf_page_to_bytes(pdf, page_num):
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
    pdf_bytes = buffer.getvalue()

    # Now you can use the `pdf_bytes` to convert it to an image or further process
    buffer.close()

    #images = convert_from_bytes(pdf_bytes)
    #image = images[0]

    return pdf_bytes


def json_to_ocrresult(json_data, page_width, page_height):
    '''
    Convert the json response from textract to the OCRResult format used elsewhere in the code.
    '''
    all_ocr_results = []
    signature_or_handwriting_recogniser_results = []
    signatures = []
    handwriting = []

    for text_block in json_data:

        is_signature = False
        is_handwriting = False

        if (text_block['BlockType'] == 'WORD') | (text_block['BlockType'] == 'LINE'):
            text = text_block['Text']

            # Extract BoundingBox details
            bbox = text_block["Geometry"]["BoundingBox"]
            left = bbox["Left"]
            top = bbox["Top"]
            width = bbox["Width"]
            height = bbox["Height"]

            # Convert proportional coordinates to absolute coordinates
            left_abs = int(left * page_width)
            top_abs = int(top * page_height)
            width_abs = int(width * page_width)
            height_abs = int(height * page_height)

            # Create OCRResult with absolute coordinates
            ocr_result = OCRResult(text, left_abs, top_abs, width_abs, height_abs)

            # If handwriting or signature, add to bounding box
            confidence = text_block['Confidence']            

            if 'TextType' in text_block:
                text_type = text_block["TextType"]
                
                if text_type == "HANDWRITING":
                    is_handwriting = True
                    entity_name = "HANDWRITING"
                    word_end = len(entity_name)
                    recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= text, score= confidence, start=0, end=word_end, left=left_abs, top=top_abs, width=width_abs, height=height_abs)
                    handwriting.append(recogniser_result)                    
                    print("Handwriting found:", handwriting[-1]) 
            
            all_ocr_results.append(ocr_result)

        elif (text_block['BlockType'] == 'SIGNATURE'):
            text = "SIGNATURE"

            # Extract BoundingBox details
            bbox = text_block["Geometry"]["BoundingBox"]
            left = bbox["Left"]
            top = bbox["Top"]
            width = bbox["Width"]
            height = bbox["Height"]

            # Convert proportional coordinates to absolute coordinates
            left_abs = int(left * page_width)
            top_abs = int(top * page_height)
            width_abs = int(width * page_width)
            height_abs = int(height * page_height)

            # Create OCRResult with absolute coordinates
            ocr_result = OCRResult(text, left_abs, top_abs, width_abs, height_abs)


            is_signature = True
            entity_name = "Signature"
            word_end = len(entity_name)
            recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= text, score= confidence, start=0, end=word_end, left=left_abs, top=top_abs, width=width_abs, height=height_abs)
            signatures.append(recogniser_result)
            print("Signature found:", signatures[-1])

            all_ocr_results.append(ocr_result)

        is_signature_or_handwriting = is_signature | is_handwriting

        # If it is signature or handwriting, will overwrite the default behaviour of the PII analyser
        if is_signature_or_handwriting:
            signature_or_handwriting_recogniser_results.append(recogniser_result)
    
    return all_ocr_results, signature_or_handwriting_recogniser_results