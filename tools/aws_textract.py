import boto3
from PIL import Image
import io
import json
import pikepdf
# Example: converting this single page to an image
from pdf2image import convert_from_bytes
from tools.custom_image_analyser_engine import OCRResult, CustomImageRecognizerResult

def extract_textract_metadata(response):
    """Extracts metadata from an AWS Textract response."""

    print("Document metadata:", response['DocumentMetadata'])

    request_id = response['ResponseMetadata']['RequestId']
    pages = response['DocumentMetadata']['Pages']
    #number_of_pages = response['DocumentMetadata']['NumberOfPages']

    return str({
        'RequestId': request_id,
        'Pages': pages
        #,
        #'NumberOfPages': number_of_pages
    })

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
    request_metadata = extract_textract_metadata(response) # Metadata comes out as a string

    # Write the response to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(response, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

    print("Response has been written to output:", json_file_path)       
            
    return text_blocks, request_metadata


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
    Convert the json response from textract to the OCRResult format used elsewhere in the code. Looks for lines, words, and signatures. Handwriting and signatures are set aside especially for later in case the user wants to override the default behaviour and redact all handwriting/signatures.
    '''
    all_ocr_results = []
    signature_or_handwriting_recogniser_results = []
    signature_recogniser_results = []
    handwriting_recogniser_results = []
    signatures = []
    handwriting = []

    combined_results = {}

    for text_block in json_data:

        is_signature = False
        is_handwriting = False

        

        if (text_block['BlockType'] == 'LINE') | (text_block['BlockType'] == 'SIGNATURE'): # (text_block['BlockType'] == 'WORD') |

            if text_block['BlockType'] == 'LINE':
                # Extract text and bounding box for the line
                line_text = text_block.get('Text', '')
                line_bbox = text_block["Geometry"]["BoundingBox"]
                line_left = int(line_bbox["Left"] * page_width)
                line_top = int(line_bbox["Top"] * page_height)
                line_right = int((line_bbox["Left"] + line_bbox["Width"]) * page_width)
                line_bottom = int((line_bbox["Top"] + line_bbox["Height"]) * page_height)

                words = []
                if 'Relationships' in text_block:
                    for relationship in text_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                child_block = next((block for block in json_data if block['Id'] == child_id), None)
                                if child_block and child_block['BlockType'] == 'WORD':
                                    word_text = child_block.get('Text', '')
                                    word_bbox = child_block["Geometry"]["BoundingBox"]
                                    confidence = child_block.get('Confidence','')
                                    word_left = int(word_bbox["Left"] * page_width)
                                    word_top = int(word_bbox["Top"] * page_height)
                                    word_right = int((word_bbox["Left"] + word_bbox["Width"]) * page_width)
                                    word_bottom = int((word_bbox["Top"] + word_bbox["Height"]) * page_height)

                                    # Extract BoundingBox details
                                    width = word_bbox["Width"]
                                    height = word_bbox["Height"]

                                    # Convert proportional coordinates to absolute coordinates
                                    width_abs = int(width * page_width)
                                    height_abs = int(height * page_height)
                                    
                                    words.append({
                                        'text': word_text,
                                        'bounding_box': (word_left, word_top, word_right, word_bottom)
                                    })
                                    # Check for handwriting
                                    text_type = child_block.get("TextType", '')

                                    if text_type == "HANDWRITING":
                                        is_handwriting = True
                                        entity_name = "HANDWRITING"
                                        word_end = len(entity_name)
                                        recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= word_text, score= confidence, start=0, end=word_end, left=word_left, top=word_top, width=width_abs, height=height_abs)
                                        handwriting.append(recogniser_result)                    
                                        print("Handwriting found:", handwriting[-1]) 

                combined_results[line_text] = {
                    'bounding_box': (line_left, line_top, line_right, line_bottom),
                    'words': words
                }

                

                # If handwriting or signature, add to bounding box               

            elif (text_block['BlockType'] == 'SIGNATURE'):
                line_text = "SIGNATURE"

                is_signature = True
                entity_name = "SIGNATURE"
                confidence = text_block['Confidence']
                word_end = len(entity_name)

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

                recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= line_text, score= confidence, start=0, end=word_end, left=left_abs, top=top_abs, width=width_abs, height=height_abs)
                signatures.append(recogniser_result)
                print("Signature found:", signatures[-1])

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
            ocr_result = OCRResult(line_text, left_abs, top_abs, width_abs, height_abs)
            all_ocr_results.append(ocr_result)

            is_signature_or_handwriting = is_signature | is_handwriting

            # If it is signature or handwriting, will overwrite the default behaviour of the PII analyser
            if is_signature_or_handwriting:
                signature_or_handwriting_recogniser_results.append(recogniser_result)

                if is_signature: signature_recogniser_results.append(recogniser_result)
                if is_handwriting: handwriting_recogniser_results.append(recogniser_result)
    
    return all_ocr_results, signature_or_handwriting_recogniser_results, signature_recogniser_results, handwriting_recogniser_results, combined_results