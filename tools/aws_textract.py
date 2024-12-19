import boto3
#from PIL import Image
from typing import List
import io
#import json
import pikepdf
import time
# Example: converting this single page to an image
#from pdf2image import convert_from_bytes
from tools.custom_image_analyser_engine import OCRResult, CustomImageRecognizerResult

def extract_textract_metadata(response):
    """Extracts metadata from an AWS Textract response."""

    #print("Document metadata:", response['DocumentMetadata'])

    request_id = response['ResponseMetadata']['RequestId']
    pages = response['DocumentMetadata']['Pages']
    #number_of_pages = response['DocumentMetadata']['NumberOfPages']

    return str({
        'RequestId': request_id,
        'Pages': pages
        #,
        #'NumberOfPages': number_of_pages
    })

def analyse_page_with_textract(pdf_page_bytes, page_no, client="", handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"]):
    '''
    Analyse page with AWS Textract
    '''
    if client == "":
        try:
            client = boto3.client('textract')
        except:
            print("Cannot connect to AWS Textract")
            return [], ""  # Return an empty list and an empty string

    #print("Analysing page with AWS Textract")
    #print("pdf_page_bytes:", pdf_page_bytes)
    #print("handwrite_signature_checkbox:", handwrite_signature_checkbox)
    
    # Redact signatures if specified
    if "Redact all identified signatures" in handwrite_signature_checkbox:
        #print("Analysing document with signature detection")
        try:
            response = client.analyze_document(Document={'Bytes': pdf_page_bytes}, FeatureTypes=["SIGNATURES"])
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 3 seconds.")
            time.sleep(3)
            response = client.analyze_document(Document={'Bytes': pdf_page_bytes}, FeatureTypes=["SIGNATURES"])
    else:
        #print("Analysing document without signature detection")
        # Call detect_document_text to extract plain text
        try:
            response = client.detect_document_text(Document={'Bytes': pdf_page_bytes})
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 5 seconds.")
            time.sleep(5)
            response = client.detect_document_text(Document={'Bytes': pdf_page_bytes})

    # Wrap the response with the page number in the desired format
    wrapped_response = {
        'page_no': page_no,
        'data': response
    }

    request_metadata = extract_textract_metadata(response)  # Metadata comes out as a string

    # Return a list containing the wrapped response and the metadata
    return wrapped_response, request_metadata  # Return as a list to match the desired structure


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


def json_to_ocrresult(json_data, page_width, page_height, page_no):
    '''
    Convert the json response from textract to the OCRResult format used elsewhere in the code. Looks for lines, words, and signatures. Handwriting and signatures are set aside especially for later in case the user wants to override the default behaviour and redact all handwriting/signatures.
    '''
    all_ocr_results = []
    signature_or_handwriting_recogniser_results = []
    signature_recogniser_results = []
    handwriting_recogniser_results = []
    signatures = []
    handwriting = []
    ocr_results_with_children = {}
    text_block={}

    i = 1

    # Assuming json_data is structured as a dictionary with a "pages" key
    #if "pages" in json_data:
    # Find the specific page data
    page_json_data = json_data #next((page for page in json_data["pages"] if page["page_no"] == page_no), None)

    if "Blocks" in page_json_data:
        # Access the data for the specific page
        text_blocks = page_json_data["Blocks"]  # Access the Blocks within the page data
    # This is a new page
    elif "page_no" in page_json_data:
        text_blocks = page_json_data["data"]["Blocks"]

    is_signature = False
    is_handwriting = False

    for text_block in text_blocks:
    
        if (text_block['BlockType'] == 'LINE') | (text_block['BlockType'] == 'SIGNATURE'): # (text_block['BlockType'] == 'WORD') |

            # Extract text and bounding box for the line
            line_bbox = text_block["Geometry"]["BoundingBox"]
            line_left = int(line_bbox["Left"] * page_width)
            line_top = int(line_bbox["Top"] * page_height)
            line_right = int((line_bbox["Left"] + line_bbox["Width"]) * page_width)
            line_bottom = int((line_bbox["Top"] + line_bbox["Height"]) * page_height)

            width_abs = int(line_bbox["Width"] * page_width)
            height_abs = int(line_bbox["Height"] * page_height)

            if text_block['BlockType'] == 'LINE':
                
                # Extract text and bounding box for the line
                line_text = text_block.get('Text', '')

                words = []
                if 'Relationships' in text_block:
                    for relationship in text_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                child_block = next((block for block in text_blocks if block['Id'] == child_id), None)
                                if child_block and child_block['BlockType'] == 'WORD':
                                    word_text = child_block.get('Text', '')
                                    word_bbox = child_block["Geometry"]["BoundingBox"]
                                    confidence = child_block.get('Confidence','')
                                    word_left = int(word_bbox["Left"] * page_width)
                                    word_top = int(word_bbox["Top"] * page_height)
                                    word_right = int((word_bbox["Left"] + word_bbox["Width"]) * page_width)
                                    word_bottom = int((word_bbox["Top"] + word_bbox["Height"]) * page_height)

                                    # Extract BoundingBox details
                                    word_width = word_bbox["Width"]
                                    word_height = word_bbox["Height"]

                                    # Convert proportional coordinates to absolute coordinates
                                    word_width_abs = int(word_width * page_width)
                                    word_height_abs = int(word_height * page_height)
                                    
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

                                        recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= word_text, score= confidence, start=0, end=word_end, left=word_left, top=word_top, width=word_width_abs, height=word_height_abs)

                                        if recogniser_result not in handwriting:
                                            handwriting.append(recogniser_result)
                                            #print("Handwriting found:", handwriting[-1]) 

            # If handwriting or signature, add to bounding box               

            elif (text_block['BlockType'] == 'SIGNATURE'):
                line_text = "SIGNATURE"

                is_signature = True
                entity_name = "SIGNATURE"
                confidence = text_block['Confidence']
                word_end = len(entity_name)

                recogniser_result = CustomImageRecognizerResult(entity_type=entity_name, text= line_text, score= confidence, start=0, end=word_end, left=line_left, top=line_top, width=width_abs, height=height_abs)

                if recogniser_result not in signatures:
                    signatures.append(recogniser_result)
                    #print("Signature found:", signatures[-1])

                words = []
                words.append({
                            'text': line_text,
                            'bounding_box': (line_left, line_top, line_right, line_bottom)
                        })

            ocr_results_with_children["text_line_" + str(i)] = {
                "line": i,
                'text': line_text,
                'bounding_box': (line_left, line_top, line_right, line_bottom),
                'words': words
            }         

            # Create OCRResult with absolute coordinates
            ocr_result = OCRResult(line_text, line_left, line_top, width_abs, height_abs)
            all_ocr_results.append(ocr_result)

            is_signature_or_handwriting = is_signature | is_handwriting

            # If it is signature or handwriting, will overwrite the default behaviour of the PII analyser
            if is_signature_or_handwriting:
                if recogniser_result not in signature_or_handwriting_recogniser_results: 
                    signature_or_handwriting_recogniser_results.append(recogniser_result)

                if is_signature:
                    if recogniser_result not in signature_recogniser_results:          
                        signature_recogniser_results.append(recogniser_result)

                if is_handwriting: 
                    if recogniser_result not in handwriting_recogniser_results: 
                        handwriting_recogniser_results.append(recogniser_result)

            i += 1

    return all_ocr_results, signature_or_handwriting_recogniser_results, signature_recogniser_results, handwriting_recogniser_results, ocr_results_with_children