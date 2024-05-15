from PIL import Image
from typing import List
import pandas as pd
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
from pdfminer.high_level import extract_pages
from tools.file_conversion import process_file
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine, LTAnno
from pikepdf import Pdf, Dictionary, Name
from gradio import Progress
import time

from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold
from tools.helper_functions import get_file_path_end
from tools.file_conversion import process_file, is_pdf, is_pdf_or_image
import gradio as gr

def choose_and_run_redactor(file_path:str, image_paths:List[str], language:str, chosen_redact_entities:List[str], in_redact_method:str, in_allow_list:List[List[str]]=None, progress=gr.Progress(track_tqdm=True)):

    tic = time.perf_counter()

    out_message = ''
    out_file_paths = []

    in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    if file_path:
         file_path_without_ext = get_file_path_end(file_path)
    else:
         out_message = "No file selected"
         print(out_message)
         return out_message, out_file_paths

    if in_redact_method == "Image analysis":
        # Analyse and redact image-based pdf or image
        # if is_pdf_or_image(file_path) == False:
        #     return "Please upload a PDF file or image file (JPG, PNG) for image analysis.", None

        pdf_images = redact_image_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat)
        out_image_file_path = "output/" + file_path_without_ext + "_result_as_img.pdf"
        pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])

        out_file_paths.append(out_image_file_path)
        out_message = "Image-based PDF successfully redacted and saved to file."

    elif in_redact_method == "Text analysis":
        if is_pdf(file_path) == False:
            return "Please upload a PDF file for text analysis.", None

        # Analyse text-based pdf
        pdf_text = redact_text_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat)
        out_text_file_path = "output/" + file_path_without_ext + "_result_as_text.pdf"
        pdf_text.save(out_text_file_path)

        out_file_paths.append(out_text_file_path)


        

    else:
        out_message = "No redaction method selected"
        print(out_message)
        return out_message, out_file_paths
    
    toc = time.perf_counter()
    out_time = f"Time taken: {toc - tic:0.1f} seconds."
    print(out_time)

    out_message = out_message + "\n\n" + out_time

    return out_message, out_file_paths


def redact_image_pdf(file_path:str, language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, progress=Progress(track_tqdm=True)):
    '''
    take an path for an image of a document, then run this image through the Presidio ImageAnalyzer to get a redacted page back
    '''

    out_message = "Converting pages to image"
    print(out_message)
    progress(0, desc=out_message)

    image_paths = process_file(file_path)

    # Create a new PDF
    #pdf = pikepdf.new()

    images = []
    number_of_pages = len(image_paths)

    out_message = "Redacting pages"
    print(out_message)
    progress(0.1, desc=out_message)

    for i in progress.tqdm(range(0,number_of_pages), total=number_of_pages, unit="pages", desc="Redacting pages"):

        print("Redacting page ", str(i + 1))

        # Get the image to redact using PIL lib (pillow)
        image = image_paths[i] #Image.open(image_paths[i])

        # %%
        image_analyser = ImageAnalyzerEngine(nlp_analyser)
        engine = ImageRedactorEngine(image_analyser)

        if language == 'en':
            ocr_lang = 'eng'
        else: ocr_lang = language

        # %%
        # Redact the image with pink color
        redacted_image = engine.redact(image,
            fill=(0, 0, 0),
            ocr_kwargs={"lang": ocr_lang},
            allow_list=allow_list,
            ad_hoc_recognizers= None,
            **{
                "language": language,
                "entities": chosen_redact_entities,
                "score_threshold": score_threshold
            },
            )
        
        images.append(redacted_image)

    return images

def redact_text_pdf(filename:str, language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, progress=Progress(track_tqdm=True)):
    '''
    Redact chosen entities from a pdf that is made up of multiple pages that are not images.
    '''
    
    combined_analyzer_results = []
    analyser_explanations = []
    annotations_all_pages = []
    analyzed_bounding_boxes_df = pd.DataFrame()

    pdf = Pdf.open(filename)

    for page_num, page in progress.tqdm(enumerate(pdf.pages), total=len(pdf.pages), unit="pages", desc="Redacting pages"):

        print("Page number is: ", page_num)

        annotations_on_page = []
        analyzed_bounding_boxes = []

        for page_layout in extract_pages(filename, page_numbers = [page_num], maxpages=1):
            analyzer_results = []

            for text_container in page_layout:
                if isinstance(text_container, LTTextContainer):
                    text_to_analyze = text_container.get_text()

                    analyzer_results = []
                    characters = []

                    analyzer_results = nlp_analyser.analyze(text=text_to_analyze,
                                                            language=language, 
                                                            entities=chosen_redact_entities,
                                                            score_threshold=score_threshold,
                                                            return_decision_process=False,
                                                            allow_list=allow_list)

                        #if analyzer_results:
                        #    pass
                        #explanation = analyzer_results[0].analysis_explanation.to_dict()
                        #analyser_explanations.append(explanation)
                    characters = [char                    # This is what we want to include in the list
                            for line in text_container          # Loop through each line in text_container
                            if isinstance(line, LTTextLine)    # Check if the line is an instance of LTTextLine
                            for char in line]                   # Loop through each character in the line
                            #if isinstance(char, LTChar)]  # Check if the character is not an instance of LTAnno #isinstance(char, LTChar) or


                    #print(characters)

                    # Collect unique types
                    # unique_types = set()

                    # for line in text_container:
                    #     if isinstance(line, LTTextLine):
                    #         print("Line: ", line)
                    #         for char in line:
                    #             unique_types.add(type(char))
                    #             if isinstance(char, LTAnno):
                    #                 print(char)

                    # # Print the unique types
                    # print("Unique types in text_container:")
                    # for t in unique_types:
                    #     print(t)
                    
                    # If any results found
                    print(analyzer_results)

                    if len(analyzer_results) > 0 and len(characters) > 0:
                        analyzed_bounding_boxes.extend({"boundingBox": char.bbox, "result": result} for result in analyzer_results for char in characters[result.start:result.end] if isinstance(char, LTChar))
                        combined_analyzer_results.extend(analyzer_results)

            if len(analyzer_results) > 0:
                # Create summary df of annotations to be made
                analyzed_bounding_boxes_df_new = pd.DataFrame(analyzed_bounding_boxes)
                analyzed_bounding_boxes_df_text = analyzed_bounding_boxes_df_new['result'].astype(str).str.split(",",expand=True).replace(".*: ", "", regex=True)
                analyzed_bounding_boxes_df_text.columns = ["type", "start", "end", "score"]
                analyzed_bounding_boxes_df_new = pd.concat([analyzed_bounding_boxes_df_new, analyzed_bounding_boxes_df_text], axis = 1)
                analyzed_bounding_boxes_df_new['page'] = page_num + 1
                analyzed_bounding_boxes_df = pd.concat([analyzed_bounding_boxes_df, analyzed_bounding_boxes_df_new], axis = 0)

            for analyzed_bounding_box in analyzed_bounding_boxes:
                bounding_box = analyzed_bounding_box["boundingBox"]
                annotation = Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Highlight,
                    QuadPoints=[bounding_box[0], bounding_box[3], bounding_box[2], bounding_box[3], bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[1]],
                    Rect=[bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
                    C=[0, 0, 0],
                    CA=1, # Transparency
                    T=analyzed_bounding_box["result"].entity_type
                )
                annotations_on_page.append(annotation)

            annotations_all_pages.extend([annotations_on_page])
   
            print("For page number: ", page_num, " there are ", len(annotations_all_pages[page_num]), " annotations")
            page.Annots = pdf.make_indirect(annotations_on_page)

        # Extracting data from dictionaries
        # extracted_data = []
        # for item in annotations_all_pages:
        #     temp_dict = {}
        #     #print(item)
        #     for key, value in item.items():
        #         if isinstance(value, Decimal):
        #             temp_dict[key] = float(value)
        #         elif isinstance(value, list):
        #             temp_dict[key] = [float(v) if isinstance(v, Decimal) else v for v in value]
        #         else:
        #             temp_dict[key] = value
        #     extracted_data.append(temp_dict)

        # Creating DataFrame
        # annotations_out = pd.DataFrame(extracted_data)
        #print(df)

        #annotations_out.to_csv("examples/annotations.csv")
    
    analyzed_bounding_boxes_df.to_csv("output/annotations_made.csv")

    return pdf