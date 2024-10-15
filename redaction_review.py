import gradio as gr
from gradio_image_annotation import image_annotator
from gradio_image_annotation.image_annotator import AnnotatedImageData

from tools.file_conversion import is_pdf, convert_pdf_to_images
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_redaction import redact_page_with_pymupdf
import json
import pymupdf
from PIL import ImageDraw, Image

file_path = "output/page_as_img_example_complaint_letter_pages_1.png"
#file_path = "examples/graduate-job-example-cover-letter.pdf"


if is_pdf(file_path):
    images = convert_pdf_to_images(file_path)
    image = images[0]
    doc = pymupdf.open(file_path)
else:
    doc = []

with open('output/gradio_annotation_boxes.json', 'r') as f:
    gradio_annotation_boxes = json.load(f)

example_annotation = {
    "image": file_path,
    "boxes": gradio_annotation_boxes
}

def apply_redactions(image_annotated:AnnotatedImageData, file_path:str, doc=[]):
    #print(image_annotated['image'])

    file_base = get_file_path_end(file_path)

    image = Image.fromarray(image_annotated['image'].astype('uint8'))

    draw = ImageDraw.Draw(image)

    if is_pdf(file_path) == False:
        for img_annotation_box in image_annotated['boxes']:
            coords = [img_annotation_box["xmin"],
            img_annotation_box["ymin"],
            img_annotation_box["xmax"],
            img_annotation_box["ymax"]]

            fill = img_annotation_box["color"]

            draw.rectangle(coords, fill=fill)

            image.save(output_folder + file_base + "_additional.png")

    # If it's a pdf, assume a doc object is available
    else:
        doc = redact_page_with_pymupdf(doc, image_annotated, 1, image)


def crop(annotations):
    if annotations["boxes"]:
        box = annotations["boxes"][0]
        return annotations["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]
        ]
    return None

def get_boxes_json(annotations):
    return annotations["boxes"]

with gr.Blocks() as demo:
    with gr.Tab("Object annotation", id="tab_object_annotation"):

        doc_state = gr.State(doc)

        file_path_textbox = gr.Textbox(value=file_path)
        annotator = image_annotator(
            example_annotation,
            label_list=["Redaction"],
            label_colors=[(0, 0, 0)],
        )
        button_get = gr.Button("Get bounding boxes")
        button_apply = gr.Button("Apply redactions")
        json_boxes = gr.JSON()
        button_get.click(get_boxes_json, annotator, json_boxes)
        button_apply.click(apply_redactions, inputs=[annotator, file_path_textbox, doc_state])

if __name__ == "__main__":
    demo.launch(inbrowser=True)