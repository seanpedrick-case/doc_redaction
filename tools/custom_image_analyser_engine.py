import pytesseract
from PIL import Image
import numpy as np
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    left: int
    top: int
    width: int
    height: int

@dataclass
class CustomImageRecognizerResult:
    entity_type: str
    start: int
    end: int
    score: float
    left: int
    top: int
    width: int
    height: int
    text: str

class CustomImageAnalyzerEngine:
    def __init__(
        self,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        tesseract_config: Optional[str] = None
    ):
        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()
        self.analyzer_engine = analyzer_engine
        self.tesseract_config = tesseract_config or '--oem 3 --psm 11'

    def perform_ocr(self, image: Union[str, Image.Image, np.ndarray]) -> List[OCRResult]:
        # Ensure image is a PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=self.tesseract_config)
        
        # Filter out empty strings and low confidence results
        valid_indices = [i for i, text in enumerate(ocr_data['text']) if text.strip() and int(ocr_data['conf'][i]) > 0]
        
        return [
            OCRResult(
                text=ocr_data['text'][i],
                left=ocr_data['left'][i],
                top=ocr_data['top'][i],
                width=ocr_data['width'][i],
                height=ocr_data['height'][i]
            )
            for i in valid_indices
        ]

    def analyze_text(
        self, 
        ocr_results: List[OCRResult], 
        **text_analyzer_kwargs
    ) -> List[CustomImageRecognizerResult]:
        # Define English as default language, if not specified
        if "language" not in text_analyzer_kwargs:
            text_analyzer_kwargs["language"] = "en"
        
        allow_list = text_analyzer_kwargs.get('allow_list', [])
        combined_results = []

        for ocr_result in ocr_results:
            # Analyze each OCR result (line) individually
            analyzer_result = self.analyzer_engine.analyze(
                text=ocr_result.text, **text_analyzer_kwargs
            )
            
            for result in analyzer_result:
                # Extract the relevant portion of text based on start and end
                relevant_text = ocr_result.text[result.start:result.end]
                
                # Create a new OCRResult with the relevant text and adjusted position
                relevant_ocr_result = OCRResult(
                    text=relevant_text,
                    left=ocr_result.left + self.estimate_x_offset(ocr_result.text, result.start),
                    top=ocr_result.top,
                    width=self.estimate_width(ocr_result, result.start, result.end),
                    height=ocr_result.height
                )
                
                # Map the analyzer results to bounding boxes for this line
                line_results = self.map_analyzer_results_to_bounding_boxes(
                    [result], [relevant_ocr_result], relevant_text, allow_list
                )
                
                combined_results.extend(line_results)

        return combined_results

    @staticmethod
    def map_analyzer_results_to_bounding_boxes(
        text_analyzer_results: List[RecognizerResult],
        ocr_results: List[OCRResult],
        full_text: str,
        allow_list: List[str],
    ) -> List[CustomImageRecognizerResult]:
        pii_bboxes = []
        text_position = 0

        for ocr_result in ocr_results:
            word_end = text_position + len(ocr_result.text)
            
            for result in text_analyzer_results:
                if (max(text_position, result.start) < min(word_end, result.end)) and (ocr_result.text not in allow_list):
                    pii_bboxes.append(
                        CustomImageRecognizerResult(
                            entity_type=result.entity_type,
                            start=result.start,
                            end=result.end,
                            score=result.score,
                            left=ocr_result.left,
                            top=ocr_result.top,
                            width=ocr_result.width,
                            height=ocr_result.height,
                            text=ocr_result.text
                        )
                    )
                    break
            
            text_position = word_end + 1  # +1 for the space between words

        return pii_bboxes

    @staticmethod
    def estimate_x_offset(full_text: str, start: int) -> int:
        # Estimate the x-offset based on character position
        # This is a simple estimation and might need refinement for variable-width fonts
        return int(start / len(full_text) * len(full_text))

    @staticmethod
    def estimate_width(ocr_result: OCRResult, start: int, end: int) -> int:
        # Estimate the width of the relevant text portion
        full_width = ocr_result.width
        full_length = len(ocr_result.text)
        return int((end - start) / full_length * full_width)

# Function to combine OCR results into line-level results
def combine_ocr_results(ocr_results, x_threshold = 20, y_threshold = 10):
    # Sort OCR results by 'top' to ensure line order
    ocr_results = sorted(ocr_results, key=lambda x: (x.top, x.left))
    
    combined_results = []
    current_line = []
    current_bbox = None

    for result in ocr_results:
        if not current_line:
            # Start a new line
            current_line.append(result)
            current_bbox = result
        else:
            # Check if the result is on the same line (y-axis) and close horizontally (x-axis)
            last_result = current_line[-1]
            if abs(result.top - last_result.top) <= y_threshold and \
               (result.left - (last_result.left + last_result.width)) <= x_threshold:
                # Update the bounding box to include the new word
                new_right = max(current_bbox.left + current_bbox.width, result.left + result.width)
                current_bbox = OCRResult(
                    text=f"{current_bbox.text} {result.text}",
                    left=current_bbox.left,
                    top=current_bbox.top,
                    width=new_right - current_bbox.left,
                    height=max(current_bbox.height, result.height)
                )
                current_line.append(result)
            else:
                # Commit the current line and start a new one
                combined_results.append(current_bbox)
                current_line = [result]
                current_bbox = result

    # Append the last line
    if current_bbox:
        combined_results.append(current_bbox)

    return combined_results