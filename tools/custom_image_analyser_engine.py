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
        # Combine all OCR text
        full_text = ' '.join([result.text for result in ocr_results])
        
        # Define English as default language, if not specified
        if "language" not in text_analyzer_kwargs:
            text_analyzer_kwargs["language"] = "en"
        
        analyzer_result = self.analyzer_engine.analyze(
            text=full_text, **text_analyzer_kwargs
        )
        
        allow_list = text_analyzer_kwargs.get('allow_list', [])
        
        return self.map_analyzer_results_to_bounding_boxes(
            analyzer_result, ocr_results, full_text, allow_list
        )

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