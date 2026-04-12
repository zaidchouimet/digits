"""Recognition backends for the digit benchmark system."""

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr
from .easyocr_recognizer import EasyOCRRecognizer
from .tesseract_recognizer import TesseractRecognizer
from .paddleocr_recognizer import PaddleOCRRecognizer

__all__ = [
    "BaseRecognizer",
    "RecognitionOutput",
    "extract_digit_text",
    "preprocess_crop_for_ocr",
    "EasyOCRRecognizer",
    "TesseractRecognizer",
    "PaddleOCRRecognizer",
]
