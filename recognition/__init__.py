"""Recognition backends for the digit benchmark system."""

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr, preprocess_for_easyocr
from .easyocr_recognizer import EasyOCRRecognizer
from .tesseract_recognizer import TesseractRecognizer

__all__ = [
    "BaseRecognizer",
    "RecognitionOutput",
    "extract_digit_text",
    "preprocess_crop_for_ocr",
    "preprocess_for_easyocr",
    "EasyOCRRecognizer",
    "TesseractRecognizer",
]