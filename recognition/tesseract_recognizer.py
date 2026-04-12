"""Tesseract OCR backend."""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr


class TesseractRecognizer(BaseRecognizer):
    """Digit recognizer built on Tesseract OCR."""

    def __init__(self):
        super().__init__("tesseract")
        self.tesseract_module = None

    def load(self) -> bool:
        try:
            import pytesseract

            tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.name == "nt" and os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path

            pytesseract.get_tesseract_version()
            self.model = pytesseract
            self.tesseract_module = pytesseract
            self.is_loaded = True
            self.last_error = None
            self.active_backend_name = "tesseract"
            return True
        except Exception as error:
            self.is_loaded = False
            self.last_error = f"Failed to load Tesseract recognizer: {error}"
            return False

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        if not self.is_loaded or self.tesseract_module is None:
            raise RuntimeError(self.last_error or "Tesseract recognizer is not loaded")

        recognitions: List[RecognitionOutput] = []
        # Use psm 8 for single word (multi-digit numbers) or psm 10 for single character
        config = "--psm 10 -c tessedit_char_whitelist=0123456789" if single_char else (
            "--psm 8 -c tessedit_char_whitelist=0123456789"
        )

        for crop in crops:
            if crop.size == 0:
                recognitions.append(RecognitionOutput("", 0.0, crop, self.active_backend_name))
                continue

            prepared = preprocess_crop_for_ocr(crop)
            if len(prepared.shape) == 2:
                prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB)

            text = self.tesseract_module.image_to_string(prepared, config=config).strip()
            digits = extract_digit_text(text)
            confidence = 0.8 if digits else 0.0
            recognitions.append(RecognitionOutput(digits, confidence, crop, self.active_backend_name))

        return recognitions
