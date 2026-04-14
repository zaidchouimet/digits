"""Tesseract OCR backend — digits-only output with bbox-level filtering."""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr


def _longest_digit_run(text: str) -> str:
    """Return the longest contiguous run of digits in a string.

    Example: 'Street 8 Community 373' → '373'  (longest run)
    Example: '987652 114 16'          → '987652' but we want all…
    For plate numbers we join ALL digit runs separated only by spaces.
    """
    # Split on non-digit characters, keep non-empty runs
    runs = [r for r in ''.join(c if c.isdigit() else ' ' for c in text).split() if r]
    if not runs:
        return ''
    # If only one run, return it
    if len(runs) == 1:
        return runs[0]
    # If all runs are short (≤ 3 digits) except one, return the longest
    # (e.g. '8' vs '373' → return '373')
    # If multiple long runs exist, return the longest
    return max(runs, key=len)


class TesseractRecognizer(BaseRecognizer):
    """Digit recognizer built on Tesseract OCR."""

    def __init__(self):
        super().__init__('tesseract')
        self.tesseract_module = None

    def load(self) -> bool:
        try:
            import pytesseract
            tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.name == 'nt' and os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
            pytesseract.get_tesseract_version()
            self.model = pytesseract
            self.tesseract_module = pytesseract
            self.is_loaded = True
            self.last_error = None
            self.active_backend_name = 'tesseract'
            return True
        except Exception as error:
            self.is_loaded = False
            self.last_error = f'Failed to load Tesseract recognizer: {error}'
            return False

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        if not self.is_loaded or self.tesseract_module is None:
            raise RuntimeError(self.last_error or 'Tesseract recognizer is not loaded')

        recognitions: List[RecognitionOutput] = []

        if single_char:
            psm_order = [10, 8]
        else:
            psm_order = [11, 3, 7, 13, 8, 6]

        whitelist = ''

        for crop in crops:
            if crop.size == 0:
                recognitions.append(RecognitionOutput('', 0.0, crop, self.active_backend_name))
                continue

            prepared = preprocess_crop_for_ocr(crop)
            if len(prepared.shape) == 2:
                prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB)

            def try_tess(img: np.ndarray) -> str:
                for psm in psm_order:
                    config = f'--oem 3 --psm {psm}'
                    if whitelist:
                        config += f' -c {whitelist}'
                    try:
                        raw = self.tesseract_module.image_to_string(img, config=config).strip()
                        candidate = _longest_digit_run(raw)
                        if candidate:
                            return candidate
                    except Exception:
                        continue
                return ''

            # Pass 1: Normal image
            digits = try_tess(prepared)
            
            # Pass 2: Inverted image (essential for white-on-black signs)
            if not digits:
                inverted = cv2.bitwise_not(prepared)
                digits = try_tess(inverted)

            confidence = 0.8 if digits else 0.0
            recognitions.append(RecognitionOutput(digits, confidence, crop, self.active_backend_name))

        return recognitions