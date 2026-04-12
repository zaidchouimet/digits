"""EasyOCR backend."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr


class EasyOCRRecognizer(BaseRecognizer):
    """Digit recognizer built on EasyOCR."""

    def __init__(self):
        super().__init__("easyocr")

    def load(self) -> bool:
        try:
            import easyocr

            self.model = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            self.is_loaded = True
            self.last_error = None
            self.active_backend_name = "easyocr"
            return True
        except Exception as error:
            self.is_loaded = False
            self.last_error = f"Failed to load EasyOCR recognizer: {error}"
            return False

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or "EasyOCR recognizer is not loaded")

        recognitions: List[RecognitionOutput] = []
        for crop in crops:
            if crop.size == 0:
                recognitions.append(RecognitionOutput("", 0.0, crop, self.active_backend_name))
                continue

            prepared = preprocess_crop_for_ocr(crop)
            try:
                results = self.model.readtext(
                    prepared,
                    allowlist="0123456789",
                    paragraph=not single_char,
                    detail=1,
                )
            except Exception:
                results = []

            if results and len(results) > 0 and len(results[0]) >= 3:
                text = extract_digit_text("".join(item[1] for item in results if len(item) >= 2))
                confidence = float(np.mean([item[2] for item in results if len(item) >= 3]))
            else:
                text = ""
                confidence = 0.0

            recognitions.append(RecognitionOutput(text, confidence, crop, self.active_backend_name))

        return recognitions
