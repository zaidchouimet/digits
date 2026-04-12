"""PaddleOCR backend with automatic EasyOCR fallback."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr
from .easyocr_recognizer import EasyOCRRecognizer


LOGGER = logging.getLogger(__name__)


class PaddleOCRRecognizer(BaseRecognizer):
    """Digit recognizer that prefers PaddleOCR and falls back to EasyOCR."""

    def __init__(self, fallback_recognizer: Optional[BaseRecognizer] = None):
        super().__init__("paddleocr")
        self.fallback_recognizer = fallback_recognizer or EasyOCRRecognizer()
        self.using_fallback = False

    def load(self) -> bool:
        self.using_fallback = False
        try:
            from paddleocr import PaddleOCR

            self.model = PaddleOCR(lang="en", show_log=False, use_angle_cls=False)
            self.is_loaded = True
            self.last_error = None
            self.active_backend_name = "paddleocr"
            return True
        except Exception as error:
            LOGGER.warning(
                "PaddleOCR could not be loaded. Falling back to EasyOCR. Reason: %s",
                error,
            )
            self.last_error = f"PaddleOCR unavailable: {error}"
            fallback_loaded = self.fallback_recognizer.load()
            self.is_loaded = fallback_loaded
            self.using_fallback = fallback_loaded
            self.active_backend_name = self.fallback_recognizer.active_backend_name
            if fallback_loaded:
                return True

            self.last_error = (
                f"{self.last_error}. EasyOCR fallback also failed: "
                f"{self.fallback_recognizer.last_error}"
            )
            return False

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        if self.using_fallback:
            return self.fallback_recognizer.recognize_batch(crops, single_char=single_char)

        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.last_error or "PaddleOCR recognizer is not loaded")

        recognitions: List[RecognitionOutput] = []
        for crop in crops:
            if crop.size == 0:
                recognitions.append(RecognitionOutput("", 0.0, crop, self.active_backend_name))
                continue

            prepared = preprocess_crop_for_ocr(crop)
            try:
                results = self.model.ocr(prepared, cls=False)
            except Exception:
                results = []

            if results and results[0]:
                texts = []
                confidences = []
                for result in results[0]:
                    if len(result) < 2 or not result[1]:
                        continue
                    text_info = result[1]
                    texts.append(extract_digit_text(text_info[0]))
                    confidences.append(float(text_info[1]) if len(text_info) > 1 else 0.0)

                filtered_texts = [text for text in texts if text]
                text = "".join(filtered_texts)
                confidence = float(np.mean(confidences)) if confidences else 0.0
            else:
                text = ""
                confidence = 0.0

            recognitions.append(RecognitionOutput(text, confidence, crop, self.active_backend_name))

        return recognitions
