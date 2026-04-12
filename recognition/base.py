"""Base abstractions and helpers for OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


def extract_digit_text(text: str) -> str:
    """Keep only numeric characters from OCR output."""
    return "".join(character for character in text if character.isdigit())


def preprocess_crop_for_ocr(crop: np.ndarray, scale: int = 2) -> np.ndarray:
    """Apply enhanced preprocessing for small digit crops."""
    if crop.size == 0:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Step 1: Ensure minimum size - Tesseract needs at least 64px height
    h, w = gray.shape
    if h < 64:
        scale_factor = 64 / h
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply additional scaling if needed
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Step 3: Denoise
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # Step 4: Threshold - makes text black on white
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


@dataclass
class RecognitionOutput:
    """OCR output for one crop."""

    text: str
    confidence: float
    crop: np.ndarray
    backend: str


class BaseRecognizer(ABC):
    """Abstract OCR backend interface."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_loaded = False
        self.last_error: Optional[str] = None
        self.active_backend_name = name

    @abstractmethod
    def load(self) -> bool:
        """Load OCR runtime and required models."""

    @abstractmethod
    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        """Recognize text from a list of crops."""

    def get_info(self) -> dict:
        """Return recognizer metadata for reporting."""
        return {
            "name": self.name,
            "active_backend": self.active_backend_name,
            "is_loaded": self.is_loaded,
            "error": self.last_error,
        }
