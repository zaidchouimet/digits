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
    """Apply enhanced preprocessing for digit crops.

    Key fix: use ADAPTIVE thresholding instead of Otsu global threshold.
    Otsu fails on SVHN images because the background is complex/colorful —
    adaptive thresholding handles uneven illumination and noisy backgrounds
    much better.  Also skip Otsu entirely for larger images where the
    background dominates and causes the threshold to invert text polarity.
    """
    if crop.size == 0:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape

    # Step 1: Ensure minimum height — Tesseract needs at least 64 px
    if h < 64:
        scale_factor = 64 / h
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor,
                          interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape

    # Step 2: Upscale
    resized = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # Step 3: CLAHE — improves local contrast before thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Step 4: Mild denoise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Step 5: DYNAMIC Adaptive threshold
    # If h > 500, we are likely looking at a full frame, not a small crop.
    # We increase the block size to ignore fine wall texture.
    dynamic_block = 15
    if h > 500:
        dynamic_block = 71

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        dynamic_block, 10,
    )

    # Step 6: Light morphological cleanup — fills small holes in digit strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


def preprocess_for_easyocr(crop: np.ndarray) -> np.ndarray:
    """Minimal preprocessing for EasyOCR (neural-network OCR).

    EasyOCR's deep-learning model was trained on NATURAL images.
    Applying binary thresholding (like the Tesseract pipeline does) destroys
    the image appearance and causes empty / garbage predictions.

    This function only resizes small crops and applies gentle bilateral
    denoising — preserving full colour and tone information so EasyOCR's
    network can recognise digits correctly.
    """
    if crop.size == 0:
        return crop

    # Ensure 3-channel colour image (EasyOCR prefers BGR/RGB)
    if len(crop.shape) == 2:
        image = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        image = crop.copy()

    # Upscale only when the crop is very small (EasyOCR struggles < 32 px tall)
    h, w = image.shape[:2]
    if h < 32:
        scale_factor = max(32 / h, 2.0)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                           interpolation=cv2.INTER_CUBIC)
    elif h < 64:
        image = cv2.resize(image, None, fx=2.0, fy=2.0,
                           interpolation=cv2.INTER_CUBIC)

    # Gentle bilateral denoise: preserves edges (digit strokes) while
    # smoothing background noise — does NOT binarize.
    image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    return image


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