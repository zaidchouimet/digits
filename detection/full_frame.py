"""Detection backend that forwards the entire frame as one region."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseDetector, DetectedBox


class FullFrameDetector(BaseDetector):
    """Treat the full image as a single OCR region."""

    def __init__(self):
        super().__init__("full_frame")

    def load(self) -> bool:
        self.is_loaded = True
        self.last_error = None
        return True

    def detect(self, frame: np.ndarray) -> List[DetectedBox]:
        height, width = frame.shape[:2]
        return [DetectedBox(0, 0, width, height, score=1.0, label="full_frame")]
