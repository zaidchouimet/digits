"""Base abstractions for detection backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectedBox:
    """A single detected digit region."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    label: str = "digit"
    class_id: Optional[int] = None

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the box as an OpenCV-friendly tuple."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


class BaseDetector(ABC):
    """Abstract interface for detector implementations."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_loaded = False
        self.last_error: Optional[str] = None

    @abstractmethod
    def load(self) -> bool:
        """Load detector weights or runtime dependencies."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedBox]:
        """Detect digit regions in a BGR frame."""

    def get_info(self) -> dict:
        """Return detector metadata for benchmark reporting."""
        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
            "error": self.last_error,
        }
