"""Detection backends for the digit benchmark system."""

from .base import BaseDetector, DetectedBox
from .full_frame import FullFrameDetector
from .yolo import YOLODetector
from .efficientdet import EfficientDetDetector

__all__ = [
    "BaseDetector",
    "DetectedBox",
    "FullFrameDetector",
    "YOLODetector",
    "EfficientDetDetector",
]
