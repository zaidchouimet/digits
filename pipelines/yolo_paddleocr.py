"""YOLO + PaddleOCR benchmark pipelines with EasyOCR fallback."""

from __future__ import annotations

from detection import YOLODetector
from recognition import PaddleOCRRecognizer

from .base_pipeline import BaseDigitPipeline


class YOLOPaddleOCRPipeline(BaseDigitPipeline):
    """Two-stage detector + PaddleOCR recognizer pipeline."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        display_name = f"YOLO + PaddleOCR ({yolo_model_path})"
        pipeline_id = f"{yolo_model_path.replace('.pt', '').lower()}_paddleocr"
        
        # Use lower confidence threshold for custom SVHN models
        if "svhn" in yolo_model_path:
            confidence_threshold = 0.1
        else:
            confidence_threshold = 0.3
            
        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(model_path=yolo_model_path, confidence_threshold=confidence_threshold),
            recognizer=PaddleOCRRecognizer(),
            category="Detection + OCR",
            description=(
                "YOLO digit detection followed by PaddleOCR recognition. "
                "If PaddleOCR is unavailable, EasyOCR is used automatically."
            ),
        )


class YOLO26nPaddleOCRPipeline(YOLOPaddleOCRPipeline):
    """Custom YOLO26n detector with PaddleOCR recognition."""

    def __init__(self):
        super().__init__("yolo26n.pt")


class YOLOv8nPaddleOCRPipeline(YOLOPaddleOCRPipeline):
    """Official YOLOv8n detector with PaddleOCR recognition."""

    def __init__(self):
        super().__init__("yolov8n.pt")
