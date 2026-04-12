"""YOLO + EasyOCR benchmark pipelines."""

from __future__ import annotations

from detection import YOLODetector
from recognition import EasyOCRRecognizer

from .base_pipeline import BaseDigitPipeline


class YOLOEasyOCRPipeline(BaseDigitPipeline):
    """Two-stage detector + EasyOCR recognizer pipeline."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        display_name = f"YOLO + EasyOCR ({yolo_model_path})"
        pipeline_id = f"{yolo_model_path.replace('.pt', '').lower()}_easyocr"
        
        # Use lower confidence threshold for custom SVHN models
        if "svhn" in yolo_model_path:
            confidence_threshold = 0.1
        else:
            confidence_threshold = 0.3
            
        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(model_path=yolo_model_path, confidence_threshold=confidence_threshold),
            recognizer=EasyOCRRecognizer(),
            category="Detection + OCR",
            description="YOLO digit detection followed by EasyOCR recognition.",
        )


class YOLO26nEasyOCRPipeline(YOLOEasyOCRPipeline):
    """Custom YOLO26n detector with EasyOCR recognition."""

    def __init__(self):
        super().__init__("yolo26n.pt")


class YOLOv8nEasyOCRPipeline(YOLOEasyOCRPipeline):
    """Official YOLOv8n detector with EasyOCR recognition."""

    def __init__(self):
        super().__init__("yolov8n.pt")
