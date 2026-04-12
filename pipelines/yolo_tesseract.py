"""YOLO + Tesseract benchmark pipelines."""

from __future__ import annotations

from detection import YOLODetector
from recognition import TesseractRecognizer

from .base_pipeline import BaseDigitPipeline


class YOLOTesseractPipeline(BaseDigitPipeline):
    """Two-stage detector + Tesseract recognizer pipeline."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        display_name = f"YOLO + Tesseract ({yolo_model_path})"
        pipeline_id = f"{yolo_model_path.replace('.pt', '').lower()}_tesseract"
        
        # Use lower confidence threshold for custom SVHN models
        if "svhn" in yolo_model_path:
            confidence_threshold = 0.1
        else:
            confidence_threshold = 0.3
            
        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(model_path=yolo_model_path, confidence_threshold=confidence_threshold),
            recognizer=TesseractRecognizer(),
            category="Detection + OCR",
            description="YOLO digit detection followed by classical Tesseract OCR.",
        )


class YOLO26nTesseractPipeline(YOLOTesseractPipeline):
    """Custom YOLO26n detector with Tesseract recognition."""

    def __init__(self):
        super().__init__("yolo26n.pt")


class YOLOv8nTesseractPipeline(YOLOTesseractPipeline):
    """Official YOLOv8n detector with Tesseract recognition."""

    def __init__(self):
        super().__init__("yolov8n.pt")
