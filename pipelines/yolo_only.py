"""Pipeline that relies strictly on YOLO labels without a separate OCR step."""

from __future__ import annotations

import time
from typing import List

import numpy as np

from detection import YOLODetector
from recognition.base import BaseRecognizer, RecognitionOutput

from .base_pipeline import BasePipeline, DetectionResult, PipelineResult, RecognitionResult


class DummyRecognizer(BaseRecognizer):
    """A dummy recognizer to satisfy BasePipeline interface."""
    
    def __init__(self):
        super().__init__("None")
        self.is_loaded = True
        
    def load(self) -> bool:
        self.is_loaded = True
        return True
        
    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        return []


class YOLOOnlyPipeline(BasePipeline):
    """Uses YOLO detections and their corresponding class labels directly, skipping OCR."""

    def __init__(self, yolo_model: str):
        display_name = f"YOLO Only ({yolo_model})"
        pipeline_id = f"{yolo_model.replace('.pt', '').lower()}_only"

        is_plate_model = False
        if "best2" in yolo_model:
            confidence_threshold = 0.20
        elif "best3" in yolo_model:
            confidence_threshold = 0.05
            is_plate_model = True
        elif "best1" in yolo_model:
            confidence_threshold = 0.50
        elif "best" in yolo_model:
            confidence_threshold = 0.50
            is_plate_model = True
        elif "yolov8s" in yolo_model:
            confidence_threshold = 0.70
        else:
            confidence_threshold = 0.30

        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(
                model_path=yolo_model,
                confidence_threshold=confidence_threshold,
                is_plate_model=is_plate_model,
            ),
            recognizer=DummyRecognizer(),
            category="Detection Only",
            description="YOLO digit detection reading class labels directly without second-stage OCR.",
        )

    def run(self, frame: np.ndarray) -> PipelineResult:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        
        detection = self.detect(frame)
        
        # Build RecognitionResults directly from the detection labels
        recognitions = []
        frame_h, frame_w = frame.shape[:2]
        
        for i, box in enumerate(detection.boxes):
            label = detection.labels[i] if i < len(detection.labels) else ""
            score = detection.scores[i] if i < len(detection.scores) else 0.0
            
            x1, y1, x2, y2 = box
            cy1, cy2 = max(0, y1), min(frame_h, y2)
            cx1, cx2 = max(0, x1), min(frame_w, x2)
            crop = frame[cy1:cy2, cx1:cx2]
            
            recognitions.append(
                RecognitionResult(
                    text=label,
                    confidence=float(score),
                    crop=crop,
                    backend="YOLO Class Label",
                )
            )
            
        annotated_image = self._annotate(frame, detection, recognitions)
        full_text = self._combine_recognitions(recognitions)
        processing_time = time.time() - started_at

        return PipelineResult(
            detection=detection,
            recognitions=recognitions,
            full_text=full_text,
            processing_time=processing_time,
            annotated_image=annotated_image,
            metadata={
                "pipeline_id": self.pipeline_id,
                "detector": self.detector.name,
                "recognizer": "None",
                "recognizer_backend": "YOLO Label",
                "category": self.category,
            },
        )
