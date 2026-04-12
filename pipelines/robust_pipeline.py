"""Enhanced pipeline with better handling for uploaded images."""

from __future__ import annotations

import time
import cv2
import numpy as np
from detection import YOLODetector, FullFrameDetector
from recognition import TesseractRecognizer
from pipelines.base_pipeline import BasePipeline, DetectionResult


class RobustDigitPipeline(BasePipeline):
    """Enhanced pipeline with better uploaded image handling."""
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        display_name = f"Robust YOLO + Tesseract ({yolo_model_path})"
        pipeline_id = f"robust_{yolo_model_path.replace('.pt', '').lower()}_tesseract"
        
        # Create detector with lower confidence threshold for uploaded images
        self.yolo_detector = YOLODetector(model_path=yolo_model_path, confidence_threshold=0.05)
        self.full_frame_detector = FullFrameDetector()
        
        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=self.yolo_detector,
            recognizer=TesseractRecognizer(),
            category="Detection + OCR",
            description="Enhanced YOLO detection with fallback for uploaded images.",
        )
    
    def run(self, frame: np.ndarray):
        """Override run to use our enhanced detect method."""
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        
        # Use OUR enhanced detect method, not the base class one
        detection = self.detect(frame)
        
        crops, valid_indices = self._crop_regions(frame, detection.boxes)
        recognitions = self.recognize(crops)
        
        # Build a detection view aligned with recognitions
        valid_detection = DetectionResult(
            boxes=[detection.boxes[i] for i in valid_indices],
            scores=[detection.scores[i] for i in valid_indices] if detection.scores else [],
            image=detection.image,
            labels=[detection.labels[i] for i in valid_indices] if detection.labels else [],
        )
        annotated_image = self._annotate(frame, valid_detection, recognitions)
        full_text = self._combine_recognitions(recognitions)
        processing_time = time.time() - started_at

        from pipelines.base_pipeline import PipelineResult
        return PipelineResult(
            detection=detection,
            recognitions=recognitions,
            full_text=full_text,
            processing_time=processing_time,
            annotated_image=annotated_image,
            metadata={
                "pipeline_id": self.pipeline_id,
                "pipeline_name": self.name,
            },
        )
    
    def detect(self, frame: np.ndarray):
        """Enhanced detection with fallback strategies."""
        
        # Try YOLO detection first
        try:
            yolo_detections = self.yolo_detector.detect(frame)
            if yolo_detections:
                # Sort by confidence and keep top detections
                yolo_detections.sort(key=lambda x: x.score, reverse=True)
                return self._convert_to_detection_result(frame, yolo_detections)
        except Exception as e:
            print(f"YOLO detection failed: {e}")
        
        # Fallback: try full-frame OCR if YOLO fails
        print("Falling back to full-frame detection")
        full_frame_detections = self.full_frame_detector.detect(frame)
        return self._convert_to_detection_result(frame, full_frame_detections)
    
    def _convert_to_detection_result(self, frame, detections):
        """Convert detections to DetectionResult format."""
        from pipelines.base_pipeline import DetectionResult
        
        if not detections:
            return DetectionResult(
                boxes=[],
                scores=[],
                image=frame,
                labels=[],
            )
        
        # Use the best detection or full frame
        if len(detections) > 1:
            # Multiple detections - use all of them
            boxes = [d.as_tuple() for d in detections]
            scores = [float(d.score) for d in detections]
            labels = [d.label for d in detections]
        else:
            # Single detection
            d = detections[0]
            boxes = [d.as_tuple()]
            scores = [float(d.score)]
            labels = [d.label]
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            image=frame,
            labels=labels,
        )


class RobustOCRPipeline(BasePipeline):
    """Robust OCR-only pipeline for uploaded images."""
    
    def __init__(self):
        super().__init__(
            pipeline_id="robust_ocr_only",
            name="Robust OCR (End-to-End)",
            detector=FullFrameDetector(),
            recognizer=TesseractRecognizer(),
            category="OCR-Only Baseline",
            description="Enhanced OCR pipeline optimized for uploaded images.",
            crop_padding=0,
        )
