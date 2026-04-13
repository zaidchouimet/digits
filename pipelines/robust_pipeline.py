"""Enhanced pipeline with better handling for uploaded images and SVHN data.

Key fixes vs original:
- RobustDigitPipeline now also falls back to full-frame OCR when YOLO
  finds boxes but ALL recognitions come back empty (not just when 0 boxes).
- RobustOCRPipeline uses the improved SmartOCROnlyPipeline base so it
  benefits from smart pre-cropping and the new adaptive preprocessing.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from detection import FullFrameDetector, YOLODetector
from recognition import TesseractRecognizer, EasyOCRRecognizer
from pipelines.base_pipeline import BasePipeline, DetectionResult, PipelineResult
from pipelines.ocr_only import SmartOCROnlyPipeline


class RobustDigitPipeline(BasePipeline):
    """Enhanced pipeline with better uploaded image handling."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        display_name = f"Robust YOLO + Tesseract ({yolo_model_path})"
        pipeline_id = f"robust_{yolo_model_path.replace('.pt', '').lower()}_tesseract"

        self.yolo_detector = YOLODetector(model_path=yolo_model_path, confidence_threshold=0.05)
        self.full_frame_detector = FullFrameDetector()

        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=self.yolo_detector,
            recognizer=TesseractRecognizer(),
            category="Detection + OCR",
            description="Enhanced YOLO detection with full-frame fallback for uploaded images.",
        )

    def run(self, frame: np.ndarray) -> PipelineResult:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        detection = self.detect(frame)
        crops, valid_indices = self._crop_regions(frame, detection.boxes)
        recognitions = self.recognize(crops)

        # FIX: if YOLO found boxes but recognition still returned all-empty
        # strings, fall back to full-frame OCR on the whole image.
        if not any(r.text for r in recognitions):
            from pipelines.ocr_only import _smart_crop
            smart = _smart_crop(frame)
            h, w = smart.shape[:2]
            fallback_det = DetectionResult(
                boxes=[(0, 0, w, h)], scores=[1.0], image=smart, labels=["full_frame"]
            )
            fallback_recog = self.recognize([smart])
            if any(r.text for r in fallback_recog):
                detection = fallback_det
                recognitions = fallback_recog
                valid_indices = [0]
                frame = smart

        valid_detection = DetectionResult(
            boxes=[detection.boxes[i] for i in valid_indices],
            scores=[detection.scores[i] for i in valid_indices] if detection.scores else [],
            image=detection.image,
            labels=[detection.labels[i] for i in valid_indices] if detection.labels else [],
        )
        annotated_image = self._annotate(frame, valid_detection, recognitions)
        full_text = self._combine_recognitions(recognitions)
        processing_time = time.time() - started_at

        return PipelineResult(
            detection=detection,
            recognitions=recognitions,
            full_text=full_text,
            processing_time=processing_time,
            annotated_image=annotated_image,
            metadata={"pipeline_id": self.pipeline_id, "pipeline_name": self.name},
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        try:
            yolo_detections = self.yolo_detector.detect(frame)
            if yolo_detections:
                yolo_detections.sort(key=lambda x: x.score, reverse=True)
                return self._to_detection_result(frame, yolo_detections)
        except Exception as e:
            print(f"YOLO detection failed: {e}")

        full_frame_detections = self.full_frame_detector.detect(frame)
        return self._to_detection_result(frame, full_frame_detections)

    def _to_detection_result(self, frame: np.ndarray, detections) -> DetectionResult:
        if not detections:
            return DetectionResult(boxes=[], scores=[], image=frame, labels=[])
        return DetectionResult(
            boxes=[d.as_tuple() for d in detections],
            scores=[float(d.score) for d in detections],
            image=frame,
            labels=[d.label for d in detections],
        )


class RobustOCRPipeline(SmartOCROnlyPipeline):
    """Robust OCR-only pipeline — uses smart pre-cropping + EasyOCR.

    Changed from Tesseract to EasyOCR as the default because EasyOCR handles
    natural-image backgrounds (SVHN, photos) better out of the box.
    """

    def __init__(self):
        super().__init__(
            pipeline_id="robust_ocr_only",
            name="Robust OCR (End-to-End)",
            recognizer=EasyOCRRecognizer(),
            description="Enhanced OCR pipeline with smart pre-cropping and EasyOCR recognition.",
        )