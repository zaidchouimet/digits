"""Cascade pipeline: best3 plate detector followed by digit detector."""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from detection import YOLODetector
from recognition.base import BaseRecognizer, RecognitionOutput, extract_digit_text

from .base_pipeline import BasePipeline, DetectionResult, PipelineResult, RecognitionResult


class _NoopRecognizer(BaseRecognizer):
    """No-op recognizer used to satisfy BasePipeline interface."""

    def __init__(self) -> None:
        super().__init__("cascade_noop")
        self.is_loaded = True

    def load(self) -> bool:
        self.is_loaded = True
        self.last_error = None
        return True

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        return [RecognitionOutput(text="", confidence=0.0, crop=crop, backend="noop") for crop in crops]


class CascadePlatePipeline(BasePipeline):
    """Two-stage detector: best3 finds plate region, bestN reads digits."""

    def __init__(
        self,
        digit_model_path: str = "best2.pt",
        digit_confidence_threshold: float = 0.20,
    ) -> None:
        self.plate_detector = YOLODetector(
            model_path="best3.pt",
            confidence_threshold=0.05,
            is_plate_model=True,
            inference_imgsz=1280,
        )
        self.digit_model_path = digit_model_path
        self.digit_detector = YOLODetector(
            model_path=digit_model_path,
            confidence_threshold=digit_confidence_threshold,
            is_plate_model=False,
        )

        model_stub = digit_model_path.replace(".pt", "").lower()
        pipeline_id = f"cascade_best3_{model_stub}"
        display_name = f"Cascade: Best3 Plate + {model_stub.upper()} Digits"

        super().__init__(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=self.plate_detector,
            recognizer=_NoopRecognizer(),
            category="Detection + OCR",
            description=f"Two-stage: best3 detects plates, {digit_model_path} reads digits class-by-class.",
        )

    def load_models(self) -> bool:
        """Load both YOLO detectors."""
        plate_loaded = self.plate_detector.load()
        digit_loaded = self.digit_detector.load()
        recognizer_loaded = self.recognizer.load()
        self.is_loaded = plate_loaded and digit_loaded and recognizer_loaded
        if self.is_loaded:
            self.last_error = None
            return True
        errors = [msg for msg in (self.plate_detector.last_error, self.digit_detector.last_error) if msg]
        self.last_error = " | ".join(errors) if errors else "Failed to load cascade detectors."
        return False

    def run(self, frame: np.ndarray) -> PipelineResult:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        frame_h, frame_w = frame.shape[:2]

        plate_dets = self.plate_detector.detect(frame)
        if not plate_dets:
            empty_detection = DetectionResult(boxes=[], scores=[], image=frame, labels=[])
            return PipelineResult(
                detection=empty_detection,
                recognitions=[],
                full_text="",
                processing_time=time.time() - started_at,
                annotated_image=frame.copy(),
                metadata={
                    "pipeline_id": self.pipeline_id,
                    "plate_count": 0,
                    "digit_count": 0,
                },
            )

        best_plate = max(plate_dets, key=lambda item: float(item.score))
        px1, py1, px2, py2 = best_plate.as_tuple()
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(frame_w, px2)
        py2 = min(frame_h, py2)

        plate_crop = frame[py1:py2, px1:px2]
        digit_dets = self.digit_detector.detect(plate_crop) if plate_crop.size > 0 else []
        digit_dets.sort(key=lambda item: item.x1)

        full_boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        labels: List[str] = []
        recognitions: List[RecognitionResult] = []

        for digit in digit_dets:
            dx1, dy1, dx2, dy2 = digit.as_tuple()
            gx1 = max(0, min(frame_w, px1 + dx1))
            gy1 = max(0, min(frame_h, py1 + dy1))
            gx2 = max(0, min(frame_w, px1 + dx2))
            gy2 = max(0, min(frame_h, py1 + dy2))
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            crop = frame[gy1:gy2, gx1:gx2]
            label_text = extract_digit_text(digit.label) or str(digit.label)

            full_boxes.append((gx1, gy1, gx2, gy2))
            scores.append(float(digit.score))
            labels.append(label_text)
            recognitions.append(
                RecognitionResult(
                    text=label_text,
                    confidence=float(digit.score),
                    crop=crop,
                    backend=f"{self.digit_model_path}_class_label",
                )
            )

        detection = DetectionResult(boxes=full_boxes, scores=scores, image=frame, labels=labels)
        annotated = frame.copy()
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(
            annotated,
            "plate",
            (px1, max(py1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        annotated = self._annotate(annotated, detection, recognitions)
        full_text = self._combine_recognitions(recognitions)

        return PipelineResult(
            detection=detection,
            recognitions=recognitions,
            full_text=full_text,
            processing_time=time.time() - started_at,
            annotated_image=annotated,
            metadata={
                "pipeline_id": self.pipeline_id,
                "plate_count": len(plate_dets),
                "digit_count": len(recognitions),
            },
        )
