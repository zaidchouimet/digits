"""Unified pipeline abstractions for digit recognition benchmarking."""

from __future__ import annotations

import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection import BaseDetector, DetectedBox, FullFrameDetector
from recognition import BaseRecognizer, RecognitionOutput, extract_digit_text


@dataclass
class DetectionResult:
    """Grouped detection output for one frame."""

    boxes: List[Tuple[int, int, int, int]]
    scores: List[float]
    image: np.ndarray
    labels: List[str] = field(default_factory=list)


@dataclass
class RecognitionResult:
    """Recognition output for one crop."""

    text: str
    confidence: float
    crop: np.ndarray
    backend: str = ""


@dataclass
class PipelineResult:
    """Complete pipeline output for a single frame."""

    detection: DetectionResult
    recognitions: List[RecognitionResult]
    full_text: str
    processing_time: float
    annotated_image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePipeline(ABC):
    """Base class for academically valid digit recognition pipelines."""

    def __init__(
        self,
        pipeline_id: str,
        name: str,
        detector: Optional[BaseDetector],
        recognizer: BaseRecognizer,
        category: str,
        description: str,
        crop_padding: int = 4,
    ):
        self.pipeline_id = pipeline_id
        self.name = name
        self.detector = detector or FullFrameDetector()
        self.recognizer = recognizer
        self.category = category
        self.description = description
        self.crop_padding = crop_padding
        self.is_loaded = False
        self.last_error: Optional[str] = None

        self.detection_model = getattr(self.detector, "model", None)
        self.recognition_model = getattr(self.recognizer, "model", None)

    def load_models(self) -> bool:
        """Load detector and recognizer backends."""
        detector_loaded = self.detector.load()
        recognizer_loaded = self.recognizer.load()
        self.detection_model = getattr(self.detector, "model", None)
        self.recognition_model = getattr(self.recognizer, "model", None)

        self.is_loaded = detector_loaded and recognizer_loaded
        if self.is_loaded:
            self.last_error = None
            return True

        errors = [message for message in (self.detector.last_error, self.recognizer.last_error) if message]
        self.last_error = " | ".join(errors) if errors else f"Failed to load pipeline {self.name}"
        return False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect digit regions in the frame."""
        detections = self.detector.detect(frame)
        # Sort by Y first (top-to-bottom), then X (left-to-right) for multi-row layouts
        detections.sort(key=lambda item: (item.y1, item.x1))
        return DetectionResult(
            boxes=[detection.as_tuple() for detection in detections],
            scores=[float(detection.score) for detection in detections],
            image=frame,
            labels=[detection.label for detection in detections],
        )

    def recognize(self, crops: List[np.ndarray]) -> List[RecognitionResult]:
        """Recognize digits from a batch of cropped regions."""
        single_char = not isinstance(self.detector, FullFrameDetector)
        outputs = self.recognizer.recognize_batch(crops, single_char=single_char)
        return [
            RecognitionResult(
                text=extract_digit_text(output.text),
                confidence=float(output.confidence),
                crop=output.crop,
                backend=output.backend,
            )
            for output in outputs
        ]

    def run(self, frame: np.ndarray) -> PipelineResult:
        """Execute the full pipeline on one frame."""
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        detection = self.detect(frame)
        crops, valid_indices = self._crop_regions(frame, detection.boxes)
        recognitions = self.recognize(crops)
        # Build a detection view aligned with recognitions (only boxes that
        # produced valid non-empty crops) so _annotate never index-errors.
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
            metadata={
                "pipeline_id": self.pipeline_id,
                "detector": self.detector.name,
                "recognizer": self.recognizer.name,
                "recognizer_backend": self.recognizer.active_backend_name,
                "category": self.category,
            },
        )

    def process_image(self, image: np.ndarray, debug: bool = False) -> PipelineResult:
        """Compatibility wrapper used by the earlier project layout."""
        result = self.run(image)
        if debug:
            print(f"{self.name}: detected {len(result.detection.boxes)} regions, prediction='{result.full_text}'")
        return result

    def process_batch(self, images: List[np.ndarray]) -> List[PipelineResult]:
        """Process multiple images while preserving benchmark robustness."""
        results = []
        for image in images:
            try:
                results.append(self.run(image))
            except Exception:
                results.append(
                    PipelineResult(
                        detection=DetectionResult([], [], image, []),
                        recognitions=[],
                        full_text="",
                        processing_time=0.0,
                        annotated_image=image,
                        metadata={"pipeline_id": self.pipeline_id, "error": self.last_error},
                    )
                )
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return structured pipeline metadata."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.name,
            "category": self.category,
            "description": self.description,
            "is_loaded": self.is_loaded,
            "detector": self.detector.get_info(),
            "recognizer": self.recognizer.get_info(),
        }

    def _crop_regions(
        self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Return (crops, valid_indices) keeping only non-empty crops.

        Returns the *integer indices* of boxes that produced a valid crop so
        that score/label arrays can be sliced directly without a fragile
        .index() call.  This ensures len(crops) == len(recognitions) always.
        """
        crops: List[np.ndarray] = []
        valid_indices: List[int] = []
        frame_height, frame_width = frame.shape[:2]

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            padded_x1 = max(0, x1 - self.crop_padding)
            padded_y1 = max(0, y1 - self.crop_padding)
            padded_x2 = min(frame_width, x2 + self.crop_padding)
            padded_y2 = min(frame_height, y2 + self.crop_padding)
            crop = frame[padded_y1:padded_y2, padded_x1:padded_x2]
            if crop.size > 0:
                crops.append(crop)
                valid_indices.append(idx)

        return crops, valid_indices

    def _combine_recognitions(self, recognitions: List[RecognitionResult]) -> str:
        return "".join(result.text for result in recognitions if result.text)

    def _annotate(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        recognitions: List[RecognitionResult],
    ) -> np.ndarray:
        annotated = frame.copy()
        for index, box in enumerate(detection.boxes):
            x1, y1, x2, y2 = box
            text = recognitions[index].text if index < len(recognitions) else ""
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if text:
                cv2.putText(
                    annotated,
                    text,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (36, 255, 12),
                    2,
                )
        return annotated

