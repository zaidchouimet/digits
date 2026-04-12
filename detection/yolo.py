"""YOLO-based digit detection backend."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseDetector, DetectedBox


class YOLODetector(BaseDetector):
    """Digit detector powered by Ultralytics YOLO models."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.15):
        super().__init__(f"yolo:{model_path}")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.is_loaded = True
            self.last_error = None
            return True
        except Exception as error:
            self.is_loaded = False
            self.last_error = f"Failed to load YOLO detector '{self.model_path}': {error}"
            return False

    def detect(self, frame: np.ndarray) -> List[DetectedBox]:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Detector {self.name} is not loaded")

        try:
            results = self.model(frame, verbose=False)
        except Exception as error:
            raise RuntimeError(f"YOLO detection failed for '{self.model_path}': {error}") from error

        detections: List[DetectedBox] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        names = getattr(result, "names", {})

        for index, (box, score) in enumerate(zip(boxes, scores)):
            if float(score) < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.astype(int).tolist()
            class_id = int(class_ids[index]) if class_ids is not None else None
            label = names.get(class_id, "digit") if isinstance(names, dict) and class_id is not None else "digit"
            detections.append(
                DetectedBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    score=float(score),
                    label=str(label),
                    class_id=class_id,
                )
            )

        detections.sort(key=lambda item: item.x1)
        return detections
