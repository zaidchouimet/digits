"""YOLO-based digit detection backend."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseDetector, DetectedBox


class YOLODetector(BaseDetector):
    """Digit detector powered by Ultralytics YOLO models."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.15,
        is_plate_model: bool = False,
        inference_imgsz: int | None = None,
    ):
        super().__init__(f"yolo:{model_path}")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.is_plate_model = is_plate_model  # skip digit geometry filter for plate detectors
        self.inference_imgsz = inference_imgsz

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
            # Apply threshold at inference time as well; otherwise Ultralytics
            # default conf may suppress boxes before post-processing sees them.
            inference_kwargs = {
                "conf": float(self.confidence_threshold),
                "verbose": False,
            }
            if self.inference_imgsz:
                inference_kwargs["imgsz"] = int(self.inference_imgsz)
            results = self.model(frame, **inference_kwargs)
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

        # Skip geometry filter for plate detectors (wide aspect ratio would be rejected)
        if not self.is_plate_model:
            detections = self._filter_geometry(detections, frame.shape[:2])
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        detections.sort(key=lambda item: item.x1)
        return detections

    def _filter_geometry(self, detections: List[DetectedBox], img_shape) -> List[DetectedBox]:
        """Remove detections that don't match digit geometry.
        
        Filters out:
        - Screw holes (too small: < 0.1% of image area)
        - Text lines (extreme aspect ratios)
        - Full-plate detections (too large: > 60% of image)
        - Sticker/noise digits much shorter than the main plate digits
        """
        if not detections:
            return detections
            
        h, w = img_shape
        img_area = h * w
        filtered: List[DetectedBox] = []
        
        for det in detections:
            box_w = det.x2 - det.x1
            box_h = det.y2 - det.y1
            area = box_w * box_h
            aspect = box_w / max(box_h, 1e-6)
            
            # Filter 1: Minimum size (remove screw holes, specks)
            min_area = max(img_area * 0.001, 150)
            if area < min_area:
                continue
                
            # Filter 2: Maximum size (remove full-frame detections)
            if area > img_area * 0.6:
                continue
                
            # Filter 3: Aspect ratio for digits
            if not (0.25 <= aspect <= 2.5):
                continue
                
            filtered.append(det)
        
        if not filtered:
            return filtered

        # Filter 4: Remove boxes much shorter than the tallest detection.
        # Date stickers and small labels have noticeably shorter bounding boxes
        # than the main plate digits — keep only boxes >= 45% of the max height.
        max_h = max(d.y2 - d.y1 for d in filtered)
        filtered = [d for d in filtered if (d.y2 - d.y1) >= max_h * 0.45]

        return filtered


    def _apply_nms(self, detections: List[DetectedBox], iou_threshold: float = 0.5) -> List[DetectedBox]:
        """Non-Maximum Suppression to handle overlapping boxes.
        
        Keeps the highest confidence box when detections overlap significantly.
        This handles cases where '26' is detected both as one box and as two separate boxes.
        """
        if not detections:
            return detections
            
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x.score, reverse=True)
        keep: List[DetectedBox] = []
        
        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU overlap with current
            remaining = []
            for det in sorted_dets:
                if self._iou(current, det) < iou_threshold:
                    remaining.append(det)
            sorted_dets = remaining
        
        return keep

    def _iou(self, a: DetectedBox, b: DetectedBox) -> float:
        """Calculate Intersection over Union."""
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter_area
        
        return inter_area / union if union > 0 else 0