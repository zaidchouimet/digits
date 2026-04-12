"""EfficientDet-based digit detection backend."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from .base import BaseDetector, DetectedBox


class EfficientDetDetector(BaseDetector):
    """Experimental EfficientDet detector retained for backward compatibility."""

    def __init__(self, model_name: str = "tf_efficientdet_d0", confidence_threshold: float = 0.3):
        super().__init__(f"efficientdet:{model_name}")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

    def load(self) -> bool:
        try:
            from effdet import create_model_from_config, get_efficientdet_config

            config = get_efficientdet_config(self.model_name)
            self.model = create_model_from_config(config, bench_task="predict", pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.is_loaded = True
            self.last_error = None
            return True
        except ImportError as error:
            self.is_loaded = False
            self.last_error = (
                "EfficientDet is unavailable because 'effdet' or one of its dependencies "
                f"could not be imported: {error}"
            )
            return False
        except Exception as error:
            self.is_loaded = False
            self.last_error = f"Failed to load EfficientDet detector '{self.model_name}': {error}"
            return False

    def detect(self, frame: np.ndarray) -> List[DetectedBox]:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Detector {self.name} is not loaded")

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = ((image_tensor - mean) / std).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            results = self.model(image_tensor)

        detections: List[DetectedBox] = []
        if results is None or len(results) == 0:
            return detections

        for detection in results[0].detach().cpu().numpy():
            score = float(detection[4])
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, detection[:4])
            detections.append(DetectedBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

        detections.sort(key=lambda item: item.x1)
        return detections
