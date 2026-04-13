"""OCR-only benchmark pipelines used as full-image baselines."""

from __future__ import annotations

import cv2
import numpy as np

from detection import FullFrameDetector
from recognition import EasyOCRRecognizer, TesseractRecognizer

from .base_pipeline import BasePipeline


def _smart_crop(image: np.ndarray) -> np.ndarray:
    """Find the region most likely to be a digit plate.

    Scoring: aspect_ratio × contrast_std
      → plates score high  (wide shape + high digit contrast)
      → plain walls score low (wide but uniform)
      → grass/dark backgrounds are rejected by brightness filter

    Brightness filter (mean pixel value ≥ 100):
      → keeps white plates, yellow plates, light backgrounds
      → rejects dark grass, dark road, dark sky
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h_img, w_img = image.shape[:2]

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, thresh   = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    best_score = 0.0
    best_box   = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area   = cw * ch
        aspect = cw / max(ch, 1)

        # Must cover at least 5 % of image and be landscape-oriented
        if area   < h_img * w_img * 0.05:
            continue
        if aspect < 2.0:
            continue

        # ── Brightness gate: reject dark regions (grass, road, sky) ──────────
        # Use the ORIGINAL image (not CLAHE-enhanced) for an honest brightness
        # reading.  Licence plates are white or yellow → mean ≥ 100.
        # Grass / dark backgrounds typically have mean < 80.
        region_orig = image[y:y + ch, x:x + cw]
        mean_bright = float(region_orig.mean())
        if mean_bright < 100:
            continue

        # ── Contrast score: high std + wide aspect = digit plate ─────────────
        region_gray = gray[y:y + ch, x:x + cw]
        std   = float(region_gray.std())
        score = aspect * std

        if score > best_score:
            best_score = score
            best_box   = (x, y, cw, ch)

    if best_box is None:
        # Fallback: nothing passed brightness gate → use full frame
        return image

    x, y, cw, ch = best_box
    pad = 10
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(w_img, x + cw + pad)
    y2  = min(h_img, y + ch + pad)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else image


class SmartOCROnlyPipeline(BasePipeline):
    """Baseline that applies smart pre-cropping before full-image OCR."""

    def __init__(self, pipeline_id: str, name: str, recognizer, description: str):
        super().__init__(
            pipeline_id=pipeline_id,
            name=name,
            detector=FullFrameDetector(),
            recognizer=recognizer,
            category="OCR-Only Baseline",
            description=description,
            crop_padding=0,
        )

    def run(self, frame: np.ndarray):
        import time
        from .base_pipeline import DetectionResult, PipelineResult

        if not self.is_loaded:
            raise RuntimeError(self.last_error or f"Pipeline {self.name} is not loaded")

        started_at = time.time()
        cropped    = _smart_crop(frame)
        h, w       = cropped.shape[:2]

        detection = DetectionResult(
            boxes=[(0, 0, w, h)],
            scores=[1.0],
            image=cropped,
            labels=["full_frame"],
        )
        recognitions    = self.recognize([cropped])
        annotated       = self._annotate(cropped, detection, recognitions)
        full_text       = self._combine_recognitions(recognitions)
        processing_time = time.time() - started_at

        return PipelineResult(
            detection=detection,
            recognitions=recognitions,
            full_text=full_text,
            processing_time=processing_time,
            annotated_image=annotated,
            metadata={
                "pipeline_id":        self.pipeline_id,
                "detector":           self.detector.name,
                "recognizer":         self.recognizer.name,
                "recognizer_backend": self.recognizer.active_backend_name,
                "category":           self.category,
            },
        )


class OCROnlyPipeline(BasePipeline):
    """Benchmark baseline that applies OCR to the whole image (no pre-crop)."""

    def __init__(self, pipeline_id: str, name: str, recognizer, description: str):
        super().__init__(
            pipeline_id=pipeline_id,
            name=name,
            detector=FullFrameDetector(),
            recognizer=recognizer,
            category="OCR-Only Baseline",
            description=description,
            crop_padding=0,
        )


class EasyOCRFullPipeline(SmartOCROnlyPipeline):
    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_easyocr",
            name="EasyOCR (End-to-End)",
            recognizer=EasyOCRRecognizer(),
            description="OCR-only baseline with smart plate-aware pre-cropping and EasyOCR.",
        )


class TesseractFullPipeline(SmartOCROnlyPipeline):
    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_tesseract",
            name="Tesseract (End-to-End)",
            recognizer=TesseractRecognizer(),
            description="OCR-only baseline with smart plate-aware pre-cropping and Tesseract.",
        )