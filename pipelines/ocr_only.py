"""OCR-only benchmark pipelines used as full-image baselines."""

from __future__ import annotations

import cv2
import numpy as np

from detection import FullFrameDetector
from recognition import EasyOCRRecognizer, TesseractRecognizer

from .base_pipeline import BasePipeline


def _hsv_plate_crop(image: np.ndarray) -> np.ndarray | None:
    """Detect yellow or white licence plate region using HSV colour filtering and edge density."""
    h_img, w_img = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow plate: hue 15–35, high saturation, medium-high value
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    # White plate: low saturation, high value
    white_mask  = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    best_score = 0.0
    best_box   = None

    def evaluate_mask(mask, color_weight):
        nonlocal best_score, best_box
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area   = cw * ch
            aspect = cw / max(ch, 1)

            # Relaxed the area constraint slightly to 2% to catch plates further back
            if area < h_img * w_img * 0.02:
                continue
            if aspect < 2.5 or aspect > 12.0:
                continue

            region = image[y:y + ch, x:x + cw]
            
            # Use Edge Density to find text instead of standard deviation
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_region, 100, 200)
            edge_density = np.sum(edges > 0) / area

            # Score: wide aspect + edge density (finds text) + area ratio + color priority
            score = aspect * edge_density * (area / (h_img * w_img)) * color_weight
            
            if score > best_score:
                best_score = score
                best_box   = (x, y, cw, ch)

    # Evaluate yellow contours with a 3x multiplier (highly likely to be a plate)
    evaluate_mask(yellow_mask, color_weight=3.0)
    # Evaluate white contours with a 1x multiplier (prone to false positives on white cars)
    evaluate_mask(white_mask, color_weight=1.0)

    if best_box is None:
        return None

    x, y, cw, ch = best_box
    pad = 8
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(w_img, x + cw + pad)
    y2  = min(h_img, y + ch + pad)
    crop = image[y1:y2, x1:x2]
    
    return crop if crop.size > 0 else None


def _contrast_plate_crop(image: np.ndarray) -> np.ndarray | None:
    """Fallback: find the brightest, most-contrast wide region.

    Used when HSV finds nothing (e.g. dark or non-standard plate colours).
    Brightness threshold is lowered to 80 so yellow plates (~98 mean) pass.
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h_img, w_img = image.shape[:2]

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh   = cv2.threshold(enhanced, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_box   = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area   = cw * ch
        aspect = cw / max(ch, 1)

        if area   < h_img * w_img * 0.05:
            continue
        if aspect < 2.0:
            continue

        region_orig = image[y:y + ch, x:x + cw]
        mean_bright = float(region_orig.mean())
        # Lowered from 100 → 80 so yellow plates (~98) are not rejected
        if mean_bright < 60:
            continue

        region_gray = gray[y:y + ch, x:x + cw]
        std   = float(region_gray.std())
        score = aspect * std
        if score > best_score:
            best_score = score
            best_box   = (x, y, cw, ch)

    if best_box is None:
        return None

    x, y, cw, ch = best_box
    pad = 10
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(w_img, x + cw + pad)
    y2  = min(h_img, y + ch + pad)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

# In pipelines/ocr_only.py, add this function after _contrast_plate_crop:

def _black_plate_crop(image: np.ndarray) -> np.ndarray | None:
    """Detect dark plates with light text (black, dark blue, dark green backgrounds).
    
    Many address plaques and modern license plates use white text on dark backgrounds.
    """
    h_img, w_img = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Look for dark backgrounds (low value) across all hues
    # Black/dark gray: low value, any saturation
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    
    # Also look for white/light text to confirm
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
    
    # Dilate the dark mask to capture the full plate region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    dark_dilated = cv2.dilate(dark_mask, kernel, iterations=2)
    
    # Find contours in the dark regions
    contours, _ = cv2.findContours(dark_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_score = 0.0
    best_box = None
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / max(ch, 1)
        
        # Plate-like aspect ratio (wider than tall)
        if area < h_img * w_img * 0.02:  # At least 2% of image
            continue
        if aspect < 2.0 or aspect > 12.0:
            continue
            
        # Check if this region contains white pixels (text)
        roi_white = white_mask[y:y+ch, x:x+cw]
        white_density = np.sum(roi_white > 0) / area
        
        # Score: high aspect ratio + presence of white text
        score = aspect * white_density * (area / (h_img * w_img))
        
        if score > best_score:
            best_score = score
            best_box = (x, y, cw, ch)
    
    if best_box is None:
        return None
        
    x, y, cw, ch = best_box
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + cw + pad)
    y2 = min(h_img, y + ch + pad)
    
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _smart_crop(image: np.ndarray) -> np.ndarray:
    """Three-stage plate finder supporting yellow, white, and black plates."""
    # Stage 1: Yellow plates (Algerian standard)
    crop = _hsv_plate_crop(image)
    if crop is not None:
        return crop

    # Stage 2: White plates  
    crop = _contrast_plate_crop(image)
    if crop is not None:
        return crop
        
    # Stage 3: Black/dark plates (NEW)
    crop = _black_plate_crop(image)
    if crop is not None:
        return crop

    # Last resort: full frame
    return image


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


class EasyOCRFullPipeline(OCROnlyPipeline):
    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_easyocr",
            name="EasyOCR (End-to-End)",
            recognizer=EasyOCRRecognizer(),
            description="OCR-only baseline using EasyOCR natively on the full image.",
        )


class TesseractFullPipeline(OCROnlyPipeline):
    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_tesseract",
            name="Tesseract (End-to-End)",
            recognizer=TesseractRecognizer(),
            description="OCR-only baseline using Tesseract natively on the full image.",
        )