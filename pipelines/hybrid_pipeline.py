"""
Hybrid pipeline that combines YOLO detection with OCR fallback.
Works well for both license plates and SVHN images.
"""

import cv2
import numpy as np
from typing import List, Tuple
from detection import YOLODetector
from recognition import EasyOCRRecognizer, TesseractRecognizer
from pipelines.base_pipeline import BasePipeline, DetectionResult, RecognitionResult


class HybridDigitPipeline(BasePipeline):
    """Hybrid pipeline that tries YOLO first, falls back to OCR-only if no detections."""
    
    def __init__(self, yolo_model: str = "best.pt", ocr_backend: str = "easyocr"):
        super().__init__(
            pipeline_id=f"hybrid_{yolo_model.replace('.pt', '')}_{ocr_backend}",
            name=f"Hybrid {yolo_model} + {ocr_backend.title()}",
            detector=YOLODetector(model_path=yolo_model, confidence_threshold=0.1),
            recognizer=EasyOCRRecognizer() if ocr_backend == "easyocr" else TesseractRecognizer(),
            category="Detection + OCR",
            description=f"YOLO detection with OCR fallback for challenging images.",
        )
        self.ocr_backend = ocr_backend
        self.ocr_only_fallback = EasyOCRRecognizer() if ocr_backend == "easyocr" else TesseractRecognizer()
        
    def load_models(self) -> bool:
        """Load both YOLO and OCR models."""
        try:
            # Load YOLO detector
            if not self.detector.load():
                return False
                
            # Load OCR recognizer
            if not self.recognizer.load():
                return False
                
            # Load fallback OCR
            if not self.ocr_only_fallback.load():
                return False
                
            self.is_loaded = True
            return True
        except Exception:
            self.is_loaded = False
            return False
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect digits using YOLO, with full-frame fallback."""
        try:
            # Try YOLO detection first
            detections = self.detector.detect(image)
            
            if detections:
                # Convert to DetectionResult format
                boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
                scores = [d.score for d in detections]
                labels = [d.label for d in detections]
                
                result = DetectionResult(
                    boxes=boxes,
                    scores=scores,
                    image=image,
                    labels=labels
                )
                self._last_detection_labels = labels
                return result
            else:
                # No YOLO detections - use full frame as fallback
                height, width = image.shape[:2]
                result = DetectionResult(
                    boxes=[(0, 0, width, height)],
                    scores=[1.0],
                    image=image,
                    labels=["full_frame"]
                )
                self._last_detection_labels = ["full_frame"]
                return result
                
        except Exception as e:
            # Error in YOLO detection - fall back to full frame
            height, width = image.shape[:2]
            result = DetectionResult(
                boxes=[(0, 0, width, height)],
                scores=[1.0],
                image=image,
                labels=["fallback"]
            )
            self._last_detection_labels = ["fallback"]
            return result
    
    def recognize(self, crops: List[np.ndarray]) -> List[RecognitionResult]:
        """Recognize digits from detected regions with hybrid approach."""
        recognitions = []
        
        try:
            # For full frame fallback, use OCR-only approach directly
            if hasattr(self, '_last_detection_labels') and 'full_frame' in self._last_detection_labels:
                results = self.ocr_only_fallback.recognize_batch(crops)
            else:
                # Try standard recognition first
                results = self.recognizer.recognize_batch(crops)
                
                # If standard recognition fails or gives empty results, try OCR-only approach
                if not results or all(not r.text.strip() for r in results):
                    results = self.ocr_only_fallback.recognize_batch(crops)
            
            # Convert RecognitionOutput to RecognitionResult
            for result in results:
                if result and result.text.strip():
                    recognitions.append(RecognitionResult(
                        text=result.text,
                        confidence=result.confidence,
                        crop=result.crop
                    ))
                        
        except Exception:
            # Continue with empty results if recognition fails
            pass
                
        return recognitions


def create_hybrid_pipeline(yolo_model: str = "best.pt", ocr_backend: str = "easyocr"):
    """Factory function to create hybrid pipeline."""
    return HybridDigitPipeline(yolo_model, ocr_backend)
