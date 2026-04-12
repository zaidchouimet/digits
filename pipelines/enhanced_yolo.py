"""
Enhanced YOLO pipeline that works with existing models
Uses class 'clock' (74) as proxy for digit regions
"""

import cv2
import numpy as np
from typing import List, Tuple
from detection import YOLODetector, FullFrameDetector
from recognition import TesseractRecognizer
from pipelines.base_pipeline import BasePipeline, DetectionResult
    """Enhanced YOLO pipeline that works with existing COCO-trained models."""
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        super().__init__(f"Enhanced YOLO + Tesseract ({yolo_model_path})")
        self.yolo_model_path = yolo_model_path
        self.profiler = ModelProfiler()
        
    def load_models(self) -> bool:
        """Load YOLO and Tesseract models."""
        try:
            # Load YOLO model
            from ultralytics import YOLO
            self.detection_model = YOLO(self.yolo_model_path)
            
            # Load Tesseract
            import pytesseract
            self.recognition_model = pytesseract
            
            self.is_loaded = True
            print(f"✅ Loaded {self.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {self.name}: {e}")
            return False
    
    def detect_digits(self, image: np.ndarray) -> DetectionResult:
        """Detect digits using YOLO with enhanced logic."""
        try:
            # Run YOLO detection with very low threshold
            results, _ = self.profiler.profile_inference(
                self.detection_model, image, conf=0.01, verbose=False
            )
            
            boxes = []
            scores = []
            
            if results and len(results[0].boxes) > 0:
                det_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                det_scores = results[0].boxes.conf.cpu().numpy()
                det_classes = results[0].boxes.cls.cpu().numpy()
                
                # Filter for classes that might contain digits
                digit_related_classes = [74]  # 'clock' class
                min_confidence = 0.02  # 2% threshold
                
                for box, score, cls in zip(det_boxes, det_scores, det_classes):
                    if score >= min_confidence and int(cls) in digit_related_classes:
                        x1, y1, x2, y2 = box
                        # Expand box to capture more content
                        padding = 20
                        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
                        
                        boxes.append((x1, y1, x2, y2))
                        scores.append(float(score))
            
            # If no detections, fall back to full frame
            if not boxes:
                height, width = image.shape[:2]
                boxes.append((0, 0, width, height))
                scores.append(1.0)
                print("🔄 No digit regions found, using full frame")
            
            return DetectionResult(boxes=boxes, scores=scores, image=image)
            
        except Exception as e:
            print(f"Detection error: {e}")
            # Fallback to full frame
            height, width = image.shape[:2]
            return DetectionResult([(0, 0, width, height)], [1.0], image)
    
    def recognize_digits(self, crops: List[np.ndarray]) -> List[RecognitionResult]:
        """Recognize digits using Tesseract with digit-only configuration."""
        recognitions = []
        
        for i, crop in enumerate(crops):
            try:
                # Configure Tesseract for digits only
                config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                text = self.recognition_model.image_to_string(crop, config=config).strip()
                
                # Filter to keep only digits
                clean_text = ''.join([c for c in text if c.isdigit()])
                
                if clean_text:
                    recognitions.append(RecognitionResult(
                        text=clean_text,
                        confidence=0.8,  # Default confidence
                        crop=crop
                    ))
                    
            except Exception as e:
                print(f"Recognition error for crop {i}: {e}")
                continue
        
        return recognitions

# Factory function
def create_enhanced_pipeline(model_path: str = "yolov8n.pt"):
    """Create enhanced YOLO pipeline."""
    return EnhancedYOLOPipeline(model_path)
