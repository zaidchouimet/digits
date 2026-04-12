"""
Enhanced YOLO pipeline that works with existing models
Uses class 'clock' (74) as proxy for digit regions
"""

import cv2
import numpy as np
from typing import List
from detection import YOLODetector, FullFrameDetector
from recognition import TesseractRecognizer
from pipelines.base_pipeline import BasePipeline

class EnhancedYOLOPipeline(BasePipeline):
    """Enhanced YOLO pipeline that works with existing COCO-trained models."""
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        # Create YOLO detector with very low confidence threshold
        self.yolo_detector = YOLODetector(model_path=yolo_model_path, confidence_threshold=0.01)
        self.full_frame_detector = FullFrameDetector()
        
        super().__init__(
            pipeline_id=f"enhanced_{yolo_model_path.replace('.pt', '').lower()}_tesseract",
            name=f"Enhanced YOLO + Tesseract ({yolo_model_path})",
            detector=self.yolo_detector,
            recognizer=TesseractRecognizer(),
            category="Detection + OCR",
            description="Enhanced YOLO detection with digit-specific logic.",
        )
    
    def detect(self, frame: np.ndarray):
        """Enhanced detection with digit-specific logic."""
        
        # Try YOLO detection first
        try:
            yolo_detections = self.yolo_detector.detect(frame)
            
            # Filter for classes that might contain digits (clock class = 74)
            digit_detections = []
            for detection in yolo_detections:
                if detection.cls == 74:  # 'clock' class
                    # Expand the bounding box to capture more content
                    padding = 20
                    x1 = max(0, detection.x1 - padding)
                    y1 = max(0, detection.y1 - padding)
                    x2 = min(frame.shape[1], detection.x2 + padding)
                    y2 = min(frame.shape[0], detection.y2 + padding)
                    
                    # Create expanded detection
                    from detection import DetectedBox
                    expanded_detection = DetectedBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        score=detection.score,
                        label=detection.label
                    )
                    digit_detections.append(expanded_detection)
            
            if digit_detections:
                print(f"🎯 Found {len(digit_detections)} digit-like regions")
                return digit_detections
                
        except Exception as e:
            print(f"YOLO detection failed: {e}")
        
        # Fallback to full frame
        print("🔄 Falling back to full-frame detection")
        return self.full_frame_detector.detect(frame)

# Factory function
def create_enhanced_pipeline(model_path: str = "yolov8n.pt"):
    """Create enhanced YOLO pipeline."""
    return EnhancedYOLOPipeline(model_path)
