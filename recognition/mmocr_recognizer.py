"""MMOCR backend for digit/plate recognition."""

from typing import List
import numpy as np
from .base import BaseRecognizer, RecognitionOutput, extract_digit_text

class MMOCRRecognizer(BaseRecognizer):
    """OCR using OpenMMLab's MMOCR."""
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None):
        super().__init__('mmocr')
        # Use a pre-trained text recognition model
        self.config_path = config_path or 'configs/textrecog/sar/sar_resnet31_parallel-decoder_5e_st_mj.py'
        self.checkpoint_path = checkpoint_path
        
    def load(self) -> bool:
        try:
            from mmocr.apis import init_model
            import mmcv
            
            self.model = init_model(self.config_path, self.checkpoint_path, device='cuda:0')
            self.is_loaded = True
            self.active_backend_name = 'mmocr'
            return True
        except Exception as e:
            self.last_error = f"Failed to load MMOCR: {e}"
            return False
            
    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        from mmocr.apis import inference_recognizer
        
        results = []
        for crop in crops:
            if crop.size == 0:
                results.append(RecognitionOutput('', 0.0, crop, self.active_backend_name))
                continue
                
            # MMOCR expects RGB
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            elif crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            result = inference_recognizer(self.model, crop)
            text = result['text']
            score = result.get('score', 0.0)
            
            # Extract digits only
            digits = extract_digit_text(text)
            results.append(RecognitionOutput(digits, score, crop, self.active_backend_name))
            
        return results