"""EasyOCR backend — digits-only with combined height + confidence filtering."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .base import BaseRecognizer, RecognitionOutput, extract_digit_text, preprocess_crop_for_ocr


# ── Tunable filter constants ──────────────────────────────────────────────────
# A detected box is kept if it passes EITHER the height test OR the confidence
# test.  This handles both large-digit signs and smaller plate digits.
_MIN_HEIGHT_PCT  = 0.15   # box height >= 15% of image height  → large digit
_MIN_CONFIDENCE  = 0.60   # confidence >= 0.60                 → high-quality read
_FALLBACK_TOP_N  = 3      # if nothing passes, keep top-N by confidence


def _filter_digit_boxes(results: list, image_height: int) -> list:
    """Keep only EasyOCR boxes that are either large or high-confidence.

    Why two criteria:
      • Height ≥ 15%  → catches big digits on street/community signs where
                         Arabic text misreads land in many small noisy boxes.
      • Confidence ≥ 0.60 → catches correctly-read digits on licence plates
                             that may be smaller but are recognised cleanly.

    Fallback: if nothing survives, return top-N by confidence so we never
    return an empty result when digits are genuinely present.
    """
    if not results:
        return results

    min_h = image_height * _MIN_HEIGHT_PCT

    kept = []
    for item in results:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], item[1], float(item[2])

        # Only consider items that contain at least one digit
        if not any(c.isdigit() for c in text):
            continue

        try:
            ys    = [p[1] for p in bbox]
            box_h = max(ys) - min(ys)
        except Exception:
            box_h = min_h  # can't measure → treat as large enough

        height_ok = box_h >= min_h
        conf_ok   = conf  >= _MIN_CONFIDENCE

        if height_ok or conf_ok:
            kept.append(item)

    if kept:
        return kept

    # Fallback: nothing passed → return top-N by confidence (not the whole list)
    digit_items = [r for r in results if len(r) >= 3 and any(c.isdigit() for c in r[1])]
    digit_items.sort(key=lambda r: float(r[2]), reverse=True)
    return digit_items[:_FALLBACK_TOP_N] if digit_items else results[:_FALLBACK_TOP_N]


def _sort_left_to_right(results: list) -> list:
    """Sort EasyOCR result items by left-edge x-coordinate (reading order)."""
    def left_x(item):
        try:
            return min(p[0] for p in item[0])
        except Exception:
            return 0
    return sorted(results, key=left_x)


class EasyOCRRecognizer(BaseRecognizer):
    """Digit recognizer built on EasyOCR."""

    def __init__(self):
        super().__init__('easyocr')

    def load(self) -> bool:
        try:
            import easyocr
            self.model = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            self.is_loaded = True
            self.last_error = None
            self.active_backend_name = 'easyocr'
            return True
        except Exception as error:
            self.is_loaded = False
            self.last_error = f'Failed to load EasyOCR recognizer: {error}'
            return False

    def recognize_batch(self, crops: List[np.ndarray], single_char: bool = False) -> List[RecognitionOutput]:
        if not self.is_loaded:
            raise RuntimeError(self.last_error or 'EasyOCR recognizer is not loaded')

        recognitions: List[RecognitionOutput] = []

        for crop in crops:
            if crop.size == 0:
                recognitions.append(RecognitionOutput('', 0.0, crop, self.active_backend_name))
                continue

            prepared  = preprocess_crop_for_ocr(crop)
            img_h     = prepared.shape[0]
            text      = ''
            confidence = 0.0

            attempts = [
                dict(allowlist='0123456789', paragraph=False, detail=1,
                     min_size=5, contrast_ths=0.1, adjust_contrast=0.5),
                dict(allowlist='0123456789', paragraph=False, detail=1),
                dict(allowlist='0123456789', paragraph=True,  detail=1,
                     min_size=5, contrast_ths=0.1, adjust_contrast=0.5),
            ]

            for kwargs in attempts:
                try:
                    results = self.model.readtext(prepared, **kwargs)
                except Exception:
                    results = []

                if not results:
                    continue

                # ── Filter: keep large OR high-confidence digit boxes ─────────
                filtered = _filter_digit_boxes(results, img_h)

                # ── Sort left-to-right for correct reading order ──────────────
                ordered = _sort_left_to_right(filtered)

                candidate = extract_digit_text(
                    ''.join(item[1] for item in ordered if len(item) >= 2)
                )
                if candidate:
                    text       = candidate
                    confidence = float(np.mean(
                        [item[2] for item in ordered if len(item) >= 3]
                    ))
                    break

            recognitions.append(RecognitionOutput(text, confidence, crop, self.active_backend_name))

        return recognitions