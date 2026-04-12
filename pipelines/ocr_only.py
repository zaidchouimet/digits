"""OCR-only benchmark pipelines used as full-image baselines."""

from __future__ import annotations

from detection import FullFrameDetector
from recognition import EasyOCRRecognizer, PaddleOCRRecognizer, TesseractRecognizer

from .base_pipeline import BasePipeline


class OCROnlyPipeline(BasePipeline):
    """Benchmark baseline that applies OCR to the whole image."""

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


class PaddleOCRFullPipeline(OCROnlyPipeline):
    """PaddleOCR OCR-only baseline with automatic EasyOCR fallback."""

    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_paddleocr",
            name="PaddleOCR (End-to-End)",
            recognizer=PaddleOCRRecognizer(),
            description=(
                "OCR-only baseline on the full image. Uses PaddleOCR when available "
                "and falls back to EasyOCR automatically."
            ),
        )


class EasyOCRFullPipeline(OCROnlyPipeline):
    """EasyOCR full-image baseline."""

    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_easyocr",
            name="EasyOCR (End-to-End)",
            recognizer=EasyOCRRecognizer(),
            description="OCR-only baseline that applies EasyOCR to the full image.",
        )


class TesseractFullPipeline(OCROnlyPipeline):
    """Tesseract full-image baseline."""

    def __init__(self):
        super().__init__(
            pipeline_id="ocr_only_tesseract",
            name="Tesseract (End-to-End)",
            recognizer=TesseractRecognizer(),
            description="OCR-only baseline that applies Tesseract to the full image.",
        )
