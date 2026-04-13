"""Pipeline registry for the digit benchmark system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from detection import YOLODetector
from recognition import EasyOCRRecognizer, TesseractRecognizer

from .base_pipeline import BasePipeline, DetectionResult, PipelineResult, RecognitionResult
from .ocr_only import EasyOCRFullPipeline, TesseractFullPipeline
from .robust_pipeline import RobustDigitPipeline, RobustOCRPipeline
from .hybrid_pipeline import HybridDigitPipeline


@dataclass(frozen=True)
class PipelineSpec:
    """Registry entry for a benchmarkable pipeline."""

    pipeline_id: str
    display_name: str
    category: str
    description: str
    builder: Callable[[], BasePipeline]
    aliases: tuple = ()


def _create_robust_yolo_pipeline(yolo_model: str) -> Callable[[], BasePipeline]:
    def pipeline_factory():
        return RobustDigitPipeline(yolo_model)
    return pipeline_factory


def _create_yolo_pipeline(yolo_model: str, recognizer_class, recognizer_name: str) -> Callable[[], BasePipeline]:
    """Factory function to create YOLO pipeline classes."""
    
    def pipeline_factory():
        display_name = f"YOLO + {recognizer_name} ({yolo_model})"
        pipeline_id = f"{yolo_model.replace('.pt', '').lower()}_{recognizer_name.lower()}"
        
        # Use higher confidence thresholds for problematic models
        if "best" in yolo_model:
            confidence_threshold = 0.5  # Higher threshold for best.pt
        elif "yolov8s" in yolo_model:
            confidence_threshold = 0.7  # Much higher threshold for yolov8s.pt
        else:
            confidence_threshold = 0.3  # Default for COCO models
            
        return BasePipeline(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(model_path=yolo_model, confidence_threshold=confidence_threshold),
            recognizer=recognizer_class(),
            category="Detection + OCR",
            description=f"YOLO digit detection followed by {recognizer_name} recognition.",
        )
    
    return pipeline_factory


PIPELINE_SPECS: List[PipelineSpec] = [
    # ── Standard YOLO pipelines ──────────────────────────────────────────────
    PipelineSpec(
        pipeline_id="yolov8n_tesseract",
        display_name="YOLOv8n + Tesseract",
        category="Detection + OCR",
        description="Official YOLOv8n detector with Tesseract OCR.",
        builder=_create_yolo_pipeline("yolov8n.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="yolov8n_easyocr",
        display_name="YOLOv8n + EasyOCR",
        category="Detection + OCR",
        description="Official YOLOv8n detector with EasyOCR.",
        builder=_create_yolo_pipeline("yolov8n.pt", EasyOCRRecognizer, "EasyOCR"),
    ),

    # ── OCR-only baselines ───────────────────────────────────────────────────
    PipelineSpec(
        pipeline_id="ocr_only_easyocr",
        display_name="EasyOCR (End-to-End)",
        category="OCR-Only Baseline",
        description="Full-image OCR baseline using EasyOCR with smart pre-cropping.",
        builder=EasyOCRFullPipeline,
    ),
    PipelineSpec(
        pipeline_id="ocr_only_tesseract",
        display_name="Tesseract (End-to-End)",
        category="OCR-Only Baseline",
        description="Full-image OCR baseline using Tesseract with smart pre-cropping.",
        builder=TesseractFullPipeline,
    ),

    # ── Robust pipelines ─────────────────────────────────────────────────────
    PipelineSpec(
        pipeline_id="robust_yolov8n_tesseract",
        display_name="Robust YOLOv8n + Tesseract",
        category="Detection + OCR",
        description="Enhanced YOLOv8n with full-frame fallback when detection is empty.",
        builder=_create_robust_yolo_pipeline("yolov8n.pt"),
    ),
    PipelineSpec(
        pipeline_id="robust_ocr_only",
        display_name="Robust OCR (End-to-End)",
        category="OCR-Only Baseline",
        description="Enhanced OCR pipeline with smart pre-cropping and EasyOCR.",
        builder=RobustOCRPipeline,
        aliases=("robust_ocr",),
    ),

    # ── Custom models ────────────────────────────────────────────────────────
    PipelineSpec(
        pipeline_id="best_tesseract",
        display_name="Best + Tesseract",
        category="Detection + OCR",
        description="Custom best.pt model with Tesseract OCR.",
        builder=_create_yolo_pipeline("best.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="best_easyocr",
        display_name="Best + EasyOCR",
        category="Detection + OCR",
        description="Custom best.pt model with EasyOCR OCR.",
        builder=_create_yolo_pipeline("best.pt", EasyOCRRecognizer, "EasyOCR"),
    ),
    PipelineSpec(
        pipeline_id="yolov8s_tesseract",
        display_name="YOLOv8s + Tesseract",
        category="Detection + OCR",
        description="YOLOv8s model with Tesseract OCR.",
        builder=_create_yolo_pipeline("yolov8s.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="yolov8s_easyocr",
        display_name="YOLOv8s + EasyOCR",
        category="Detection + OCR",
        description="YOLOv8s model with EasyOCR OCR.",
        builder=_create_yolo_pipeline("yolov8s.pt", EasyOCRRecognizer, "EasyOCR"),
    ),
    PipelineSpec(
        pipeline_id="best1_tesseract",
        display_name="Best1 + Tesseract",
        category="Detection + OCR",
        description="Custom newly synthetic data trained best1.pt model with Tesseract OCR.",
        builder=_create_yolo_pipeline("best1.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="best1_easyocr",
        display_name="Best1 + EasyOCR",
        category="Detection + OCR",
        description="Custom newly synthetic data trained best1.pt model with EasyOCR OCR.",
        builder=_create_yolo_pipeline("best1.pt", EasyOCRRecognizer, "EasyOCR"),
    ),

    # ── Hybrid pipelines (YOLO + OCR fallback) ─────────────────────────────────
    PipelineSpec(
        pipeline_id="hybrid_best_easyocr",
        display_name="Hybrid Best + EasyOCR",
        category="Detection + OCR",
        description="Best.pt YOLO with OCR fallback for challenging images.",
        builder=lambda: HybridDigitPipeline("best.pt", "easyocr"),
    ),
    PipelineSpec(
        pipeline_id="hybrid_best_tesseract",
        display_name="Hybrid Best + Tesseract",
        category="Detection + OCR",
        description="Best.pt YOLO with Tesseract fallback for challenging images.",
        builder=lambda: HybridDigitPipeline("best.pt", "tesseract"),
    ),
    ]


def _build_lookup() -> Dict[str, PipelineSpec]:
    lookup: Dict[str, PipelineSpec] = {}
    for spec in PIPELINE_SPECS:
        lookup[spec.pipeline_id]   = spec
        lookup[spec.display_name]  = spec
        for alias in spec.aliases:
            lookup[alias] = spec
    return lookup


PIPELINE_LOOKUP = _build_lookup()


def get_pipeline(pipeline_name: str) -> BasePipeline:
    if pipeline_name not in PIPELINE_LOOKUP:
        raise ValueError(
            f"Unknown pipeline: {pipeline_name}. "
            f"Available: {[s.pipeline_id for s in PIPELINE_SPECS]}"
        )
    return PIPELINE_LOOKUP[pipeline_name].builder()


def get_pipeline_spec(pipeline_name: str) -> PipelineSpec:
    if pipeline_name not in PIPELINE_LOOKUP:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINE_LOOKUP[pipeline_name]


def get_available_pipelines(use_display_names: bool = True) -> List[str]:
    if use_display_names:
        return [spec.display_name for spec in PIPELINE_SPECS]
    return [spec.pipeline_id for spec in PIPELINE_SPECS]


def get_pipeline_categories() -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    for spec in PIPELINE_SPECS:
        categories.setdefault(spec.category, []).append(spec.display_name)
    return categories


def list_pipeline_specs() -> List[PipelineSpec]:
    return list(PIPELINE_SPECS)


__all__ = [
    "BasePipeline",
    "DetectionResult",
    "RecognitionResult",
    "PipelineResult",
    "EasyOCRFullPipeline",
    "TesseractFullPipeline",
    "RobustDigitPipeline",
    "RobustOCRPipeline",
    "PipelineSpec",
    "get_pipeline",
    "get_pipeline_spec",
    "get_available_pipelines",
    "get_pipeline_categories",
    "list_pipeline_specs",
]