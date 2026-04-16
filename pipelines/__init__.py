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
from .yolo_only import YOLOOnlyPipeline
from .cascade_pipeline import CascadePlatePipeline


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
        pipeline_id  = f"{yolo_model.replace('.pt', '').lower()}_{recognizer_name.lower()}"

        # Confidence thresholds and detector flags tuned per model
        is_plate_model = False
        inference_imgsz = None
        if "best2" in yolo_model:
            # SVHN fine-tuned, only 5 epochs → lower threshold to catch more detections
            confidence_threshold = 0.20
        elif "best3" in yolo_model:
            # best3 behaves like plate-level detector; skip digit geometry filtering
            confidence_threshold = 0.05
            is_plate_model = True
            # Use larger inference size for small/distant plate recall.
            inference_imgsz = 1280
        elif "best1" in yolo_model:
            # Synthetic-trained, overfitted → higher threshold reduces false positives
            confidence_threshold = 0.50
        elif "best" in yolo_model:
            # Plate detector — wide boxes, skip digit geometry filter
            confidence_threshold = 0.30
            is_plate_model = True
        elif "yolov8s" in yolo_model:
            confidence_threshold = 0.70
        else:
            confidence_threshold = 0.30   # default for COCO pretrained

        return BasePipeline(
            pipeline_id=pipeline_id,
            name=display_name,
            detector=YOLODetector(
                model_path=yolo_model,
                confidence_threshold=confidence_threshold,
                is_plate_model=is_plate_model,
                inference_imgsz=inference_imgsz,
            ),
            recognizer=recognizer_class(),
            category="Detection + OCR",
            description=f"YOLO digit detection followed by {recognizer_name} recognition.",
        )

    return pipeline_factory


PIPELINE_SPECS: List[PipelineSpec] = [

    # ── Standard YOLO (COCO pretrained) ──────────────────────────────────────
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

    # ── Custom models ─────────────────────────────────────────────────────────
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
        description="Custom best.pt model with EasyOCR.",
        builder=_create_yolo_pipeline("best.pt", EasyOCRRecognizer, "EasyOCR"),
    ),
    PipelineSpec(
        pipeline_id="best_only",
        display_name="Best Only",
        category="Detection Only",
        description="Custom best.pt model without OCR.",
        builder=lambda: YOLOOnlyPipeline("best.pt"),
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
        description="YOLOv8s model with EasyOCR.",
        builder=_create_yolo_pipeline("yolov8s.pt", EasyOCRRecognizer, "EasyOCR"),
    ),

    # ── Synthetic-trained model (best1) ───────────────────────────────────────
    PipelineSpec(
        pipeline_id="best1_tesseract",
        display_name="Best1 + Tesseract",
        category="Detection + OCR",
        description="Synthetic-data YOLOv8n (best1.pt) with Tesseract OCR.",
        builder=_create_yolo_pipeline("best1.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="best1_easyocr",
        display_name="Best1 + EasyOCR",
        category="Detection + OCR",
        description="Synthetic-data YOLOv8n (best1.pt) with EasyOCR.",
        builder=_create_yolo_pipeline("best1.pt", EasyOCRRecognizer, "EasyOCR"),
    ),

    # ── SVHN fine-tuned model (best2) — NEW ───────────────────────────────────
    PipelineSpec(
        pipeline_id="best2_tesseract",
        display_name="Best2 + Tesseract (SVHN 5ep)",
        category="Detection + OCR",
        description="YOLOv8n fine-tuned on SVHN for 5 epochs (best2.pt) with Tesseract OCR.",
        builder=_create_yolo_pipeline("best2.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="best2_easyocr",
        display_name="Best2 + EasyOCR (SVHN 5ep)",
        category="Detection + OCR",
        description="YOLOv8n fine-tuned on SVHN for 5 epochs (best2.pt) with EasyOCR.",
        builder=_create_yolo_pipeline("best2.pt", EasyOCRRecognizer, "EasyOCR"),
    ),
    PipelineSpec(
        pipeline_id="best2_only",
        display_name="Best2 Only (SVHN 5ep)",
        category="Detection Only",
        description="YOLOv8n fine-tuned on SVHN for 5 epochs (best2.pt) without OCR.",
        builder=lambda: YOLOOnlyPipeline("best2.pt"),
    ),
    PipelineSpec(
        pipeline_id="best3_tesseract",
        display_name="Best3 + Tesseract",
        category="Detection + OCR",
        description="Custom YOLOv8n (best3.pt) with Tesseract OCR.",
        builder=_create_yolo_pipeline("best3.pt", TesseractRecognizer, "Tesseract"),
    ),
    PipelineSpec(
        pipeline_id="best3_easyocr",
        display_name="Best3 + EasyOCR",
        category="Detection + OCR",
        description="Custom YOLOv8n (best3.pt) with EasyOCR.",
        builder=_create_yolo_pipeline("best3.pt", EasyOCRRecognizer, "EasyOCR"),
    ),
    PipelineSpec(
        pipeline_id="best3_only",
        display_name="Best3 Only",
        category="Detection Only",
        description="Custom YOLOv8n (best3.pt) without OCR.",
        builder=lambda: YOLOOnlyPipeline("best3.pt"),
    ),
    PipelineSpec(
        pipeline_id="cascade_best3_best2",
        display_name="Cascade: Best3 Plate + Best2 Digits",
        category="Detection + OCR",
        description="Two-stage: best3 detects plates, best2 reads digits class-by-class.",
        builder=lambda: CascadePlatePipeline("best2.pt", digit_confidence_threshold=0.20),
    ),
    PipelineSpec(
        pipeline_id="cascade_best3_best1",
        display_name="Cascade: Best3 Plate + Best1 Digits",
        category="Detection + OCR",
        description="Two-stage: best3 detects plates, best1 reads digits class-by-class.",
        builder=lambda: CascadePlatePipeline("best1.pt", digit_confidence_threshold=0.50),
    ),

    # ── Hybrid pipelines ──────────────────────────────────────────────────────
    PipelineSpec(
        pipeline_id="hybrid_best_easyocr",
        display_name="Hybrid Best + EasyOCR",
        category="Detection + OCR",
        description="best.pt YOLO with OCR fallback for challenging images.",
        builder=lambda: HybridDigitPipeline("best.pt", "easyocr"),
    ),
    PipelineSpec(
        pipeline_id="hybrid_best_tesseract",
        display_name="Hybrid Best + Tesseract",
        category="Detection + OCR",
        description="best.pt YOLO with Tesseract fallback for challenging images.",
        builder=lambda: HybridDigitPipeline("best.pt", "tesseract"),
    ),
]


def _build_lookup() -> Dict[str, PipelineSpec]:
    lookup: Dict[str, PipelineSpec] = {}
    for spec in PIPELINE_SPECS:
        lookup[spec.pipeline_id]  = spec
        lookup[spec.display_name] = spec
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
    "YOLOOnlyPipeline",
    "CascadePlatePipeline",
    "PipelineSpec",
    "get_pipeline",
    "get_pipeline_spec",
    "get_available_pipelines",
    "get_pipeline_categories",
    "list_pipeline_specs",
]