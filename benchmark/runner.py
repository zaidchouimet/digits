"""Benchmark runner for academically valid digit recognition evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

from pipelines.base_pipeline import PipelineResult
from utils import (
    BenchmarkTracker,
    apply_image_conditions,
    calculate_cer,
    calculate_digit_accuracy,
    calculate_metrics_batch,
    calculate_sequence_accuracy,
    calculate_wer,
)


@dataclass
class BenchmarkSample:
    """A single labeled sample used during a benchmark run."""

    image: np.ndarray
    ground_truth: str
    sample_id: str
    source: str


@dataclass
class SamplePrediction:
    """Stored prediction details for one benchmark sample."""

    sample_id: str
    source: str
    ground_truth: str
    prediction: str
    processing_time: float
    cer: float
    wer: float
    digit_accuracy: float
    sequence_accuracy: float
    result: PipelineResult


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark statistics and per-sample results."""

    pipeline_id: str
    pipeline_name: str
    detector_name: str
    recognizer_name: str
    recognizer_backend: str
    data_source: str
    condition: str
    num_samples: int
    evaluated_samples: int
    avg_fps: float
    avg_latency_ms: float
    avg_cpu: float
    avg_memory: float
    peak_memory: float
    energy_kwh: float
    cer: float
    wer: float
    digit_accuracy: float
    sequence_accuracy: float
    created_at: str
    sample_predictions: List[SamplePrediction] = field(default_factory=list)

    def to_row(self) -> dict:
        """Flatten the benchmark summary into a CSV-friendly dict."""
        return {
            "Timestamp": self.created_at,
            "Pipeline ID": self.pipeline_id,
            "Pipeline": self.pipeline_name,
            "Detector": self.detector_name,
            "Recognizer": self.recognizer_name,
            "Active Recognizer": self.recognizer_backend,
            "Data Source": self.data_source,
            "Condition": self.condition,
            "Num Samples": self.num_samples,
            "Evaluated Samples": self.evaluated_samples,
            "Avg FPS": self.avg_fps,
            "Avg Latency (ms)": self.avg_latency_ms,
            "CPU %": self.avg_cpu,
            "Memory (MB)": self.avg_memory,
            "Peak Memory (MB)": self.peak_memory,
            "Energy (kWh)": self.energy_kwh,
            "CER": self.cer,
            "WER": self.wer,
            "Digit Accuracy": self.digit_accuracy,
            "Sequence Accuracy": self.sequence_accuracy,
        }


class BenchmarkRunner:
    """Run a benchmark over a set of labeled digit samples."""

    def __init__(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        self.progress_callback = progress_callback

    def run(self, pipeline, samples: List[BenchmarkSample], condition: str, data_source: str) -> BenchmarkSummary:
        if not samples:
            raise ValueError("No samples were provided to the benchmark runner.")

        tracker = BenchmarkTracker(project_name=f"digit_benchmark_{pipeline.pipeline_id}")
        tracker.start_benchmark()

        ground_truths: List[str] = []
        predictions: List[str] = []
        scored_ground_truths: List[str] = []
        scored_predictions: List[str] = []
        per_sample: List[SamplePrediction] = []

        for index, sample in enumerate(samples, start=1):
            processed_image = apply_image_conditions(sample.image, condition)
            result = pipeline.run(processed_image)
            prediction = result.full_text

            ground_truths.append(sample.ground_truth)
            predictions.append(prediction)
            tracker.increment_frame_count()

            is_labeled_sample = any(character.isdigit() for character in sample.ground_truth)
            if is_labeled_sample:
                scored_ground_truths.append(sample.ground_truth)
                scored_predictions.append(prediction)
                cer = calculate_cer(sample.ground_truth, prediction)
                wer = calculate_wer(sample.ground_truth, prediction)
                digit_accuracy = calculate_digit_accuracy(sample.ground_truth, prediction)
                sequence_accuracy = calculate_sequence_accuracy(sample.ground_truth, prediction)
            else:
                cer = float("nan")
                wer = float("nan")
                digit_accuracy = float("nan")
                sequence_accuracy = float("nan")

            per_sample.append(
                SamplePrediction(
                    sample_id=sample.sample_id,
                    source=sample.source,
                    ground_truth=sample.ground_truth,
                    prediction=prediction,
                    processing_time=result.processing_time,
                    cer=cer,
                    wer=wer,
                    digit_accuracy=digit_accuracy,
                    sequence_accuracy=sequence_accuracy,
                    result=result,
                )
            )

            if self.progress_callback:
                self.progress_callback(index, len(samples))

        performance = tracker.stop_benchmark()
        metrics = (
            calculate_metrics_batch(scored_ground_truths, scored_predictions)
            if scored_ground_truths
            else {
                "cer_mean": float("nan"),
                "wer_mean": float("nan"),
                "digit_accuracy_mean": float("nan"),
                "sequence_accuracy_mean": float("nan"),
            }
        )
        latencies_ms = [sample.processing_time * 1000 for sample in per_sample]

        return BenchmarkSummary(
            pipeline_id=pipeline.pipeline_id,
            pipeline_name=pipeline.name,
            detector_name=pipeline.detector.name,
            recognizer_name=pipeline.recognizer.name,
            recognizer_backend=pipeline.recognizer.active_backend_name,
            data_source=data_source,
            condition=condition,
            num_samples=len(samples),
            evaluated_samples=len(scored_ground_truths),
            avg_fps=performance.avg_fps,
            avg_latency_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            avg_cpu=performance.avg_cpu_percent,
            avg_memory=performance.avg_memory_mb,
            peak_memory=performance.peak_memory_mb,
            energy_kwh=performance.total_energy_kwh,
            cer=float(metrics["cer_mean"]),
            wer=float(metrics["wer_mean"]),
            digit_accuracy=float(metrics["digit_accuracy_mean"]),
            sequence_accuracy=float(metrics["sequence_accuracy_mean"]),
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            sample_predictions=per_sample,
        )
