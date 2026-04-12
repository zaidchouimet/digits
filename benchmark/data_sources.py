"""Data loading helpers for benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from utils import CustomDigitDataset, SVHNDataset, SVHNFormat1Dataset

from .runner import BenchmarkSample


def create_sample_video(path: str = "results/generated_sample_video.mp4", duration: int = 10, fps: int = 30) -> str:
    """Create a deterministic sample video with digit strings moving across the frame.

    If the file already exists it is reused without re-rendering, which saves
    several seconds on repeated benchmark runs.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip generation if the file already exists.
    if output_path.exists():
        return str(output_path)

    width, height = 640, 480
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_index in range(duration * fps):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        digits = []
        for digit_index in range(5):
            digit = str((frame_index + digit_index) % 10)
            digits.append(digit)
            x = 50 + digit_index * 110
            y = 240 + int(40 * np.sin(frame_index / 18 + digit_index))
            cv2.putText(frame, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (0, 0, 0), 4)
        writer.write(frame)

    writer.release()
    return str(output_path)


def load_sample_video_samples(limit: int = 50) -> List[BenchmarkSample]:
    """Load labeled frames from the generated sample video."""
    video_path = create_sample_video()
    capture = cv2.VideoCapture(video_path)
    samples: List[BenchmarkSample] = []

    frame_index = 0
    while frame_index < limit:
        success, frame = capture.read()
        if not success:
            break
        ground_truth = "".join(str((frame_index + offset) % 10) for offset in range(5))
        samples.append(
            BenchmarkSample(
                image=frame,
                ground_truth=ground_truth,
                sample_id=f"sample_video_{frame_index:04d}",
                source="sample_video",
            )
        )
        frame_index += 1

    capture.release()
    return samples


def load_streamlit_upload_samples(uploaded_files, ground_truth_input: str) -> List[BenchmarkSample]:
    """Decode uploaded Streamlit images into benchmark samples."""
    labels = [value.strip() for value in ground_truth_input.split(",")] if ground_truth_input else []
    samples: List[BenchmarkSample] = []

    for index, uploaded_file in enumerate(uploaded_files):
        # Reset file pointer to beginning - this fixes the empty prediction issue!
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            continue

        ground_truth = labels[index] if index < len(labels) else ""
        samples.append(
            BenchmarkSample(
                image=image,
                ground_truth=ground_truth,
                sample_id=f"upload_{index:04d}",
                source="uploaded_images",
            )
        )

    return samples


def load_directory_samples(image_dir: str, label_file: Optional[str] = None) -> List[BenchmarkSample]:
    """Load CLI image directory samples."""
    dataset = CustomDigitDataset()
    images, labels = dataset.load_from_folder(image_dir, label_file=label_file)
    samples: List[BenchmarkSample] = []
    for index, (image, label) in enumerate(zip(images, labels)):
        samples.append(
            BenchmarkSample(
                image=image,
                ground_truth=label,
                sample_id=f"dir_{index:04d}",
                source=str(Path(image_dir)),
            )
        )
    return samples


def load_svhn_samples(split: str, dataset_size: int, target_size: int = 256) -> List[BenchmarkSample]:
    """Load SVHN samples and upscale them for OCR pipelines."""
    dataset = SVHNDataset()
    images, labels = dataset.load_data(split, max_samples=dataset_size)
    samples: List[BenchmarkSample] = []
    for index, (image, label) in enumerate(zip(images, labels)):
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        samples.append(
            BenchmarkSample(
                image=resized,
                ground_truth=label,
                sample_id=f"svhn_{split}_{index:04d}",
                source=f"svhn_{split}",
            )
        )
    return samples


def load_svhn_format1_samples(split: str, dataset_size: int, target_size: int = 256) -> List[BenchmarkSample]:
    """Load SVHN Format 1 samples from our downloaded dataset with separate label files."""
    dataset = SVHNFormat1Dataset()
    images, labels = dataset.load_data(split, max_samples=dataset_size)
    samples: List[BenchmarkSample] = []
    for index, (image, label) in enumerate(zip(images, labels)):
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        samples.append(
            BenchmarkSample(
                image=resized,
                ground_truth=str(label),  # Convert to string for consistency
                sample_id=f"svhn_format1_{split}_{index:04d}",
                source=f"svhn_format1_{split}",
            )
        )
    return samples
