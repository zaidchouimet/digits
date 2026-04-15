"""CLI and Streamlit entrypoints for the digit recognition benchmark system."""

from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
import sys
from typing import List

from benchmark.data_sources import (
    iter_video_samples,
    load_dataset_dataset_samples,
    load_directory_samples,
    load_sample_video_samples,
    load_streamlit_upload_samples,
    load_svhn_samples,
    load_svhn_format1_samples,
)
from benchmark import data_sources as benchmark_data_sources
from benchmark.runner import BenchmarkRunner
from benchmark.storage import append_benchmark_summary, load_benchmark_table
from pipelines import get_pipeline, get_pipeline_categories, get_pipeline_spec, list_pipeline_specs


LOGGER = logging.getLogger("digit_benchmark")


def format_metric_value(value: float, decimals: int = 3, suffix: str = "") -> str:
    """Format a numeric metric while preserving unlabeled samples as N/A."""
    if value is None:
        return "N/A"
    try:
        if math.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value:.{decimals}f}{suffix}"


def configure_logging() -> None:
    """Configure a process-wide logger once."""
    if LOGGER.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_pipeline_or_raise(pipeline_name: str):
    """Instantiate and load a pipeline or raise a descriptive error."""
    pipeline = get_pipeline(pipeline_name)
    if pipeline.load_models():
        return pipeline
    raise RuntimeError(pipeline.last_error or f"Failed to load pipeline '{pipeline_name}'")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Scientifically valid digit recognition benchmark runner.",
    )
    parser.add_argument("--pipeline", help="Pipeline ID or alias, for example 'yolo_paddleocr'.")
    parser.add_argument(
        "--data-source",
        default="svhn_format1",
        choices=["svhn_format1", "dataset", "images", "video"],
        help="Benchmark source.",
    )
    parser.add_argument(
        "--condition",
        default="clean",
        choices=["clean", "blurry", "noisy", "low_contrast"],
        help="Visual corruption applied before inference.",
    )
    parser.add_argument("--svhn-size", type=int, default=50, help="Number of SVHN samples to benchmark.")
    parser.add_argument(
        "--svhn-split",
        default="test",
        choices=["train", "test"],
        help="SVHN split used during benchmarking.",
    )
    parser.add_argument("--image-dir", help="Directory containing input images for --data-source images.")
    parser.add_argument("--video-path", help="Path to a video file when --data-source video is selected.")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum video frames to benchmark.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame from the video.")
    parser.add_argument("--label-file", help="Optional CSV file with image_name,label columns.")
    parser.add_argument("--output-csv", default="results.csv", help="CSV file used to append benchmark results.")
    parser.add_argument("--list-pipelines", action="store_true", help="List available pipeline IDs and exit.")
    return parser


def get_cli_samples(args: argparse.Namespace) -> List[BenchmarkSample]:
    """Load samples for requested CLI data source."""
    if args.data_source == "svhn_format1":
        return load_svhn_format1_samples(split=args.svhn_split, dataset_size=args.svhn_size)
    if args.data_source == "dataset":
        return load_dataset_dataset_samples(split=args.svhn_split, dataset_size=args.svhn_size)
    if args.data_source == "images":
        if not args.image_dir:
            raise ValueError("--image-dir is required when --data-source images is selected.")
        return load_directory_samples(args.image_dir, label_file=args.label_file)
    if args.data_source == "video":
        if not args.video_path:
            raise ValueError("--video-path is required when --data-source video is selected.")
        return benchmark_data_sources.load_video_samples(
            video_path=args.video_path,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            source_name="video_cli",
        )
    raise ValueError(f"Unsupported data source: {args.data_source}")


def print_pipeline_list() -> None:
    """Print available benchmark pipelines."""
    for spec in list_pipeline_specs():
        alias_text = f" aliases={', '.join(spec.aliases)}" if spec.aliases else ""
        print(f"{spec.pipeline_id}: {spec.display_name} [{spec.category}]{alias_text}")


def print_summary(summary, output_path: str) -> None:
    """Print a compact CLI summary for one benchmark run."""
    print()
    print(f"Pipeline: {summary.pipeline_name}")
    print(f"Detector: {summary.detector_name}")
    print(f"Recognizer: {summary.recognizer_name} -> {summary.recognizer_backend}")
    print(f"Source: {summary.data_source} | Condition: {summary.condition}")
    print(f"Samples: {summary.num_samples}")
    print(f"FPS: {summary.avg_fps:.2f} | Latency: {summary.avg_latency_ms:.1f} ms")
    print(f"CPU: {summary.avg_cpu:.1f}% | Memory: {summary.avg_memory:.1f} MB | Energy: {summary.energy_kwh:.6f} kWh")
    print(
        "CER: "
        f"{summary.cer:.3f} | WER: {summary.wer:.3f} | "
        f"Digit Accuracy: {summary.digit_accuracy:.3f} | Sequence Accuracy: {summary.sequence_accuracy:.3f}"
    )
    print(f"Results saved to {output_path}")


def run_cli(args: argparse.Namespace) -> int:
    """Execute a benchmark run from the terminal."""
    if args.list_pipelines:
        print_pipeline_list()
        return 0

    if not args.pipeline:
        print("error: --pipeline is required unless --list-pipelines is used.", file=sys.stderr)
        return 2

    pipeline = load_pipeline_or_raise(args.pipeline)
    samples = get_cli_samples(args)
    if not samples:
        raise RuntimeError("No benchmark samples were loaded.")

    def on_progress(current: int, total: int) -> None:
        print(f"\rProcessing {current}/{total}", end="", flush=True)
        if current == total:
            print()

    runner = BenchmarkRunner(progress_callback=on_progress)
    summary = runner.run(pipeline, samples, condition=args.condition, data_source=args.data_source)
    output_path = append_benchmark_summary(summary, args.output_csv)
    print_summary(summary, str(output_path))
    return 0


def is_running_under_streamlit() -> bool:
    """Detect whether the file is being executed by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            return True
    except Exception:
        pass

    argv_text = " ".join(sys.argv).lower()
    return "streamlit" in argv_text


def render_streamlit_app() -> None:
    """Run the Streamlit dashboard."""
    import cv2
    import streamlit as st

    st.set_page_config(
        page_title="Digit Recognition Benchmark System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Digit Recognition Benchmark System")
    st.caption(
        "Academic benchmark for digit recognition using explicit detection + recognition pipelines "
        "and OCR-only baselines."
    )

    def load_pipeline_cached(pipeline_name: str):
        return load_pipeline_or_raise(pipeline_name)

    with st.sidebar:
        st.header("Benchmark Configuration")
        pipeline_options = list_pipeline_specs()
        pipeline_name = st.selectbox(
            "Pipeline",
            options=[spec.display_name for spec in pipeline_options],
        )
        # Note: PIPELINE_LOOKUP maps both pipeline_ids and display_names to PipelineSpec
        # objects, so passing the chosen display_name here works natively.
        pipeline_spec = get_pipeline_spec(pipeline_name)
        st.markdown(f"**Category:** {pipeline_spec.category}")
        st.caption(pipeline_spec.description)


        data_source = st.radio("Data Source", options=["Upload Images", "Upload Video", "SVHN Format 1", "Dataset Folder"])
        condition = st.selectbox("Condition", options=["clean", "blurry", "noisy", "low_contrast"])

        dataset_size = 50
        dataset_split = "test"
        uploaded_files = []
        uploaded_video = None
        max_frames = 50
        frame_stride = 1
        live_preview = False
        preview_every_n = 1
        skip_empty_video_frames = True
        adaptive_streaming = True
        target_stream_fps = 12
        max_adaptive_skip = 8
        ground_truth_input = ""

        if data_source == "SVHN Format 1":
            st.markdown("**SVHN Format 1 Dataset (Our Downloaded Version)**")
            st.caption("Using datasets/svhn_format1 with separate image and label files")
            
            dataset_size = st.selectbox("Dataset Size", options=[10, 50, 100, 500], index=1)
            dataset_split = st.selectbox("Dataset Split", options=["test", "train"], index=0)
            
            # Show dataset info
            if dataset_split == "train":
                st.info(f"Training set: 73,257 samples with digit labels 0-9")
            else:
                st.info(f"Test set: 26,032 samples with digit labels 0-9")
        elif data_source == "Dataset Folder":
            st.markdown("**Dataset Folder (datasets/dataset)**")
            st.caption("Uses datasets/dataset/labels/*.txt with matching image filenames.")

            dataset_size = st.selectbox("Dataset Size", options=[10, 50, 100, 500], index=1)
            dataset_split = st.selectbox("Dataset Split", options=["test", "train"], index=0)
            st.info("Split is deterministic: first 80% train, last 20% test.")
        elif data_source == "Upload Images":
            uploaded_files = st.file_uploader(
                "Upload Images",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
            )
            ground_truth_input = st.text_input(
                "Ground Truth (comma-separated)",
                placeholder="01234, 56789",
            )
        elif data_source == "Upload Video":
            uploaded_video = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                accept_multiple_files=False,
            )
            max_frames = st.selectbox("Max Frames", options=[10, 25, 50, 100, 200], index=2)
            frame_stride = st.selectbox("Frame Stride", options=[1, 2, 3, 5, 10], index=0)
            skip_empty_video_frames = st.checkbox(
                "Ignore frames with no detections (faster stream)",
                value=True,
            )
            adaptive_streaming = st.checkbox(
                "Adaptive real-time streaming (dynamic frame skipping)",
                value=True,
            )
            if adaptive_streaming:
                target_stream_fps = st.slider("Target Stream FPS", min_value=5, max_value=30, value=12, step=1)
                max_adaptive_skip = st.slider("Max Adaptive Frame Skip", min_value=1, max_value=20, value=8, step=1)
            live_preview = st.checkbox("Live Stream Preview", value=True)
            if live_preview:
                preview_every_n = st.selectbox("Preview Every N Processed Frames", options=[1, 2, 3, 5, 10], index=0)
            st.caption("Processes frames without labels (metrics like CER/WER will show N/A).")

        run_button = st.button("Run Benchmark", type="primary")

    summary = st.session_state.get("benchmark_summary")

    if run_button:
        temp_video_path_for_cleanup = None
        try:
            pipeline = load_pipeline_cached(pipeline_name)
            total_samples = None
            if data_source == "Sample Video":
                samples = load_sample_video_samples(limit=50)
                source_name = "sample_video"
            elif data_source == "Upload Images":
                if not uploaded_files:
                    st.error("Please upload at least one image.")
                    return
                samples = load_streamlit_upload_samples(uploaded_files, ground_truth_input)
                if ground_truth_input:
                    expected_count = len([value.strip() for value in ground_truth_input.split(",") if value.strip()])
                    if expected_count and expected_count != len(samples):
                        st.error("The number of ground-truth entries must match the number of uploaded images.")
                        return
                source_name = "uploaded_images"
            elif data_source == "SVHN Format 1":
                samples = load_svhn_format1_samples(split=dataset_split, dataset_size=dataset_size)
                source_name = f"svhn_format1_{dataset_split}"
            elif data_source == "Dataset Folder":
                samples = load_dataset_dataset_samples(split=dataset_split, dataset_size=dataset_size)
                source_name = f"dataset_{dataset_split}"
            elif data_source == "Upload Video":
                if not uploaded_video:
                    st.error("Please upload a video.")
                    return
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_video.name}") as temp_video:
                    temp_video.write(uploaded_video.getbuffer())
                    temp_video_path = temp_video.name
                temp_video_path_for_cleanup = temp_video_path
                samples = iter_video_samples(
                    video_path=temp_video_path,
                    max_frames=max_frames,
                    frame_stride=frame_stride,
                    source_name="uploaded_video",
                )
                total_samples = max_frames
                source_name = "uploaded_video"
            else:
                samples = load_svhn_samples(split=dataset_split, dataset_size=dataset_size)
                source_name = f"svhn_{dataset_split}"

            progress_bar = st.progress(0.0)
            status = st.empty()
            preview_area = st.empty() if data_source == "Upload Video" and live_preview else None
            preview_text = st.empty() if data_source == "Upload Video" and live_preview else None

            def on_progress(current: int, total: int) -> None:
                safe_total = max(total, 1)
                progress_bar.progress(min(current / safe_total, 1.0))
                status.write(f"Processing sample {current}/{total}")

            def on_sample(current: int, total: int, sample, result) -> None:
                if preview_area is None or preview_text is None:
                    return
                if current % max(preview_every_n, 1) != 0:
                    return
                image_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
                preview_area.image(
                    image_rgb,
                    caption=f"Live stream preview ({current}/{total})",
                    use_container_width=True,
                )
                realtime_fps = result.metadata.get("realtime_fps", 0.0)
                adaptive_skip = int(result.metadata.get("adaptive_skip", 0))
                target_fps = float(result.metadata.get("target_fps", target_stream_fps))
                preview_text.markdown(
                    f"**Current OCR:** `{result.full_text or '(empty)'}`  \n"
                    f"**Realtime FPS:** `{realtime_fps:.2f}` / target `{target_fps:.1f}`  \n"
                    f"**Adaptive Skip:** `{adaptive_skip}`"
                )

            try:
                runner = BenchmarkRunner(progress_callback=on_progress, sample_callback=on_sample)
            except TypeError:
                # Backward compatibility if an older BenchmarkRunner signature is loaded.
                runner = BenchmarkRunner(progress_callback=on_progress)
            summary = runner.run(
                pipeline,
                samples,
                condition=condition,
                data_source=source_name,
                total_samples=total_samples,
                stream_skip_empty_frames=(data_source == "Upload Video" and skip_empty_video_frames),
                adaptive_streaming=(data_source == "Upload Video" and adaptive_streaming),
                target_fps=float(target_stream_fps),
                max_adaptive_skip=int(max_adaptive_skip),
            )
            output_path = append_benchmark_summary(summary)
            st.session_state["benchmark_summary"] = summary
            progress_bar.progress(1.0)
            status.write(f"Saved benchmark to {output_path}")
        except Exception as error:
            import traceback
            error_msg = str(error)
            st.error(f"Pipeline Execution Failed: {error_msg}")
            
            # Common failure modes get friendlier hints
            if "tesseract" in error_msg.lower():
                st.info("💡 Hint: Ensure Tesseract OCR is installed and TESSERACT_PATH is set.")
            elif "weights" in error_msg.lower() or ".pt" in error_msg.lower() or "not found" in error_msg.lower():
                st.info("💡 Hint: Ensure required model weights (e.g., yolov8n.pt) exist in the project root.")
            
            with st.expander("Show detailed traceback"):
                st.code(traceback.format_exc(), language="python")
        finally:
            if temp_video_path_for_cleanup:
                try:
                    os.unlink(temp_video_path_for_cleanup)
                except OSError:
                    pass

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.header("Results Visualization")
        if not summary:
            st.info("Run a benchmark to inspect metrics and sample predictions.")
        else:
            st.success(f"Benchmark completed for {summary.pipeline_name}")
            st.caption(
                f"Detector: {summary.detector_name} | Recognizer: {summary.recognizer_name} -> {summary.recognizer_backend}"
            )
            if summary.evaluated_samples < summary.num_samples:
                st.warning(
                    f"Accuracy metrics were computed on {summary.evaluated_samples} of "
                    f"{summary.num_samples} samples because some uploads had no digit labels."
                )

            metric_columns = st.columns(4)
            metric_columns[0].metric("FPS", f"{summary.avg_fps:.2f}")
            metric_columns[1].metric("Latency", f"{summary.avg_latency_ms:.1f} ms")
            metric_columns[2].metric("CPU", f"{summary.avg_cpu:.1f}%")
            metric_columns[3].metric("Energy", f"{summary.energy_kwh:.6f} kWh")

            accuracy_columns = st.columns(4)
            accuracy_columns[0].metric("CER", format_metric_value(summary.cer))
            accuracy_columns[1].metric("WER", format_metric_value(summary.wer))
            accuracy_columns[2].metric("Digit Accuracy", format_metric_value(summary.digit_accuracy))
            accuracy_columns[3].metric("Sequence Accuracy", format_metric_value(summary.sequence_accuracy))

            if len(summary.sample_predictions) == 1:
                sample = summary.sample_predictions[0]
                st.caption("Showing the only available sample.")
            else:
                sample_index = st.slider(
                    "Sample Index",
                    min_value=0,
                    max_value=len(summary.sample_predictions) - 1,
                    value=0,
                )
                sample = summary.sample_predictions[sample_index]
            image_rgb = cv2.cvtColor(sample.result.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Processed Image", use_container_width=True)
            ground_truth_text = sample.ground_truth if sample.ground_truth else "(not provided)"
            prediction_text = sample.prediction if sample.prediction else "(empty)"
            st.markdown(
                f"**Ground Truth:** `{ground_truth_text}`  \n"
                f"**Prediction:** `{prediction_text}`  \n"
                f"**Latency:** `{sample.processing_time * 1000:.1f} ms`  \n"
                f"**CER:** `{format_metric_value(sample.cer)}`  \n"
                f"**Digit Accuracy:** `{format_metric_value(sample.digit_accuracy)}`"
            )

    with right_col:
        st.header("Leaderboard")
        results_table = load_benchmark_table()
        if results_table.empty:
            st.info("No benchmark runs saved yet.")
        else:
            st.dataframe(results_table, use_container_width=True)

        st.subheader("Pipeline Registry")
        for category, names in get_pipeline_categories().items():
            st.markdown(f"**{category}**")
            for name in names:
                st.write(name)


def main(argv: List[str] | None = None) -> int:
    """Select the CLI or Streamlit execution mode."""
    configure_logging()

    if is_running_under_streamlit():
        render_streamlit_app()
        return 0

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
