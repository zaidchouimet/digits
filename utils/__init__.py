"""
Utilities module for digit recognition benchmarking.
"""

from .metrics import (
    calculate_cer, 
    calculate_wer, 
    calculate_digit_accuracy, 
    calculate_sequence_accuracy,
    calculate_metrics_batch,
    format_metrics_summary
)
from .dataset_loader import (
    SVHNDataset, 
    SVHNFormat1Dataset,
    CustomDigitDataset, 
    get_dataset_loader,
    apply_image_conditions
)
from .energy_tracker import (
    PerformanceMetrics,
    ResourceMonitor,
    BenchmarkTracker
)
