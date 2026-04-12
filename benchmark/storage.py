"""CSV persistence helpers for benchmark outputs.

All benchmark results are written to a single canonical file:
    results/benchmark_results.csv

This avoids the previous situation where the CLI defaulted to root
results.csv and the Streamlit app wrote to the same file accidentally,
while results/benchmark_results.csv grew separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .runner import BenchmarkSummary


# Single canonical results file \u2014 the results/ directory is created on first write.
DEFAULT_RESULTS_PATH = Path("results") / "benchmark_results.csv"


def append_benchmark_summary(summary: BenchmarkSummary, output_path: Optional[str] = None) -> Path:
    """Append a benchmark summary to the canonical CSV file.

    Args:
        summary: The completed benchmark summary to persist.
        output_path: Override path (used by CLI --output-csv flag). Defaults
            to results/benchmark_results.csv.

    Returns:
        The Path that was written to.
    """
    path = Path(output_path) if output_path else DEFAULT_RESULTS_PATH
    # Ensure the results directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([summary.to_row()])
    if path.exists():
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(path, index=False)
    return path


def load_benchmark_table(output_path: Optional[str] = None) -> pd.DataFrame:
    """Load the benchmark CSV into a DataFrame.

    Falls back to the legacy root results.csv if the canonical file has
    not been written yet, so old data is still visible in the leaderboard.
    """
    path = Path(output_path) if output_path else DEFAULT_RESULTS_PATH
    if path.exists():
        return pd.read_csv(path)
    # Legacy fallback \u2014 old root-level results.csv from earlier runs.
    legacy = Path("results.csv")
    if legacy.exists():
        return pd.read_csv(legacy)
    return pd.DataFrame()
