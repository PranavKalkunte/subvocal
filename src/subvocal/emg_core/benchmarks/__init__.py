"""EMG benchmarks package — EMGBench (NeurIPS 2024)."""

from .emgbench import DEFAULT_DATASETS, EMGBench, aggregate_loso_scores, summarize_results

__all__ = ["EMGBench", "DEFAULT_DATASETS", "summarize_results", "aggregate_loso_scores"]
