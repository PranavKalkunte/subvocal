---
title: emg_core.ml.benchmark
sidebar_label: benchmark
---

Inference benchmarking harness for sEMG classifiers.

## Functions

### `estimate_flops`

```python
def estimate_flops(model_type: str, num_channels: int = 4, segment_length: int = 150, num_classes: int = 4, hidden_size: int = 64, num_layers: int = 2) -> int
```

Estimate floating point operations (FLOPs) per forward inference pass.

Formulas derived from structural layers (convolutions, linear projections, attention, and recurrences).

### `run_benchmark`

```python
def run_benchmark(user_id: str, model_type: str, num_runs: int = 200) -> Dict[str, Any]
```

Benchmark model inference latency, disk footprint, and parameter size.

Estimates computational FLOPs and electrical energy footprint.
