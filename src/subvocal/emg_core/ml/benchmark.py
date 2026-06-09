"""Inference benchmarking harness for sEMG classifiers."""

import os
import time
from typing import Any

import numpy as np

from subvocal.emg_core.ml.infer import InferenceEngine
from subvocal.emg_core.ml.model_io import get_model_path


def estimate_flops(model_type: str, num_channels: int = 4, segment_length: int = 150, num_classes: int = 4, hidden_size: int = 64, num_layers: int = 2) -> int:
    """Estimate floating point operations (FLOPs) per forward inference pass.

    Formulas derived from structural layers (convolutions, linear projections, attention, and recurrences).
    """
    if model_type in ["rf", "svm"]:
        # Classical feature extraction (TD10 sliding average + window context) + RF/SVM forest/kernel evaluation
        # Approximately 250,000 operations for DSP and 500,000 for classification.
        return 750000

    elif model_type == "cnn":
        # Conv1: Conv1d(4, 32, 5) -> 2 * 5 * 4 * 32 * 150 = 192,000
        # Conv2: Conv1d(32, 64, 5) -> Pool 2x -> 2 * 5 * 32 * 64 * 75 = 1,536,000
        # Conv3: Conv1d(64, 128, 3) -> Pool 2x -> 2 * 3 * 64 * 128 * 37 = 1,818,624
        # FC1: Linear(128 * 8 -> 64) -> 2 * 1024 * 64 = 131,072
        # FC2: Linear(64 -> num_classes) -> 2 * 64 * num_classes = 512
        conv1_flops = 2 * 5 * num_channels * 32 * segment_length
        conv2_flops = 2 * 5 * 32 * 64 * (segment_length // 2)
        conv3_flops = 2 * 3 * 64 * 128 * (segment_length // 4)
        fc1_flops = 2 * (128 * 8) * 64
        fc2_flops = 2 * 64 * num_classes
        return conv1_flops + conv2_flops + conv3_flops + fc1_flops + fc2_flops

    elif model_type == "gru":
        # GRU Cell Step FLOPs per direction: 3 * 2 * (hidden_size * (hidden_size + input_size) + hidden_size)
        # Layer 1 (bidirectional, input=4, hidden=64): 2 * 3 * 2 * (64 * (64 + 4) + 64) * 150 = 2 * 26,496 * 150 = 7,948,800
        # Layer 2 (bidirectional, input=128, hidden=64): 2 * 3 * 2 * (64 * (64 + 128) + 64) * 150 = 2 * 74,112 * 150 = 22,233,600
        # FC: Linear(hidden_size * 2 -> 32) -> 2 * 128 * 32 = 8,192
        # FC: Linear(32 -> num_classes) -> 2 * 32 * num_classes = 256
        l1_step = 2 * 3 * 2 * (hidden_size * (hidden_size + num_channels) + hidden_size)
        l2_step = 2 * 3 * 2 * (hidden_size * (hidden_size + hidden_size * 2) + hidden_size)
        gru_flops = (l1_step + l2_step) * segment_length
        fc1_flops = 2 * (hidden_size * 2) * 32
        fc2_flops = 2 * 32 * num_classes
        return gru_flops + fc1_flops + fc2_flops

    elif model_type == "transformer":
        # Linear projection: 2 * num_channels * d_model * segment_length = 76,800
        # QKV Proj: 2 * d_model * d_model * 3 * segment_length = 3,686,400
        # Attention score QK^T: 2 * d_model * segment_length * segment_length = 2,880,000
        # Attention over values V: 2 * segment_length * segment_length * d_model = 2,880,000
        # Out Proj: 2 * d_model * d_model * segment_length = 1,228,800
        # FFN Linear 1: 2 * d_model * dim_feedforward * segment_length = 2,457,600
        # FFN Linear 2: 2 * dim_feedforward * d_model * segment_length = 2,457,600
        # Multiply by num_layers=2: ~31,200,000 FLOPs
        proj_flops = 2 * num_channels * hidden_size * segment_length
        qkv_flops = 2 * hidden_size * hidden_size * 3 * segment_length
        attn_flops = 2 * hidden_size * segment_length * segment_length
        val_flops = 2 * segment_length * segment_length * hidden_size
        out_flops = 2 * hidden_size * hidden_size * segment_length
        ffn_flops = 2 * hidden_size * (hidden_size * 2) * segment_length * 2
        layer_flops = (qkv_flops + attn_flops + val_flops + out_flops + ffn_flops) * num_layers
        fc1_flops = 2 * hidden_size * 32
        fc2_flops = 2 * 32 * num_classes
        return proj_flops + layer_flops + fc1_flops + fc2_flops

    else:
        return 0


def run_benchmark(user_id: str, model_type: str, num_runs: int = 200) -> dict[str, Any]:
    """Benchmark model inference latency, disk footprint, and parameter size.

    Estimates computational FLOPs and electrical energy footprint.
    """
    engine = InferenceEngine(user_id=user_id, model_type=model_type)
    num_classes = len(engine.labels)

    # Derive shape from the loaded model so it works for any channel count
    num_channels = getattr(engine, "_mean", None)
    if num_channels is not None:
        num_channels = engine._mean.shape[1]  # mean shape is (1, C, 1)
    else:
        num_channels = 4  # classical models don't store mean; 4 is the synthetic default
    segment_length = 150

    dummy_input = np.random.normal(0.0, 1.0, (segment_length, num_channels))
    
    # Warmup runs to stabilize CPU cache / thread pool
    for _ in range(20):
        engine.predict_raw(dummy_input)

    # Benchmark loop
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        engine.predict_raw(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000.0) # in ms

    # Latency stats
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))

    # Memory/Footprint metrics
    model_path = get_model_path(user_id, model_type)
    disk_size_kb = os.path.getsize(model_path) / 1024.0 if os.path.exists(model_path) else 0.0
    
    param_count = 0
    if hasattr(engine, "_model") and hasattr(engine._model, "parameters"):
        param_count = sum(p.numel() for p in engine._model.parameters())

    # FLOPs and energy estimation (Assuming 100 pJ per FLOP for mobile/edge execution)
    base_type = model_type.replace("_quantized", "")
    flops = estimate_flops(base_type, num_classes=num_classes)
    
    # Quantized model requires less computational resources
    if "quantized" in model_type:
        flops = int(flops * 0.7)  # approx 30% reduction due to int8 ops

    # 100 pJ = 0.0001 microjoules
    energy_uj = flops * 100 * 1e-6

    metrics = {
        "model_type": model_type,
        "latency_mean_ms": mean_lat,
        "latency_median_ms": med_lat,
        "latency_p95_ms": p95_lat,
        "latency_std_ms": std_lat,
        "disk_size_kb": disk_size_kb,
        "parameter_count": param_count,
        "estimated_flops": flops,
        "estimated_energy_uj": energy_uj
    }
    return metrics


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Benchmark sEMG inference performance.")
    parser.add_argument("--user_id", type=str, default="pretrained", help="User ID of model")
    parser.add_argument("--model_type", type=str, default="cnn", help="Model type to benchmark")
    parser.add_argument("--runs", type=int, default=200, help="Number of benchmark iterations")
    args = parser.parse_args()

    try:
        res = run_benchmark(args.user_id, args.model_type, num_runs=args.runs)
        print(f"\n================ BENCHMARK RESULTS ({args.model_type.upper()}) ================")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Failed to benchmark model {args.model_type}: {e}")
