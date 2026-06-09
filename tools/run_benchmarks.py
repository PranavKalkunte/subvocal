#!/usr/bin/env python3
"""Automated Benchmark Suite Runner for Subvocal Middleware.

Runs ML inference benchmarks (latency, disk, parameters, FLOPs) and
Heuristic Intent Reconstruction benchmarks, outputting a structured Markdown report.
"""

import json
import os
import subprocess
import sys
from typing import Any

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "platform", "benchmark_report.md")


def run_ml_benchmark(model_type: str) -> dict[str, Any]:
    """Run the ML inference benchmarking script."""
    cmd = [
        sys.executable,
        "-m", "subvocal.emg_core.ml.benchmark",
        "--user_id", "pretrained",
        "--model_type", model_type,
        "--runs", "100"
    ]
    try:
        env = os.environ.copy()
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env, check=True)
        # Parse the JSON portion of stdout
        lines = result.stdout.strip().split("\n")
        json_str = ""
        capture = False
        for line in lines:
            if line.strip().startswith("{"):
                capture = True
            if capture:
                json_str += line
        if json_str:
            return json.loads(json_str)
    except Exception as e:
        print(f"Warning: Failed to benchmark {model_type}: {e}")
    return {}


def run_intent_benchmark() -> str:
    """Run the intent reconstruction benchmark harness."""
    cmd = [
        sys.executable,
        "benchmarks/intent_eval.py"
    ]
    try:
        env = os.environ.copy()
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        return result.stdout
    except Exception as e:
        return f"Failed to run intent runner: {e}"


def main():
    print("======================================================================")
    print("Running Subvocal Middleware Release Benchmarks...")
    print("======================================================================")
    
    # 1. Run ML Benchmarks
    print("Benchmarking ML Classifier Inferences...")
    cnn_results = run_ml_benchmark("cnn")
    gru_results = run_ml_benchmark("gru")
    svm_results = run_ml_benchmark("svm")
    
    # 2. Run Intent Benchmarks
    print("Benchmarking Shorthand Intent Reconstruction...")
    intent_output = run_intent_benchmark()
    
    # Extract intent accuracy line (e.g., "Accuracy: 37/50 (74.0%)")
    accuracy_str = "N/A"
    latency_str = "N/A"
    for line in intent_output.split("\n"):
        if "Accuracy:" in line:
            accuracy_str = line.strip()
        if "Average latency:" in line:
            latency_str = line.strip()

    # 3. Write Markdown Report
    print(f"Writing compilation report to {REPORT_PATH}...")
    
    report = f"""# Automated Performance Benchmark Report

**Generated on:** {subprocess.check_output(["date"]).decode("utf-8").strip()}  
**Version:** v0.3.0-alpha (Automated release run)

---

## 1. sEMG Classifier Inference Benchmarks
These metrics profile real-time execution speeds, computational footprint, and energy consumption estimates.

| Model Type | Mean Latency | Median Latency | Disk Footprint | Params Size | Est. FLOPs | Est. Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D CNN** | {cnn_results.get('latency_mean_ms', 0.0):.2f} ms | {cnn_results.get('latency_median_ms', 0.0):.2f} ms | {cnn_results.get('disk_size_kb', 0.0):.1f} KB | {cnn_results.get('parameter_count', 0):,} | {cnn_results.get('estimated_flops', 0):,} | {cnn_results.get('estimated_energy_uj', 0.0):.3f} µJ |
| **GRU** | {gru_results.get('latency_mean_ms', 0.0):.2f} ms | {gru_results.get('latency_median_ms', 0.0):.2f} ms | {gru_results.get('disk_size_kb', 0.0):.1f} KB | {gru_results.get('parameter_count', 0):,} | {gru_results.get('estimated_flops', 0):,} | {gru_results.get('estimated_energy_uj', 0.0):.3f} µJ |
| **SVM Baseline** | {svm_results.get('latency_mean_ms', 0.0):.2f} ms | {svm_results.get('latency_median_ms', 0.0):.2f} ms | {svm_results.get('disk_size_kb', 0.0):.1f} KB | {svm_results.get('parameter_count', 0):,} | {svm_results.get('estimated_flops', 0):,} | {svm_results.get('estimated_energy_uj', 0.0):.3f} µJ |

---

## 2. Shorthand Intent Reconstruction Benchmark
These metrics profile shorthand-to-intent phonetic alignment and LLM resolution loops evaluated over 50 gold-standard phrases.

*   **Intent Resolution Accuracy**: {accuracy_str}
*   **Intent Resolution Latency**: {latency_str}

### Full Intent Runner Output
```text
{intent_output}
```
"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
        
    print("Benchmarks completed and saved successfully.")


if __name__ == "__main__":
    main()
