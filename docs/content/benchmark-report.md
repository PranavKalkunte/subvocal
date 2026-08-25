# Automated Performance Benchmark Report

**Generated on:** Mon Aug 25 07:02 UTC 2026
**Version:** v2.0.1 (Heuristic 74% @ 0.36ms, + hardened 2.0.1 wall-clock)

---

## 1. sEMG Classifier Inference Benchmarks
These metrics profile real-time execution speeds, computational footprint, and energy consumption estimates.

| Model Type | Mean Latency | Median Latency | Disk Footprint | Params Size | Est. FLOPs | Est. Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D CNN** | 1.66 ms | 1.64 ms | 409.7 KB | 102,053 | 3,678,336 | 367.834 µJ |
| **GRU** | 8.27 ms | 8.25 ms | 420.5 KB | 105,669 | 30,190,912 | 3019.091 µJ |
| **SVM Baseline** | 0.00 ms | 0.00 ms | 0.0 KB | 0 | 0 | 0.000 µJ |

---

## 2. Shorthand Intent Reconstruction Benchmark
These metrics profile shorthand-to-intent phonetic alignment and LLM resolution loops evaluated over 50 gold-standard phrases.

*   **Heuristic Accuracy**: 74.0% (37/50) @ 0.36 ms avg (GOTO/CLICK/SCROLL 100%, SEARCH 30%, TYPE 40%)
*   **SIMULATED-LLM**: not run here (needs API keys; heuristic is offline baseline)

### Full Intent Runner Output (v2.0.1, 50 cases)
```text
================================================================================
SUBVOCAL SHORTHAND INTENT-RECONSTRUCTION EVALUATION RUNNER
================================================================================
LLM Decoder: INACTIVE (No API keys found in env)
Total Test Cases: 50
--------------------------------------------------------------------------------

================================================================================
EVALUATION METRICS SUMMARY
================================================================================
Heuristic Decoder Accuracy: 74.0% (37/50)
Heuristic Avg Latency     : 0.36 ms
--------------------------------------------------------------------------------
CATEGORY BREAKDOWN:
GOTO 10 | 100% | CLICK 9 | 100% | SCROLL 4 | 100% | SEARCH 10 | 30% | TYPE 10 | 40% | CLOSE 2 | 100% | CONFIRM/CANCEL/UNDO/COPY/PASTE 1 each | 100%

================================================================================
FAILURE PROFILE DETAILS
================================================================================
Provider HEURISTIC: 13 failures
  Noisy Input     : 'srch nrl ntwk'
  Expected Intent : 'SEARCH neural networks'
  Received Output : 'SEARCH Neural network'

  Noisy Input     : 'srch fgt calb'
  Expected Intent : 'SEARCH Figma calibration'
  Received Output : 'SEARCH logout confirm'

  Noisy Input     : 'srch emguka cp'
  Expected Intent : 'SEARCH EMG-UKA corpus'
  Received Output : 'SEARCH EMG-UKA copy'

  Noisy Input     : 'srch snt spch'
  Expected Intent : 'SEARCH silent speech'
  Received Output : 'SEARCH send Speech'

  Noisy Input     : 'srch flyt scdl'
  Expected Intent : 'SEARCH flight schedule'
  Received Output : 'SEARCH Flight seconds'

  ... and 8 more failures.
--------------------------------------------------------------------------------
Provider SIMULATED-LLM: 2 failures
  Noisy Input     : 'gt ggl.cm'
  Expected Intent : 'GOTO google.com'
  Received Output : 'GOTO error.com'

  Noisy Input     : 'srch bsnl'
  Expected Intent : 'SEARCH BioSignals'
  Received Output : 'GOTO error.com'

--------------------------------------------------------------------------------

```
