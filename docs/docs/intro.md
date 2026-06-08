# Subvocal SDK Documentation

Welcome to the **Subvocal Middleware SDK** developer surface.

Subvocal SDK is a library for electromyography (sEMG) bio-signal capture, digital signal processing (DSP), gesture classification, and context-aware natural language intent reconstruction.

```text
[sEMG Hardware] ────> [DSP Filtering] ────> [ML Gesture Classifier]
                                                    │
                                                    ▼
[Action Dispatcher] <─── [Intent Reconstruction] <─── [Command Tokens]
       ▲
       └─────── [Context Snapshots]
```

## Core Ecosystem Modules

1. **`core`**: Orchestrates pipelines, security policy gates, dry-run simulations, and JSONL log tracing.
2. **`hardware`**: Conforms to pluggable abstraction drivers for File Replay, OpenBCI boards, Delsys stations, and custom dataset readers.
3. **`emg_core`**: Physiological bandpass filtering, notch shaping, TD10 feature extraction, and multi-model PyTorch classifiers (CNN, GRU, SVM, Transformer).
4. **`context`**: Queries on-screen UI visibility states, calendars, contacts, and active desktop environments.
5. **`mcp`**: Exposes intent reconstruction services as standardized tools to Model Context Protocol (MCP) clients.

Select **Getting Started** to set up the software environment and initialize your first subvocal controller.
