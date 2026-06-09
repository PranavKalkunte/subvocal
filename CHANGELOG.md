# Changelog

All notable changes to the Subvocal Middleware Platform will be documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-rc1] - 2026-06-08
### Added
*   **Documentation Site**: Full Docusaurus site configuration under `docs/` containing detailed guides for getting-started, custom agents, hardware abstractions, LLM systems, context, MCP integration, and model calibration.
*   **API Auto-Generation**: AST-based python docstring parser in `tools/generate_api_docs.py` to compile Markdown pages directly from code.
*   **Walkthrough Notebook**: Google Colab-compatible Jupyter notebook `notebooks/subvocal_walkthrough.ipynb` demonstrating the end-to-end signal-to-intent pipeline.
*   **E2E Smoke Tests**: Added `sdk/core/test_smoke.py` simulating full pipelines.

### Changed
*   Stabilized core class constructors and public interface signatures.

### Removed
*   **Observability Dashboard**: Removed the local HTTP dashboard (`sdk/core/dashboard.py`) in favor of the static landing page; JSONL trace logs remain the observability surface.

---

## [0.3.0] - 2026-06-08
### Added
*   **MCP Server Integration**: Implemented a stdio-based Model Context Protocol (MCP) server under `sdk/mcp/server.py` exposing pipeline status, token buffers, token injection, phrase processing, and user calibration.
*   **Pluggable Security Policies**: Added `ConfidenceThresholdPolicy`, `CommandWhitelistPolicy`, and `ContextBoundPolicy` coordinated by a `PolicyEngine` to validate actions before dispatching.
*   **Dry-Run Mode**: Added a `dry_run` flag to the orchestrator to compile intents without executing side-effects.
*   **Structured JSONL Tracing**: Implemented log tracing saving pipeline execution traces to `sdk/data/pipeline_traces.jsonl`.
*   **Observability Dashboard**: Built a zero-dependency local HTTP server (`sdk/core/dashboard.py`) serving a glassmorphic dashboard visualizing statistics, latency, and confidence charts.
*   **License Auditing**: Added `tools/check_licenses.py` validating third-party package compliance.

---

## [0.2.0] - 2026-05-25
### Added
*   **HAL Drivers**: Added `FileReplayDriver`, `SyntheticSignalGenerator`, `OpenBCICytonDriver`, and `DelsysTrignoDriver`.
*   **Research Dataset Streamers**: Added loaders for Ninapro (`.mat`), PutEMG (`.h5`), and CSL-HDEMG.
*   **EMG Classifiers**: Implemented Random Forest, 1D CNN, GRU, and Transformer classifiers with reproducible `TrainingConfig` schemas.
*   **Dynamic Calibration**: Per-user Head training calibration routine.
*   **Model Exporters**: PyTorch-to-ONNX serialization and dynamic int8 quantization.
*   **Hardware BOM**: Documented minimum $25 and full $227 wearable bio-sensing BOMs.

---

## [0.1.0] - 2026-05-10
### Added
*   **Core SDK**: Public API data structures (`Sample`, `Frame`, `CommandToken`, `Intent`, `Action`).
*   **TTS Engines**: Multi-backend Text-to-Speech offline generator prioritizing native macOS `say`/`afplay` commands.
*   **Heuristic Decoder**: Shorthand-to-intent hybrid phonetic alignment decoder.
*   **Evaluation Benchmarks**: 50-case intent-reconstruction benchmarks and LLM provider REST adapters (Claude, OpenAI, Gemini).
*   **Correction Capture**: Logging overrides to local JSONL and fine-tuning format exporters.
