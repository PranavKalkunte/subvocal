# Changelog

All notable changes to the Subvocal Middleware Platform will be documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-06-09
### Added
*   **LiveKit-Inspired Concurrency**: Introduced `subvocal/utils/concurrency.py` implementing `OpsQueue` (serialized thread worker execution), `IncrementalDispatcher` (thread-safe condition-based fan-out), `ChangeNotifier` (async keyed callback registry), and resettable `Debouncer` timers.
*   **Unified Configuration Management**: Integrated strict Pydantic configurations in `subvocal/config.py` with `extra="forbid"` to reject unknown YAML keys. Added support for double-underscore nested environment variable overrides (e.g. `SUBVOCAL_HARDWARE__SAMPLE_RATE`).
*   **Physiological Signal Monitoring**: Upgraded stream processing in `subvocal/stream/`:
    *   `FrameRing` & `StreamStats`: Circular buffer frame ingestion with windowed statistics.
    *   `SignalLevel`: EMA-smoothed signal activity tracking tailored for sEMG amplitudes.
    *   `StreamTracker`: Hysteresis-based stream activity and drop tracking.
    *   `SignalQualityScorer`: MOS-like (Mean Opinion Score) signal quality evaluator factoring saturation, drift, and dropouts.
*   **Session State Machine & Watchdogs**: Implemented `Session` lifecycle (`STARTING` -> `ACTIVE` -> `DEGRADED` -> `CLOSED`) with liveness watchdogs monitoring hardware stream health, and `SessionWorker` coordinating multi-session capacity.
*   **Prometheus Telemetry & Observability**: Created `subvocal/telemetry/` package supporting a `TelemetryService` interface, `NullTelemetry` defaults, and a `PrometheusTelemetry` exporter reporting active sessions, intent accuracy, physiological quality, and error rates. Included a ready-to-use Grafana dashboard configuration (`grafana_dashboard.json`).
*   **HMAC-Signed Auth Grants**: Implemented capability-scoped authorization in `subvocal/auth/` using HMAC-SHA256 signed JSON claims tokens (`ActionGrants`) containing permitted command scopes and dry-run enforcements. Added context propagation helpers (`set_context_grants`) and a `GrantsPolicy` security provider for the `PolicyEngine`.
*   **Routing & Node Selection**: Added Node selectors (`CPULoadSelector`, `SessionCountSelector`) managing session worker distribution under load.
*   **Persistent State Storage**: Added SQLite session store (`SQLiteSessionStore`) persisting active states and session configurations to disk with configuration scrubbing.
*   **Biometric Data Channel**: Added TCP socket server (`BiometricDataChannelServer`) and client broadcasting live sEMG metrics, signal levels, and classifications to visualization dashboards.
*   **Ingress/Egress Orchestration**: Added Ingress manager supporting sensor registration and automated failovers; egress manager coordinating speech synthesizer queues and trace database logs.
*   **Zero-Dependency BrainFlow Compatibility Layer**:
    *   `subvocal/hardware/brainflow_compat.py`: Implemented a pure-Python fallback for `BoardShim`, `BoardIds`, and `BrainFlowInputParams`. Automatically delegates to the official C++ `brainflow` library if installed; otherwise, runs natively. Supports simulated signal generator threads (`SYNTHETIC_BOARD`) and a direct USB dongle serial packet parser (`CYTON_BOARD`) to enable edge node acquisition.
    *   `subvocal/emg_core/dsp/brainflow_filter.py`: Re-implemented the `DataFilter` signal processing suite in pure Python utilizing NumPy and SciPy. Includes Butterworth, Chebyshev, and Bessel causal/zero-phase filters, environmental notch filtering, moving averages, running medians, downsampling, windowing, Welch PSD estimation, and bandpower integration.

---

## [1.0.0rc1] - 2026-06-09
### Added
*   **PyPI Packaging**: The SDK is now a proper installable package (`pip install subvocal`) with a src-layout (`src/subvocal/`), hatchling build backend, single-source version (`subvocal.__version__`), PEP 561 `py.typed` marker, and optional extras `[ml]`, `[hardware]`, `[tts]`, `[export]`, `[all]`, `[dev]`.
*   **`subvocal-mcp` Console Command**: The MCP stdio server installs as an entry point.
*   **Writable Path Resolution**: `subvocal.paths` resolves per-user data/model directories (overridable via `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR`), replacing package-relative paths that break after installation; `SubvocalPipeline` accepts a `trace_path` parameter.
*   **CI Quality Gates**: GitHub Actions matrix (Python 3.10–3.12) running ruff, pyright, pytest with coverage, license audit, and a packaging job that builds, twine-checks, and smoke-tests the wheel in a clean environment.
*   **Typed Exception Hierarchy**: `subvocal.exceptions` — every SDK error derives from `SubvocalError` while remaining compatible with the builtin exception types previously raised.
*   **Offline Provider & Auto-Selection**: `HeuristicProvider` reconstructs intents fully offline via the articulatory-distance decoder; `resolve_provider()` selects the best provider from the environment.
*   **Provider Resilience**: configurable HTTP timeouts and exponential-backoff retries for transient failures across all LLM providers.
*   **Pipeline Observability**: `PipelineStats` counters, `on_token`/`on_intent`/`on_action`/`on_error` observer callbacks, and opt-in `raise_on_policy_violation`.
*   **Platform Corpus on the Site**: all specification documents and the end-to-end walkthrough render as site pages from `docs/content/` via `tools/build_site.py`.
*   **Documentation Site**: Static GitHub Pages site under `docs/` with a landing page, quickstart/development/MCP guides, and an auto-generated API reference (`tools/build_api_page.py`).
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
