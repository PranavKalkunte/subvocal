# Changelog

All notable changes to the Subvocal Middleware Platform will be documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2026-08-25 (Research SOTA Expansion)

> **Note:** `2.0.1` was security & reliability hardening (C1–C8/H1–H10); `2.1.0` is the research expansion — Wave 1 (already in code, now documented) + Wave 2 (just added). All new modules are optional under `subvocal[ml]` (lazy `torch` guard), reuse path-traversal/`weights_only=True` hardening, and carry paper citations in docstrings.

### Added

#### DSP (Wave 1)
*   **Handcrafted 112-D features** (`emg_core/dsp/handcrafted`): `extract_handcrafted_features` (112 = 4×28: temporal 11 + stats 7 + spectral 10 via FFT/Welch) and `extract_handcrafted_timevarying` (50 ms/20 ms sliding) with numpy/scipy fallback. References Mohapatra et al. ACL 2025; Jou et al. 2006; Gaddy & Klein 2020.
*   **SPD manifold** (`emg_core/dsp/spd`): `compute_covariance_matrix`, `compute_spd_matrix` (+`eps·I`), `compute_spd_timevarying`, `spd_logm` via `scipy.linalg.eigh`, `spd_flatten_upper` (C=4→K=10), `spd_riemannian_features` + aliases. References Gowda & Miller ACL 2026 Findings; J Neural Eng 2024; Gowda 2025/2026 SPD-GRU.

#### ML (Wave 1)
*   **SPD-GRU CTC** (`emg_core/ml/spd_gru`): `SPDGRU` (`(B,T,C,C)→eigh logm→upper→Linear→3-layer GRU→FC`) with CTC helpers `ctc_loss`/`greedy_decode`/`train_spd_gru`, batched `torch.linalg.eigh` inside forward. Gowda ACL 2025 & Findings 2026.
*   **EMG Adaptor 112→768→3072** (`emg_core/ml/adaptor`): `EMGAdaptor` 2-layer MLP + `EMGAdaptorProvider` (`encode_emg_features`/`encode`/`adapt`, `LLMProvider` wrapper) handling raw `(T,4)`/`(B,T,4)` or precomputed vectors; guarded `MissingDependencyError`. Mohapatra et al. ACL 2025.
*   **MONA losses + DTW + BatchSampler** (`emg_core/ml/mona`): `cross_contrastive_loss` (bidirectional InfoNCE), `supervised_temporal_contrastive_loss`, `dtw_align` (numpy/torch, batched), `mona_loss` combined, `BatchSampler` undersampling LibriSpeech 50% per epoch (MONA Algo 1). Benster et al. arXiv:2403.05583.
*   **LISA rescoring** (`emg_core/ml/lisa`): `LISA` (`alpha`/`beta`, `rescore`/`beam_search_rescore`, numpy fallback) + `lisa_rescore` functional (`(1-α)·acoustic+α·llm+β·length`), LLMProvider scoring with heuristic fallback. Same arXiv:2403.05583.
*   **SpeechNet 15k CNN** (`emg_core/ml/speechnet`): `SpeechNet` depthwise-separable 3-block CNN + `AdaptiveAvgPool→Linear`, `count_parameters` (~15k), `estimate_energy` 63.9 µJ / `estimate_latency` on GAP9, `finetune_inter_session` head-only (<10 min). Spacone et al. SilentWear 2026 (ETH Zurich, GAP9).

#### Hardware / Datasets (Wave 1)
*   **GaddyDriver** (`hardware/datasets GaddyDriver`): Zenodo 4064409, 8-ch facial EMG @1000 Hz (ACL 2021 800 Hz), `*_emg.npy` + `*_info.json` + `*_audio.flac` index, `silent`/`vocalized` split, `allow_pickle=False`, traversal sanitization, `list_utterances`/`get_transcript`.

#### Foundation (Wave 2 — SOTA)
*   **TinyMyo 3.6M Transformer** (`emg_core/foundation/tinymyo`): `TinyMyoEncoder` channel-independent patching (`patch_size` 10→`embed_dim` 128), SimMIM 50% masking via `mask_token`, 8 bidirectional Transformer blocks with RoPE (`_rope_cos_sin`/`_apply_rope`, pre-norm, SwiGLU-free), linear decoder; `TinyMyoFoundation` task heads + `pretrain_step`/`finetune_step`. *TinyMyo arXiv:2512.15729*.
*   **AEMG NCT + VQ vocabulary** (`emg_core/foundation/aemg_tokenizer`): `EMGTokenizer` sliding-window 32/16 contraction primitives → `token_dim` 64, codebook K=512, `torch.cdist` (fallback numpy chunked), overlap-add `decode`/`decode_with_channels`, `AEMGFramework` masked modeling (mask_ratio 0.15, BERT 80/10/10 simplified). *Huang et al. CVPR 2026*.
*   **SPECTRE STFT K-means + CyRoPE** (`emg_core/foundation/spectre`): `stft_kmeans_pseudolabels` (scipy `stft` → magnitude concat → sklearn/numpy K-means), `CyRoPE` factorized temporal+annular (8×16 forearm grid, `2π·c/num_channels`) rotary, `SPECTREEncoder` (depthwise Conv1d front-end + CyRoPE + Transformer + `D→n_clusters` head, `ssl_loss`). *SPECTRE arXiv:2512.22481*.

#### Adaptation (Wave 2 — SOTA)
*   **SAL/LBN 7-param spatial adaptation** (`emg_core/adaptation/sal_lbn`): `SAL` 2-D affine `2×3` (6 DoF) or 2-DoF translation via `affine_grid`/`grid_sample` + per-channel `scale/shift` (2·C); `LBN` per-channel `x-bias`; `SAL_LBN` + `adapt_sal_lbn()` (freeze backbone, Adam on SAL/LBN only, 3–5 epochs). NumPy fallback when `torch` absent. *Pereira et al. arXiv:2409.08058*.
*   **CPEP pose-EMG contrastive** (`emg_core/adaptation/cpep`): `EMGEncoder`/`PoseEncoder` (4-layer Transformers d_model 256, 8 heads, 512 FFN, CLS token, 1-layer 256-d projection + L2), `pose_emg_contrastive_loss` symmetric InfoNCE `τ=0.02` learnable, `CPEPFramework` (pose frozen), `knn_classify` (sklearn or numpy majority vote, `k=10`). *Cui et al. arXiv:2509.04699*.
*   **Variance Transfer Bayesian GCM** (`emg_core/adaptation/variance_transfer`): `GaussianClassificationModel` QDA `log p(x|k) = -0.5·mahal +0.5·log|Λ|+const`, `pretrain_variance_transfer` Normal-Wishart posterior (`m0=0, β0=1, ν0=D+1, W0=I`), `transfer_to_target` `w_s`-scaled source posterior + regularized 1-trial update, `VarianceTransferGCM` wrapper. *Yoneda & Furui EMBC 2024 / arXiv:2505.15381*.

#### Hardware / Benchmarks (Wave 2)
*   **MetaEMGDriver** (`hardware/datasets MetaEMGDriver`): Meta sEMG-RD 16×2000 Hz wristband + hand pose 63-D (HDF5 `/emg`/`/pose` + `*_emg.npy`/`*_pose.npy`), `train`/`val`/`test` split sanitization, download note to `ai.meta.com/blog` + `github.com/facebookresearch/emg2pose`, traversal-safe, `read_frame_with_pose`/`load_pair`. Salter et al. NeurIPS 2024 `arXiv:2412.02725`; Sivakumar & Landau Nature 2025.
*   **EMGBench LOSO harness** (`emg_core/benchmarks/emgbench` + `benchmarks/`): `EMGBench` with 9 datasets (`DEFAULT_DATASETS`: Ninapro DB2/DB3/DB5, CapgMyo DB-b, Myo Armband, UCI EMG, MCS, Hyser, FlexWear-HD), `evaluate_loso`/`evaluate_adaptation` (`n_shot`)/`evaluate_all` + `summarize_results`, synthetic deterministic fallback when `DatasetsProcessed_hdf5` absent. *Yang et al. NeurIPS 2024 D&B Track arXiv:2410.23625*.

### Changed
*   `pyproject.toml` / README Development: new research modules documented as optional `subvocal[ml]`; CI still base-install (`pydantic+numpy`) + `torch` only via `[ml]`.
*   Hardening reuse: all new loaders (`*_emg.npy`, HDF5 via `h5py`) use `allow_pickle=False` / context managers, traversal checks (`is_relative_to` + `relative_to` fallback + regex sanitization), `pip-audit` retains green.

---

## [2.0.1] - 2026-08-25 (Security & Reliability Hardening)
### Added
*   **Trace opt-out**: `telemetry.trace_enabled` and `runtime.trace_enabled` (env `SUBVOCAL_TELEMETRY__TRACE_ENABLED` / `SUBVOCAL_RUNTIME__TRACE_ENABLED`) to disable JSONL PII tracing; `false` skips file creation entirely.
*   **Validated core models**: `Frame` ordering invariant and `CommandToken.confidence` clamped to 0–1 via Pydantic validators.
*   **Bounded `OpsQueue` introspection**: `qsize()` and `is_full()` helpers; queue constructed with `maxsize=min_size` to enforce backpressure.
*   **Trace rotation cap**: JSONL trace files rotate at 10 MB to bound disk use.

### Fixed
*   **C1 — Secure deserialization (RCE)**: `torch.load(..., weights_only=True)` in `subvocal/emg_core/ml/` prevents arbitrary code execution on untrusted checkpoints.
*   **C2 — Path traversal sanitization**: `model_path` / file I/O helpers reject `..` and absolute-path escapes outside `get_models_dir()` / `get_data_dir()`.
*   **C3 — HMAC padding**: correct `=` padding in `subvocal/auth/` grant verification fixes ~25% spurious failures.
*   **C4 — Deadlock-free `pipeline.step()`**: 5 s queue-get timeout raises `HardwareError` instead of hanging forever.
*   **C5 — Race on `token_buffer`/`stats`**: `token_buffer` migrated to `collections.deque` with `pipeline.inject_token()` under lock; `stats` increments protected.
*   **C6 — `BoardShim` unbounded buffer leak**: streaming buffer now bounded (`maxsize`-capped deque) to prevent OOM on long sessions.
*   **C7 — TTS `--` injection**: `tts` shell invocation sanitizes arguments to block flag injection.
*   **C8 — Watchdog rollback**: session watchdog no longer rolls back `CLOSED` → `DEGRADED`; lifecycle is monotonic.
*   **H1 — Env coercion (`int` vs `bool`)**: `SUBVOCAL_*` env overrides now coerce correctly (bool fields no longer accept `0`/`1` as int).
*   **H2 — Notch frequency ignored**: `dsp.notch_freq` (50/60 Hz) correctly plumbed through `BrainFlow`/`DSP` filter chain.
*   **H5 — `h5` file leak**: `h5py.File` handles in dataset loaders now use context managers to guarantee close.
*   **H9 — Prometheus cardinality**: removed high-cardinality `session_id` label and bounded collector registry / `prometheus_port` range.
*   **H10 — Routing status filter**: selectors now correctly filter to `ACTIVE` sessions only.

### Changed
*   **Ruff**: enabled `S` (bandit) rules; `E501`/`E741` remain ignored by policy.
*   **Pyright**: `basic` → `standard` mode; CI requires 0 errors.
*   **Coverage floor**: `65` → `75` (`--cov-fail-under=75`).
*   **CI hardening**: `pip-audit` added to GitHub Actions quality gates.
*   **Buffer type**: `SubvocalPipeline.token_buffer` is now `collections.deque`; external mutation should use `pipeline.inject_token()`.

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

### Fixed
*   **Critical config/env collision**: `merge_env_overrides()` consumed the reserved flat variables `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR` (used by `subvocal.paths`) as unknown config keys, crashing `load_config()` under `extra="forbid"` for any process — including CI — that set them. Only `SUBVOCAL_<SECTION>__<KEY>` variables are now treated as overrides.
*   **Routing coherence**: `SessionWorker` now exposes `id` and `cpu_usage`, satisfying the `WorkerNode` protocol so real workers (not just test doubles) can be ranked by `CPULoadSelector` / `SessionCountSelector`.
*   **Type-checker cleanliness**: resolved all pyright/Pyrefly errors (SVC kernel typing, pyttsx3 rate coercion) and silenced optional native-backend import warnings; the tree is now `0 errors / 0 warnings` under a venv-bound type check.

### Changed
*   `[hardware]` extra now includes `pyserial`; new `[metrics]` extra bundles `prometheus-client` and `psutil`. Added `types-pyyaml` / `types-psutil` to `[dev]`.

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
