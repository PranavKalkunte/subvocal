# Subvocal SDK: Physiological Silent Speech Interface Middleware

The **Subvocal SDK** is an open-source, hardware-agnostic middleware platform that connects surface electromyography (sEMG) interfaces to LLM-driven AI agents.

Rather than locking developers to a proprietary neckband or a closed whole-word vocabulary, the Subvocal SDK provides the software rails—signal conditioning, deep learning training skeletons, articulatory phonetic shorthand simulators, and context-aware decoders—to enable high-accuracy, low-latency, and open-vocabulary silent speech control.

---

## 🛠️ Installation

```bash
pip install subvocal
```

The base install is lightweight (pydantic + numpy) and covers the pipeline, hardware drivers, shorthand decoding, context, and the MCP server. Optional extras pull in heavier subsystems:

| Extra | Enables | Installs |
|-------|---------|----------|
| `subvocal[ml]` | Classifier training, inference, calibration (`subvocal.emg_core`) | scipy, scikit-learn, joblib, torch |
| `subvocal[hardware]` | Public-dataset drivers (Ninapro, PutEMG, CSL-HDEMG) and live serial boards | scipy, h5py, pyserial |
| `subvocal[metrics]` | Prometheus telemetry exporter and worker CPU-load reporting | prometheus-client, psutil |
| `subvocal[tts]` | Audio feedback outside macOS | pyttsx3 |
| `subvocal[export]` | ONNX model export | onnx |
| `subvocal[all]` | Everything above | — |

Security hardening is enforced in CI and requires no extra install: `pip-audit` for dependency CVEs, `ruff` bandit rules (`S`), validated core models, sanitized model paths (path-traversal protection), and `torch.load(weights_only=True)` for safe deserialization. Research frontier modules (Wave 1: `dsp/handcrafted`, `dsp/spd`, `ml/spd_gru`, `ml/adaptor`, `ml/mona`, `ml/lisa`, `ml/speechnet`, `hardware/datasets GaddyDriver`; Wave 2: `foundation/tinymyo`, `foundation/aemg_tokenizer`, `foundation/spectre`, `adaptation/sal_lbn`, `adaptation/cpep`, `adaptation/variance_transfer`, `hardware/datasets MetaEMGDriver`, `benchmarks/emgbench`) are optional under `subvocal[ml]` (require `torch`) and reuse the same hardening (path sanitization, `weights_only=True`, guarded lazy imports).

## 🚀 Quickstart

A complete pipeline—synthetic sEMG source through intent reconstruction to action execution—runs offline in a few lines:

```python
from subvocal import SubvocalPipeline
from subvocal.core.testing import MockActionExecutor, MockContextProvider, MockLLMProvider
from subvocal.hardware.drivers import SyntheticSignalGenerator
from subvocal.core.models import CommandToken
import time

hardware = SyntheticSignalGenerator(fs=1000.0, num_channels=8)

def classify(frame):
    """Replace with subvocal.emg_core.ml.infer.InferenceEngine for real models."""
    arr = frame.to_numpy()
    if abs(arr).max() > 1.0:  # a command burst is present
        return CommandToken(text="gt", confidence=0.95, timestamp=time.time())
    return None

pipeline = SubvocalPipeline(
    hardware=hardware,
    classify_fn=classify,
    llm_provider=MockLLMProvider(),       # or resolve_provider() / ClaudeProvider() ...
    context_provider=MockContextProvider(),
    executor=MockActionExecutor(),
    phrase_timeout_seconds=0.5,
    on_action=lambda action, status: print("observed:", action.action_type, status),
)
# To inject tokens externally, prefer pipeline.inject_token(token) over direct
# token_buffer append — it is thread-safe (deque + lock) and respects backpressure.

hardware.start()
hardware.trigger_command("gt", duration_ms=120)
for _ in range(30):
    action = pipeline.step(window_ms=50)
    if action:
        print("Executed:", action.action_type, action.params)
        # -> Executed: goto {'arguments': ['google.com'], 'resolved_text': 'GOTO google.com', ...}
        break
    time.sleep(0.05)  # real-time pacing: the phrase ends after 0.5 s of silence
```

Swap in a real LLM provider (`subvocal.core.llm_providers.ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`, `LlamaProvider`), a real driver (`OpenBCICytonDriver`, `DelsysTrignoDriver`, `FileReplayDriver`), and a trained classifier (`subvocal.emg_core.ml.infer.InferenceEngine`) without changing the pipeline code. `subvocal.resolve_provider()` picks the best provider for the environment automatically — a real LLM when an API key is present, the offline `HeuristicProvider` otherwise.

See `examples/silent-typing/demo.py` for a standalone 20-line copy-paste runnable (no hardware or API keys).

### Production behavior

- **Typed errors**: everything the SDK raises derives from `subvocal.SubvocalError` (`HardwareError`, `ProviderError`, `ConfigurationError`, `PolicyViolationError`, ...), each compatible with the builtin exception type it replaces.
- **Resilient providers**: configurable per-request timeouts and exponential-backoff retries for transient failures (connection errors, HTTP 408/429/5xx); non-retryable statuses fail fast.
- **Observability**: `pipeline.stats` exposes running counters (frames, tokens, intents, executed/blocked actions, errors, uptime), and `on_token` / `on_intent` / `on_action` / `on_error` observer callbacks stream pipeline lifecycle events without ever breaking the pipeline. Every phrase is JSONL-traced for audit with 10 MB rotation; opt-out via `telemetry.trace_enabled` / `runtime.trace_enabled` (`SUBVOCAL_TELEMETRY__TRACE_ENABLED=false`).
- **Safety**: pluggable policy engine with dry-run mode; set `raise_on_policy_violation=True` to turn rejections into `PolicyViolationError`.
- **Hardening (2.0.1)**:
  - **Bounded `OpsQueue`** (`maxsize=min_size`, `qsize()`/`is_full()`) with backpressure — no unbounded OOM growth.
  - **Thread-safe `pipeline.inject_token()`** — preferred over direct `token_buffer` mutation; `token_buffer` is now a `collections.deque`.
  - **Deadlock-free `pipeline.step()`** — 5 s timeout raises `HardwareError` instead of hanging.
  - **Validated core models** — `Frame` ordering enforced, `CommandToken.confidence` clamped 0–1.
  - **Sanitized model paths** — `model_path` traversal protection; rejects `..` / absolute escapes.
  - **Secure deserialization** — `torch.load(weights_only=True)` prevents RCE.
  - **HMAC padding fix** — correct `=` padding restores ~25% previously-failing grant verifications.
  - **DSP notch fix** — `notch_freq` (50/60 Hz) now correctly plumbed to filters.
  - **Prometheus low-cardinality** — `session_id` label removed, port range validated and registry bounded to avoid cardinality explosion.

### MCP server

The SDK ships a stdio Model Context Protocol server so Claude Desktop (or any MCP client) can ingest subvocal commands as tools:

```bash
subvocal-mcp
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "subvocal": { "command": "subvocal-mcp" }
  }
}
```

### Configuration

Every subsystem reads from one strict, validated config tree (`subvocal.config.SubvocalConfig`). Load it from a YAML file and/or environment overrides:

```python
from subvocal.config import load_config

config = load_config("subvocal.yaml")          # optional path; all keys have defaults
print(config.hardware.sample_rate, config.dsp.bandpass_high)
```

Unknown keys are rejected (`extra="forbid"`), so typos fail loudly instead of being silently ignored. Override any nested field with a `SUBVOCAL_<SECTION>__<KEY>` environment variable (double underscore separates section from key):

```bash
export SUBVOCAL_HARDWARE__SAMPLE_RATE=500
export SUBVOCAL_TELEMETRY__ENABLED=true
```

The flat `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR` variables are reserved for writable-path resolution (`subvocal.paths`) and are not config keys. See [`subvocal-sample.yaml`](subvocal-sample.yaml) for every option with its default.

### Sessions, monitoring, and telemetry (v2)

For long-running deployments, `subvocal.runtime.Session` wraps the pipeline with a lifecycle state machine (`STARTING → ACTIVE → DEGRADED → CLOSED`), a liveness watchdog with fixed lifecycle (no rollback from `CLOSED`), real-time signal-quality scoring (`subvocal.stream`), and an async work queue. `token_buffer` is now a `collections.deque` drained via thread-safe `pipeline.inject_token()`. `subvocal.runtime.SessionWorker` manages a pool of sessions with load reporting, and `subvocal.telemetry.PrometheusTelemetry` (install `subvocal[metrics]`) exports session, intent, action, and signal-quality metrics with low-cardinality labels (no `session_id`) and bounded port registry — with a ready-to-import Grafana dashboard at [`src/subvocal/telemetry/grafana_dashboard.json`](src/subvocal/telemetry/grafana_dashboard.json). `BoardShim` streaming buffers are bounded to prevent leaks, and JSONL traces rotate at 10 MB and can be disabled via `trace_enabled`.

### BrainFlow-compatible API

`subvocal.hardware.brainflow_compat` and `subvocal.emg_core.dsp.brainflow_filter` provide a pure-Python, zero-native-dependency drop-in for BrainFlow's `BoardShim` and `DataFilter`. Code written against BrainFlow runs unchanged; if the official `brainflow` package is installed it is used transparently as the backend.

```python
from subvocal.hardware.brainflow_compat import BoardShim, BoardIds, BrainFlowInputParams

board = BoardShim(BoardIds.SYNTHETIC_BOARD, BrainFlowInputParams())
board.prepare_session(); board.start_stream()
data = board.get_board_data()   # (channels x samples) ndarray
board.stop_stream(); board.release_session()
```

---

## 📂 Repository Structure

```
subvocal/
├── src/subvocal/           # The installable package
│   ├── core/               # Data models, interfaces, pipeline, security policies, LLM providers
│   ├── hardware/           # HAL drivers (file replay, synthetic, OpenBCI, Delsys) + dataset loaders (Ninapro/PutEMG/CSL-HDEMG/Gaddy/MetaEMG)
│   ├── emg_core/           # sEMG stack: dsp (filters/handcrafted 112/spd), ml (adaptor/spd_gru/mona/lisa/speechnet), foundation (tinymyo/aemg_tokenizer/spectre), adaptation (sal_lbn/cpep/variance_transfer), benchmarks (emgbench)
│   ├── shorthand/          # Phonetic shorthand vocabulary, simulator, hybrid decoder
│   ├── context/            # User context schemas and phonetic context matching
│   ├── mcp/                # Model Context Protocol stdio server
│   └── tts/                # Multi-backend TTS feedback engine
├── tests/                  # Pytest suite (incl. foundation/adaptation/emgbench)
├── benchmarks/             # 50-case intent-reconstruction eval harnesses + EMGBench LOSO harness
├── tools/                  # Site/API-page builders, license audit, benchmark runner
└── docs/                   # GitHub Pages site (landing, docs, platform corpus, API reference)
    └── content/            # Markdown sources for the platform corpus and walkthrough
```

---

## 🚀 Core Features

1. **Articulatory Shorthand Decoder**: Overcomes the whole-word sEMG vocabulary ceiling. Decodes compressed phonetic consonant shorthand inputs (e.g. `g gl` -> `Google`) under heavy muscle-movement noise.
2. **Asymmetric Levenshtein Distance**: A dynamic programming string alignment cost matrix configured with physiological sEMG confusion clusters (Glottal, Labial, Alveolar, Velar, Rhotic) to discount vowel/consonant omissions in silent speech.
3. **Command-Aware Context Prioritization**: Dynamic target matching against active user contacts (`TYPE`), calendar events (`SEARCH`), browser URLs (`GOTO`), and active application screen elements (`CLICK`).
4. **Physiological Signal Conditioning**: Preprocessing filter configurations defaulting to AlterEgo's `1.3–50.0 Hz` bandpass filter (designed for low-velocity articulatory gestures) with configuration support for standard `20.0–450.0 Hz` EMG; notch 50/60 Hz correctly plumbed.
5. **Handcrafted 112-D sEMG Features** (`emg_core/dsp/handcrafted`): Per-channel 28 (temporal 11 MAV/RMS/VAR/WL/ZC/SSC/WAMP/IEMG/SSI/DASDV/LOGVAR + stats 7 mean/std/min/max/ptp/skew/kurt + spectral 10 MNF/MDF/centroid/bandpower/peak/entropy/spread/rolloff) → 112 for 4-ch; FFT/Welch, numpy-only. Mohapatra ACL 2025; Jou 2006; Gaddy & Klein 2020.
6. **SPD Manifold & Riemannian Features** (`emg_core/dsp/spd`): Sample covariance + `eps·I` (1e-6), affine-invariant `logm` via `eigh` in sparse spectral domain, upper-tri flatten (C=4→K=10), time-varying 50 ms/20 ms windows. Gowda & Miller ACL 2026 Findings; J Neural Eng 2024.
7. **SPD-GRU CTC Decoder** (`emg_core/ml/spd_gru`): `SPD (B,T,C,C) → logm → Linear(K→hidden) → 3-layer GRU (hidden 64, dropout 0.2) → Linear → CTC logits (B,T,V)`, batch `torch.linalg.eigh` inside forward for gradient flow. Gowda ACL 2025/2026.
8. **EMG Adaptor 112→768→3072 + Provider** (`emg_core/ml/adaptor`): 2-layer MLP `Linear→ReLU→Dropout→Linear` mapping handcrafted 112 (or 768 speech-encoder) to frozen Llama-3.2-3B input (3072); `EMGAdaptorProvider.encode_emg_features()` handles raw `(T,4)`/`(B,T,4)` or precomputed. Mohapatra ACL 2025.
9. **MONA Cross-Modal Losses + DTW** (`emg_core/ml/mona`): crossCon bidirectional InfoNCE EMG↔audio, supTcon SupCon, latent DTW warping (numpy/torch, batched), `mona_loss` + `BatchSampler` undersampling LibriSpeech to 50% per epoch (MONA Algo 1). Benster et al. arXiv:2403.05583.
10. **LISA LLM Rescoring** (`emg_core/ml/lisa`): `final = (1-α)·acoustic + α·llm + β·length`, `LISA.rescore`/`lisa_rescore`/`beam_search_rescore` via `LLMProvider`, min-max normalization for scale mismatch. Same MONA arXiv:2403.05583.
11. **SpeechNet Tiny CNN 15k** (`emg_core/ml/speechnet`): 3× depthwise-separable Conv1d blocks (groups → pointwise, BN/ReLU/MaxPool2/Dropout) + AdaptiveAvgPool + Linear, ~15k params, 63.9 µJ on GAP9, inter-session finetune <10 min head-only. Spacone SilentWear 2026 (ETH Zurich).
12. **Gaddy Silent Speech Driver** (`hardware/datasets GaddyDriver`): Zenodo 4064409, 8-ch facial EMG @1000 Hz (ACL 2021 resampled 800 Hz) + audio + `info.json` transcripts, recursive `*_emg.npy` index, `silent`/`vocalized` split filter, `allow_pickle=False`, path-traversal sanitized.
13. **TinyMyo 3.6M Transformer** (`foundation/tinymyo`): Channel-independent patching (`patch_size` 10), shared linear proj, SimMIM 50% masking via learned `mask_token`, 8 bidirectional Transformer blocks with RoPE and pre-norm, linear decoder `embed_dim→patch_size`; RoPE enables length extrapolation. arXiv:2512.15729.
14. **AEMG NCT + VQ Vocabulary** (`foundation/aemg_tokenizer`): Sliding-window 32/16 contraction primitives (~30–60 ms) → `token_dim` 64 via interpolation, codebook K=512 (0.1σ init), numpy `cdist` or `torch.cdist` VQ, overlap-add decode, BERT-style masked modeling (15% masking, Transformer LM). Huang et al. CVPR 2026.
15. **SPECTRE Spectral + CyRoPE** (`foundation/spectre`): Per-channel STFT magnitude (scipy with numpy Hann fallback) → per-frame concat → K-means pseudo-labels, CyRoPE factorized temporal (linear) + spatial annular (8×16 grid, `2π·c/num_channels`) rotary, depthwise CNN front-end + Transformer + masked spectral head `D→n_clusters`. arXiv:2512.22481.
16. **SAL/LBN 7-Param Spatial Adaptation** (`adaptation/sal_lbn`): SAL prependable warp — 2-D affine `2×3` (6 DoF) or 2-DoF translation via `affine_grid`+`grid_sample` on 8×16 HDEMG grid plus per-channel `x·scale+shift` (2·C); LBN per-channel bias `x-bias`; `adapt_sal_lbn()` freezes backbone, Adam on SAL/LBN only (3–5 epochs, <2 min calib). Pereira et al. arXiv:2409.08058.
17. **CPEP Pose-EMG Contrastive** (`adaptation/cpep`): Dual 4-layer Transformers (EMG 8-ch/pose 63-D, d_model 256, 8 heads) → 1-layer 256-d projection + L2 norm, symmetric InfoNCE `τ=0.02` learnable, pose encoder frozen, zero-shot `k=10` kNN cosine. Cui et al. arXiv:2509.04699.
18. **Variance Transfer Bayesian GCM** (`adaptation/variance_transfer`): Normal-Wishart shared precision (means subject-specific), priors `m0=0, β0=1, ν0=D+1, W0=I`, `w_s`-scaled source posterior, 1-trial target calibration with diagonally-regularized update, QDA `log p(x|k)` with `0.5·log|Λ|`. Yoneda & Furui EMBC 2024 / arXiv:2505.15381.
19. **MetaEMGDriver + EMGBench 9-Dataset LOSO** (`hardware/datasets MetaEMGDriver` + `emg_core/benchmarks/emgbench`): Meta sEMG-RD 16-ch×2000 Hz wristband + pose 63-D (HDF5 `/emg`/`/pose` or `*_emg.npy`+`*_pose.npy`), split-filtered; EMGBench harness evaluates LOSO-CV and few-shot `n_shot` adaptation across 9 datasets (Ninapro DB2/DB3/DB5, CapgMyo DB-b, Myo Armband, UCI EMG, MCS, Hyser, FlexWear-HD) with `evaluate_loso`/`evaluate_adaptation`/`evaluate_all`. Yang et al. NeurIPS 2024 D&B (arXiv:2410.23625).
20. **Hardened V2 Runtime & Infra**: Bounded `OpsQueue` (`maxsize=min_size`, `qsize`/`is_full`), thread-safe `deque` + `inject_token()`, 5 s `pipeline.step()` timeout → `HardwareError`, validated `Frame`/`CommandToken`, sanitized `model_path` + `torch.load(weights_only=True)`, HMAC `=` padding fix, Prometheus low-cardinality (no `session_id`, bounded port/registry) + `trace_enabled` opt-out (10 MB rotation), `BoardShim` bounded buffers, TTS flag-injection sanitization; MCP stdio server, SQLite session store, TCP biometric channel, ingress/egress failover, `CPULoadSelector`/`SessionCountSelector` routing, pure-Python BrainFlow `BoardShim`/`DataFilter` fallback.

---

## Research Frontier (v2.1) — SOTA Expansion

Wave 2 foundation/adaptation modules complement Wave 1 (handcrafted/SPD/adaptor/MONA/LISA/SpeechNet/Gaddy) and live under `subvocal[ml]` (`torch` guarded, citation in every docstring):

- **TinyMyo** (`foundation/tinymyo`): 3.6M Transformer encoder, channel-independent patching + SimMIM 50% masking, RoPE for variable-length extrapolation — *TinyMyo arXiv:2512.15729*.
- **AEMG NCT + VQ** (`foundation/aemg_tokenizer`): Neuromuscular Contraction Tokenizer sliding-window primitives + codebook K=512 VQ + overlap-add + masked collective token prediction — *Huang et al. CVPR 2026*.
- **SPECTRE** (`foundation/spectre`): STFT per-channel magnitude → K-means pseudo-labels + Cylindrical RoPE (temporal linear + annular 8×16 forearm grid) + depthwise CNN front-end — *SPECTRE arXiv:2512.22481*.
- **SAL/LBN** (`adaptation/sal_lbn`): 7-param spatial adaptation layer (2-D affine warp via `affine_grid`/`grid_sample` + per-channel scale/shift) + learnable baseline norm `x-bias`, <2 min calibration — *Pereira et al. arXiv:2409.08058*.
- **CPEP** (`adaptation/cpep`): Dual 4-layer Transformers (EMG+pose) to shared 256-d L2 space, symmetric InfoNCE τ=0.02 (pose frozen), zero-shot kNN `k=10` — *Cui et al. arXiv:2509.04699*.
- **Variance Transfer Bayesian GCM** (`adaptation/variance_transfer`): Normal-Wishart shared precision, `w_s`-scaled source posterior → target per-class means + shared covariance, 1-trial regularization — *Yoneda & Furui EMBC 2024 / arXiv:2505.15381*.
- **MetaEMGDriver + EMGBench** (`hardware/datasets MetaEMGDriver` + `emg_core/benchmarks/emgbench`): LOSO-CV and few-shot adaptation across 9 datasets (Ninapro DB2/DB3/DB5, CapgMyo DB-b, Myo, UCI, MCS, Hyser, FlexWear-HD) — *Yang et al. NeurIPS 2024 D&B arXiv:2410.23625*.

Wave 1 recap (v2.0): handcrafted 112 (`dsp/handcrafted`), SPD matrices + SPD-GRU CTC (`dsp/spd` + `ml/spd_gru`), EMG Adaptor 112→768→3072 + provider (`ml/adaptor`), MONA crossCon/supTcon + DTW + LISA rescoring (`ml/mona` + `ml/lisa`), SpeechNet 15k CNN 63.9 µJ (`ml/speechnet`), GaddyDriver (`hardware/datasets` Zenodo 4064409). All research code is optional (`subvocal[ml]`) and documented with paper citations in docstrings.

---

## 🧪 Development

```bash
git clone https://github.com/PranavKalkunte/subvocal.git
cd subvocal
pip install -e ".[all,dev]"

pytest --cov=subvocal --cov-report=term-missing --cov-fail-under=65  # test suite (65% floor, covers foundation/tinymyo/aemg_tokenizer/spectre, adaptation/sal_lbn/cpep/variance_transfer, benchmarks/emgbench, dsp/handcrafted/spd, ml/*)
ruff check src tests benchmarks tools   # lint (E,F,I,UP,B; E501/E741 ignored)
pyright                                # type check (standard mode, 0 errors required)
pip-audit                              # dependency CVE audit
python benchmarks/eval_runner.py       # 50-case heuristic benchmark (74% @0.36ms)
python -m subvocal.emg_core.benchmarks.emgbench  # EMGBench LOSO harness sanity (synthetic fallback when DatasetsProcessed_hdf5 absent)
```

Research modules are under `subvocal[ml]` (lazy `torch` guard → `MissingDependencyError` with `pip install "subvocal[ml]"` hint) and are exercised by `tests/emg_core/test_foundation.py`, `test_adaptation.py`, `test_spd.py`, `test_mona_lisa.py`, etc. New research code must carry its paper citation in the module docstring (see `CONTRIBUTING.md`).

Runtime artifacts (traces, trained models) are written to the per-user data directory; override with `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR`.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and quality gates, and [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## 📄 License
This repository is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.
