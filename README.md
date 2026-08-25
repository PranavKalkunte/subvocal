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

Security hardening is enforced in CI and requires no extra install: `pip-audit` for dependency CVEs, `ruff` bandit rules (`S`), validated core models, sanitized model paths (path-traversal protection), and `torch.load(weights_only=True)` for safe deserialization.

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
│   ├── hardware/           # HAL drivers (file replay, synthetic, OpenBCI, Delsys) + dataset loaders
│   ├── emg_core/           # DSP filters, TD10 features, classifiers (RF/CNN/GRU/Transformer)
│   ├── shorthand/          # Phonetic shorthand vocabulary, simulator, hybrid decoder
│   ├── context/            # User context schemas and phonetic context matching
│   ├── mcp/                # Model Context Protocol stdio server
│   └── tts/                # Multi-backend TTS feedback engine
├── tests/                  # Pytest suite
├── benchmarks/             # 50-case intent-reconstruction eval harnesses
├── tools/                  # Site/API-page builders, license audit, benchmark runner
└── docs/                   # GitHub Pages site (landing, docs, platform corpus, API reference)
    └── content/            # Markdown sources for the platform corpus and walkthrough
```

---

## 🚀 Core Features

1. **Articulatory Shorthand Decoder**: Overcomes the whole-word sEMG vocabulary ceiling. Decodes compressed phonetic consonant shorthand inputs (e.g. `g gl` -> `Google`) under heavy muscle-movement noise.
2. **Asymmetric Levenshtein Distance**: A dynamic programming string alignment cost matrix configured with physiological sEMG confusion clusters (Glottal, Labial, Alveolar, Velar, Rhotic) to discount vowel/consonant omissions in silent speech.
3. **Command-Aware Context Prioritization**: Dynamic target matching against active user contacts (`TYPE`), calendar events (`SEARCH`), browser URLs (`GOTO`), and active application screen elements (`CLICK`).
4. **Physiological Signal Conditioning**: Preprocessing filter configurations defaulting to AlterEgo's `1.3–50.0 Hz` bandpass filter (designed for low-velocity articulatory gestures) with configuration support for standard `20.0–450.0 hz` EMG.
5. **Classifiers (RF + Deep Learning)**: Custom pipelines to train scikit-learn **Random Forest**, PyTorch **1D CNN**, **GRU**, and **Transformer** architectures on raw multi-channel sEMG traces.
6. **Asynchronous Execution (V2 Architecture)**: Low-latency, thread-safe orchestration on LiveKit's `OpsQueue`/`IncrementalDispatcher`; `OpsQueue` is now bounded (`maxsize=min_size`) with backpressure and `qsize()`/`is_full()`, `pipeline.step()` has deadlock protection (5 s timeout → `HardwareError`), and `pipeline.inject_token()` provides thread-safe deque-based injection.
7. **Physiological Signal Monitoring**: Real-time EMA-smoothed signal level activity detection and MOS-like connection quality scoring (saturation, drift, dropouts); validated `Frame` ordering and `CommandToken` confidence 0–1.
8. **Prometheus Telemetry**: Integrated exporter and Grafana dashboard with low-cardinality metrics (no `session_id` label), bounded port registry, and `trace_enabled` opt-out plus 10 MB JSONL rotation.
9. **HMAC-Signed Capability Grants**: Capability-scoped `ActionGrants` (HMAC-SHA256) with corrected `=` padding (fixes ~25% verification failures), verified via `GrantsPolicy`.
10. **MCP Integration**: Zero-dependency stdio JSON-RPC server exposing pipeline status, thread-safe token injection, phrase processing, and calibration as MCP tools.
11. **Persistent Session Storage**: SQLite and in-memory backends to serialize and reload session configurations, states, and active metrics.
12. **Real-Time TCP Biometric Streaming**: Dedicated TCP socket server broadcasting live signal attributes (quality, levels, tokens) to visualization dashboards; `BoardShim` buffers are now bounded to prevent leaks.
13. **Ingress/Egress Orchestration**: Ingress failover for primary sensors/simulation streams; egress dispatcher for audio TTS and trace logs with sanitized model paths and `torch.load(weights_only=True)` deserialization.
14. **Intelligent Node Routing**: Load-balanced session assignment via CPU or session-count selectors; routing status filter now correctly excludes non-ACTIVE sessions.
15. **Zero-Dependency BrainFlow & DSP**: Pure-Python fallback emulating `SYNTHETIC_BOARD`/`CYTON_BOARD` and `DataFilter` (filtering, windowing, Welch PSD, bandpower) with correctly-plumbed notch `50/60 Hz` and sanitized file I/O.

---

## 🧪 Development

```bash
git clone https://github.com/PranavKalkunte/subvocal.git
cd subvocal
pip install -e ".[all,dev]"

pytest --cov=subvocal --cov-report=term-missing --cov-fail-under=65  # test suite (65% floor)
ruff check src tests benchmarks tools   # lint (E,F,I,UP,B; E501/E741 ignored)
pyright                                # type check (standard mode, 0 errors required)
pip-audit                              # dependency CVE audit
python benchmarks/eval_runner.py       # 50-case heuristic benchmark (74% @0.36ms)
```

Runtime artifacts (traces, trained models) are written to the per-user data directory; override with `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR`.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and quality gates, and [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## 📄 License
This repository is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.
