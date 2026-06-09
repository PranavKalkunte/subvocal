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
| `subvocal[hardware]` | Public-dataset drivers (Ninapro, PutEMG, CSL-HDEMG) | scipy, h5py |
| `subvocal[tts]` | Audio feedback outside macOS | pyttsx3 |
| `subvocal[export]` | ONNX model export | onnx |
| `subvocal[all]` | Everything above | — |

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
    llm_provider=MockLLMProvider(),       # or ClaudeProvider(), OpenAIProvider(), ...
    context_provider=MockContextProvider(),
    executor=MockActionExecutor(),
    phrase_timeout_seconds=0.5,
)

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

Swap in a real LLM provider (`subvocal.core.llm_providers.ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`, `LlamaProvider`), a real driver (`OpenBCICytonDriver`, `DelsysTrignoDriver`, `FileReplayDriver`), and a trained classifier (`subvocal.emg_core.ml.infer.InferenceEngine`) without changing the pipeline code.

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
6. **Pluggable Safety Policies**: `ConfidenceThresholdPolicy`, `CommandWhitelistPolicy`, and `ContextBoundPolicy` gate every action, with dry-run mode and JSONL audit tracing built into the pipeline.
7. **MCP Integration**: A zero-dependency stdio JSON-RPC server exposing pipeline status, token injection, phrase processing, and calibration as MCP tools.

---

## 🧪 Development

```bash
git clone https://github.com/PranavKalkunte/subvocal.git
cd subvocal
pip install -e ".[all,dev]"

pytest                      # test suite
ruff check src tests       # lint
pyright                     # type check
python benchmarks/eval_runner.py   # 50-case heuristic benchmark
```

Runtime artifacts (traces, trained models) are written to the per-user data directory; override with `SUBVOCAL_DATA_DIR` / `SUBVOCAL_MODELS_DIR`.

---

## 📄 License
This repository is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.
