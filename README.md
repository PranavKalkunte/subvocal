# Subvocal SDK: Physiological Silent Speech Interface Middleware

The **Subvocal SDK** is an open-source, hardware-agnostic middleware platform that connects surface electromyography (sEMG) interfaces to LLM-driven AI agents. 

Rather than locking developers to a proprietary neckband or a closed whole-word vocabulary, the Subvocal SDK provides the software rails—signal conditioning, deep learning training skeletons, articulatory phonetic shorthand simulators, and context-aware decoders—to enable high-accuracy, low-latency, and open-vocabulary silent speech control.

---

## 📂 Repository Structure

The SDK is structured as a monorepo containing modular packages:

```
subvocal/
├── sdk/
│   ├── shorthand/          # Compressed shorthand simulator & intent reconstructor
│   │   ├── vocab.py        # 17-command phonetic target vocabulary
│   │   ├── spec.py         # Shorthand grammar and sEMG phonetic clusters
│   │   ├── simulator.py    # Physiological sEMG noise channel simulator
│   │   ├── decoder.py      # Asymmetric Levenshtein & hybrid LLM decoder
│   │   └── eval_runner.py  # 50-case benchmark execution harness
│   ├── context/            # Pydantic schemas and search manager for user context
│   │   ├── schema.py       # Contacts, Calendar, and AppState context models
│   │   └── manager.py      # Phonetic context matching
│   ├── tts/                # Zero-dependency local TTS audio feedback engine
│   │   └── engine.py       # macOS native CLI say/afplay fallback + OpenAI Audio
│   └── emg_core/           # Digital Signal Processing (DSP) & ML classifiers
│       ├── config.py       # Central hardware and training configurations
│       ├── dsp/
│       │   ├── filters.py  # Reconciled 1.3-50 Hz bandpass and notch filters
│       │   └── features.py # TD10 segment feature extraction (840 dims)
│       └── ml/
│           ├── train.py    # Random Forest, 1D CNN, and GRU PyTorch models
│           ├── infer.py    # Unified InferenceEngine with confidence & cooldown gating
│           └── model_io.py # Model serialization (.pth and .joblib)
├── platform/               # Publishable specifications (thesis positioning & declaration post)
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🚀 Core Features

1. **Articulatory Shorthand Decoder**: Overcomes the whole-word sEMG vocabulary ceiling. Decodes compressed phonetic consonant shorthand inputs (e.g. `g gl` -> `Google`) under heavy muscle-movement noise.
2. **Asymmetric Levenshtein Distance**: A dynamic programming string alignment cost matrix configured with physiological sEMG confusion clusters (Glottal, Labial, Alveolar, Velar, Rhotic) to discount vowel/consonant omissions in silent speech.
3. **Command-Aware Context Prioritization**: Dynamic target matching against active user contacts (`TYPE`), calendar events (`SEARCH`), browser URLs (`GOTO`), and active application screen elements (`CLICK`).
4. **Physiological Signal Conditioning**: Preprocessing filter configurations defaulting to AlterEgo's `1.3–50.0 Hz` bandpass filter (designed for low-velocity articulatory gestures) with configuration support for standard `20.0–450.0 Hz` EMG.
5. **Classifiers (RF + Deep Learning)**: Custom pipelines to train scikit-learn **Random Forest**, PyTorch **1D CNN**, and PyTorch **GRU** architectures on raw multi-channel sEMG traces.
6. **Zero-Dependency TTS Engine**: Local, offline audio playback relying on macOS command-line utilities (`say` and `afplay`) for instant audio confirmations.

---

## 🛠️ Quickstart

### 1. Installation & Requirements
The Subvocal SDK runs fully offline with zero mandatory heavy dependencies. You only need standard packages for data modeling:

```bash
pip install pydantic scikit-learn joblib torch
```

*Note: The PyTorch models are configured to run on CPU to avoid device-pooling limitations on Apple Silicon (MPS).*

### 2. Verify the SDK
Run the integration test suite to verify that all packages, decoders, and PyTorch classifiers are compiling and training correctly:

```bash
# Test 1: ML Core Classifiers (RF, CNN, GRU)
python3 sdk/emg_core/test_ml.py

# Test 2: Shorthand Phonetic Alignment
python3 sdk/shorthand/test_shorthand.py

# Test 3: Context-Aware Intent Resolution
python3 sdk/context/test_context.py
```

### 3. Run the 50-Case Reconstruction Benchmark
Execute the evaluation harness to measure intent expansion accuracy and average processing latencies across 50 realistic shorthand command examples:

```bash
python3 sdk/shorthand/eval_runner.py
```

---

## 🗺️ Roadmap & Model Context Protocol (MCP)

Our immediate roadmap focuses on standardizing the API surface. In upcoming releases, we will publish the **Model Context Protocol (MCP)** server integration, exposing the intent-reconstruction decoders as standard MCP tools. This will allow Claude Desktop or any MCP-compatible agent framework to ingest low-bandwidth subvocal inputs natively.

---

## 📄 License
This repository is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.
