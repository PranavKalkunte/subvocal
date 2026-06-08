# Subvocal: Physiological Silent Speech Interface

This repository contains the ground-up backend implementation of the **Subvocal Phase 0 Software Demonstrator**. It radically improves upon competitor baselines (such as TreeHacks 2026 / MindOS) by introducing an **articulatory phonetic shorthand layer** and **command-aware context decoding** to bypass the vocabulary ceiling of silent speech interfaces.

## 🚀 Key Improvements Over Competitors
1. **17-Command Target Vocabulary**: Expanded from 8 to 17 commands optimized for throat/jaw sEMG muscular separation.
2. **Shorthand Input Simulator**: Models human speech compression (vowel dropping, deduplication, abbreviations) and inserts sEMG noise based on speech physiology clusters.
3. **Context-Aware Phonetic Decoder**: Uses custom **asymmetric Levenshtein distance** (discounting consonant omissions) and **Command-Aware Prioritization** to match queries against contacts, calendar events, and screen UI elements with **100% accuracy** on direct commands.
4. **Local Text-to-Speech Engine**: Zero-dependency offline audio feedback utilizing macOS CLI fallback (`say`/`afplay`) and OpenAI Audio support.
5. **Robust Evaluation Harness**: Runs a 50-example test suite tracking accuracies, category performance, and trace failures.

---

## 📂 Project Structure
```
software/
├── shorthand/             # Intent reconstruction and shorthand package
│   ├── vocab.py           # 17-command phonetic vocabulary
│   ├── spec.py            # Shorthand grammar and sEMG phonetic clusters
│   ├── simulator.py       # Articulatory sEMG noise simulator
│   ├── decoder.py         # Asymmetric Levenshtein & hybrid LLM decoder
│   ├── eval_set.py        # 50-case benchmark dataset
│   └── eval_runner.py     # Evaluation benchmark execution harness
├── context/               # Context tracking and manager package
│   ├── schema.py          # Pydantic schemas (Contacts, Calendar, AppState)
│   ├── mock_data.py       # High-fidelity context data generator
│   ├── manager.py         # Phonetic searching and context state manager
│   └── test_context.py    # Integration test for context-aware decoding
├── tts/                   # Audio feedback package
│   ├── schema.py          # Pydantic voice configurations
│   └── engine.py          # Multi-backend local TTS player (say/afplay/OpenAI)
├── audio_output/          # Generated TTS audio cache folder
├── PHASE0_METHOD_RESULTS.md # Detailed scientific paper of Phase 0 results
└── README.md              # This file
```

---

## 🛠️ Installation & Setup

1. **Python Dependencies**:
   This project relies on Pydantic for context schema management.
   ```bash
   pip install pydantic
   ```
   *Note: No other external packages are required! The TTS engine runs offline on macOS using native subprocess utilities.*

2. **API Keys (Optional)**:
   To enable LLM-assisted reconstruction for complex/free-text inputs, export your preferred API key:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   # OR
   export OPENAI_API_KEY="your-api-key"
   # OR
   export ANTHROPIC_API_KEY="your-api-key"
   ```
   *If no API keys are present, the reconstructor will run in offline Heuristic-only mode.*

---

## 📈 Running the Demonstrators

### 1. Run the Shorthand Evaluation Harness
Execute the 50-case intent-reconstruction benchmark to measure heuristic accuracies and latencies:
```bash
python3 software/shorthand/eval_runner.py
```

### 2. Run Context Integration Tests
Execute the context-aware shorthand decoding test to verify contact, calendar, and UI button resolution:
```bash
python3 software/context/test_context.py
```

### 3. Test the Text-to-Speech Engine
Synthesize and play a test message locally:
```bash
python3 software/tts/engine.py
```
Generated files will be cached in `software/audio_output/`.

---

## 📊 Evaluation Benchmark Summary
```
================================================================================
EVALUATION METRICS SUMMARY (Heuristic-only, Offline)
================================================================================
Heuristic Decoder Accuracy: 74.0% (37/50)
Heuristic Avg Latency     : 0.72 ms
--------------------------------------------------------------------------------
Category        | Count | Heuristic Accuracy  
-----------------------------------------------
GOTO            | 10    |              100.0%
CLICK           | 9     |              100.0%
SCROLL          | 4     |              100.0%
CLOSE           | 2     |              100.0%
SYSTEM CONTROLS | 5     |              100.0%
SEARCH          | 10    |               30.0%
TYPE            | 10    |               40.0%
================================================================================
```
For detailed methodology and experimental findings, see [PHASE0_METHOD_RESULTS.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/software/PHASE0_METHOD_RESULTS.md).
