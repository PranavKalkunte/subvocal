# Phase 0 Software Demonstrator: Articulatory Phonetic Shorthand & Context-Aware Intent Reconstruction

## Abstract
Traditional silent speech interfaces (such as MindOS/TreeHacks 2026) are bottlenecked by a **vocabulary ceiling** (typically limited to 8 closed-vocabulary whole-word commands) and a high sensitivity to electromyographic (sEMG) signal noise. 

This paper introduces the **Phase 0 Software Demonstrator** for a neckband silent speech interface. We overcome the vocabulary ceiling by defining a **17-command phonetically diverse vocabulary** and introducing a **Compressed Shorthand Layer** that simulates noisy subvocal speech. To decode this shorthand under heavy signal degradation, we implement a **Hybrid Intent Reconstructor** utilizing:
1. **Asymmetric sEMG Levenshtein Alignment**: A dynamic-programming string matcher configured with physiological sEMG confusion clusters.
2. **Command-Aware Context Prioritization**: Dynamic ranking of candidate pools (UI elements, contacts, calendar events) based on command intent.
3. **Phrase-Level Alignment**: Reconstructing multi-word commands (like `Sign In`) by aligning concatenated shorthand tokens.
4. **Model-Agnostic LLM Fallback**: Reconstructing free-text intents using standard, SDK-free REST requests to OpenAI, Anthropic, or Gemini APIs.

We evaluate this system on a 50-case intent-reconstruction test suite, demonstrating **74.0% overall heuristic accuracy** (reaching **100%** on GOTO, CLICK, SCROLL, and system controls) with an average execution latency of **0.72 ms** per case.

---

## 1. Target Command Vocabulary
We expanded the vocabulary from TreeHacks' 8 commands to **17 commands**, categorizing them by action type and identifying dominant articulatory features to maximize sEMG separation:

| Command | Action Category | Phonetic IPA | Dominant sEMG Articulation |
| :--- | :--- | :--- | :--- |
| **GOTO** | Navigation | `ˈɡoʊ.tuː` | Velar plosive + Glottal vowel + Dental/Alveolar plosive |
| **SEARCH** | Navigation | `sɜːrtʃ` | Dental/Alveolar fricative + Rhotic + Palatal affricate |
| **CLICK** | Interaction | `klɪk` | Velar plosive + Dental/Alveolar lateral + Velar plosive |
| **SCROLL** | Interaction | `skroʊl` | Dental/Alveolar fricative + Velar plosive + Rhotic + Lateral |
| **TYPE** | Input | `taɪp` | Dental/Alveolar plosive + Diphthong vowel + Labial plosive |
| **ENTER** | Action | `ˈɛntər` | Glottal vowel + Alveolar nasal + Alveolar plosive + Rhotic |
| **CONFIRM** | Confirmation | `kənˈfɜːrm` | Velar plosive + Alveolar nasal + Rhotic + Labial nasal |
| **CANCEL** | Abort | `ˈkænsəl` | Velar plosive + Alveolar nasal + Alveolar fricative + Lateral |
| **BACK** | Navigation | `bæk` | Labial plosive + Glottal vowel + Velar plosive |
| **FORWARD** | Navigation | `ˈfɔːrwərd` | Labial fricative + Rhotic + Labial semivowel + Alveolar plosive |
| **REFRESH** | Control | `rɪˈfrɛʃ` | Rhotic + Labial fricative + Rhotic + Palatal fricative |
| **ZOOM** | Control | `zuːm` | Alveolar fricative + Glottal vowel + Labial nasal |
| **CLOSE** | Control | `kloʊz` | Velar plosive + Alveolar lateral + Alveolar fricative |
| **COPY** | Clipboard | `ˈkɑːpi` | Velar plosive + Glottal vowel + Labial plosive |
| **PASTE** | Clipboard | `peɪst` | Labial plosive + Glottal vowel + Alveolar fricative + Plosive |
| **UNDO** | Control | `ʌnˈduː` | Glottal vowel + Alveolar nasal + Alveolar plosive |
| **WAIT** | Control | `weɪt` | Labial semivowel + Diphthong vowel + Alveolar plosive |

---

## 2. Compressed Shorthand Specification
Subvocal speech is highly compressed. When users silent-speak under muscular constraints, they tend to drop unstressed vowels and contract syllables. The spelling-shortening rules are formalized as:
1. **Vowel Dropping**: All non-initial vowels (`a`, `e`, `i`, `o`, `u`, `y`) and the weak glottal consonant `h` are omitted.
2. **Duplicate Collapse**: Consecutive duplicate letters are simplified to a single character (e.g., `google` -> `ggl`).
3. **Common Abbreviations**: Mapped via a static dictionary (e.g. `the` -> `th`, `and` -> `nd`, `for` -> `fr`).

---

## 3. Physiological sEMG Noise Model
To realistically evaluate decoder robustness, we grouped the English alphabet into 5 **articulatory phonetic clusters** matching jaw/throat sEMG activation zones:
* **Glottal/Vowels (`G`)**: `A, E, I, O, U, Y, H` (Vocal fold vibration, open mouth)
* **Labials (`L`)**: `B, P, M, W, F, V` (Lip closure, front jaw)
* **Alveolars/Dentals (`A`)**: `T, D, N, S, Z, L` (Tongue tip to teeth ridge)
* **Velars/Palatals (`V`)**: `K, G, J, C, Q, X` (Tongue back to soft palate)
* **Rhotics (`R`)**: `R` (Tongue curl/retroflex)

Our simulator applies three forms of channel noise based on configurable rates:
* **Articulatory Substitution**: Swapping a character with another in its same phonetic cluster (e.g., `t` -> `d` or `l` -> `n` at cost 0.25).
* **Signal Droplet (Deletion)**: Dropping a character entirely to simulate electrode contact loss.
* **Artifact Insertion**: Inserting spurious characters to model swallowing or muscle twitch spikes.

---

## 4. Intent Reconstructor (Decoder)
The intent reconstructor processes raw shorthand inputs using a three-stage pipeline:

### 4.1 Asymmetric sEMG Levenshtein Distance
Standard Levenshtein alignment penalizes deletions and insertions equally. In shorthand decoding, this fails because users *intentionally* drop consonants (resulting in extra characters in candidate words relative to the query). 
We implement an **asymmetric cost matrix**:
* Substituting characters in the same articulatory group: **0.25**
* Substituting characters in different articulatory groups: **1.0**
* Deleting characters from the query (query has extra char): **1.0**
* Inserting characters into the candidate (query omitted char): **0.4** (highly discounted)

### 4.2 Command-Aware Context Prioritization & Phrase-Level Alignment
When decoding arguments, candidate lists are prioritized dynamically based on the decoded command:
* `CLICK`/`SCROLL` -> Prioritizes active App State UI labels (`ui_elements`).
* `GOTO` -> Prioritizes common navigation domains.
* `TYPE`/`MESSAGE` -> Prioritizes user contacts (`contacts`).
* `SEARCH` -> Prioritizes upcoming calendar events (`calendar`).

Instead of matching word-by-word, multi-word queries are concatenated (e.g. `sgn n`) and matched directly against the concatenated shorthand of full labels (e.g. `Sign In` -> `sgnn`), reducing word-boundary errors. Longer titles (e.g. `BioSignals Research Review`) are expanded into constituent words to allow sub-phrase matches.

### 4.3 Domain Suffix Splitting
Tokens containing dots (such as `mail.google.com`) are split into sub-tokens, decoded independently, mapped to common domain extensions (`cm` -> `com`, `rg` -> `org`), and rejoined with a dot, allowing high-accuracy URL reconstruction under noise.

---

## 5. Text-to-Speech Engine
To close the feedback loop without visual overhead, we implemented a modular local TTS engine. It attempts:
1. **Cloud Speech API**: Synthesizes via OpenAI audio API if a key is present.
2. **macOS Native CLI Fallback**: Invokes macOS native `say` and `afplay` utilities via subprocess (completely offline, zero-dependency, high performance).
3. **pyttsx3**: Falls back to the cross-platform python library if installed.

---

## 6. Experimental Results & Benchmarks

### 6.1 Accuracy & Latency Breakdown
The evaluation harness ([eval_runner.py](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/software/shorthand/eval_runner.py)) was executed against 50 diverse test cases. Under a simulated noisy channel with no LLM connection (`LLM Decoder: INACTIVE`), the heuristic decoder achieved:
* **Overall Accuracy**: **74.0%** (37/50)
* **Average Latency**: **0.72 ms** per phrase

Category-specific performance:
* **GOTO**: **100.0%** (10/10)
* **CLICK**: **100.0%** (9/9)
* **SCROLL**: **100.0%** (4/4)
* **CLOSE**: **100.0%** (2/2)
* **SYSTEM CONTROLS**: **100.0%** (5/5)
* **SEARCH & TYPE (Free-Text)**: **35.0%** (7/20)

### 6.2 Comparison: Our Phase 0 vs. TreeHacks MindOS

| Dimension | TreeHacks MindOS (2026) | Our Phase 0 Ground Up | Competitive Impact |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size** | 8 closed whole-word commands | **17 commands** + free-text | Expanded capability and utility |
| **Argument Input** | None (Discrete classification) | **Compressed Shorthand Layer** | Allows typing arbitrary strings, URLs, and queries |
| **Context Awareness** | None (Tracks execution state only) | **Full User Context Schema** | Leverages App UI, Contacts, and Calendar to resolve shorthand |
| **Phonetic Matching** | None | **Asymmetric sEMG Levenshtein** | High noise resilience using speech physiology clusters |
| **Model Portability** | Locked to cloud GPT-4o | **Model-Agnostic + Local Fallbacks** | Offline viability and zero-dependency local TTS |

---

## 7. sEMG Signal Processing & Classifier Training (Section 6)

We implemented and verified the digital signal processing (DSP) and machine learning (ML) training skeleton in the `emg_core` package:

### 7.1 Reconciled Bandpass Filtering
Silent speech sEMG signals differ from traditional physical/prosthetic EMG. We resolved this divergence by implementing:
* **AlterEgo sEMG configuration (1.3–50.0 Hz, default)**: Rejects powerline hum (60 Hz) and high-frequency noise, focusing on slow-velocity articulation movements. This fits low sampling rates (e.g. 250 Hz) and acts as the default.
* **Prosthetics configuration (20.0–450.0 Hz, option)**: Built-in configuration toggle for testing standard high-frequency limb-movement signals.

### 7.2 Classifier Architectures
We implemented three model classes in `software/emg_core/ml/train.py`:
1. **Random Forest + LDA**: Standard ML model extracting 840 TD10 features.
2. **1D CNN (PyTorch)**: A deep learning model that applies 1D convolutions over raw sEMG channels.
3. **GRU (PyTorch)**: A temporal bidirectional GRU model with average pooling over time.

### 7.3 Integration Test Verification
The validation suite `test_ml.py` verified compilation, training, and inference across all three architectures on synthetic multichannel sEMG calibration segments (60 segments, 4 channels, 150-sample window).

All classifiers achieved successful compilation and training execution, generating valid accuracy metrics and confusion matrices on CPU:
* **Test results**: `Ran 4 tests. OK` (100% test pass).

