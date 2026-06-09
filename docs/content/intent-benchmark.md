# Subvocal SDK: Intent Reconstruction Benchmark Specs

**Status:** Published  
**Version:** v0.1.0-alpha  
**Date:** June 2026  
**Audience:** Platform Developers, AI Researchers  

---

## 1. Executive Summary

This document establishes the initial performance baseline of the Subvocal SDK's intent reconstruction layer. The benchmark evaluates the **Heuristic Decoder** and **Simulated LLM Decoder** across a test set of **50 diverse shorthand-to-intent test cases**.

### High-Level Metrics Summary:

| Decoder / Provider | Accuracy | Average Latency |
|---|---|---|
| **HEURISTIC** (Local Articulatory Distance) | 74.0% (37/50) | 0.67 ms |
| **SIMULATED LLM** (Target Threshold) | 96.0% (48/50) | 0.68 ms |

*Note: Simulated LLM represents the target accuracy baseline when utilizing the contextual prompt and model-agnostic REST adapters (such as Claude 3.5 Haiku or GPT-4o-mini).*

---

## 2. Test Set Structure & Category Breakdowns

The evaluation dataset comprises 50 cases spanning five critical categories of wearable device usage:

### 1. `GOTO` Navigation (10 cases)
* **Description:** Direct browser navigation to popular websites with varying consonant deletions (e.g., `gt ggl.cm` -> `GOTO google.com`).
* **Heuristic Performance:** Excellent. Fixed command patterns and structured domain expansions yield high accuracy.

### 2. Viewport & Window Controls (15 cases)
* **Description:** Page interactions like `CLICK`, `SCROLL`, and `CLOSE` (e.g. `clk sbt` -> `CLICK Submit`, `scrl dwn` -> `SCROLL Down`).
* **Heuristic Performance:** Very high. Command-aware prioritizations limit the search pool to visible UI elements.

### 3. `SEARCH` Queries (10 cases)
* **Description:** Free-text searches requiring spelling recovery and calendar/context cross-referencing (e.g., `srch wthr dstn` -> `SEARCH weather Austin`).
* **Heuristic Performance:** Moderate. Simple Levenshtein alignment struggles with word pluralizations or phonetic extensions.

### 4. `TYPE` Text Inputs (10 cases)
* **Description:** Free-text typing inputs, including typing user contact names (e.g., `typ alc smth` -> `TYPE Alice Smith`).
* **Heuristic Performance:** High when matching contacts; moderate when typing arbitrary phrases.

### 5. System Controls (5 cases)
* **Description:** Clipboard and modal actions (e.g., `cnfm` -> `CONFIRM`, `und` -> `UNDO`).
* **Heuristic Performance:** 100%. Single-token commands match vocabulary abbreviations cleanly.

---

## 3. Failure Profile Analysis

Analysis of the 13 failures under the Heuristic Decoder highlights the necessity of the hybrid LLM reconstruction layer:

1. **Pluralization Failures:**
   * *Noisy Input:* `srch nrl ntwk`
   * *Expected:* `SEARCH neural networks`
   * *Heuristic Got:* `SEARCH Neural network`
   * *Root Cause:* The heuristic aligner selected the singular form because it had a shorter distance than the plural. An LLM solves this using language modeling semantics.
2. **Contextual Confusions:**
   * *Noisy Input:* `srch fgt calb`
   * *Expected:* `SEARCH Figma calibration`
   * *Heuristic Got:* `SEARCH logout confirm`
   * *Root Cause:* The shorthand `fgt` was mapped to `logout` and `calb` was mapped to `confirm` due to raw phonetic distance ties. The LLM resolves this because "Figma calibration" is a valid calendar event, whereas "logout confirm" is not a semantically valid search phrase.
3. **Phonetic Word Boundary Shifts:**
   * *Noisy Input:* `srch snt spch`
   * *Expected:* `SEARCH silent speech`
   * *Heuristic Got:* `SEARCH send Speech`
   * *Root Cause:* `snt` mapped to `send` (present in vocabulary dictionary) rather than `silent`. An LLM resolves this because "silent speech" is a highly cohesive bigram, whereas "send speech" is atypical in browser searches.

---

## 4. How to Execute the Benchmarks

Developers can reproduce these benchmarks locally using the unified test runner:

```bash
# 1. Run in Simulated Mode (No API keys required)
PYTHONPATH=sdk python3 sdk/shorthand/test_intent_runner.py

# 2. Run in Live Mode (Evaluates live API performance)
export GEMINI_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"

PYTHONPATH=sdk python3 sdk/shorthand/test_intent_runner.py
```
