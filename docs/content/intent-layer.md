# Subvocal SDK: Intent Reconstruction Layer

**Status:** Draft Spec  
**Version:** v0.1.0-alpha  
**Date:** June 2026  
**Audience:** Platform Developers, Systems Integrators  

---

## 1. Subsystem Overview

The **Intent Reconstruction Layer** forms the core decoding pipeline of the Subvocal SDK. It bridges raw electromyographic signal classification (Command Tokens) with LLM-driven actions (Agent Execution).

Surface electromyography (sEMG) from the throat and neck features a raw word error rate (WER) of 15% to 25% due to physiological noise and dropped vowels in subvocal articulation. Rather than attempting direct speech-to-text transcription, the Subvocal SDK reframes the input as a **low-bandwidth intent channel**. 

It uses a **Hybrid Decoder Pipeline**:
1. **Heuristic Alignment (Stage A):** Performs fast, local Levenshtein alignment on the phonetic group level using a customized articulatory distance matrix to resolve commands and prioritize arguments based on visible UI elements, contacts, or schedule items.
2. **Context-Aware LLM Reconstruction (Stage B):** If the heuristic decoder's confidence score falls below a threshold (default: 0.90), or if the query requires complex phrase expansion, the SDK dispatches the noisy shorthand and heuristic recommendations to an LLM provider client along with active device contexts.

```
[ Noisy Shorthand ] ────> [ Heuristic Alignment ] ───( Conf >= 0.90 )───> [ Fast Resolved Intent ]
                                  │
                          ( Conf < 0.90 )
                                  ▼
[ UserContext Snapshot ] ─> [ LLM Provider Client ] ────────────────────> [ Context-Reconstructed Intent ]
```

---

## 2. Pluggable LLM Providers

The SDK defines a model-agnostic REST client architecture implemented in `sdk/core/llm_providers.py`. These clients utilize only the Python standard library's `urllib` to hit endpoints, avoiding bloated external SDK packages:

* **`ClaudeProvider`:** Communicates with Anthropic's Messages REST API (`https://api.anthropic.com/v1/messages`).
* **`OpenAIProvider`:** Communicates with OpenAI's Chat Completions REST API (`https://api.openai.com/v1/chat/completions`).
* **`GeminiProvider`:** Communicates with Google's Gemini Content Generation API (`https://generativelanguage.googleapis.com/v1beta/models`).
* **`LlamaProvider`:** Communicates with local Llama/Ollama endpoints via their OpenAI-compatible `/v1/chat/completions` REST interface.

Maintainers can register new providers by subclassing the abstract `LLMProvider` base class and implementing the `reconstruct_intent` method.

---

## 3. Prompt Template Versioning

Prompts are stored, versioned, and resolved via `PromptManager` in `sdk/core/prompts.py`. Versioning is critical for ensuring regression-free model upgrades:

* **`v1` (Default):** Targets basic translation instructions, listing shorthand rules, vocabulary descriptions, and user contexts as flat text lists. Optimized for standard chat models.
* **`v2`:** A structured system-instruction template that explicitly defines boundaries, inputs, and outputs, designed for next-generation thinking models.

### Dynamic Prompt Registration
Developers can register custom prompt templates at runtime:
```python
from core.prompts import PromptManager

manager = PromptManager()
manager.register_template("v3-custom", "System: Resolve shorthand... Inputs: {noisy_input}")
```

---

## 4. Composite Context Provider Pattern

To resolve ambiguous shorthand tokens (e.g. mapping `clk sbt` to `CLICK Submit`), the LLM requires structured context snapshots. In `sdk/context/providers.py`, context retrieval is modularized:

1. **`CalendarContextProvider`:** Exposes upcoming calendar titles and start/end times.
2. **`ContactsContextProvider`:** Exposes contacts name and pre-computed phonetic shorthands.
3. **`LocationContextProvider`:** Exposes user GPS coordinates and place name.
4. **`AppStateContextProvider`:** Exposes the active application and active interactive UI elements.
5. **`CompositeContextProvider`:** Takes a list of these sub-providers and aggregates their context data into a single unified `UserContext` model on demand.

### Example Configuration:
```python
from context.providers import (
    ContactsContextProvider,
    AppStateContextProvider,
    CompositeContextProvider
)

# Initialize modular sources
contacts_src = ContactsContextProvider(load_contacts_database())
app_state_src = AppStateContextProvider(poll_screen_elements())

# Merge sources into a unified provider
context_provider = CompositeContextProvider([contacts_src, app_state_src])
live_context = context_provider.get_context()
```

---

## 5. Correction-Capture Loop and Fine-Tuning Hook

No machine learning pipeline is perfect. When an incorrect action is triggered, the user registers a correction (e.g., through an voice undo, physical interface override, or subsequent keyboard entry).

The SDK implements a feedback loop in `sdk/core/corrections.py`:

### The Correction Logging Loop
The `CorrectionManager` automatically logs correction entries to a local JSONL file (`sdk/data/corrections_log.jsonl`):
```python
from core.corrections import CorrectionManager

# Initialize the manager
manager = CorrectionManager()

# Log a correction
manager.log_correction(
    raw_shorthand="typ alc",
    decoded_intent="TYPE Alex",
    corrected_intent="TYPE Alice",
    context=active_context
)
```

Each log line represents a serialized `CorrectionLogEntry` Pydantic model:
```json
{
  "timestamp": 1780823400.0,
  "raw_shorthand": "typ alc",
  "decoded_intent": "TYPE Alex",
  "corrected_intent": "TYPE Alice",
  "context_snapshot": {
    "contacts": [{"id": "1", "name": "Alice", "shorthand_name": "alc"}],
    "app_state": {"current_app": "Messages"}
  }
}
```

### The Fine-Tuning Export Hook
To close the loop, the `FinetuningHook` class converts logged corrections into training examples formatted for model fine-tuning:

```python
from core.corrections import CorrectionManager, FinetuningHook

manager = CorrectionManager()
corrections = manager.get_corrections()

# 1. Export to OpenAI Chat Fine-Tuning Format (JSONL)
openai_dataset = FinetuningHook.export_to_openai(corrections)
FinetuningHook.export_to_jsonl(openai_dataset, "sdk/data/openai_tuning.jsonl")

# 2. Export to Gemini Fine-Tuning Format (JSONL)
gemini_dataset = FinetuningHook.export_to_gemini(corrections)
FinetuningHook.export_to_jsonl(gemini_dataset, "sdk/data/gemini_tuning.jsonl")
```

This local log can be uploaded periodically to cloud APIs or used to fine-tune a local Llama model (via Ollama or LLaMA-Factory) to personalize the intent reconstruction layer for a specific user's physiological speech patterns.
