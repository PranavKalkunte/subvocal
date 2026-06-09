---
title: shorthand.decoder
sidebar_label: decoder
---

Hybrid decoder for subvocal compressed shorthand.

Combines an articulatory-distance heuristic alignment algorithm (dynamic programming)
with a model-agnostic LLM reconstruction layer.

## Functions

### `articulatory_distance`

```python
def articulatory_distance(word1: str, word2: str) -> float
```

Calculates Levenshtein distance customized for sEMG articulatory phonetic groups.

Substitutions between letters in the same articulatory group cost 0.25 (highly likely sEMG error).
Substitutions across different articulatory groups cost 1.0.
Deletion in candidate (extra character in query) costs 1.0.
Insertion in candidate (skipped character in query shorthand) costs 0.4.

### `find_best_shorthand_match`

```python
def find_best_shorthand_match(noisy_token: str, candidate_words: list[str]) -> list[tuple[str, float]]
```

Scores a noisy shorthand token against a list of candidate words.

Compresses candidates to shorthand and computes articulatory distance.
Returns a list of tuples (candidate, distance) sorted by distance.

### `heuristic_decode_phrase`

```python
def heuristic_decode_phrase(noisy_phrase: str, web_context_words: list[str] | None = None, calendar_words: list[str] | None = None, contacts_words: list[str] | None = None, ui_elements: list[str] | None = None, contacts: list[str] | None = None, calendar_events: list[str] | None = None) -> tuple[str, float]
```

Decodes a full noisy shorthand phrase using articulatory heuristics.

Uses Command-Aware Context Prioritization and Phrase-Level Matching.
If structured contexts are provided, they are prioritized based on the command.
Otherwise, falls back to flat context words.

### `call_llm_api`

```python
def call_llm_api(provider: str, api_key: str, model: str | None, prompt: str) -> str | None
```

Helper to perform model-agnostic LLM calls via urllib without external SDK dependencies.

### `reconstruct_intent_llm`

```python
def reconstruct_intent_llm(noisy_input: str, heuristic_candidate: str, web_context: str | None = None, calendar: str | None = None, contacts: str | None = None, history: str | None = None) -> str | None
```

Reconstructs the target command phrase from noisy shorthand using a model-agnostic LLM.

Discovers API keys dynamically from the environment.

### `hybrid_decode`

```python
def hybrid_decode(noisy_phrase: str, web_context_words: list[str] | None = None, calendar_words: list[str] | None = None, contacts_words: list[str] | None = None, history: str | None = None, ui_elements: list[str] | None = None, contacts: list[str] | None = None, calendar_events: list[str] | None = None) -> tuple[str, float, str]
```

Combines heuristic alignment and LLM disambiguation for premium accuracy.


**Returns:**

- A tuple of (decoded_phrase, confidence, method_used)
- Method used is either "heuristic" or "llm".
