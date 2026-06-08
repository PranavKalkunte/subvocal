# Code Review Findings — Subvocal SDK (Tasks 1–3)
Review date: 2026-06-07  
Scope: `sdk/core/`, `sdk/shorthand/decoder.py`, `sdk/context/`  
Method: 7-angle multi-agent review (correctness × 3, reuse, simplification, efficiency, altitude)  
Verified: 1-vote verifier per candidate (CONFIRMED / PLAUSIBLE / REFUTED)

---

## Summary

| # | Severity | Status | File | Issue |
|---|----------|--------|------|-------|
| 1 | Critical | Fixed | `corrections.py:112` | heuristic_recommendation is wrong LLM output in fine-tuning |
| 2 | Critical | Fixed | `corrections.py:106,145` | Calendar format missing `start_time` in fine-tuning exports |
| 3 | Critical | Fixed | `corrections.py:107,146` | UI element format missing `element_type` in fine-tuning exports |
| 4 | High | Fixed | `pipeline.py:105` | No exception handling — tokens lost on LLM network failure |
| 5 | High | Fixed | `decoder.py:459` | LLM called unconditionally despite documented confidence threshold |
| 6 | High | Fixed | `schema.py:69` | `UserContext.app_state` required with no default — latent crash |
| 7 | Medium | Fixed | `corrections.py:33` | Log path relative to `__file__`, breaks on pip install |
| 8 | Medium | Fixed | `decoder.py:388` | Inline prompt duplicates `PROMPT_v1` — two divergent sources |
| 9 | Medium | Fixed | `llm_providers.py:130` | `PromptManager` rebuilt on every inference call |
| 10 | Medium | Fixed | `llm_providers.py:100` | `Intent.confidence` hardcoded `0.95` — disables correction-loop trigger |
| 11 | Medium | Open | `llm_providers.py:119` | heuristic preamble copy-pasted across all 4 providers |
| 12 | Medium | Open | `models.py:46,55` | Confidence fields have no range validator [0.0, 1.0] |
| 13 | Medium | Open | `decoder.py:179` | GOTO hardcoded URL list baked into decoder layer |
| 14 | Low | Open | `llm_providers.py:117` | Default model names hardcoded per subclass, no central registry |
| 15 | Low | Open | `context/providers.py:116` | CompositeContextProvider merge strategy implicit (last-writer-wins) |
| 16 | Low | Open | `corrections.py:179` | `export_to_jsonl` does not validate empty `output_path` |
| 17 | Low | Open | `llm_providers.py:45` | `_get_api_key` returns `""` when `mock_response` is set; Gemini embeds empty key in URL |

---

## Fixed Findings

### 1. Fine-tuning `heuristic_recommendation` set to wrong value
**File:** `corrections.py:112`  **Severity:** Critical

`export_to_openai/export_to_gemini` passed `entry.decoded_intent` (the prior wrong LLM output) as `heuristic_recommendation`. At serving time all providers pass the output of `heuristic_decode_phrase()` there. Every fine-tuning example taught the model that the heuristic field means "what the LLM guessed wrong last time" — the exact inverse of its real meaning. Training/serving signal corruption.

**Fix:** Extracted `_build_user_prompt()` static helper that calls `heuristic_decode_phrase()` on `entry.raw_shorthand` to regenerate the correct heuristic output before building the prompt.

---

### 2. Fine-tuning calendar format missing `start_time`
**File:** `corrections.py:106` (and `:145` in Gemini export)  **Severity:** Critical

`export_to_openai` formatted calendar as `"{e.title} ({e.shorthand_title})"`.  
`_format_context` formats it as `"{e.title} at {e.start_time} ({e.shorthand_title})"`.  
Every fine-tuning example with calendar context used a structurally different string than what the deployed model receives.

**Fix:** Canonical formatting now lives in `_build_user_prompt()` and matches `_format_context` exactly. Both export methods call the shared helper.

---

### 3. Fine-tuning UI element format missing `element_type`
**File:** `corrections.py:107` (and `:146` in Gemini export)  **Severity:** Critical

`export_to_openai` formatted UI elements as `"{el.label} ({el.shorthand_label})"`.  
`_format_context` formats them as `"{el.label} [{el.element_type}] ({el.shorthand_label})"`.  
CLICK disambiguation — the command most dependent on UI element context — was the primary casualty.

**Fix:** Same as #2. `_build_user_prompt()` uses the correct format string.

---

### 4. Pipeline drops token buffer on LLM network failure
**File:** `pipeline.py:105`  **Severity:** High

`process_phrase()` called `clear_buffer()` at line 99, then `reconstruct_intent()` at line 105 with no exception handling. `_execute_http_post` raises `RuntimeError` on HTTP errors and timeouts. Any transient failure crashed the pipeline loop and permanently lost all accumulated tokens with no retry or log.

**Fix:** Wrapped `reconstruct_intent()` in `try/except Exception: return None` so the pipeline continues on failure.

---

### 5. `hybrid_decode` calls LLM unconditionally, ignoring confidence threshold
**File:** `decoder.py:459`  **Severity:** High

Comment said "We trigger the LLM if confidence is low (< 0.90)" but code did `if has_key:` with no reference to `heur_conf`. Every decode call with an API key set made a live HTTP round-trip regardless of heuristic confidence — including cases where confidence was 1.0.

**Fix:** Changed condition to `if has_key and heur_conf < 0.9:`.

---

### 6. `UserContext.app_state` required with no default — latent crash
**File:** `schema.py:69`  **Severity:** High

`app_state: AppState = Field(description=...)` had no `default` or `default_factory`. Any caller building `UserContext(contacts=[...])` without an explicit `app_state` raises a Pydantic `ValidationError`. All four LLM providers then crash inside `_format_context` at `context.app_state.visible_elements`.

**Fix:** Added `default_factory=lambda: AppState(current_app="")`.

---

### 7. Correction log path resolves relative to `__file__` — breaks on `pip install`
**File:** `corrections.py:33`  **Severity:** Medium

`os.path.join(os.path.dirname(__file__), "..", "data", "corrections_log.jsonl")` resolves correctly in a dev tree but resolves into `site-packages` after `pip install`, which is typically not writable.

**Fix:** Changed default to `~/.subvocal/data/corrections_log.jsonl` via `os.path.expanduser("~")`.

---

### 8. `reconstruct_intent_llm` duplicates `PROMPT_v1` as an inline f-string
**File:** `decoder.py:388`  **Severity:** Medium

`reconstruct_intent_llm()` built a full prompt inline including its own `vocab_desc` rebuild from `COMMANDS` — substantively identical to `PROMPT_v1` in `sdk/core/prompts.py`. Any prompt change had to be applied in two places independently.

**Fix:** Replaced the inline f-string with `PromptManager().format_prompt(...)`.

---

### 9. `PromptManager` re-instantiated on every inference call
**File:** `llm_providers.py:130`  **Severity:** Medium

Every call to `reconstruct_intent()` in all four providers created a local `PromptManager(...)`, which rebuilds `vocab_desc` by iterating `COMMANDS` and allocates the templates dict. `COMMANDS` is a module-level constant.

**Fix:** `PromptManager` is now instantiated once in `BaseLLMProvider.__init__()` as `self._prompt_mgr`. Module-level `from .prompts import PromptManager` import added; lazy per-method imports removed.

---

### 10. `Intent.confidence` hardcoded to `0.95` for all LLM-resolved intents
**File:** `llm_providers.py:100`  **Severity:** Medium

`_parse_llm_output` always set `confidence=0.95` regardless of input token quality. Any downstream logic gating on `intent.confidence` (e.g., surfacing a correction prompt below a threshold) would never trigger because the value was always the same.

**Fix:** Added `default_confidence: float = 0.95` parameter to `_parse_llm_output`. Each provider computes `mean_conf = sum(t.confidence for t in tokens) / len(tokens)` from the input token stream and passes it through, so confidence reflects actual classifier certainty.

---

## Open Findings

### 11. heuristic preamble copy-pasted across all 4 providers
**File:** `llm_providers.py:119`  **Severity:** Medium  **Status:** Open

The 6-line block computing `raw_shorthand`, calling `heuristic_decode_phrase()`, and calling `_format_context()` is duplicated verbatim across `ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`, and `LlamaProvider`. `BaseLLMProvider` already exists as the shared base. The prompt-building step was fixed (finding #9), but this heuristic preamble remains duplicated.

**Recommended fix:** Extract `_build_prompt(self, tokens, context) -> str` into `BaseLLMProvider`. Each subclass `reconstruct_intent` reduces to: compute `mean_conf`, call `self._build_prompt(tokens, context)` to get the prompt, make the HTTP call, parse the response.

---

### 12. Confidence fields have no range validator
**File:** `models.py:46,55`  **Severity:** Medium  **Status:** Open

`CommandToken.confidence` and `Intent.confidence` are documented as "between 0.0 and 1.0" but Pydantic never enforces it. A malformed API response or unclamped classifier softmax can produce values outside [0, 1] and silently corrupt downstream gating logic and fine-tuning exports.

**Recommended fix:**
```python
confidence: float = Field(description="...", ge=0.0, le=1.0)
```

---

### 13. GOTO hardcoded URL list baked into the decoder layer
**File:** `decoder.py:179`  **Severity:** Medium  **Status:** Open

The GOTO candidate pool is a hardcoded list of 10 URLs inside `heuristic_decode_phrase`. SCROLL and ZOOM also hardcode their argument vocabularies inline. Any site not in the list gets no heuristic match. Adding a new command with a fixed argument vocabulary requires modifying the decoder's elif chain directly, which is the wrong layer.

**Recommended fix:** Add a `static_args: Optional[List[str]]` field to each `COMMANDS` entry in `shorthand/vocab.py`. The decoder dispatches generically: `prioritized_pool = COMMANDS[best_cmd].get("static_args", [])`. URL lists and direction lists move into the vocabulary spec, out of the decoder logic.

---

### 14. Default model names hardcoded per subclass with no central registry
**File:** `llm_providers.py:117`  **Severity:** Low  **Status:** Open

Each provider subclass has its default model string (`"claude-3-5-sonnet-20241022"`, `"gpt-4o"`, `"gemini-1.5-flash"`, `"llama3"`) as a literal inside `reconstruct_intent`. When a model is deprecated, every subclass must be hunted and patched individually. The benchmark runner also hardcodes model names separately.

**Recommended fix:** `PROVIDER_DEFAULTS: Dict[str, str]` dict in `llm_providers.py` or a config module. Each provider references `PROVIDER_DEFAULTS[self.get_provider_name()]`.

---

### 15. `CompositeContextProvider` merge strategy is implicit
**File:** `context/providers.py:116`  **Severity:** Low  **Status:** Open

App state uses last-writer-wins with a `!= "System"` heuristic. Contacts and calendar are concatenated with no deduplication. Both behaviors are invisible to callers — the outcome depends silently on provider ordering. Two providers supplying the same contact (e.g., a local contacts DB and a Google Contacts provider) will double-inject that contact into the decoder pool.

**Recommended fix:** Accept an explicit `merge_strategy` enum parameter, or at minimum deduplicate contacts/calendar entries by `id` field before returning the merged context.

---

### 16. `export_to_jsonl` does not validate empty `output_path`
**File:** `corrections.py:179`  **Severity:** Low  **Status:** Open

If called with `output_path=""`, `os.path.abspath("")` resolves to the CWD; `os.path.dirname` returns the parent; `open("", "w")` raises `FileNotFoundError` with a confusing message. No validation that `output_path` is non-empty and points to a writable path is performed before the filesystem operations.

**Recommended fix:** Add `if not output_path: raise ValueError("output_path must be non-empty")` at the top of the method.

---

### 17. `_get_api_key` returns `""` when `mock_response` is set; Gemini embeds it in URL
**File:** `llm_providers.py:45`  **Severity:** Low  **Status:** Open

When `mock_response` is set, `_get_api_key` returns `""` without raising. `GeminiProvider` embeds the key directly in the URL: `?key={key}`. If `mock_response` is later cleared without supplying a real key, the live request fires to `?key=` — a silent empty-key request rather than a clear error. Claude and OpenAI use headers so the empty string is less dangerous there (the server returns 401 with a readable message), but the empty-string path should raise for all providers in non-mock mode.

**Recommended fix:** Remove the `return key or ""` fallback. If `mock_response` is None and no key, always raise. The mock path should never reach `_get_api_key`.

---

## Files changed in this review

| File | Changes made |
|------|-------------|
| `sdk/core/corrections.py` | Fixed log path → `~/.subvocal/data/`; extracted `_build_user_prompt()`; fixed calendar/UI format strings to match `_format_context`; fixed `heuristic_recommendation` to use articulatory heuristic output; deduplicated both export methods |
| `sdk/core/llm_providers.py` | Added `PromptManager` to module-level imports; moved to `self._prompt_mgr` in `__init__`; added `default_confidence` param to `_parse_llm_output`; threaded `mean_conf` from input tokens through all 4 providers; removed 4 lazy per-method PromptManager instantiations |
| `sdk/core/pipeline.py` | Wrapped `reconstruct_intent()` in `try/except Exception: return None` |
| `sdk/shorthand/decoder.py` | Added `heur_conf < 0.9` gate to LLM trigger; replaced inline prompt f-string with `PromptManager().format_prompt()` |
| `sdk/context/schema.py` | Added `default_factory=lambda: AppState(current_app="")` to `app_state` field |
