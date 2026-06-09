---
title: core.corrections
sidebar_label: corrections
---

Correction-capture loop and fine-tuning hook implementation for Subvocal SDK.

## Classes

### `class CorrectionLogEntry(BaseModel)`

Represents a single correction entry logged when a user fixes an incorrect reconstruction.

### `class CorrectionManager`

Manages local storage and retrieval of user correction log entries.

#### Methods

##### `__init__`

```python
def __init__(self, log_path: str | None = None)
```

Initializes the manager.


**Arguments:**

- ` log_path `: Path to the JSONL log file. Defaults to 'sdk/data/corrections_log.jsonl'.

##### `log_correction`

```python
def log_correction(self, raw_shorthand: str, decoded_intent: str, corrected_intent: str, context: UserContext) -> CorrectionLogEntry
```

Logs a new correction entry to the local JSONL file.

##### `get_corrections`

```python
def get_corrections(self) -> list[CorrectionLogEntry]
```

Retrieves all logged corrections from the local file.

##### `clear_logs`

```python
def clear_logs(self) -> None
```

Clears all logged corrections in the file.

### `class FinetuningHook`

Converts correction logs into training datasets for fine-tuning LLMs.

#### Methods

##### `export_to_openai`

```python
def export_to_openai(entries: list[CorrectionLogEntry], system_instruction: str = 'Translate silent speech shorthand and context into correct system actions.') -> list[dict[str, Any]]
```

Converts entries to OpenAI chat fine-tuning format.

Format:
    \{"messages": [\{"role": "system", "content": ...\}, \{"role": "user", "content": ...\}, \{"role": "assistant", "content": ...\}]\}

##### `export_to_gemini`

```python
def export_to_gemini(entries: list[CorrectionLogEntry], system_instruction: str = 'Translate silent speech shorthand and context into correct system actions.') -> list[dict[str, Any]]
```

Converts entries to Google Gemini fine-tuning format.

Format:
    \{"contents": [...], "systemInstruction": \{...\}\}

##### `export_to_jsonl`

```python
def export_to_jsonl(data: list[dict[str, Any]], output_path: str) -> int
```

Writes the exported dataset list to a JSONL file.
