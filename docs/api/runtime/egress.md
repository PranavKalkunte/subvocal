---
title: runtime.egress
sidebar_label: egress
---

## Classes

### `class EgressManager`

Coordinates downstream data exports, JSONL tracing, and audio speech feedback.

#### Methods

##### `__init__`

```python
def __init__(self, trace_path: str | None = None, tts_engine: Any | None = None)
```

No description.

##### `speak`

```python
def speak(self, text: str) -> None
```

Plays speech feedback to the user via TTS.

##### `write_trace`

```python
def write_trace(self, trace_entry: dict) -> None
```

Appends a trace entry to the pipeline trace log file.

##### `record_signal`

```python
def record_signal(self, output_path: str, frames: list[Any]) -> None
```

Saves raw biometric frames to a JSON document for training/fine-tuning models.
