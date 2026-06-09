---
title: tts.engine
sidebar_label: engine
---

Text-to-Speech (TTS) engine for subvocal feedback.

Supports native macOS 'say' / 'afplay' utilities, OpenAI Audio API, and pyttsx3.

## Classes

### `class TTSEngine`

Multi-backend Text-to-Speech generator and player.

#### Methods

##### `__init__`

```python
def __init__(self, config: TTSConfig | None = None)
```

No description.

##### `speak`

```python
def speak(self, text: str, filename: str | None = None) -> str
```

Synthesize and speak the text, saving the audio to a file.


**Arguments:**

- ` text `: The plain text statement to speak.
- ` filename `: Target file name (optional). If omitted, a file name
- is auto-generated based on text hash.
              

**Returns:**

- The absolute path of the generated audio file.
