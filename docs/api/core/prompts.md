---
title: core.prompts
sidebar_label: prompts
---

Prompt templates and version manager for intent reconstruction.

## Classes

### `class PromptManager`

Manages versioned prompt templates for LLM intent decoding.

#### Methods

##### `__init__`

```python
def __init__(self, default_version: str = 'v1')
```

No description.

##### `register_template`

```python
def register_template(self, version: str, template: str) -> None
```

Register a new custom prompt template version.

##### `get_template`

```python
def get_template(self, version: str | None = None) -> str
```

Retrieve a raw prompt template string by version.

##### `format_prompt`

```python
def format_prompt(self, noisy_input: str, heuristic_recommendation: str, web_context: str, calendar: str, contacts: str, history: str, version: str | None = None) -> str
```

Formats the selected prompt template with active inputs.
