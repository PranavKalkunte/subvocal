---
title: core.llm_providers
sidebar_label: llm_providers
---

Concrete implementations of the LLMProvider interface using urllib.

## Classes

### `class BaseLLMProvider(LLMProvider)`

Base provider containing prompt formatting and request helpers.

#### Methods

##### `__init__`

```python
def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, mock_response: Optional[str] = None, prompt_version: str = 'v1')
```

Initializes the base LLM provider.


**Arguments:**

- ` api_key `: API authorization key. Resolves from environment if None.
- ` model_name `: Name of the LLM model to request.
- ` mock_response `: Optional mock string response to bypass HTTP calls (for testing).
- ` prompt_version `: The version string of the prompt template to use.

### `class ClaudeProvider(BaseLLMProvider)`

Anthropic Claude LLM provider client.

#### Methods

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

No description.

### `class OpenAIProvider(BaseLLMProvider)`

OpenAI GPT completions provider client.

#### Methods

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

No description.

### `class GeminiProvider(BaseLLMProvider)`

Google Gemini content generation provider client.

#### Methods

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

No description.

### `class LlamaProvider(BaseLLMProvider)`

Local Llama provider client (running via Ollama).

#### Methods

##### `__init__`

```python
def __init__(self, ollama_host: Optional[str] = None, model_name: Optional[str] = None, mock_response: Optional[str] = None, prompt_version: str = 'v1')
```

Initializes the Ollama Llama provider client.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

No description.
