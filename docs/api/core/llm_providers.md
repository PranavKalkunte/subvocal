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
def __init__(self, api_key: str | None = None, model_name: str | None = None, mock_response: str | None = None, prompt_version: str = 'v1', timeout: float = 10.0, max_retries: int = 2, backoff_seconds: float = 0.5)
```

Initializes the base LLM provider.


**Arguments:**

- ` api_key `: API authorization key. Resolves from environment if None.
- ` model_name `: Name of the LLM model to request.
- ` mock_response `: Optional mock string response to bypass HTTP calls (for testing).
- ` prompt_version `: The version string of the prompt template to use.
- ` timeout `: Per-request HTTP timeout in seconds.
- ` max_retries `: Additional attempts after the first failure for transient
- errors (connection failures and retryable HTTP statuses).
- ` backoff_seconds `: Base delay for exponential backoff between retries.

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
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
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
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
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
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
```

No description.

### `class LlamaProvider(BaseLLMProvider)`

Local Llama provider client (running via Ollama).

#### Methods

##### `__init__`

```python
def __init__(self, ollama_host: str | None = None, model_name: str | None = None, mock_response: str | None = None, prompt_version: str = 'v1')
```

Initializes the Ollama Llama provider client.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
```

No description.

### `class HeuristicProvider(LLMProvider)`

Fully offline intent reconstruction using the articulatory-distance decoder.

Resolves shorthand phrases against the command vocabulary and the active
user context with no network access or API keys. Useful as a development
default, an air-gapped deployment mode, and a graceful fallback when no
LLM credentials are configured (see :func:`resolve_provider`).

#### Methods

##### `__init__`

```python
def __init__(self, min_confidence: float = 0.0)
```

Initializes the heuristic provider.


**Arguments:**

- ` min_confidence `: Confidence floor below which the intent command is
- reported as UNKNOWN rather than a low-quality guess.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
```

No description.

## Functions

### `resolve_provider`

```python
def resolve_provider(prefer: str | None = None, **kwargs) -> LLMProvider
```

Returns the best available LLM provider for this environment.

Selection order: an explicit ``prefer`` name ("anthropic", "openai",
"gemini", "ollama", "heuristic"), then the first provider whose API key is
present in the environment, then the offline :class:`HeuristicProvider`.


**Arguments:**

- ` prefer `: Optional provider name to force.
- ` **kwargs `: Passed through to the chosen provider constructor.
