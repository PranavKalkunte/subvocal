# LLM Intent Decoder Guide

The LLM Provider layer translates classified shorthand sequences (e.g., `clk gt srch`) into structured semantic intents containing commands and context-bound arguments.

---

## 1. Abstract Interface

All LLM decoders must implement the `LLMProvider` interface (defined in `sdk/core/interfaces.py`):

```python
from abc import ABC, abstractmethod
from typing import List
from core.models import CommandToken, Intent
from context.schema import UserContext

class LLMProvider(ABC):
    @abstractmethod
    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        """Resolves gesture shorthand to structured Intent.
        
        Args:
            tokens: Buffer list of classified command tokens.
            context: UserContext snapshot.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Identifier name of the LLM provider."""
        pass
```

---

## 2. Prebuilt Providers

The SDK implements REST integrations for major cloud and local backends under `sdk/core/llm_providers.py`:

*   **`ClaudeProvider`**: Integrates with Anthropic's Messages REST API.
*   **`OpenAIProvider`**: Integrates with OpenAI's Chat Completions REST API.
*   **`GeminiProvider`**: Integrates with Google's generativeContent REST API.
*   **`LlamaProvider`**: Connects to local Llama instances running via **Ollama** or custom local endpoints.

---

## 3. Prompts & System Instructions

The prompt templates are version-managed by `PromptManager` in `sdk/core/prompts.py`. The instructions configure the LLM to act as a strict translator:

1.  **Strict JSON Output**: The LLM is forced to output structured JSON matching the keys `command`, `arguments`, `resolved_text`, and `confidence`.
2.  **Context-Aware Binding**: The system instructions instruct the LLM to inspect the injected context (contacts, visible page elements, calendar) and map shorthand parameters to valid context values. For example:
    *   Shorthand: `gt g` + context contact list containing `name: George`
    *   Resolved: Command `GOTO`, Arguments `["contact: George"]`

### Versioning Prompt Templates

You can customize prompts or register new template versions:

```python
from core.prompts import PromptManager

prompt_manager = PromptManager()
prompt_manager.register_template(
    version="v3",
    template="Your custom system prompt instruction goes here... Shorthand: {shorthand_text}"
)
```

---

## 4. Local Deployment with Ollama

To maintain biometric privacy and run completely offline, deploy a local LLM via Ollama:

1.  Download and install Ollama from [https://ollama.com](https://ollama.com).
2.  Pull a lightweight model (e.g., Llama 3 or Mistral):
    ```bash
    ollama pull llama3:8b
    ```
3.  Ensure the Ollama local daemon is running.
4.  Configure the SDK to use the local `LlamaProvider`:
    ```python
    from core.llm_providers import LlamaProvider
    
    provider = LlamaProvider(model_name="llama3:8b", endpoint="http://localhost:11434")
    ```
