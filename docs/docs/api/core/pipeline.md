---
title: core.pipeline
sidebar_label: pipeline
---

SubvocalPipeline class coordinating the hardware, classification, and execution layers.

## Classes

### `class SubvocalPipeline`

Orchestrates hardware ingestion, classifier decoding, intent reconstruction, and action execution.

#### Methods

##### `__init__`

```python
def __init__(self, hardware: HardwareSource, classify_fn: Callable[[Frame], Optional[CommandToken]], llm_provider: LLMProvider, context_provider: ContextProvider, executor: ActionExecutor, phrase_timeout_seconds: float = 1.5, policy_engine: Optional[Any] = None, dry_run: bool = False)
```

Initializes the subvocal pipeline.


**Arguments:**

- ` hardware `: The hardware source to read sEMG frames from.
- ` classify_fn `: A function that takes a Frame and returns an optional CommandToken.
- ` llm_provider `: The LLM intent decoder.
- ` context_provider `: The active user context source.
- ` executor `: The device or tool action dispatcher.
- ` phrase_timeout_seconds `: Duration of silence (no tokens) to wait before triggering intent reconstruction.
- ` policy_engine `: Optional authorization and security policy checker.
- ` dry_run `: If True, resolves intents and compiles actions but does not run them.

##### `token_buffer`

```python
def token_buffer(self) -> List[CommandToken]
```

Returns the current accumulated tokens in the buffer.

##### `clear_buffer`

```python
def clear_buffer(self) -> None
```

Clears the accumulated command token buffer.

##### `step`

```python
def step(self, window_ms: int = 100) -> Optional[Action]
```

Performs a single step of the pipeline.

1. Ingests a new frame of raw sEMG data from the hardware source.
2. Classifies the frame into a potential command token.
3. Accumulates the token and checks for phrase completion timeout.
4. If timed out, reconstructs the intent and executes the action.


**Arguments:**

- ` window_ms `: Buffer window duration in milliseconds to acquire.


**Returns:**

- An executed Action if an intent was resolved and dispatched, otherwise None.

##### `process_phrase`

```python
def process_phrase(self) -> Optional[Action]
```

Forces immediate decoding and execution of the accumulated tokens.


**Returns:**

- The executed Action if successful, otherwise None.
