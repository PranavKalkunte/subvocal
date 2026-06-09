---
title: core.pipeline
sidebar_label: pipeline
---

SubvocalPipeline class coordinating the hardware, classification, and execution layers.

## Classes

### `class PipelineStats`

Running counters for a pipeline instance, exposed as ``pipeline.stats``.

#### Methods

##### `as_dict`

```python
def as_dict(self) -> dict[str, Any]
```

Returns the counters as a plain dictionary (for logging/export).

### `class SubvocalPipeline`

Orchestrates hardware ingestion, classifier decoding, intent reconstruction, and action execution.

#### Methods

##### `__init__`

```python
def __init__(self, hardware: HardwareSource, classify_fn: Callable[[Frame], CommandToken | None], llm_provider: LLMProvider, context_provider: ContextProvider, executor: ActionExecutor, phrase_timeout_seconds: float = 1.5, policy_engine: Any | None = None, dry_run: bool = False, trace_path: str | None = None, raise_on_policy_violation: bool = False, on_token: Callable[[CommandToken], None] | None = None, on_intent: Callable[[Intent], None] | None = None, on_action: Callable[[Action, str], None] | None = None, on_error: Callable[[Exception], None] | None = None, telemetry_service: Any | None = None)
```

No description.

##### `trace_path`

```python
def trace_path(self) -> str | None
```

No description.

##### `trace_path`

```python
def trace_path(self, val: str | None) -> None
```

No description.

##### `hardware`

```python
def hardware(self) -> HardwareSource
```

No description.

##### `hardware`

```python
def hardware(self, val: HardwareSource) -> None
```

No description.

##### `classify_fn`

```python
def classify_fn(self) -> Callable[[Frame], CommandToken | None]
```

No description.

##### `classify_fn`

```python
def classify_fn(self, val: Callable[[Frame], CommandToken | None]) -> None
```

No description.

##### `llm_provider`

```python
def llm_provider(self) -> LLMProvider
```

No description.

##### `llm_provider`

```python
def llm_provider(self, val: LLMProvider) -> None
```

No description.

##### `context_provider`

```python
def context_provider(self) -> ContextProvider
```

No description.

##### `context_provider`

```python
def context_provider(self, val: ContextProvider) -> None
```

No description.

##### `executor`

```python
def executor(self) -> ActionExecutor
```

No description.

##### `executor`

```python
def executor(self, val: ActionExecutor) -> None
```

No description.

##### `policy_engine`

```python
def policy_engine(self) -> Any
```

No description.

##### `policy_engine`

```python
def policy_engine(self, val: Any) -> None
```

No description.

##### `phrase_timeout_seconds`

```python
def phrase_timeout_seconds(self) -> float
```

No description.

##### `phrase_timeout_seconds`

```python
def phrase_timeout_seconds(self, val: float) -> None
```

No description.

##### `dry_run`

```python
def dry_run(self) -> bool
```

No description.

##### `dry_run`

```python
def dry_run(self, val: bool) -> None
```

No description.

##### `raise_on_policy_violation`

```python
def raise_on_policy_violation(self) -> bool
```

No description.

##### `raise_on_policy_violation`

```python
def raise_on_policy_violation(self, val: bool) -> None
```

No description.

##### `token_buffer`

```python
def token_buffer(self) -> list[CommandToken]
```

Returns the current accumulated tokens in the buffer.

##### `clear_buffer`

```python
def clear_buffer(self) -> None
```

Clears the accumulated command token buffer.

##### `step`

```python
def step(self, window_ms: int = 100) -> Action | None
```

Performs a single step of the pipeline.


**Arguments:**

- ` window_ms `: Buffer window duration in milliseconds to acquire.


**Returns:**

- An executed Action if an intent was resolved and dispatched, otherwise None.

##### `process_phrase`

```python
def process_phrase(self) -> Action | None
```

Forces immediate decoding and execution of the accumulated tokens.


**Returns:**

- The executed Action if successful, otherwise None.
