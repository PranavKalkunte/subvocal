---
title: runtime.session
sidebar_label: session
---

## Classes

### `class Session`

Represents an active silent speech processing session.

Co-ordinates real-time signal level monitoring, stream tracking, quality scoring,
and intent decoding on a background OpsQueue thread.

#### Methods

##### `__init__`

```python
def __init__(self, id: str, config: SubvocalConfig, hardware: HardwareSource, classify_fn: Callable[[Frame], CommandToken | None], llm_provider: LLMProvider, context_provider: ContextProvider, executor: ActionExecutor, policy_engine: Any | None = None, trace_path: str | None = None, on_state_changed: Callable[[str], None] | None = None, on_token: Callable[[CommandToken], None] | None = None, on_intent: Callable[[Intent], None] | None = None, on_action: Callable[[Action, str], None] | None = None, on_error: Callable[[Exception], None] | None = None, telemetry: Any | None = None, data_channel: Any | None = None)
```

No description.

##### `state`

```python
def state(self) -> str
```

Returns the current state of the session state machine.

##### `token_buffer`

```python
def token_buffer(self) -> list[CommandToken]
```

Returns the raw reference to the accumulated token buffer.

##### `start`

```python
def start(self) -> None
```

Initializes and starts the session.

##### `stop`

```python
def stop(self) -> None
```

Gracefully stops and closes the session.

##### `push_frame`

```python
def push_frame(self, frame: Frame) -> None
```

Ingests a new Frame, running stream checks and classification asynchronously on OpsQueue.

##### `process_phrase`

```python
def process_phrase(self) -> Action | None
```

Forces immediate decoding of the current token buffer and runs the resolved action.
