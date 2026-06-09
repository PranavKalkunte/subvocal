---
title: core.testing
sidebar_label: testing
---

Shared test fixtures for Subvocal SDK unit tests.

## Classes

### `class MockHardwareSource(HardwareSource)`

Generates continuous synthetic sEMG samples (sine + noise).

#### Methods

##### `__init__`

```python
def __init__(self, fs: float = 1000.0, num_channels: int = 4)
```

No description.

##### `start`

```python
def start(self) -> None
```

No description.

##### `stop`

```python
def stop(self) -> None
```

No description.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

No description.

##### `is_connected`

```python
def is_connected(self) -> bool
```

No description.

### `class MockLLMProvider(LLMProvider)`

Maps shorthand token text patterns to GOTO or CLICK intents.

#### Methods

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: list[CommandToken], context: UserContext) -> Intent
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

### `class MockActionExecutor(ActionExecutor)`

Records dispatched actions for assertion in tests.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `execute`

```python
def execute(self, action: Action) -> Any
```

No description.

##### `can_execute`

```python
def can_execute(self, action: Action) -> bool
```

No description.

### `class MockContextProvider(ContextProvider)`

Returns a static browser UserContext snapshot.

#### Methods

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.
