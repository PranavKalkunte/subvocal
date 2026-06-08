---
title: core.interfaces
sidebar_label: interfaces
---

Abstract base interfaces for subvocal middleware components.

Enables pluggable integration of various:
- Hardware sources (real-time stream, replay file, synthetic generators).
- LLM providers (Gemini, Claude, GPT, local Llama).
- Action executors (browser control, device APIs, notifications).
- Context providers (OS, active application, contacts, location).

## Classes

### `class HardwareSource(ABC)`

Abstract interface for subvocal/sEMG sensor data acquisition.

#### Methods

##### `start`

```python
def start(self) -> None
```

Initialize connection and start streaming raw sEMG data.

##### `stop`

```python
def stop(self) -> None
```

Stop data acquisition and disconnect hardware clean.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

Reads a buffered window of raw sEMG samples.


**Arguments:**

- ` window_ms `: Time duration in milliseconds to buffer and retrieve.


**Returns:**

- A Frame containing the Sample points.

##### `is_connected`

```python
def is_connected(self) -> bool
```

Returns True if the sensor connection is healthy.

### `class LLMProvider(ABC)`

Abstract interface for turning noisy classified shorthand into semantic Intents.

#### Methods

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

Resolves a noisy shorthand token stream to a structured Intent.


**Arguments:**

- ` tokens `: A sequence of classified CommandTokens.
- ` context `: The UserContext state snapshot at execution time.


**Returns:**

- The reconstructed Intent.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

Returns the name of the LLM provider (e.g., 'gemini', 'anthropic').

### `class ActionExecutor(ABC)`

Abstract interface for dispatching resolved agentic Actions.

#### Methods

##### `execute`

```python
def execute(self, action: Action) -> Any
```

Dispatches the action to device APIs or agent tools.


**Arguments:**

- ` action `: The Action object to execute.


**Returns:**

- The result of the execution.

##### `can_execute`

```python
def can_execute(self, action: Action) -> bool
```

Returns True if the executor supports this specific action type.

### `class ContextProvider(ABC)`

Abstract interface for retrieving live device or environment context.

#### Methods

##### `get_context`

```python
def get_context(self) -> UserContext
```

Retrieves a snapshot of the active user/system context.


**Returns:**

- The UserContext populated with active state metadata.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

Returns the name/identifier of this context provider.

### `class Classifier(ABC)`

Abstract interface for classifying physiological raw signals into command tokens.

#### Methods

##### `predict`

```python
def predict(self, frame: Union[Frame, Any]) -> Optional[CommandToken]
```

Classifies a Frame of raw signals into a CommandToken (applies gating/cooldown if configured).

##### `predict_raw`

```python
def predict_raw(self, frame: Union[Frame, Any]) -> Tuple[str, float, List[float]]
```

Predicts the probability distribution for a Frame of raw signals.


**Returns:**

- (predicted_class_label, max_probability, all_probabilities_list)

##### `labels`

```python
def labels(self) -> List[str]
```

Returns the list of output labels/classes supported by the classifier.
