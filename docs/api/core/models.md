---
title: core.models
sidebar_label: models
---

Core data models representing the flow from raw physiological signals to actions.

Defines:
- Sample: A single point-in-time multi-channel sensor reading.
- Frame: A window of Samples buffered for classifier consumption.
- CommandToken: A token/shorthand output by a classifier.
- Intent: Reconstructed semantic command and arguments.
- Action: Executable instruction dispatched to an executor.

## Classes

### `class Sample(BaseModel)`

A single multichannel biometric time-series sample.

### `class Frame(BaseModel)`

A window/segment of Samples buffered for signal processing or ML inference.

#### Methods

##### `to_numpy`

```python
def to_numpy(self) -> np.ndarray
```

Converts the frame samples into a 2D NumPy array.


**Returns:**

- np.ndarray of shape (num_samples, num_channels)

### `class CommandToken(BaseModel)`

A token predicted by the raw classifier with associated confidence.

### `class Intent(BaseModel)`

Semantic intent reconstructed from a token stream and active context.

### `class Action(BaseModel)`

The executable instruction dispatched to the system or agent executor.
