---
title: emg_core.ml.infer
sidebar_label: infer
---

## Classes

### `class InferenceEngine(Classifier)`

Inference engine with debounce and confidence thresholding.

Implements the core.interfaces.Classifier interface.

#### Methods

##### `__init__`

```python
def __init__(self, user_id: str, model_type: str = 'rf', confidence_threshold: float = config.CONFIDENCE_THRESHOLD, cooldown_ms: int = config.COOLDOWN_MS)
```

No description.

##### `labels`

```python
def labels(self) -> list[str]
```

No description.

##### `predict`

```python
def predict(self, frame: Frame | np.ndarray) -> CommandToken | None
```

Classify a segment and apply gating logic.


**Arguments:**

- ` frame `: A Frame object or 2D array of shape (segment_length, num_channels)


**Returns:**

- CommandToken if successful, else None.

##### `predict_raw`

```python
def predict_raw(self, frame: Frame | np.ndarray) -> tuple[str, float, list[float]]
```

Classify a segment/frame without debounce gates.


**Returns:**

- (best_command, probability, all_probabilities)
