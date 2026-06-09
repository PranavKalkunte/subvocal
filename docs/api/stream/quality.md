---
title: stream.quality
sidebar_label: quality
---

## Classes

### `class SignalQualityScorer`

Scores sEMG signal quality using physical metrics (saturation, drift, dropouts, SNR).

Outputs a MOS-like score mapped to EXCELLENT, GOOD, POOR, and LOST states.
Equivalent to LiveKit's connectionquality Scorer.

#### Methods

##### `__init__`

```python
def __init__(self, clip_max: float = 5.0, increase_factor: float = 0.4, decrease_factor: float = 0.8)
```

No description.

##### `score`

```python
def score(self) -> float
```

Returns the current MOS-like quality score (1.0 - 4.5).

##### `quality`

```python
def quality(self) -> str
```

Returns the current mapped quality state.

##### `on_status_changed`

```python
def on_status_changed(self, callback: Callable[[str], None]) -> None
```

Registers a callback to be run on quality state transitions.

##### `update`

```python
def update(self, frame: Frame) -> None
```

Evaluates a Frame, computes metrics, updates the quality score, and checks transitions.
