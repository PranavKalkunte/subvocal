---
title: stream.level
sidebar_label: level
---

## Classes

### `class SignalLevel`

Tracks and smooths the biological/electrical signal level of sEMG channels.

Equivalent to LiveKit's AudioLevel.

#### Methods

##### `__init__`

```python
def __init__(self, active_level: float = 0.1, min_percentile: float = 40.0, update_interval_ms: int = 400, smooth_intervals: int = 2)
```

No description.

##### `observe`

```python
def observe(self, frame: Frame) -> None
```

Observes a new Frame, computes its MAV, and updates sliding window metrics.

##### `get_level`

```python
def get_level(self, now: float) -> tuple[float, bool]
```

Returns the current smoothed signal level and active status.

Resets level to 0 if signal is stale (no updates received for 2x update_interval).
