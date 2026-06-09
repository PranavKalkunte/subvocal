---
title: stream.buffer
sidebar_label: buffer
---

## Classes

### `class FrameRing`

Thread-safe circular queue (ring buffer) storing biometric Frame segments.

#### Methods

##### `__init__`

```python
def __init__(self, max_size: int = 100)
```

No description.

##### `push`

```python
def push(self, frame: Frame) -> None
```

Pushes a new Frame onto the ring buffer, evicting the oldest if full.

##### `pop`

```python
def pop(self) -> Frame | None
```

Pops and returns the oldest Frame from the ring buffer, or None if empty.

##### `clear`

```python
def clear(self) -> None
```

Clears the ring buffer.

##### `get_all`

```python
def get_all(self) -> list[Frame]
```

Returns a copy of all current frames in chronological order.

### `class StreamStats`

Windowed tracking for frame arrivals, sample count, gaps, and inter-frame arrival jitter.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `observe`

```python
def observe(self, frame: Frame) -> None
```

Observes a new frame and updates statistics.

##### `get_stats`

```python
def get_stats(self) -> dict[str, Any]
```

Returns a snapshot of the current stats.

##### `reset`

```python
def reset(self) -> None
```

Resets all stats to zero.
