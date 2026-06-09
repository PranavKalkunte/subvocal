---
title: stream.tracker
sidebar_label: tracker
---

## Classes

### `class StreamTracker`

Monitors whether a biometric sEMG stream is active or stopped.

Applies a hysteresis threshold to prevent bouncing between states. Equivalent
to LiveKit's StreamTracker.

#### Methods

##### `__init__`

```python
def __init__(self, samples_required: int = 3, cycles_required: int = 5)
```

No description.

##### `status`

```python
def status(self) -> str
```

Returns the current status (active/stopped).

##### `generation`

```python
def generation(self) -> int
```

Returns the current tracker worker generation.

##### `on_status_changed`

```python
def on_status_changed(self, callback: Callable[[str], None]) -> None
```

Registers a callback to execute on status transitions.

##### `observe`

```python
def observe(self, has_activity: bool) -> None
```

Observes an activity tick and updates status state using hysteresis.

##### `set_paused`

```python
def set_paused(self, paused: bool) -> None
```

Pauses or resumes tracking.

Increments generation to invalidate older checks.

##### `reset`

```python
def reset(self) -> None
```

Resets the tracker to initial stopped state.

##### `stop`

```python
def stop(self) -> None
```

Permanently stops the tracker.
