---
title: runtime.worker
sidebar_label: worker
---

## Classes

### `class SessionWorker`

Manages active sessions and load-tracking across a pool of silent speech sessions.

Equivalent to LiveKit's SessionWorker/AgentWorker.

#### Methods

##### `__init__`

```python
def __init__(self, config: SubvocalConfig, max_sessions: int = 10, store: Any | None = None)
```

No description.

##### `load`

```python
def load(self) -> float
```

Returns the current load fraction of the worker (0.0 to 1.0).

##### `status`

```python
def status(self) -> str
```

Returns status string based on load.

##### `create_session`

```python
def create_session(self, session_id: str, *args, **kwargs) -> Session
```

Creates, registers, and returns a new Session.

Raises ValueError if session pool capacity is exhausted.

##### `get_session`

```python
def get_session(self, session_id: str) -> Session | None
```

Retrieves a registered Session by ID, or None if not found.

##### `remove_session`

```python
def remove_session(self, session_id: str) -> Session | None
```

Stops and unregisters a Session by ID.

##### `start_session`

```python
def start_session(self, session_id: str) -> None
```

Starts a registered session by ID.

##### `close`

```python
def close(self) -> None
```

Gracefully closes all sessions and stops the worker.
