---
title: runtime.store
sidebar_label: store
---

## Classes

### `class SessionStore(ABC)`

Abstract interface for session configuration and state storage.

#### Methods

##### `save_session`

```python
def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None
```

Saves or updates a session configuration and state snapshot.

##### `get_session`

```python
def get_session(self, session_id: str) -> dict[str, Any] | None
```

Retrieves a session snapshot by ID, returning metadata, config, and state.

##### `delete_session`

```python
def delete_session(self, session_id: str) -> None
```

Deletes a session snapshot by ID.

##### `list_sessions`

```python
def list_sessions(self) -> list[dict[str, Any]]
```

Lists all stored sessions.

### `class InMemorySessionStore(SessionStore)`

In-memory session registry for lightweight, transient deployments.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `save_session`

```python
def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None
```

No description.

##### `get_session`

```python
def get_session(self, session_id: str) -> dict[str, Any] | None
```

No description.

##### `delete_session`

```python
def delete_session(self, session_id: str) -> None
```

No description.

##### `list_sessions`

```python
def list_sessions(self) -> list[dict[str, Any]]
```

No description.

### `class SQLiteSessionStore(SessionStore)`

SQLite-backed session registry for persistent edge configurations.

#### Methods

##### `__init__`

```python
def __init__(self, db_path: str)
```

No description.

##### `save_session`

```python
def save_session(self, session_id: str, state: str, config_yaml: str, metadata: dict[str, Any]) -> None
```

No description.

##### `get_session`

```python
def get_session(self, session_id: str) -> dict[str, Any] | None
```

No description.

##### `delete_session`

```python
def delete_session(self, session_id: str) -> None
```

No description.

##### `list_sessions`

```python
def list_sessions(self) -> list[dict[str, Any]]
```

No description.
