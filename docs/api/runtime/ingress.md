---
title: runtime.ingress
sidebar_label: ingress
---

## Classes

### `class IngressManager`

Manages active hardware source registration, monitoring, and failovers.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `register_source`

```python
def register_source(self, name: str, source: HardwareSource, is_fallback: bool = False) -> None
```

Registers a biometric input source.

##### `start`

```python
def start(self) -> None
```

Starts the active ingress source stream.

##### `stop`

```python
def stop(self) -> None
```

Stops all registered ingress sources.

##### `active_name`

```python
def active_name(self) -> str | None
```

No description.

##### `get_active_source`

```python
def get_active_source(self) -> HardwareSource | None
```

No description.

##### `trigger_failover`

```python
def trigger_failover(self) -> None
```

Fails over dynamically to the registered fallback source.
