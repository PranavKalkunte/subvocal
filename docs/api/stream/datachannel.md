---
title: stream.datachannel
sidebar_label: datachannel
---

## Classes

### `class BiometricDataChannelServer`

TCP server that broadcasts real-time biometric metrics to connected dashboard/client nodes.

#### Methods

##### `__init__`

```python
def __init__(self, host: str = '127.0.0.1', port: int = 8100)
```

No description.

##### `start`

```python
def start(self) -> None
```

Starts the TCP server in a background thread.

##### `broadcast`

```python
def broadcast(self, payload: dict[str, Any]) -> None
```

Sends a JSON-serialized dictionary to all active listeners.

##### `close`

```python
def close(self) -> None
```

Closes the server and drops all client connections.

### `class BiometricDataChannelClient`

Helper client to connect to the BiometricDataChannelServer and parse message frames.

#### Methods

##### `__init__`

```python
def __init__(self, host: str = '127.0.0.1', port: int = 8100)
```

No description.

##### `connect`

```python
def connect(self) -> None
```

Connects to the server socket.

##### `read_messages`

```python
def read_messages(self)
```

Generator yielding parsed messages from the streaming buffer.

##### `close`

```python
def close(self) -> None
```

Disconnects the client socket.
