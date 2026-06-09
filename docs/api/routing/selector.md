---
title: routing.selector
sidebar_label: selector
---

## Classes

### `class WorkerNode(Protocol)`

Protocol matching standard attributes of routing nodes.

#### Methods

##### `id`

```python
def id(self) -> str
```

No description.

##### `load`

```python
def load(self) -> float
```

No description.

##### `status`

```python
def status(self) -> str
```

No description.

##### `cpu_usage`

```python
def cpu_usage(self) -> float
```

No description.

### `class NodeSelector(ABC)`

Base interface for all worker node routing selectors.

#### Methods

##### `select_node`

```python
def select_node(self, nodes: list[WorkerNode]) -> WorkerNode
```

Selects the best node from candidate worker nodes.


**Arguments:**

- ` nodes `: Candidates list.


**Returns:**

- The chosen WorkerNode.


**Raises:**

    ValueError: If nodes list is empty.

### `class SessionCountSelector(NodeSelector)`

Selects the worker node with the minimum active session load.

#### Methods

##### `select_node`

```python
def select_node(self, nodes: list[WorkerNode]) -> WorkerNode
```

No description.

### `class CPULoadSelector(NodeSelector)`

Selects the worker node with the lowest CPU load report.

#### Methods

##### `select_node`

```python
def select_node(self, nodes: list[WorkerNode]) -> WorkerNode
```

No description.
