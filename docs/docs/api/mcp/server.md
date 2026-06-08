---
title: mcp.server
sidebar_label: server
---

Model Context Protocol (MCP) reference server for Subvocal Middleware.

## Classes

### `class MockLLMProvider(LLMProvider)`

Fallback LLM Provider that maps command abbreviations to intents.

#### Methods

##### `reconstruct_intent`

```python
def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

### `class MockContextProvider(ContextProvider)`

Exposes mock system context state.

#### Methods

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

### `class MockActionExecutor(ActionExecutor)`

Executes actions and logs execution history.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `execute`

```python
def execute(self, action: Action) -> Any
```

No description.

##### `can_execute`

```python
def can_execute(self, action: Action) -> bool
```

No description.

### `class SubvocalMCPServer`

Zero-dependency Model Context Protocol server communicating via stdio.

#### Methods

##### `__init__`

```python
def __init__(self)
```

No description.

##### `handle_request`

```python
def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]
```

Process an incoming JSON-RPC 2.0 request and return the response dictionary.

##### `run`

```python
def run(self)
```

Standard input/output reader loop.
