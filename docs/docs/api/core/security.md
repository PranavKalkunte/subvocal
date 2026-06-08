---
title: core.security
sidebar_label: security
---

Security policies and authorization gating for subvocal actions.

## Classes

### `class AuthorizationPolicy(ABC)`

Abstract base class for all subvocal execution policies.

#### Methods

##### `is_authorized`

```python
def is_authorized(self, action: Action, context: UserContext) -> bool
```

Evaluate if the given action is authorized to execute.


**Arguments:**

- ` action `: The proposed Action object.
- ` context `: The UserContext state snapshot at proposal time.


**Returns:**

- True if authorized, False otherwise.

### `class ConfidenceThresholdPolicy(AuthorizationPolicy)`

Restricts execution of actions if classifier/reconstruction confidence is too low.

#### Methods

##### `__init__`

```python
def __init__(self, threshold: float = 0.8)
```

No description.

##### `is_authorized`

```python
def is_authorized(self, action: Action, context: UserContext) -> bool
```

No description.

### `class CommandWhitelistPolicy(AuthorizationPolicy)`

Restricts commands to a strict whitelist of allowed actions.

#### Methods

##### `__init__`

```python
def __init__(self, allowed_commands: List[str])
```

No description.

##### `is_authorized`

```python
def is_authorized(self, action: Action, context: UserContext) -> bool
```

No description.

### `class ContextBoundPolicy(AuthorizationPolicy)`

Blocks execution of sensitive commands unless the active application is safe.

#### Methods

##### `__init__`

```python
def __init__(self, sensitive_commands: List[str], safe_applications: List[str])
```

No description.

##### `is_authorized`

```python
def is_authorized(self, action: Action, context: UserContext) -> bool
```

No description.

### `class PolicyEngine`

Orchestrates authorization consensus across multiple active policies.

#### Methods

##### `__init__`

```python
def __init__(self, policies: List[AuthorizationPolicy] = None)
```

No description.

##### `add_policy`

```python
def add_policy(self, policy: AuthorizationPolicy) -> None
```

Add a new policy to the engine.

##### `is_authorized`

```python
def is_authorized(self, action: Action, context: UserContext) -> bool
```

Returns True only if all configured policies approve the action.
